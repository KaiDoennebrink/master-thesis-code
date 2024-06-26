import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from tensorflow import keras

from deepal_for_ecg.data.augmentation import random_crop
from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.models.inception_network import (
    InceptionNetworkConfig,
    InceptionNetworkBuilder,
)
from deepal_for_ecg.strategies.initalize import InitializationStrategy, apply_pt4al, apply_representation_clustering
from deepal_for_ecg.strategies.query.random import RandomQueryStrategy
from deepal_for_ecg.train.test_util import test_step_with_sliding_windows
from deepal_for_ecg.train.time_series import MultiLabelTimeSeriesTrainer


@dataclass
class InitializationExperimentConfig:
    # general experiment setup
    base_experiment_dir: Path = Path("./experiments/init_strategy")
    model_dir_name: str = "models"
    log_dir_name: str = "logs"
    results_dir_name: str = "results"
    runs_per_strategy: int = 10
    # setup for the initialization strategies
    pretrained_model_dir: Path = Path("./models")
    initial_samples: int = 300
    pretext_model_base_name: str = "PretextInception"
    transfer_learning_model_name: str = "ICBEBInception"
    # data
    saved_data_base_dir: str | Path = Path("./data/saved/ptbxl")


@dataclass
class ExperimentRunResult:
    """The result of an experiment run."""

    auc: float
    loss: float
    accuracy: float
    experiment_name: str
    run: int
    data_module_state: Dict
    epoch: int

    def __str__(self):
        return f"Experiment: {self.experiment_name}, Run: {self.run}, AUC: {self.auc}, Accuracy: {self.accuracy}, Loss: {self.loss}, Epoch: {self.epoch}"


class InitializationStrategyExperiment:
    """
    Code class for the initialization strategy experiments.

    The experiment compares five different initialization strategies:
    - random selection (baseline)
    - PT4AL initial selection with just splitting the unlabeled data in one big pool
    - PT4AL initial selection with splitting the unlabeled data in 10 pools based on pretext task loss
    - Representation clustering selection with representations from a pre-trained model
    - Representation clustering selection with representations from a model trained on another dataset
    """

    def __init__(self, config: InitializationExperimentConfig):
        self.config = config
        self._data_loader = PTBXLDataLoader(load_saved_data=True, saved_data_base_dir=config.saved_data_base_dir)
        self._data_loader.load_data()
        self._experiment_model_dir = Path(config.base_experiment_dir, config.model_dir_name)
        self._experiment_model_dir.mkdir(parents=True, exist_ok=True)
        self._init_model_dir = Path(self._experiment_model_dir, "initial")
        self._init_model_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir = Path(config.base_experiment_dir, config.log_dir_name)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_result_dir = Path(config.base_experiment_dir, config.results_dir_name)
        self._experiment_result_dir.mkdir(parents=True, exist_ok=True)

        # for model evaluation
        self._loss_object = keras.losses.BinaryCrossentropy()
        self._test_loss = keras.metrics.Mean(name="validation_loss")
        self._test_accuracy = keras.metrics.BinaryAccuracy(name="validation_accuracy")
        self._test_auc = keras.metrics.AUC(
            multi_label=True,
            name="validation_auc",
            num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES,
        )

        with open(Path(config.base_experiment_dir, "config.pkl"), "wb") as data_file:
            pickle.dump(config, data_file)

    def run(self):
        all_run_results = []
        for strategy in InitializationStrategy:
            for run_number in range(1, self.config.runs_per_strategy + 1):
                print("-------------------------------------------------------")
                print(f"Running strategy {strategy} - run number {run_number}")
                result_file_name = f"{strategy.value}_{run_number}.pkl"
                if self._is_run_completed(result_file_name):
                    run_result = self._load_previous_result(result_file_name)
                    print(f"Found result: {run_result}")
                else:
                    run_result = self._do_experiment_run(strategy, run_number)
                    self._save_run_result(run_result)
                    print(f"Result: {run_result}")

                all_run_results.append(run_result)
                print("")

        run_results = [vars(run_result) for run_result in all_run_results]
        df = pd.DataFrame(run_results)
        df.to_csv(Path(self._experiment_result_dir, "results.csv"))

    def _do_experiment_run(
        self, strategy: InitializationStrategy, run_number: int
    ) -> ExperimentRunResult:
        data_module = PTBXLActiveLearningDataModule(
            train_samples=self._data_loader.X_train,
            test_samples=self._data_loader.X_test,
            val_samples=self._data_loader.X_valid,
            train_labels_12sl=self._data_loader.Y_train_12sl,
            train_labels_ptb_xl=self._data_loader.Y_train_ptb_xl,
            test_labels=self._data_loader.Y_test,
            val_labels=self._data_loader.Y_valid,
        )

        # select initial samples
        data_module = self._apply_initial_query_strategy(
            data_module, strategy, run_number=run_number
        )

        # prepare the model
        model = self._get_model(run_number)

        # train the classifier
        trainer = MultiLabelTimeSeriesTrainer(
            model=model,
            model_name=f"{strategy.value}_{run_number}",
            num_labels=data_module.NUM_CLASSES,
            model_base_dir=str(self._experiment_model_dir),
            log_base_dir=str(self._log_dir),
            epochs=100,
        )
        trainer.experiment_name = f"{strategy.value}_{run_number}"
        trainer.fit(data_module.train_dataset, data_module.validation_dataset, verbose=True)
        best_model = trainer.get_model(best=True)

        # evaluate the performance
        self._reset_metrics()
        for samples, labels in data_module.test_dataset.batch(128):
            test_step_with_sliding_windows(
                sliding_window_batch=samples,
                label_batch=labels,
                model=best_model,
                loss_object=self._loss_object,
                loss_based_metrics=[self._test_loss],
                prediction_based_metrics=[self._test_accuracy, self._test_auc],
            )

        return self._get_experiment_run_result(
            experiment_name=strategy.value,
            run=run_number,
            data_module_state=data_module.state_dict(),
            epoch=trainer.best_epoch,
        )

    def _reset_metrics(self):
        self._test_loss.reset_states()
        self._test_accuracy.reset_states()
        self._test_auc.reset_states()

    def _get_experiment_run_result(
        self, experiment_name: str, run: int, data_module_state: Dict, epoch: int
    ) -> ExperimentRunResult:
        return ExperimentRunResult(
            auc=self._test_auc.result().numpy(),
            loss=self._test_loss.result().numpy(),
            accuracy=self._test_accuracy.result().numpy(),
            experiment_name=str(experiment_name),
            run=run,
            data_module_state=data_module_state,
            epoch=epoch,
        )

    def _apply_initial_query_strategy(
        self,
        data_module: PTBXLActiveLearningDataModule,
        strategy: InitializationStrategy,
        run_number: int,
    ):
        """Applies the initial query strategy to select the initial batch of samples that have to be labeled."""
        selected_samples = set()

        if strategy == InitializationStrategy.RANDOM:
            unlabeled_indices = data_module.state_dict()["unlabeled_indices"]
            selected_samples = RandomQueryStrategy().select_samples(self.config.initial_samples, unlabeled_indices)
        if strategy == InitializationStrategy.PT4AL_ONE or strategy == InitializationStrategy.PT4AL_TEN:
            num_of_al_batches = 1 if strategy == InitializationStrategy.PT4AL_ONE else 10
            selected_samples = apply_pt4al(
                self.config.initial_samples,
                self._data_loader.X_train,
                model_dir=self.config.pretrained_model_dir,
                pretext_model_base_name=self.config.pretext_model_base_name,
                run_number=run_number,
                num_of_al_batches=num_of_al_batches
            )
        if strategy == InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT or strategy == InitializationStrategy.REPRESENTATION_CLUSTER_TL:
            if strategy == InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT:
                model_base_name = self.config.pretext_model_base_name
                augmentation_method = lambda x: x
            else:
                model_base_name = self.config.transfer_learning_model_name
                augmentation_method = random_crop
            selected_samples = apply_representation_clustering(
                num_samples=self.config.initial_samples,
                unlabeled_dataset=data_module.unlabeled_dataset,
                model_base_name=model_base_name,
                augmentation_method=augmentation_method,
                run_number=run_number,
                model_dir=self.config.pretrained_model_dir
            )

        data_module.update_annotations(
            buy_idx_ptb_xl=selected_samples, buy_idx_12sl=set()
        )
        return data_module

    def _is_run_completed(self, result_file: str) -> bool:
        """Checks whether a run was completed earlier."""
        return Path(self._experiment_result_dir, result_file).exists()

    def _save_run_result(self, result: ExperimentRunResult):
        """Saves the run result to disc."""
        with open(
            Path(
                self._experiment_result_dir,
                f"{result.experiment_name}_{result.run}.pkl",
            ),
            "wb",
        ) as data_file:
            pickle.dump(result, data_file)

    def _load_previous_result(self, result_file: str) -> ExperimentRunResult:
        """Loads the previous run result."""
        with open(Path(self._experiment_result_dir, result_file), "rb") as pickle_file:
            return pickle.load(pickle_file)

    def _get_model(self, run_number) -> keras.Model:
        """
        Returns the initial model for the given run number so that for each run with the different strategies the same
        initial model is used.
        """
        initial_model_path = Path(self._init_model_dir, f"initial_model_{run_number}.keras")
        if initial_model_path.exists():
            # load the existing initial model
            return keras.models.load_model(initial_model_path)

        model_config = InceptionNetworkConfig()
        builder = InceptionNetworkBuilder()
        model = builder.build_model(model_config)
        model.save(initial_model_path)
        return model
