import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.augmentation import random_crop
from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.models.inception_network import InceptionNetworkConfig, InceptionNetworkBuilder
from deepal_for_ecg.strategies.initalize.pt4al import PreTextLossInitQueryStrategy
from deepal_for_ecg.strategies.initalize.representation import RepresentationClusteringInitQueryStrategy
from deepal_for_ecg.strategies.query.random import RandomQueryStrategy
from deepal_for_ecg.train.time_series import MultiLabelTimeSeriesTrainer


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


class InitializationStrategy(Enum):
    """Enumeration of initialization strategies."""
    RANDOM = "random"
    PT4AL_ONE = "pt4al_one"
    PT4AL_TEN = "pt4al_ten"
    REPRESENTATION_CLUSTER_PRETEXT = "representation_cluster_pretext"
    REPRESENTATION_CLUSTER_TL = "representation_cluster_tl"


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

    RANDOM_EXPERIMENT_ID = "random"
    PT4AL_ONE_EXPERIMENT_ID = "pt4al_one"
    PT4AL_TEN_EXPERIMENT_ID = "pt4al_ten"
    REP_CLUSTER__PRETEXT_EXPERIMENT_ID = "rep_cluster_pretext"
    REP_CLUSTER_TL_EXPERIMENT_ID = "rep_cluster_tl"

    def __init__(self, model_dir: Path = Path("./models"), log_dir: Path = Path("./logs/experiments/init_strategy"),
                 pretext_model_base_name: str = "PretextInception", transfer_learning_model_name: str =
                 "ICBEBInception", initial_samples: int = 300, saved_data_base_dir: str | Path =
                 Path("./data/saved/ptbxl"), runs_per_strategy: int = 5, experiment_result_dir: Path =
                 Path("./experiments/init_strategy")):
        self._data_loader = PTBXLDataLoader(load_saved_data=True, saved_data_base_dir=saved_data_base_dir)
        self._data_loader.load_data()
        self._initial_samples = initial_samples
        self._load_model_dir = model_dir
        self._experiment_model_dir = Path(model_dir, "init_experiment")
        self._experiment_model_dir.mkdir(parents=True, exist_ok=True)
        self._pretext_model_base_name = pretext_model_base_name
        self._transfer_learning_model_name = transfer_learning_model_name
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._runs_per_strategy = runs_per_strategy
        self._experiment_result_dir = experiment_result_dir
        self._experiment_result_dir.mkdir(parents=True, exist_ok=True)

        # for model evaluation
        self._loss_object = keras.losses.BinaryCrossentropy()
        self._test_loss = keras.metrics.Mean(name="validation_loss")
        self._test_accuracy = keras.metrics.BinaryAccuracy(name="validation_accuracy")
        self._test_auc = keras.metrics.AUC(multi_label=True, name="validation_auc",
                                           num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES)

    def run(self):
        all_run_results = []
        for strategy in InitializationStrategy:
            for run_number in range(1, self._runs_per_strategy+1):
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

    def _do_experiment_run(self, strategy: InitializationStrategy, run_number: int) -> ExperimentRunResult:
        data_module = PTBXLActiveLearningDataModule(
            train_samples=self._data_loader.X_train,
            test_samples=self._data_loader.X_test,
            val_samples=self._data_loader.X_valid,
            train_labels_12sl=self._data_loader.Y_train_12sl,
            train_labels_ptb_xl=self._data_loader.Y_train_ptb_xl,
            test_labels=self._data_loader.Y_test,
            val_labels=self._data_loader.Y_valid
        )

        # select initial samples
        data_module = self._apply_initial_query_strategy(data_module, strategy, run_number=run_number)

        # prepare the model
        model_config = InceptionNetworkConfig()
        builder = InceptionNetworkBuilder()
        model = builder.build_model(model_config)

        # train the classifier
        trainer = MultiLabelTimeSeriesTrainer(model=model, model_name=f"{strategy.value}_{run_number}",
                                              num_labels=data_module.NUM_CLASSES,
                                              model_base_dir=str(self._experiment_model_dir),
                                              log_base_dir=str(self._log_dir), epochs=100)
        trainer.experiment_name = f"{strategy.value}_{run_number}"
        trainer.fit(data_module.train_dataset, data_module.validation_dataset)
        best_model = trainer.get_model(best=True)

        # evaluate the performance
        self._reset_metrics()
        for samples, labels in data_module.test_dataset.batch(128):
            self._test_step(samples, labels, best_model)

        return self._get_experiment_run_result(experiment_name=strategy.value, run=run_number,
                                               data_module_state=data_module.state_dict(), epoch=trainer.best_epoch)

    def _reset_metrics(self):
        self._test_loss.reset_states()
        self._test_accuracy.reset_states()
        self._test_auc.reset_states()

    def _get_experiment_run_result(self, experiment_name: str, run: int, data_module_state: Dict, epoch: int) -> ExperimentRunResult:
        return ExperimentRunResult(auc=self._test_auc.result().numpy(), loss=self._test_loss.result().numpy(),
                                   accuracy=self._test_accuracy.result().numpy(), experiment_name=str(experiment_name),
                                   run=run, data_module_state=data_module_state, epoch=epoch)

    @tf.function
    def _test_step(self, samples_batch, label_batch, model: keras.Model):
        sliding_window_predictions = []
        for sliding_window in samples_batch:
            sliding_window_predictions.append(model(sliding_window, training=False))

        predictions = tf.reduce_max(sliding_window_predictions, axis=0)
        t_loss = self._loss_object(label_batch, predictions)

        self._test_loss(t_loss)
        self._test_accuracy(label_batch, predictions)
        self._test_auc(label_batch, predictions)

    def _apply_initial_query_strategy(self, data_module: PTBXLActiveLearningDataModule,
                                      strategy: InitializationStrategy, run_number: int):
        """Applies the initial query strategy to select the initial batch of samples that have to be labeled."""
        selected_samples = set()

        if strategy == InitializationStrategy.RANDOM:
            unlabeled_indices = data_module.state_dict()["unlabeled_indices"]
            selected_samples = RandomQueryStrategy().select_samples(self._initial_samples, unlabeled_indices)
        if strategy == InitializationStrategy.PT4AL_ONE:
            selected_samples = self._apply_pt4al(run_number, num_of_al_batches=1)
        if strategy == InitializationStrategy.PT4AL_TEN:
            selected_samples = self._apply_pt4al(run_number, num_of_al_batches=10)
        if strategy == InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT:
            selected_samples = self._apply_representation_clustering(run_number, self._pretext_model_base_name,
                                                                     lambda x: x, data_module.unlabeled_dataset)
        if strategy == InitializationStrategy.REPRESENTATION_CLUSTER_TL:
            selected_samples = self._apply_representation_clustering(run_number, self._transfer_learning_model_name,
                                                                     random_crop, data_module.unlabeled_dataset)

        data_module.update_annotations(buy_idx_ptb_xl=selected_samples, buy_idx_12sl=set())
        return data_module

    def _apply_pt4al(self, run_number: int, num_of_al_batches: int):
        # load pretext model
        model_name = f"{self._pretext_model_base_name}{run_number}"
        model = keras.models.load_model(Path(self._load_model_dir, model_name, "best_model.keras"))

        init_strategy = PreTextLossInitQueryStrategy(model, model_name)
        init_strategy.prepare(self._data_loader.X_train, num_of_al_batches=num_of_al_batches, load_losses=True)
        return init_strategy.select_samples(self._initial_samples, current_batch=0)

    def _apply_representation_clustering(self, run_number: int, model_base_name: str, augmentation_method: callable,
                                         unlabeled_dataset: tf.data.Dataset):
        # load pretext model
        model_name = f"{model_base_name}{run_number}"
        model = keras.models.load_model(Path(self._load_model_dir, model_name, "best_model.keras"))

        init_strategy = RepresentationClusteringInitQueryStrategy(model, self._initial_samples, augmentation_method)
        init_strategy.prepare(unlabeled_dataset)
        return init_strategy.select_samples()

    def _is_run_completed(self, result_file: str) -> bool:
        """Checks whether a run was completed earlier."""
        return Path(self._experiment_result_dir, result_file).exists()

    def _save_run_result(self, result: ExperimentRunResult):
        """Saves the run result to disc."""
        with open(Path(self._experiment_result_dir, f"{result.experiment_name}_{result.run}.pkl"), 'wb') as data_file:
            pickle.dump(result, data_file)

    def _load_previous_result(self, result_file: str) -> ExperimentRunResult:
        """Loads the previous run result."""
        with open(Path(self._experiment_result_dir, result_file), "rb") as pickle_file:
            return pickle.load(pickle_file)

