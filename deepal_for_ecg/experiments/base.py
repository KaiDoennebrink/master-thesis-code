import pickle
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Set, List

from matplotlib import pyplot as plt
import pandas as pd
from tensorflow import keras

from deepal_for_ecg.data.augmentation import random_crop
from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.models.inception_network import InceptionNetworkConfig, InceptionNetworkBuilder
from deepal_for_ecg.strategies.initalize import InitializationStrategy, apply_pt4al, apply_representation_clustering
from deepal_for_ecg.strategies.query import SelectionStrategy
from deepal_for_ecg.strategies.query.badge import BadgeSamplingStrategy
from deepal_for_ecg.strategies.query.entropy import EntropySamplingStrategy
from deepal_for_ecg.strategies.query.plvi_ce import PredictedLabelVectorInconsistencyCrossEntropyStrategy
from deepal_for_ecg.strategies.query.random import RandomQueryStrategy
from deepal_for_ecg.train.test_util import test_step_with_sliding_windows
from deepal_for_ecg.train.time_series import MultiLabelTimeSeriesTrainer


@dataclass
class BaseExperimentConfig:
    name: str
    strategy: SelectionStrategy
    num_al_iterations: int = 20
    num_al_samples: int = 300
    # directories
    base_experiment_dir: Path = Path("./experiments/al")
    models_folder_name: str = "models"
    tensorboard_logs_folder_name: str = "tensorboard_logs"
    plots_folder_name: str = "plots"
    results_folder_name: str = "results"
    # initialization
    num_initial_samples: int = 300
    init_strategy: InitializationStrategy = InitializationStrategy.RANDOM
    init_strategy_pretrained_model_dir: Path | None = None
    init_strategy_pretrained_model_base_name: str | None = None
    init_strategy_pretrained_model_run_number: int = 1
    # start model
    start_model_dir: Path | None = None
    # data
    ptbxl_data_base_dir: Path = Path("./data/saved/ptbxl")


@dataclass(kw_only=True)
class BaseExperimentALIterationResult:
    auc: float
    accuracy: float
    loss: float
    label_coverage: float
    training_time: float
    selection_time: float
    all_selected_samples: Set[int]
    newly_selected_samples: Set[int]
    al_iteration: int

    # Generated attribute
    num_samples: int = field(init=False)
    num_newly_samples: int = field(init=False)

    @property
    def num_samples(self) -> int:
        return len(self.all_selected_samples)

    @num_samples.setter
    def num_samples(self, _):
        pass  # read-only attribute

    @property
    def num_newly_samples(self) -> int:
        return len(self.newly_selected_samples)

    @num_newly_samples.setter
    def num_newly_samples(self, _):
        pass  # read-only attribute

    def __str__(self) -> str:
        return f"{self.auc = }, {self.accuracy = }, {self.loss = }, {self.label_coverage = }, {self.selection_time = }, {self.training_time = }"


class BaseExperiment:
    """
    The base for all active learning experiments with the PTB-XL dataset and the InceptionTime network.
    In the base setting samples can just be selected from the human-annotator.
    """

    def __init__(self, config: BaseExperimentConfig):
        self.config = config
        # directory setup
        self._experiment_dir = Path(config.base_experiment_dir, config.strategy.value, config.name)
        self._model_dir = Path(self._experiment_dir, config.models_folder_name)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_logs_dir = Path(self._experiment_dir, config.tensorboard_logs_folder_name)
        self._tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir = Path(self._experiment_dir, config.plots_folder_name)
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        self._results_dir = Path(self._experiment_dir, config.results_folder_name)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        # get the models, that should be used as a starting point in each iteration
        self._start_model = self._get_or_create_start_model()
        self._best_model = None
        # load the data and prepare the data module
        self._data_loader = PTBXLDataLoader(load_saved_data=True, saved_data_base_dir=config.ptbxl_data_base_dir)
        self._data_loader.load_data()
        self._data_module = PTBXLActiveLearningDataModule(
            train_samples=self._data_loader.X_train,
            test_samples=self._data_loader.X_test,
            val_samples=self._data_loader.X_valid,
            train_labels_12sl=self._data_loader.Y_train_12sl,
            train_labels_ptb_xl=self._data_loader.Y_train_ptb_xl,
            test_labels=self._data_loader.Y_test,
            val_labels=self._data_loader.Y_valid
        )
        # test utilities
        self._loss_object = keras.losses.BinaryCrossentropy()
        self._test_loss = keras.metrics.Mean(name="test_loss")
        self._test_accuracy = keras.metrics.BinaryAccuracy(name="test_accuracy")
        self._test_auc = keras.metrics.AUC(multi_label=True, name="test_auc",
                                           num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES)
        # results
        self._auc_results = []
        self._accuracy_results = []
        self._loss_results = []
        self._coverage_results = []
        self._training_times = []
        self._sampling_times = []
        self._selected_samples = set()

        # save initialisation
        with open(Path(self._experiment_dir, "config.pkl"), "wb") as data_file:
            pickle.dump(config, data_file)

    def run(self):
        """
        Main function to run the experiment. In the initialization phase of the active learning cycle, samples are
        selected with the initial selection strategy. In all other phases, the selection strategy is used.
        """
        self._print_heading("AL initialization")
        self._run_al_iteration(name="initial", selection=self._initial_selection, al_iteration=0)
        for i in range(1, self.config.num_al_iterations + 1):
            self._print_heading(f"AL iteration {i}")
            self._run_al_iteration(name=f"al_iteration_{i}", selection=self._al_selection, al_iteration=i)
            self._reset_fields()

        self._save_final_results()

    def _run_al_iteration(self, name: str, selection: Callable, al_iteration: int):
        """
        Runs a single AL iteration.

        Args:
            name (str): Name of the active learning iteration.
            selection (callable): The selection method to choose the samples in the current iteration.
            al_iteration (int): Current iteration
        """
        if self._is_iteration_complete(al_iteration):
            result = self._load_previous_result(al_iteration)
            self._update_current_state(result)
            self._load_best_model(name)
            print(f"Found result: {result}")
        else:
            self._print_heading("sampling")
            self._sampling(selection, al_iteration)

            self._print_heading("training")
            self._train(name=name)

            self._print_heading("testing")
            self._test()
            self._save_al_iteration_results(al_iteration)
        print("")
        print("")

    def _sampling(self, selection: Callable, al_iteration: int):
        """Performs the sampling."""
        sample_selection_start = time.time()
        self._selected_samples = selection()
        sampling_time = time.time() - sample_selection_start
        # validate the sampling
        expected_num_samples = self.config.num_al_samples if al_iteration > 0 else self.config.num_initial_samples
        actual_num_samples = len(self._selected_samples)
        assert actual_num_samples == expected_num_samples, f"{actual_num_samples = }, {expected_num_samples = }"
        self._sampling_times.append(sampling_time)
        self._buy_samples(self._selected_samples)
        coverage = self._data_module.calculate_label_coverage()
        self._coverage_results.append(coverage)
        print(f"{coverage = }")
        print(f"sampling took {sampling_time:.4f} seconds")

    def _train(self, name: str):
        """Trains the model on the currently selected data points."""
        training_start = time.time()
        current_model = keras.models.clone_model(self._start_model)
        trainer = MultiLabelTimeSeriesTrainer(model=current_model, model_name=name,
                                              num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES,
                                              model_base_dir=str(self._model_dir),
                                              log_base_dir=str(self._tensorboard_logs_dir), epochs=100)
        trainer.experiment_name = name
        trainer.fit(self._data_module.train_dataset, self._data_module.validation_dataset, verbose=True)
        self._best_model = trainer.get_model(best=True)
        training_time = time.time() - training_start
        self._training_times.append(training_time)
        print(f"training took {training_time:.4f} seconds")

    def _test(self):
        """Tests the model."""
        self._reset_metrics()
        for samples, labels in self._data_module.test_dataset.batch(128):
            test_step_with_sliding_windows(
                sliding_window_batch=samples,
                label_batch=labels,
                model=self._best_model,
                loss_object=self._loss_object,
                loss_based_metrics=[self._test_loss],
                prediction_based_metrics=[self._test_accuracy, self._test_auc],
            )

    def _save_al_iteration_results(self, al_iteration: int):
        self._set_test_results()
        result = self._get_al_iteration_result(al_iteration)
        with open(self._get_result_file(al_iteration), "wb") as data_file:
            pickle.dump(result, data_file)

    def _set_test_results(self):
        auc = self._test_auc.result().numpy()
        loss = self._test_loss.result().numpy()
        accuracy = self._test_accuracy.result().numpy()
        print(f"{auc = }, {accuracy = }, {loss = }")
        self._auc_results.append(auc)
        self._loss_results.append(loss)
        self._accuracy_results.append(accuracy)

    def _get_al_iteration_result(self, al_iteration: int) -> BaseExperimentALIterationResult:
        return BaseExperimentALIterationResult(
            auc=self._auc_results[-1],
            loss=self._loss_results[-1],
            accuracy=self._accuracy_results[-1],
            label_coverage=self._coverage_results[-1],
            training_time=self._training_times[-1],
            selection_time=self._sampling_times[-1],
            all_selected_samples=self._data_module.state_dict()["labeled_indices_ptb_xl"],
            newly_selected_samples=self._selected_samples,
            al_iteration=al_iteration
        )

    def _reset_metrics(self):
        self._test_loss.reset_states()
        self._test_accuracy.reset_states()
        self._test_auc.reset_states()

    def _al_selection(self) -> Set[int]:
        """Performs the sample selection with the specified strategy."""
        if self.config.strategy == SelectionStrategy.RANDOM:
            strategy = RandomQueryStrategy()
            unlabeled_indices = self._data_module.state_dict()["unlabeled_indices"]
            return strategy.select_samples(self.config.num_initial_samples, unlabeled_indices)
        if (self.config.strategy == SelectionStrategy.PLVI_CE_TOPK
                or self.config.strategy == SelectionStrategy.PLVI_CE_KNN):
            strategy = PredictedLabelVectorInconsistencyCrossEntropyStrategy()
            use_top_k = self.config.strategy == SelectionStrategy.PLVI_CE_TOPK
            return strategy.select_samples(
                num_of_samples=self.config.num_initial_samples,
                data_module=self._data_module,
                data_loader=self._data_loader,
                model=self._best_model,
                top_k_selection=use_top_k
            )
        if self.config.strategy == SelectionStrategy.ENTROPY:
            strategy = EntropySamplingStrategy()
            return strategy.select_samples(self.config.num_initial_samples, self._data_module, self._best_model)
        if self.config.strategy == SelectionStrategy.BADGE:
            strategy = BadgeSamplingStrategy()
            return strategy.select_samples(self.config.num_initial_samples, self._data_module, self._best_model)
        else:
            raise NotImplementedError(f"The selection strategy {self.config.strategy} is not implemented.")

    def _initial_selection(self) -> Set[int]:
        """Performs the selection of the first samples."""
        if self.config.init_strategy == InitializationStrategy.RANDOM:
            strategy = RandomQueryStrategy()
            unlabeled_indices = self._data_module.state_dict()["unlabeled_indices"]
            return strategy.select_samples(self.config.num_initial_samples, unlabeled_indices)
        if self.config.init_strategy == InitializationStrategy.PT4AL_ONE or self.config.init_strategy == InitializationStrategy.PT4AL_TEN:
            num_of_al_batches = 1 if self.config.init_strategy == InitializationStrategy.PT4AL_ONE else 10
            return apply_pt4al(
                num_samples=self.config.num_initial_samples,
                training_samples=self._data_loader.X_train,
                model_dir=self.config.init_strategy_pretrained_model_dir,
                pretext_model_base_name=self.config.init_strategy_pretrained_model_base_name,
                run_number=self.config.init_strategy_pretrained_model_run_number,
                num_of_al_batches=num_of_al_batches
            )
        if self.config.init_strategy == InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT or self.config.init_strategy == InitializationStrategy.REPRESENTATION_CLUSTER_TL:
            if self.config.init_strategy == InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT:
                augmentation_method = lambda x: x
            else:
                augmentation_method = random_crop
            return apply_representation_clustering(
                num_samples=self.config.num_initial_samples,
                unlabeled_dataset=self._data_module.unlabeled_dataset,
                model_base_name=self.config.init_strategy_pretrained_model_base_name,
                augmentation_method=augmentation_method,
                run_number=self.config.init_strategy_pretrained_model_run_number,
                model_dir=self.config.init_strategy_pretrained_model_dir
            )
        else:
            raise NotImplementedError(f"The initialisation strategy {self.config.init_strategy} is not implemented.")

    @classmethod
    def _print_heading(cls, heading: str):
        """Prints a heading in a unified way to std out."""
        print(f"{heading:=^50}")

    def _get_or_create_start_model(self) -> keras.Model:
        """
        Returns the model that should be used as a starting point for each iteration.
        Can either create a new model for that purpose or use an existing one.
        If a new model is created the model is saved.

        Returns: The model that should be used as a starting model in each active learning iteration.
        """
        if not (self.config.start_model_dir is None) and self.config.start_model_dir.exists():
            return keras.models.load_model(self.config.start_model_dir)
        elif Path(self._model_dir, "start_model.keras").exists():
            # load the start model from a previous point
            return keras.models.load_model(Path(self._model_dir, "start_model.keras"))
        else:
            # if no model was loaded, create a new one
            model_config = InceptionNetworkConfig()
            builder = InceptionNetworkBuilder()
            model = builder.build_model(model_config)
            model.save(Path(self._model_dir, "start_model.keras"))
            return model

    def _is_iteration_complete(self, al_iteration: int) -> bool:
        """Checks whether an iteration was completed earlier."""
        return self._get_result_file(al_iteration).exists()

    def _get_result_file(self, al_iteration: int) -> Path:
        """Returns the path to the result file of the given iteration."""
        return Path(self._results_dir, f"al_iteration_{al_iteration}.pkl")

    def _load_previous_result(self, al_iteration: int) -> BaseExperimentALIterationResult:
        """Loads the previous run result."""
        with open(self._get_result_file(al_iteration), "rb") as pickle_file:
            return pickle.load(pickle_file)

    def _load_best_model(self, name: str) -> keras.Model:
        self._best_model = keras.models.load_model(Path(self._model_dir, name, "best_model.keras"))

    def _update_current_state(self, result: BaseExperimentALIterationResult):
        """
        Updates the current state of the class with the given result.
        Args:
            result: A result from a previous iteration.
        """
        self._update_data_module(result)
        self._update_result_lists(result)

    def _update_data_module(self, result: BaseExperimentALIterationResult):
        """Updates the data module with the previous selections."""
        self._buy_samples(result.newly_selected_samples)

    def _buy_samples(self, selected_indices: Set[int]):
        self._data_module.update_annotations(buy_idx_ptb_xl=selected_indices, buy_idx_12sl=set())

    def _update_result_lists(self, result: BaseExperimentALIterationResult):
        """Updates the result lists with the previous results."""
        self._auc_results.append(result.auc)
        self._accuracy_results.append(result.accuracy)
        self._loss_results.append(result.loss)
        self._coverage_results.append(result.label_coverage)
        self._training_times.append(result.training_time)
        self._sampling_times.append(result.selection_time)

    def _save_final_results(self):
        self._plot_metric(self._auc_results, "Macro AUC")
        self._plot_metric(self._accuracy_results, "Binary accuracy")
        self._plot_metric(self._loss_results, "Loss")
        self._plot_metric(self._coverage_results, "Label coverage")
        self._plot_metric(self._training_times, "Training time")
        self._plot_metric(self._sampling_times, "Sampling time")

        data_dict = self._get_final_data_dict()
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(Path(self._results_dir, "results.csv"))

    def _plot_metric(self, values: List, value_name: str):
        plt.clf()  # clear previous plots
        fig = plt.figure(1, figsize=(10, 5))
        fig.suptitle(f"{value_name} per active learning iteration")
        plt.plot(values, label=value_name)
        plt.ylabel(value_name)
        plt.xlabel("AL iteration")
        fig_name = value_name.lower().replace(" ", "_")
        fig.savefig(Path(self._plots_dir, f"{fig_name}_plot.png"))

    def _get_final_data_dict(self):
        return {
            "auc": self._auc_results,
            "accuracy": self._accuracy_results,
            "loss": self._loss_results,
            "coverage": self._coverage_results,
            "training_time": self._training_times,
            "sampling_time": self._sampling_times
        }

    def _reset_fields(self):
        self._selected_samples = set()
