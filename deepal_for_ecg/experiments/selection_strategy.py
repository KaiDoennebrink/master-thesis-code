import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Set, Callable, List

import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.experiments.initialization_strategy import InitializationStrategy
from deepal_for_ecg.models.inception_network import InceptionNetworkConfig, InceptionNetworkBuilder
from deepal_for_ecg.strategies.query.badge import BadgeSamplingStrategy
from deepal_for_ecg.strategies.query.entropy import EntropySamplingStrategy
from deepal_for_ecg.strategies.query.plvi_ce import PredictedLabelVectorInconsistencyCrossEntropyStrategy
from deepal_for_ecg.strategies.query.random import RandomQueryStrategy
from deepal_for_ecg.train.test_util import test_step_with_sliding_windows
from deepal_for_ecg.train.time_series import MultiLabelTimeSeriesTrainer


class SelectionStrategy(Enum):
    PLVI_CE_KNN = "plvi_ce_knn"
    PLVI_CE_TOPK = "plvi_ce_topk"
    RANDOM = "random"
    ENTROPY = "entropy"
    BADGE = "badge"


@dataclass
class SelectionStrategyExperimentConfig:
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
    # initialisation
    num_initial_samples: int = 300
    initialisation_strategy: InitializationStrategy = InitializationStrategy.RANDOM
    load_model: bool = False
    start_model_dir: Path | None = None
    # data
    ptbxl_data_base_dir: Path = Path("./data/saved/ptbxl")


@dataclass
class SelectionStrategyExperimentALIterationResult:
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


class SelectionStrategyExperiment:

    def __init__(self, config: SelectionStrategyExperimentConfig):
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
        # get the start model that is used in each active learning iteration
        self.start_model = self._create_model()
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
        self._test_loss = keras.metrics.Mean(name="validation_loss")
        self._test_accuracy = keras.metrics.BinaryAccuracy(name="validation_accuracy")
        self._test_auc = keras.metrics.AUC(multi_label=True, name="validation_auc",
                                           num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES)
        # results
        self._auc_results = []
        self._accuracy_results = []
        self._loss_results = []
        self._coverage_results = []
        self._training_times = []
        self._sampling_times = []

        # save initialisation
        with open(Path(self._experiment_dir, "config.pkl"), "wb") as data_file:
            pickle.dump(config, data_file)

    def run(self):
        self._print_step(f"AL initialisation")
        self._run_al_iteration(name="initial", selection=self._initial_selection, al_iteration=0)
        for i in range(1, self.config.num_al_iterations+1):
            self._print_step(f"AL iteration {i}")
            self._run_al_iteration(name=f"al_iteration_{i}", selection=self._al_selection, al_iteration=i)

        self._save_final_results()

    def _run_al_iteration(self, name: str, selection: Callable, al_iteration: int):
        if self._is_iteration_complete(al_iteration):
            result = self._load_previous_result(self._get_result_file(al_iteration))
            self._load_best_model(name)
            self._data_module.update_annotations(buy_idx_ptb_xl=result.newly_selected_samples, buy_idx_12sl=set())
            print(f"Found result: "
                  f"{result.auc = }, "
                  f"{result.accuracy = }, "
                  f"{result.loss = }, "
                  f"{result.label_coverage = }, "
                  f"{result.selection_time = }, "
                  f"{result.training_time = }")
            self._auc_results.append(result.auc)
            self._accuracy_results.append(result.accuracy)
            self._loss_results.append(result.loss)
            self._coverage_results.append(result.label_coverage)
            self._training_times.append(result.training_time)
            self._sampling_times.append(result.selection_time)
        else:
            self._print_step("sampling")
            sample_selection_start = time.time()
            selected_samples = selection()
            sampling_time = time.time() - sample_selection_start
            expected_num_samples = self.config.num_al_samples if al_iteration > 0 else self.config.num_initial_samples
            assert len(selected_samples) == expected_num_samples, f"{len(selected_samples) = }, {expected_num_samples = }"
            self._sampling_times.append(sampling_time)
            self._data_module.update_annotations(buy_idx_ptb_xl=selected_samples, buy_idx_12sl=set())
            coverage = self._data_module.calculate_label_coverage_ptbxl()
            self._coverage_results.append(coverage)
            print(f"{coverage = }")
            print(f"labeled samples: {len(self._data_module.state_dict()['labeled_indices_ptb_xl'])}")
            print(f"sampling took {sampling_time:.4f} seconds")

            self._print_step("training")
            training_start = time.time()
            self._best_model = self._train(name=name)
            training_time = time.time() - training_start
            self._training_times.append(training_time)
            print(f"training took {training_time:.4f} seconds")

            self._print_step("testing")
            self._test()
            self._save_al_iteration_results(al_iteration, selected_samples)
        print("")
        print("")

    def _print_step(self, heading: str):
        """Prints a heading in a unified way to std out."""
        print(f"{heading:=^30}")

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

    def _initial_selection(self) -> Set[int]:
        """Performs the selection of the first samples."""
        if self.config.initialisation_strategy == InitializationStrategy.RANDOM:
            strategy = RandomQueryStrategy()
            unlabeled_indices = self._data_module.state_dict()["unlabeled_indices"]
            return strategy.select_samples(self.config.num_initial_samples, unlabeled_indices)
        else:
            raise NotImplementedError(f"The initialisation strategy {self.config.initialisation_strategy} is not implemented.")

    def _train(self, name: str) -> keras.Model:
        """
        Trains the model on the currently selected data points.

        Args:
            name (str): The name of the active learning iteration.
        Returns: The best model of the current active learning iteration that was selected using the validation dataset.
        """
        current_model = keras.models.clone_model(self.start_model)
        trainer = MultiLabelTimeSeriesTrainer(model=current_model, model_name=name,
                                              num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES,
                                              model_base_dir=str(self._model_dir),
                                              log_base_dir=str(self._tensorboard_logs_dir), epochs=100)
        trainer.experiment_name = name
        trainer.fit(self._data_module.train_dataset, self._data_module.validation_dataset, verbose=True)
        return trainer.get_model(best=True)

    def _create_model(self) -> keras.Model:
        """
        Returns the model that should be used as a starting point for each iteration.
        Can either create a new model for that purpose or use an existing one.
        If a new model is created the model is saved.

        Returns: The model that should be used as a starting model in each active learning iteration.
        """
        if self.config.load_model and not (self.config.start_model_dir is None):
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

    def _reset_metrics(self):
        self._test_loss.reset_states()
        self._test_accuracy.reset_states()
        self._test_auc.reset_states()

    def _test(self):
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

    def _save_al_iteration_results(self, al_iteration: int, selected_samples: Set[int]):
        auc = self._test_auc.result().numpy()
        loss = self._test_loss.result().numpy()
        accuracy = self._test_accuracy.result().numpy()
        print(f"{auc = }, {accuracy = }, {loss = }")

        self._auc_results.append(auc)
        self._loss_results.append(loss)
        self._accuracy_results.append(accuracy)

        result = SelectionStrategyExperimentALIterationResult(
            auc=auc,
            loss=loss,
            accuracy=accuracy,
            label_coverage=self._coverage_results[-1],
            training_time=self._training_times[-1],
            selection_time=self._sampling_times[-1],
            all_selected_samples=self._data_module.state_dict()["labeled_indices_ptb_xl"],
            newly_selected_samples=selected_samples,
            al_iteration=al_iteration
        )

        with open(self._get_result_file(al_iteration), "wb") as data_file:
            pickle.dump(result, data_file)

    def _save_final_results(self):
        self._plot_metric(self._auc_results, "Macro AUC")
        self._plot_metric(self._accuracy_results, "Binary accuracy")
        self._plot_metric(self._loss_results, "Loss")
        self._plot_metric(self._coverage_results, "Label coverage")
        self._plot_metric(self._training_times, "Training time")
        self._plot_metric(self._sampling_times, "Sampling time")

        data_dict = {
            "auc": self._auc_results,
            "accuracy": self._accuracy_results,
            "loss": self._loss_results,
            "coverage": self._coverage_results,
            "training_time": self._training_times,
            "sampling_time": self._sampling_times
        }
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

    def _is_iteration_complete(self, al_iteration: int) -> bool:
        """Checks whether an iteration was completed earlier."""
        return self._get_result_file(al_iteration).exists()

    def _get_result_file(self, al_iteration: int) -> Path:
        """Returns the path to the result file of the given iteration."""
        return Path(self._results_dir, f"al_iteration_{al_iteration}.pkl")

    def _load_previous_result(self, result_file_path: Path) -> SelectionStrategyExperimentALIterationResult:
        """Loads the previous run result."""
        with open(result_file_path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def _load_best_model(self, name: str) -> keras.Model:
        self._best_model = keras.models.load_model(Path(self._model_dir, name, "best_model.keras"))
