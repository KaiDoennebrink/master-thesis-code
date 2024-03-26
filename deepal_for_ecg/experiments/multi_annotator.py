from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Set

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.data.module.annotator import AnnotatorDataModule
from deepal_for_ecg.experiments.base import BaseExperimentConfig, BaseExperiment, BaseExperimentALIterationResult
from deepal_for_ecg.models.classification_heads import annotator_model, BasicClassificationHeadConfig
from deepal_for_ecg.models.util import get_representation_part_of_model
from deepal_for_ecg.strategies.annotator import HybridAnnotatorModelSetting
from deepal_for_ecg.strategies.annotator.selection import AnnotatorSelectionStrategy
from deepal_for_ecg.strategies.initalize import InitializationStrategy
from deepal_for_ecg.strategies.query import SelectionStrategy
from deepal_for_ecg.strategies.annotator.agreement import measure_relative_agreement_to_positive_true_labels


@dataclass
class HybridAnnotatorSelectionExperimentConfig(BaseExperimentConfig):
    run_number: int = 1
    strategy: SelectionStrategy = SelectionStrategy.ENTROPY
    # directories
    base_experiment_dir: Path = Path("./experiments/hybrid_label")
    # initialisation
    init_strategy: InitializationStrategy = InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT
    init_strategy_pretrained_model_dir: Path | None = Path("./models")
    init_strategy_pretrained_model_base_name: str | None = "PretextInception"
    # data
    annotator_agreement_method: Callable = measure_relative_agreement_to_positive_true_labels
    annotator_agreement_threshold: float = 1
    annotator_test_size: float = 0.1
    # annotator selection
    annotator_selection_threshold: float = 0.65
    start_annotator_model_dir: Path | None = None
    # cost factors
    cost_factor_ha: float = 1
    cost_factor_wsa: float = 0.1
    # annotator model
    annotator_model_setting: HybridAnnotatorModelSetting = HybridAnnotatorModelSetting.LABEL_BASED_MODEL
    annotator_model_pretrained_model_dir: Path | None = Path("./models")
    annotator_model_pretrained_model_base_name: str | None = "PretextInception"


@dataclass(kw_only=True)
class HybridAnnotatorSelectionExperimentALIterationResult(BaseExperimentALIterationResult):
    # selection
    all_selected_samples_from_wsa: Set[int] = field(default_factory=set)
    newly_selected_samples_from_wsa: Set[int] = field(default_factory=set)

    # Generated attribute
    num_samples_from_wsa: int = field(init=False)
    new_samples_from_wsa: int = field(init=False)

    @property
    def num_samples_from_wsa(self) -> int:
        return len(self.all_selected_samples_from_wsa)

    @num_samples_from_wsa.setter
    def num_samples_from_wsa(self, _):
        pass  # read-only attribute

    @property
    def num_newly_samples_from_wsa(self) -> int:
        return len(self.newly_selected_samples_from_wsa)

    @num_newly_samples_from_wsa.setter
    def num_newly_samples_from_wsa(self, _):
        pass  # read-only attribute


class HybridAnnotatorSelectionExperiment(BaseExperiment):
    """
    Class to test whether the architecture proposed by Rauch et al. (2022) in their paper "Enhancing active learning
    with weak supervision and transfer learning by leveraging information and knowledge sources" can be used in the
    case of multi-label time series data, too.

    Rauch et al. (2022) used a black box model as a second annotation source. The black box model is cheaper as a human
    annotator. But it is also more error prune. They see it therefore as a weak label source. In addition to the weak
    supervision, they added transfer learning to decrease labeling costs further.
    """

    def __init__(self, config: HybridAnnotatorSelectionExperimentConfig):
        super().__init__(config)
        self.config = config
        # model
        self._start_annotator_model = self._get_or_create_start_annotator_model()
        self._best_annotator_model = None
        # data
        self._annotator_data_module = AnnotatorDataModule(
            agreement_method=config.annotator_agreement_method,
            agreement_threshold=config.annotator_agreement_threshold,
            test_size=config.annotator_test_size
        )
        # results
        self._samples_from_ha = []
        self._samples_from_wsa = []
        self._selected_samples_from_wsa = set()
        self._annotator_selector = AnnotatorSelectionStrategy(config.annotator_selection_threshold)
        self._is_initialized: bool = False

    def _get_or_create_start_annotator_model(self) -> keras.Model:
        """
        Returns the model that should be used as a starting point for each iteration.
        Can either create a new model for that purpose or use an existing one.
        If a new model is created the model is saved.

        Returns: The model that should be used as a starting model in each active learning iteration.
        """
        if not (self.config.start_annotator_model_dir is None) and self.config.start_annotator_model_dir.exists():
            return keras.models.load_model(self.config.start_annotator_model_dir)
        elif Path(self._model_dir, "start_annotator_model.keras").exists():
            # load the start model from a previous point
            return keras.models.load_model(Path(self._model_dir, "start_annotator_model.keras"))
        else:
            # if no model was loaded, create a new one
            if self.config.annotator_model_setting == HybridAnnotatorModelSetting.LABEL_BASED_MODEL:
                config = BasicClassificationHeadConfig(num_input_units=PTBXLActiveLearningDataModule.NUM_CLASSES, num_output_units=1)
                model = annotator_model(config)
                model.save(Path(self._model_dir, "start_annotator_model.keras"))
                return model
            else:
                input_layer = keras.layers.Input((1000, 12), name=f"Input")
                pretrained_model = keras.models.load_model(
                    Path(
                        self.config.annotator_model_pretrained_model_dir,
                        f"{self.config.annotator_model_pretrained_model_base_name}{self.config.run_number}",
                        "best_model.keras"
                    )
                )
                pretrained_representation_model = get_representation_part_of_model(pretrained_model)
                pretrained_representation_model.trainable = False
                representation_out = pretrained_representation_model(input_layer)
                config = BasicClassificationHeadConfig(num_input_units=representation_out.shape[1],
                                                       num_output_units=1)
                annotator_model_head = annotator_model(config)
                output_layer = annotator_model_head(representation_out)
                model = keras.Model(inputs=input_layer, outputs=output_layer, name="Annotator_Model_Signal_Based")
                model.save(Path(self._model_dir, "start_annotator_model.keras"))
                return model

    def _load_best_model(self, name: str) -> keras.Model:
        super()._load_best_model(name)
        # Load the best annotator model as well
        self._best_annotator_model = keras.models.load_model(Path(self._model_dir, name, "best_annotator_model.keras"))

    def _update_data_module(self, result: HybridAnnotatorSelectionExperimentALIterationResult):
        """Updates the data module with the previous selections."""
        all_selected_samples = set()
        if result.newly_selected_samples is not None:
            all_selected_samples = all_selected_samples.union(result.newly_selected_samples)
        if result.newly_selected_samples_from_wsa is not None:
            all_selected_samples = all_selected_samples.union(result.newly_selected_samples_from_wsa)
        self._buy_samples(all_selected_samples)

    def _buy_samples(self, selected_indices: Set[int]):
        if not self._is_initialized:
            self._data_module.update_annotations(buy_idx_ptb_xl=selected_indices, buy_idx_12sl=set())
            # update counts
            self._samples_from_ha.append(len(selected_indices))
            self._samples_from_wsa.append(0)
            self._is_initialized = True
            wsa_labels = self._data_module.request_wsa_labels(selected_indices)
            ha_labels = self._data_module.request_ha_labels(list(selected_indices))
            input_data = self._get_input_data_for_data_module(selected_indices)
            self._annotator_data_module.update_data(ha_labels, wsa_labels, input_data)
        else:
            input_data = self._get_input_data_for_data_module(selected_indices)
            ds = tf.data.Dataset.from_tensor_slices(input_data)
            selected_annotators = self._annotator_selector.select_annotator(self._best_annotator_model, ds)
            # update counts
            num_of_wsa_samples = np.sum(selected_annotators)
            num_of_ha_samples = len(selected_annotators) - num_of_wsa_samples
            self._samples_from_ha.append(num_of_ha_samples)
            self._samples_from_wsa.append(num_of_wsa_samples)
            print(f"Selected {num_of_wsa_samples} samples from WSA and {num_of_ha_samples} samples from HA.")
            # update the active learning data module
            all_wsa_labels = self._data_module.request_wsa_labels(selected_indices)
            wsa_indices = list(np.array(list(selected_indices))[selected_annotators])
            ha_indices = list(np.array(list(selected_indices))[~selected_annotators])
            self._data_module.update_annotations(buy_idx_ptb_xl=set(ha_indices), buy_idx_12sl=set(wsa_indices))
            # update the annotator data module
            wsa_labels = all_wsa_labels[~selected_annotators]
            ha_labels = self._data_module.request_ha_labels(ha_indices)
            # TODO: If the experiment is killed and it is rerun there will be different samples in the train and val data
            input_data = self._get_input_data_for_data_module(ha_indices)
            self._annotator_data_module.update_data(ha_labels, wsa_labels, input_data)
            # update instance fields so that the result will be correct
            self._selected_samples = set(ha_indices)
            self._selected_samples_from_wsa = set(wsa_indices)

    def _get_input_data_for_data_module(self, selected_indices):
        if self.config.annotator_model_setting == HybridAnnotatorModelSetting.LABEL_BASED_MODEL:
            return self._data_module.request_wsa_labels(selected_indices)
        elif self.config.annotator_model_setting == HybridAnnotatorModelSetting.SIGNAL_BASED_MODEL:
            return self._data_module.request_raw_signals(selected_indices)
        else:
            raise NotImplementedError(f"Not implemented for {self.config.annotator_model_setting = }")

    def _train(self, name: str):
        super()._train(name)

        model = keras.models.clone_model(self._start_annotator_model)
        loss_fn = keras.losses.BinaryCrossentropy()
        auc = keras.metrics.AUC(name="auc")
        model.compile(optimizer=keras.optimizers.AdamW(), loss=loss_fn, metrics=['accuracy', auc])
        callbacks = []  # [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)]
        model.fit(
            self._annotator_data_module.train_dataset,
            epochs=100,
            validation_data=self._annotator_data_module.validation_dataset,
            callbacks=callbacks,
            class_weight=self._annotator_data_module.get_class_weights(),
            verbose=True
        )
        self._best_annotator_model = model
        self._best_annotator_model.save(Path(self._model_dir, name, "best_annotator_model.keras"))

    def _get_al_iteration_result(self, al_iteration: int) -> HybridAnnotatorSelectionExperimentALIterationResult:
        return HybridAnnotatorSelectionExperimentALIterationResult(
            auc=self._auc_results[-1],
            loss=self._loss_results[-1],
            accuracy=self._accuracy_results[-1],
            label_coverage=self._coverage_results[-1],
            training_time=self._training_times[-1],
            selection_time=self._sampling_times[-1],
            all_selected_samples=self._data_module.state_dict()["labeled_indices_ptb_xl"],
            newly_selected_samples=self._selected_samples,
            al_iteration=al_iteration,
            all_selected_samples_from_wsa=self._data_module.state_dict()["labeled_indices_12sl"],
            newly_selected_samples_from_wsa=self._selected_samples_from_wsa
        )

    def _save_final_results(self):
        super()._save_final_results()
        self._create_plots()

    def _get_final_data_dict(self):
        data_dict = super()._get_final_data_dict()
        data_dict["samples_from_ha"] = self._samples_from_ha
        data_dict["samples_from_wsa"] = self._samples_from_wsa
        return data_dict

    def _create_plots(self):
        # prepare the data
        cumsum_of_ha_samples = np.cumsum(self._samples_from_ha)
        cumsum_of_wsa_samples = np.cumsum(self._samples_from_wsa)
        cumsum_of_all_samples = cumsum_of_ha_samples + cumsum_of_wsa_samples
        hybrid_cost = cumsum_of_ha_samples * (self.config.cost_factor_ha + self.config.cost_factor_wsa) + cumsum_of_wsa_samples * self.config.cost_factor_wsa
        full_human_cost = cumsum_of_all_samples * self.config.cost_factor_ha
        full_wsa_cost = cumsum_of_all_samples * self.config.cost_factor_wsa
        cost = np.array((hybrid_cost, full_human_cost, full_wsa_cost)).T
        cost_df = pd.DataFrame(cost, columns=['hybrid', 'full_human', 'full_wsa'])
        samples = np.array((cumsum_of_ha_samples, cumsum_of_wsa_samples)).T
        samples_df = pd.DataFrame(samples, columns=['human annotator', 'weak supervision annotator'])

        # cost plot
        plt.clf()  # clear previous plots
        fig = plt.figure(1, figsize=(10, 5))
        sns.set_style("whitegrid")
        fig.suptitle(f"Aggregated cost in different settings")
        plt.ylabel("Total cost")
        plt.xlabel("AL iteration")
        sns.lineplot(cost_df, markers=True)
        fig.savefig(Path(self._plots_dir, f"agg_cost_plot.png"))

        # samples plot
        plt.clf()  # clear previous plots
        fig = plt.figure(1, figsize=(10, 5))
        sns.set_style("whitegrid")
        fig.suptitle(f"Aggregated samples from different annotators")
        plt.ylabel("Number of total samples")
        plt.xlabel("AL iteration")
        sns.lineplot(samples_df, markers=True)
        fig.savefig(Path(self._plots_dir, f"agg_samples_plot.png"))

    def _reset_fields(self):
        super()._reset_fields()
        self._selected_samples_from_wsa = set()
