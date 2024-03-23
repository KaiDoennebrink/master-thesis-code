from dataclasses import dataclass
from pathlib import Path
from typing import Set

from deepal_for_ecg.experiments.base import BaseExperimentConfig, BaseExperiment, BaseExperimentALIterationResult
from deepal_for_ecg.strategies.initalize import InitializationStrategy
from deepal_for_ecg.strategies.query import SelectionStrategy


@dataclass
class WSAExperimentConfig(BaseExperimentConfig):
    strategy: SelectionStrategy = SelectionStrategy.ENTROPY
    # directories
    base_experiment_dir: Path = Path("./experiments/wsa")
    # initialisation
    init_strategy: InitializationStrategy = InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT
    init_strategy_pretrained_model_dir: Path | None = Path("./models")
    init_strategy_pretrained_model_base_name: str | None = "PretextInception"


class WeakSupervisionAnnotatorExperiment(BaseExperiment):
    """Selects samples from the weak supervision annotator instead of the human annotator."""

    def __init__(self, config: WSAExperimentConfig):
        super().__init__(config)
        self.config = config

    def _buy_samples(self, samples_to_buy: Set[int]):
        """Buys sample from the weak supervision annotator (12 SL)."""
        self._data_module.update_annotations(buy_idx_ptb_xl=set(), buy_idx_12sl=samples_to_buy)

    def _get_al_iteration_result(self, al_iteration: int) -> BaseExperimentALIterationResult:
        """Use the labeled indices from the WSA instead of the HA."""
        return BaseExperimentALIterationResult(
            auc=self._auc_results[-1],
            loss=self._loss_results[-1],
            accuracy=self._accuracy_results[-1],
            label_coverage=self._coverage_results[-1],
            training_time=self._training_times[-1],
            selection_time=self._sampling_times[-1],
            all_selected_samples=self._data_module.state_dict()["labeled_indices_12sl"],
            newly_selected_samples=self._selected_samples,
            al_iteration=al_iteration
        )
