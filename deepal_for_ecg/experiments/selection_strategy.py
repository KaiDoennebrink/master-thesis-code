from dataclasses import dataclass, field
from typing import Set


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
