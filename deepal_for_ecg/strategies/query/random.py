from typing import Set

import numpy as np


class RandomQueryStrategy:
    """
    This query strategy selects random samples from a pool of unlabeled query strategies.
    """

    def __init__(self):
        self._rng = np.random.default_rng()

    def select_samples(self, num_of_samples: int, unlabeled_indices: Set[int]) -> Set[int]:
        """
        Selects random samples from the pool of unlabeled samples.

        Args:
            num_of_samples (int): The number of samples that should be selected.
            unlabeled_indices (set[int]): The indices of the unlabeled samples pool.
        Returns:
            A set of selected sample indices.
        """
        return set(self._rng.choice(list(unlabeled_indices), num_of_samples))
