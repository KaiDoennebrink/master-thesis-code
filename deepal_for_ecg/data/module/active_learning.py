from typing import List

import numpy as np
import tensorflow as tf

from deepal_for_ecg.data.augmentation import generate_sliding_window


class PTBXLActiveLearningDataModule:
    """
    The active learning data module for the PTB-XL dataset.
    It provides the labeled and unlabeled data for the active learning process, keeps track of the queried instances,
    and can augment the samples.
    """

    NUM_CLASSES = 90

    def __init__(
        self,
        train_samples: np.ndarray,
        val_samples: np.ndarray,
        test_samples: np.ndarray,
        train_labels_ptb_xl: np.ndarray,
        train_labels_12sl: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray,
        warning_mode_activated: bool = True
    ):
        """Initializes the data module with the previously loaded data."""
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._test_samples = test_samples
        self._train_labels_ptb_xl = train_labels_ptb_xl
        self._train_labels_12sl = train_labels_12sl
        self._val_labels = val_labels
        self._test_labels = test_labels
        self._warning_mode_activated = warning_mode_activated

        self._unlabeled_indices = set(range(len(self._train_samples)))
        self._labeled_indices_ptb_xl = set()
        self._labeled_indices_12sl = set()

        self._validation_dataset = None
        self._test_dataset = None
        self._sliding_window_training_samples = None

    def state_dict(self):
        """Returns the indices of the currently labeled and unlabeled samples."""
        return {
            "unlabeled_indices": self._unlabeled_indices,
            "labeled_indices_ptb_xl": self._labeled_indices_ptb_xl,
            "labeled_indices_12sl": self._labeled_indices_12sl,
        }

    def update_annotations(self, buy_idx_ptb_xl: set, buy_idx_12sl: set):
        """Updates the labeled pool with newly annotated instances."""
        if self._warning_mode_activated:
            # check whether some labels are already present
            intersection_ptbxl = buy_idx_ptb_xl.intersection(self._labeled_indices_ptb_xl)
            if len(intersection_ptbxl) > 0:
                print(f"WARNING: {len(intersection_ptbxl)} samples are already labeled with PTB-XL")
                print(f"indices: {intersection_ptbxl }")
            intersection_12sl = buy_idx_12sl.intersection(self._labeled_indices_12sl)
            if len(intersection_12sl) > 0:
                print(f"WARNING: {len(intersection_12sl)} samples are already labeled with 12SL")
                print(f"indices: {intersection_12sl}")
        self._labeled_indices_ptb_xl = self._labeled_indices_ptb_xl.union(
            buy_idx_ptb_xl
        )
        self._labeled_indices_12sl = self._labeled_indices_12sl.union(buy_idx_12sl)
        self._unlabeled_indices = self._unlabeled_indices.difference(
            buy_idx_ptb_xl.union(buy_idx_12sl)
        )

    def prepare_sliding_windows_data(self):
        """Prepares the training data to be used with sliding windows."""
        self._sliding_window_training_samples = generate_sliding_window(self._train_samples, window_size=250, stride=125)

    def calculate_label_coverage_ptbxl(self):
        """Calculates the label coverage of the currently selected PTB-XL instances."""
        selected_indices = list(self._labeled_indices_ptb_xl)
        selected_labels = self._train_labels_ptb_xl[selected_indices]
        samples_per_label = np.sum(selected_labels, axis=0)
        return np.sum(samples_per_label >= 1) / PTBXLActiveLearningDataModule.NUM_CLASSES

    @property
    def label_cardinality(self) -> int:
        """
        Returns the average label cardinality of the currently labeled samples.
        """
        current_labels_12_sl = self._train_labels_12sl[list(self._labeled_indices_12sl)]
        current_labels_ptb_xl = self._train_labels_ptb_xl[list(self._labeled_indices_ptb_xl)]
        num_of_labels = np.sum(current_labels_12_sl) + np.sum(current_labels_ptb_xl)
        num_of_samples = len(self._labeled_indices_12sl) + len(self._labeled_indices_ptb_xl)
        return int(np.ceil(num_of_labels / num_of_samples))

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        if self._validation_dataset is None:
            self._validation_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    generate_sliding_window(
                        self._val_samples, window_size=250, stride=125
                    ),
                    self._val_labels,
                )
            )
        return self._validation_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        if self._test_dataset is None:
            self._test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    generate_sliding_window(
                        self._test_samples, window_size=250, stride=125
                    ),
                    self._test_labels,
                )
            )
        return self._test_dataset

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        Constructs and returns the current training dataset without data augmentations.
        It contains the labels from the selected PTB-XL and 12SL samples.
        """
        idx_ptb_xl = list(self._labeled_indices_ptb_xl)
        train_dataset_ptb_xl = tf.data.Dataset.from_tensor_slices(
            (self._train_samples[idx_ptb_xl], self._train_labels_ptb_xl[idx_ptb_xl])
        )

        idx_12sl = list(self._labeled_indices_12sl)
        train_dataset_12sl = tf.data.Dataset.from_tensor_slices(
            (self._train_samples[idx_12sl], self._train_labels_12sl[idx_12sl])
        )

        return train_dataset_ptb_xl.concatenate(train_dataset_12sl)

    @property
    def unlabeled_dataset(self) -> tf.data.Dataset:
        """
        Constructs and returns the current unlabeled dataset without data augmentations.
        """
        indices = list(self._unlabeled_indices)
        return tf.data.Dataset.from_tensor_slices((self._train_samples[indices]))

    @property
    def unlabeled_sliding_window_sample_dataset(self) -> tf.data.Dataset:
        """Constructs and returns the current unlabeled dataset as a sliding window dataset."""
        indices = list(self._unlabeled_indices)
        return self._get_sliding_window_dataset(indices)

    @property
    def labeled_sliding_window_sample_dataset(self) -> tf.data.Dataset:
        """Constructs and returns the current labeled dataset as a sliding window dataset."""
        indices = list(self._labeled_indices_ptb_xl.union(self._labeled_indices_12sl))
        return self._get_sliding_window_dataset(indices)

    def _get_sliding_window_dataset(self, indices: List[int]) -> tf.data.Dataset:
        if self._sliding_window_training_samples is None:
            self.prepare_sliding_windows_data()
        labeled_data = tuple([sample[indices] for sample in self._sliding_window_training_samples])
        return tf.data.Dataset.from_tensor_slices(labeled_data)