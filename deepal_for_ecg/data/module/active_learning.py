import numpy as np
import tensorflow as tf

from deepal_for_ecg.data.augmentation import generate_sliding_window


class PTBXLActiveLearningDataModule:
    """
    The active learning data module for the PTB-XL dataset.
    It provides the labeled and unlabeled data for the active learning process, keeps track of the queried instances,
    and can augment the samples.
    """
    NUM_CLASSES = 93

    def __init__(
            self,
            train_samples: np.ndarray,
            val_samples: np.ndarray,
            test_samples: np.ndarray,
            train_labels_ptb_xl: np.ndarray,
            train_labels_12sl: np.ndarray,
            val_labels: np.ndarray,
            test_labels: np.ndarray
    ):
        """Initializes the data module with the previously loaded data."""
        self._train_samples = train_samples
        self._val_samples = val_samples
        self._test_samples = test_samples
        self._train_labels_ptb_xl = train_labels_ptb_xl
        self._train_labels_12sl = train_labels_12sl
        self._val_labels = val_labels
        self._test_labels = test_labels

        self._unlabeled_indices = set(range(len(self._train_samples)))
        self._labeled_indices_ptb_xl = set()
        self._labeled_indices_12sl = set()

        self._validation_dataset = None
        self._test_dataset = None

    def state_dict(self):
        """Returns the indices of the currently labeled and unlabeled samples."""
        return {
            'unlabeled_indices': self._unlabeled_indices,
            'labeled_indices_ptb_xl': self._labeled_indices_ptb_xl,
            'labeled_indices_12sl': self._labeled_indices_12sl
        }

    def update_annotations(self, buy_idx_ptb_xl: set, buy_idx_12sl: set):
        """Updates the labeled pool with newly annotated instances."""
        self._labeled_indices_ptb_xl = self._labeled_indices_ptb_xl.union(buy_idx_ptb_xl)
        self._labeled_indices_12sl = self._labeled_indices_12sl.union(buy_idx_12sl)
        self._unlabeled_indices = self._unlabeled_indices.difference(buy_idx_ptb_xl.union(buy_idx_12sl))

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        if self._validation_dataset is None:
            self._validation_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    generate_sliding_window(self._val_samples, window_size=250, stride=125),
                    self._val_labels
                )
            )
        return self._validation_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        if self._test_dataset is None:
            self._test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    generate_sliding_window(self._test_samples, window_size=250, stride=125),
                    self._test_labels
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
