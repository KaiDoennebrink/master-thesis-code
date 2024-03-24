from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from deepal_for_ecg.strategies.annotator.agreement import measure_relative_agreement_to_positive_true_labels


class AnnotatorDataModule:
    """
    The data module to train the annotator model with.
    """

    def __init__(self, agreement_method: Callable = measure_relative_agreement_to_positive_true_labels, agreement_threshold: float = 1.0, test_size: float = 0.1):
        self._agreement_method = agreement_method
        self._agreement_threshold = agreement_threshold
        self._test_size = test_size

        self._train_samples = None
        self._val_samples = None
        self._train_labels = None
        self._val_labels = None

        self._batch_size = 1000

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """Constructs and returns the current training dataset."""
        return tf.data.Dataset.from_tensor_slices((self._train_samples, self._train_labels)).batch(self._batch_size)

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        """Constructs and returns the current validation dataset."""
        return tf.data.Dataset.from_tensor_slices((self._val_samples, self._val_labels)).batch(self._batch_size)

    def update_data(self, ha_labels: np.ndarray, wsa_labels: np.ndarray):
        """
        Updates the data module with new additional datapoints.

        Args:
            ha_labels (np.ndarray): The labels from the human annotator.
            wsa_labels (np.ndarray): The labels from the weak supervision annotator.
        """
        agreement_measure = self._agreement_method(ha_labels, wsa_labels)
        annotator_labels = self._decide_which_label_to_take(agreement_measure).astype(int).reshape((-1, 1))
        print(f"Update labels with {np.sum(annotator_labels)} samples where to select WSA.")
        num_of_new_datapoints = annotator_labels.shape[0]
        if num_of_new_datapoints == 1:
            if self._train_samples is None:
                self._train_samples = wsa_labels
                self._train_labels = annotator_labels
            else:
                self._train_samples = np.concatenate((self._train_samples, wsa_labels), axis=0)
                self._train_labels = np.concatenate((self._train_labels, annotator_labels), axis=0)
        elif num_of_new_datapoints > 1:
            new_train_samples, new_val_samples, new_train_labels, new_val_labels = train_test_split(wsa_labels, annotator_labels, test_size=self._test_size)
            if self._train_samples is None:
                self._train_samples = new_train_samples
                self._val_samples = new_val_samples
                self._train_labels = new_train_labels
                self._val_labels = new_val_labels
            else:
                self._train_samples = np.concatenate((self._train_samples, new_train_samples), axis=0)
                self._val_samples = np.concatenate((self._val_samples, new_val_samples), axis=0)
                self._train_labels = np.concatenate((self._train_labels, new_train_labels), axis=0)
                self._val_labels = np.concatenate((self._val_labels, new_val_labels), axis=0)

    def _decide_which_label_to_take(self, agreement_measure: np.ndarray) -> np.ndarray:
        return agreement_measure >= self._agreement_threshold
