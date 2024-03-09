import logging
from typing import Dict, Set

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.augmentation import generate_sliding_window


logger = logging.getLogger(__name__)


class PredictedLabelVectorInconsistencyCrossEntropyStrategy:
    """
    The strategy is inspired by Gu et al. (2023) and their paper "PLVI-CE: a multi-label active learning algorithm with
    simultaneously considering uncertainty and diversity". It combines two different selection criteria to determine
    which unlabeled samples should be selected next. The first selection criteria is based on uncertainty and measures
    the inconsistency between the predicted label and a label propagation from k-nearest neighbors. The second selection
    criteria is based on diversity and measures the cross entropy between the predicted label and all predictions of the
    labeled samples.

    The main differences between the approach in the paper and this implementation are:
    - This implementation does not utilize adopted versions of the label-weighted extreme learning machines (LW_-ELM).
      Instead, InceptionTime networks are used.
    - In the original paper, just one sample is selected in each turn. Since this implementation is used in a batch-mode
      setting, multiple samples are returned.
    """

    def __init__(self):
        self._model = None
        self._pred_of_unlabeled_samples = None
        self._pred_of_labeled_samples = None
        self._unlabeled_indices = None
        self._labeled_indices = None
        self._unlabeled_samples = None
        self._labeled_samples = None
        self._labeled_labels = None

        self._num_labeled_samples = None
        self._num_unlabeled_samples = None
        self._num_labels = None

    def select_samples(
        self, num_of_samples: int, data_module, data_loader, model: keras.Model
    ) -> Set[int]:
        # prepare the current selection
        self._model = model
        self._set_predictions(data_module)
        self._set_data(data_loader, data_module.state_dict())

        # get the measures
        uncertainty_measure = self._calc_uncertainty()
        diversity_measure = self._calc_diversity()

        # select the samples
        # TODO: In the original paper just one unlabeled sample was selected for annotation
        # Start with the top-k selection criteria samples
        # Test an incorporation of k-means clustering and selecting in each cluster the top selection criteria sample
        # that extend the uncertain + divers selection with representative selection
        selection_criteria = uncertainty_measure * diversity_measure
        sorted_criteria_idx = np.flip(np.argsort(selection_criteria))
        top_k_indices = sorted_criteria_idx[:num_of_samples]
        query_indices = np.array(self._unlabeled_indices)[top_k_indices]
        return set(query_indices)

    def _calc_uncertainty(self) -> np.ndarray:
        propagated_labels = self._propagate_labels()
        # TODO: Discuss whether euclidean distance can be used instead of hamming distance since I do not have 0/1 outputs for the prediction
        return np.linalg.norm(
            self._pred_of_unlabeled_samples - propagated_labels, axis=1
        )

    def _calc_diversity(self) -> np.ndarray:
        # clip to prevent errors due to log(0)
        labeled_probability = np.clip(self._pred_of_labeled_samples, 1e-15, 1)

        d = np.zeros(self._num_unlabeled_samples)
        for i in range(self._num_unlabeled_samples):
            d[i] = np.sum(
                self._pred_of_unlabeled_samples[i] * np.log(labeled_probability)
            )

        # Adjust the calculation and return the result
        return -d / (self._num_labels * self._num_labeled_samples)

    def _propagate_labels(self) -> np.ndarray:
        sigma = self._calc_sigma()
        rbf_distances = self._calc_rbf_distances(sigma)
        k = self._calc_neighborhood_size()

        # get the top-k labeled instances with largest RBF-distance
        distances_sorted = np.flip(
            np.argsort(rbf_distances, axis=1), axis=1
        )  # first sort in ascending order, then reverse it so that max value is first
        knn_indices = distances_sorted[:, 0:k]

        soft_label_propagation = self._propagate_soft_label(rbf_distances, knn_indices)
        # calculate the average label cardinality of each neighborhood
        average_label_cardinality_of_neighborhood = np.ceil(
            np.sum(self._labeled_labels[knn_indices[:], :], axis=(1, 2)) / k
        ).astype(int)

        # hardcode the label propagation
        hard_label_propagation = np.zeros(
            (self._num_unlabeled_samples, self._num_labels)
        ).astype(int)
        sorted_confidence_idx = np.flip(np.argsort(soft_label_propagation), axis=1)
        for i in range(self._num_unlabeled_samples):
            hard_label_propagation[
                i,
                sorted_confidence_idx[
                    i, 0 : average_label_cardinality_of_neighborhood[i]
                ],
            ] = 1
        return hard_label_propagation

    def _calc_rbf_distances(self, sigma):
        rbf_distances = np.zeros(
            (self._num_unlabeled_samples, self._num_labeled_samples)
        )
        for i in range(self._num_unlabeled_samples):
            for j in range(i, self._num_labeled_samples):
                dist = self._rbf_distance(
                    self._unlabeled_samples[i], self._labeled_samples[j], sigma
                )
                rbf_distances[i, j] = dist
                rbf_distances[j, i] = dist
        return rbf_distances

    def _calc_neighborhood_size(self):
        return np.ceil(np.sqrt(self._num_labeled_samples)).astype(int)

    def _propagate_soft_label(
        self, rbf_distances: np.ndarray, knn_indices: np.ndarray
    ) -> np.ndarray:
        soft_label_propagation = np.zeros(
            (self._num_unlabeled_samples, self._num_labels)
        )
        for i in range(self._num_unlabeled_samples):
            soft_label_propagation[i, :] = np.sum(
                rbf_distances[i, knn_indices[i]]
                * self._labeled_labels[knn_indices[i], :].T,
                axis=1,
            )
        return soft_label_propagation

    def _rbf_distance(self, x, y, sigma):
        diff = x - y
        rbf_distance = np.exp(np.sum(diff**2) / (-2 * sigma**2))
        return rbf_distance

    def _calc_sigma(self):
        """
        Calculates the value of sigma for the current data.
        Sigma is used during the calculation of the RDF distances.
        """
        dist_list = []
        for i in range(self._num_labeled_samples - 1):
            for j in range(i + 1, self._num_labeled_samples):
                dist = np.linalg.norm(
                    self._labeled_samples[i] - self._labeled_samples[j]
                )
                dist_list.append(dist)
        num_divisor = (self._num_labeled_samples * (self._num_labeled_samples - 1)) / 2
        return np.sum(dist_list) / num_divisor

    def _set_predictions(self, data_module):
        _pred_of_unlabeled_samples = []

        # TODO: Discuss whether i should use random crop or sliding window aggregation
        unlabeled_np_iterator = data_module.unlabeled_dataset.as_numpy_iterator()
        self._pred_of_unlabeled_samples = self._get_sliding_window_predictions(
            unlabeled_np_iterator
        )

        labeled_np_iterator = data_module.train_dataset.map(
            lambda x, _: x
        ).as_numpy_iterator()
        self._pred_of_labeled_samples = self._get_sliding_window_predictions(
            labeled_np_iterator
        )

    def _get_sliding_window_predictions(self, numpy_iterator):
        data = np.concatenate([[sample] for sample in numpy_iterator], axis=0)
        sliding_window_ds = tf.data.Dataset.from_tensor_slices(
            generate_sliding_window(data, window_size=250, stride=125)
        )
        pred = []
        for batch in sliding_window_ds.batch(128):
            pred.append(self._aggregate_sliding_window_predictions(batch))
        return np.concatenate(pred, axis=0)

    def _aggregate_sliding_window_predictions(self, time_series_batch: tf.data.Dataset):
        sliding_window_predictions = []
        for sliding_window in time_series_batch:
            sliding_window_predictions.append(
                self._model(sliding_window, training=False)
            )

        return tf.reduce_max(sliding_window_predictions, axis=0).numpy()

    def _set_data(self, data_loader, state: Dict[str, Set]):
        self._unlabeled_indices = list(state["unlabeled_indices"])
        self._labeled_indices = list(state["labeled_indices_ptb_xl"])
        self._unlabeled_samples = data_loader.X_train[self._unlabeled_indices]
        self._labeled_samples = data_loader.X_train[self._labeled_indices]
        self._labeled_labels = data_loader.Y_train_ptb_xl[self._labeled_indices]

        self._num_labeled_samples = self._labeled_samples.shape[0]
        self._num_unlabeled_samples = self._unlabeled_samples.shape[0]
        self._num_labels = self._labeled_labels.shape[1]
