import logging
from typing import Dict, Set

from funcy import log_durations
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from scipy.spatial import distance
from tensorflow import keras

from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.models.util import get_representation_part_of_model

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
        self._representation_model = None
        self._pred_of_unlabeled_samples = None
        self._pred_of_labeled_samples = None
        self._representation_of_unlabeled_samples = None
        self._representation_of_labeled_samples = None

        self._unlabeled_indices = None
        self._labeled_indices = None
        self._labeled_labels = None

        self._num_labeled_samples = None
        self._num_unlabeled_samples = None
        self._num_labels = None

        self._batch_size = 256

    @log_durations(print, threshold=0.5)
    def select_samples(
        self,
        num_of_samples: int,
        data_module: PTBXLActiveLearningDataModule,
        data_loader: PTBXLDataLoader,
        model: keras.Model,
        top_k_selection: bool = True
    ) -> Set[int]:
        """
        Select samples based on a diversity and an uncertainty measure for the next active learning iteration.
        If top_k_selection is set to False, samples are selected by clustering the feature space of the unlabeled data
        and selecting the top sample from each cluster.

        Args:
            num_of_samples (int): The number of samples that should be selected.
            data_module (PTBXLActiveLearningDataModule): The data module that provides the current state of the
                selected samples.
            data_loader (PTBXLDataLoader): The data loader that provides the data.
            model (keras.Model): The model that was trained during the last active learning iteration.
            top_k_selection (bool): Indicator whether to use a top-k selection or knn clusters to sample batches of
                unlabeled data.

        Returns:
            The selected sample indices.
        """
        # prepare the current selection
        print("Start selecting samples")
        self._model = model
        self._representation_model = get_representation_part_of_model(self._model)
        self._set_predictions(data_module)
        self._set_representations(data_module)
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
        if top_k_selection:
            sorted_criteria_idx = np.flip(np.argsort(selection_criteria))
            top_k_indices = sorted_criteria_idx[:num_of_samples]
            query_indices = np.array(self._unlabeled_indices)[top_k_indices]
            return set(query_indices)
        else:
            # k-means clustering of representations
            kmeans = KMeans(n_clusters=num_of_samples)
            kmeans.fit(self._representation_of_unlabeled_samples)
            # get from each cluster the top sample
            cluster_labels = kmeans.labels_
            full_cluster_indices = np.arange(self._num_unlabeled_samples, dtype=int)
            selected_samples = set()
            # TODO: Fix returned indices
            for center_idx, _ in enumerate(kmeans.cluster_centers_):
                cluster_selection_condition = cluster_labels == center_idx
                cluster_indices = full_cluster_indices[cluster_selection_condition]
                cluster_selection_criteria = selection_criteria[cluster_selection_condition]
                max_index = np.argmax(cluster_selection_criteria)
                selected_samples.add(int(self._unlabeled_indices[cluster_indices[max_index]]))
            return selected_samples

    @log_durations(print, threshold=0.5)
    def _calc_uncertainty(self) -> np.ndarray:
        """
        Calculates the uncertainty of each unlabeled sample by comparing the current prediction with a propagated label.
        """
        propagated_labels = self._propagate_labels()
        # TODO: Discuss whether euclidean distance can be used instead of hamming distance since I do not have 0/1 outputs for the prediction
        return np.linalg.norm(
            self._pred_of_unlabeled_samples - propagated_labels, axis=1
        )

    @log_durations(print, threshold=0.5)
    def _calc_diversity(self) -> np.ndarray:
        """
        Calculates the diversity of each unlabeled sample by taking the cross entropy between each unlabeled sample
        prediction and the probabilities of all labeled samples.
        """
        # clip to prevent errors due to log(0)
        labeled_probability = np.clip(self._pred_of_labeled_samples, 1e-15, 1)

        d = np.zeros(self._num_unlabeled_samples)
        for i in range(self._num_unlabeled_samples):
            d[i] = np.sum(
                self._pred_of_unlabeled_samples[i] * np.log(labeled_probability)
            )

        # Adjust the calculation and return the result
        return -d / (self._num_labels * self._num_labeled_samples)

    @log_durations(print, threshold=0.5)
    def _propagate_labels(self) -> np.ndarray:
        """
        Propagate labels by selecting the neighborhood of each unlabeled sample, using the labels of the labeled samples
        in the neighborhood to calculate a soft label based on the distance between the samples and hardcode it with
        regard to the average label cardinality in the neighborhood.
        """
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
        # TODO: Discuss whether to use a global label cardinality like the paper or a neighborhood specific.
        average_label_cardinality_of_neighborhood = np.ceil(
            np.sum(self._labeled_labels[knn_indices[:], :], axis=(1, 2)) / k
        ).astype(int)

        # hardcode the label propagation
        hard_label_propagation = np.zeros(
            (self._num_unlabeled_samples, self._num_labels)
        ).astype(int)
        sorted_confidence_idx = np.flip(np.argsort(soft_label_propagation), axis=1)
        for i in range(self._num_unlabeled_samples):
            hard_label_propagation[i, sorted_confidence_idx[i, 0:average_label_cardinality_of_neighborhood[i]]] = 1
        return hard_label_propagation

    @log_durations(print, threshold=0.5)
    def _calc_rbf_distances(self, sigma: float) -> np.ndarray:
        x = self._representation_of_unlabeled_samples
        y = self._representation_of_labeled_samples
        # I use here the l2-distance here instead of the l1-distance and use the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 * x^T * y
        x_norm = np.sum(x ** 2, axis=-1)
        y_norm = np.sum(y ** 2, axis=-1)
        return np.exp((x_norm[:, None] + y_norm[None, :] - 2 * np.dot(x, y.T))/(-2 * sigma**2))

    def _calc_neighborhood_size(self):
        return np.ceil(np.sqrt(self._num_labeled_samples)).astype(int)

    @log_durations(print, threshold=0.5)
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

    @log_durations(print, threshold=0.5)
    def _rbf_distance(self, x: np.ndarray, y: np.ndarray, sigma: float):
        """Calculates the RBF distance between two datapoints."""
        diff = x - y
        rbf_distance = np.exp(np.sum(diff**2) / (-2 * sigma**2))
        return rbf_distance

    @log_durations(print, threshold=0.5)
    def _calc_sigma(self):
        """
        Calculates the value of sigma for the current data.
        Sigma is used during the calculation of the RDF distances.
        """
        d = distance.cdist(self._representation_of_labeled_samples, self._representation_of_labeled_samples)
        num_divisor = (self._num_labeled_samples * (self._num_labeled_samples - 1)) / 2
        return np.sum(np.triu(d, k=1)) / num_divisor

    @log_durations(print, threshold=0.5)
    def _set_predictions(self, data_module: PTBXLActiveLearningDataModule):
        # TODO: Discuss whether i should use random crop or sliding window aggregation
        self._pred_of_unlabeled_samples = self._get_sliding_window_predictions(
            data_module.unlabeled_sliding_window_sample_dataset
        )

        self._pred_of_labeled_samples = self._get_sliding_window_predictions(
            data_module.labeled_sliding_window_sample_dataset
        )

    @log_durations(print, threshold=0.5)
    def _set_representations(self, data_module: PTBXLActiveLearningDataModule):
        self._representation_of_unlabeled_samples = self._get_representations(
            data_module.unlabeled_sliding_window_sample_dataset)
        self._representation_of_labeled_samples = self._get_representations(
            data_module.labeled_sliding_window_sample_dataset)

    @log_durations(print, threshold=0.5)
    def _get_sliding_window_predictions(self, sliding_window_ds: tf.data.Dataset):
        pred = []
        for batch in sliding_window_ds.batch(self._batch_size):
            pred.append(self._aggregate_sliding_window_predictions(batch))
        return np.concatenate(pred, axis=0)

    def _aggregate_sliding_window_predictions(self, time_series_batch: tf.data.Dataset):
        sliding_window_predictions = []
        for sliding_window in time_series_batch:
            sliding_window_predictions.append(
                self._model(sliding_window, training=False)
            )

        return tf.reduce_max(sliding_window_predictions, axis=0).numpy()

    @log_durations(print, threshold=0.5)
    def _set_data(self, data_loader: PTBXLDataLoader, state: Dict[str, Set]):
        self._unlabeled_indices = list(state["unlabeled_indices"])
        self._labeled_indices = list(state["labeled_indices_ptb_xl"])
        self._labeled_labels = data_loader.Y_train_ptb_xl[self._labeled_indices]

        self._num_labeled_samples = self._representation_of_labeled_samples.shape[0]
        self._num_unlabeled_samples = self._representation_of_unlabeled_samples.shape[0]
        self._num_labels = self._labeled_labels.shape[1]

    @log_durations(print, threshold=0.5)
    def _get_representations(self, sliding_window_ds: tf.data.Dataset) -> np.ndarray:
        """
        Gets the mean representations of each sample in the dataset.

        Args:
            sliding_window_ds (tf.data.Dataset): A dataset where each sample has multiple windows that must be cumulated.
        Returns: The representations of each sample as a numpy array with the dimensions [number of samples,
            representation dimension]
        """
        all_sample_representation = []
        for sample_batch in sliding_window_ds.batch(self._batch_size):
            sliding_window_predictions = []
            for sliding_window in sample_batch:
                sliding_window_predictions.append(
                    self._representation_model(sliding_window, training=False)
                )
            # TODO: Discuss whether I can use the mean to get a single representation for each sample
            sample_representations = tf.reduce_mean(sliding_window_predictions, axis=0)
            all_sample_representation.append(sample_representations)
        return tf.concat(all_sample_representation, axis=0).numpy()
