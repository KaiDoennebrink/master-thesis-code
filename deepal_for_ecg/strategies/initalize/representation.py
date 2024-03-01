from typing import Set

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow import keras

from deepal_for_ecg.data.augmentation import random_crop


class RepresentationClusteringInitQueryStrategy:
    """
    This query strategy just can be used to select the initial samples for an active learning cycle. Subsequent samples
    have to be selected by other query strategies. It is not affected by the cold start problem as it utilizes
    representations from a pre-trained model. Parts of the query strategy are inspired by the paper "Making your first
    choice: To address cold start problem in medical active learning" from Chen et al. (2023).

    Chen et al. (2023) utilize in their query strategy HaCon (i) features extracted by contrastive learning, (ii)
    K-means clustering to cluster the features for label diversity, and (iii) hard-to-contrast samples to select samples
    from each cluster. This query strategy just will use the idea of using pre-trained features and the K-means
    clustering to cluster the features for label diversity. Within each cluster the sample that is the nearest to the
    cluster center is selected.
    """

    def __init__(self, pretrained_model: keras.Model, num_clusters: int, augmentation_method: callable = random_crop,
                 augmentation_kwargs: dict | None = None):
        self._pretrained_model = pretrained_model
        self._representation_model = self._get_representation_part_of_model()
        self._num_clusters = num_clusters
        self._kmeans = KMeans(n_clusters=self._num_clusters)
        self._augmentation_method = augmentation_method
        self._augmentation_kwargs = augmentation_kwargs if augmentation_kwargs else dict()
        self._all_sample_representation = None

    def prepare(self, unlabeled_samples: tf.data.Dataset):
        # if the pre-trained representation model expect a smaller input than the original time series length
        # the unlabeled samples need to be augmented with the random crop method for example
        self._all_sample_representation = self._get_representations(unlabeled_samples)
        self._kmeans.fit(self._all_sample_representation)

    def select_samples(self) -> Set[int]:
        """Selects the sample from each cluster that is the nearest sample to the cluster center."""
        cluster_labels = self._kmeans.labels_
        orig_indices = np.arange(len(cluster_labels), dtype=int)
        selected_samples = set()
        for center_idx, center in enumerate(self._kmeans.cluster_centers_):
            cluster_selection_condition = cluster_labels == center_idx
            cluster_points = self._all_sample_representation[cluster_selection_condition]
            orig_cluster_point_indices = orig_indices[cluster_selection_condition]

            distances = np.linalg.norm(cluster_points - center, axis=1)
            nearest_point_idx = np.argmin(distances)
            selected_samples.add(int(orig_cluster_point_indices[nearest_point_idx]))
        return selected_samples

    def _get_representations(self, unlabeled_samples: tf.data.Dataset) -> np.ndarray:
        augmented_ds = unlabeled_samples.map(lambda x: self._augmentation_method(x, **self._augmentation_kwargs))

        all_sample_representation = []
        for sample_batch in augmented_ds.batch(128):
            sample_representations = self._representation_model(sample_batch)
            all_sample_representation.append(sample_representations)
        return tf.concat(all_sample_representation, axis=0).numpy()

    def _get_representation_part_of_model(self):
        for layer in self._pretrained_model.layers:
            if layer.name == "Representation":
                return layer
