from typing import Set, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule


class BadgeSamplingStrategy:

    def __init__(self):
        self._model = None
        self._feature_layer_name = "Representation"
        self._batch_size = 256

    def select_samples(self, num_of_samples: int, data_module: PTBXLActiveLearningDataModule, model: keras.Model) -> Set[int]:
        self._model = model
        features, probabilities = self._get_features_and_probabilities(data_module.unlabeled_sliding_window_sample_dataset)
        # do get a label for the gradient computation we need the label cardinality to transform the probabilities into labels
        label_cardinality = data_module.label_cardinality
        sorted_probability_indexes = np.flip(np.argsort(probabilities))
        proxy_labels = np.zeros(probabilities.shape)
        proxy_labels[sorted_probability_indexes[:label_cardinality]] = 1
        # calculate the gradient embeddings
        proxy_labels_bool = proxy_labels.astype(bool)
        total_samples = features.shape[0]
        num_of_classes = data_module.NUM_CLASSES
        feature_dim = features.shape[-1]
        embeddings = np.empty((total_samples, feature_dim * num_of_classes))
        for n in range(total_samples):
            for c in range(num_of_classes):
                if proxy_labels_bool[n, c]:
                    embeddings[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (1 - probabilities[n, c])
                else:
                    embeddings[n, feature_dim * c: feature_dim * (c + 1)] = features[n] * (
                                -1 * probabilities[n, c])
        # cluster the embeddings with k-means ++
        center_indices = self.kmeans_plusplus(embeddings, num_of_samples, np.random.default_rng())

        unlabeled_indices = list(data_module.state_dict()["unlabeled_indices"])
        query_indices = np.array(unlabeled_indices)[center_indices]
        return set(query_indices)

    def kmeans_plusplus(self, embeddings, n_clusters, rng) -> List[int]:
        # Start with the highest grad norm since it is the "most uncertain"
        grad_norm = np.linalg.norm(embeddings, ord=2, axis=1)
        idx = np.argmax(grad_norm)

        indices = [idx]
        centers = [embeddings[idx]]
        dist_mat = []
        for _ in range(1, n_clusters):
            # Compute the distance of the last center to all samples
            dist = np.sqrt(np.sum((embeddings - centers[-1]) ** 2, axis=-1))
            dist_mat.append(dist)
            # Get the distance of each sample to its closest center
            min_dist = np.min(dist_mat, axis=0)
            min_dist_squared = min_dist ** 2
            if np.all(min_dist_squared == 0):
                raise ValueError('All distances to the centers are zero!')
            # sample idx with probability proportional to the squared distance
            p = min_dist_squared / np.sum(min_dist_squared)
            if np.any(p[indices] != 0):
                print('Already sampled centers have probability', p)
            idx = rng.choice(range(len(embeddings)), p=p.squeeze())
            indices.append(idx)
            centers.append(embeddings[idx])
        return indices

    def _get_features_and_probabilities(self, sliding_window_ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        # aggregate probabilities with max and features with mean
        features = []
        probabilities = []
        for batch in sliding_window_ds.batch(batch_size=self._batch_size):
            features_batch, probabilities_batch = self._aggregate_sliding_windows(batch)
            features.append(features_batch)
            probabilities.append(probabilities_batch)

        return tf.concat(features, axis=0).numpy(), tf.concat(probabilities, axis=0).numpy()

    def _aggregate_sliding_windows(self, sliding_windows) -> Tuple[tf.Tensor, tf.Tensor]:
        feature_windows = []
        probabilities_windows = []
        # pass each window through the model
        for window in sliding_windows:
            layer_in = window
            for layer in self._model.layers:
                layer_out = layer(layer_in, training=False)
                if layer.name == self._feature_layer_name:
                    feature_windows.append(layer_out)
                layer_in = layer_out
            # the last output are the probabilities
            probabilities_windows.append(layer_out)

        features = tf.reduce_mean(feature_windows, axis=0)
        probabilities = tf.reduce_max(probabilities_windows, axis=0)
        return features, probabilities

