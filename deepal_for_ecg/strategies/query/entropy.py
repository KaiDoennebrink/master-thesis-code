from typing import Set

import numpy as np
import tensorflow as tf
from tensorflow import keras
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule


class EntropySamplingStrategy:

    def __init__(self, batch_size: int = 256):
        self._batch_size = batch_size
        self._model = None

    def select_samples(
            self, num_of_samples: int, data_module: PTBXLActiveLearningDataModule, model: keras.Model
    ) -> Set[int]:
        self._model = model
        # get posterior probabilities
        probabilities = self._get_sliding_window_predictions(data_module.unlabeled_sliding_window_sample_dataset)
        probabilities = np.clip(probabilities, 1e-15, 1)
        # calculate the entropy
        entropy = - np.sum(probabilities * np.log(probabilities), axis=-1)
        # sort and select top-k
        unlabeled_indices = list(data_module.state_dict()["unlabeled_indices"])
        sorted_entropy_indices = np.flip(np.argsort(entropy))
        top_k_indices = sorted_entropy_indices[:num_of_samples]
        query_indices = np.array(unlabeled_indices)[top_k_indices]
        return set(query_indices)

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
