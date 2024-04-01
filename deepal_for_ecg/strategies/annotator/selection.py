import numpy as np
import tensorflow as tf
from tensorflow import keras


class AnnotatorSelectionStrategy:

    def __init__(self, selection_threshold: float = 0.8):
        self._batch_size = 300
        self._model = None
        self._selection_threshold = selection_threshold

    def select_annotator(self, annotator_model: keras.Model, selected_samples_ds: tf.data.Dataset) -> np.ndarray:
        """Returns a boolean numpy array where true means use label from WSA and false means use label from HA."""
        self._model = annotator_model
        predictions = self._get_predictions(selected_samples_ds)
        return (predictions >= self._selection_threshold).reshape((-1,))

    def _get_predictions(self, sliding_window_ds: tf.data.Dataset) -> np.ndarray:
        pred = []
        for batch in sliding_window_ds.batch(self._batch_size):
            pred.append(self._model(batch).numpy())
        return np.concatenate(pred, axis=0)

