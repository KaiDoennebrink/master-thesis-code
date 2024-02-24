from abc import ABC

import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.augmentation import random_crop
from deepal_for_ecg.train.base import BaseTrainer


class TimeSeriesClassificationTrainer(BaseTrainer, ABC):
    """
    Trainer for a time series classification model.
    It applies random crop augmentation to the training dataset.
    The validation dataset is expected in sliding windows and for each original time series the sliding windows are
    reduced by taking the maximum.
    """

    def __init__(self, model: keras.Model, model_name: str, batch_size: int = 256, epochs: int = 50,
                 keep_best_model: bool = True, model_base_dir: str = "../models", log_base_dir: str = "../logs",
                 crop_length: int = 250):
        """Initializes a trainer."""
        super().__init__(model, model_name, batch_size, epochs, keep_best_model, model_base_dir, log_base_dir)
        self.crop_length = crop_length

    def _get_training_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Returns a random crop from each sample in the dataset.
        """
        # do not use buffer_size=dataset.cardinality() here to prevent the loading of the full dataset into memory
        cropped_dataset = dataset.map(lambda x, y: (random_crop(x, self.crop_length), y))
        return super()._get_training_dataset(cropped_dataset)

    @tf.function
    def _validation_step(self, time_series_batch, label_batch):
        """Custom validation step that calculates the loss and accuracy of the model over the sliding windows."""
        sliding_window_predictions = []
        for sliding_window in time_series_batch:
            sliding_window_predictions.append(self.model(sliding_window, training=False))

        predictions = tf.reduce_max(sliding_window_predictions, axis=0)
        loss = self._loss_object(label_batch, predictions)

        # update the measures
        self._update_validation_metrics(loss, label_batch, predictions)
