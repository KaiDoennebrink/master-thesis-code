from typing import List

import tensorflow as tf
from tensorflow import keras


@tf.function
def test_step_with_sliding_windows(
        sliding_window_batch: tf.Tensor,
        label_batch: tf.Tensor,
        model: keras.Model,
        loss_object: keras.losses.Loss,
        loss_based_metrics: List[keras.metrics.Metric],
        prediction_based_metrics: List[keras.metrics.Metric]
):
    """
    Tests a model based on a sliding window dataset.

    :param sliding_window_batch: A batch with sliding windows for multiple samples.
    :param label_batch: The associated labels for the batch.
    :param model: The model that should be tested.
    :param loss_object: The loss object to calculate the loss.
    :param loss_based_metrics: A list of metrics that are based on loss values.
    :param prediction_based_metrics: A list of metrics that are based on prediction and the real label values.
    """
    sliding_window_predictions = []
    for sliding_window in sliding_window_batch:
        sliding_window_predictions.append(model(sliding_window, training=False))

    predictions = tf.reduce_max(sliding_window_predictions, axis=0)
    t_loss = loss_object(label_batch, predictions)

    for metric in loss_based_metrics:
        metric(t_loss)
    for metric in prediction_based_metrics:
        metric(label_batch, predictions)