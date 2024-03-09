import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.train.base import BaseTrainer


class TransformationRecognitionTrainer(BaseTrainer):
    """The trainer class for training a neural network on a pretext task to recognize time series transformations."""

    def __init__(
        self,
        model: keras.Model,
        model_name: str,
        batch_size: int = 128,
        epochs: int = 50,
        keep_best_model: bool = True,
        model_base_dir: str = "./models",
        log_base_dir: str = "./logs",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        """Initializes a trainer."""
        super().__init__(
            model,
            model_name,
            batch_size,
            epochs,
            keep_best_model,
            model_base_dir,
            log_base_dir,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _initialize_metrics(self):
        self._train_loss = keras.metrics.Mean(name="train_loss")
        self._train_accuracy = keras.metrics.CategoricalAccuracy(name="train_accuracy")
        self._validation_loss = keras.metrics.Mean(name="validation_loss")
        self._validation_accuracy = keras.metrics.CategoricalAccuracy(
            name="validation_accuracy"
        )

    def _reset_metrics(self):
        self._train_loss.reset_states()
        self._train_accuracy.reset_states()
        self._validation_loss.reset_states()
        self._validation_accuracy.reset_states()

    def _get_main_validation_metric_result(self):
        return self._validation_accuracy.result()

    def _get_loss_object(self) -> keras.losses.Loss:
        return keras.losses.CategoricalCrossentropy(from_logits=True)

    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        # return keras.optimizers.Adam()

    def _log_training_metrics(self, epoch: int):
        self._log_metrics(
            epoch,
            self._train_summary_writer,
            metrics=[self._train_loss, self._train_accuracy],
            metric_names=["loss", "accuracy"],
        )

    def _log_validation_metrics(self, epoch: int):
        self._log_metrics(
            epoch,
            self._validation_summary_writer,
            metrics=[self._validation_loss, self._validation_accuracy],
            metric_names=["loss", "accuracy"],
        )

    def _print_metrics(self):
        print(
            f"Loss: {self._train_loss.result()}, "
            f"Accuracy: {self._train_accuracy.result() * 100}, "
            f"Validation Loss: {self._validation_loss.result()}, "
            f"Validation Accuracy: {self._validation_accuracy.result() * 100}"
        )

    def _update_training_metrics(self, loss_value, true_labels, predicted_labels):
        self._train_loss(loss_value)
        self._train_accuracy(true_labels, predicted_labels)

    def _update_validation_metrics(self, loss_value, true_labels, predicted_labels):
        self._validation_loss(loss_value)
        self._validation_accuracy(true_labels, predicted_labels)
