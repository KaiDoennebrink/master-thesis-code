import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm

from deepal_for_ecg.data.augmentation import random_crop


class PTBXLTrainer:
    """Trains a neural network on the ptb-xl dataset."""

    def __init__(
            self,
            model: keras.Model,
            model_name: str,
            num_labels: int,
            batch_size: int = 256,
            learning_rate: float = 1e-3,
            epochs: int = 50,
            weight_decay: float = 1e-2,
            keep_best_model: bool = True,
            model_base_dir: str = './models',
            log_base_dir: str = './logs'
    ):
        """Initializes a trainer."""
        self.model = model
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.keep_best_model = keep_best_model
        self.model_base_dir = model_base_dir
        self.log_base_dir = log_base_dir

    def fit(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset):
        """Trains the model on the PTB-XL dataset."""
        self._prepare()

        for epoch in range(self.epochs):

            self._reset_metrics()

            # trainings data augmentation
            cropped_ds = train_ds.map(lambda x, y: (random_crop(x), y))
            for time_series, labels in tqdm(cropped_ds.shuffle(100000).batch(self.batch_size), desc=f"Epoch: {epoch}"):
                self._training_step(time_series, labels)

            # log the training metrics
            self._log_training_metrics(epoch)

            for validation_time_series, validation_labels in validation_ds.batch(self.batch_size):
                self._validation_step(validation_time_series, validation_labels)

            self._log_validation_metrics(epoch)
            self._save_best_model()
            self._print_metrics()

    @tf.function
    def _training_step(self, time_series_batch, label_batch):
        with tf.GradientTape() as tape:
            # get the predictions
            predictions = self.model(time_series_batch, training=True)

            # measure the loss and apply it
            loss = self._loss_object(label_batch, predictions)

        tf.debugging.check_numerics(loss, "Loss is NaN or Inf")

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update the measures
        self._train_loss(loss)
        self._train_accuracy(label_batch, predictions)
        self._train_auc(label_batch, predictions)

    @tf.function
    def _validation_step(self, time_series_batch, label_batch):
        sliding_window_predictions = []
        for sliding_window in time_series_batch:
            sliding_window_predictions.append(self.model(sliding_window, training=False))

        predictions = tf.reduce_max(sliding_window_predictions, axis=0)
        loss = self._loss_object(label_batch, predictions)

        self._validation_loss(loss)
        self._validation_accuracy(label_batch, predictions)
        self._validation_auc(label_batch, predictions)

    def _prepare(self):
        """Prepares the training process."""
        Path(self.model_base_dir, self.model_name).mkdir(parents=True, exist_ok=True)
        self._initialize_log_writer()
        self._initialize_metrics()
        self._log_model()

        self._loss_object = keras.losses.BinaryCrossentropy()
        self._optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        self._best_auc = -1

    def _initialize_log_writer(self):
        """Initializes the log writer used for the current run."""
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._train_log_dir = f"logs/gradient_tape/{current_time}/train"
        self._validation_log_dir = f"logs/gradient_tape/{current_time}/validation"
        self._graph_log_dir = f"logs/func/{current_time}"
        self._train_summary_writer = tf.summary.create_file_writer(self._train_log_dir)
        self._validation_summary_writer = tf.summary.create_file_writer(self._validation_log_dir)

    def _initialize_metrics(self):
        """Initializes the training and validation metrics."""
        self._train_loss = keras.metrics.Mean(name="train_loss")
        self._train_accuracy = keras.metrics.BinaryAccuracy(name="train_accuracy")
        self._train_auc = keras.metrics.AUC(multi_label=True, name="train_auc", num_labels=self.num_labels)

        self._validation_loss = keras.metrics.Mean(name="validation_loss")
        self._validation_accuracy = keras.metrics.BinaryAccuracy(name="validation_accuracy")
        self._validation_auc = keras.metrics.AUC(multi_label=True, name="validation_auc", num_labels=self.num_labels)

    def _reset_metrics(self):
        """Resets the training and validation metric states."""
        self._train_loss.reset_states()
        self._train_accuracy.reset_states()
        self._train_auc.reset_states()
        self._validation_loss.reset_states()
        self._validation_accuracy.reset_states()
        self._validation_auc.reset_states()

    def _log_training_metrics(self, epoch: int):
        """Logs the training metric values."""
        self._log_metrics(epoch,
                          self._train_summary_writer,
                          loss_metric=self._train_loss,
                          accuracy_metric=self._train_accuracy,
                          auc_metric=self._train_auc)

    def _log_validation_metrics(self, epoch: int):
        """Logs the validation metric values."""
        self._log_metrics(epoch,
                          self._validation_summary_writer,
                          loss_metric=self._validation_loss,
                          accuracy_metric=self._validation_accuracy,
                          auc_metric=self._validation_auc)

    def _save_best_model(self):
        """Saves the model if the current validation AUC is the currently best one."""
        if self.keep_best_model and self._validation_auc.result() >= self._best_auc:
            self._best_auc = self._validation_auc.result()
            print(f"Saving new best model with AUC: {self._best_auc}")
            self.model.save(Path(self.model_base_dir, self.model_name, "best_model.keras"))

    def _print_metrics(self):
        print(
            f'Loss: {self._train_loss.result()}, '
            f'Accuracy: {self._train_accuracy.result() * 100}, '
            f'AUC: {self._train_auc.result() * 100}, '
            f'Validation Loss: {self._validation_loss.result()}, '
            f'Validation Accuracy: {self._validation_accuracy.result() * 100}, '
            f'Validation AUC: {self._validation_auc.result() * 100}'
        )

    def _log_model(self):
        """Logs the model to visualize it in tensorboard."""
        TensorBoard(log_dir=self._graph_log_dir).set_model(self.model)

    @staticmethod
    def _log_metrics(epoch: int,
                     summary_writer: tf.summary.SummaryWriter,
                     loss_metric: keras.metrics.Metric,
                     accuracy_metric: keras.metrics.Metric,
                     auc_metric: keras.metrics.Metric):
        """Logs the metrics with the specified summary writer."""
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss_metric.result(), step=epoch)
            tf.summary.scalar('accuracy', accuracy_metric.result(), step=epoch)
            tf.summary.scalar('auc', auc_metric.result(), step=epoch)
