import abc
import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tqdm import tqdm


class BaseTrainer(abc.ABC):
    """Base class for training a neural network."""

    def __init__(
        self,
        model: keras.Model,
        model_name: str,
        batch_size: int = 256,
        epochs: int = 50,
        keep_best_model: bool = True,
        model_base_dir: str = "./models",
        log_base_dir: str = "./logs",
    ):
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.keep_best_model = keep_best_model
        self._best_main_metric_result = -1.0
        self._best_epoch = -1
        self.model_base_dir = model_base_dir
        self.log_base_dir = log_base_dir
        self._loss_object = None
        self._optimizer = None
        self._shuffle_buffer_size = 1000
        self._experiment_name = None
        self._early_stopping_patience = 25

    def fit(
        self,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        verbose: bool = False,
    ):
        """Trains the model on the given dataset."""
        self._prepare()

        # prepare the validation dataset
        prepared_validation_ds = self._get_validation_dataset(validation_ds)

        for epoch in range(self.epochs):

            self._reset_metrics()

            # trainings data augmentation
            augmented_train_ds = self._get_training_dataset(train_ds)
            for samples, labels in tqdm(
                augmented_train_ds.batch(self.batch_size),
                desc=f"Training - Epoch: {epoch}",
                disable=verbose,
            ):
                self._training_step(samples, labels)
            self._log_training_metrics(epoch)

            for validation_samples, validation_labels in tqdm(
                prepared_validation_ds.batch(self.batch_size),
                desc=f"Validation - Epoch: {epoch}",
                disable=verbose,
            ):
                self._validation_step(validation_samples, validation_labels)
            self._log_validation_metrics(epoch)

            new_best = self._save_best_model()

            if not verbose:
                self._print_metrics()

            if new_best:
                self._best_epoch = epoch

            # stop the training if there was no improvement for a too long time
            if (epoch - self._best_epoch) == self._early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch} - last improvement was in epoch {self._best_epoch} with {self._best_main_metric_result}"
                )
                break

    def get_model(self, best: bool = False) -> tf.keras.Model:
        """Returns either the current or the best model."""
        return (
            keras.models.load_model(
                Path(self.model_base_dir, self.model_name, "best_model.keras")
            )
            if best
            else self.model
        )

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, value):
        self._experiment_name = value

    @tf.function
    def _training_step(self, data_batch, label_batch):
        """
        Standard training step for a classification model.
        Can be overridden by subclasses to have a custom training step.
        """
        with tf.GradientTape() as tape:
            # get the predictions
            predictions = self.model(data_batch, training=True)

            # measure the loss and apply it
            loss = self._loss_object(label_batch, predictions)

        tf.debugging.check_numerics(loss, "Loss is NaN or Inf")

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # update the measures
        self._update_training_metrics(loss, label_batch, predictions)

    @tf.function
    def _validation_step(self, data_batch, label_batch):
        """
        Standard validation step for a classification model.
        Can be overridden by subclasses to have a custom validation step.
        """

        predictions = self.model(data_batch, training=False)
        loss = self._loss_object(label_batch, predictions)

        # update the measures
        self._update_validation_metrics(loss, label_batch, predictions)

    def _save_best_model(self) -> bool:
        """Saves the model if the current validation AUC is the currently best one."""
        if (
            self.keep_best_model
            and self._get_main_validation_metric_result()
            >= self._best_main_metric_result
        ):
            self._best_main_metric_result = self._get_main_validation_metric_result()
            print(
                f"Saving new best model with main validation metric: {self._best_main_metric_result}"
            )
            self.model.save(
                Path(self.model_base_dir, self.model_name, "best_model.keras")
            )
            return True
        return False

    def _prepare(self):
        """Prepares the training process."""
        Path(self.model_base_dir, self.model_name).mkdir(parents=True, exist_ok=True)
        self._initialize_log_writer()
        self._log_model()
        self._initialize_metrics()

        self._loss_object = self._get_loss_object()
        self._optimizer = self._get_optimizer()

    def _initialize_log_writer(self):
        """Initializes the log writer used for the current run."""
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = (
            f"{self._experiment_name}_{current_time}"
            if self._experiment_name
            else current_time
        )
        self._train_log_dir = str(
            Path(self.log_base_dir, "gradient_tape", dir_name, "train")
        )
        self._validation_log_dir = str(
            Path(self.log_base_dir, "gradient_tape", dir_name, "validation")
        )
        self._graph_log_dir = str(Path(self.log_base_dir, "func", dir_name))
        self._train_summary_writer = tf.summary.create_file_writer(self._train_log_dir)
        self._validation_summary_writer = tf.summary.create_file_writer(
            self._validation_log_dir
        )

    def _log_model(self):
        """Logs the model to visualize it in tensorboard."""
        TensorBoard(log_dir=self._graph_log_dir).set_model(self.model)

    def _get_training_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Returns the shuffled but un-batched training dataset that should be used in the current iteration.
        Can be overridden by subclasses to add augmentations to the training dataset.
        """
        # do not use buffer_size=dataset.cardinality() here to prevent the loading of the full dataset into memory
        return dataset.shuffle(buffer_size=self._shuffle_buffer_size)

    def _get_validation_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepares the validation dataset for the training.
        Can be overridden by subclasses to add custom preparations to the dataset.
        """
        return dataset

    @staticmethod
    def _log_metrics(
        epoch: int,
        summary_writer: tf.summary.SummaryWriter,
        metrics: list[keras.metrics.Metric],
        metric_names: list[str],
    ):
        """Logs the metrics with the specified summary writer."""
        with summary_writer.as_default():
            for metric, name in zip(metrics, metric_names):
                tf.summary.scalar(name, metric.result(), step=epoch)

    @abc.abstractmethod
    def _initialize_metrics(self):
        """Initializes the training and validation metrics."""

    @abc.abstractmethod
    def _reset_metrics(self):
        """Resets the training and validation metric states."""

    @abc.abstractmethod
    def _get_main_validation_metric_result(self):
        """Returns the result of the main validation metric."""

    @abc.abstractmethod
    def _get_loss_object(self) -> keras.losses.Loss:
        """Returns the loss object that should be used to compute the losses."""

    @abc.abstractmethod
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Returns the optimizer that should be used to train the network."""

    @abc.abstractmethod
    def _log_training_metrics(self, epoch: int):
        """Logs the training metric values."""

    @abc.abstractmethod
    def _log_validation_metrics(self, epoch: int):
        """Logs the validation metric values."""

    @abc.abstractmethod
    def _print_metrics(self):
        """Prints the current result of the metrics."""

    @abc.abstractmethod
    def _update_training_metrics(self, loss_value, true_labels, predicted_labels):
        """Updates the training metric values."""

    @abc.abstractmethod
    def _update_validation_metrics(self, loss_value, true_labels, predicted_labels):
        """Updates the validation metric values."""
