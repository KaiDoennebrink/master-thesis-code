from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.random
from tensorflow import keras
from tqdm import tqdm

from deepal_for_ecg.data.tranformation_recognition import Transformation
from deepal_for_ecg.util import save_numpy


class PreTextLossInitQueryStrategy:
    """
    This query strategy can be used to select initial samples for an active learning cycle.
    It is not affected by the cold start problem as it utilizes the pretext task losses from a pre-trained model.
    The pre-trained model is trained with a pretext task in a self-supervised way on the training data that
    should be used in the active learning cycle. It is inspired by the paper "PT4AL: Using self-supervised pretext
    task for active learning" from Yi et al. (2022).

    The pre-text task used in this strategy is time series transformation recognition. In total six transformations,
    i.e., noise addition, scaling, temporal inversion, negation, permutation, and time-warping, are performed on the
    data and these needs to be distinguished from each other and the original signal.
    """

    def __init__(self, pretrained_model: keras.Model, loss_storage_dir: Path = Path("./data/loss_storage"),
                 transformation_dir: Path = Path("./data/saved/transformation_recognition/transform")):
        self._pretrained_model = pretrained_model
        self._loss_storage_dir = loss_storage_dir
        self._loss_storage_dir.mkdir(parents=True, exist_ok=True)
        self._transformation_dir = transformation_dir

        self._losses = None
        self._loss_indices = None
        self._loss_indices_batches = None
        self._current_batch = 0

    def prepare(self, training_samples: np.ndarray, num_of_al_batches: int = 10, save_losses: bool = True, load_losses: bool = False):
        """
        Prepares the query strategy by calculating the average pretext task loss to recognize each transformation for
        each training sample. Afterward the losses are sorted in descending order and stored together with the original
        training samples indices.

        Args:
            training_samples (np.ndarray): The training samples for which the pretext task loss should be calculated.
            num_of_al_batches (int): The number of batches the training samples should be split into. A batch refers to
            a pool of unlabeled data that should be sampled in an active learning iteration (default = 10).
            save_losses (bool): Whether to save the losses after calculating them (default = true).
            load_losses (bool): Whether to load saved losses (default = true).
        """
        if load_losses:
            self._losses = np.load(Path(self._loss_storage_dir, "losses.npy"))
            self._loss_indices = np.load(Path(self._loss_storage_dir, "loss_indices.npy"))
        else:
            self._losses = self._calculate_losses(training_samples)
            self._loss_indices = np.argsort(self._losses)

            if save_losses:
                save_numpy(self._losses, Path(self._loss_storage_dir, "losses.npy"))
                save_numpy(self._loss_indices, Path(self._loss_storage_dir, "loss_indices.npy"))

        # split losses into batches
        samples_per_split = training_samples.shape[0] // num_of_al_batches + 1
        split_indices = [samples_per_split * i for i in range(1, num_of_al_batches)]
        self._loss_indices_batches = np.split(self._loss_indices, split_indices)

    def select_samples(self, num_of_samples: int, previous_round_model: keras.Model = None, current_batch: int = -1) -> np.ndarray:
        """
        Select samples from the current batch and return the indices of the samples.

        Args:
            num_of_samples (int): Number of samples to select.
            previous_round_model (keras.Model): The trained model from the previous active learning iteration.
            current_batch (int): Can be used to override the current batch from which the samples should be selected.
        Returns:
            The indices of the selected samples as a numpy array.
        """
        local_batch = current_batch if current_batch != -1 else self._current_batch

        # no trained model is available before the first data was selected
        # thus, according to the PT4AL paper we sample uniformly
        if local_batch == 0:
            return self._select_uniformly_from_batch(num_of_samples, local_batch)

        # otherwise compute uncertainties in the current batch and sample from them
        # in the paper the top-1 posterior probability was calculated in the current batch with the previous round model
        # but since we hava a multi-label classification we cannot compute the top-1 posterior probability
        # for now we just return uniformly sampled data from the
        # TODO: If query strategy should be used further implement appropriate uncertainty query strategy
        return self._select_uniformly_from_batch(num_of_samples, local_batch)

    def _select_uniformly_from_batch(self, num_of_samples: int, current_batch: int) -> np.ndarray:
        """Selects samples uniformly from the current batch of samples."""
        batch = self._loss_indices_batches[current_batch]
        rng = numpy.random.default_rng()
        return rng.choice(batch, num_of_samples, replace=False)

    def _calculate_losses(self, training_samples: np.ndarray) -> np.ndarray:
        """ Calculates the losses for all training samples."""
        noisy_data, scaled_data, negation_data, temporal_inversed_data, permuted_data, time_warped_data = (
            self._get_transformation_data(training_samples))

        loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_metric = keras.metrics.Mean()
        expected_model_output = np.array([Transformation.ORIG.one_hot_label, Transformation.ADD_NOISE.one_hot_label,
                                          Transformation.SCALE.one_hot_label, Transformation.NEGATE.one_hot_label,
                                          Transformation.TEMPORAL_INVERSE.one_hot_label,
                                          Transformation.PERMUTATION.one_hot_label,
                                          Transformation.TIME_WARPING.one_hot_label])
        num_samples = training_samples.shape[0]
        stored_losses = np.zeros(num_samples)

        model_inputs_base = np.stack([
            training_samples, noisy_data, scaled_data, negation_data,
            temporal_inversed_data, permuted_data, time_warped_data
        ], axis=1)

        for i, model_input in tqdm(enumerate(model_inputs_base), desc="Calculating losses", total=num_samples):
            loss_metric.reset_states()
            logits = self._pretrained_model(model_input)
            loss_metric(loss_object(expected_model_output, logits))
            stored_losses[i] = loss_metric.result()
        return stored_losses

    def _get_transformation_data(self, training_samples: np.ndarray) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the transformation data.
        """
        if self._transformation_dir.exists():
            # try to load the transformations from disc
            noisy_data = np.load(Path(self._transformation_dir, f"{Transformation.ADD_NOISE.data_name}.npy"))
            scaled_data = np.load(Path(self._transformation_dir, f"{Transformation.SCALE.data_name}.npy"))
            negation_data = np.load(Path(self._transformation_dir, f"{Transformation.NEGATE.data_name}.npy"))
            temporal_inversed_data = np.load(
                Path(self._transformation_dir, f"{Transformation.TEMPORAL_INVERSE.data_name}.npy"))
            permuted_data = np.load(Path(self._transformation_dir, f"{Transformation.PERMUTATION.data_name}.npy"))
            time_warped_data = np.load(Path(self._transformation_dir, f"{Transformation.TIME_WARPING.data_name}.npy"))
            return noisy_data, scaled_data, negation_data, temporal_inversed_data, permuted_data, time_warped_data

        # otherwise compute the transformations and return them
        noisy_data = Transformation.ADD_NOISE.augmentation(training_samples)
        scaled_data = Transformation.SCALE.augmentation(training_samples)
        negation_data = Transformation.NEGATE.augmentation(training_samples)
        temporal_inversed_data = Transformation.TEMPORAL_INVERSE.augmentation(training_samples)
        permuted_data = Transformation.PERMUTATION.augmentation(training_samples)
        time_warped_data = Transformation.TIME_WARPING.augmentation(training_samples)
        return noisy_data, scaled_data, negation_data, temporal_inversed_data, permuted_data, time_warped_data
