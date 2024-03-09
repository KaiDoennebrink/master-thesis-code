from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from deepal_for_ecg.data.augmentation import (
    noise_addition,
    scaling,
    negation,
    temporal_inversion,
    permutation,
    time_warping,
    generate_sliding_window,
)


class Transformation(Enum):
    ORIG = ("orig", lambda x: x, [1, 0, 0, 0, 0, 0, 0])
    ADD_NOISE = ("noisy", noise_addition, [0, 1, 0, 0, 0, 0, 0])
    SCALE = ("scaled", scaling, [0, 0, 1, 0, 0, 0, 0])
    NEGATE = ("negated", negation, [0, 0, 0, 1, 0, 0, 0])
    TEMPORAL_INVERSE = ("temporal_inversed", temporal_inversion, [0, 0, 0, 0, 1, 0, 0])
    PERMUTATION = ("permuted", permutation, [0, 0, 0, 0, 0, 1, 0])
    TIME_WARPING = ("time_warped", time_warping, [0, 0, 0, 0, 0, 0, 1])

    def __init__(
        self, data_name: str, augmentation: callable, one_hot_label: list[int]
    ):
        self.data_name = data_name
        self.augmentation = augmentation
        self.one_hot_label = one_hot_label


class TransformationRecognitionDataModule:
    """
    The data module for the transformation recognition pretext task.
    It provides the training, validation and test data to train a model in a self-supervised way.
    The module either can load prepared datasets from disk or can create datasets from scratch.
    If a dataset should be created from scratch, it is recommended to just use the training data as input.
    """

    NUM_TRANSFORMATIONS = 7

    def __init__(
        self,
        saved_data_base_dir: Path = Path("./data/saved/transformation_recognition"),
        seed: int = 304,
        window_size: int = 250,
        stride: int = 125,
    ):
        self._saved_data_base_dir = saved_data_base_dir
        self._rng = np.random.default_rng(seed)
        self._orig_data = None
        self._shuffle_buffer_size = 97552

        self._validation_dataset = None
        self._test_dataset = None
        self._train_dataset = None
        self._window_size = window_size
        self._stride = stride

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        if self._validation_dataset is None:
            path_to_saved_dataset = Path(
                self._saved_data_base_dir, "datasets", "validation"
            )
            self._validation_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._validation_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        if self._test_dataset is None:
            path_to_saved_dataset = Path(self._saved_data_base_dir, "datasets", "test")
            self._test_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._test_dataset

    @property
    def train_dataset(self) -> tf.data.Dataset:
        if self._train_dataset is None:
            path_to_saved_dataset = Path(self._saved_data_base_dir, "datasets", "train")
            self._train_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._train_dataset

    def generate_and_split_data(
        self, signal_data: np.ndarray, from_checkpoint: bool = True
    ):
        """
        Splits the given signal data into training, test, and validation sets.
        These sets are stored on disk and can be used to construct the tf.data.Dataset objects.

        Args:
            signal_data (np.ndarray): The signal data that should be split.
            from_checkpoint (bool): An indicator whether to use the last checkpoint, e.g., saved splits or
                transformations, in the process.
        """
        self._orig_data = signal_data
        # calculate the buffer size for later use
        self._shuffle_buffer_size = len(self._orig_data) * self.NUM_TRANSFORMATIONS

        train_idx, valid_idx, test_idx = self._get_splits(from_checkpoint)
        (
            noisy_data,
            scaled_data,
            negation_data,
            temporal_inversed_data,
            permuted_data,
            time_warped_data,
        ) = self._generate_transformations(from_checkpoint)

        self._prepare_transformed_split_dir()
        self._split_and_save(
            noisy_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.ADD_NOISE.data_name,
        )
        self._split_and_save(
            scaled_data, train_idx, valid_idx, test_idx, Transformation.SCALE.data_name
        )
        self._split_and_save(
            negation_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.NEGATE.data_name,
        )
        self._split_and_save(
            temporal_inversed_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.TEMPORAL_INVERSE.data_name,
        )
        self._split_and_save(
            permuted_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.PERMUTATION.data_name,
        )
        self._split_and_save(
            time_warped_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.TIME_WARPING.data_name,
        )
        self._split_and_save(
            self._orig_data,
            train_idx,
            valid_idx,
            test_idx,
            Transformation.ORIG.data_name,
        )

        # reset class variable
        self._orig_data = None

    def prepare_datasets(
        self,
        shuffle_train_data: bool = True,
        sliding_windows_for_test_and_val_data: bool = False,
    ):
        """
        Prepares the test, validation and training datasets.
        Therefore, all transformations for each split are concatenated together and saved on disk again.

        Args:
            shuffle_train_data (bool): An indicator whether to shuffle the training datasets before writing it to disk.
            sliding_windows_for_test_and_val_data (bool): An indicator whether to create sliding window datasets from
                the validation and test datasets.
        """
        self._prepare_dataset(
            "test",
            with_sliding_windows=sliding_windows_for_test_and_val_data,
            with_shuffle=False,
        )
        self._prepare_dataset(
            "validation",
            with_sliding_windows=sliding_windows_for_test_and_val_data,
            with_shuffle=False,
        )
        self._prepare_dataset(
            "train", with_sliding_windows=False, with_shuffle=shuffle_train_data
        )

    def _prepare_dataset(
        self, name: str, with_sliding_windows: bool = False, with_shuffle: bool = True
    ):
        """
        Prepares a single dataset by iterating over all saved transformation data for the given split.
        It uses the CPU because otherwise the dataset is maybe too big to fit into the GPU memory.
        The resulting tf.data.Dataset is stored on disc.

        Args:
            name (str): The name of the dataset to prepare, i.e., "test", "validation", "train".
            with_sliding_windows (bool): An indicator whether to generate sliding windows from the data set.
            with_shuffle (bool): An indicator whether to shuffle the dataset before writing it to disk.
        """
        # use CPU for this work
        with tf.device("/cpu:0"):
            base_dir = Path(self._saved_data_base_dir, "transformed_split", name)
            ds = None
            for transformation in Transformation:
                data = np.load(Path(base_dir, f"{transformation.data_name}.npy"))
                label = np.array([transformation.one_hot_label] * data.shape[0])
                if with_sliding_windows:
                    data = generate_sliding_window(
                        data, self._window_size, self._stride
                    )
                if ds is not None:
                    ds = ds.concatenate(
                        tf.data.Dataset.from_tensor_slices((data, label))
                    )
                else:
                    ds = tf.data.Dataset.from_tensor_slices((data, label))

            if with_shuffle:
                # shuffles the whole dataset at once so that later smaller buffer sizes can be used
                ds = ds.shuffle(buffer_size=self._shuffle_buffer_size)

            # save dataset to disk
            dataset_dir = Path(self._saved_data_base_dir, "datasets")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            ds.save(str(Path(dataset_dir, name)))

    def _split_and_save(
        self,
        data: np.ndarray,
        train_indices: np.ndarray,
        valid_indices: np.ndarray,
        test_indices: np.ndarray,
        name: str,
    ):
        transformed_split_dir = Path(self._saved_data_base_dir, "transformed_split")
        train_data = data[train_indices]
        self._save_numpy(
            train_data, Path(transformed_split_dir, "train", f"{name}.npy")
        )
        validation_data = data[valid_indices]
        self._save_numpy(
            validation_data, Path(transformed_split_dir, "validation", f"{name}.npy")
        )
        test_data = data[test_indices]
        self._save_numpy(test_data, Path(transformed_split_dir, "test", f"{name}.npy"))

    def _prepare_transformed_split_dir(self):
        base_dir = Path(self._saved_data_base_dir, "transformed_split")
        Path(base_dir, "train").mkdir(parents=True, exist_ok=True)
        Path(base_dir, "test").mkdir(parents=True, exist_ok=True)
        Path(base_dir, "validation").mkdir(parents=True, exist_ok=True)

    def _generate_transformations(self, from_checkpoint: bool):
        transform_dir = Path(self._saved_data_base_dir, "transform")
        if from_checkpoint and transform_dir.exists():
            # try to load the transformations from disc
            noisy_data = np.load(
                Path(transform_dir, f"{Transformation.ADD_NOISE.data_name}.npy")
            )
            scaled_data = np.load(
                Path(transform_dir, f"{Transformation.SCALE.data_name}.npy")
            )
            negation_data = np.load(
                Path(transform_dir, f"{Transformation.NEGATE.data_name}.npy")
            )
            temporal_inversed_data = np.load(
                Path(transform_dir, f"{Transformation.TEMPORAL_INVERSE.data_name}.npy")
            )
            permuted_data = np.load(
                Path(transform_dir, f"{Transformation.PERMUTATION.data_name}.npy")
            )
            time_warped_data = np.load(
                Path(transform_dir, f"{Transformation.TIME_WARPING.data_name}.npy")
            )
            return (
                noisy_data,
                scaled_data,
                negation_data,
                temporal_inversed_data,
                permuted_data,
                time_warped_data,
            )

        # otherwise compute new transformations
        noisy_data = Transformation.ADD_NOISE.augmentation(self._orig_data)
        scaled_data = Transformation.SCALE.augmentation(self._orig_data)
        negation_data = Transformation.NEGATE.augmentation(self._orig_data)
        temporal_inversed_data = Transformation.TEMPORAL_INVERSE.augmentation(
            self._orig_data
        )
        permuted_data = Transformation.PERMUTATION.augmentation(self._orig_data)
        time_warped_data = Transformation.TIME_WARPING.augmentation(self._orig_data)

        # save the indices
        transform_dir.mkdir(parents=True, exist_ok=True)
        self._save_numpy(
            noisy_data, Path(transform_dir, f"{Transformation.ADD_NOISE.data_name}.npy")
        )
        self._save_numpy(
            scaled_data, Path(transform_dir, f"{Transformation.SCALE.data_name}.npy")
        )
        self._save_numpy(
            negation_data, Path(transform_dir, f"{Transformation.NEGATE.data_name}.npy")
        )
        self._save_numpy(
            temporal_inversed_data,
            Path(transform_dir, f"{Transformation.TEMPORAL_INVERSE.data_name}.npy"),
        )
        self._save_numpy(
            permuted_data,
            Path(transform_dir, f"{Transformation.PERMUTATION.data_name}.npy"),
        )
        self._save_numpy(
            time_warped_data,
            Path(transform_dir, f"{Transformation.TIME_WARPING.data_name}.npy"),
        )

        return (
            noisy_data,
            scaled_data,
            negation_data,
            temporal_inversed_data,
            permuted_data,
            time_warped_data,
        )

    def _get_splits(
        self, from_checkpoint: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        splits_dir = Path(self._saved_data_base_dir, "split")
        if from_checkpoint and splits_dir.exists():
            # try to load the splits from disc
            train_idx = np.load(Path(splits_dir, "train_idx.npy"))
            test_idx = np.load(Path(splits_dir, "test_idx.npy"))
            validation_idx = np.load(Path(splits_dir, "validation_idx.npy"))
            return train_idx, test_idx, validation_idx

        # otherwise compute new indices
        total_samples = len(self._orig_data)
        per_split_samples = total_samples // 10
        tmp_idx = np.arange(total_samples, dtype=np.int16)

        # get the test, validation and train indices
        test_idx = self._rng.choice(tmp_idx, size=per_split_samples, replace=False)
        tmp_idx = np.setdiff1d(tmp_idx, test_idx)
        validation_idx = self._rng.choice(
            tmp_idx, size=per_split_samples, replace=False
        )
        train_idx = np.setdiff1d(tmp_idx, validation_idx)

        # save the indices
        splits_dir.mkdir(parents=True, exist_ok=True)
        self._save_numpy(train_idx, Path(splits_dir, "train_idx.npy"))
        self._save_numpy(test_idx, Path(splits_dir, "test_idx.npy"))
        self._save_numpy(validation_idx, Path(splits_dir, "validation_idx.npy"))

        return train_idx, test_idx, validation_idx

    @staticmethod
    def _save_numpy(array: np.ndarray, path_to_file: Path):
        with open(path_to_file, "wb") as f:
            np.save(f, array)
