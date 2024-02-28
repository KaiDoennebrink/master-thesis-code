from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from deepal_for_ecg.data.augmentation import generate_sliding_window


class ICBEBDataModule:
    """
    The data module for the ICBEB 2018 dataset.
    """

    NUM_CLASSES = 9

    def __init__(self, saved_datasets_base_dir: Path = Path("./data/saved/icbeb/datasets"),):
        """
        Initializes the data module. Since the ICBEB 2018 challenge just offers a training set and a small validation
        set we have to create our own training, validation, and test datasets from it.

        Args:
            saved_datasets_base_dir (Path): The path to the dataset directory where the prepared datasets are stored.
        """
        self._stride = 125
        self._window_size = 250
        self._saved_datasets_base_dir = saved_datasets_base_dir
        self._saved_datasets_base_dir.mkdir(parents=True, exist_ok=True)

        self._validation_dataset = None
        self._test_dataset = None
        self._train_dataset = None

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        if self._validation_dataset is None:
            path_to_saved_dataset = Path(self._saved_datasets_base_dir, "validation")
            self._validation_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._validation_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        if self._test_dataset is None:
            path_to_saved_dataset = Path(self._saved_datasets_base_dir, "test")
            self._test_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._test_dataset

    @property
    def train_dataset(self) -> tf.data.Dataset:
        if self._train_dataset is None:
            path_to_saved_dataset = Path(self._saved_datasets_base_dir, "train")
            self._train_dataset = tf.data.Dataset.load(str(path_to_saved_dataset))
        return self._train_dataset

    def prepare_datasets(self, train_samples: np.ndarray, val_samples: np.ndarray, train_labels: np.ndarray,
                         val_labels: np.ndarray):
        """
        Splits the ICBEB dataset into training, validation, and test datasets with a ratio of 8:1:1 and saves them.

        Args:
            train_samples (np.ndarray): The training samples of the ICBEB 2018 challenge dataset. Resampled at 100Hz
                and each 10 seconds long.
            val_samples (np.ndarray): The validation samples of the ICBEB 2018 challenge dataset. Resampled at 100Hz
                and each 10 seconds long.
            train_labels (np.ndarray): The multi-label encoded training labels of the ICBEB 2018 challenge dataset.
            val_labels (np.ndarray): The multi-label encoded validation labels of the ICBEB 2018 challenge dataset.
        """

        samples = np.concatenate((train_samples, val_samples))
        labels = np.concatenate((train_labels, val_labels))
        x_train_val, x_test, y_train_val, y_test = train_test_split(samples, labels, train_size=0.9,
                                                                    stratify=labels)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=0.8888888888888,
                                                          stratify=y_train_val)

        self._save_dataset("train", x_train, y_train)
        self._save_dataset("validation", x_val, y_val, with_sliding_windows=True)
        self._save_dataset("test", x_test, y_test, with_sliding_windows=True)

    def _save_dataset(self, name: str, x: np.ndarray, y: np.ndarray, with_sliding_windows: bool = False):
        """
        Saves the given data and labels as a dataset.

        Args:
            name (str): The name of the dataset.
            x (np.ndarray): The input data of the dataset.
            y (np.ndarray): The output data of the dataset.
            with_sliding_windows (bool): Whether to create a sliding window dataset or not.
        """
        if with_sliding_windows:
            x = generate_sliding_window(x, self._window_size, self._stride)
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds.save(str(Path(self._saved_datasets_base_dir, name)))
