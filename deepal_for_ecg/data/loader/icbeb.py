from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm import tqdm

from deepal_for_ecg.data.loader.base import BaseDataLoader
from deepal_for_ecg.data.loader.util import (
    resample_multichannel_signal,
    apply_standardizer,
)


class ICBEBDataLoader(BaseDataLoader):
    """
    Loads and preprocesses the ICBEB 2018 dataset.

    Preprocessing steps:
    - Resamples the original signals with 100Hz sampling rate from the 500Hz original sample rate
    - Removes samples shorter than 10 seconds
    - Samples that are longer than 10 seconds will be randomly shortened to 10 seconds.
    - Standardizes the signals
    - Multi-label encoding of the labels
    """

    def __init__(
        self,
        load_saved_data: bool = False,
        saved_data_base_dir: str | Path = Path("./data/saved/icbeb"),
        raw_data_base_dir: str | Path = Path("./data/raw/icbeb_2018"),
    ):
        super().__init__(load_saved_data, saved_data_base_dir, raw_data_base_dir)

        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None

        # internal fields
        self._train_samples = None
        self._train_labels_indices = None
        self._val_samples = None
        self._val_labels_indices = None
        self._train_labels = None
        self._val_labels = None
        self._mlb = None
        self._scaler = None

    def _internal_loading_and_processing(self):
        # sample loading and processing
        self._train_samples, self._train_labels_indices = self._load_samples("training")
        self._val_samples, self._val_labels_indices = self._load_samples("validation")
        self._preprocess_signals()

        # label loading and processing
        self._train_labels = self._load_labels(self._train_labels_indices, "training")
        self._val_labels = self._load_labels(self._val_labels_indices, "validation")
        self._encode_labels()

    def _set_data(self, data: Dict[str, Any]):
        self.X_train = data["X_train"]
        self.X_val = data["X_val"]
        self.Y_train = data["Y_train"]
        self.Y_val = data["Y_val"]
        self._mlb = data["mlb"]
        self._scaler = data["scaler"]

    def _get_data(self) -> Dict[str, Any]:
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "Y_train": self.Y_train,
            "Y_val": self.Y_val,
            "mlb": self._mlb,
            "scaler": self._scaler,
        }

    def _load_samples(
        self, sample_kind: str = "training", orig_frequency: int = 500
    ) -> Tuple[List[np.ndarray], Set[str]]:
        """
        Loads the given samples from the ICBEB dataset and resamples them at 100Hz.

        Args:
            sample_kind (str): The kind of samples to load. Possible values are "training" and "validation".
        """
        signal_dir = Path(self._raw_data_base_dir, sample_kind)
        df = self._get_df(sample_kind)
        recordings = []
        label_idx = []
        for recording in tqdm(df["Recording"], desc=f"Loading {sample_kind} samples"):
            data = loadmat(str(Path(signal_dir, f"{recording}.mat")))
            ecg_signal = data["ECG"][0]["data"][0].T
            if ecg_signal.shape[0] >= 10 * orig_frequency:
                recordings.append(
                    resample_multichannel_signal(
                        ecg_signal, orig_frequency=orig_frequency, target_frequency=100
                    )
                )
                label_idx.append(recording)
        return recordings, set(label_idx)

    def _load_labels(
        self, indices: Set[str], sample_kind: str = "training"
    ) -> List[List[int]]:
        """
        Loads the given labels from the ICBEB dataset.

        Args:
            indices (List[str]): A list with indices that should be loaded.
            sample_kind (str): The kind of samples to load. Possible values are "training" and "validation".
        """
        df = self._get_df(sample_kind)
        labels = []
        for _, row in df.iterrows():
            if row["Recording"] in indices:
                sample_label = []
                if not pd.isnull(row["First_label"]):
                    sample_label.append(int(row["First_label"]))
                if not pd.isnull(row["Second_label"]):
                    sample_label.append(int(row["Second_label"]))
                if not pd.isnull(row["Third_label"]):
                    sample_label.append(int(row["Third_label"]))
                labels.append(sample_label)
        return labels

    def _get_df(self, sample_kind: str = "training") -> pd.DataFrame:
        signal_dir = Path(self._raw_data_base_dir, sample_kind)
        signal_reference_file = Path(signal_dir, "REFERENCE.csv")
        return pd.read_csv(signal_reference_file)

    def _preprocess_signals(self):
        # Standardize data such that mean 0 and variance 1
        self._train_samples = self._shorten_signals(self._train_samples)
        self._val_samples = self._shorten_signals(self._val_samples)
        self._standardize()

    def _encode_labels(self):
        """
        Encodes the labels with a multi-label encoding.
        """
        self._mlb = MultiLabelBinarizer()
        self._mlb.fit(self._train_labels)

        # apply it to both label sources
        self.Y_train = self._mlb.transform(self._train_labels)
        self.Y_val = self._mlb.transform(self._val_labels)

    def _shorten_signals(self, signals: List[np.ndarray]) -> np.ndarray:
        """Shortens the data such that each sample is 10 seconds or 1000 data point long."""
        rng = np.random.default_rng()
        shorten_signals = []
        for s in signals:
            signal_length = s.shape[0]
            if signal_length > 1000:
                start_idx = rng.integers(signal_length - 1000)
                shorten_signals.append(s[start_idx : start_idx + 1000])
            else:
                shorten_signals.append(s)
        return np.array(shorten_signals)

    def _standardize(self):
        """Standardize the data such that mean is zero and variance one."""
        self._scaler = StandardScaler()
        self._scaler.fit(
            np.vstack(self._train_samples).flatten()[:, np.newaxis].astype(np.float32)
        )

        self.X_train = apply_standardizer(self._train_samples, self._scaler)
        self.X_val = apply_standardizer(self._val_samples, self._scaler)
