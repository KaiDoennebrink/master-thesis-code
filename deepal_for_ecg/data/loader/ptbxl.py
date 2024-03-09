import ast
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm import tqdm
import wfdb

from deepal_for_ecg.data.loader.base import BaseDataLoader
from deepal_for_ecg.data.loader.util import apply_standardizer


logger = logging.getLogger(__name__)


class PTBXLDataLoader(BaseDataLoader):
    """Loads and preprocesses the PTB-XL and PTB-XL+ data to train, validate and test a neural network."""

    def __init__(
        self,
        load_saved_data: bool = False,
        saved_data_base_dir: str | Path = Path("./data/saved/ptbxl"),
        raw_data_base_dir: str | Path = Path("./data/raw"),
    ):
        super().__init__(load_saved_data, saved_data_base_dir, raw_data_base_dir)
        self._raw_ptb_xl_data_dir = Path(self._raw_data_base_dir, "ptb_xl_1.0.3")
        self._raw_ptb_xl_plus_data_dir = Path(
            self._raw_data_base_dir, "ptb_xl_plus_1.0.1"
        )

        self.X_train = None
        self.Y_train_ptb_xl = None
        self.Y_train_12sl = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None

        # internal fields
        self._relevant_snomed_ct_codes = None
        self._samples = None
        self._folds = None
        self._ptb_xl_snomed_labels = None
        self._12sl_snomed_labels = None
        self._ptb_xl_snomed_labels_encoded = None
        self._12sl_snomed_labels_encoded = None
        self._mlb = None

    def _internal_loading_and_processing(self):
        """Load and process the data."""
        self._init_relevant_snomed_ct_codes()
        self._load_samples()
        self._load_labels()
        self._split_data()
        self._preprocess_signals()
        self._ensure_label_availability()

    def _set_data(self, data: Dict[str, Any]):
        """Sets the loaded and processed data."""
        self.X_train = data["X_train"]
        self.Y_train_ptb_xl = data["Y_train_ptb_xl"]
        self.Y_train_12sl = data["Y_train_12sl"]
        self.X_valid = data["X_valid"]
        self.Y_valid = data["Y_valid"]
        self.X_test = data["X_test"]
        self.Y_test = data["Y_test"]
        self._mlb = data["mlb"]

    def _get_data(self) -> Dict[str, Any]:
        """Returns the loaded and processed data."""
        return {
            "X_train": self.X_train,
            "Y_train_ptb_xl": self.Y_train_ptb_xl,
            "Y_train_12sl": self.Y_train_12sl,
            "X_valid": self.X_valid,
            "Y_valid": self.Y_valid,
            "X_test": self.X_test,
            "Y_test": self.Y_test,
            "mlb": self._mlb,
        }

    def _init_relevant_snomed_ct_codes(self):
        """
        Initializes the relevant SNOMED CT codes based on the classification of the SNOMED CT codes in the PTB-XL+ dataset.
        A SNOMED CT code is relevant if it is used in both mappings, i.e., the original SCP codes of the PTB-XL data
        set and the 12SL statements of the PTB-XL+ dataset, and it is classified as informative.
        """
        snomed_df = pd.read_csv(
            Path(self._raw_ptb_xl_plus_data_dir, "labels", "snomed_description.csv")
        )
        self._relevant_snomed_ct_codes = set(
            snomed_df[snomed_df["in_both"] & snomed_df["informative"]].snomed_id
        )

    def _load_samples(self):
        """
        Loads the 100Hz samples and the stratified folds from the PTB-XL dataset that are used in the experiments.
        """
        tmp_ptb_xl_df = pd.read_csv(
            Path(self._raw_ptb_xl_data_dir, "ptbxl_database.csv"), index_col="ecg_id"
        )
        ptb_xl_df = tmp_ptb_xl_df[["strat_fold", "filename_lr"]]

        file_location_series = ptb_xl_df["filename_lr"]
        data = [
            wfdb.rdsamp(str(self._raw_ptb_xl_data_dir.joinpath(str(f))))
            for f in tqdm(file_location_series, desc="Load samples")
        ]
        self._samples = np.array([signal for signal, _ in data])
        self._folds = ptb_xl_df["strat_fold"]

    def _load_labels(self):
        """
        Loads the labels from the original PTB-XL dataset and the 12SL statements mapped to the SNOMED CT codes.
        """
        # load the PTB-XL SNOMED CT labels
        _ptb_xl_snomed_df = pd.read_csv(
            Path(self._raw_ptb_xl_plus_data_dir, "labels/ptbxl_statements.csv"),
            index_col="ecg_id",
        )
        _ptb_xl_snomed_df["scp_codes_ext_snomed"] = _ptb_xl_snomed_df[
            "scp_codes_ext_snomed"
        ].apply(lambda x: ast.literal_eval(x))
        _ptb_xl_snomed_df["relevant_snomed"] = _ptb_xl_snomed_df[
            "scp_codes_ext_snomed"
        ].apply(self._filter_snomed_codes)
        self._ptb_xl_snomed_labels = _ptb_xl_snomed_df["relevant_snomed"]

        # load the 12SL statement SNOMED CT labels
        _12sl_snomed_df = pd.read_csv(
            Path(self._raw_ptb_xl_plus_data_dir, "labels/12sl_statements.csv")
        )
        _12sl_snomed_df["statements_ext_snomed"] = _12sl_snomed_df[
            "statements_ext_snomed"
        ].apply(lambda x: ast.literal_eval(x))
        _12sl_snomed_df["relevant_snomed"] = _12sl_snomed_df[
            "statements_ext_snomed"
        ].apply(self._filter_snomed_codes)
        self._12sl_snomed_labels = _12sl_snomed_df["relevant_snomed"]

    def _split_data(self):
        """
        Splits the data into training, validation and test data using the folds from the PTB-XL dataset.
        For the validation and test data, the PTB-XL SNOMED labels are just provided as labels.
        For the training data, the PTB-XL SNOMED labels and the 12SL SNOMED labels are provided.
        During the process the SNOMED labels will be encoded with a multi label binarizer.
        """
        # initialize multi label binarizer
        self._mlb = MultiLabelBinarizer()
        self._mlb.fit(self._ptb_xl_snomed_labels.values)

        # apply it to both label sources
        self._ptb_xl_snomed_labels_encoded = self._mlb.transform(
            self._ptb_xl_snomed_labels.values
        )
        self._12sl_snomed_labels_encoded = self._mlb.transform(
            self._12sl_snomed_labels.values
        )

        # split the data
        self.X_test = self._samples[self._folds == 10]
        self.Y_test = self._ptb_xl_snomed_labels_encoded[self._folds == 10]
        self.X_valid = self._samples[self._folds == 9]
        self.Y_valid = self._ptb_xl_snomed_labels_encoded[self._folds == 9]
        self.X_train = self._samples[self._folds <= 8]
        self.Y_train_ptb_xl = self._ptb_xl_snomed_labels_encoded[self._folds <= 8]
        self.Y_train_12sl = self._12sl_snomed_labels_encoded[self._folds <= 8]

    def _preprocess_signals(self):
        # Standardize data such that mean 0 and variance 1
        ss = StandardScaler()
        ss.fit(np.vstack(self.X_train).flatten()[:, np.newaxis].astype(float))

        self.X_train = apply_standardizer(self.X_train, ss)
        self.X_valid = apply_standardizer(self.X_valid, ss)
        self.X_test = apply_standardizer(self.X_test, ss)

    def _filter_snomed_codes(self, codes_with_probs: list) -> set:
        """Filters the given SNOMED CT codes with probabilities and just keeps the relevant codes."""
        all_codes = set([code_with_prob[0] for code_with_prob in codes_with_probs])
        return self._relevant_snomed_ct_codes.intersection(all_codes)

    def _ensure_label_availability(self):
        """Ensures that each label is present in each split."""
        logger.debug(f"Check for missing labels in each split.")
        missing_label_indices = self._check_for_missing_labels(self.Y_test)
        missing_label_indices.extend(self._check_for_missing_labels(self.Y_valid))
        missing_label_indices.extend(self._check_for_missing_labels(self.Y_train_12sl))
        missing_label_indices.extend(
            self._check_for_missing_labels(self.Y_train_ptb_xl)
        )
        logger.debug(
            f"Found {len(missing_label_indices)} missing labels: {missing_label_indices}"
        )

        # remove missing label indices from the
        if len(missing_label_indices) > 0:
            logger.debug(f"Delete missing label columns from all splits.")
            self.Y_train_ptb_xl = self._delete_missing_labels(
                self.Y_train_ptb_xl, missing_label_indices
            )
            self.Y_train_12sl = self._delete_missing_labels(
                self.Y_train_12sl, missing_label_indices
            )
            self.Y_valid = self._delete_missing_labels(
                self.Y_valid, missing_label_indices
            )
            self.Y_test = self._delete_missing_labels(
                self.Y_test, missing_label_indices
            )

        assert (
            self.Y_train_ptb_xl.shape[1]
            == self.Y_train_12sl.shape[1]
            == self.Y_test.shape[1]
            == self.Y_valid.shape[1]
        ), (
            f"Label splits have different number of labels "
            f"{self.Y_train_ptb_xl.shape[1] = }, {self.Y_train_12sl.shape[1] = }, "
            f"{self.Y_test.shape[1] = }, {self.Y_valid.shape[1] =}"
        )
        logger.debug(f"{self.Y_train_ptb_xl.shape[1]} labels are available.")

    def _check_for_missing_labels(self, label_split: np.ndarray) -> list[int]:
        """Checks whether labels are missing in the split and returns the indices of the missing labels."""
        return np.where(label_split.sum(axis=0) == 0)[0].tolist()

    def _delete_missing_labels(
        self, label_split: np.ndarray, column_indices: list[int]
    ) -> np.ndarray:
        """Deletes the given columns from the split label array."""
        return np.delete(label_split, column_indices, axis=1)
