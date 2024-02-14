import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tqdm import tqdm
import wfdb


class PTBXLDataProvider:
    """Provides the PTB-XL data to train, validate and test a neural network."""

    def __init__(self,
                 load_saved_data: bool = False,  # TODO: Change to true
                 saved_data_base_dir: str | Path = Path("./data/saved"),
                 raw_ptb_xl_data_dir: str | Path = Path("./data/raw/ptb_xl_1.0.3"),
                 raw_ptb_xl_plus_data_dir: str | Path = Path("./data/raw/ptb_xl_plus_1.0.1")):
        self._load_saved_data = load_saved_data
        self._saved_data_base_dir = saved_data_base_dir
        self._raw_ptb_xl_data_dir = raw_ptb_xl_data_dir
        self._raw_ptb_xl_plus_data_dir = raw_ptb_xl_plus_data_dir

        self._init_relevant_snomed_ct_codes()
        self._load_samples()
        self._load_labels()

    def _init_relevant_snomed_ct_codes(self):
        """
        Initializes the relevant SNOMED CT codes based on the classification of the SNOMED CT codes in the PTB-XL+ data set.
        A SNOMED CT code is relevant if it is used in both mappings, i.e., the original SCP codes of the PTB-XL data
        set and the 12SL statements of the PTB-XL+ data set, and it is classified as informative.
        """
        # TODO: Check whether the data exists somewhere
        if self._load_saved_data:
            # TODO: Load saved data
            pass
        else:
            snomed_df = pd.read_csv(Path(self._raw_ptb_xl_plus_data_dir, "labels", "snomed_description.csv"))
            self._relevant_snomed_ct_codes = set(snomed_df[snomed_df["in_both"] & snomed_df["informative"]].snomed_id)
            # TODO: Save relevant SNOMED CT codes

    def _load_samples(self):
        """
        Loads the 100Hz samples and the stratified folds from the PTB-XL data set that are used in the experiments.
        """
        # TODO: Check whether the data exists somewhere
        if self._load_saved_data:
            # TODO: Load saved data
            pass
        else:
            tmp_ptb_xl_df = pd.read_csv(Path(self._raw_ptb_xl_data_dir, "ptbxl_database.csv"), index_col="ecg_id")
            ptb_xl_df = tmp_ptb_xl_df[["strat_fold", "filename_lr"]]

            file_location_series = ptb_xl_df["filename_lr"]
            data = [wfdb.rdsamp(str(self._raw_ptb_xl_data_dir.joinpath(str(f)))) for f in tqdm(file_location_series)]
            self._samples = np.array([signal for signal, _ in data])
            self._folds = ptb_xl_df["strat_fold"]
            # TODO: Save samples and folds

    def _load_labels(self):
        """
        Loads the labels from the original PTB-XL data set and the 12SL statements mapped to the SNOMED CT codes.
        """
        # TODO: Check whether the data exists somewhere
        if self._load_saved_data:
            # TODO: Load saved data
            pass
        else:
            # load the PTB-XL SNOMED CT labels
            _ptb_xl_snomed_df = pd.read_csv(Path(self._raw_ptb_xl_plus_data_dir, "labels/ptbxl_statements.csv"),
                                           index_col="ecg_id")
            _ptb_xl_snomed_df["scp_codes_ext_snomed"] = _ptb_xl_snomed_df["scp_codes_ext_snomed"].apply(
                lambda x: ast.literal_eval(x))
            _ptb_xl_snomed_df["relevant_snomed"] = _ptb_xl_snomed_df["scp_codes_ext_snomed"].apply(
                self._filter_snomed_codes)
            self._ptb_xl_snomed_labels = _ptb_xl_snomed_df["relevant_snomed"]

            # load the 12SL statement SNOMED CT labels
            _12sl_snomed_df = pd.read_csv(Path(self._raw_ptb_xl_plus_data_dir, "labels/12sl_statements.csv"))
            _12sl_snomed_df["statements_ext_snomed"] = _12sl_snomed_df["statements_ext_snomed"].apply(
                lambda x: ast.literal_eval(x))
            _12sl_snomed_df["relevant_snomed"] = _12sl_snomed_df["statements_ext_snomed"].apply(
                self._filter_snomed_codes)
            self._12sl_snomed_labels = _12sl_snomed_df["relevant_snomed"]
            # TODO: Save labels

    def _split_data(self):
        """
        Splits the data into training, validation and test data using the folds from the PTB-XL data set.
        For the validation and test data, the PTB-XL SNOMED labels are just provided as labels.
        For the training data, the PTB-XL SNOMED labels and the 12SL SNOMED labels are provided.
        During the process the SNOMED labels will be encoded with a multi label binarizer.
        """
        # initialize multi label binarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(self._ptb_xl_snomed_labels.values)

        # apply it to both label sources
        self._ptb_xl_snomed_labels_encoded = mlb.transform(self._ptb_xl_snomed_labels.values)
        self._12sl_snomed_labels_encoded = mlb.transform(self._12sl_snomed_labels.values)

        # split the data
        self.X_test = self._samples[self._folds == 10]
        self.Y_test = self._ptb_xl_snomed_labels_encoded[self._folds == 10]
        self.X_val = self._samples[self._folds == 9]
        self.Y_val = self._ptb_xl_snomed_labels_encoded[self._folds == 9]
        self.X_train = self._samples[self._folds <= 8]
        self.Y_train_ptb_xl = self._ptb_xl_snomed_labels_encoded[self._folds <= 8]
        self.Y_train_12sl = self._12sl_snomed_labels_encoded[self._folds <= 8]

    def _preprocess_signals(self):
        # Standardize data such that mean 0 and variance 1
        ss = StandardScaler()
        ss.fit(np.vstack(self.X_train).flatten()[:, np.newaxis].astype(float))

        self.X_train = self.apply_standardizer(self.X_train, ss)
        self.X_val = self.apply_standardizer(self.X_val, ss)
        self.X_test = self.apply_standardizer(self.X_test, ss)

    def _filter_snomed_codes(self, codes_with_probs: list) -> set:
        """Filters the given SNOMED CT codes with probabilities and just keeps the relevant codes."""
        all_codes = set([code_with_prob[0] for code_with_prob in codes_with_probs])
        return self._relevant_snomed_ct_codes.intersection(all_codes)

    @staticmethod
    def apply_standardizer(data, standard_scaler: StandardScaler):
        x_tmp = []
        for x in data:
            x_shape = x.shape
            x_tmp.append(standard_scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
        return np.array(x_tmp)
