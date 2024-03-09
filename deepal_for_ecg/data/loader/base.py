import abc
import pickle
from pathlib import Path
from typing import Dict, Any


class BaseDataLoader(abc.ABC):
    """
    The base class to load, preprocess and save different datasets that should be used to train models.
    """

    def __init__(
        self,
        load_saved_data: bool = False,
        saved_data_base_dir: str | Path = Path("./data/saved"),
        raw_data_base_dir: str | Path = Path("./data/raw"),
    ):
        self._load_saved_data = load_saved_data
        self._saved_data_base_dir = Path(saved_data_base_dir)
        self._raw_data_base_dir = Path(raw_data_base_dir)
        self._saved_data_pickle_file_name = "loaded_and_preprocessed_data.pkl"

        # ensure that the save base dir exists
        self._saved_data_base_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Loads the dataset and preprocesses it."""
        if (
            self._load_saved_data
            and Path(
                self._saved_data_base_dir, self._saved_data_pickle_file_name
            ).exists()
        ):
            with open(
                Path(self._saved_data_base_dir, self._saved_data_pickle_file_name), "rb"
            ) as pickle_file:
                data_dict = pickle.load(pickle_file)
                self._set_data(data_dict)
        else:
            self._internal_loading_and_processing()

    def save_data(self):
        """Saves the data."""
        data_dict = self._get_data()
        with open(
            Path(self._saved_data_base_dir, self._saved_data_pickle_file_name), "wb"
        ) as data_file:
            pickle.dump(data_dict, data_file)

    @abc.abstractmethod
    def _internal_loading_and_processing(self):
        """Load and process the data."""

    @abc.abstractmethod
    def _set_data(self, data: Dict[str, Any]):
        """Sets the loaded and processed data."""

    @abc.abstractmethod
    def _get_data(self) -> Dict[str, Any]:
        """Returns the loaded and processed data."""
