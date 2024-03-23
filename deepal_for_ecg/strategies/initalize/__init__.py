from enum import Enum
from pathlib import Path
from typing import Set

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.strategies.initalize.pt4al import PreTextLossInitQueryStrategy
from deepal_for_ecg.strategies.initalize.representation import RepresentationClusteringInitQueryStrategy


class InitializationStrategy(Enum):
    """Enumeration of initialization strategies."""

    RANDOM = "random"
    PT4AL_ONE = "pt4al_one"
    PT4AL_TEN = "pt4al_ten"
    REPRESENTATION_CLUSTER_PRETEXT = "representation_cluster_pretext"
    REPRESENTATION_CLUSTER_TL = "representation_cluster_tl"


def apply_pt4al(
        num_samples: int,
        training_samples: np.ndarray,
        run_number: int = 1,
        pretext_model_base_name: str = "PretextInception",
        num_of_al_batches: int = 10,
        model_dir: Path = Path("./models")
) -> Set[int]:
    """
    Applies the PT4AL algorithm to select the initial samples from the training data.
    Notice: The path model_dir / (model_base_name + run_number) / best_model.keras has to exist.
    Args:
        num_samples (int): The number of samples to select.
        training_samples (np.ndarray): The raw training samples from which to choose the initial samples.
        run_number (int): The number of the model that should be used to get the representations.
        pretext_model_base_name (str): The base name of the model that is used to get the pretext losses.
        num_of_al_batches (int): The number of splits for the PT4AL algorithm.
        model_dir (Path): The directory where the model is stored.
    Returns:
        The selected indices.
    """
    # load pretext model
    model_name = f"{pretext_model_base_name}{run_number}"
    model = keras.models.load_model(
        Path(model_dir, model_name, "best_model.keras")
    )

    init_strategy = PreTextLossInitQueryStrategy(model, model_name)
    init_strategy.prepare(training_samples, num_of_al_batches=num_of_al_batches, load_losses=True)
    return init_strategy.select_samples(num_samples, current_batch=0)


def apply_representation_clustering(
        num_samples: int,
        unlabeled_dataset: tf.data.Dataset,
        model_base_name: str = "PretextInception",
        augmentation_method: callable = lambda x: x,
        run_number: int = 1,
        model_dir: Path = Path("./models")
) -> Set[int]:
    """
    Applies the representation clustering to the unlabeled dataset and returns the selected indices.
    Notice: The path model_dir / (model_base_name + run_number) / best_model.keras has to exist.

    Args:
        num_samples (int): The number of samples to select.
        unlabeled_dataset (tf.data.Dataset): The unlabeled dataset from which to choose the initial samples.
        model_base_name (str): The base name of the model that is used to get the representations.
        augmentation_method (callable): The function that augments the unlabeled dataset so that it can be feed to the model.
        run_number (int): The number of the model that should be used to get the representations.
        model_dir (Path): The directory where the model is stored.
    Returns:
        The selected indices.
    """
    # load model
    model_name = f"{model_base_name}{run_number}"
    model = keras.models.load_model(
        Path(model_dir, model_name, "best_model.keras")
    )

    init_strategy = RepresentationClusteringInitQueryStrategy(model, num_samples, augmentation_method)
    init_strategy.prepare(unlabeled_dataset)
    return init_strategy.select_samples()

