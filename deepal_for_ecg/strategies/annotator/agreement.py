import numpy as np


def measure_absolute_agreement(true_labels: np.ndarray, weak_labels: np.ndarray) -> np.ndarray:
    """
    Measures the absolute agreement between the two label collections. Therefore, it checks how many labels are the same
    for each sample.

    Args:
        true_labels (np.ndarray): The labels of the samples that are considered to be the ground-truth labels. The shape
            is (number of samples, number of classes). Each element should be zero or one.
        weak_labels (np.ndarray): The labels of the samples that are considered to be the weak labels. The shape
            is (number of samples, number of classes). Each element should be zero or one.
    Returns:
        A numpy array with the absolute agreement between the true labels and the weak labels. The shape is (number of
        samples, ).
    """
    same_labels = true_labels == weak_labels
    return np.sum(same_labels, axis=1)


def measure_relative_agreement_to_positive_true_labels(true_labels: np.ndarray, weak_labels: np.ndarray) -> np.ndarray:
    """
    Measures the relative agreement between a true label collection and a weak label collection.
    For each sample the relative agreement between the two multi-label distributions is measured as follows:

    - Count the number of the agreement between the positive classes of the true label and the positive classes of the
      weak label
    - Count the total number of positive classes of the true label
    - Divide the first count by the second count

    Args:
        true_labels (np.ndarray): The labels of the samples that are considered to be the ground-truth labels. The shape
            is (number of samples, number of classes). Each element should be zero or one.
        weak_labels (np.ndarray): The labels of the samples that are considered to be the weak labels. The shape
            is (number of samples, number of classes). Each element should be zero or one.
    Returns:
        A numpy array with the relative agreement between the true labels and the weak labels. The shape is (number of
        samples, ).
    """
    true_labels_bool = true_labels.astype(bool)
    weak_labels_bool = weak_labels.astype(bool)
    agreement = true_labels_bool & weak_labels_bool
    return np.sum(agreement, axis=1) / np.sum(true_labels, axis=1)
