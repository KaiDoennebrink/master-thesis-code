from typing import Tuple

import cv2 as cv
import numpy as np
import tensorflow as tf


def random_crop(x: tf.Tensor, timeseries_len: int = 250) -> tf.Tensor:
    """Returns a random crop from the input tensor."""
    crop_idx = np.random.randint(0, x.shape[0]-timeseries_len)
    cropped_tensor = x[crop_idx:crop_idx+timeseries_len]

    return cropped_tensor


def generate_sliding_window(data: np.ndarray, window_size: int = 250, stride: int = 125) -> Tuple[np.ndarray]:
    """
    Takes a three-dimensional array, where the first dimension is the sample number, the second dimension is time,
    and the third dimension are the channels. From this array sliding windows are generated on the time dimension.
    The sliding windows arrays are returned as a tuple.
    """
    sliding_window_array = []
    for start_idx in range(0, data.shape[1] - window_size, stride):
        sliding_window_array.append(data[:, start_idx:start_idx + window_size, :])

    return tuple(sliding_window_array)
