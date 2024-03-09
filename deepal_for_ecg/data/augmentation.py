from typing import Tuple

import cv2 as cv
import numpy as np
import tensorflow as tf


def random_crop(x: tf.Tensor, timeseries_len: int = 250) -> tf.Tensor:
    """Returns a random crop from the input tensor."""
    crop_idx = np.random.randint(0, x.shape[0] - timeseries_len)
    cropped_tensor = x[crop_idx : crop_idx + timeseries_len]

    return cropped_tensor


def generate_sliding_window(
    data: np.ndarray, window_size: int = 250, stride: int = 125
) -> Tuple[np.ndarray]:
    """
    Takes a three-dimensional array, where the first dimension is the sample number, the second dimension is time,
    and the third dimension are the channels. From this array sliding windows are generated on the time dimension.
    The sliding windows arrays are returned as a tuple.
    """
    sliding_window_array = []
    for start_idx in range(0, data.shape[1] - window_size, stride):
        sliding_window_array.append(data[:, start_idx : start_idx + window_size, :])

    return tuple(sliding_window_array)


def noise_addition(
    data: np.ndarray,
    signal_to_noise_ratio: int = 15,
    output_type: np.dtype = np.float32,
) -> np.ndarray:
    """
    Adds noise to the signal as described by Sarkar and Etemad (2022).
    https://code.engineering.queensu.ca/pritam/SSL-ECG/-/blob/master/implementation/signal_transformation_task.py?ref_type=heads
    """
    # calculate signal power per sample and channel and convert it to dB
    data_watts = data**2
    avg_watts_per_channel = np.mean(data_watts, axis=1)
    avg_db_per_channel = 10 * np.log10(avg_watts_per_channel)
    # calculate the noise and convert to watts
    noise_avg_db = avg_db_per_channel - signal_to_noise_ratio
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # calculate the variance per sample and channel
    variance = np.sqrt(noise_avg_watts)
    # create the noise
    rng = np.random.default_rng()
    noise = rng.normal(0, 1, data.shape)
    noise = noise * np.sqrt(variance)[:, np.newaxis, :]

    # add noise to signal
    output = data + noise
    return output.astype(output_type)


def scaling(
    data: np.ndarray, scaling_factor: float = 0.9, output_type: np.dtype = np.float32
) -> np.ndarray:
    """Scales the original data with the given scaling factor."""
    output = data * scaling_factor
    return output.astype(output_type)


def negation(data: np.ndarray, output_type: np.dtype = np.float32) -> np.ndarray:
    """Negates the original data."""
    output = data * (-1)
    return output.astype(output_type)


def temporal_inversion(
    data: np.ndarray, output_type: np.dtype = np.float32
) -> np.ndarray:
    """Inverts the original data on its time axis."""
    output = np.flip(data, axis=1)
    return output.astype(output_type)


def permutation(
    data: np.ndarray, num_segments: int = 20, output_type: np.dtype = np.float32
) -> np.ndarray:
    """
    Transforms the original data by dividing it into segments and randomly perturbing the temporal location
    of each segment.
    """
    orig_shape = data.shape
    output_data = (
        data.copy()
        .reshape([orig_shape[0], num_segments, -1, orig_shape[-1]])
        .astype(output_type)
    )
    rng = np.random.default_rng()
    rng.shuffle(output_data, axis=1)
    return output_data.reshape(orig_shape)


def time_warping(
    data: np.ndarray,
    num_segments: int = 9,
    stretch_factor: float = 1.05,
    output_type: np.dtype = np.float32,
) -> np.ndarray:
    """Squeezes or stretches randomly selected segments of the original data along the time axis."""
    num_samples, num_time_steps, num_channels = data.shape
    segment_time = num_time_steps // num_segments

    num_stretches = (num_segments - 1) // 2 + 1
    stretch_idx = np.random.choice(9, num_stretches, replace=False)
    squeeze_idx = set(range(num_segments)).difference(set(stretch_idx))

    time_warped_data = []
    for sample_idx in range(num_samples):
        time_warped_signal = []
        for i in range(num_segments):
            segment_start = int(i * segment_time)
            segment_end = segment_start + segment_time
            orig_data = data[sample_idx, segment_start:segment_end, :]

            if i in stretch_idx:
                output_shape = int(np.ceil(orig_data.shape[0] * stretch_factor))
            elif i in squeeze_idx:
                output_shape = int(np.ceil(orig_data.shape[0] * (1 / stretch_factor)))
            else:
                output_shape = orig_data.shape[0]

            new_signal = cv.resize(
                orig_data, (num_channels, output_shape), interpolation=cv.INTER_LINEAR
            )
            time_warped_signal.append(new_signal)

        time_warped_data.append(np.vstack(time_warped_signal))

    # reshape time dimension to the original shape
    time_warped_data = np.array(time_warped_data)
    if time_warped_data.shape[1] >= num_time_steps:
        start_idx = (time_warped_data.shape[1] - num_time_steps) // 2
        end_idx = start_idx + num_time_steps
        time_warped_data = time_warped_data[:, start_idx:end_idx, :]
    else:
        padding_before = (time_warped_data.shape[1] - num_time_steps) // 2
        padding_after = time_warped_data.shape[1] - num_time_steps + padding_before
        time_warped_data = np.pad(
            time_warped_data, ((0, 0), (padding_before, padding_after), (0, 0)), "edge"
        )
    return time_warped_data.astype(output_type)
