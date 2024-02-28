import numpy as np
from sklearn.preprocessing import StandardScaler
from wfdb.processing import resample_sig


def apply_standardizer(data, standard_scaler: StandardScaler) -> np.ndarray:
    """Scales the data with the given standardizer."""
    x_tmp = []
    for x in data:
        x_shape = x.shape
        x_tmp.append(standard_scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    return np.array(x_tmp).astype(np.float32)


def resample_multichannel_signal(orig_signal: np.ndarray, orig_frequency: int = 500, target_frequency: int = 100) -> np.ndarray:
    """Resamples a multichannel signal from the original frequency to the target frequency."""
    resampled_channels = []
    for channel in range(orig_signal.shape[1]):
        resampled_x, _ = resample_sig(orig_signal[:, channel], orig_frequency, target_frequency)
        resampled_channels.append(resampled_x)
    return np.column_stack(resampled_channels)
