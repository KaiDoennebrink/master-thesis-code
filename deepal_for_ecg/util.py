import tensorflow as tf
from tensorflow.keras import mixed_precision


def improve_gpu_capacity(
        increase_memory: bool = True,
        memory_limit: int = 5292,
        use_mixed_precision: bool = False,
        use_dynamic_growth: bool = False
):
    """
    Improves the GPU capacity that can be used for training a deep neural network.

    Args:
        increase_memory (bool): Indicator whether to increase the memory limit to the given limit.
        memory_limit (int): The memory limit to use when the memory should be increased.
        use_mixed_precision (bool): Indicator whether to use mixed precisions during training to speed up training.
        use_dynamic_growth (bool): Indicator whether to use dynamic growth for the GPU capacity. Just has an effect
            if increase_memory is set to False.
    """
    if use_mixed_precision:
        # set global policy
        mixed_precision.set_global_policy('mixed_float16')

    # increase memory or use dynamic memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if increase_memory:
            tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
        elif use_dynamic_growth:
            tf.config.experimental.set_memory_growth(gpus[0], True)