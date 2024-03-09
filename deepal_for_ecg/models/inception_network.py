from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.models.classification_heads import BasicClassificationHeadConfig, simple_multi_label_classification_head


@dataclass
class InceptionNetworkConfig:
    """
    Configuration class to build Inception networks.

    Args:
        input_shape: Shape of the input tensor.
        num_classes: Number of classes to classify.
        num_filters (int): The number of filters each convolution in the inception modules should use.
        bottleneck_size (int): The number of filter that the bottleneck layer should use in the inception modules.
        use_residual (bool): Whether to add residual connections between the inception blocks.
        use_bottleneck (bool): Whether to use bottleneck layers in each inception module.
        depth (int): The depth of the inception network which is specified by the number of inception modules used.
            Should be divisible by three since each three inception modules a residual connection is added to the
            network.
        max_kernel_size (int): The maximum kernel size that should be used in the inception modules.
        use_handcrafted_filters (bool): Whether the network should be use handcrafted filters in the first inception
            module to detected typical time series pattern, i.e., positive trend, negative trend, and peaks
        max_cf_length (int): The maximum number of custom filters length to use per typ (increase, decrease, peak)
        with_classification_head (bool): Indicator whether a classification head should be added to the network.
             If false, just the representation part of the network will be returned.
        create_classification_head (callable): A function that can create the classification head from a
            BasicClassificationHEadConfig object.
    """
    input_shape: Tuple[int, int] = (250, 12)
    num_classes: int | None = PTBXLActiveLearningDataModule.NUM_CLASSES
    num_filters: int = 32
    bottleneck_size: int = 32
    use_residual: bool = True
    use_bottleneck: bool = True
    depth: int = 6
    max_kernel_size: int = 40
    use_handcrafted_filters: bool = False
    max_cf_length: int = 6
    with_classification_head: bool = True
    create_classification_head: callable = simple_multi_label_classification_head
    kernel_sizes: list[int] = field(init=False)
    increasing_trend_kernels: list[int] = field(init=False)
    decreasing_trend_kernels: list[int] = field(init=False)
    peak_kernels: list[int] = field(init=False)

    def __post_init__(self):
        self._calculate_kernel_sizes()

    def _calculate_kernel_sizes(self):
        """
        Calculates the kernel size used for convolutions in the inception modules.
        """
        self.kernel_sizes = [self.max_kernel_size // (2 ** i) for i in range(3)]
        handcrafted_kernel_sizes = [2 ** i for i in range(1, self.max_cf_length + 1)]
        self.increasing_trend_kernels = handcrafted_kernel_sizes
        self.decreasing_trend_kernels = handcrafted_kernel_sizes
        self.peak_kernels = handcrafted_kernel_sizes[1:]


class InceptionNetworkBuilder:
    """Builder class for inception networks."""

    def __init__(self):
        """
        Create a builder for inception networks.
        """
        self._config = None

    def build_model(
            self, config: InceptionNetworkConfig, representation_model: keras.Model | None = None
    ) -> keras.Model:
        """
        Builds the inception network model for the input and number of classes.

        Args:
            config (InceptionNetworkConfig): The configuration of the model that should be created.
            representation_model: (keras.Model): An optional representation model to use to extract the features. If
                specified, this representation model is used instead of creating a new one.

        Returns:
            The full inception network model as a composition of the representation part and the classification head or
            just the representation part as a keras.Model.
        """
        self._config = config

        input_layer = keras.layers.Input(self._config.input_shape, name=f"Input")

        # create the part of the network that extracts the features
        if representation_model is None:
            x = input_layer
            input_res = input_layer

            for d in range(self._config.depth):

                x = self._inception_module(x, d)

                if self._config.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x, d)
                    input_res = x

            tmp_representation_out = keras.layers.GlobalAveragePooling1D(name="GAP")(x)
            representation_model = keras.Model(
                inputs=input_layer,
                outputs=tmp_representation_out,
                name="Representation"
            )

        # output of the feature extraction part
        representation_out = representation_model(input_layer)

        if self._config.with_classification_head:
            config = BasicClassificationHeadConfig(
                num_input_units=representation_model.output_shape[-1],
                num_output_units=self._config.num_classes
            )
            classification_head = self._config.create_classification_head(config)
            output_layer = classification_head(representation_out)

            # return the full inception network
            return keras.Model(inputs=input_layer, outputs=output_layer, name="InceptionNetwork")
        else:
            # just return the feature extractor of the inception network
            return representation_model

    def _hybrid_layer(
            self,
            input_tensor: tf.Tensor,
            input_channels: int
    ):
        """
        Function to create the hybrid layer consisting of non-trainable Conv1D layers with custom filters.

        Args:
            input_tensor (tf.Tensor): The input tensor of the hybrid layer.
            input_channels : The number of input channels.
        """

        conv_list = []

        # for increasing detection filters
        for kernel_size in self._config.increasing_trend_kernels:
            # formula of increasing detection filter
            filter_ = np.ones(shape=(kernel_size, input_channels, 1))
            indices_ = np.arange(kernel_size)
            filter_[indices_ % 2 == 0] *= -1 

            conv = keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding="same", use_bias=False, 
                                       kernel_initializer=tf.keras.initializers.Constant(filter_), trainable=False,
                                       name=f"Hybrid_Increase_{kernel_size}")(input_tensor)
            conv_list.append(conv)

        # for decreasing detection filters
        for kernel_size in self._config.decreasing_trend_kernels:
            # formula of decreasing detection filter
            filter_ = np.ones(shape=(kernel_size, input_channels, 1))  
            indices_ = np.arange(kernel_size)
            filter_[indices_ % 2 > 0] *= -1  

            conv = keras.layers.Conv1D(filters=1, kernel_size=kernel_size, padding="same", use_bias=False, 
                                       kernel_initializer=tf.keras.initializers.Constant(filter_),trainable=False,
                                       name=f"Hybrid_Decrease_{kernel_size}")(input_tensor)
            conv_list.append(conv)

        # for peak detection filters
        for kernel_size in self._config.peak_kernels:
            # formula of peak detection filter
            filter_ = np.zeros(shape=(kernel_size + kernel_size // 2, input_channels, 1))
            xmesh = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))
            filter_left = xmesh ** 2
            filter_right = filter_left[::-1]
            filter_[0:kernel_size // 4] = -filter_left
            filter_[kernel_size // 4:kernel_size // 2] = -filter_right
            filter_[kernel_size // 2:3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4:kernel_size] = 2 * filter_right
            filter_[kernel_size:5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4:] = -filter_right

            conv = keras.layers.Conv1D(filters=1, kernel_size=kernel_size + kernel_size // 2, padding="same", 
                                       use_bias=False, kernel_initializer=tf.keras.initializers.Constant(filter_), 
                                       trainable=False, name=f"Hybrid_Peaks_{kernel_size}")(input_tensor)
            conv_list.append(conv)

        hybrid_layer = keras.layers.Concatenate(axis=2, name="Hybrid_Concat")(conv_list)
        hybrid_layer = keras.layers.Activation(activation="relu", name="Hybrid_Activation")(hybrid_layer)

        return hybrid_layer

    def _inception_module(
            self,
            input_tensor: tf.Tensor,
            current_depth: int,
            stride: int = 1,
            activation: str = "linear"
    ) -> tf.Tensor:
        """
        Adds an inception module to the network.

        Args:
            input_tensor (tf.Tensor): The input to the inception module.
            current_depth (int): The current depth of the network. Used to name the layers in the inception module.
            stride (int): The stride that the convolutions use (default = 1).
            activation (str): The activation function that the convolutions use (default = "linear").

        Returns:
            The output of the inception module.
        """
        # to have better layer names prepare a prefix for the module
        inception_module_name_prefix = f"InceptionModule{current_depth + 1}_"

        if self._use_bottleneck(int(input_tensor.shape[-1])):
            input_inception = keras.layers.Conv1D(name=f"{inception_module_name_prefix}Bottleneck",
                                                  filters=self._config.bottleneck_size, kernel_size=1, padding="same",
                                                  activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        conv_list = []

        for i, kernel_size in enumerate(self._config.kernel_sizes):
            conv_list.append(keras.layers.Conv1D(name=f"{inception_module_name_prefix}Convolution{kernel_size}",
                                                 filters=self._config.num_filters, kernel_size=kernel_size,
                                                 strides=stride, padding="same", activation=activation,
                                                 use_bias=False)(input_inception)
                             )

        max_pool_1 = keras.layers.MaxPool1D(name=f"{inception_module_name_prefix}MaxPool",
                                            pool_size=3, strides=stride, padding="same")(input_tensor)

        # TODO: Evaluate whether to just add the bottleneck layer if input channels > num_filters and input channels > 1
        conv_6 = keras.layers.Conv1D(name=f"{inception_module_name_prefix}MaxPool_Bottleneck",
                                     filters=self._config.num_filters, kernel_size=1,
                                     padding="same", activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        if self._config.use_handcrafted_filters and current_depth == 0:
            print("add handcrafted filters")
            hybrid = self._hybrid_layer(input_tensor, int(input_tensor.shape[-1]))
            conv_list.append(hybrid)

        x = keras.layers.Concatenate(name=f"{inception_module_name_prefix}Concat", axis=2)(conv_list)
        x = keras.layers.BatchNormalization(name=f"{inception_module_name_prefix}BatchNorm")(x)
        x = keras.layers.Activation(name=f"{inception_module_name_prefix}Activation", activation="relu")(x)
        return x

    def _shortcut_layer(
            self,
            residual_input_tensor: tf.Tensor,
            other_tensor: tf.Tensor,
            current_depth: int
    ) -> tf.Tensor:
        """
        Adds a shortcut layer to the network.

        Args:
            residual_input_tensor (tf.Tensor): The residual input that should be transferred through the shortcut layer.
            other_tensor (tf.Tensor): The other input the residual input should be added to.
            current_depth (int): The current depth of the network. Is used to name the layers in the network.
        Returns:
            The output of the shortcut layer.
        """
        # to have better layer names prepare a prefix for the module
        num_shortcut = int(current_depth / 3) + 1
        shortcut_layer_name_prefix = f"Shortcut{num_shortcut}_"

        shortcut_y = keras.layers.Conv1D(name=f"{shortcut_layer_name_prefix}ResidualConvolution",
                                         filters=int(other_tensor.shape[-1]), kernel_size=1,
                                         padding="same", use_bias=False)(residual_input_tensor)
        shortcut_y = keras.layers.BatchNormalization(name=f"{shortcut_layer_name_prefix}ResidualBatchNorm")(shortcut_y)

        x = keras.layers.Add(name=f"{shortcut_layer_name_prefix}Add")([shortcut_y, other_tensor])
        x = keras.layers.Activation("relu", name=f"{shortcut_layer_name_prefix}Activation")(x)
        return x

    def _use_bottleneck(self, input_channels: int) -> bool:
        """
        Validates whether it is sensible to use a bottleneck layer. In addition to the original paper, a bottleneck
        layer is just added when the number of input channels is greater than the number of bottleneck channels.

        Args:
            input_channels (int): The number of input channels.
        Returns:
            An indicator whether to use the bottleneck layer or not.
        """
        return self._config.use_bottleneck and input_channels > self._config.bottleneck_size and input_channels > 1
