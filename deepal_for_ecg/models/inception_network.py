from typing import Tuple

import tensorflow as tf
from tensorflow import keras


class InceptionNetworkBuilder:
    """Builder class for inception networks."""

    def __init__(self,
                 num_filters: int = 32,
                 bottleneck_size: int = 32,
                 use_residual: bool = True,
                 use_bottleneck: bool = True,
                 depth: int = 6,
                 max_kernel_size: int = 40
                 ):
        """
        Create a builder for inception networks with the given parameters.

        :param num_filters: The number of filters each convolution in the inception modules should use.
        :param bottleneck_size: The number of filter that the bottleneck layer should use in the inception modules.
        :param use_residual: Whether to add residual connections between the inception blocks.
        :param use_bottleneck: Whether to use bottleneck layers in each inception module.
        :param depth: The depth of the inception network which is specified by the number of inception modules used.
        Should be divisible by three since each three inception modules a residual connection is added to the network.
        :param max_kernel_size: The maximum kernel size that should be used in the inception modules.
        """
        self.num_filters = num_filters
        self.bottleneck_size = bottleneck_size
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self._calculate_kernel_sizes(max_kernel_size)

    def build_model(self, input_shape: Tuple[int, int], num_classes: int, output_activation: str = "softmax") -> keras.Model:
        """
        Builds the inception network model for the input and number of classes.

        :param input_shape: The shape of the input to the network in the form (time steps, channels).
        :param num_classes: The number of classes the classification head should have.
        :param output_activation: The activation function that should be used for the output layer of the network.
        For example, use 'softmax' for single label classification and 'sigmoid' for multilabel classification.
        :return: The inception network model.
        """
        input_layer = keras.layers.Input(input_shape, name=f"Input")

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x, d)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, d)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D(name="GAP")(x)

        output_layer = keras.layers.Dense(num_classes, activation=output_activation,
                                          name=f"Output_{output_activation}")(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

    def _inception_module(self,
                          input_tensor: tf.Tensor,
                          current_depth: int,
                          stride: int = 1,
                          activation: str = 'linear') -> tf.Tensor:
        """
        Adds an inception module to the network.
        :param input_tensor: The input to the inception module.
        :param current_depth: The current depth of the network. Used to name the layers in the inception module.
        :param stride: The stride that the convolutions use (default = 1).
        :param activation: The activation function that the convolutions use (default = 'linear').
        :return: The output of the inception module.
        """
        # to have better layer names prepare a prefix for the module
        inception_module_name_prefix = f"InceptionModule{current_depth + 1}_"

        if self._use_bottleneck(int(input_tensor.shape[-1])):
            input_inception = keras.layers.Conv1D(name=f"{inception_module_name_prefix}Bottleneck",
                                                  filters=self.bottleneck_size, kernel_size=1, padding='same',
                                                  activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        conv_list = []

        for i, kernel_size in enumerate(self.kernel_sizes):
            conv_list.append(keras.layers.Conv1D(name=f"{inception_module_name_prefix}Convolution{kernel_size}",
                                                 filters=self.num_filters, kernel_size=kernel_size,
                                                 strides=stride, padding='same', activation=activation,
                                                 use_bias=False)(input_inception)
                             )

        max_pool_1 = keras.layers.MaxPool1D(name=f"{inception_module_name_prefix}MaxPool",
                                            pool_size=3, strides=stride, padding='same')(input_tensor)

        # TODO: Evaluate whether to just add the bottleneck layer if input channels > num_filters and input channels > 1
        conv_6 = keras.layers.Conv1D(name=f"{inception_module_name_prefix}MaxPool_Bottleneck",
                                     filters=self.num_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(name=f"{inception_module_name_prefix}Concat", axis=2)(conv_list)
        x = keras.layers.BatchNormalization(name=f"{inception_module_name_prefix}BatchNorm")(x)
        x = keras.layers.Activation(name=f"{inception_module_name_prefix}Activation", activation='relu')(x)
        return x

    def _shortcut_layer(self,
                        residual_input_tensor: tf.Tensor,
                        other_tensor: tf.Tensor,
                        current_depth: int) -> tf.Tensor:
        """
        Adds a shortcut layer to the network.

        :param residual_input_tensor: The residual input that should be transferred through the shortcut layer.
        :param other_tensor: The other input the residual input should be added to.
        :param current_depth: The current depth of the network. Is used to name the layers in the network.
        :return: The output of the shortcut layer.
        """
        # to have better layer names prepare a prefix for the module
        num_shortcut = int(current_depth / 3) + 1
        shortcut_layer_name_prefix = f"Shortcut{num_shortcut}_"

        shortcut_y = keras.layers.Conv1D(name=f"{shortcut_layer_name_prefix}ResidualConvolution",
                                         filters=int(other_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(residual_input_tensor)
        shortcut_y = keras.layers.BatchNormalization(name=f"{shortcut_layer_name_prefix}ResidualBatchNorm")(shortcut_y)

        x = keras.layers.Add(name=f"{shortcut_layer_name_prefix}Add")([shortcut_y, other_tensor])
        x = keras.layers.Activation('relu', name=f"{shortcut_layer_name_prefix}Activation")(x)
        return x

    def _calculate_kernel_sizes(self, max_kernel_size: int):
        """
        Calculates based on the maximum kernel size the kernel sized used for convolutions in the inception modules.
        :param max_kernel_size:
        :return:
        """
        self.kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]

    def _use_bottleneck(self, input_channels: int) -> bool:
        """
        Validates whether it is sensible to use a bottleneck layer. In addition to the original paper, a bottleneck
        layer is just added when the number of input channels is greater than the number of bottleneck channels.

        :param input_channels: The number of input channels.
        :return: An indicator whether to use the bottleneck layer or not.
        """
        return self.use_bottleneck and input_channels > self.bottleneck_size and input_channels > 1
