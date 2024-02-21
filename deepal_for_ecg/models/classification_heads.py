from dataclasses import dataclass

from tensorflow import keras


@dataclass
class BasicClassificationHeadConfig:
    """Configuration class for a basic classification head."""
    num_input_units: int
    num_output_units: int


def simple_multi_label_classification_head(config: BasicClassificationHeadConfig) -> keras.Model:
    """
    Creates a simple classification head for a multi-label classification task.
    It consists of a simple dense layer between the input and the output with a sigmoid activation function.

    Args:
        config (BasicClassificationHeadConfig): The configuration of the classification head.
    Returns:
        The classification head as a keras.Model for a multi-label classification task.
    """
    input_layer = keras.layers.Input([config.num_input_units], name=f"Classification_Input")
    output_layer = keras.layers.Dense(
        config.num_output_units,
        activation="sigmoid",
        name=f"Classification_Output"
    )(input_layer)

    return keras.Model(inputs=input_layer, outputs=output_layer, name="Simple_Multi_Label_Classification_Head")


def simple_classification_head(config: BasicClassificationHeadConfig):
    """
    Creates a simple classification head with two dense layers and a softmax output layer.

    Args:
        config (BasicClassificationHeadConfig): The configuration of the classification head.
    Returns:
        The classification head as a keras.Model for a classification task.
    """
    input_layer = keras.layers.Input([config.num_input_units], name=f"Classification_Input")
    x = keras.layers.Dense(128, activation="relu", name=f"Classification_Dense_1")(input_layer)
    x = keras.layers.Dense(128, activation="relu", name=f"Classification_Dense_2")(input_layer)
    output_layer = keras.layers.Dense(config.num_output_units, activation="softmax", name=f"Classification_Output")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer, name="Simple_Multi_Label_Classification_Head")
