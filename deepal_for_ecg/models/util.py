from tensorflow import keras


def get_representation_part_of_model(model: keras.Model) -> keras.Model:
    """Returns the representation part of a given model. The part has to be named 'Representation'."""
    for layer in model.layers:
        if layer.name == "Representation":
            return layer
