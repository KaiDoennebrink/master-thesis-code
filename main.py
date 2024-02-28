from pathlib import Path

import numpy as np
import typer
from tensorflow import keras

from deepal_for_ecg.data.load import PTBXLDataLoader
from deepal_for_ecg.data.tranformation_recognition import TransformationRecognitionDataModule
from deepal_for_ecg.models.classification_heads import simple_classification_head
from deepal_for_ecg.models.inception_network import InceptionNetworkConfig, InceptionNetworkBuilder
from deepal_for_ecg.strategies.initalize.pt4al import PreTextLossInitQueryStrategy
from deepal_for_ecg.train.transformation_recognition import TransformationRecognitionTrainer
from deepal_for_ecg.util import improve_gpu_capacity

improve_gpu_capacity()

app = typer.Typer()


@app.command()
def train_pretext_model(base_name: str = "PretextInception", num_models: int = 1):
    """
    Trains models on the pretext tasks.

    Args:
        base_name (str): The base name of the model that is used to store it.
        num_models (int): The number of models that should be trained on the pretext tasks (default = 1).
    """
    # load data
    data_module = TransformationRecognitionDataModule()

    # prepare the model builder
    config = InceptionNetworkConfig(create_classification_head=simple_classification_head,
                                    num_classes=data_module.NUM_TRANSFORMATIONS, input_shape=(1000, 12))
    builder = InceptionNetworkBuilder()

    # train the models
    for i in range(num_models):
        model = builder.build_model(config)
        trainer = TransformationRecognitionTrainer(model=model, model_name=f"{base_name}{i+1}")
        trainer.fit(data_module.train_dataset, data_module.validation_dataset)


@app.command()
def prepare_data_ptbxl():
    """
    Loads, preprocesses, and saves the PTB-XL dataset.
    """
    data_loader = PTBXLDataLoader()
    data_loader.load_data()
    data_loader.save_data()


@app.command()
def prepare_data_for_pretext_task():
    """
    Prepares the losses for the PT4AL query strategy for the given pre-trained model.
    """
    # load saved PTB-XL data
    loader = PTBXLDataLoader(load_saved_data=True)
    loader.load_data()
    signal_data = loader.X_train.astype(np.float32)

    # delete the loader to free up memory
    del loader

    data_module = TransformationRecognitionDataModule()
    data_module.generate_and_split_data(signal_data)
    data_module.prepare_datasets()


@app.command()
def prepare_loss_for_pt4al(model_name: str):
    """
    Prepares the losses for the PT4AL query strategy for the given pre-trained model.
    """
    # load the data
    loader = PTBXLDataLoader(load_saved_data=True)
    loader.load_data()
    signal_data = loader.X_train.astype(np.float32)
    # load the model
    b_model = keras.models.load_model(f"./models/{model_name}/best_model.keras")
    # prepare the losses
    init_strategy = PreTextLossInitQueryStrategy(b_model, Path(f"./data/loss_storage/{model_name}"))
    init_strategy.prepare(signal_data)


if __name__ == "__main__":
    app()
