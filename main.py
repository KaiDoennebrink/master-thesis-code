from pathlib import Path

import numpy as np
import typer
from tensorflow import keras

from deepal_for_ecg.data.load import PTBXLDataLoader
from deepal_for_ecg.strategies.initalize.pt4al import PreTextLossInitQueryStrategy

app = typer.Typer()


@app.command()
def hello(name: str):
    print(f"Hello {name}")


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
