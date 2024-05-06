from pathlib import Path

from tensorflow import keras
import numpy as np

from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.train.test_util import test_step_with_sliding_windows


def test_all_supervised_models(base_name: str = "full_supervised_PTBXLInception", num_models: int = 5):
    # load the prepared data
    data_loader = PTBXLDataLoader(load_saved_data=True, saved_data_base_dir=Path("./data/saved/ptbxl"))
    data_loader.load_data()
    data_module = PTBXLActiveLearningDataModule(
        train_samples=data_loader.X_train,
        test_samples=data_loader.X_test,
        val_samples=data_loader.X_valid,
        train_labels_12sl=data_loader.Y_train_12sl,
        train_labels_ptb_xl=data_loader.Y_train_ptb_xl,
        test_labels=data_loader.Y_test,
        val_labels=data_loader.Y_valid
    )

    losses = []
    aucs = []
    accuracies = []

    for i in range(num_models):
        # load the model
        model_name = f"{base_name}{i + 1}"
        model = keras.models.load_model(Path(f"./models/{model_name}/best_model.keras"))

        # prepare the loss and metrics
        loss_object = keras.losses.BinaryCrossentropy()
        test_loss = keras.metrics.Mean(name="test_loss")
        test_accuracy = keras.metrics.BinaryAccuracy(name="test_accuracy")
        test_auc = keras.metrics.AUC(multi_label=True, name="test_auc", num_labels=PTBXLActiveLearningDataModule.NUM_CLASSES)

        for samples, labels in data_module.test_dataset.batch(128):
            test_step_with_sliding_windows(
                sliding_window_batch=samples,
                label_batch=labels,
                model=model,
                loss_object=loss_object,
                loss_based_metrics=[test_loss],
                prediction_based_metrics=[test_accuracy, test_auc],
            )

        losses.append(test_loss.result().numpy())
        accuracies.append(test_accuracy.result().numpy())
        aucs.append(test_auc.result().numpy())

    print(f"Average loss: {np.mean(losses)}")
    print(f"Average auc: {np.mean(aucs)}")
    print(f"Average accuracy: {np.mean(accuracies)}")
