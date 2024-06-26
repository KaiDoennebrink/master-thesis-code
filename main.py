from pathlib import Path

import numpy as np
import typer
from tensorflow import keras

from deepal_for_ecg.data.loader.icbeb import ICBEBDataLoader
from deepal_for_ecg.data.loader.ptbxl import PTBXLDataLoader
from deepal_for_ecg.data.module.active_learning import PTBXLActiveLearningDataModule
from deepal_for_ecg.data.module.icbeb import ICBEBDataModule
from deepal_for_ecg.data.module.tranformation_recognition import TransformationRecognitionDataModule
from deepal_for_ecg.experiments.initialization_strategy import InitializationStrategyExperiment, \
    InitializationExperimentConfig
from deepal_for_ecg.experiments.base import BaseExperimentConfig, BaseExperiment
from deepal_for_ecg.experiments.multi_annotator import HybridAnnotatorSelectionExperimentConfig, \
    HybridAnnotatorSelectionExperiment
from deepal_for_ecg.experiments.wsa import WSAExperimentConfig, WeakSupervisionAnnotatorExperiment
from deepal_for_ecg.strategies.annotator import HybridAnnotatorModelSetting
from deepal_for_ecg.strategies.initalize import InitializationStrategy
from deepal_for_ecg.strategies.query import SelectionStrategy
from deepal_for_ecg.evaluation import selection, initialization, hybrid, supervised
from deepal_for_ecg.models.classification_heads import simple_classification_head
from deepal_for_ecg.models.inception_network import InceptionNetworkConfig, InceptionNetworkBuilder
from deepal_for_ecg.strategies.initalize.pt4al import PreTextLossInitQueryStrategy
from deepal_for_ecg.train.time_series import MultiLabelTimeSeriesTrainer
from deepal_for_ecg.train.transformation_recognition import TransformationRecognitionTrainer
from deepal_for_ecg.util import improve_gpu_capacity

improve_gpu_capacity()

app = typer.Typer()


@app.command()
def experiment_init_strategy(runs_per_strategy: int = 10, initial_samples: int = 300):
    """Executes the initialization strategy experiment. The results are saved in a csv-file."""
    config = InitializationExperimentConfig(runs_per_strategy=runs_per_strategy, initial_samples=initial_samples)
    experiment = InitializationStrategyExperiment(config)
    experiment.run()


@app.command()
def experiment_selection_strategy(strategy: SelectionStrategy, experiment_name: str, num_al_iterations: int = 20):
    """Executes a selection strategy experiment with the full human setting."""
    config = BaseExperimentConfig(
        name=experiment_name,
        strategy=strategy,
        num_al_iterations=num_al_iterations
    )
    experiment = BaseExperiment(config=config)
    experiment.run()


@app.command()
def experiment_full_human(experiment_base_name: str, experiment_num: int = 1, num_al_iterations: int = 20):
    """Executes a selection strategy experiment with the full human setting."""
    experiment_name = f"{experiment_base_name}_{experiment_num}"
    config = BaseExperimentConfig(
        base_experiment_dir=Path("./experiments/full_human"),
        strategy=SelectionStrategy.ENTROPY,
        name=experiment_name,
        num_al_iterations=num_al_iterations,
        start_model_dir=Path(f"./experiments/hybrid_label/entropy/run_{experiment_num}/models/start_model.keras"),
        init_strategy=InitializationStrategy.REPRESENTATION_CLUSTER_PRETEXT,
        init_strategy_pretrained_model_run_number=experiment_num,
        init_strategy_pretrained_model_dir=Path("./models"),
        init_strategy_pretrained_model_base_name="PretextInception"
    )
    experiment = BaseExperiment(config=config)
    experiment.run()


@app.command()
def experiment_full_wsa(experiment_base_name: str, experiment_num: int = 1, num_al_iterations: int = 20):
    """Executes a selection strategy experiment with the full wsa setting."""
    experiment_name = f"{experiment_base_name}_{experiment_num}"
    config = WSAExperimentConfig(
        name=experiment_name,
        num_al_iterations=num_al_iterations,
        init_strategy_pretrained_model_run_number=experiment_num,
        start_model_dir=Path(f"./experiments/hybrid_label/entropy/run_{experiment_num}/models/start_model.keras")
    )
    experiment = WeakSupervisionAnnotatorExperiment(config=config)
    experiment.run()


@app.command()
def experiment_hybrid(
        experiment_base_name: str,
        experiment_num: int = 1,
        num_al_iterations: int = 20,
        annotator_model_setting: HybridAnnotatorModelSetting = HybridAnnotatorModelSetting.LABEL_BASED_MODEL
):
    """Executes a selection strategy experiment with the full wsa setting."""
    experiment_name = f"{experiment_base_name}_{experiment_num}"
    base_dir = Path(f"./experiments/hybrid_{annotator_model_setting.value}")
    config = HybridAnnotatorSelectionExperimentConfig(
        name=experiment_name,
        run_number=experiment_num,
        base_experiment_dir=base_dir,
        num_al_iterations=num_al_iterations,
        init_strategy_pretrained_model_run_number=experiment_num,
        annotator_model_setting=annotator_model_setting
    )
    experiment = HybridAnnotatorSelectionExperiment(config=config)
    experiment.run()


@app.command()
def evaluate_selection_strategy(min_experiment_iterations: int = 21, with_title: bool = False):
    """Evaluates the selection strategy experiments."""
    raw_results = selection.collect_data(min_experiment_iterations=min_experiment_iterations)
    plotting_df = selection.create_dataframe_for_plotting(raw_results)
    selection.auc_coverage_plot(plotting_df)
    selection.results_over_time_plot(plotting_df, with_title=with_title, figure_filename="plots/results_over_iteration.png", x_max=min_experiment_iterations-1)
    selection.results_over_time_plot(plotting_df, time_value_to_use="Percentage of samples", figure_filename="plots/results_over_percentage.png", with_title=with_title)
    selection.results_over_time_plot(plotting_df, result_value_to_use="Label coverage", figure_filename="plots/coverage_over_iteration.png", with_title=with_title, x_max=min_experiment_iterations-1)

@app.command()
def evaluate_hybrid(min_experiment_iterations: int = 21):
    """Evaluates the selection strategy experiments."""
    raw_results = selection.collect_data(base_path=Path("./experiments/hybrid_Label"), min_experiment_iterations=min_experiment_iterations)
    plotting_df = selection.create_dataframe_for_plotting(raw_results)
    selection.results_over_time_plot(plotting_df, figure_filename="plots/results_over_iteration_hybrid.png")
    selection.results_over_time_plot(plotting_df, time_value_to_use="Percentage of samples", figure_filename="plots/results_over_percentage_hybrid.png")
    selection.results_over_time_plot(plotting_df, result_value_to_use="Label coverage", figure_filename="plots/coverage_over_iteration_hybrid.png")


@app.command()
def evaluate_hybrid_annotator(min_experiment_iterations: int = 21, with_title: bool = False):
    """Evaluates the selection strategy experiments."""
    raw_results = hybrid.collect_data(min_iterations=min_experiment_iterations)
    plotting_df = hybrid.create_dataframe(raw_results)
    hybrid.cost_plot(plotting_df, with_title=with_title)
    hybrid.auc_plot(plotting_df, with_title=with_title)
    hybrid.samples_plot(plotting_df, with_title=with_title)
    hybrid.auc_cost_plot(plotting_df)


@app.command()
def evaluate_init_strategy():
    initialization.plot_init_strategy_results()


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
    config = InceptionNetworkConfig(
        create_classification_head=simple_classification_head,
        num_classes=data_module.NUM_TRANSFORMATIONS,
        input_shape=(1000, 12),
    )
    builder = InceptionNetworkBuilder()

    # train the models
    for i in range(num_models):
        model = builder.build_model(config)
        trainer = TransformationRecognitionTrainer(
            model=model, model_name=f"{base_name}{i+1}"
        )
        trainer.fit(data_module.train_dataset, data_module.validation_dataset)


@app.command()
def train_icbeb_model(base_name: str = "ICBEBInception", num_models: int = 1):
    """
    Trains models on the ICBEB 2018 dataset.

    Args:
        base_name (str): The base name of the model that is used to store it.
        num_models (int): The number of models that should be trained on the pretext tasks (default = 1).
    """
    # load data
    data_module = ICBEBDataModule()

    # prepare the model builder
    config = InceptionNetworkConfig(
        num_classes=data_module.NUM_CLASSES, input_shape=(250, 12)
    )
    builder = InceptionNetworkBuilder()

    # train the models
    for i in range(num_models):
        model = builder.build_model(config)
        trainer = MultiLabelTimeSeriesTrainer(
            model=model,
            model_name=f"{base_name}{i+1}",
            num_labels=data_module.NUM_CLASSES,
        )
        trainer.fit(data_module.train_dataset, data_module.validation_dataset)


@app.command()
def train_ptbxl_model_fully_supervised(
    base_name: str = "full_supervised_PTBXLInception", num_models: int = 1
):
    """
    Trains models on the PTBXL dataset in a fully supervised fashion.

    Args:
        base_name (str): The base name of the model that is used to store it.
        num_models (int): The number of models that should be trained on the pretext tasks (default = 1).
    """
    # load data
    data_loader = PTBXLDataLoader(load_saved_data=True)
    data_loader.load_data()
    data_module = PTBXLActiveLearningDataModule(
        train_samples=data_loader.X_train,
        test_samples=data_loader.X_test,
        val_samples=data_loader.X_valid,
        train_labels_12sl=data_loader.Y_train_12sl,
        train_labels_ptb_xl=data_loader.Y_train_ptb_xl,
        test_labels=data_loader.Y_test,
        val_labels=data_loader.Y_valid,
    )

    # buy all labels from the original ptb-xl dataset
    data_module_state = data_module.state_dict()
    data_module.update_annotations(
        buy_idx_ptb_xl=data_module_state["unlabeled_indices"], buy_idx_12sl=set()
    )

    # prepare the model builder
    config = InceptionNetworkConfig()
    builder = InceptionNetworkBuilder()

    # train the models
    for i in range(num_models):
        model = builder.build_model(config)
        trainer = MultiLabelTimeSeriesTrainer(
            model=model,
            model_name=f"{base_name}{i+1}",
            num_labels=data_module.NUM_CLASSES,
            epochs=100
        )
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
def prepare_data_icbeb():
    """
    Loads, preprocesses, and saves the ICBEB 2018 dataset with the data loader. Afterward, the train, test, and
    validation datasets are prepared with the data module.
    """
    data_loader = ICBEBDataLoader()
    data_loader.load_data()
    data_loader.save_data()

    data_module = ICBEBDataModule()
    data_module.prepare_datasets(
        data_loader.X_train, data_loader.X_val, data_loader.Y_train, data_loader.Y_val
    )


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
    init_strategy = PreTextLossInitQueryStrategy(b_model, model_name)
    init_strategy.prepare(signal_data)


@app.command()
def get_supervised_results():
    supervised.test_all_supervised_models()


if __name__ == "__main__":
    app()
