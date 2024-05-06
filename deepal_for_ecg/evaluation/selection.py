from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from deepal_for_ecg.evaluation.util import collect_experiment_runs_data
from deepal_for_ecg.strategies.query import SelectionStrategy


def collect_data(base_path: Path = Path("./experiments/al"), min_experiment_iterations: int = 21, trim_to_min: bool = True) -> Dict:
    """
    Collects the results of each iteration from each experiment for each strategy from the given directory.

    Args:
        base_path (Path): The base path of the experiment which data should be collected.
        min_experiment_iterations (int): The minimum iterations each experiment should have been run.
        trim_to_min (bool): An indicator whether just to collect the data until the minimum number of iterations has
            been reached or all data should be collected.

    Returns:
        A dictionary that is grouped by strategies holding the experiment results.
    """
    data_dict = dict()
    for strategy in SelectionStrategy:
        strategy_base_path = Path(base_path, strategy.value)
        if not strategy_base_path.exists():
            continue

        data_dict[strategy.value] = collect_experiment_runs_data(strategy_base_path, min_experiment_iterations, trim_to_min)
    return data_dict


def create_dataframe_for_plotting(data_dict: Dict, num_total_samples: int = 17418) -> pd.DataFrame:
    """Creates a dataframe that can be used for plotting."""
    all_data = None
    for strategy in SelectionStrategy:
        if strategy.value not in data_dict:
            continue

        strategy_name = get_plotting_name(strategy)

        for experiment, results in data_dict[strategy.value].items():
            auc_list = []
            num_samples_list = []
            al_iterations_list = []
            coverage_list = []
            for result in results:
                auc_list.append(result.auc)
                num_samples_list.append(result.num_samples)
                al_iterations_list.append(result.al_iteration)
                coverage_list.append(result.label_coverage)
            experiment_data = pd.DataFrame(np.array([auc_list, num_samples_list, al_iterations_list, coverage_list]).T, columns=["Macro AUC", "Number of samples", "AL iteration", "Label coverage"])
            experiment_data["Experiment"] = experiment
            experiment_data["Strategy"] = strategy_name
            experiment_data["Percentage of samples"] = experiment_data["Number of samples"] / num_total_samples * 100

            if all_data is None:
                all_data = experiment_data
            else:
                all_data = pd.concat([all_data, experiment_data])

    return all_data


def auc_coverage_plot(
        plotting_df: pd.DataFrame,
        time_value_to_use: str = "AL iteration",
        figure_filename: str = "plots/auc_coverage_plot.png",
        auc_value_to_use: str = "Macro AUC",
        coverage_value_to_use: str = "Label coverage",
        x_max: int = 20,
        auc_supervised_result: float = 0.8955587148666382
):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(ncols=2, figsize=(9, 4))
    # auc part
    g = sns.lineplot(plotting_df, y=auc_value_to_use, x=time_value_to_use, hue="Strategy", errorbar=("ci", 95),
                 ax=axes[0])
    g.axhline(auc_supervised_result, color="grey", linestyle="--")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    axes[0].set_xlim([0, x_max])

    # coverage part
    sns.lineplot(data=plotting_df, y=coverage_value_to_use, x=time_value_to_use, hue="Strategy",
                 errorbar=("ci", 95), ax=axes[1])
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    axes[1].set_xlim([0, x_max])

    fig.tight_layout()
    fig.savefig(figure_filename, dpi=600)


def results_over_time_plot(
        plotting_df: pd.DataFrame,
        time_value_to_use: str = "AL iteration",
        figure_filename: str = "results_over_iteration.png",
        result_value_to_use: str = "Macro AUC",
        with_title: bool = True,
        x_max: int = 20,
        auc_supervised_result: float = 0.8955587148666382
):
    """
    Creates a results over time plot from the given data.

    Args:
        plotting_df (pd.DataFrame): The data that should be plotted.
        time_value_to_use (str): The column name that should be used for the time dimension.
        figure_filename (str): The name of the figure file. Is used to store the figure.
        result_value_to_use (str): The colum name of the result that should be used for the y dimension.
        with_title (bool): Whether to add a title or not.
        x_max (int): The maximum value for the x axis. Defaults to 20
        auc_supervised_result (float): The average result of fully supervised training on all data
    """
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(7, 4))
    if with_title:
        fig.suptitle("Results of different selection strategies over time")
    g = sns.lineplot(plotting_df, y=result_value_to_use, x=time_value_to_use, hue="Strategy", errorbar=("ci", 95))

    if result_value_to_use == "Macro AUC":
        g.axhline(auc_supervised_result, color="grey", linestyle="--")

    if time_value_to_use == "AL iteration":
        fig.axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
        fig.axes[0].set_xlim(0, x_max)
    else:
        fig.axes[0].set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=600)


def get_plotting_name(strategy: SelectionStrategy) -> str:
    """Returns the name that should be used in plots."""
    if strategy == SelectionStrategy.RANDOM:
        return "Random"
    if strategy == SelectionStrategy.BADGE:
        return "BADGE"
    if strategy == SelectionStrategy.ENTROPY:
        return "Entropy"
    if strategy == SelectionStrategy.PLVI_CE_TOPK:
        return "PLVI-CE (top-k)"
    if strategy == SelectionStrategy.PLVI_CE_KNN:
        return "PLVI-CE (clust)"
