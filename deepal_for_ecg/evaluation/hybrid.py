from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator

from deepal_for_ecg.evaluation.util import collect_experiment_runs_data


def collect_data(
        full_human_experiment_dir: Path = Path("./experiments/full_human/entropy"),
        hybrid_experiment_dir: Path = Path("./experiments/hybrid_label/entropy"),
        full_wsa_experiment_dir: Path = Path("./experiments/wsa/entropy"),
        min_iterations: int = 21
) -> Dict:
    data_dict = dict()
    data_dict["full_human"] = collect_experiment_runs_data(full_human_experiment_dir, min_experiment_iterations=min_iterations)
    data_dict["hybrid"] = collect_experiment_runs_data(hybrid_experiment_dir, min_experiment_iterations=min_iterations)
    data_dict["full_wsa"] = collect_experiment_runs_data(full_wsa_experiment_dir, min_experiment_iterations=min_iterations)
    return data_dict


def create_dataframe(data_dict: Dict, cost_factor_ha: float = 1, cost_factor_wsa: float = 0.1):
    cost_df = None
    for experiment, results in data_dict["hybrid"].items():
        auc_list = []
        samples_from_ha = []
        cum_samples_from_ha = []
        samples_from_wsa = []
        cum_samples_from_wsa = []
        al_iterations_list = []

        for result in results:
            auc_list.append(result.auc)
            samples_from_ha.append(result.num_newly_samples)
            cum_samples_from_ha.append(result.num_samples)
            samples_from_wsa.append(result.num_newly_samples_from_wsa)
            cum_samples_from_wsa.append(result.num_samples_from_wsa)
            al_iterations_list.append(result.al_iteration)

        experiment_data = pd.DataFrame(np.array([auc_list, samples_from_ha, cum_samples_from_ha, samples_from_wsa, cum_samples_from_wsa, al_iterations_list]).T,
                                       columns=["Macro AUC", "samples_from_ha", "cum_samples_from_ha", "samples_from_wsa", "cum_samples_from_wsa", "AL iteration"])
        experiment_data["Cumulative costs"] = experiment_data["cum_samples_from_ha"] * (cost_factor_ha + cost_factor_wsa) + experiment_data["cum_samples_from_wsa"] * cost_factor_wsa
        experiment_data["Annotator setting"] = "hybrid"
        experiment_data["AL iteration"] = experiment_data["AL iteration"].astype(int)

        if cost_df is None:
            cost_df = experiment_data
        else:
            cost_df = pd.concat([cost_df, experiment_data])

    for experiment, results in data_dict["full_human"].items():
        auc_list = []
        samples_from_ha = []
        cum_samples_from_ha = []
        samples_from_wsa = []
        cum_samples_from_wsa = []
        al_iterations_list = []

        for result in results:
            auc_list.append(result.auc)
            samples_from_ha.append(result.num_newly_samples)
            cum_samples_from_ha.append(result.num_samples)
            samples_from_wsa.append(0)
            cum_samples_from_wsa.append(0)
            al_iterations_list.append(result.al_iteration)

        experiment_data = pd.DataFrame(np.array([auc_list, samples_from_ha, cum_samples_from_ha, samples_from_wsa, cum_samples_from_wsa, al_iterations_list]).T,
                                       columns=["Macro AUC", "samples_from_ha", "cum_samples_from_ha", "samples_from_wsa", "cum_samples_from_wsa", "AL iteration"])
        experiment_data["Cumulative costs"] = experiment_data["cum_samples_from_ha"] * cost_factor_ha + experiment_data["cum_samples_from_wsa"] * cost_factor_wsa
        experiment_data["Annotator setting"] = "full-human"
        cost_df = pd.concat([cost_df, experiment_data])

    for experiment, results in data_dict["full_wsa"].items():
        auc_list = []
        samples_from_ha = []
        cum_samples_from_ha = []
        samples_from_wsa = []
        cum_samples_from_wsa = []
        al_iterations_list = []

        for result in results:
            auc_list.append(result.auc)
            samples_from_ha.append(0)
            cum_samples_from_ha.append(0)
            samples_from_wsa.append(result.num_newly_samples)
            cum_samples_from_wsa.append(result.num_samples)
            al_iterations_list.append(result.al_iteration)

        experiment_data = pd.DataFrame(np.array([auc_list, samples_from_ha, cum_samples_from_ha, samples_from_wsa, cum_samples_from_wsa, al_iterations_list]).T,
                                       columns=["Macro AUC", "samples_from_ha", "cum_samples_from_ha", "samples_from_wsa", "cum_samples_from_wsa", "AL iteration"])
        experiment_data["Cumulative costs"] = experiment_data["cum_samples_from_ha"] * cost_factor_ha + experiment_data["cum_samples_from_wsa"] * cost_factor_wsa
        experiment_data["Annotator setting"] = "full-wsa"
        cost_df = pd.concat([cost_df, experiment_data])

    return cost_df


def cost_plot(
        plotting_df: pd.DataFrame,
        cost_value_to_use: str = "Cumulative costs",
        time_value_to_use: str = "AL iteration",
        figure_filename: str = "annotator_cost_plot.png",
        with_title: bool = True
):
    """
    Plots the cumulative costs of each experiment run grouped by the annotator setting over time.

    Args:
        plotting_df (pd.DataFrame): The data that should be plotted.
        cost_value_to_use (str): The column name of the cost value to use for the y dimension.
        time_value_to_use (str): The column name of the time value to use for the x dimension.
        figure_filename (str): The name of the figure file. Is used to store the figure.
        with_title (bool): Whether to add a title or not.
    """
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    if with_title:
        fig.suptitle("Cumulative costs of different annotator settings")
    sns.lineplot(data=plotting_df, y=cost_value_to_use, x=time_value_to_use, hue="Annotator setting", errorbar=("ci", 95))
    fig.axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    fig.tight_layout()
    fig.savefig(figure_filename)


def auc_plot(
        plotting_df: pd.DataFrame,
        time_value_to_use: str = "AL iteration",
        figure_filename: str = "annotator_auc_plot.png",
        result_value_to_use: str = "Macro AUC",
        with_title: bool = True
):
    """
    Plots the Macro AUC of each experiment run grouped by the annotator setting over time.

    Args:
        plotting_df (pd.DataFrame): The data that should be plotted.
        time_value_to_use (str): The column name that should be used for the time dimension.
        figure_filename (str): The name of the figure file. Is used to store the figure.
        result_value_to_use (str): The column name that should be used for the y dimension.
        with_title (bool): Whether to add a title or not.
    """
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    if with_title:
        fig.suptitle("Results of different annotator settings over time")
    sns.lineplot(plotting_df, y=result_value_to_use, x=time_value_to_use, hue="Annotator setting", errorbar=("ci", 95))
    fig.axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    fig.tight_layout()
    fig.savefig(figure_filename)


def samples_plot(
        plotting_df: pd.DataFrame,
        figure_filename: str = "annotator_samples_plot.png",
        with_title: bool = True
):
    """
    Plots the number of requested samples from each annotator over time for the hybrid annotator setting.

    Args:
        plotting_df (pd.DataFrame): The data that should be plotted.
        figure_filename (str): The name of the figure file. Is used to store the figure.
        with_title (bool): Whether to add a title or not.
    """
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    if with_title:
        fig.suptitle("Number of samples from the different annotators over time in the hybrid setting")

    # prepare the data
    hybrid_plotting_df = plotting_df[plotting_df["Annotator setting"] == "hybrid"]
    hybrid_plotting_df = hybrid_plotting_df.rename(columns={"samples_from_ha": "Human", "samples_from_wsa": "Weak supervision"})
    prepared_plotting_df = pd.melt(hybrid_plotting_df, ["AL iteration"], value_vars=["Human", "Weak supervision"], value_name="Number of samples", var_name="Annotator")

    sns.lineplot(x="AL iteration", y="Number of samples", hue="Annotator",
                 data=prepared_plotting_df, palette=sns.color_palette()[1:3])
    fig.axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 4, 5, 10]))
    fig.tight_layout()
    fig.savefig(figure_filename)
