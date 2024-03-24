import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from deepal_for_ecg.experiments.base import BaseExperimentALIterationResult
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
        experiments_dir = dict()
        strategy_base_path = Path(base_path, strategy.value)
        for experiment_dir in strategy_base_path.iterdir():
            experiment_results_dir = Path(experiment_dir, "results")
            al_iteration_result_paths = []
            for p in experiment_results_dir.iterdir():
                if p.match("al_iteration_[0-9]*.pkl"):
                    al_iteration_result_paths.append(p)

            # check whether the experiment has enough iterations
            num_of_experiment_iterations = len(al_iteration_result_paths)
            if num_of_experiment_iterations >= min_experiment_iterations:
                iterations_to_consider = min_experiment_iterations if trim_to_min else num_of_experiment_iterations
                experiments_dir[experiment_dir.name] = load_result_files(experiment_results_dir, iterations_to_consider)

        data_dict[strategy.value] = experiments_dir
    return data_dict


def load_result_files(results_dir: Path, num_of_experiment_iterations: int) -> List[BaseExperimentALIterationResult]:
    """Loads the result files from a given directory and returns them as a list."""
    results = []
    for i in range(num_of_experiment_iterations):
        p = Path(results_dir, f"al_iteration_{i}.pkl")
        with open(p, "rb") as pickle_file:
            results.append(pickle.load(pickle_file))
    return results


def create_dataframe_for_plotting(data_dict: Dict, num_total_samples: int = 17418) -> pd.DataFrame:
    """Creates a dataframe that can be used for plotting."""
    all_data = None
    for strategy in SelectionStrategy:
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


def results_over_time_plot(plotting_df: pd.DataFrame, time_value_to_use: str = "AL iteration", figure_filename: str = "results_over_iteration.png", result_value_to_use: str = "Macro AUC"):
    """
    Creates a results over time plot from the given data.

    Args:
        plotting_df (pd.DataFrame): The data that should be plotted.
        time_value_to_use (str): The column name that should be used for the time dimension.
        figure_filename (str): The name of the figure file. Is used to store the figure.
    """
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Results of different selection strategies over time")
    sns.lineplot(plotting_df, y=result_value_to_use, x=time_value_to_use, hue="Strategy", errorbar=("ci", 95))
    fig.savefig(figure_filename)


def get_plotting_name(strategy: SelectionStrategy) -> str:
    """Returns the name that should be used in plots."""
    if strategy == SelectionStrategy.RANDOM:
        return "Random"
    if strategy == SelectionStrategy.BADGE:
        return "BADGE"
    if strategy == SelectionStrategy.ENTROPY:
        return "Entropy"
    if strategy == SelectionStrategy.PLVI_CE_TOPK:
        return "PLVI-CE with top-k"
    if strategy == SelectionStrategy.PLVI_CE_KNN:
        return "PLVI-CE with clustering"
