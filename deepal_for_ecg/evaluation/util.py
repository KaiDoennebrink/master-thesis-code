import pickle
from pathlib import Path
from typing import List, Dict

from deepal_for_ecg.experiments.base import BaseExperimentALIterationResult


def collect_experiment_runs_data(experiment_base_dir: Path, min_experiment_iterations: int = 21, trim_to_min: bool = True) -> Dict[str, List[BaseExperimentALIterationResult]]:
    experiment_runs_data = dict()

    for experiment_run_dir in experiment_base_dir.iterdir():
        experiment_results_dir = Path(experiment_run_dir, "results")
        al_iteration_result_paths = get_all_al_iteration_result_pickle_file_paths(experiment_results_dir)

        # check whether the experiment has enough iterations
        num_of_experiment_iterations = len(al_iteration_result_paths)
        if num_of_experiment_iterations >= min_experiment_iterations:
            iterations_to_consider = min_experiment_iterations if trim_to_min else num_of_experiment_iterations
            experiment_runs_data[experiment_run_dir.name] = load_result_files(experiment_results_dir, iterations_to_consider)

    return experiment_runs_data


def get_all_al_iteration_result_pickle_file_paths(base_dir: Path) -> List[Path]:
    al_iteration_result_paths = []
    for p in base_dir.iterdir():
        if p.match("al_iteration_[0-9]*.pkl"):
            al_iteration_result_paths.append(p)
    return al_iteration_result_paths


def load_result_files(results_dir: Path, num_of_experiment_iterations: int) -> List[BaseExperimentALIterationResult]:
    """Loads the result files from a given directory and returns them as a list."""
    results = []
    for i in range(num_of_experiment_iterations):
        p = Path(results_dir, f"al_iteration_{i}.pkl")
        with open(p, "rb") as pickle_file:
            results.append(pickle.load(pickle_file))
    return results
