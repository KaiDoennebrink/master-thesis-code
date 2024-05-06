from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_init_strategy_results(path_to_result_file: Path = Path("./experiments/init_strategy/results/results.csv")):
    # prepare the data
    df = pd.read_csv(path_to_result_file)
    plotting_df = df[["auc", "experiment_name"]].copy()
    plotting_df.rename(columns={"auc": "Macro AUC", "experiment_name": "Initialization strategy"}, inplace=True)

    def _rename_column_values(s: str):
        if s == "random":
            return "Random"
        if s == "pt4al_one":
            return "PT4AL\n(k=1)"
        if s == "pt4al_ten":
            return "PT4AL\n(k=10)"
        if s == "representation_cluster_pretext":
            return "RepClust\n(pretext)"
        if s == "representation_cluster_tl":
            return "RepClust\n(CPSC 2018)"

    plotting_df["Initialization strategy"] = plotting_df["Initialization strategy"].apply(_rename_column_values)

    # plot everything
    plt.figure(figsize=(7, 4))
    sns.set_style("whitegrid")
    sns.boxplot(plotting_df, x="Initialization strategy", y="Macro AUC")

    plt.tight_layout()
    plt.savefig("./plots/results_init_strategy.png", dpi=600)
