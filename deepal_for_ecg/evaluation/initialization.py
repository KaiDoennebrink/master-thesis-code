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
            return "PT4AL (k=1)"
        if s == "pt4al_ten":
            return "PT4AL (k=10)"
        if s == "representation_cluster_pretext":
            return "Representation\nclustering with\npretext model"
        if s == "representation_cluster_tl":
            return "Representation\nclustering with\nTL model"

    plotting_df["Initialization strategy"] = plotting_df["Initialization strategy"].apply(_rename_column_values)

    # plot everything
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.boxplot(plotting_df, x="Initialization strategy", y="Macro AUC")
    # sns.scatterplot(df, x="experiment_name", y="auc", alpha=0.6, color="black")
    plt.tight_layout()
    # plt.show()
    plt.savefig("./result_init_strategy.png")
