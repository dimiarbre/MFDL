import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from housing_plotter import *
from plotters import plot_housing_results
from utils import METHOD_COLORS, METHOD_DISPLAY_NAMES


def plot_final_test_loss_vs_epsilon(df, filters, graph_name):
    plt.figure(figsize=(8, 5))

    means = []
    cis = []
    epsilons = []
    for epsilon in filters["epsilon"]:
        epsilon = float(epsilon)
        current_df = df[(df["graph_name"] == graph_name) & (df["epsilon"] == epsilon)]
        assert (
            not current_df.empty
        ), f"Empty dataframe for graph {graph_name}, epsilon {epsilon}"

        # For each method, compute mean and CI of final test_loss across seeds
        methods = [m for m in METHOD_COLORS if m in current_df["method"].unique()]
        for method in methods:
            method_df = current_df[current_df["method"] == method]
            # Get the last step for each seed
            last_steps = method_df.groupby("dataloader_seed")["step"].max().values
            final_losses = []
            for seed, last_step in zip(
                method_df["dataloader_seed"].unique(), last_steps
            ):
                loss = method_df[
                    (method_df["dataloader_seed"] == seed)
                    & (method_df["step"] == last_step)
                ]["test_loss"].values
                if len(loss) > 0:
                    final_losses.append(loss[0])
            if final_losses:
                mean_loss = np.mean(final_losses)
                std_loss = np.std(final_losses)
                n = len(final_losses)
                ci = 1.96 * std_loss / np.sqrt(n) if n > 1 else 0
                means.append((epsilon, method, mean_loss))
                cis.append((epsilon, method, ci))
                epsilons.append(epsilon)

    # Plot for each method
    # Order methods by METHOD_COLORS keys
    ordered_methods = [m for m in METHOD_COLORS if m in set([m for _, m, _ in means])]
    for method in ordered_methods:
        # Compute sigma = 1/epsilon and sort by sigma
        method_data = [
            (1.0 / eps, mean, ci)
            for eps, m, mean in means
            if m == method
            for _, _, ci in [
                next((c for c in cis if c[0] == eps and c[1] == m), (eps, m, 0))
            ]
        ]
        method_data.sort(key=lambda x: x[0])  # sort by sigma

        sigmas = [sigma for sigma, _, _ in method_data]
        method_means = [mean for _, mean, _ in method_data]
        method_cis = [ci for _, _, ci in method_data]
        color = METHOD_COLORS.get(method, None)
        plt.errorbar(
            sigmas,
            method_means,
            yerr=method_cis,
            label=f"{method}",
            color=color,
            marker="o",
            capsize=4,
        )

    plt.xlabel("Sigma ($\\sigma = 1/\\epsilon$)", fontsize=18)
    plt.ylabel("Final Test Loss", fontsize=18)
    plt.legend(fontsize=14, frameon=True, facecolor="white", edgecolor="black")
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    figpath = (
        f"figures/housing/final_test_loss_vs_sigma_multigraphs_graph{graph_name}.pdf"
    )
    plt.savefig(figpath)
    print(f"Saved fig to {figpath}")
    # plt.show()


def main():
    graph_names = [
        "peertube (connex component)",
        "florentine",
        "ego",
    ]
    for graph_name in graph_names:
        filters: Dict[str, List] = {
            "graph_name": [graph_name],
            "num_passes": [20],
            "epsilon": [
                10.0,
                0.5,
                0.2,
                1.0,
                0.1,
                2.0,
                5.0,
            ],  # Remember to put floats here (1.0,...)
        }
        methods_to_remove = ["OPTIMAL_DL_MSG", "BSR_LOCAL", "ANTIPGD"]
        df = load_housing_data(
            param_filters=filters,
            methods_to_remove=methods_to_remove,
            only_last_step=True,
        )
        assert not df.empty, "Empty dataframe, check you used floats in epsilon"

        # Rename on plots:
        for method_source, method_display_name in METHOD_DISPLAY_NAMES.items():
            df["method"] = df["method"].replace(method_source, method_display_name)

        current_df = df[df["graph_name"] == graph_name]
        plot_final_test_loss_vs_epsilon(
            current_df, filters=filters, graph_name=graph_name
        )

    print("Finished plotting")
    plt.show()


# Example usage:
if __name__ == "__main__":
    main()
