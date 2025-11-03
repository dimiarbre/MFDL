import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epsilon_computation import solve_epsilon
from housing_plotter import *
from plotters import plot_housing_results
from utils import METHOD_COLORS, METHOD_DISPLAY_NAMES, METHOD_LINESTYLES


def plot_final_test_loss_vs_epsilon(
    df, filters, graph_name, dataset_name, y_axis="test_loss"
):
    plt.figure(figsize=(8, 5))

    means = []
    cis = []
    mus = []
    unnoised_baseline_mean = None
    unnoised_baseline_ci = None
    baseline_points = []
    ylim_max = 4
    ylim_min = 5e-1
    if "acc" in y_axis:
        ylim_max = 1
        ylim_min = 0

    for mu in filters["mu"]:
        # Here, mu means 1/sigma (hence mu-GDP guarantees).
        mu = float(mu)
        current_df = df[(df["graph_name"] == graph_name) & (df["mu"] == mu)]
        assert not current_df.empty, f"Empty dataframe for graph {graph_name}, mu {mu}"

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
                ][y_axis].values
                if len(loss) > 0:
                    final_losses.append(loss[0])
            if final_losses:
                mean_loss = np.mean(final_losses)
                std_loss = np.std(final_losses)
                n = len(final_losses)
                # 95% confidence interval
                ci = 1.96 * std_loss / np.sqrt(n) if n > 1 else 0
                if method == "Unnoised baseline":
                    # Store only once (should be same for all mus)
                    if unnoised_baseline_mean is None:
                        unnoised_baseline_mean = mean_loss
                        unnoised_baseline_ci = ci
                    baseline_points.append((mu, mean_loss))
                else:
                    means.append((mu, method, mean_loss))
                    cis.append((mu, method, ci))
                    mus.append(mu)

    ordered_methods = [m for m in METHOD_COLORS if m in set([m for _, m, _ in means])]
    for method in ordered_methods:
        if method == "Unnoised baseline":
            continue
        delta = 1e-6
        method_data = [
            (solve_epsilon(mu=mu, delta_target=delta), mean, ci)
            for mu, m, mean in means
            if m == method
            for _, _, ci in [
                next((c for c in cis if c[0] == mu and c[1] == m), (mu, m, 0))
            ]
        ]
        method_data.sort(key=lambda x: x[0])  # sort by epsilon

        epsilons_plot = [eps for eps, _, _ in method_data]
        method_means = [mean for _, mean, _ in method_data]
        method_cis = [ci for _, _, ci in method_data]
        color = METHOD_COLORS.get(method, None)
        plt.errorbar(
            epsilons_plot,
            method_means,
            yerr=method_cis,
            label=f"{method}",
            color=color,
            marker="o",
            capsize=4,
            linestyle=METHOD_LINESTYLES.get(method, "-"),
        )
        # Print method and points
        print(f"{method}: {list(zip(epsilons_plot, method_means))}")

    # plt.yscale("log")
    plt.xscale("log")
    plt.tight_layout()
    # Plot Unnoised baseline as a horizontal line
    if unnoised_baseline_mean is not None:
        color = METHOD_COLORS.get("Unnoised baseline", "gray")
        # Use infinite bounds so the line always spans the axis, regardless of xlim
        plt.axhline(
            unnoised_baseline_mean,
            label="Unnoised baseline",
            color=color,
            linestyle="--",
            linewidth=2,
        )
        # add CI as shaded area
        if unnoised_baseline_ci is not None and unnoised_baseline_ci > 0:
            plt.axhspan(
                unnoised_baseline_mean - unnoised_baseline_ci,
                unnoised_baseline_mean + unnoised_baseline_ci,
                color=color,
                alpha=0.2,
            )
        # Print baseline points
        print(f"Unnoised baseline: {baseline_points}")

    plt.xlabel("$\\epsilon$", fontsize=24)
    plt.ylabel(f"Final {y_axis}", fontsize=24)
    plt.legend(fontsize=15, frameon=True, facecolor="white", edgecolor="black")
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.grid()
    ylims = plt.ylim()
    if ylims[0] < ylim_min:
        plt.ylim(bottom=ylim_min)
    if ylims[1] > ylim_max:
        plt.ylim(top=ylim_max)
    plt.tight_layout()
    figpath = f"figures/{dataset_name}/final_{y_axis}_vs_epsilon_graph{graph_name}.pdf"
    plt.savefig(figpath)
    print(f"Saved fig to {figpath}")
    # plt.show()


def main():
    dataset_name = "housing"
    y_axis = "test_loss"
    graph_names = [
        "florentine",
        "peertube (connex component)",
        "ego",
        "chain",
        "hypercube",
    ]
    for graph_name in graph_names:
        filters: Dict[str, List] = {
            "graph_name": [graph_name],
            "num_passes": [20],
            "mu": [
                0.1,
                0.2,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
            ],  # Remember to put floats here (1.0,...)
        }
        methods_to_remove = [
            "OPTIMAL_DL_MSG",
            "BSR_LOCAL",
            "BSR_BANDED_LOCAL",
            # "OPTIMAL_LOCAL",
        ]
        if "peertube" in graph_name:
            methods_to_remove.append("ANTIPGD")
        df = load_decentralized_simulation_data(
            base_dir=f"results/{dataset_name}",
            param_filters=filters,
            methods_to_remove=methods_to_remove,
            only_last_step=True,
        )
        assert (
            not df.empty
        ), "Empty dataframe, check you used floats in mu and lr filters"

        # Rename on plots:
        for method_source, method_display_name in METHOD_DISPLAY_NAMES.items():
            df["method"] = df["method"].replace(method_source, method_display_name)

        current_df = df[df["graph_name"] == graph_name]
        plot_final_test_loss_vs_epsilon(
            current_df,
            filters=filters,
            graph_name=graph_name,
            dataset_name=dataset_name,
            y_axis=y_axis,
        )

    print("Finished plotting")
    plt.show()


# Example usage:
if __name__ == "__main__":
    plt.rcParams["axes.linewidth"] = 2.0  # Axis lines
    plt.rcParams["grid.linewidth"] = 1.5  # Grid lines
    main()
