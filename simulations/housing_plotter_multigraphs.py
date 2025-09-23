import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epsilon_computation import solve_epsilon
from housing_plotter import *
from plotters import plot_housing_results
from utils import METHOD_COLORS, METHOD_DISPLAY_NAMES


def plot_final_test_loss_vs_epsilon(df, filters, graph_name):
    plt.figure(figsize=(8, 5))

    means = []
    cis = []
    epsilons = []
    unnoised_baseline_mean = None
    unnoised_baseline_ci = None
    ylim_max = 4
    ylim_min = 5e-1

    for epsilon in filters["epsilon"]:
        # Here, epsilon means 1/sigma (so mu, for mu-GDP guarantees), and is not a proper (epsilon, delta)-DP guarantee.
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
                # 95% confidence interval
                ci = 1.96 * std_loss / np.sqrt(n) if n > 1 else 0
                if method == "Unnoised baseline":
                    # Store only once (should be same for all epsilons)
                    if unnoised_baseline_mean is None:
                        unnoised_baseline_mean = mean_loss
                        unnoised_baseline_ci = ci
                    # else:
                    #     assert np.isclose(
                    #         mean_loss, unnoised_baseline_mean
                    #     ), f"{mean_loss} vs {unnoised_baseline_mean}"
                else:
                    means.append((epsilon, method, mean_loss))
                    cis.append((epsilon, method, ci))
                    epsilons.append(epsilon)

    # Plot for each method except Unnoised baseline
    ordered_methods = [m for m in METHOD_COLORS if m in set([m for _, m, _ in means])]
    for method in ordered_methods:
        if method == "Unnoised baseline":
            continue
        delta = 1e-6
        method_data = [
            (solve_epsilon(mu=eps, delta_target=delta), mean, ci)
            for eps, m, mean in means
            if m == method
            for _, _, ci in [
                next((c for c in cis if c[0] == eps and c[1] == m), (eps, m, 0))
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
        )

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

    plt.xlabel("$\\epsilon$", fontsize=24)
    plt.ylabel("Final Test Loss", fontsize=24)
    plt.legend(fontsize=15, frameon=True, facecolor="white", edgecolor="black")
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.grid()
    ylims = plt.ylim()
    if ylims[0] < ylim_min:
        plt.ylim(bottom=ylim_min)
    if ylims[1] > ylim_max:
        plt.ylim(top=ylim_max)
    plt.tight_layout()
    figpath = (
        f"figures/housing/final_test_loss_vs_sigma_multigraphs_graph{graph_name}.pdf"
    )
    plt.savefig(figpath)
    print(f"Saved fig to {figpath}")
    # plt.show()


def main():
    graph_names = [
        "florentine",
        "peertube (connex component)",
        "ego",
    ]
    for graph_name in graph_names:
        filters: Dict[str, List] = {
            "graph_name": [graph_name],
            "num_passes": [20],
            "epsilon": [
                # 10.0,
                0.5,
                0.2,
                1.0,
                0.1,
                2.0,
                5.0,
            ],  # Remember to put floats here (1.0,...)
        }
        methods_to_remove = [
            "OPTIMAL_DL_MSG",
            "BSR_LOCAL",
            "BSR_BANDED_LOCAL",
            "OPTIMAL_LOCAL",
        ]
        if "peertube" in graph_name:
            methods_to_remove.append("ANTIPGD")
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
