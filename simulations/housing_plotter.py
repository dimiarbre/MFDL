import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotters import plot_housing_results
from scipy.stats import t
from utils import METHOD_COLORS, METHOD_DISPLAY_NAMES

# Map from the name stored in filenames to the name in the columns.
PARAM_MAP = {
    "_graph": "graph_name",
    "_nodes": "num_nodes",
    "_eps": "epsilon",
    "_reps": "num_passes",
    "_nbbatches": "nb_big_batches",
    "_mbps": "micro_batches_per_step",
    "_lr": "lr",
    "_seed": "dataloader_seed",
    "_mbsize": "micro_batch_size",
    "_steps": "total_steps",
}


def extract_params_from_filename(filename: str) -> Dict[str, str]:
    """
    Extracts parameter values from the filename using regular expressions.
    Example filename:
    simulation_graphego_nodes148_eps0.5_reps5_nbbatches16_mbps1_lr0.1_seed421_mbsize6_steps95_.csv
    Returns a dictionary of parameter names to values.
    """
    basename = os.path.basename(filename)
    pattern = r"(_graph[ A-Za-z0-9\(\)]+|_nodes\d+|_eps[\d\.]+|_reps\d+|_nbbatches\d+|_mbps\d+|_lr[\d\.]+|_seed\d+|_mbsize\d+|_steps\d+)"
    matches = re.findall(pattern, basename)
    params = {}
    float_params = {"epsilon", "lr"}
    int_params = {
        "num_nodes",
        "num_passes",
        "nb_big_batches",
        "micro_batches_per_step",
        "dataloader_seed",
        "micro_batch_size",
        "total_steps",
    }
    for match in matches:
        for key in PARAM_MAP:
            if match.startswith(key):
                param_name = PARAM_MAP[key]
                value = match[len(key) :]
                if param_name in float_params:
                    params[param_name] = float(value)
                elif param_name in int_params:
                    params[param_name] = int(value)
                else:
                    params[param_name] = value
                break
    return params


def load_housing_data(
    base_dir: str = "results/housing",
    param_filters: Optional[Dict[str, List]] = None,
    methods_to_remove=[],
    only_last_step=False,
) -> pd.DataFrame:
    """
    Loads dataframes from files matching the parameter filters.
    param_filters: dict of parameter name to list of allowed values (as strings)
    Returns concatenated dataframe of all matching files.
    """
    dfs = []
    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".csv"):
            continue
        try:
            params = extract_params_from_filename(fname)
        except ValueError:
            continue
        # Filter by param_filters
        if param_filters:
            match = True
            for k, allowed in param_filters.items():
                if k in params and str(params[k]) not in [str(v) for v in allowed]:
                    match = False
                    break
            if not match:
                continue
        df = pd.read_csv(os.path.join(base_dir, fname))
        for k, v in params.items():
            assert len(df[k].unique()) == 1
            assert df[k].iloc[0] == v

        # Optimization to reduce memory consumption.
        for method in methods_to_remove:
            df = df[df["method"] != method]
        if only_last_step:
            # For each dataloader_seed, select all rows with the maximum step
            max_steps = df.groupby("dataloader_seed")["step"].transform("max")
            df = df[df["step"] == max_steps].reset_index(drop=True)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {fname}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Empty if no matches


def plot_housing_results_with_ci(
    df: pd.DataFrame,
    loss_attr: str = "test_loss",
    debug: bool = True,
    log_scale: bool = False,
    min_max: bool = False,
    filename: str = "",
):
    """
    Plots the mean and confidence interval of loss_attr for each method as a function of step.
    """
    # Order methods by METHOD_COLORS, keeping only those present in df
    unique_methods = df["method"].unique()
    methods = [m for m in METHOD_COLORS if m in unique_methods]
    steps = sorted(df["step"].unique())
    nodes = sorted(df["node"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    all_results_df = pd.DataFrame({})

    for method in methods:
        method_df = df[df["method"] == method]
        # Group by step, aggregate over nodes for each seed and iteration
        grouped = (
            method_df.groupby(["step", "dataloader_seed"])[loss_attr]
            .mean()
            .reset_index()
        )
        # For each step, get mean and std over seeds
        means = []
        stds = []
        ci95s = []
        ci99s = []
        for step in steps:
            seed_means = grouped[grouped["step"] == step][loss_attr].values
            means.append(np.mean(seed_means))
            n = len(seed_means)
            stds.append(np.std(seed_means))
            sem = np.std(seed_means, ddof=1) / np.sqrt(n) if n > 1 else 0
            ci95 = sem * t.ppf((1 + 0.95) / 2, n - 1) if n > 1 else 0
            ci95s.append(ci95)
            ci99 = sem * t.ppf((1 + 0.99) / 2, n - 1) if n > 1 else 0
            ci99s.append(ci99)
        means_np = np.array(means)
        stds_np = np.array(stds)
        ci95s_np = np.array(ci95s)
        ci99s_np = np.array(ci99s)
        # Save means and stds for each method
        results_df = pd.DataFrame(
            {
                "step": steps,
                f"{method}_mean": means_np,
                f"{method}_std": stds_np,
                f"{method}_ci95": ci95s_np,
                f"{method}_ci99": ci99s_np,
                "method": method,
            }
        )

        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

        color = METHOD_COLORS[method]
        ax.plot(steps, means, label=method, color=color)
        ax.fill_between(
            steps, means_np - ci95s_np, means_np + ci95s_np, alpha=0.2, color=color
        )

    ymax_plot = 8  # Manual bound for nicer figures
    ymax = ax.get_ylim()[1]
    if ymax > ymax_plot:
        ax.set_ylim(top=ymax_plot)
    ymin = ax.get_ylim()[0]
    if ymin < 0:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Step", fontsize=18)
    ax.set_ylabel(loss_attr.replace("_", " ").title(), fontsize=18)
    if log_scale:
        ax.set_yscale("log")
    plt.tick_params(axis="both", which="major", labelsize=16)
    legend_ncols = 2 if len(methods) >= 4 else 1
    ax.legend(
        fontsize=16,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        ncols=legend_ncols,
    )
    plt.grid()
    plt.tight_layout()
    if filename == "":
        filename = f"housing_{loss_attr}_ci_plot"
    csv_path = f"results/housing_data/{filename}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    all_results_df.to_csv(csv_path)
    if debug:
        plt.show()
    else:
        plt.savefig(f"figures/housing/{filename}.pdf")


def main():
    # Only load files with eps=0.5 and seed=421
    filters: Dict[str, List] = {
        "graph_name": ["ego"],
        "num_passes": [20],
        "epsilon": [
            10.0,
            0.5,
            0.2,
            1.0,
            0.1,
            2.0,
        ],  # Remember to put floats here (1.0,...)
    }
    methods_to_remove = [
        "OPTIMAL_DL_MSG",
        "BSR_LOCAL",
        "BSR_BANDED_LOCAL",
        "OPTIMAL_LOCAL",
    ]

    df = load_housing_data(param_filters=filters, methods_to_remove=methods_to_remove)
    assert not df.empty, "Empty dataframe, check you used floats in epsilon"

    for epsilon in filters["epsilon"]:
        epsilon = float(epsilon)
        current_df = df[df["epsilon"] == epsilon]

        # Ensures we are on an unique setting in all the experiments
        for _, param in PARAM_MAP.items():
            if "seed" not in param:  # Allow seeding arguments.
                assert (
                    len(current_df[param].unique()) == 1
                ), f"Got multiple values for parameter {param}"

        # df = df[df["method"] != "LDP"]
        if epsilon < 0.5:  # Remove them from the plot as they don't converge.
            current_df = current_df[current_df["method"] != "ANTIPGD"]
            current_df = current_df[current_df["method"] != "BSR_BANDED_LOCAL"]

        # Rename on plots:
        current_df = current_df.copy()
        for method_source, method_display_name in METHOD_DISPLAY_NAMES.items():
            current_df["method"] = current_df["method"].replace(
                method_source, method_display_name
            )

        assert not current_df.empty, "Empty dataframe, consider relaxing filters."

        # Usage:
        plot_housing_results_with_ci(
            current_df,
            loss_attr="test_loss",  # Change to "train_loss" if needed
            debug=False,
            log_scale=False,
            min_max=False,
            filename=f"housing_test_loss_ci_plot_epsilon{epsilon}",
        )

    print("Finished plotting")


# Example usage:
if __name__ == "__main__":
    main()
