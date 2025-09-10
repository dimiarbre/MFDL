import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotters import plot_housing_results
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
    pattern = r"(_graph[A-Za-z0-9]+|_nodes\d+|_eps[\d\.]+|_reps\d+|_nbbatches\d+|_mbps\d+|_lr[\d\.]+|_seed\d+|_mbsize\d+|_steps\d+)"
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
) -> pd.DataFrame:
    """
    Loads dataframes from files matching the parameter filters.
    param_filters: dict of parameter name to list of allowed values (as strings)
    Returns concatenated dataframe of all matching files.
    """
    dfs = []
    for fname in os.listdir(base_dir):
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
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {fname}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Empty if no matches


def plot_housing_results_with_ci(
    df: pd.DataFrame,
    loss_attr: str = "test_loss",
    details: str = "",
    experiment_properties: str = "",
    debug: bool = True,
    log_scale: bool = False,
    min_max: bool = False,
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
        for step in steps:
            seed_means = grouped[grouped["step"] == step][loss_attr].values
            means.append(np.mean(seed_means))
            stds.append(np.std(seed_means))
        means_np = np.array(means)
        stds_np = np.array(stds)
        # Save means and stds for each method
        results_df = pd.DataFrame(
            {
                "step": steps,
                f"{method}_mean": means_np,
                f"{method}_std": stds_np,
                "method": method,
            }
        )

        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

        color = METHOD_COLORS[method]
        ax.plot(steps, means, label=method, color=color)
        ax.fill_between(
            steps, means_np - stds_np, means_np + stds_np, alpha=0.2, color=color
        )

    # Limit y-axis upper bound to 8 if needed
    ymax = ax.get_ylim()[1]
    if ymax > 8:
        ax.set_ylim(top=8)
    ymin = ax.get_ylim()[0]
    if ymin < 0:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Step", fontsize=18)
    ax.set_ylabel(loss_attr.replace("_", " ").title(), fontsize=18)
    ax.set_title(
        f"{loss_attr.replace('_', ' ').title()} vs Step {details} {experiment_properties}",
        fontsize=20,
    )
    if log_scale:
        ax.set_yscale("log")
    plt.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
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
        "epsilon": [10.0],  # Remember to put floats here (1.0,...)
    }
    df = load_housing_data(param_filters=filters)
    assert not df.empty, "Empty dataframe, check you used floats in epsilon"

    # Ensures we are on an unique setting in all the experiments
    for _, param in PARAM_MAP.items():
        if "seed" not in param:  # Allow seeding arguments.
            assert (
                len(df[param].unique()) == 1
            ), f"Got multiple values for parameter {param}"

    # df = df[df["method"] != "LDP"]
    # df = df[df["method"] != "ANTIPGD"]
    # df = df[df["method"] != "BSR_LOCAL"]
    df = df[df["method"] != "OPTIMAL_DL_MSG"]
    # df = df[df["method"] != "OPTIMAL_LOCAL"]

    # Rename on plots:
    for method_source, method_display_name in METHOD_DISPLAY_NAMES.items():
        df["method"] = df["method"].replace(method_source, method_display_name)

    assert not df.empty, "Empty dataframe, consider relaxing filters."

    # Usage:
    plot_housing_results_with_ci(
        df,
        loss_attr="test_loss",  # Change to "train_loss" if needed
        details="",
        experiment_properties="",
        debug=False,
        log_scale=False,
        min_max=False,
    )

    print("Finished plotting")


# Example usage:
if __name__ == "__main__":
    main()
