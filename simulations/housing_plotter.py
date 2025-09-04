import os
import re
from typing import Dict, List, Optional

import pandas as pd
from plotters import plot_housing_results


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
    param_map = {
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
        for key in param_map:
            if match.startswith(key):
                param_name = param_map[key]
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
            assert df[k].unique()
            assert df[k].iloc[0] == v
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {fname}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Empty if no matches


# Example usage:
if __name__ == "__main__":
    # Only load files with eps=0.5 and seed=421
    filters: Dict[str, List] = {
        "graph_name": ["ego"],
        "num_passes": [5],
        "epsilon": [0.5],
    }
    df = load_housing_data(param_filters=filters)
    df = df[df["method"] != "ANTIPGD"]
    df = df[df["method"] != "BSR_LOCAL"]

    test_loss_dict = {
        method: group.sort_values(["step", "node"])["test_loss"].values.reshape(
            -1, len(group["node"].unique())
        )
        for method, group in df.groupby("method")
    }
    num_steps = df["total_steps"].iloc[0]

    plot_housing_results(
        all_test_losses=test_loss_dict,
        num_steps=num_steps,
        details="",
        experiment_properties="",
        debug=True,
    )
