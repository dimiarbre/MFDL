import argparse
import os
import shutil
from typing import Literal

import factorizations_comparison
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
from scipy.stats import t
from utils import (
    METHOD_COLORS,
    METHOD_DISPLAY_NAMES,
    METHOD_MARKERS,
    GraphName,
    get_graph,
    graph_require_seed,
    profile_memory_usage,
)

GRAPH_NAMES: list[GraphName] = [
    # "expander",
    # "cycle",
    # "empty",
    "complete",
    "erdos",
    # "star",
    # "florentine",
    # "ego",
    # "peertube (connex component)",
    "regular",
]
# GRAPH_NAMES = ["expander", "complete"]

NB_NODES = [10 * i for i in range(1, 11)]
# NB_NODES = [10 * i for i in range(1, 21)]
# NB_NODES = [10, 20, 50]

# NB_REPETITIONS = [4, 10]
NB_REPETITIONS = [4]

PARTICIPATION_INTERVALS = [16]

# SEEDS = [421 + i for i in range(10)]
SEEDS = [421 + i for i in range(20)]


def generate_all_configurations(
    nb_repetitions, nb_nodes_list, graph_names, participation_intervals, seeds
) -> list[tuple[GraphName, int, int, int, int]]:
    params = []
    for nb_repetition in nb_repetitions:
        for participation_interval in participation_intervals:
            for graph_name in graph_names:
                current_graph_seeds = seeds
                # For non-random graphs, just run once, seeds won't have an impact
                if not graph_require_seed(graph_name):
                    current_graph_seeds = [seeds[0]]
                current_graph_nbnodes_list = nb_nodes_list
                # Florentine is a special case with only one configuration
                if graph_name == "florentine":
                    current_graph_nbnodes_list = [30]
                for nb_nodes in current_graph_nbnodes_list:
                    for seed in current_graph_seeds:
                        params.append(
                            (
                                graph_name,
                                nb_nodes,
                                nb_repetition,
                                seed,
                                participation_interval,
                            )
                        )
    # Order: nb_repetition < nb_nodes < graph_name < seed < participation_interval
    params = sorted(params, key=lambda x: (x[2], x[1], x[0], x[3], x[4]))
    return params


@profile_memory_usage
def compute_all_experiments(recompute: bool, verbose: bool = False) -> pd.DataFrame:

    data_path = "results/factorization/experiment.csv"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df = pd.DataFrame({})
    if not recompute:  # Try to load already existing dataframe
        try:
            df = pd.read_csv(data_path)
            # Clean up Unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        except FileNotFoundError:
            pass

    params = generate_all_configurations(
        nb_repetitions=NB_REPETITIONS,
        nb_nodes_list=NB_NODES,
        graph_names=GRAPH_NAMES,
        participation_intervals=PARTICIPATION_INTERVALS,
        seeds=SEEDS,
    )
    nb_expe = len(params)

    for i, (
        graph_name,
        nb_nodes,
        nb_repetition,
        seed,
        participation_interval,
    ) in enumerate(params):
        graph_name: GraphName
        prefix = f"[{i+1:0{len(str(nb_expe))}d}/{nb_expe}] - "

        G = get_graph(graph_name, nb_nodes, seed=seed)
        if nb_nodes != G.number_of_nodes():
            can_change: list[GraphName] = [
                "florentine",
                "ego",
                "peertube",
                "peertube (connex component)",
            ]
            assert (
                graph_name in can_change
            ), f"The number of node was changed on a non-florentine/ego/peertube graph: {graph_name}"
            # Florentine graph is a special case, with always 30 nodes
            nb_nodes = G.number_of_nodes()
        del G

        # Skip already_existing data.
        if (
            not df.empty
            and (
                (df.get("graph_name") == graph_name)
                & (df.get("nb_nodes") == nb_nodes)
                & (df.get("nb_repetition") == nb_repetition)
                & (df.get("participation_interval") == participation_interval)
                & (df.get("seed") == seed)
            ).any()
        ):
            print(f"{prefix}Skipping already computed configuration")
            continue
        print(
            f"{prefix}Running experiment with nb_repetition={nb_repetition}, nb_nodes={nb_nodes}, graph_name={graph_name}, seed={seed}, participation_interval={participation_interval}"
        )
        experiment_results = factorizations_comparison.run_experiment(
            graph_name=graph_name,
            nb_nodes=nb_nodes,
            nb_repetition=nb_repetition,
            participation_interval=participation_interval,
            seed=seed,
            verbose=verbose,
        )
        df = pd.concat([df, experiment_results])
        df.to_csv(
            data_path, index=False
        )  # Save as we go, so that we can recover from interruption.
    return df


def plot_one_figure(
    sub_df,
    label_marker_map,
    title,
    savepath,
    y_axis_data="loss_optimization",
    y_axis_name="Optimization loss",
):
    plt.figure(figsize=(10, 6))

    # Group by all experiment parameters except seed, aggregate over seeds
    group_cols = [
        "factorization_name",
        "nb_nodes",
        "nb_repetition",
        "participation_interval",
    ]

    # Check that each group is unique (i.e., only one row per group)
    group_check = sub_df.groupby(group_cols + ["seed"]).size().reset_index(name="count")
    if not (group_check["count"] == 1).all():
        duplicate_groups = group_check[group_check["count"] > 1]
        print("Duplicate group(s) found:")
        print(duplicate_groups.iloc[0])
        raise ValueError(
            "Some (nb_nodes, nb_repetition, participation_interval, seed) groups have multiple entries."
        )
    grouped = sub_df  # No aggregation needed since each group is unique
    summary = (
        grouped.groupby(group_cols)[y_axis_data]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / summary["count"] ** 0.5
    summary["ci95"] = summary["sem"] * t.ppf(0.975, summary["count"] - 1)

    # Plot with error bars (confidence intervals)
    # Order factorization_name according to METHOD_COLORS.keys()
    factorization_order = list(METHOD_COLORS.keys())
    # Get unique combinations of factorization_name, nb_repetition, participation_interval
    configs = summary.groupby(
        ["factorization_name", "nb_repetition", "participation_interval"]
    )
    # Sort the configs by factorization_name order, then nb_repetition, then participation_interval
    sorted_keys = sorted(
        configs.groups.keys(),
        key=lambda x: (
            (
                factorization_order.index(x[0])
                if x[0] in factorization_order
                else len(factorization_order)
            ),
            x[1],
            x[2],
        ),
    )
    for key in sorted_keys:
        config = configs.get_group(key)
        factorization_name = config["factorization_name"].iloc[0]
        label = f"{factorization_name}"
        # Assign marker if not already assigned
        color = METHOD_COLORS[factorization_name]
        marker = METHOD_MARKERS[factorization_name]
        plt.errorbar(
            config["nb_nodes"],
            config["mean"],
            yerr=config["ci95"],
            label=label,
            marker=marker,
            capsize=4,
            color=color,
        )

    # plt.title(title, fontsize=26)
    plt.xlabel("n", fontsize=27)
    plt.ylabel(y_axis_name, fontsize=27)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.legend(
        fontsize=20,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)


def generate_plots(df, hide_figures: bool = False, only_optimal: bool = False):
    # Ensure necessary columns exist
    required_columns = [
        "graph_name",
        "factorization_name",
        "nb_nodes",
        "nb_repetition",
        "participation_interval",
        "seed",
        "loss_optimization",
        "loss_message",
    ]
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return

    df["mean_loss_optimization"] = df["loss_optimization"] / df["nb_nodes"]

    # Additional filters
    # TODO: remove
    df = df[df["nb_repetition"] == 4]
    if only_optimal:
        df = df[
            df["factorization_name"].isin(
                [
                    "OPTIMAL_LOCAL",
                    "OPTIMAL_DL_MSG",
                    "OPTIMAL_DL_POSTAVG",
                    "LDP",
                    # "BSR_LOCAL",
                ]
            )
        ]

    df = df.copy()
    # Rename on plots:
    for method_source, method_display_name in METHOD_DISPLAY_NAMES.items():
        df["factorization_name"] = df["factorization_name"].replace(
            method_source, method_display_name
        )

    # Set up plotting style
    sns.set_theme(style="whitegrid")
    plt.style.use("science")

    # Define a list of markers to cycle through

    # Create a mapping from config label to marker to keep it consistent
    label_marker_map = {}

    # Plot for each graph type
    for graph_name in df["graph_name"].unique():
        sub_df = df[df["graph_name"] == graph_name]
        title = f"Optimization Loss vs Number of Nodes ({graph_name})"
        csvpath = f"figures/factorization_simulation/{"only_optimal_" if only_optimal else ""}optim_loss_vs_nbnodes_{graph_name}.pdf"
        plot_one_figure(
            sub_df=sub_df,
            label_marker_map=label_marker_map,
            title=title,
            savepath=csvpath,
            y_axis_name=r"$\mathcal{L}_{\mathrm{opti}}$",
        )

        # title = f"Mean Optimization Loss vs Number of Nodes ({graph_name})"
        # csvpath = f"figures/factorization_simulation/{"only_optimal_" if only_optimal else ""}mean_optim_loss_vs_nbnodes_{graph_name}.pdf"
        # plot_one_figure(
        #     sub_df=sub_df,
        #     label_marker_map=label_marker_map,
        #     title=title,
        #     savepath=csvpath,
        #     y_axis_name="Mean optimization loss",
        #     y_axis_data="mean_loss_optimization",
        # )

    if not hide_figures:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run factorization experiments.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all experiments, even if results exist.",
    )
    parser.add_argument(
        "--hidefigs",
        action="store_true",
        help="Do not show figures after the experiment, only store them.Useful when estimating timing or running on a server.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose prints to check progress inside steps.",
    )
    args = parser.parse_args()

    if args.recompute:
        cache_path = "cache/"
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

    df = compute_all_experiments(args.recompute, verbose=args.verbose)

    # TODO: Save this df

    # Remove unneeded plot data for the paper.
    methods_to_remove = [
        "OPTIMAL_DL_MSG",
        "BSR_LOCAL",
        "BSR_BANDED_LOCAL",
        "OPTIMAL_LOCAL",
    ]
    for method in methods_to_remove:
        df = df[df["factorization_name"] != method]
    df = df.copy()

    generate_plots(df, args.hidefigs, only_optimal=False)
    generate_plots(df, args.hidefigs, only_optimal=True)


if __name__ == "__main__":
    main()
