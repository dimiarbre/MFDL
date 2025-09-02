import argparse

import factorizations_comparison
import pandas as pd

GRAPH_NAMES = ["expander", "cycle", "empty", "complete"]
# GRAPH_NAMES = ["expander", "complete"]

NB_NODES = [10 * i for i in range(1, 11)]
# NB_NODES = [10, 20, 50]

NB_REPETITIONS = [4, 10]
# NB_REPETITIONS = [4]

PARTICIPATION_INTERVALS = [16]

SEEDS = [421]


def main():
    args = parse()
    recompute = args.recompute

    data_path = "results/factorization/experiment.csv"
    df = pd.DataFrame({})
    if not recompute:  # Try to load already existing dataframe
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            pass

    params = [
        (graph_name, nb_nodes, nb_repetition, participation_interval, seed)
        for graph_name in GRAPH_NAMES
        for nb_repetition in NB_REPETITIONS
        for nb_nodes in NB_NODES
        for participation_interval in PARTICIPATION_INTERVALS
        for seed in SEEDS
    ]
    nb_expe = len(params)

    for i, (
        graph_name,
        nb_nodes,
        nb_repetition,
        participation_interval,
        seed,
    ) in enumerate(params):
        if not recompute:
            # Skip already_existing data.
            if (
                (df.get("graph_name") == graph_name)
                & (df.get("nb_nodes") == nb_nodes)
                & (df.get("nb_repetition") == nb_repetition)
                & (df.get("participation_interval") == participation_interval)
                & (df.get("seed") == seed)
            ).any():
                print(f"Skipping already computed configuration")
                continue
        print(
            f"[{i:0{len(str(nb_expe))}d}/{nb_expe}] Running experiment with graph_name={graph_name}, nb_nodes={nb_nodes}, nb_repetition={nb_repetition}, participation_interval={participation_interval}, seed={seed}"
        )
        experiment_results = factorizations_comparison.run_experiment(
            graph_name=graph_name,
            nb_nodes=nb_nodes,
            nb_repetition=nb_repetition,
            participation_interval=participation_interval,
            seed=seed,
        )
        df = pd.concat([df, experiment_results])
        df.to_csv(data_path)  # Save as we go, so that we can recover from interruption.


def parse():
    parser = argparse.ArgumentParser(description="Run factorization experiments.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all experiments, even if results exist.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
