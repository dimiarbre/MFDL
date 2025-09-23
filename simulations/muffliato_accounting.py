import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scienceplots
import utils
import workloads_generator
from matplotlib.lines import Line2D
from utils import GRAPH_RENAME, GraphName


def epsilon_upper_bound(
    G: nx.Graph, u: int, v: int, T: int, delta_phi: float, alpha: float, sigma: float
) -> float:
    """
    Compute the upper bound eps^T_{u->v}(alpha) following theorem 11 in https://arxiv.org/pdf/2206.05091. This is the original paper's CORRECTED formula (different from the NeurIPS paper).
    This code assumes a fixed communication graph G.

    We also clip the output to be the one of LDP when it is not, as recommanded after discussion with the original authors.
    """
    W = utils.get_communication_matrix(G)
    n = W.shape[0]
    assert W.shape[1] == n, "W must be square"
    assert 0 <= u < n and 0 <= v < n, "u and v must be valid node indices"

    coeff = ((delta_phi**2) * alpha) / (2.0 * (sigma**2))
    total = 0.0

    # Formula in the paper:
    # for t in range(1, T + 1):
    #     for j in range(t, T + 1):
    #         product = np.linalg.matrix_power(W, j - t)
    #         norms = np.linalg.norm(product, axis=0) ** 2
    #         for w in G.neighbors(v):
    #             total += product[u, w] / norms[w]

    # Optimized via a change of variable (s = j-t) to reduce the number of calls to matrix_power
    # And to remove a loop.
    for s in range(T):
        product = np.linalg.matrix_power(W, s)
        norms = np.linalg.norm(product, axis=0) ** 2
        for w in G.neighbors(v):
            total += product[u, w] / norms[w] * (T - s)

    if total > T:  # Clip to LDP bound: this is composition theorem.
        total = T
    epsilon = coeff * total
    return epsilon


def compute_muffliato_privacy_loss(
    communication_matrix: np.ndarray,
    nb_steps: int,
    attacker: int,
    participation_interval: int,
) -> list[float]:
    """Compute the PNDP privacy loss of the Muffliato algorithm (k=1) using our paper's accounting. This assumes a cyclic (participation_interval,nb_steps)-participation. Note participation_interval should be 1 to compare to the original paper, as they consider user-level privacy.

    Args:
        communication_matrix (np.ndarray): Communication graph as a np matrix
        nb_steps (int): Number of iterations
        attacker (int): Id of the attacker node, will compute eps(u->attacker), for all node u.
        participation_interval (int): Interval between two succinct participation (for batch-level privacy).

    Returns:
        list[float]: List containing eps(u->attacker), for all node u, using our accounting.
    """
    nb_nodes = len(communication_matrix)
    # Messages observed by an attacker
    LDP_workload = workloads_generator.build_DL_workload(
        matrix=communication_matrix, nb_steps=nb_steps, initial_power=0
    )

    projection_workload = workloads_generator.build_projection_workload(
        communication_matrix=communication_matrix,
        attacker_node=attacker,
        nb_steps=nb_steps,
    )

    # TODO: Simplify the return of projection matrix
    S = np.argmax(projection_workload, axis=1)
    # We project like this for efficiency. This is P @ W.
    PNDP_workload = LDP_workload[S, :]

    # This is the way to do it, but can be very slow. Optimization is done below for the Muffliato specific case.
    # sens = workloads_generator.compute_sensitivity_rectangularworkload(
    #     PNDP_workload,
    #     np.identity(nb_steps * nb_nodes),
    #     participation_interval=participation_interval,
    #     nb_steps=nb_steps,
    # )
    B = PNDP_workload
    B_old = B.copy()
    # Create the "additional knowledge" workload.
    for t in range(nb_steps):
        line = np.zeros(nb_nodes * nb_steps)
        line[nb_nodes * t + attacker] = 1
        B = np.vstack([B, line])

    # Main method (slow)
    PW_proj = np.linalg.pinv(B) @ B

    # Method 2: use SVD decomposition
    # We compute PW_proj = np.linalg.pinv(B) @ B
    # U, S, Vt = np.linalg.svd(B, full_matrices=False)
    # tol = max(B.shape) * S[0] * np.finfo(float).eps
    # r = (S > tol).sum()
    # PW_proj = Vt[:r].T @ Vt[:r]

    # The below does not really hold when considering additional lines of information for B like above...

    # # Let B = P @ W.
    # # For Muffliato, the sensitivity is B^+ @ B (since C = Identity)
    # # W is lower triangular of full rank, thus B is of maximal rank (P being a projection).
    # # Thus, B^+ @ B = B.T @ (B @ B.T)^{-1} @ B
    # # Now, we know (B @ B.T)^{-1} @ B is the solution to the linear system (B @ B.T) X = B
    # # Thus, we solve and get X = (B @ B.T)^{-1} @ B, and just need B.T @ X

    # X = np.linalg.solve(B_old @ B_old.T, B_old)  # shape (m, n)
    # PW_proj_old = B_old.T @ X
    PW_proj_old = np.linalg.pinv(B_old) @ B_old

    print("Norm of PW_proj_old - PW_proj:", np.linalg.norm(PW_proj_old - PW_proj))

    # TODO: compute the sensitivity (max of sum... cf paper).
    sensitivities = []
    for node in range(nb_nodes):
        if node == attacker:
            sensitivities.append(np.inf)
        else:
            sensitivities.append(
                workloads_generator.compute_cyclic_repetitions_1node(
                    X=PW_proj,
                    participation_interval=participation_interval,
                    nb_steps=nb_steps,
                    nb_nodes=nb_nodes,
                    node=node,
                )
            )
            # sensitivity_old = workloads_generator.compute_cyclic_repetitions_1node(
            #     X=PW_proj_old,
            #     participation_interval=participation_interval,
            #     nb_steps=nb_steps,
            #     nb_nodes=nb_nodes,
            #     node=node,
            # )
            # sensitivitiy_new = sensitivities[-1]
            # print(
            #     f"[{sensitivity_old>=sensitivitiy_new}]Old - new: {sensitivity_old - sensitivitiy_new}."
            # )

    # sens = workloads_generator.compute_cyclic_repetitions(
    #     PW_proj,
    #     participation_interval=participation_interval,
    #     nb_steps=nb_steps,
    #     nb_nodes=nb_nodes,
    # )

    return sensitivities


def privacy_loss_by_distance(
    graph: nx.Graph,
    attacker,
    participation_interval,
    nb_repetitions,
    alpha: float,
):
    """
    Computes privacy loss as a function of shortest path length from the attacker node.
    Returns a dict: {distance: [privacy_losses]}, plus arrays for best, worst, and average per distance.
    """
    matrix = utils.get_communication_matrix(graph)
    nb_steps = nb_repetitions * participation_interval

    sensitivities = compute_muffliato_privacy_loss(
        communication_matrix=matrix,
        nb_steps=nb_steps,
        attacker=attacker,
        participation_interval=participation_interval,
    )
    lengths = nx.single_source_shortest_path_length(graph, attacker)

    # Group sensitivities by distance
    distance_dict_MF = {}
    distance_dict_Muffliato = {}
    for i, (node, dist) in enumerate(lengths.items()):
        print(f"{i}/{len(lengths)}", end="\r")
        if dist not in distance_dict_MF:
            distance_dict_MF[dist] = []
            distance_dict_Muffliato[dist] = []

        # u-GDP implies (alpha, alpha * u**2 /2)-RDP
        # and Muffilato is 1/sens(C)-GDP since we do not rescale by the sensitivity.
        distance_dict_MF[dist].append(alpha * sensitivities[node] ** 2 / 2)
        muffliato_eps = epsilon_upper_bound(
            G=graph,
            u=node,
            v=attacker,
            T=nb_steps,
            delta_phi=1,
            alpha=alpha,
            sigma=1,
        )
        distance_dict_Muffliato[dist].append(muffliato_eps)

    return distance_dict_MF, distance_dict_Muffliato


def plot_privacy_loss_for_graph(
    graph_name: GraphName,
    nb_nodes: int,
    seed: int,
    attacker: int,
    participation_interval: int,
    nb_repetitions: int,
    alpha: float,
):
    """
    Generates a graph, computes privacy loss, and plots the sensitivity histogram.
    """
    graph = utils.get_graph(graph_name, nb_nodes, seed)

    ploss_dict_MF, ploss_dict_Muffliato = privacy_loss_by_distance(
        graph=graph,
        attacker=attacker,
        participation_interval=participation_interval,
        nb_repetitions=nb_repetitions,
        alpha=alpha,
    )

    plt.style.use("science")
    plt.figure(figsize=(12, 4))

    plot_data = []

    for name, ploss_dict in (
        ("MF", ploss_dict_MF),
        ("Muffliato-SGD", ploss_dict_Muffliato),
    ):

        distances = []
        avg_ploss = []
        min_ploss = []
        max_ploss = []

        for distance in sorted(ploss_dict.keys()):
            if distance == 0.0:
                continue
            distances.append(distance)
            avg_val = np.mean(ploss_dict[distance])
            min_val = np.min(ploss_dict[distance])
            max_val = np.max(ploss_dict[distance])
            avg_ploss.append(avg_val)
            min_ploss.append(min_val)
            max_ploss.append(max_val)
            plot_data.append(
                {
                    "approach": name,
                    "distance": distance,
                    "avg_privacy_loss": avg_val,
                    "min_privacy_loss": min_val,
                    "max_privacy_loss": max_val,
                }
            )

        # Plot average sensitivity and use its color for error bars
        (avg_line,) = plt.plot(distances, avg_ploss)
        color = avg_line.get_color()
        plt.errorbar(
            distances,
            avg_ploss,
            yerr=[
                np.array(avg_ploss) - np.array(min_ploss),
                np.array(max_ploss) - np.array(avg_ploss),
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,
            label=f"{name}",
        )

    # Create a pandas DataFrame with all plotting data
    plot_df = pd.DataFrame(plot_data)
    plt.xlabel("Shortest Path Length from Attacker", fontsize=18)
    plt.ylabel("Privacy loss", fontsize=18)
    plt.yscale("log")
    plt.grid()
    plt.title(
        f"Privacy loss accounting\nGraph: {graph_name}, {nb_nodes} nodes", fontsize=20
    )
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    file_name = (
        f"privacy_loss_{graph_name}_nodes{nb_nodes}_T{nb_repetitions}_alpha{alpha}"
    )

    pdf_path = f"figures/accounting/{file_name}.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plt.savefig(pdf_path)

    csv_path = f"results/accounting/{file_name}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    plot_df.to_csv(csv_path)

    plt.show()


def plot_privacy_loss_multiple_graphs(
    graph_configs: list[dict],
    participation_interval: int,
    nb_repetitions: int,
    alpha: float,
):
    """
    Plots privacy loss for multiple graphs on one figure.
    Each config in graph_configs should be a dict with keys:
        'graph_name', 'nb_nodes', 'seed', 'attacker'
    MF: solid line, Muffliato-SGD: dotted line, same color per graph.
    Legend: "...": Muffliato-SGD, "—" MF-DL, plus color for each graph.
    """
    plt.style.use(["science", "tableau-colorblind10"])
    plt.figure(figsize=(12, 4))
    plot_data = []
    graph_lines = []

    # Use colorblind-friendly colors from tab10
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(graph_configs))]
    markers = ["o", "s", "D", "^", "v", "P", "*", "X"]  # Add more if needed
    for idx, config in enumerate(graph_configs):
        graph_name = config["graph_name"]
        graph_display_name = graph_name
        if graph_name in GRAPH_RENAME:
            graph_display_name = GRAPH_RENAME[graph_name]
        nb_nodes = config["nb_nodes"]
        seed = config["seed"]
        attacker = config["attacker"]
        marker = markers[idx]
        color = colors[idx]

        graph = utils.get_graph(graph_name, nb_nodes, seed)
        nb_nodes = graph.number_of_nodes()
        ploss_dict_MF, ploss_dict_Muffliato = privacy_loss_by_distance(
            graph=graph,
            attacker=attacker,
            participation_interval=participation_interval,
            nb_repetitions=nb_repetitions,
            alpha=alpha,
        )

        distances = []
        avg_ploss_MF = []
        avg_ploss_Muffliato = []

        for distance in sorted(ploss_dict_MF.keys()):
            if distance == 0.0:
                continue
            distances.append(distance)
            avg_ploss_MF.append(np.mean(ploss_dict_MF[distance]))
            avg_ploss_Muffliato.append(np.mean(ploss_dict_Muffliato[distance]))
            plot_data.append(
                {
                    "graph": graph_name,
                    "distance": distance,
                    "avg_privacy_loss_MF": np.mean(ploss_dict_MF[distance]),
                    "max_privacy_loss_MF": np.max(ploss_dict_MF[distance]),
                    "min_privacy_loss_MF": np.min(ploss_dict_MF[distance]),
                    "avg_privacy_loss_Muffliato": np.mean(
                        ploss_dict_Muffliato[distance]
                    ),
                    "max_privacy_loss_Muffliato": np.max(
                        ploss_dict_Muffliato[distance]
                    ),
                    "min_privacy_loss_Muffliato": np.min(
                        ploss_dict_Muffliato[distance]
                    ),
                }
            )

        # Plot MF (solid) and Muffliato-SGD (dotted) with same color
        plt.plot(
            distances,
            avg_ploss_MF,
            linewidth=2,
            color=color,
        )
        plt.plot(
            distances,
            avg_ploss_Muffliato,
            linestyle="--",
            color=color,
            linewidth=2,
        )

        # Add error bars for MF (min/max)
        min_ploss_MF = [np.min(ploss_dict_MF[d]) for d in distances]
        max_ploss_MF = [np.max(ploss_dict_MF[d]) for d in distances]
        plt.errorbar(
            distances,
            avg_ploss_MF,
            yerr=[
                np.array(avg_ploss_MF) - np.array(min_ploss_MF),
                np.array(max_ploss_MF) - np.array(avg_ploss_MF),
            ],
            fmt=marker,
            markersize=8,
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,
            # markerfacecolor="none",  # hollow marker
            label=f"{graph_display_name} MF",
        )

        # Add error bars for Muffliato-SGD (min/max)
        min_ploss_Muffliato = [np.min(ploss_dict_Muffliato[d]) for d in distances]
        max_ploss_Muffliato = [np.max(ploss_dict_Muffliato[d]) for d in distances]
        plt.errorbar(
            distances,
            avg_ploss_Muffliato,
            yerr=[
                np.array(avg_ploss_Muffliato) - np.array(min_ploss_Muffliato),
                np.array(max_ploss_Muffliato) - np.array(avg_ploss_Muffliato),
            ],
            fmt=marker,
            markersize=8,
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,
            # markerfacecolor="none",  # hollow marker
            label=f"{graph_display_name} Muffliato-SGD",
        )

        graph_lines.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=3,
                label=graph_display_name,
                marker=marker,
                markersize=10,
            )
        )

    # Separate legends: one for accounting methods (bottom left), one for graphs (below it)
    # Accounting method legend (placed above graph legend)
    accounting_elements = [
        Line2D(
            [0], [0], color="black", linestyle="--", lw=2, label="Cyffers et al. (2022)"
        ),
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="Ours"),
    ]

    plt.xlabel("Shortest Path Length from Attacker", fontsize=18)
    plt.ylabel("Privacy loss (Rényi Divergence)", fontsize=18)
    plt.yscale("log")
    plt.grid()
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Graph legend
    graph_legend = plt.legend(
        handles=graph_lines,
        title="Graph",
        loc="lower left",
        fontsize=13,
        title_fontsize=15,
        handlelength=3,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )
    plt.gca().add_artist(graph_legend)

    plt.draw()  # Update positions

    def add_accounting_legend(accounting_elements, graph_legend):
        """
        Adds the accounting legend to the plot, aligned with the graph legend if possible.
        """
        try:
            bbox = graph_legend.get_window_extent()
            inv = plt.gca().transAxes.inverted()
            bbox_axes = inv.transform(bbox)
            x0, y0 = bbox_axes[0]  # bottom-left corner of graph legend
            x1, _ = bbox_axes[1]  # top-right corner

            # Accounting legend aligned to same y0
            accounting_legend = plt.legend(
                handles=accounting_elements,
                title="Accounting",
                fontsize=13,
                title_fontsize=15,
                handlelength=3,
                frameon=True,
                facecolor="white",
                edgecolor="black",
                loc="lower left",
                bbox_to_anchor=(x1, y0),
            )
            plt.gca().add_artist(accounting_legend)
        except Exception:
            # Fallback: plot accounting legend at bottom right
            accounting_legend = plt.legend(
                handles=accounting_elements,
                title="Accounting",
                fontsize=13,
                title_fontsize=15,
                handlelength=3,
                frameon=True,
                facecolor="white",
                edgecolor="black",
                loc="lower right",
            )
            plt.gca().add_artist(accounting_legend)

    add_accounting_legend(accounting_elements, graph_legend)

    # Save plot data to CSV
    plot_df = pd.DataFrame(plot_data)
    file_name = f"privacy_loss_all_graphs_T{nb_repetitions}_alpha{alpha}"
    csv_path = f"results/accounting/{file_name}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    plot_df.to_csv(csv_path)

    # Save figure to PDF
    pdf_path = f"figures/accounting/{file_name}.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plt.savefig(pdf_path)
    plt.show()


def plot_privacy_loss_histogram(
    matrix: np.ndarray, nb_steps: int, participation_interval: int, attacker: int
):
    # Call the privacy loss computation function
    sensitivities = compute_muffliato_privacy_loss(
        communication_matrix=matrix,
        nb_steps=nb_steps,
        participation_interval=participation_interval,
        attacker=attacker,
    )

    labels = [str(i) for i in range(len(sensitivities))]
    values = [s if not np.isinf(s) else 0 for s in sensitivities]

    plt.style.use("science")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="skyblue")

    for i, s in enumerate(sensitivities):
        if np.isinf(s):
            plt.text(
                i, 0.05, r"$\infty$", ha="center", va="bottom", color="red", fontsize=14
            )

    plt.xlabel("Node")
    plt.ylabel("Sensitivity")
    plt.title("Sensitivity Histogram per Node")
    plt.tight_layout()
    plt.show()


def main():
    # Set parameters
    graph_name = "erdos"
    # graph_name = "ego"
    nb_nodes = 100
    seed = 421
    attacker = 0
    participation_interval = 1
    nb_repetitions = 10
    alpha = 2

    nb_steps = nb_repetitions * participation_interval

    import random

    random.seed(seed)
    np.random.seed(seed)

    # graph = utils.get_graph(graph_name, 20, 421)
    # nb_nodes = graph.number_of_nodes()
    # matrix = utils.get_communication_matrix(graph)

    # plot_privacy_loss_for_graph(
    #     graph_name=graph_name,
    #     nb_nodes=nb_nodes,
    #     seed=seed,
    #     attacker=attacker,
    #     participation_interval=participation_interval,
    #     nb_repetitions=nb_repetitions,
    #     alpha=alpha,
    # )

    # Debug config
    # graph_configs = [
    #     {"graph_name": "cycle", "nb_nodes": 100, "seed": 42, "attacker": 0},
    # ]

    graph_configs = [
        {"graph_name": "erdos", "nb_nodes": 100, "seed": 42, "attacker": 0},
        {"graph_name": "ego", "nb_nodes": 148, "seed": 421, "attacker": 0},
        {
            "graph_name": "peertube (connex component)",
            "nb_nodes": 271,
            "seed": 421,
            "attacker": 0,
        },
        {"graph_name": "florentine", "nb_nodes": 15, "seed": 421, "attacker": 0},
    ]

    plot_privacy_loss_multiple_graphs(
        graph_configs=graph_configs,
        participation_interval=participation_interval,
        nb_repetitions=nb_repetitions,
        alpha=alpha,
    )


if __name__ == "__main__":
    main()
