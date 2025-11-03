import time

import matplotlib.pyplot as plt
import numpy as np
import plotters
import utils
import workloads_generator
from matplotlib.colors import ListedColormap, Normalize


def plot_correlations(
    communication_matrix, graph_name, num_batches, num_repetition, seed=421, debug=False
):
    num_steps = num_batches * num_repetition

    nb_nodes = len(communication_matrix)

    C_NONOISE = np.zeros((num_steps, num_steps))
    C_LDP = workloads_generator.MF_LDP(nb_nodes=1, nb_iterations=num_steps)
    C_ANTIPGD = workloads_generator.MF_ANTIPGD(nb_nodes=1, nb_iterations=num_steps)
    C_SR_LOCAL = workloads_generator.SR_local_factorization(nb_iterations=num_steps)
    C_BSR_LOCAL = workloads_generator.BSR_local_factorization(
        nb_iterations=num_steps, nb_epochs=num_repetition
    )
    C_OPTIMAL_LOCAL = workloads_generator.MF_OPTIMAL_local(
        communication_matrix=communication_matrix,
        nb_nodes=nb_nodes,
        nb_steps=num_steps,
        nb_epochs=num_repetition,
        caching=True,
        verbose=True,
    )
    # C_OPTIMAL_DL = workloads_generator.MF_OPTIMAL_DL(
    #     communication_matrix=communication_matrix,
    #     nb_nodes=nb_nodes,
    #     nb_steps=num_steps,
    #     nb_epochs=num_repetition,
    #     post_average=False,
    #     graph_name=graph_name,
    #     seed=seed,
    #     caching=True,
    #     verbose=True,
    # )
    C_OPTIMAL_DL_POSTAVERAGE = workloads_generator.MF_OPTIMAL_DL(
        communication_matrix=communication_matrix,
        nb_nodes=nb_nodes,
        nb_steps=num_steps,
        nb_epochs=num_repetition,
        post_average=True,
        graph_name=graph_name,
        seed=seed,
        caching=True,
        verbose=True,
    )

    configs = [
        # ("Unnoised baseline", C_NONOISE),
        ("LDP", C_LDP),
        ("ANTIPGD", C_ANTIPGD),
        # ("BSR_LOCAL", C_SR_LOCAL),
        # ("BSR_BANDED_LOCAL", C_BSR_LOCAL),
        # ("OPTIMAL_DL_MSG", C_OPTIMAL_DL),
        ("OPTIMAL_LOCAL", C_OPTIMAL_LOCAL),
        ("OPTIMAL_DL_POSTAVG", C_OPTIMAL_DL_POSTAVERAGE),
    ]

    # Plot all the C_daggers side by side with a unified colorbar
    correlation_types = [r"$C$", r"$C^\dagger$"]
    fig, axes = plt.subplots(2, len(configs), figsize=(8 * len(configs), 8))
    colorbar_axes = [
        fig.add_axes(
            [0.92, 0.56, 0.01, 0.32]
        ),  # [left, bottom, width, height] for top row
        fig.add_axes([0.92, 0.12, 0.01, 0.32]),  # for bottom row
    ]
    for j, correlation_type in enumerate(["C", "C^+"]):
        C_displays = []
        for method_name, C in configs:
            C_display = C
            if correlation_type == "C^+":
                C_display = np.linalg.pinv(C_display)
            # C_display = np.kron(C_display, np.identity(nb_nodes))
            C_displays.append(C_display)

        # Find global vmin and vmax for unified colorbar
        vmin = min(C.min() for C in C_displays)
        vmax = max(C.max() for C in C_displays)
        vmax_abs = max(abs(vmin), abs(vmax))

        ims = []
        for i, (method_name, C_display) in enumerate(
            zip([name for name, _ in configs], C_displays)
        ):
            method_name = utils.METHOD_DISPLAY_NAMES[method_name]
            ax = axes[j][i]
            im = ax.imshow(C_display, cmap="bwr", vmin=-vmax_abs, vmax=vmax_abs)
            if j == 0:
                ax.set_title(method_name)
            ims.append(im)
        # Set the ylabel for each row to the correlation type (rendered as LaTeX)
        axes[j][0].set_ylabel(
            correlation_types[j], fontsize=14, rotation=0, labelpad=10
        )
        fig.colorbar(
            ims[0],
            cax=colorbar_axes[j],
            orientation="vertical",
        )

    plt.tight_layout()
    characteristics = [
        "n = {}".format(len(communication_matrix)),
        "Graph: {}".format(graph_name),
        "Batches: {}".format(num_batches),
        "Repetitions: {}".format(num_repetition),
        "Seed: {}".format(seed),
    ]
    fig.suptitle(
        " | ".join(characteristics),
        fontsize=10,
    )
    # plt.show()


def plot_correlations_across_graphs(
    graphs_dict,
    num_batches,
    num_repetition,
    seed=421,
    debug=False,
    caching=True,
):
    """
    For each graph in graph_names, plot C^+ for each method/config.
    Each row corresponds to a graph, each column to a method/config.
    """
    num_steps = num_batches * num_repetition
    configs = [
        (
            "LDP",
            lambda nb_nodes, num_steps, cm, graph_name: workloads_generator.MF_LDP(
                nb_nodes=1, nb_iterations=num_steps
            ),
        ),
        (
            "ANTIPGD",
            lambda nb_nodes, num_steps, cm, graph_name: workloads_generator.MF_ANTIPGD(
                nb_nodes=1, nb_iterations=num_steps
            ),
        ),
        (
            "OPTIMAL_LOCAL",
            lambda nb_nodes, num_steps, cm, graph_name: workloads_generator.MF_OPTIMAL_local(
                communication_matrix=cm,
                nb_nodes=nb_nodes,
                nb_steps=num_steps,
                nb_epochs=num_repetition,
                caching=True,
                verbose=True,
            ),
        ),
        (
            "OPTIMAL_DL_POSTAVG",
            lambda nb_nodes, num_steps, cm, graph_name: workloads_generator.MF_OPTIMAL_DL(
                communication_matrix=cm,
                nb_nodes=nb_nodes,
                nb_steps=num_steps,
                nb_epochs=num_repetition,
                post_average=True,
                graph_name=graph_name,
                seed=seed,
                caching=caching,  # Cannot cache, we are playing around with the graph too much (probability of an erdos)
                verbose=False,
            ),
        ),
    ]

    fig, axes = plt.subplots(
        len(graphs_dict), len(configs), figsize=(8 * len(configs), 4 * len(graphs_dict))
    )
    if len(graphs_dict) == 1:
        axes = [axes]
    if len(configs) == 1:
        axes = [[axs] for axs in axes]
    colorbar_axes = fig.add_axes([0.92, 0.12, 0.01, 0.76])  # unified colorbar for all

    # Compute all C^+ matrices
    C_plus_displays = []
    for graph_idx, (graph_name, cm) in enumerate(graphs_dict.items()):
        nb_nodes = len(cm)
        row_C_plus = []
        for method_name, config_fn in configs:
            base_graphname = graph_name.split(" ")[0]
            C = config_fn(nb_nodes, num_steps, cm, base_graphname)
            C_plus = np.linalg.inv(C)
            row_C_plus.append(C_plus)
        C_plus_displays.append(row_C_plus)

    # Find global vmin/vmax for unified colorbar
    vmin = min(C.min() for row in C_plus_displays for C in row)
    vmax = max(C.max() for row in C_plus_displays for C in row)
    vmax_abs = max(abs(vmin), abs(vmax))

    # Create a cividis colormap with white at zero
    base_cmap = plt.get_cmap("cividis")
    n_colors = 256
    colors = base_cmap(np.linspace(0, 1, n_colors))
    # Find the index corresponding to zero in the normalization
    norm = Normalize(vmin=-vmax_abs, vmax=vmax_abs)
    zero_idx = int(n_colors * norm(0))
    colors[zero_idx] = [1, 1, 1, 1]  # RGBA for white
    custom_cmap = ListedColormap(colors)

    # Plot
    for i, graph_name in enumerate(graphs_dict):
        for j, (method_name, _) in enumerate(configs):
            print(
                f"Graph: {graph_name}, Method: {method_name}, L1 Norms: {np.linalg.norm(C_plus_displays[i][j], ord=1)}"
            )
            ax = axes[i][j]
            current_C_display = C_plus_displays[i][j]
            # Clip values near zero to exactly zero (e.g., threshold 1e-10)
            current_C_display[np.abs(current_C_display) < 1e-2] = 0
            im = ax.imshow(
                current_C_display, cmap=custom_cmap, vmin=-vmax_abs, vmax=vmax_abs
            )
            if i == 0:
                ax.set_title(utils.METHOD_DISPLAY_NAMES[method_name])
            if j == 0:
                ax.set_ylabel(graph_name, fontsize=12)
    fig.colorbar(
        im,
        cax=colorbar_axes,
        orientation="vertical",
    )
    plt.tight_layout()
    characteristics = [
        "Batches: {}".format(num_batches),
        "Repetitions: {}".format(num_repetition),
        "Seed: {}".format(seed),
    ]
    fig.suptitle(
        " | ".join(characteristics),
        fontsize=10,
    )
    # plt.show()


@utils.profile_memory_usage
def main():
    graph_name = "cycle"
    n = 1000
    seed = 421
    num_batches = 16
    num_repetition = 4
    debug = True
    G = utils.get_graph(graph_name, n, seed)
    communication_matrix = utils.get_communication_matrix(G)
    caching = True

    # plot_correlations(
    #     communication_matrix=communication_matrix,
    #     graph_name=graph_name,
    #     num_batches=num_batches,
    #     num_repetition=num_repetition,
    #     seed=seed,
    #     debug=debug,
    # )

    # Example usage of plot_correlations_across_graphs
    graphs_dicts = {
        "cycle 10": utils.get_communication_matrix(utils.get_graph("cycle", 10, seed)),
        "cycle 100": utils.get_communication_matrix(
            utils.get_graph("cycle", 100, seed)
        ),
        "cycle 500": utils.get_communication_matrix(
            utils.get_graph("cycle", 500, seed)
        ),
        "cycle 1000": utils.get_communication_matrix(
            utils.get_graph("cycle", 1000, seed)
        ),
    }

    graphs_dicts = {
        f"cycle {p}": utils.get_communication_matrix(utils.get_graph("cycle", p, seed))
        for p in [10, 100, 1000]
    }

    # graphs_dicts = {
    #     "erdos 100": utils.get_communication_matrix(
    #         utils.get_graph("erdos", 100, seed)
    #     ),
    #     "cycle 100": utils.get_communication_matrix(
    #         utils.get_graph("cycle", 100, seed)
    #     ),
    #     "complete 100": utils.get_communication_matrix(
    #         utils.get_graph("complete", 100, seed)
    #     ),
    # }

    nb_nodes = 100
    probs = [1, 0.5, 0.1, 0.05]
    graphs_dicts = {}
    caching = False
    for p in probs:
        er_graph = utils.get_erdos_renyi_graph(nb_nodes, p, seed, 10000)
        print(list(er_graph.neighbors(0)))
        er_graph.add_edges_from([(i, i) for i in range(nb_nodes)])
        graphs_dicts[f"erdos p={p}"] = utils.get_communication_matrix(er_graph)

    plot_correlations_across_graphs(
        graphs_dict=graphs_dicts,
        num_batches=num_batches,
        num_repetition=num_repetition,
        seed=seed,
        debug=debug,
        caching=caching,
    )
    plt.show()


if __name__ == "__main__":
    main()
