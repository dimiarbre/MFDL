import time

import matplotlib.pyplot as plt
import numpy as np
import plotters
import utils
import workloads_generator


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
    fig, axes = plt.subplots(2, len(configs), figsize=(1 * len(configs), 1))
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
    fig.text(
        0.5,
        0.02,
        " | ".join(characteristics),
        ha="center",
        va="center",
        fontsize=10,
    )
    plt.show()


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

    plot_correlations(
        communication_matrix=communication_matrix,
        graph_name=graph_name,
        num_batches=num_batches,
        num_repetition=num_repetition,
        seed=seed,
        debug=debug,
    )


if __name__ == "__main__":
    main()
