import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use(["science"])


def plot_factorization(
    C_matrix,
    title: Optional[str] = None,
    details: Optional[str] = None,
    save_name_properties: Optional[str] = None,
    debug: bool = False,
):
    plt.figure()
    plt.imshow(C_matrix, cmap="bwr")
    plt.clim(-np.abs(C_matrix).max(), np.abs(C_matrix).max())
    plt.colorbar()
    plt.title("C_OPTI_LOCAL")
    # Add experiment details as a subtitle below the plot
    if details is not None:
        plt.subplots_adjust(bottom=0.18)
        plt.figtext(
            0.5, 0.005, details, wrap=True, horizontalalignment="center", fontsize=10
        )
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if debug:
        plt.show()
    elif save_name_properties is not None:
        os.makedirs("figures/optimal_factorization", exist_ok=True)
        plt.savefig(
            f"figures/optimal_factorization/{save_name_properties}.pdf", format="pdf"
        )

        os.makedirs("figures/optimal_factorization_jpeg", exist_ok=True)
        plt.savefig(
            f"figures/optimal_factorization_jpeg/{save_name_properties}.jpeg",
            dpi=300,
        )


def plot_housing_results(
    all_test_losses, num_steps, details, experiment_properties, debug: bool = False
):
    plt.figure()
    for name, test_losses in all_test_losses.items():
        # Plot the min and max

        avg_loss = test_losses.mean(axis=1)
        min_loss = test_losses.min(axis=1)
        max_loss = test_losses.max(axis=1)

        (line,) = plt.plot(range(num_steps), avg_loss, label=name)
        color = line.get_color()
        plt.fill_between(range(num_steps), min_loss, max_loss, alpha=0.2, color=color)

    plt.legend()
    plt.grid()

    plt.title("Test losses per model")
    plt.xlabel("Communication rounds")
    plt.ylabel("Test loss")
    # Add experiment details as a subtitle below the plot
    plt.subplots_adjust(bottom=0.18)
    plt.figtext(
        0.5, 0.005, details, wrap=True, horizontalalignment="center", fontsize=10
    )
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    if debug:
        plt.show()
    else:
        # Ensure the figures directory exists
        os.makedirs("figures/housing", exist_ok=True)
        # Create a unique filename with experiment details
        fig_filename = f"figures/housing/{experiment_properties}.pdf"

        plt.savefig(fig_filename, bbox_inches="tight", dpi=200, format="pdf")
        print(f"Figure saved to {fig_filename}")
    return
