import numpy as np
import utils
import workloads_generator


def compute_muffliato_privacy_loss(
    communication_matrix: np.ndarray,
    nb_steps: int,
    attacker: int,
    participation_interval: int,
) -> float:
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
    # Let B = P @ W.
    # For Muffliato, the sensitivity is B^+ @ B (since C = Identity)
    # W is lower triangular of full rank, thus B is of maximal rank (P being a projection).
    # Thus, B^+ @ B = B.T @ (B @ B.T)^{-1} @ B
    # Now, we know (B @ B.T)^{-1} @ B is the solution to the linear system (B @ B.T) X = B
    # Thus, we solve and get X = (B @ B.T)^{-1} @ B, and just need B.T @ X
    X = np.linalg.solve(B @ B.T, B)  # shape (m, n)
    PW_proj = B.T @ X

    # TODO: compute the sensitivity (max of sum... cf paper).
    sens = workloads_generator.compute_cyclic_repetitions(
        PW_proj,
        participation_interval=participation_interval,
        nb_steps=nb_steps,
        nb_nodes=nb_nodes,
    )

    return sens


def main():
    # Create a basic graph (e.g., a ring graph with 5 nodes)
    graph = utils.get_graph("cycle", 20, 421)
    matrix = utils.get_communication_matrix(graph)

    # Set parameters
    attacker = 0
    participation_interval = 16
    nb_steps = 20 * participation_interval

    # Call the privacy loss computation function
    compute_muffliato_privacy_loss(
        communication_matrix=matrix,
        nb_steps=nb_steps,
        attacker=attacker,
        participation_interval=participation_interval,
    )


if __name__ == "__main__":
    main()
