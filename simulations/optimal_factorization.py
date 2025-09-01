"""
This code is heavily based on Google's implementation available here:
https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py

Everything was rewritten to not use jaxlib, but np instead.

Version used for this code: https://github.com/google-research/federated/blob/a5af8c4433c9ee2d2b1565f1bcc68c89e2000a6b/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py
"""

from typing import Optional

import numpy as np
import scipy
import utils


def initialize_X_to_normalized_identity(
    nb_steps: int, np_participations: int
) -> np.ndarray:
    """Initializes X matrix to be optimized to a normalized identity matrix.

    Args:
        n_steps: Integer specifying the number of steps for which we wish to train
            the downstream ML model.
        epochs: Integer specifying the number of epochs which will be used in
            training for n_steps.

    Returns:
        A scaled identity matrix, with sensitivity 1 under the (k, b) participation
        pattern specified by the parameters.
    """
    return np.eye(nb_steps) / np_participations


def _make_permutation_matrix(n: int) -> np.ndarray:
    """Constructs a matrix of all-ones on antidiagonal, all zeros elsewhere."""
    return np.fliplr(np.eye(n))


def _permute_lower_triangle(h_lower: np.ndarray) -> np.ndarray:
    """Computes PXP^T for P permutation matrix above."""
    perm = _make_permutation_matrix(h_lower.shape[0])
    return perm @ h_lower @ perm.T


def termination_metric(dX):
    return np.abs(dX).max()


def termination_fn(dX, min_norm):
    return termination_metric(dX) <= min_norm


class MatrixFactorizer:
    def __init__(
        self,
        workload_matrix: np.ndarray,
        nb_epochs: int,
        equal_norm: bool = False,
    ) -> None:
        self.workload_matrix = workload_matrix
        self.nb_steps = self.workload_matrix.shape[0]
        self.nb_epochs = nb_epochs
        self.equal_norm = equal_norm
        self.mask = utils.get_orthogonal_mask(self.nb_steps, self.nb_epochs)

    def project_update(self, dX: np.ndarray) -> np.ndarray:
        """Project dX so that X + alpha*dX satisfies constraints for any alpha.

        Args:
            X: Current iterate, an n x n matrix
            dX: an n x n matrix, representing the gradient with respect to X.
            mask: A mask to apply constraints
            equal_norm: Flag indicating if columns should have equal norm

        Returns:
            An n x n matrix, representing the projected gradient.
        """
        if self.equal_norm:
            diag = np.zeros(dX.shape[0])
        else:
            dsum = np.diag(dX).reshape(self.nb_epochs, -1).sum(axis=0) / self.nb_epochs
            diag = np.diag(dX) - np.kron(np.ones(self.nb_epochs), dsum)
        dX[np.diag_indices_from(dX)] = diag
        dX *= self.mask
        return dX

    def lbfgs_direction(
        self, X: np.ndarray, dX: np.ndarray, X1: np.ndarray, dX1: np.ndarray
    ) -> np.ndarray:
        """Computes the LBFGS search direction.

        Args:
            X: The current iterate, an n x n matrix
            dX: The current gradient, an n x n matrix
            X1: The previous iterate, an n x n matrix
            dX1: The previous gradient, an n x n matrix

        Returns:
            The (negative) search direction, an n x n matrix
        """
        S = X - X1
        Y = dX - dX1
        rho = 1.0 / np.sum(Y * S)
        alpha = rho * np.sum(S * dX)
        gamma = np.sum(S * Y) / np.sum(Y**2)
        Z = gamma * (dX - rho * np.sum(S * dX) * Y)
        beta = rho * np.sum(Y * Z)
        Z += S * (alpha - beta)
        return Z

    def loss_and_gradient(self, X: np.ndarray) -> tuple:
        try:
            H = scipy.linalg.solve(X, self.workload_matrix.T, assume_a="pos")
        except np.linalg.LinAlgError:
            # Jax's jax.scipy.linalg.solve returns NaN when the matrix is not positive definite, whereas scipy raises an error. This trick is to obtain the same behavior as the jax implementation - the optimization function downstream will detect such a behavior.
            H = np.zeros_like(self.workload_matrix) * np.nan
        loss = np.trace(H @ self.workload_matrix)
        gradient = self.project_update(-H @ H.T)
        return loss, gradient

    def optimize(
        self,
        iters: int = 1000,
        initial_X: Optional[np.ndarray] = None,
        min_norm: float = 1e-8,  # If the gradient has norm less than this, end optimization.
        initial_step_size: float = 1.0,
    ) -> np.ndarray:

        if initial_X is None:
            X = initialize_X_to_normalized_identity(
                nb_steps=self.nb_steps,
                np_participations=self.nb_epochs,
            )
        else:
            X = initial_X

        if not np.all((1 - self.mask) * X == 0):
            raise ValueError(
                "Initial X matrix is nonzero in indices i, j where "
                "i != j and some user can participate in rounds i and "
                "j. Such entries being zero is generally assumed by the "
                "optimization code here"
            )

        loss, dX = self.loss_and_gradient(X)
        X1 = X
        dX1 = dX
        loss1 = loss
        Z = dX

        for step in range(iters):
            print(
                f"Step: {step}/{iters} - loss {loss1:.2f} - termination condition: {termination_metric(dX):.2e} <= {min_norm:.0e}"
                + 10 * " ",
                end="\r",
            )
            # Check if X is positive semidefinite
            # TODO: Remove
            # print(f"Step {step}: positive & definite: {check_positive_definite(X)}")
            step_size = initial_step_size
            for _ in range(30):
                X = X1 - step_size * Z
                # print(f"Step {step}: positive & definite: {check_positive_definite(X)}")
                loss, dX = self.loss_and_gradient(X)
                if np.isnan(loss) or np.isnan(dX).any():
                    step_size *= 0.25
                elif loss < loss1:
                    loss1 = loss
                    break
            if termination_fn(dX=dX, min_norm=min_norm):
                # Early-return triggered; return X immediately.
                print("")
                return X
            Z = self.lbfgs_direction(X, dX, X1, dX1)
            X1 = X
            dX1 = dX
        print(f"\nFactorization did not finish! Aborting after {iters} iteration....")
        return X

    def get_factorization(self, gram_matrix):
        C_lower = np.linalg.cholesky(_permute_lower_triangle(gram_matrix))
        C_matrix = _permute_lower_triangle(C_lower.T).astype(self.workload_matrix.dtype)
        # C_lower = np.linalg.cholesky(gram_matrix)
        # C_matrix = C_lower.T.astype(self.workload_matrix.dtype)

        B_matrix = self.workload_matrix @ np.linalg.pinv(C_matrix)
        return B_matrix, C_matrix


def get_optimal_factorization(
    workload, nb_steps: int, nb_epochs: int, optimizer_iterations: int = 10000
):
    optimizer = MatrixFactorizer(workload_matrix=workload, nb_epochs=nb_epochs)
    gram_encoder = optimizer.optimize(optimizer_iterations)
    B_optimized, C_optimized = optimizer.get_factorization(gram_encoder)
    return B_optimized, C_optimized


def get_optimal_factorization_gram(
    gram_matrix,
    nb_steps: int,
    nb_epochs: int,
    optimizer_iterations: int = 10000,
):
    workload = np.linalg.cholesky(_permute_lower_triangle(gram_matrix))
    optimizer = MatrixFactorizer(
        workload_matrix=workload,
        nb_epochs=nb_epochs,
    )
    gram_encoder_C = optimizer.optimize(optimizer_iterations)
    B_optimized, C_optimized = optimizer.get_factorization(gram_encoder_C)
    return B_optimized, C_optimized


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use(["science"])

    nb_epochs = 4  # Number of passes over each data point. k in a (b,k)-participation scheme, and b= nb_steps//nb_epochs
    nb_batches = 16  # b, number of batches

    nb_steps = nb_batches * nb_epochs

    workload = np.tri(nb_steps)

    B_optimized, C_optimized = get_optimal_factorization(
        workload=workload, nb_steps=nb_steps, nb_epochs=nb_epochs
    )

    assert np.allclose(workload, B_optimized @ C_optimized)

    plt.figure()
    # C_optimized = np.linalg.pinv(C_optimized)
    plt.imshow(C_optimized, cmap="bwr")
    plt.clim(-np.abs(C_optimized).max(), np.abs(C_optimized).max())
    plt.colorbar()

    X = C_optimized.T @ C_optimized

    B_optimized_gram, C_optimized_gram = get_optimal_factorization_gram(
        gram_matrix=workload.T @ workload,
        nb_steps=nb_steps,
        nb_epochs=nb_epochs,
    )

    assert np.allclose(workload, B_optimized_gram @ C_optimized_gram)

    plt.figure()
    # C_optimized = np.linalg.pinv(C_optimized)
    plt.imshow(C_optimized_gram, cmap="bwr")
    plt.clim(-np.abs(C_optimized_gram).max(), np.abs(C_optimized_gram).max())
    plt.colorbar()
    plt.show()
