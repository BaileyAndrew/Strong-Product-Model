"""
This contains helper functions for the main code
"""

import numpy as np
from scipy import linalg

def sim_diag(
    X: np.ndarray,
    Y: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray
]:
    """
    Simultaneously diagonalize two positive definite matrices by congruence
    (Technically only X must be positive definite, the other must be symmetric)

    Returns P, D such that X = PP^T and Y = PDP^T
    """

    X_sqrt = linalg.sqrtm(X)
    X_sqrt_inv = linalg.inv(X_sqrt)
    S = X_sqrt_inv @ Y @ X_sqrt_inv
    D, V = linalg.eigh(S)
    P = X_sqrt @ V

    return P, D

def blockwise_trace_ks(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr_d2[(Lam kronsum D)^-1]

    Lam, D are diagonal matrices
    """

    internal = 1 / (Lam[:, None] + D[None, :])
    return internal.sum(axis=1)

def stridewise_trace_ks(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr^d1[(Lam kronsum D)^-1]

    Lam, D are diagonal matrices
    """

    internal = 1 / (Lam[:, None] + D[None, :])
    return internal.sum(axis=0)

def stridewise_trace_mult(
    Lam: np.ndarray,
    D: np.ndarray
) -> np.ndarray:
    """
    Computes tr^d1[(Lam kronsum D)^-1 * (Lam kronprod I)]

    Lam, D are diagonal matrices
    """

    internal = Lam[:, None] / (Lam[:, None] * D[None, :])
    return internal.sum(axis=0)

def vec_kron_sum(Xs: list) -> np.array:
    """Compute the Kronecker vector-sum"""
    if len(Xs) == 1:
        return Xs[0]
    elif len(Xs) == 2:
        return np.kron(Xs[0], np.ones(Xs[1].shape[0])) + np.kron(np.ones(Xs[0].shape[0]), Xs[1])
    else:
        d_slash0 = np.prod([X.shape[0] for X in Xs[1:]])
        return (
            np.kron(Xs[0], np.ones(d_slash0))
            + np.kron(np.ones(Xs[0].shape[0]), vec_kron_sum(Xs[1:]))
        )