"""Module containing common loss functions."""

from jax import numpy as jnp
from jax.typing import ArrayLike


def poisson_nll(g_est: ArrayLike, g_true: ArrayLike):
    """Compute the (scaled) negative log-likelihood loss between `g_est` and `g_true`.

    Args:
        g_est: The estimated proxy distribution, shape `(n_proxy_bins,)`.
        g_true: The observed proxy distribution, shape `(n_proxy_bins,)`.

    Returns:
        The value of the (scaled) negative log likelihood.
    """
    eps = 1e-8

    """
    P(g_true | g_est) = g_est^g_true * exp(-g_est) / g_true!    <- likelihood function Poisson dist.

    log(P) = g_true * log(g_est) - g_est - log(g_true!)          <- apply log
 
    log(P) = g_true * log(g_est) - g_est                          <- remove constant term since optimization stays the same

    -log(P) = g_est - g_true * log(g_est)                         <- multiply with -1
    """
    loss = g_est - g_true * jnp.log(g_est + eps)

    return loss.sum()  # TODO need to evaluate whether .mean() should be used


def tikhonov_regularization(f: ArrayLike, scaling_factors: ArrayLike | None = None):
    """Compute the Tikhonov regularization.

    Args:
        f: A candidate solution or a matrix of candidate solutions to regularize, shape `(n_classes,)` or `(n_solutions, n_classes)`.
        scaling_factors (optional): A vector of `(n_classes,)` factors that scale the rows of the Tikhonov matrix or `None` for unit scales. Defaults to `None`.

    Returns:
        The value of the Tikhonov regularization with shape `(n_solutions,)` where `n_solutions=1` if `f.shape==(n_classes,)`.
    """
    if f.ndim > 2:
        raise ValueError("f.ndim must be <= 2")

    # create the (square root of the) Tikhonov matrix
    n_classes = f.shape[-1]
    T = (
        jnp.diag(jnp.full(n_classes, 2))
        + jnp.diag((jnp.full(n_classes - 1, -1)), -1)
        + jnp.diag(jnp.full(n_classes - 1, -1), 1)
    )[1:-1, :]
    if scaling_factors is not None:
        T *= scaling_factors[1:-1].reshape(-1, 1)

    # compute the penalty
    Tf = T @ f.T  # shape (n_classes-2, n_solutions)
    return jnp.sum(Tf * Tf, axis=0) / 2  # shape (n_solutions,)
