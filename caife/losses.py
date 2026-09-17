"""Module containing common loss functions."""

from __future__ import annotations

import operator
from functools import partial

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

_EPS = 1e-8


def poisson_nll(g_est, g_true, reduce=True):
    """Compute the (scaled) negative log-likelihood loss between `g_est` and `g_true`.

    Args:
        g_est: A JAX pytree of estimated proxy distributions; could be a single vector of shape `(n_proxy_bins,)`.
        g_true: A JAX pytree of observed proxy distributions; could be a single vector of shape `(n_proxy_bins,)`.
        reduce (optional): Whether to reduce the leaf-wise values through addition. Defaults to `True`.

    Returns:
        The value of the (scaled) negative log likelihood. If `reduce` is `False`, return a JAX pytree of leaf-wise values.
    """
    values = jax.tree.map(lambda a, b: jnp.sum(a - b * jnp.log(a + _EPS)), g_est, g_true)
    if reduce:
        return jax.tree.reduce(operator.add, values)
    return values


def _tikhonov_regularization_fn(f: ArrayLike, scaling_factors: ArrayLike | None = None):
    if f.ndim > 2:
        raise ValueError("f.ndim must be <= 2")

    # create the (square root of the) Tikhonov matrix
    T = ( # f.shape[-1] = n_classes
        jnp.diag(jnp.full(f.shape[-1], 2))
        + jnp.diag((jnp.full(f.shape[-1] - 1, -1)), -1)
        + jnp.diag(jnp.full(f.shape[-1] - 1, -1), 1)
    )[1:-1, :]
    if scaling_factors is not None:
        T *= scaling_factors[1:-1].reshape(-1, 1)

    # compute the penalty
    Tf = T @ f.T  # shape (n_classes-2, n_solutions)
    return jnp.sum(Tf * Tf, axis=0) / 2  # shape (n_solutions,)


def tikhonov_regularization(f, scaling_factors=None, reduce=True):
    """Compute the Tikhonov regularization.

    Args:
        f: A candidate solution or a matrix of candidate solutions to regularize, shape `(n_classes,)` or `(n_solutions, n_classes)`.
        scaling_factors (optional): A vector of `(n_classes,)` factors that scale the rows of the Tikhonov matrix or `None` for unit scales. Defaults to `None`.

    Returns:
        The value of the Tikhonov regularization with shape `(n_solutions,)` where `n_solutions=1` if `f.shape==(n_classes,)`.
    """
    if jax.tree.structure(scaling_factors) == jax.tree.structure(f):
        values = jax.tree.map(_tikhonov_regularization_fn, f, scaling_factors)
    else:
        fn = partial(_tikhonov_regularization_fn, scaling_factors=scaling_factors)
        values = jax.tree.map(fn, f)
    if reduce:
        return jax.tree.reduce(operator.add, values)
    return values
