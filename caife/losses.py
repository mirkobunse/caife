"""Module containing common loss functions."""

from __future__ import annotations

import math
import operator

import jax
from jax import numpy as jnp

from .utils.pytree import promote_structure

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


def _tikhonov_regularization_fn(f, scaling_factors, order):
    if f.ndim > 2:
        raise ValueError("f.ndim must be <= 2")

    # create the (square root of the) Tikhonov matrix of any order
    T_tmp = ( # Eq. 40 in Bunse et al. (2024, DMKD)
        jnp.diag(jnp.full(f.shape[-1], 1))
        + jnp.diag(jnp.full(f.shape[-1] - 1, -1), 1)
    )
    T = T_tmp
    for _ in range(order):
        T = T.T @ T_tmp
    i_start = math.ceil(order / 2)
    i_end = -(order + 1) // 2
    T = T[i_start:i_end] # [:-1], [1:-1], [1:-2], [2:-2], ...

    # apply scaling factors
    if scaling_factors is not None:
        T *= scaling_factors[i_start:i_end].reshape(-1, 1)

    # compute the penalty
    Tf = T @ f.T  # shape (n_classes-2, n_solutions)
    return jnp.sum(Tf * Tf, axis=0) / 2  # shape (n_solutions,)


def tikhonov_regularization(f, scaling_factors=None, order=1, reduce=True):
    """Compute the Tikhonov regularization.

    Args:
        f: A candidate solution or a matrix of candidate solutions to regularize, shape `(n_classes,)` or `(n_solutions, n_classes)`.
        scaling_factors (optional): A vector of `(n_classes,)` factors that scale the rows of the Tikhonov matrix or `None` for unit scales. Defaults to `None`.
        order (optional): The largest order of a polynomial that is not penalized by this regularization term. Defaults to `1`, so that any perfectly linear function (i.e., a function of polynomial order one) produces a zero-valued regularization term.

    Returns:
        The value of the Tikhonov regularization with shape `(n_solutions,)` where `n_solutions=1` if `f.shape==(n_classes,)`.
    """
    scaling_factors, order = promote_structure(f, scaling_factors, order)
    values = jax.tree.map(
        _tikhonov_regularization_fn, f, scaling_factors, order)
    if reduce:
        return jax.tree.reduce(operator.add, values)
    return values
