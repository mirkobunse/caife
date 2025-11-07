"""Module contaning common loss functions."""

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp

def poisson_nll(g_est: NDArray, g_true: NDArray, sample_weight: NDArray | None = None):
    """Compute the (scaled) negative log-likelihood loss between `g_est` and `g_true`.

    Args:
        g_est: The estimated proxy distribution, shape `(n_proxy_bins,)`.
        g_true: The observed proxy distribution, shape `(n_proxy_bins,)`.
        sample_weight (optional): Per-bin weights, shape `(n_proxy_bins,)`.

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
    
    """
    np.average normalizes the weights to sum up to 1; its equivalent to:
        >>loss *= sample_weight / sample_weight.sum()
        >>return np.sum(loss)

    TODO: Need to evaluate whether weights should be un-normalized or normalized to len(g_est) instead of normalizing to one.
    """
    return jnp.average(loss, weights=sample_weight)

def tikhonov_regularization(f: NDArray):
    """TODO: add documentation"""
    
    # implemented according to current example in lucas/caife.ipynb
    # f can be of shape (n_classes,) or (n_samples, n_classes)
    # if f is of shape (n_samples, n_classes) the loss will return 
    # a result of shape (n_samples,) equivalent to:
    # [tikhonov(f[0]), tikhonov(f[1]), ...]
    if f.ndim > 2:
        raise ValueError("Invalid input! Must of dim <= 2")
    
    C = len(f)

    # f: (n_samples, n_classes)
    if f.ndim == 2:
        C = f.shape[1] 

    # (n_classes-2, n_classes)
    tik_mat = (
        jnp.diag(jnp.full(C, 2)) + 
        jnp.diag((jnp.full(C-1, -1)), -1) + 
        jnp.diag(jnp.full(C-1, -1), 1)
    )[1:C-1, :]

    Tf = tik_mat @ f.T # (n_classes-2, n_samples)
    return jnp.sum(Tf * Tf, axis=0) / 2 # (n_samples)
    
