"""Module contaning common loss functions."""

import numpy as np
from numpy.typing import ndarray

def poisson_nll(g_est: ndarray, g_true: ndarray, sample_weight: ndarray | None = None):
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
    loss = g_est - g_true * np.log(g_est + eps)

    """
    np.average normalizes the weights to sum up to 1; its equivalent to:
        >>loss *= sample_weight / sample_weight.sum()
        >>return np.sum(loss)

    TODO: Need to evaluate whether weights should be un-normalized or normalized to len(g_est) instead of normalizing to one.
    """
    return np.average(loss, weights=sample_weight)
