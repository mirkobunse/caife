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

    return loss.sum() # TODO need to evaluate whether .mean() should be used

def tikhonov_regularization(f: ArrayLike):
    """Compute the Tikhonov regularization.

    Args:
        f: A candidate solution to regularize.

    Returns:
        The (un-scaled) value of the Tikhonov regularization.
    """

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
