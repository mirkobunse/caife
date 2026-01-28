"""Module containing scores for the evaluation of a model's results."""

import jax
import numpy as np
from .solvers import Result

def total_correlation_score(f_est, nll):
    """Compute the total correlation score as the sum of all inter-bin correlations. This score should be minimal in unfolding because the true target bins should be uncorrelated, such that any correlations are artifacts that stem from the reconstruction process.

    Args:
        f_est: The estimated spectrum, shape (n_target_bins,).
        nll: The negative log-likelihood function that `f_est` minimizes.

    Returns:
        The value of the total correlation score.
    """
    if not isinstance(f_est, Result):
        raise ValueError("f_est must be of type caife.solvers.Result")

    # compute the joint Hessian for the spectrum and nuisance parameters
    if len(f_est.values) > 1:
        fn = lambda vec: nll(*f_est.zip(vec)) # assume nll: (*vec) -> loss
    else:
        fn = nll # assume nll: f -> loss
    hess = jax.jacfwd(jax.grad(fn))(f_est.zip()) # joint Hessian at result

    # compute the covariance / error matrix, as in Minuit, and derive the correlation
    cov = np.linalg.inv(np.sqrt(.5) * hess)
    corr = cov / np.sqrt(np.diagonal(cov) * np.diagonal(cov).reshape(-1,1))
    corr = corr[:len(np.array(f_est)), :len(np.array(f_est))] # ignore nuisance parameters
    corr = corr[1:-1, 1:-1] # also ignore the under- and overflow bins
    return np.sum(np.abs(np.triu(corr, k=1))) # the sum of all off-diagonal entries


def logarithmic_earth_movers_distance(f_true, f_est):
    """Compute the Earth Mover's Distance, a.k.a. Wasserstein L1 distance, between the logarithm of a true spectrum and the logarithm of its estimate. Ignore over- and underflow bins during the computation. Due to the logarithmic scaling, and due to be choice of distance, the resulting value well represents the "physicist's eye" in assessing similarity between spectra.

    Args:
        f_true: The true spectrum, shape (n_target_bins,).
        f_est: The estimated spectrum, shape (n_target_bins,).

    Returns:
        The Earth Mover's Distance between the two spectra.
    """
    return jnp.abs(jnp.cumsum( # Earth Mover's Distance in log space
        np.log10(f_tst[1:-1]) - np.log10(f_est[1:-1])
    )).sum()
