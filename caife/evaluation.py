"""Module containing scores for the evaluation of a model's results."""

import jax
import numpy as np
from .solvers import Result

def global_correlation_coefficient(f_est, nll, ignore_overflow_bins=True):
    """Compute the global correlation coefficient, as proposed by James: Statistical Methods in Experimental Physics (2006) and by Morik & Rhode: Discovery in Physics (2023). This coefficient should be minimal in unfolding because the true target bins should be independent, such that any correlations should be regarded as artifacts that stem from the reconstruction process.

    James (2006) defines the global correlation coefficient of a single parameter P as the maximum correlation between P and all possible linear combinations of all other parameters. Morik & Rhode (2023) apply this idea to unfolding by assessing the mean global correlation coefficient over all target bins. This assessment is implemented here.

    Args:
        f_est: The estimated spectrum, shape (n_target_bins,).
        nll: The negative log-likelihood function that `f_est` minimizes.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The value of the global correlation coefficient.
    """
    if not isinstance(f_est, Result):
        raise ValueError("f_est must be of type caife.solvers.Result")

    # compute the joint Hessian for the spectrum and nuisance parameters
    if len(f_est.values) > 1:
        fn = lambda vec: nll(*f_est.zip(vec)) # assume nll: (*vec) -> loss
    else:
        fn = nll # assume nll: f -> loss
    hess = jax.jacfwd(jax.grad(fn))(f_est.zip()) # joint Hessian at result
    hess = np.sqrt(.5) * hess # Minuit scales the Hessian of a likelihood
    hess = hess[:len(np.array(f_est)), :len(np.array(f_est))] # ignore systematics

    # bin-wise coefficients, see p. 28 in James (2006)
    global_correlation_coefficients = np.sqrt(
        1 - 1 / (np.diagonal(np.linalg.inv(hess)) * np.diagonal(hess)))

    # mean value, see Fig 10.9 and Eq. 10.39 in Morik & Rhode (2023)
    if ignore_overflow_bins:
        global_correlation_coefficients = global_correlation_coefficients[1:-1]
    return global_correlation_coefficients.mean()


def pairwise_correlation_score(f_est, nll, ignore_overflow_bins=True):
    """Compute the average pair-wise correlation.

    This alternative to the `global_correlation_coefficient` computes the correlation matrix from the covariance matrix and averages all off-diagonal entries. Hence, it computes the average pair-wise correlation between target bins instead of the average maximum correlation of each target bin with all linear combinations of the other bins.

    Args:
        f_est: The estimated spectrum, shape (n_target_bins,).
        nll: The negative log-likelihood function that `f_est` minimizes.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The value of the pair-wise correlation score.
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
    if ignore_overflow_bins:
        corr = corr[1:-1, 1:-1]
    return np.mean(np.abs(np.triu(corr, k=1))) # the sum of all off-diagonal entries


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
