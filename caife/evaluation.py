"""Module containing scores for the evaluation of a model's results."""

import jax
import numpy as np
from .solvers import create_split_fn

def global_correlation_coefficient(result_components, nll, ignore_overflow_bins=True):
    """Compute the global correlation coefficient, as proposed by James: Statistical Methods in Experimental Physics (2006) and by Morik & Rhode: Discovery in Physics (2023). This coefficient should be minimal in unfolding because the true target bins should be independent, such that any correlations should be regarded as artifacts that stem from the reconstruction process.

    James (2006) defines the global correlation coefficient of a single parameter P as the maximum correlation between P and all possible linear combinations of all other parameters. Morik & Rhode (2023) apply this idea to unfolding by assessing the mean global correlation coefficient over all target bins. This assessment is implemented here.

    Args:
        result_components: A tuple of result components or a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that `result_components` minimizes.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The value of the global correlation coefficient.
    """
    if not isinstance(result_components, (tuple, list)):
        result_components = (result_components,) # ensure tuple

    # compute the joint Hessian across all result_components
    split_fn = create_split_fn(result_components)
    def vector_nll(vec): # assume nll: (*result_components) -> loss
        return nll(*split_fn(vec))
    hess = jax.jacfwd(jax.grad(vector_nll))(np.concatenate(result_components))
    hess = np.sqrt(.5) * hess # Minuit scales the Hessian of a likelihood

    # how to compute the GCC for each sub-Hessian, corresponding to one result component
    def gcc_fn(hess_i, ignore_overflow_bins_i):
        # bin-wise coefficients, see p. 28 in James (2006)
        global_correlation_coefficients = np.sqrt(
            1 - 1 / (np.diagonal(np.linalg.inv(hess_i)) * np.diagonal(hess_i)))

        # TODO overflow bin handling requires extra consideration
        if ignore_overflow_bins:
            global_correlation_coefficients = global_correlation_coefficients[1:-1]

        # mean value, see Fig 10.9 and Eq. 10.39 in Morik & Rhode (2023)
        return global_correlation_coefficients.mean()

    # split the Hessian into result component-wise sub-Hessians and compute the GCCs
    split_and_project_fn = create_split_fn(
        result_components,
        projections=gcc_fn,
        two_dimensional=True,
    )
    with np.errstate(invalid="ignore"):
        gccs = split_and_project_fn(hess)

    # return the GCC for all result components
    if len(gccs) == 1:
        return gccs[0] # if no tuple is entered, no tuple should be returned
    return gccs


def pairwise_correlation_score(result_components, nll, ignore_overflow_bins=True):
    """Compute the average pair-wise correlation.

    This alternative to the `global_correlation_coefficient` computes the correlation matrix from the covariance matrix and averages all off-diagonal entries. Hence, it computes the average pair-wise correlation between target bins instead of the average maximum correlation of each target bin with all linear combinations of the other bins.

    Args:
        result_components: A tuple of result components or a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that `f_est` minimizes.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The value of the pair-wise correlation score.
    """
    if not isinstance(result_components, (tuple, list)):
        result_components = (result_components,) # ensure tuple

    # compute the joint Hessian across all result_components
    split_fn = create_split_fn(result_components)
    def vector_nll(vec): # assume nll: (*result_components) -> loss
        return nll(*split_fn(vec))
    hess = jax.jacfwd(jax.grad(vector_nll))(np.concatenate(result_components))

    # compute the covariance / error matrix, as in Minuit, and derive the correlation
    cov = np.linalg.inv(np.sqrt(.5) * hess)
    with np.errstate(invalid="ignore"):
        corr = cov / np.sqrt(np.diagonal(cov) * np.diagonal(cov).reshape(-1,1))

    def pcc_fn(corr_i, ignore_overflow_bins_i):
        # TODO overflow bin handling requires extra consideration
        if ignore_overflow_bins:
            corr_i = corr_i[1:-1]

        # PCC = the sum of all off-diagonal entries
        return np.mean(np.abs(np.triu(corr, k=1)))

    # split the correlation matrix and compute the GCCs
    split_and_project_fn = create_split_fn(
        result_components,
        projections=pcc_fn,
        two_dimensional=True,
    )
    with np.errstate(invalid="ignore"):
        pccs = split_and_project_fn(corr)

    # return the PCC for all result components
    if len(pccs) == 1:
        return pccs[0] # if no tuple is entered, no tuple should be returned
    return pccs


def effective_number_of_degrees_of_freedom(result_components, unreg_nll, tau, ignore_overflow_bins=True):
    """Compute the effective number of degrees of freedom, as of Blobel (1985, 2002).

    Args:
        result_components: A tuple of result components or a single spectrum of shape (n_target_bins,).
        unreg_nll: The un-regularized variant of the negative log-likelihood function that `f_est` minimizes.
        tau: The strength of the Tikhonov regularization.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The effective number of degrees of freedom.
    """
    if not isinstance(result_components, (tuple, list)):
        result_components = (result_components,) # ensure tuple
    elif len(result_components) != 2:
        raise NotImplementedError(
            "Only implemented for a single spectrum with optional systematics"
        )

    # compute the joint Hessian across all result_components
    split_fn = create_split_fn(result_components)
    def vector_nll(vec): # assume unreg_nll: (f, s) -> loss
        return unreg_nll(split_fn(vec)[0], result_components[1]) # fix s
    hess = jax.jacfwd(jax.grad(vector_nll))(result_components[0])
    if ignore_overflow_bins:
        hess = hess[1:-1, 1:-1]

    # create the Tikhonov regularization matrix
    C0 = np.eye(hess.shape[0]) + np.diag(-np.ones(hess.shape[0]-1), k=1)
    C1 = (C0.T @ C0)[1:-1]
    C = C1.T @ C1

    # compute the effective_number_of_degrees_of_freedom
    D, U = np.linalg.eigh(hess)
    D = np.diag(1 / np.sqrt(D)) # = D^{-1/2}
    C1 = D @ U.T @ C @ U @ D
    S, _ = np.linalg.eigh(C1)
    n_df = np.sum(1 / (1 + S / tau))
    return n_df


def logarithmic_earth_movers_distance(f_true, f_est):
    """Compute the Earth Mover's Distance, a.k.a. Wasserstein L1 distance, between the logarithm of a true spectrum and the logarithm of its estimate. Ignore over- and underflow bins during the computation. Due to the logarithmic scaling, and due to be choice of distance, the resulting value well represents the "physicist's eye" in assessing similarity between spectra.

    Args:
        f_true: The true spectrum, shape (n_target_bins,).
        f_est: The estimated spectrum, shape (n_target_bins,).

    Returns:
        The Earth Mover's Distance between the two spectra.
    """
    return np.abs(np.cumsum( # Earth Mover's Distance in log space
        np.log10(f_true[1:-1]) - np.log10(f_est[1:-1])
    )).sum().item()


def gaussian_nll_score(f_true, f_est, uncertainties, ignore_overflow_bins=True):
    """Compute the negative log-likelihood of some ground-truth under the Gaussian distributions that a solution represents.

    Args:
        f_true: The true spectrum, shape (n_target_bins,).
        f_est: The estimated spectrum, shape (n_target_bins,).
        uncertainties: The uncertainties of `f_est`, shape (n_target_bins,).
        ignore_overflow_bins (optional): Whether to ignore the two over- and underflow bins. Defaults to `True`.

    Returns:
        The negative log-likelihood of `np.ones(n_target_bins,)` under Gaussian distributions with means given by `f_est / f_true` and standard deviations given by `uncertainties / f_true`.
    """
    x = np.ones(len(f_true))
    loc = f_est / f_true
    scale = uncertainties / f_true
    log_probs = np.log(scale) + ((x - loc) / scale)**2 / 2
    if ignore_overflow_bins:
        log_probs = log_probs[1:-1]
    return np.sum(log_probs).item()
