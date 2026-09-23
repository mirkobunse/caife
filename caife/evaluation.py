"""Module containing scores for the evaluation of a model's results."""

import jax
import numpy as np


def global_correlation_coefficients(result, nll, args=(), ignore_overflow_bins=True):
    """Compute the global correlation coefficient, as proposed by James: Statistical Methods in Experimental Physics (2006) and by Morik & Rhode: Discovery in Physics (2023). This coefficient should be minimal in unfolding because the true target bins should be independent, such that any correlations should be regarded as artifacts that stem from the reconstruction process.

    James (2006) defines the global correlation coefficient of a single parameter P as the maximum correlation between P and all possible linear combinations of all other parameters. Morik & Rhode (2023) apply this idea to unfolding by assessing the mean global correlation coefficient over all target bins. This assessment is implemented here.

    Args:
        result: A JAX pytree of result components; could be a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that the `result` minimizes.
        args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
        ignore_overflow_bins (optional): Either a single `bool`, applied to every result component, or a JAX pytree of bools matching the structure of `result`, to control this per component. Defaults to `True`.

    Returns:
        A JAX pytree of the global correlation coefficients corresponding to result components.
    """
    result_vec, unravel_fn = jax.flatten_util.ravel_pytree(result)
    def vec_nll(result_vec, *args):
        return nll(unravel_fn(result_vec), *args)

    # compute the joint Hessian across all result components
    joint_hessian = jax.jacfwd(jax.grad(vec_nll))(result_vec, *args)
    joint_hessian = .5 * (joint_hessian + joint_hessian.T) # symmetrize
    joint_hessian = np.sqrt(.5) * joint_hessian # apply Minuit's scaling

    # split the joint Hessian into component-wise sub-Hessians
    sizes, treedef = jax.tree.flatten(jax.tree.map(lambda x: np.size(x), result))
    boundaries = np.cumsum(np.concatenate(([0], sizes)))
    sub_hessians = (
        joint_hessian[
            boundaries[i]:boundaries[i+1],
            boundaries[i]:boundaries[i+1],
        ]
        for i in range(len(boundaries)-1)
    )
    sub_hessians = jax.tree.unflatten( # organize in a pytree
        treedef,
        sub_hessians,
    )

    # compute the GCC for each sub-Hessian
    def gcc_fn(sub_hessian, ignore_overflow_bins):
        # bin-wise coefficients, see p. 28 in James (2006)
        sub_gccs = np.sqrt(
            1 - 1 / (np.diagonal(np.linalg.inv(sub_hessian)) * np.diagonal(sub_hessian)))

        if ignore_overflow_bins:
            sub_gccs = sub_gccs[1:-1]

        # mean value, see Fig 10.9 and Eq. 10.39 in Morik & Rhode (2023)
        return sub_gccs.mean()
    if isinstance(ignore_overflow_bins, bool):
        ignore_overflow_bins = jax.tree.unflatten( # repeat across tree structure
            treedef,
            [ignore_overflow_bins for _ in range(len(boundaries)-1)],
        )
    with np.errstate(invalid="ignore"):
        gccs = jax.tree.map(gcc_fn, sub_hessians, ignore_overflow_bins)
    return gccs


def pairwise_correlation_scores(result, nll, args=(), ignore_overflow_bins=True):
    """Compute the average pair-wise correlation.

    This alternative to the `global_correlation_coefficients` computes the correlation matrix from the covariance matrix and averages all off-diagonal entries. Hence, it computes the average pair-wise correlation between target bins instead of the average maximum correlation of each target bin with all linear combinations of the other bins.

    Args:
        result: A JAX pytree of result components; could be a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that the `result` minimizes.
        args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
        ignore_overflow_bins (optional): Either a single `bool`, applied to every result component, or a JAX pytree of bools matching the structure of `result`, to control this per component. Defaults to `True`.

    Returns:
        A JAX pytree of the pair-wise correlation scores corresponding to result components.
    """
    result_vec, unravel_fn = jax.flatten_util.ravel_pytree(result)
    def vec_nll(result_vec, *args):
        return nll(unravel_fn(result_vec), *args)

    # compute the joint Hessian across all result components
    joint_hessian = jax.jacfwd(jax.grad(vec_nll))(result_vec, *args)
    joint_hessian = .5 * (joint_hessian + joint_hessian.T) # symmetrize

    # compute the covariance / error matrix, as in Minuit, and derive the correlation
    joint_cov = np.linalg.inv(np.sqrt(.5) * joint_hessian)
    with np.errstate(invalid="ignore"):
        joint_corr = joint_cov / np.sqrt(
            np.diagonal(joint_cov) * np.diagonal(joint_cov).reshape(-1,1))

    # split the joint correlation matrix into component-wise sub-correlation matrices
    sizes, treedef = jax.tree.flatten(jax.tree.map(lambda x: np.size(x), result))
    boundaries = np.cumsum(np.concatenate(([0], sizes)))
    sub_corrs = (
        joint_corr[
            boundaries[i]:boundaries[i+1],
            boundaries[i]:boundaries[i+1],
        ]
        for i in range(len(boundaries)-1)
    )
    sub_corrs = jax.tree.unflatten( # organize in a pytree
        treedef,
        sub_corrs,
    )

    # compute the PCC for each sub-correlation matrix
    def pcs_fn(sub_corr, ignore_overflow_bins):
        if ignore_overflow_bins:
            sub_corr = sub_corr[1:-1, 1:-1]
        return np.mean(np.abs( # average off-diagonal correlation
            sub_corr[np.triu_indices(sub_corr.shape[0], k=1)]))
    if isinstance(ignore_overflow_bins, bool):
        ignore_overflow_bins = jax.tree.unflatten( # repeat across tree structure
            treedef,
            [ignore_overflow_bins for _ in range(len(boundaries)-1)],
        )
    with np.errstate(invalid="ignore"):
        pcss = jax.tree.map(pcs_fn, sub_corrs, ignore_overflow_bins)
    return pcss


def effective_number_of_degrees_of_freedom(result, unreg_nll, tau, ignore_overflow_bins=True):
    """Compute the effective number of degrees of freedom, as of Blobel (1985, 2002).

    Args:
        result: A single spectrum of shape (n_target_bins,).
        unreg_nll: The un-regularized variant of the negative log-likelihood function that the `result` minimizes.
        tau: The strength of the Tikhonov regularization.
        ignore_overflow_bins (optional): Whether to ignore the correlations with the two over- and underflow bins. Defaults to `True`.

    Returns:
        The effective number of degrees of freedom.
    """
    if jax.tree.structure(result) != jax.tree.structure(0):
        raise NotImplementedError(
            "Only implemented for a single spectrum, not for pytrees"
        )

    # compute the Hessian
    hess = jax.jacfwd(jax.grad(unreg_nll))(result)
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
    """Compute the Earth Mover's Distance, a.k.a. Wasserstein L1 distance, between the logarithm of a true spectrum and the logarithm of its estimate. Ignore over- and underflow bins during the computation. Due to the logarithmic scaling, and due to the choice of distance, the resulting value well represents the "physicist's eye" in assessing similarity between spectra.

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
