"""Module containing scores for the evaluation of a model's results."""

import jax
import jax.flatten_util
import numpy as np

from .utils.pytree import promote_structure


def gcc_from_nll(result, nll, args=(), ignore_overflow_bins=True):
    """Compute the global correlation coefficients of the result components from the target-space negative log-likelihood function.

    This function computes the correlations from the Hessian of `nll` with respect to the target-space `result`. These legacy correlations ignore the constraints of the latent parameterization that a solver actually fits. Use `gcc_from_aux` to properly characterize the fitted parameters instead. Also see `gcc_from_aux` for more information on the global correlation coefficient.

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

    # promote the pytree structure of ignore_overflow_bins to the sub_hessians
    ignore_overflow_bins = promote_structure(sub_hessians, ignore_overflow_bins)

    # compute the GCC for each sub-Hessian
    def gcc_fn(sub_hessian, ignore_overflow_bins):
        # bin-wise coefficients, see p. 28 in James (2006)
        sub_gccs = np.sqrt(
            1 - 1 / (np.diagonal(np.linalg.inv(sub_hessian)) * np.diagonal(sub_hessian)))

        if ignore_overflow_bins:
            sub_gccs = sub_gccs[1:-1]

        # mean value, see Fig 10.9 and Eq. 10.39 in Morik & Rhode (2023)
        return sub_gccs.mean()
    with np.errstate(invalid="ignore"):
        gccs = jax.tree.map(gcc_fn, sub_hessians, ignore_overflow_bins)
    return gccs


def pcs_from_nll(result, nll, args=(), ignore_overflow_bins=True):
    """Compute the pair-wise correlation scores of the result components from the target-space negative log-likelihood function.

    This function computes the correlations from the Hessian of `nll` with respect to the target-space `result`. These legacy correlations ignore the constraints of the latent parameterization that a solver actually fits. Use `pcs_from_aux` to properly characterize the fitted parameters instead. Also see `gcc_from_aux` for more information on the global correlation coefficient.

    Args:
        result: A JAX pytree of result components; could be a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that the `result` minimizes.
        args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
        ignore_overflow_bins (optional): Either a single `bool`, applied to every result component, or a JAX pytree of bools matching the structure of `result`, to control this per component. Defaults to `True`.

    Returns:
        A JAX pytree of the pair-wise correlation scores corresponding to result components. The score of a component is `nan` if it has fewer than two entries (after ignoring the overflow bins, if requested), because such a component has no pairs of entries to correlate.
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

    # promote the pytree structure of ignore_overflow_bins to the sub_corrs
    ignore_overflow_bins = promote_structure(sub_corrs, ignore_overflow_bins)

    # compute the PCC for each sub-correlation matrix
    def pcs_fn(sub_corr, ignore_overflow_bins):
        if ignore_overflow_bins:
            sub_corr = sub_corr[1:-1, 1:-1]
        if sub_corr.shape[0] < 2:
            return np.nan # there are no pairs of bins to correlate
        return np.mean(np.abs( # average off-diagonal correlation
            sub_corr[np.triu_indices(sub_corr.shape[0], k=1)]))
    with np.errstate(invalid="ignore"):
        pcss = jax.tree.map(pcs_fn, sub_corrs, ignore_overflow_bins)
    return pcss


def gcc_from_aux(aux, ignore_overflow_bins=True):
    """Compute the global correlation coefficients of the result components from the auxiliary information of a solver.

    The global correlation coefficient (GCC) was proposed by James: Statistical Methods in Experimental Physics (2006) and by Morik & Rhode: Discovery in Physics (2023). This coefficient should be minimal in unfolding because the true target bins should be independent, such that any correlations should be regarded as artifacts that stem from the reconstruction process. James (2006) defines the GCC of a single parameter P as the maximum correlation between P and all possible linear combinations of all other parameters. This correlation can be computed in multiple ways, and one of them is implemented here. Morik & Rhode (2023) apply this idea to unfolding by assessing the mean GCC over all target bins.

    Here, the correlations are computed with respect to the latent-space parameterization that a solver actually fits. Use `gcc_from_nll` to instead compute the legacy correlations with respect to the target-space result, ignoring constraints imposed by the latent variables.

    This function will score each component on the covariance of its own entries. Selecting these entries marginalizes all other entries, like the overflow bins, and all other result components, like systematic parameters, so that the correlations with these nuisance parameters are properly accounted for.

    Args:
        aux: A dict of auxiliary information, as returned by `AbstractSolver.solve(..., return_aux=True)` together with a valid result. Must contain the keys `ell`, `hess`, and `unravel_fn`.
        ignore_overflow_bins (optional): Whether to exclude the first and the last entry of a component from its score. Either a single `bool`, applied to every result component, or a JAX pytree of bools matching the structure of the result, to control this per component. For a multi-dimensional component, like the spectrum of a `MultiTargetModel`, a single `bool` applies to every axis and a tuple with one `bool` per axis controls each axis separately. Defaults to `True`.

    Returns:
        A JAX pytree of the mean global correlation coefficients corresponding to result components. The score of a component is `nan` if none of its entries remain after ignoring the overflow bins, or if the covariance of the remaining entries is singular. The covariance is singular, for instance, if all bins of a `LatentSpectrum` are kept, because the fixed number of samples makes each bin an exact linear combination of the other bins.
    """
    def gcc_fn(cov):
        if cov.shape[0] < 1:
            return np.nan # no entries (e.g., when ignore_overflow_bins==True)

        # entry-wise coefficients, see p. 28 in James (2006)
        gccs = np.sqrt(1 - 1 / (np.diagonal(np.linalg.inv(cov)) * np.diagonal(cov)))

        # mean value, see Fig 10.9 and Eq. 10.39 in Morik & Rhode (2023)
        return gccs.mean()
    with np.errstate(invalid="ignore"):
        gccs = jax.tree.map(gcc_fn, _cov_from_aux(aux, ignore_overflow_bins))
    return gccs


def pcs_from_aux(aux, ignore_overflow_bins=True):
    """Compute the pair-wise correlation scores of the result components from the auxiliary information of a solver.

    This alternative to the `gcc_from_aux` computes the correlation matrix from the covariance matrix and averages all off-diagonal entries. Hence, it computes the average pair-wise correlation between target bins instead of the average maximum correlation of each target bin with all linear combinations of the other bins.

    Here, the correlations are computed with respect to the latent-space parameterization that a solver actually fits. Use `pcs_from_nll` to instead compute the legacy correlations with respect to the target-space result, ignoring constraints imposed by the latent variables.

    Args:
        aux: A dict of auxiliary information, as returned by `AbstractSolver.solve(..., return_aux=True)` together with a valid result. Must contain the keys `ell`, `hess`, and `unravel_fn`.
        ignore_overflow_bins (optional): Whether to exclude the first and the last entry of a component from its score. Either a single `bool`, applied to every result component, or a JAX pytree of bools matching the structure of the result, to control this per component. For a multi-dimensional component, like the spectrum of a `MultiTargetModel`, a single `bool` applies to every axis and a tuple with one `bool` per axis controls each axis separately. Defaults to `True`.

    Returns:
        A JAX pytree of the pair-wise correlation scores corresponding to result components. The score of a component is `nan` if fewer than two of its entries remain after ignoring the overflow bins, because such a component has no pairs of entries to correlate.
    """
    def pcs_fn(cov):
        if cov.shape[0] < 2:
            return np.nan # there are no pairs of entries to correlate
        corr = cov / np.sqrt(np.diagonal(cov) * np.diagonal(cov).reshape(-1,1))
        return np.mean(np.abs( # average off-diagonal correlation
            corr[np.triu_indices(corr.shape[0], k=1)]))
    with np.errstate(invalid="ignore"):
        pcss = jax.tree.map(pcs_fn, _cov_from_aux(aux, ignore_overflow_bins))
    return pcss


def _cov_from_aux(aux, ignore_overflow_bins):
    """Compute the target-space covariance of each result component, possibly selecting non-overflow bins.

    Here, the target-space covariance is computed from latent-space covariance through linearization. Given the latent-space Hessian `H_latent` and the Jacobian `J_latent2target` of the latent-to-target map, the computed covariance is `J_latent2target @ H_latent^-1 @ J_latent2target.T`. If overflow bins are ignored, the latent-to-target map is altered to ignore the overflow bins, effectively marginalizing these bins to properly treat overflow as a nuisance parameter. Since all result components are treated separately, each of them marginalizes all others in the same way.

    Returns:
        A JAX pytree with the structure of the result, containing one covariance matrix per component, shape (n_selected_entries, n_selected_entries).
    """
    # the Laplace approximation of the posterior covariance in latent space
    latent_cov = np.linalg.inv(aux["hess"])
    latent_cov = .5 * (latent_cov + latent_cov.T) # symmetrize

    # unravel_fn maps to a pytree in target space; we need its component-wise
    # Jacobians, each with shape (*component_shape, n_latents)
    jacs = jax.jacfwd(aux["unravel_fn"])(aux["ell"])

    # promote the pytree structure of ignore_overflow_bins to the jacs
    ignore_overflow_bins = promote_structure(jacs, ignore_overflow_bins)

    # propagate the latent covariance to the selected entries of each component; the
    # selection marginalizes all other entries and all other components
    def cov_fn(jac, ignore_overflow_bins):
        ndim = jac.ndim - 1 # number of dimensions of the component
        if isinstance(ignore_overflow_bins, bool): # repeat
            ignore_overflow_bins = (ignore_overflow_bins,) * ndim
        elif len(ignore_overflow_bins) != ndim:
            raise ValueError(
                f"ignore_overflow_bins={ignore_overflow_bins} must have one bool per axis of a {ndim}-dimensional component")
        jac = jac[ # ignore or keep along each dimension of the component
            tuple(slice(1, -1) if ignore else slice(None) for ignore in ignore_overflow_bins)
        ]
        jac = jac.reshape(-1, jac.shape[-1]) # shape (n_selected_entries, n_latents)
        return jac @ latent_cov @ jac.T
    return jax.tree.map(cov_fn, jacs, ignore_overflow_bins)


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
