"""Module containing estimators of uncertainty."""

import jax
import numpy as np


def uncertainty_from_hessian(result, nll, args=(),):
    """Estimate statistical uncertainties in the style of Minuit.

    Args:
        result: A JAX pytree of result components; could be a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that the `result` minimizes.
        args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.

    Returns:
        A JAX pytree of the pair-wise correlation scores corresponding to result components.
    """
    result_vec, unravel_fn = jax.flatten_util.ravel_pytree(result)
    def vec_nll(result_vec, *args):
        return nll(unravel_fn(result_vec), *args)

    # compute the joint Hessian across all result components
    joint_hessian = jax.jacfwd(jax.grad(vec_nll))(result_vec, *args)
    joint_hessian = .5 * (joint_hessian + joint_hessian.T) # symmetrize

    # compute the bin-wise errors, just as Minuit does
    errors = np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * joint_hessian)))
    return unravel_fn(errors)


def uncertainty_from_aux(aux, n_samples=10_000, rng=None):
    """Estimate statistical uncertainties through sampling in the latent space.

    Args:
        aux: A dict of auxiliary information from the optimizer. Must contain keys `ell`, `hess`, and `unravel_fn`.
        n_samples (optional): The number fo samples to generate in the latent space. Defaults to `10_000`.
        rng (optional): The random number generator. Defaults to `None`.

    Returns:
        A pair of JAX pytrees, one corresponding to all lower bounds and one corresponding to all upper bounds of the results within one standard deviation.
    """
    if rng is None:
        rng = np.random.default_rng()
    ells = rng.multivariate_normal( # samples in latent space
        aux["ell"], # mean
        np.linalg.inv(aux["hess"]), # covariance matrix
        size=n_samples,
    )
    results = [aux["unravel_fn"](ell) for ell in ells] # map to target spaces
    results = jax.tree_util.tree_map( # list of pytrees -> pytree of arrays
        lambda *leaves: np.array(leaves), *results)
    lower = jax.tree_util.tree_map( # -1 sigma
        lambda leaf: np.percentile(leaf, 16, axis=0), results)
    upper = jax.tree_util.tree_map( # +1 sigma
        lambda leaf: np.percentile(leaf, 84, axis=0), results)
    return lower, upper
