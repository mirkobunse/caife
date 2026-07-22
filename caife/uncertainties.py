"""Module containing estimators of uncertainty."""

import jax
import numpy as np
from .solvers import flatten_result, unflatten_result

def uncertainty_from_hessian(result, nll, args=(),):
    """Estimate statistical uncertainties in the style of Minuit.

    Args:
        result: A JAX pytree of result components; could be a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that the `result` minimizes.
        args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.

    Returns:
        A JAX pytree of the pair-wise correlation scores corresponding to result components.
    """
    result_vec, resultdef = flatten_result(result)
    def vec_nll(result_vec, *args):
        return nll(unflatten_result(resultdef, result_vec), *args)

    # compute the joint Hessian across all result components
    joint_hessian = jax.jacfwd(jax.grad(vec_nll))(result_vec, *args)

    # compute the bin-wise errors, just as Minuit does
    errors = np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * joint_hessian)))
    return unflatten_result(resultdef, errors)
