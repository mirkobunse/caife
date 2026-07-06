"""Module containing estimators of uncertainty."""

import jax
import numpy as np
from .solvers import create_split_fn

def uncertainty_from_hessian(result_components, nll):
    """Estimate uncertainties in the style of Minuit.

    Args:
        result_components: A tuple of result components or a single spectrum of shape (n_target_bins,).
        nll: The negative log-likelihood function that `result_components` minimizes.

    Returns:
        A vector of bin-wise errors, shape (n_target_bins,).
    """
    if not isinstance(result_components, (tuple, list)):
        result_components = (result_components,) # ensure tuple

    # compute the joint Hessian across all result_components
    split_fn = create_split_fn(result_components)
    def vector_nll(vec): # assume nll: (*result_components) -> loss
        return nll(*split_fn(vec))
    hess = jax.jacfwd(jax.grad(vector_nll))(np.concatenate(result_components))

    # compute the bin-wise errors, just as Minuit does
    errors = split_fn(np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess))))
    if len(errors) == 1:
        return errors[0] # if no tuple is entered, no tuple should be returned
    return errors
