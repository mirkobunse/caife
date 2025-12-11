import jax
import numpy as np
from .solvers import Result
from jax import numpy as jnp
from functools import partial

def uncertainty_from_hessian(f_est, nll):
    """Estimate uncertainties like Minuit does."""
    if not isinstance(f_est, Result):
        raise ValueError("f_est must be of type caife.solvers.Result")

    # compute the joint Hessian for the spectrum and nuisance parameters
    if len(f_est.values) > 1:
        fn = lambda vec: nll(*f_est.zip(vec)) # assume nll: (*vec) -> loss
    else:
        fn = nll # assume nll: f -> loss
    hess = jax.jacfwd(jax.grad(fn))(f_est.zip()) # joint Hessian at result

    # compute the bin-wise errors, just as Minuit does
    errors = np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess)))
    if len(f_est.values) > 1:
        target_dims = [ len(x) for x in f_est.values ]
        boundaries = np.concatenate(([0], np.cumsum(target_dims)))
        return (
            errors[boundaries[i]:boundaries[i+1]]
            for i in range(len(boundaries)-1)
        )
    return errors
