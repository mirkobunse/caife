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
    return np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess)))[:len(np.array(f_est))]
