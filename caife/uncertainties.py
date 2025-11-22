import jax
import numpy as np
from jax import numpy as jnp
from functools import partial

def uncertainty_from_hessian(f_est, nll):
    """Estimate uncertainties like Minuit does."""
    if f_est.nuisance_parameters is not None:
        fn = lambda f: nll(f, f_est.nuisance_parameters) # fix parameters
    else:
        fn = nll # assume nll: f -> loss
    hess = jax.jacfwd(jax.grad(fn))(jnp.array(f_est)) # Hessian at result
    return np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess)))

def systematic_uncertainty_from_hessian(f_est, nll):
    """Estimate uncertainties from systematic parameters."""
    fn = lambda s: nll(f_est, s) # fix the estimate
    hess = jax.jacfwd(jax.grad(fn))(jnp.array(f_est.nuisance_parameters))
    return np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess)))
