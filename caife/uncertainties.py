import jax
import numpy as np
from jax import numpy as jnp

def uncertainty_from_hessian(f_est, nll):
    """Estimate uncertainties like Minuit does."""
    hess = jax.jacfwd(jax.grad(nll))(jnp.array(f_est)) # Hessian at result
    return np.sqrt(np.diagonal(np.linalg.inv(np.sqrt(.5) * hess)))
