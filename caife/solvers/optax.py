import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from . import AbstractSolver, Result, _jnp_softmax, _np_softmax, _rand_x0

@dataclass
class OptaxResult(Result):
    pass # TODO add properties (iteration number, error message, ...)


@dataclass
class OptaxSolver(AbstractSolver):
    """TODO: document."""

    def solve(nll, target_dim, nuisance_dim, n_samples):
        nll = lambda ell: nll( # cast to a function of the latent variable ell
            n_samples * _jnp_softmax(ell)) # TODO consider n_samples as a nuisance parameter
        jac = jax.grad(nll) # Jacobian
        hess = jax.jacfwd(jac) # Hessian through forward-mode AD
        x = _rand_x0( # random starting point
            np.random.RandomState(self.seed),
            target_dim,
        )

        # TODO optimize
        return OptaxResult(_np_softmax(x))
