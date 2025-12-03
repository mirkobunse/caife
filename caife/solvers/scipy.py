import jax
import jax.numpy as jnp
import numpy as np
import traceback
from dataclasses import dataclass, field
from scipy import optimize
from . import AbstractSolver, Result


SOLVER_OPTIONS_FACTORY = lambda: {
    "gtol": 1e-8,
    "maxiter": 1000,
}


@dataclass
class ScipySolver(AbstractSolver):
    """TODO: document."""
    solver: str = "trust-ncg"
    solver_options: dict[str,object] = field(default_factory=SOLVER_OPTIONS_FACTORY)

    def solve_latent(self, latent_nll, x0):
        opt = minimize(latent_nll, x0, self.solver, self.solver_options)
        return opt.x, { "opt": opt }


def minimize(loss_fn, x0, solver="trust-ncg", solver_options=None):
    """TODO document."""
    if solver_options is None:
        solver_options = SOLVER_OPTIONS_FACTORY()
    jac = jax.grad(loss_fn) # Jacobian
    hess = jax.jacfwd(jac) # Hessian through forward-mode AD

    # error-robust optimization with a callback state
    state = _CallbackState(x0)
    try:
        opt = optimize.minimize(
            loss_fn,
            x0,
            jac=_check_derivative(jac, "jac"), # safe-guard derivatives
            hess=_check_derivative(hess, "hess"),
            method=solver,
            options=solver_options,
            callback=state.callback(),
        )
    except (DerivativeError, ValueError):
        traceback.print_exc()
        opt = state.get_state()
    return opt


# helpers for maintaining the last result in case of an error
class DerivativeError(Exception):
    def __init__(self, name, result):
        super().__init__(f"infs and NaNs in {name}: {result}")

def _check_derivative(jac_or_hess, name):
    return lambda x: _check_derivative_at_x(jac_or_hess, name, x)

def _check_derivative_at_x(jac_or_hess, name, x):
    result = jac_or_hess(x)
    if not np.all(np.isfinite(result)):
        raise DerivativeError(name, result)
    return result

class _CallbackState():
    def __init__(self, x0):
        self._xk = x0
        self._nit = 0
    def get_state(self):
        return optimize.OptimizeResult(
            x=self._xk,
            success=False,
            message="Intermediate result",
            nit=self._nit
        )
    def _callback(self, xk):
        self._xk = xk
        self._nit += 1

    def callback(self):
        return lambda xk, *args: self._callback(xk)
