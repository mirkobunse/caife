import jax
import jax.numpy as jnp
import numpy as np
import traceback
from dataclasses import dataclass, field
from scipy import optimize
from . import AbstractSolver, Result, _jnp_softmax, _np_softmax, _rand_x0


@dataclass
class ScipyResult(Result):
    opt: object


@dataclass
class ScipySolver(AbstractSolver):
    """TODO: document."""
    solver: str = "trust-ncg"
    solver_options: dict[str,object] = field(default_factory=lambda: {
        "gtol": 1e-8,
        "maxiter": 1000,
    })
    seed: int | None = None

    def solve(self, nll, target_dim, nuisance_dim, n_samples):
        if nuisance_dim > 0: # if yes, nll has two arguments
            nll_ell = lambda ell: nll( # cast to a function of the latent variable ell
                n_samples * _jnp_softmax(ell[:(target_dim-1)]),
                jax.nn.sigmoid(ell[(target_dim-1):]),
            )
        else:
            nll_ell = lambda ell: nll(n_samples * _jnp_softmax(ell))
        x0 = np.concatenate((
            _rand_x0( # random starting point for target
                np.random.RandomState(self.seed),
                target_dim,
            ),
            np.zeros(nuisance_dim), # random starting point for nuisance parameters
        ))
        opt = minimize(nll_ell, x0, self.solver, self.solver_options)
        f_est = n_samples * _np_softmax(opt.x[:(target_dim-1)])
        nuisance_parameters = None
        if nuisance_dim > 0:
            nuisance_parameters = np.array(jax.nn.sigmoid(opt.x[(target_dim-1):]))
        return ScipyResult(f_est, nuisance_parameters, opt)


def minimize(loss_fn, x0, solver="trust-ncg", solver_options=None):
    """TODO document."""
    if solver_options is None:
        solver_options = { "gtol": 1e-8, "maxiter": 1000 }
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
