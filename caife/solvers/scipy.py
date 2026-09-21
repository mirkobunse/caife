from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import jax
import numpy as np
from scipy import optimize

from . import AbstractSolver, LatentResult, is_valid_hessian, symmetrize_hessian_fn

"""Factory function, without arguments, to create solver options."""
SOLVER_OPTIONS_FACTORY = lambda: {
    "gtol": 1e-8,
    "maxiter": 1000,
}

_SECOND_ORDER_SOLVERS = [
    "newton-cg",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
    "trust-constr",
]


@dataclass
class ScipySolver(AbstractSolver):
    """A solver with a SciPy back-end.

    Args:
        nll: The negative log-likelihood function with the signature `nll(params, *args) -> float`, where `params` is a pytree containing vectors in their natural target space, e.g., count spectra or systematic paramenter vectors, and `args` is a tuple of fixed parameters of the function.
        latent_vectors: A JAX pytree of latent vectors. These vectors describe, for all model paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
        n_trials: The number of random trials, each with a new, random starting point. Defaults to `20`.
        seed (optional): Random number generator seed. Defaults to `None`.
        solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to "trust-ncg".
        solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `caife.solvers.scipy.SOLVER_OPTIONS_FACTORY()`.
        rtol (optional): The relative tolerance for validity checks. Defaults to `1e-8`.
    """
    solver: str = "trust-ncg"
    solver_options: dict[str,object] = field(default_factory=SOLVER_OPTIONS_FACTORY)
    rtol: float = 1e-8

    def __post_init__(self):
        AbstractSolver.__post_init__(self)

        # create all derivatives
        self.latent_nll_ = jax.jit(self.latent_nll_)
        self.latent_jac_ = jax.jit(jax.grad(self.latent_nll_))
        self.latent_hess_ = jax.jit(
            symmetrize_hessian_fn(jax.jacfwd(self.latent_jac_)))

    def solve_latent(self, args):
        opt = minimize(
            self.latent_nll_,
            self.create_starting_vector(),
            args=args,
            jac=self.latent_jac_,
            hess=self.latent_hess_,
            apply_jit=False, # already applied
            solver=self.solver,
            solver_options=self.solver_options,
        )
        hess = opt.get("hess", self.latent_hess_(opt.x, *args))
        is_valid, cond = is_valid_hessian(hess, rtol=self.rtol)
        return LatentResult(
            ell=opt.x,
            is_valid=is_valid,
            value=opt.get("fun", self.latent_nll_(opt.x, *args)),
            aux={"opt": opt, "cond": cond, "hess": hess},
        )


def minimize(
        loss_fn,
        x0,
        args=(),
        jac=None,
        hess=None,
        apply_jit=True,
        solver="trust-ncg",
        solver_options=None,
    ):
    """Minimize a loss function with a SciPy back-end.

    Args:
        loss_fn: The loss function to minimize.
        x0: The initial guess (i.e., starting point) for the minimization.
        args (optional): A tuple of extra arguments that are passed to `loss_fn`. Defaults to `()`.
        jac (optional): The Jacobian of the `loss_fn`. Defaults to `None`, triggering JAX differentiation of `loss_fn`.
        hess (optional): The Hessian of the `loss_fn`. Defaults to `None`, triggering JAX differentiation of `jac`.
        apply_jit (optional): Whether to JIT-compile `loss_fn` and its derivatives. If `loss_fn` and its derivatives are already JIT-compiled, this switch does not change the run-time. Defaults to `True`.
        solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to "trust-ncg".
        solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `None`, which is interpreted as `caife.solvers.scipy.SOLVER_OPTIONS_FACTORY()`.

    Returns:
        A `scipy.optimize.OptimizeResult`.
    """
    if solver_options is None:
        solver_options = SOLVER_OPTIONS_FACTORY()
    if apply_jit:
        loss_fn = jax.jit(loss_fn)
    if jac is None:
        jac = jax.grad(loss_fn) # Jacobian
    if apply_jit:
        jac = jax.jit(jac)
    if solver.lower() in _SECOND_ORDER_SOLVERS:
        if hess is None:
            hess = symmetrize_hessian_fn(jax.jacfwd(jac)) # Hessian through forward-mode AD
        if apply_jit:
            hess = jax.jit(hess)
    else:
        hess = None # first-order solvers must not use Hessians

    # error-robust optimization with a callback state
    state = _CallbackState(x0)
    try:
        opt = optimize.minimize(
            loss_fn,
            x0,
            jac=_check_derivative(jac, "jac"), # safe-guard derivatives
            hess=_check_derivative(hess, "hess"),
            args=args,
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
    if jac_or_hess is None:
        return None
    return lambda x, *args: _check_derivative_at_x(jac_or_hess, name, x, args)

def _check_derivative_at_x(jac_or_hess, name, x, args):
    result = jac_or_hess(x, *args)
    if not np.all(np.isfinite(result)):
        raise DerivativeError(name, result)
    return result

class _CallbackState:
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
