"""Module containing solvers of unfolding equations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import compress
from typing import NamedTuple

import jax
import jax.flatten_util
import numpy as np


class LatentResult(NamedTuple):
    """A result in latent space, with extra information.

    Args:
        ell: The result in latent space, shape (n_free_parameters,)
        is_valid: Whether the result is valid; invalid results will be excluded.
        value: The loss function value associated with the result; smaller is better.
        aux: A dict of any auxiliary information.
    """
    ell: list
    is_valid: bool
    value: float
    aux: dict


def symmetrize_hessian_fn(hess_fn):
    """Alter a Hessian function to produce a symmetrized version of the Hessian.

    Args:
        hess_fn: The function computing the Hessian from arbitrary arguments.

    Returns:
        Another function that computes a symmetrized version of the same Hessian."""
    def symmetric_hess_fn(*args, **kwargs):
        hess = hess_fn(*args, **kwargs)
        return .5 * (hess + hess.T)
    return symmetric_hess_fn


def is_valid_hessian(hess, rtol=1e-8):
    """Check whether a Hessian matrix is positive definite, and thereby represents a valid solution. In addition, this check computes the condition number to allow for flagging near-invalid solutions.

    Args:
        hess: The Hessian matrix at the solution, shape (n_parameters, n_parameters).
        rtol (optional): The tolerance, relative to the largest absolute eigenvalue. This tolerance identifies a near-zero smallest eigenvalue as being compatible with zero, so that the check for positive definiteness is robust against a limited machine precision. Defaults to `1e-8`.

    Returns:
        A tuple `(is_valid, cond)` consisting of a validity boolean and the condition number.
    """
    hess = .5 * (hess + hess.T) # symmetrize to reduce autodiff noise
    eigvals = np.linalg.eigvalsh(hess) # eigenvalues in ascending order
    atol = rtol * np.max(np.abs(eigvals)) # some absolute tolerance
    is_valid = eigvals[0] > atol # the actual check for positive definiteness
    cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else np.inf # extra information
    return is_valid, cond


@dataclass
class AbstractSolver(ABC):
    """Abstract base class that all caife solvers inherit from.

    Args:
        nll: The negative log-likelihood function with the signature `nll(params, *args) -> float`, where `params` is a pytree containing vectors in their natural target space, e.g., count spectra or systematic parameter vectors, and `args` is a tuple of fixed parameters of the function.
        latent_vectors: A JAX pytree of latent vectors. These vectors describe, for all model parameters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
        min_trials (optional): The minimum number of random trials, each with a new, random starting point. Defaults to `1`.
        max_trials (optional): The maximum number of random trials. No further trials are started once this number is reached, even if `min_valid_trials` is not yet met. Defaults to `100`.
        min_valid_trials (optional): The minimum number of trials that need to produce a valid result. Further trials are started until this number is reached, unless `max_trials` is reached first. Defaults to `1`.
        seed (optional): Random number generator seed. Defaults to `None`.

    Note:
        Concrete sub-classes of this abstract class can extend the `__post_init__` method to perform additional initialization steps like differentiating the objective function. So far, the `__post_init__` method already initializes `self.latent_nll_`, which should be used in implementations of the abstract method `solve_latent`.
    """
    nll: callable
    latent_vectors: any # a pytree
    min_trials: int = 1
    max_trials: int = 100
    min_valid_trials: int = 1
    seed: int | None = None

    def __post_init__(self):
        if self.max_trials < 1:
            raise ValueError(f"max_trials={self.max_trials} must be at least 1")
        if self.min_trials > self.max_trials:
            raise ValueError(f"min_trials={self.min_trials} exceeds max_trials={self.max_trials}")
        if self.min_valid_trials > self.max_trials:
            raise ValueError(f"min_valid_trials={self.min_valid_trials} exceeds max_trials={self.max_trials}")
        self._rng = np.random.RandomState(self.seed)

        # extract the PyTree structure
        self.unravel_fn_ = jax.flatten_util.ravel_pytree(jax.tree.map(
            lambda x: x.create_starting_point(np.random.RandomState(0)),
            self.latent_vectors,
        ))[1]

        # solve the Fredholm equation through minimizing the latent objective
        def latent_nll(ell, *args):
            ell = self.unravel_fn_(ell)
            params = jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                self.latent_vectors,
                ell,
            )
            value = self.nll(params, *args)
            return value.squeeze() # ensure that a float is returned
        self.latent_nll_ = latent_nll

    def create_starting_vector(self):
        """Create a starting vector for a new random trial, in latent space.

        Returns:
            A single vector `ell` that concatenates a random starting point from each latent vector in `self.latent_vectors`, generated using `self._rng`. Implementations of `solve_latent` should minimize `self.latent_nll_` starting from this vector.
        """
        return jax.flatten_util.ravel_pytree(jax.tree.map(
            lambda x: x.create_starting_point(self._rng),
            self.latent_vectors,
        ))[0]

    def solve(self, args=(), return_aux=False):
        """Solve the unfolding problem.

        Random trials are started until at least `min_trials` trials have been executed and at least `min_valid_trials` of them have produced a valid result, but no more than `max_trials` trials are started. The best valid result among all trials is returned.

        Args:
            args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
            return_aux (optional): Whether to return auxiliary information on the solution. This information could include, for instance, the number of iterations or the wall-clock time used for solving. Defaults to `False`.

        Returns:
            A result pytree with the same structure as the given `latent_vectors` or `None` if none of the trials produced a valid result. If `return_aux` is `True`, return a pair of this result and a `dict` of auxiliary information.
        """
        # create LatentResults until the minimum numbers of trials and valid trials are met
        latent_results = []
        n_valid = 0
        while len(latent_results) < self.max_trials and (
                len(latent_results) < self.min_trials or n_valid < self.min_valid_trials):
            latent_result = self.solve_latent(args)
            latent_results.append(latent_result)
            n_valid += latent_result.is_valid # cast bool to 0 or 1

        # unravel and map these results to their target space
        def unravel_to_target_space(ell):
            return jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                self.latent_vectors,
                self.unravel_fn_(ell),
            )
        results = [unravel_to_target_space(l.ell) for l in latent_results]

        # find the best valid result
        is_valid = [l.is_valid for l in latent_results]
        result = None
        aux = {}
        if np.sum(is_valid) > 0:
            i_best = np.argmin([l.value for l in compress(latent_results, is_valid)])
            result = list(compress(results, is_valid))[i_best]
            ell = list(compress(latent_results, is_valid))[i_best].ell
            aux = list(compress(latent_results, is_valid))[i_best].aux | {"ell": ell}
        if return_aux:
            aux = aux | { # update aux
                "results": results,
                "latent_results": latent_results,
                "unravel_fn": unravel_to_target_space,
            }
            return result, aux
        else:
            return result

    @abstractmethod
    def solve_latent(self, args):
        """Solve the unfolding problem in the corresponding latent space. This abstract method is meant to be implemented by concrete sub-classes of the `AbstractSolver` but is not meant to be called by a user directly.

        Args:
            args: A tuple of extra arguments that are passed to `latent_nll`.

        Returns:
            A `LatentResult`.

        Note:
            Implementations of this abstract method should minimize `self.latent_nll_` starting from `self.create_starting_vector()`. These class members represent the negative log-likelihood function that takes only a single combined latent vector as an argument and the single combined starting point in latent space.
        """
