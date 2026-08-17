"""Module containing solvers of unfolding equations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
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


def is_valid_hessian(hess):
    """Check whether a Hessian matrix is positive definite, and thereby represents a valid solution."""
    try: # https://stackoverflow.com/a/44287862/11567260 for a symmetric matrix
        np.linalg.cholesky(hess)
        return True
    except np.linalg.LinAlgError:
        return False


@dataclass
class AbstractSolver(ABC):
    """Abstract base class that all caife solvers inherit from.

    Args:
        nll: The negative log-likelihood function with the signature `nll(params, *args) -> float`, where `params` is a pytree containing vectors in their natural target space, e.g., count spectra or systematic paramenter vectors, and `args` is a tuple of fixed parameters of the function.
        latent_vectors: A JAX pytree of latent vectors. These vectors describe, for all model paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
        n_trials: The number of random trials, each with a new, random starting point. Defaults to `20`.
        seed (optional): Random number generator seed. Defaults to `None`.

    Note:
        Concrete sub-classes of this abstract class can extend the `__post_init__` method to perform additional initialization steps like differentiating the objective function. So far, the `__post_init__` method already initializes `self.latent_nll_`, which should be used in implementations of the abstract method `solve_latent`.
    """
    nll: callable
    latent_vectors: any # a pytree
    n_trials: int = 20
    seed: int | None = None

    def __post_init__(self):
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
            return value.squeeze() # ensure that a float is retured
        self.latent_nll_ = latent_nll

    def create_starting_vector(self):
        """TODO."""
        return jax.flatten_util.ravel_pytree(jax.tree.map(
            lambda x: x.create_starting_point(self._rng),
            self.latent_vectors,
        ))[0]

    def solve(self, args=(), return_aux=False):
        """Solve the unfolding problem.

        Args:
            args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
            return_aux (optional): Whether to return auxiliary information on the solution. This information could include, for instance, the number of iterations or the wall-clock time used for solving. Defaults to `False`.

        Returns:
            A result pytree with the same structure as the given `latent_vectors`. If `return_aux` is `True`, return a pair of this pytree and a `dict` of auxiliary information.
        """
        # create LatentResults
        latent_results = [self.solve_latent(args) for _ in range(self.n_trials)]

        # unravel and map these results to their target space
        def unravel_to_target_space(ell):
            return jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                self.latent_vectors,
                self.unravel_fn_(ell),
            )
        results = [unravel_to_target_space(r.ell) for r in latent_results]

        # find the best result
        result = results[np.argmin([r.value for r in latent_results])]
        if return_aux:
            aux = {
                "results": results,
                "latent_results": latent_results,
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
        pass
