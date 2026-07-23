"""Module containing solvers of unfolding equations."""

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


def flatten_result(tree):
    """TODO"""
    shapes = jax.tree.map(lambda x: np.array(x.shape), tree)
    leaves = jax.tree.leaves(jax.tree.map(lambda x: x.reshape(-1), tree))
    boundaries = np.concatenate(
        ([0], np.cumsum([len(x) for x in leaves]))
    )
    return np.concatenate(leaves), (shapes, boundaries)


def unflatten_result(resultdef, leaves_vec):
    """TODO"""
    shapes, boundaries = resultdef
    leaves = ( # split the single vector into its leaf components
        leaves_vec[boundaries[i]:boundaries[i+1]]
        for i in range(len(boundaries)-1)
    )
    tree = jax.tree.unflatten(jax.tree.structure(shapes), leaves)
    return jax.tree.map(lambda x, s: x.reshape(s), tree, shapes)


@dataclass
class AbstractSolver(ABC):
    """Abstract base class that all caife solvers inherit from.

    Args:
        nll: The negative log-likelihood function with the signature `nll(params, *args) -> float`, where `params` is a pytree containing vectors in their natural target space, e.g., count spectra or systematic paramenter vectors, and `args` is a tuple of fixed parameters of the function.
        latent_vectors: A JAX pytree of latent vectors. These vectors describe, for all model paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
        seed (optional): Random number generator seed. Defaults to `None`.

    Note:
        Concrete sub-classes of this abstract class can extend the `__post_init__` method to perform additional initialization steps like differentiating the objective function. So far, the `__post_init__` method already initializes `self.starting_vector_` and `self.latent_nll_`, which are to be used in implementations of the abstract method `solve_latent`.
    """
    nll: callable
    latent_vectors: any # a pytree
    seed: int | None = None

    def __post_init__(self):
        # create a random starting point for each latent variable
        rng = np.random.RandomState(self.seed)
        starting_points = jax.tree.map(
            lambda x: x.create_starting_point(rng),
            self.latent_vectors,
        )

        # extract the PyTree structure
        self.starting_vector_, self.resultdef_ = flatten_result(starting_points)

        # solve the Fredholm equation through minimizing the latent objective
        def latent_nll(ell, *args):
            ell = unflatten_result(self.resultdef_, ell)
            params = jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                self.latent_vectors,
                ell,
            )
            value = self.nll(params, *args)
            return value.squeeze() # ensure that a float is retured
        self.latent_nll_ = latent_nll

    def solve(self, args=(), return_aux=False):
        """Solve the unfolding problem.

        Args:
            args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
            return_aux (optional): Whether to return auxiliary information on the solution. This information could include, for instance, the number of iterations or the wall-clock time used for solving. Defaults to `False`.

        Returns:
            A result pytree with the same structure as the given `latent_vectors`. If `return_aux` is `True`, return a pair of this pytree and a `dict` of auxiliary information.
        """
        ell, aux = self.solve_latent(args)
        result = jax.tree.map( # map to latents to target spaces
            lambda latent_vector, ell: latent_vector(ell),
            self.latent_vectors,
            unflatten_result(self.resultdef_, ell),
        )
        if return_aux:
            return result, aux
        else:
            return result

    @abstractmethod
    def solve_latent(self, args):
        """Solve the unfolding problem in the corresponding latent space. This abstract method is meant to be implemented by concrete sub-classes of the `AbstractSolver` but is not meant to be called by a user directly.

        Args:
            args: A tuple of extra arguments that are passed to `latent_nll`.

        Returns:
            A pair `(ell, aux)` of a single combined latent solution vector `ell` and an auxiliary information object `aux`.

        Note:
            Implementations of this abstract method should minimize `self.latent_nll_` starting from `self.starting_vector_`. These class members represent the negative log-likelihood function that takes only a single combined latent vector as an argument and the single combined starting point in latent space.
        """
        pass
