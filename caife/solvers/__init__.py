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
        seed: Random number generator seed. Defaults to `None`.
    """
    seed: int | None = None

    def solve(self, nll, latent_vectors, args=(), return_aux=False):
        """Solve the unfolding problem in the way it is represented by a negative log-likelihood and a sequence of latent vectors.

        Args:
            nll: The negative log-likelihood function with the signature `nll(params, *args) -> float`, where `params` is a pytree containing vectors in their natural target space, e.g., count spectra or systematic paramenter vectors, and `args` is a tuple of fixed parameters of the function.
            latent_vectors: A JAX pytree of latent vectors, which describe, for all spectra and nuisance paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
            args (optional): A tuple of extra arguments that are passed to `nll`. Defaults to `()`.
            return_aux (optional): Whether to return auxiliary information on the solution. This information could include, for instance, the number of iterations or the wall-clock time used for solving. Defaults to `False`.

        Returns:
            A result pytree with the same structure as the given `latent_vectors`. If `return_aux` is `True`, return a pair of this pytree and a `dict` of auxiliary information.
        """
        # create a random starting point for each latent variable
        rng = np.random.RandomState(self.seed)
        starting_points = jax.tree.map(
            lambda x: x.create_starting_point(rng),
            latent_vectors,
        )

        # extract the PyTree structure
        starting_vector, resultdef = flatten_result(starting_points)

        # solve the Fredholm equation through minimizing the latent objective
        def latent_nll(ell, *args):
            ell = unflatten_result(resultdef, ell)
            params = jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                latent_vectors,
                ell,
            )
            value = nll(params, *args)
            return value.squeeze() # ensure that a float is retured
        ell, aux = self.solve_latent(latent_nll, starting_vector, args)
        result = jax.tree.map( # map to latents to target spaces
            lambda latent_vector, ell: latent_vector(ell),
            latent_vectors,
            unflatten_result(resultdef, ell),
        )
        if return_aux:
            return result, aux
        else:
            return result

    @abstractmethod
    def solve_latent(self, latent_nll, x0, args):
        """Solve the unfolding problem in the way it is represented by an objective function that takes a single combined latent vector as an argument. This abstract method is meant to be implemented by specific sub-classes of the `AbstractSolver` but is not meant to be called by a user directly.

        Args:
            latent_nll: The negative log-likelihood function that takes only a single combined latent vector as an argument.
            x0: The single combined starting point in latent space.
            args: A tuple of extra arguments that are passed to `latent_nll`.

        Returns:
            A pair `(ell, aux)` of a single combined latent solution vector `ell` and an auxiliary information object `aux`.
        """
        pass
