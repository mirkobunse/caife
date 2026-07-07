"""Module containing solvers of unfolding equations."""

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import ArrayLike


def flatten_result(tree):
    """TODO"""
    leaves, treedef = jax.tree.flatten(tree)
    boundaries = np.concatenate( # TODO generalize for multi-dimensional leaves
        ([0], np.cumsum([len(x) for x in leaves]))
    )
    return np.concatenate(leaves), (treedef, boundaries)


def unflatten_result(resultdef, leaves_vec):
    """TODO"""
    treedef, boundaries = resultdef
    leaves = ( # split the single vector into its leaf components
        leaves_vec[boundaries[i]:boundaries[i+1]]
        for i in range(len(boundaries)-1)
    )
    return jax.tree.unflatten(treedef, leaves)


@dataclass
class AbstractSolver(ABC):
    """Abstract base class that all caife solvers inherit from.

    Args:
        seed: Random number generator seed. Defaults to `None`.
    """
    seed: int | None = None

    def solve(self, nll, latent_vectors, return_aux=False):
        """Solve the unfolding problem in the way it is represented by a negative log-likelihood and a sequence of latent vectors.

        Args:
            nll: The negative log-likelihood function that takes as many input arguments as there are latent vectors. Each input argument has to be a vector in its natural target space, e.g., a count spectrum or a systematic paramenter vector.
            latent_vectors: A JAX pytree of latent vectors, which describe, for all spectra and nuisance paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.
            return_aux (optional): Whether to return auxiliary information on the solution. This information could include, for instance, the number of iterations or the wall-clock time used for solving. Defaults to `False`.

        Returns:
            A tuple of results with elements that correspond to the given `latent_vectors`. If `return_aux` is `True`, return a pair of this tuple and a `dict` of auxiliary information.
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
        def latent_nll(ell):
            ell = unflatten_result(resultdef, ell)
            target_vectors = jax.tree.map( # map to latents to target spaces
                lambda latent_vector, ell: latent_vector(ell),
                latent_vectors,
                ell,
            )
            return nll(target_vectors)
        ell, aux = self.solve_latent(latent_nll, starting_vector)
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
    def solve_latent(self, latent_nll, x0):
        """Solve the unfolding problem in the way it is represented by an objective function that takes a single combined latent vector as an argument. This abstract method is meant to be implemented by specific sub-classes of the `AbstractSolver` but is not meant to be called by a user directly.

        Args:
            latent_nll: The negative log-likelihood function that takes only a single combined latent vector as an argument.
            x0: The single combined starting point in latent space.

        Returns:
            A pair `(ell, aux)` of a single combined latent solution vector `ell` and an auxiliary information object `aux`.
        """
        pass


class AbstractLatentVector(ABC):
    """Abstract base class for the latent variables over which caife solvers minimize negative log-likelihoods."""

    @abstractmethod
    def __call__(self, ell):
        """Transform a latent variable to the target space.

        Args:
            ell: The vector-valued latent variable, shape (latent_dim,).

        Returns:
            A vector in the target space, shape (target_dim,).
        """
        pass

    @abstractmethod
    def create_starting_point(self, rng=None):
        """Create a random starting point in latent space.

        Args:
            rng (optional): Random number generator.

        Returns:
            An initial vector in the latent space, shape (latent_dim,)."""
        pass


@dataclass
class LatentSpectrum(AbstractLatentVector):
    """Transforms a vector of latent variables through the "soft-max trick" by Bunse (2022) and scales it by the given number of samples. Through this design choice, the output will always be a valid count spectrum with all bin-wise counts larger than zero and the sum of all bin counts equal to the given number of samples.

    Args:
        n_samples: The desired number of samples.
        n_bins_target: The number of target bins.
    """
    n_samples: int
    n_bins_target: int

    def __call__(self, ell):
        exp_ell = jnp.exp(ell)
        p_est = jnp.concatenate((jnp.ones(1), exp_ell)) / (1. + exp_ell.sum())
        return self.n_samples * p_est

    def create_starting_point(self, rng=None):
        return jnp.array(rng.rand(self.n_bins_target-1) * 2 - 1)


@dataclass
class LatentSystematics(AbstractLatentVector):
    """Transforms a latent variable through a sigmoid that is scaled to match the given bounds. Through this design choice, the value of the systematic parameter will always be within the given bounds.

    Args:
        bounds: The minimum and maximum value for each systematic parameter, shape (n_systematic_parameters, 2).
    """
    bounds: ArrayLike

    def __call__(self, ell):
        return self.bounds[:,0] + (self.bounds[:,1] - self.bounds[:,0]) * jax.nn.sigmoid(ell)

    def create_starting_point(self, rng=None):
        return jnp.zeros(self.bounds.shape[0])
