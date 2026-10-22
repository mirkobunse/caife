"""Module containing latent-space parameterizations of model parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


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

    @abstractmethod
    def create_starting_point(self, rng=None):
        """Create a random starting point in latent space.

        Args:
            rng (optional): Random number generator.

        Returns:
            An initial vector in the latent space, shape (latent_dim,).
        """


@dataclass
class LatentSpectrum(AbstractLatentVector):
    """Transforms a vector of latent variables through the "soft-max trick" by Bunse (2022) and scales it by the given number of samples. Through this design choice, the output will always be a valid count spectrum with all bin-wise counts larger than zero and the sum of all bin counts equal to the given number of samples.

    Args:
        n_samples: The desired number of samples.
        n_bins_target: The number of target bins.
    """
    n_samples: int
    n_bins_target: int
    sharing_indices: ArrayLike | None = None

    def __call__(self, ell):
        exp_ell = jnp.concatenate((jnp.ones(1), jnp.exp(ell)))
        if self.sharing_indices is not None:
            exp_ell = exp_ell[self.sharing_indices + 1]
        return self.n_samples * (exp_ell / exp_ell.sum())

    @staticmethod
    def parameter_indices(n_bins_target):
        return jnp.arange(n_bins_target) - 1 # first index is -1 = non-fittable

    def create_starting_point(self, rng=None):
        # draw logits that are uniformly distributed after applying a softmax
        logits = jnp.log(rng.exponential(size=self.n_bins_target))
        starting_point = logits[1:] - logits[0] # fix the first dimension to zero
        if self.sharing_indices is not None:
            starting_point = starting_point[jnp.unique(
                self.sharing_indices[self.sharing_indices >= 0])]
        return starting_point


@dataclass
class LatentReshape(AbstractLatentVector):
    """Reshapes a latent vector.

    Args:
        latent_vector: The latent vector to reshape.
        shape: The desired shape.
    """
    latent_vector: AbstractLatentVector
    shape: list[int]

    def __call__(self, ell):
        return self.latent_vector(ell).reshape(self.shape)

    def create_starting_point(self, rng=None):
        return self.latent_vector.create_starting_point(rng)


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
        # draw logits, each uniformly distributed after applying a sigmoid
        uniform = rng.uniform(size=self.bounds.shape[0])
        return jnp.log(uniform / (1 - uniform))
