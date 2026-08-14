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
        # draw logits that are uniformly distributed after applying a softmax
        logits = jnp.log(rng.exponential(size=self.n_bins_target))
        return logits[:, 1:] - logits[:, [0]] # fix the first dimension to zero


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
