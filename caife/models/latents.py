"""Module containing latent-space parameterizations of model parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
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
        bin_groups (optional): A group number for each target bin, shape `(n_bins_target,)`. All bins of a group share one latent variable and, hence, have equal counts. The group with the smallest number, e.g., `-1` in `create_bin_groups`, has a fixed latent value of zero. Defaults to `None` for one group per bin.
    """
    n_samples: int
    n_bins_target: int
    bin_groups: ArrayLike | None = None

    def __post_init__(self):
        if self.bin_groups is not None:
            if len(self.bin_groups) != self.n_bins_target:
                raise ValueError("len(bin_groups) != n_bins_target")
            _, self._bin_group_map = np.unique(
                np.asarray(self.bin_groups),
                return_inverse=True,
            )

    def __call__(self, ell):
        exp_ell = jnp.concatenate((jnp.ones(1), jnp.exp(ell)))
        if self.bin_groups is not None: # one value per group -> one value per bin
            exp_ell = exp_ell[self._bin_group_map]
        return self.n_samples * (exp_ell / exp_ell.sum())

    @staticmethod
    def create_bin_groups(n_bins_target):
        return np.arange(n_bins_target) - 1 # first index is -1 = non-fittable

    def create_starting_point(self, rng=None):
        if self.bin_groups is None: # determine the number of classes
            n_classes = self.n_bins_target
        else:
            n_classes = self._bin_group_map.max() + 1

        # draw logits that are uniformly distributed after applying a softmax
        logits = jnp.log(rng.exponential(size=n_classes))
        return logits[1:] - logits[0] # fix the first dimension to zero


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
