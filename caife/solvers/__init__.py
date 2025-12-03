import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import ArrayLike

@dataclass
class Result:
    """A result object containing the spectrum, nuisance parameter values, and auxiliary information. Behaves as if it was just the spectrum if cast to a numpy or JAX array.

    Args:
        values: Components of the solution, with `values[0]` being the target spectrum and `values[i] for i>0` optionally representing systematic parameter values.
        aux (optional): A dict of auxiliary information on the result, e.g., how the result was obtained numerically. The contents of this dict differ between solvers.
    """
    values: list[ArrayLike]
    aux: dict[str,object] | None = None

    def __array__(self, dtype=None, copy=None): # enable automatic casting to a numpy array
        return np.array(self.values[0], dtype=dtype)

    def __jax_array__(self): # enable automatic casting to a JAX array
        return jnp.array(self.values[0])

    def __getitem__(self, item): # allow slicing
        return self.values[0].__getitem__(item)

    def __str__(self): # logging sugar: a concise string representation
        values = ", ".join([ str(v) for v in self.values ])
        return f"{self.__class__.__name__}({values})"

    def zip(self, *values):
        """Concatenate solution components into a single vector or split a single vector into its components.

        Args:
            *values: The solution components or, if len(values)==1, a single concatenated solution vector. If len(values)==0, take the values of this result as an input.

        Returns:
            A tuple of solution components if len(values)==1 or else a single concatenated solution vector.
        """
        if len(values) == 1: # split a single vector into its components
            target_dims = [ len(x) for x in self.values ]
            boundaries = np.concatenate(([0], np.cumsum(target_dims)))
            return (
                values[0][boundaries[i]:boundaries[i+1]]
                for i in range(len(boundaries)-1)
            )
        elif len(values) == 0:
            values = self.values
        return jnp.concatenate(values) # concatenate components into a single vector


@dataclass
class AbstractSolver(ABC):
    """Abstract Base Class all caife Solvers inherit from."""
    seed: int | None = None

    def solve(self, nll, *latent_vectors):
        """Solve the unfolding problem in the way it is represented by a negative log-likelihood and a sequence of latent vectors.

        Args:
            nll: The negative log-likelihood function that takes as many input arguments as there are latent vectors. Each input argument has to be a vector in its natural target space, e.g., a count spectrum or a systematic paramenter vector.
            *latent_vectors: The latent vectors, which describe, for all spectra and nuisance paramenters, the mapping between their latent and natural target spaces as well as the generation of their starting points in latent space.

        Returns:
            A `caife.solvers.Result` to the unfolding problem.
        """
        # create a random starting point for each latent variable
        rng = np.random.RandomState(self.seed)
        starting_points = [ x.create_starting_point(rng) for x in latent_vectors ]

        # determine where to split a concatenation of all latent variables
        latent_dims = [ len(x) for x in starting_points ]
        boundaries = np.concatenate(([0], np.cumsum(latent_dims)))

        # split latents and project each latent to its target space
        target_mapping_fn = lambda ell: [
            latent_vectors[i](ell[boundaries[i]:boundaries[i+1]])
            for i in range(len(boundaries)-1)
        ]

        # solve the Fredholm equation through minimizing the latent objective
        ell, aux = self.solve_latent(
            lambda ell: nll(*target_mapping_fn(ell)),
            jnp.concatenate(starting_points),
        )
        return Result(target_mapping_fn(ell), aux) # map ell to the target spaces

    @abstractmethod
    def solve_latent(self, latent_nll, x0):
        """Solve the unfolding problem in the way it is represented by an objective function that takes a single combined latent vector as an argument. This abstract method is meant to be implemented by specific sub-classes of the `AbstractSolver` but is not meant to be called by a user directly.

        Args:
            latent_nll: The negative log-likelihood function that takes only a single combined latent vector as an argument.
            x0: The single combined starting point in latent space.

        Returns:
            A pair of a single combined latent solution vector and an optional dict of auxiliary information on the result.
        """
        pass


class AbstractLatentVector(ABC):
    """Abstract base class for the latent variables with which caife solvers minimize negative log-likelihoods."""

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
    """TODO document: soft-max "trick"."""
    n_samples: int
    n_bins_target: int

    def __call__(self, ell):
        exp_ell = jnp.exp(ell)
        p_est = jnp.concatenate((jnp.ones(1), exp_ell)) / (1. + exp_ell.sum())
        return self.n_samples * p_est

    def create_starting_point(self, rng=None):
        return rng.rand(self.n_bins_target-1) * 2 - 1


@dataclass
class LatentSystematics(AbstractLatentVector):
    """TODO document.

    Args:
        bounds: The minimum and maximum value for each systematic parameter, shape (n_systematic_parameters, 2).
    """
    bounds: ArrayLike

    def __call__(self, ell):
        return self.bounds[:,0] + (self.bounds[:,1] - self.bounds[:,0]) * jax.nn.sigmoid(ell)

    def create_starting_point(self, rng=None):
        return jnp.zeros(self.bounds.shape[0])
