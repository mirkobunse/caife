import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import ArrayLike

@dataclass
class Result:
    """Super-class for result objects that enables them to be treated like numpy arrays."""
    _f: ArrayLike

    def __array__(self, dtype=None, copy=None): # automatically cast to a numpy array
        return self._f.__array__(dtype, copy)

    def __getitem__(self, item): # allow slicing
        return self._f.__getitem__(item)

    def __str__(self): # logging sugar: a concise string representation
        return f"{self.__class__.__name__}({self._f})"


class AbstractSolver(ABC):
    """Abstract Base Class all caife Solvers inherit from."""

    @abstractmethod
    def solve(nll, target_dim, nuisance_dim, n_samples):
        """TODO: document."""
        pass # TODO: consider pytree-based signatures for nll and solve


# helper functions for our softmax "trick" with l[0]=0
def _jnp_softmax(l):
    exp_l = jnp.exp(l)
    return jnp.concatenate((jnp.ones(1), exp_l)) / (1. + exp_l.sum())

# same as above but in numpy instead of JAX
def _np_softmax(l):
    exp_l = np.exp(l)
    return np.concatenate((np.ones(1), exp_l)) / (1. + exp_l.sum())

# random starting points for the softmax "trick"
def _rand_x0(rng, n_classes):
    return rng.rand(n_classes-1) * 2 - 1
