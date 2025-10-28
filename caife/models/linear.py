from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from models.binnings import AbstractBinning
import jax.numpy as jnp

# TODO move this class to caife.models
class AbstractModel(ABC):
    """Abstract Base Class all caife Models inherit from."""

    @abstractmethod
    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        """TODO: add documentation"""
        pass

    @abstractmethod
    def __call__(self, f):
        """Apply this model to a candidate spectrum.

        Args:
            f: A candidate spectrum, shape `(n_target_bins,)`.

        Returns:
            The predicted proxy distribution `g`, shape `(n_proxy_bins,)`, as modeled for `f`.
        """
        pass

@dataclass
class LinearModel(AbstractModel):
    """Linear Model for solving g = A @ f."""
    binning: AbstractBinning
    fit_binning: bool = True

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if self.fit_binning:
            self.binning.fit(X, y)

        # estimate the transfer matrix A from (X, y)        
        A = np.histogram2d(
            x=self.binning.transform_proxy(X),
            y=self.binning.transform_target(y),
            bins=(self.binning.proxy_bins, self.binning.target_bins),
            weights=sample_weight,
        )[0]

        M_norm = np.diag(1 / np.sum(A, axis=0))
        A = A @ M_norm
        self.A_ = jnp.array(A) # cast A to a JAX array to make __call__ differentiable

        # store the background distribution
        g_background = np.zeros(self.binning.n_bins_proxy)
        if background is not None:
            g_background = np.histogram(
                self.binning.transform_proxy(background),
                bins=self.binning.proxy_bins,
                density=True,
            )[0]
        self.g_background_ = jnp.array(g_background)

        self.systematics_ = systematics # TODO ignore systematics for now
        return self # sklearn convention; allows method chaining

    def __call__(self, f):
        g_pred = self.A_ @ f + self.g_background_
        return g_pred
