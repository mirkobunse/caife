from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from binnings import Binning

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
    binning: Binning
    fit_binning: bool = True

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if self.fit_binning:
            binning_target = None
            if self.binning.target_bins is None:
                binning_target = y
            self.binning.fit(X, binning_target)
        self.systematics = systematics

        # estimate the transfer matrix A from (X, y)
        A = np.histogram2d( # TODO: fix this for multi-dimensional X by using transform_proxy and transform_target
            x=X,
            y=y,
            bins=(self.binning.obs_bins, self.binning.target_bins),
            weights=sample_weight
        )[0]
        M_norm = np.diag(1 / np.sum(A, axis=0))
        A = A @ M_norm
        self.A = jnp.array(A) # cast A to a JAX array to make __call__ differentiable

        # store the background distribution
        g_background = np.zeros(len(self.binning.n_bins_proxy))
        if background is not None:
            g_background = np.histogram( # TODO: use transform_proxy
                background,
                bins=self.binning.obs_bins,
            )
        self.g_background = jnp.array(g_background)

        return self # sklearn convention; allows method chaining

    def __call__(self, f):
        g_pred = self.A @ f += self.g_background
        return g_pred
