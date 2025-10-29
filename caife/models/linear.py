import numpy as np
import jax.numpy as jnp
from . import AbstractModel
from .binnings import AbstractBinning
from dataclasses import dataclass

@dataclass
class LinearModel(AbstractModel):
    """Linear Model for solving g = A @ f."""
    binning: AbstractBinning
    fit_binning: bool = True

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if self.fit_binning:
            self.binning.fit(X, y)

        # estimate the transfer matrix A from (X, y)
        X = self.binning.transform_proxy(X)
        y = self.binning.transform_target(y)
        A = np.bincount( # most efficient method, see caife.tests.benchmark_transfer
            self.binning.n_bins_target * X + y, # combined X*y bins
            weights=sample_weight,
            minlength=self.binning.n_bins_proxy * self.binning.n_bins_target,
        ).reshape((self.binning.n_bins_proxy, self.binning.n_bins_target))
        A = A / A.sum(axis=0, keepdims=True) # normalize
        self.A_ = jnp.array(A) # cast A to a JAX array to make __call__ differentiable

        # store the background distribution
        g_background = np.zeros(self.binning.n_bins_proxy)
        if background is not None:
            g_background = self.proxy_view(background)
        self.g_background_ = jnp.array(g_background)

        # ignore systematics for now
        # self.systematics_ = systematics

        return self # sklearn convention; allows method chaining

    def proxy_view(self, X):
        return np.bincount( # return a histogram of counts
            self.binning.transform_proxy(X),
            minlength=self.binning.n_bins_proxy,
        )

    def __call__(self, f):
        g_pred = self.A_ @ f + self.g_background_
        return g_pred
