import numpy as np
import jax.numpy as jnp
from . import AbstractModel
from dataclasses import dataclass
from qunfold import AbstractRepresentation

@dataclass
class LinearModel(AbstractModel):
    """Linear Model for solving g = A @ f.

    Args:
        target_bins: Bin boundaries of the target quantity, shape (n_target_bins+1,).
        representation: The data representation that is computed from the proxy features.
    """
    target_bins: list[float]
    representation: AbstractRepresentation

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        y = self.target_view(y)
        A = self.representation.fit_transform(X, y, n_classes=self.n_bins_target)
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
        return self.representation.transform(X)

    def target_view(self, y):
        return np.digitize(y, self.target_bins, right=True)

    def __call__(self, f):
        g_pred = self.A_ @ f + self.g_background_
        return g_pred

    @property
    def n_bins_target(self):
        return len(self.target_bins) - 1
