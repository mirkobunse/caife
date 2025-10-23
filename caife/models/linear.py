from abc import ABC
import numpy as np
from dataclasses import dataclass
from binnings import Binning

class Model(ABC):
    """Abstract Base Class all caife Models inherit from."""
    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        """TODO: add documentation"""
        pass

    def __call__(self, f):
        """Apply this model to a candidate spectrum.

        Args:
            f: A candidate spectrum, shape `(n_target_bins,)`.

        Returns:
            The predicted proxy distribution `g`, shape `(n_proxy_bins,)`, as modeled for `f`.
        """
        pass

@dataclass
class LinearModel(Model):
    """Linear Model for solving g = A @ f."""
    binning: Binning
    fit_binning: bool = False

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if self.fit_binning:
            binning_target = None
            if self.binning.target_bins is None:
                binning_target = y
            self.binning.fit(X, binning_target)
        self.systematics = systematics
        self.g_background = None
        if background is not None:
            self.g_background = np.histogram( # TODO: use transform_proxy
                background,
                bins=self.binning.obs_bins,
            )
        self.A = np.histogram2d( # TODO: fix this for multi-dimensional X by using transform_proxy and transform_target
            x=X,
            y=y,
            bins=(self.binning.obs_bins, self.binning.target_bins),
            weights=sample_weight
        )[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = self.A @ M_norm
        return self # sklearn convention; allows method chaining

    def __call__(self, f):
        g_pred = self.A @ f
        if self.g_background is not None:
            g_pred += self.g_background
        return g_pred
