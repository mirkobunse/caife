from abc import ABC
import numpy as np
from dataclasses import dataclass
from binnings import Binning

class Model(ABC):
    """Abstract Base Class all Linear Caife Models inherit from"""
    def fit(self, g_train, f_train, sample_weight, systematics, background=None):
        pass

    def predict(self, f):
        pass

@dataclass
class LinearModel(Model):
    """Linear Model for solving g = A @ f"""
    binning: Binning

    def fit(self, g_train, f_train, sample_weight, systematics, background=None):
        if not self.binning.fitted:
            binning_target = None
            if self.binning.target_bins is None:
                binning_target = f_train
            self.binning.fit(g_train, binning_target)
        self.systematics = systematics
        self.background = background
        self.A = np.histogram2d(
            x=g_train,
            y=f_train,
            bins=(self.binning.obs_bins, self.binning.target_bins),
            weights=sample_weight
        )[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = self.A @ M_norm
        
    def __call__(self, f):
        dig_tar = self.binning.transform_targets(f)
        g_pred = self.A @ dig_tar
        if self.background is not None:
            g_pred += self.background
        return g_pred