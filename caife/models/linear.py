from abc import ABC
import numpy as np

class Model(ABC):
    """Abstract Base Class all Linear Caife Models inherit from"""
    def fit(self, g_train, f_train, sample_weight, systematics, background=None):
        pass

    def predict(self, g):
        pass

class LinearModel(Model):
    """Linear Model for solving g = A @ f"""
    def __init__(self, binning):
        super().__init__()
        self.binning = binning

    def fit(self, g_train, f_train, sample_weight, systematics, background=None):
        # digitize inputs somehow
        digitized_obs = self.binning.digitize(g_train)
        digitized_truth = self.binning.digitize(f_train)
        self.systematics = systematics
        self.background = background
        self.A = np.histogram2d(
            x=digitized_obs,
            y=digitized_truth,
            bins=(self.binning.obs_bins, self.binning.target_bins), # can we take our optim bins here?
            weights=sample_weight
        )[0]
        M_norm = np.diag(1 / np.sum(self.A, axis=0))
        self.A = self.A @ M_norm
        ...

    def predict(self, g):
        digitized_obs = self.binning.digitize(g)
        ...