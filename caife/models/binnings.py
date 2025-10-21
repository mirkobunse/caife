from abc import ABC
from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import LinearRegression

@dataclass
class Binning(ABC):
    """
        unsure what exactly is going on with regard to binning
        in ipynb values get digitized with geomspace before passing to LinearModel
        in LinearModel we calculate bins for digitized input with linspace
        how should our binning work? are both of those steps necessary?
        I assume: we can optimize one binning and then use it for the Matrix A? 
    """
    def fit(self, obs, targets=None):
        pass
    
    def digitize(self, vec):
        pass

@dataclass
class UnivariateBinning(Binning):
    target_bins: np.ndarray = None
    max_n_bins_proxy: int = 100
    n_cv: int = 10
    criterion: str = "dussap"
    preprocessor = field(default_factory=LinearRegression)

    def fit(self, obs, targets=None):
        # Need to optimize targets aswell
        if self.target_bins is None:
           assert targets is not None, "Need to specify targets if initialized with no target bins!"
           ... 
        # TODO optimize bins for observations
        self.obs_bins = ...

    # TODO
    def digitize(self, vec):
        ...
        
@dataclass
class TreeBinning(Binning):
    # TODO
    def fit(self, obs, targets=None):
        ...

    # TODO
    def digitize(self, vec):
        ...