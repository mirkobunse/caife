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

    # TODO is this what we want to do here?
    def digitize(self, vec):
        """internally work with 2D arrays for now: e.g. (n_samples, 1) instead of (n_samples,)"""
        preprocessed_vec = self._preprocess(vec)
        if preprocessed_vec.ndim == 2:
            return np.digitize(preprocessed_vec, self.obs_bins)
        if preprocessed_vec.ndim == 3:
            digitized_vec = np.zeros_like(preprocessed_vec)
            for batch in range(preprocessed_vec.shape[0]):
                digitized_vec[batch, :, :] = np.digitize(preprocessed_vec[batch, :, :], self.obs_bins)
            return digitized_vec
        
    
    def _preprocess(self, vec):
        """Helper function to assure that vector has shape (n_samples, 1) or *optionally* (batch_size, n_samples, 1)"""
        preprocessed_vec = vec
        if vec.ndim == 1:
            preprocessed_vec = np.expand_dims(vec, 1)
        elif vec.ndim == 2:
            if vec.shape[1] == self.preprocessor.n_features_in_:
                preprocessed_vec = self.preprocessor.predict(vec).reshape((-1, 1))
            elif vec.shape[1] != 1:
                raise ValueError(
                    (
                        "Invalid input shape! Vector can have shape: "
                        f"(n_samples,), (n_samples, 1) or (n_samples, {self.preprocessor.n_features_in_}). "
                        f"Got shape (n_samples, {vec.shape[1]}) instead."
                    )
                )
        elif vec.ndim == 3: 
            """optional: batched data. For now batch_first assumed (batch, n_samples, n_features).
               might be useful for different configurations? """
            batch_size, n_samples, n_features = vec.shape
            if n_features == self.preprocessor.n_features_in_:
                preprocessed_vec = np.zeros((batch_size, n_samples, 1))
                for batch in range(batch_size):
                    preprocessed_vec[batch, :, :] = self.preprocessor.predict(vec[batch, :, :]).reshape((n_samples, 1))
            elif n_features != 1:
                raise ValueError(
                    (
                        "Invalid input shape! Vector can have shape: "
                        "(batch_size, n_samples,), (batch_size, n_samples, 1) "
                        f"or (batch_size, n_samples, {self.preprocessor.n_features_in_}). "
                        f"Got shape (batch_size, n_samples, {n_features}) instead."
                    )
                )
        else:    
            raise ValueError(f"Invalid input dim! Maximum valid vector dim is 3, got {vec.ndim} instead.")
        
        return preprocessed_vec

        
@dataclass
class TreeBinning(Binning):
    # TODO
    def fit(self, obs, targets=None):
        ...

    # TODO
    def digitize(self, vec):
        ...