from abc import ABC
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Binning(ABC):
    """Base class that all binnings in caife inherit from."""
    target_bins: np.ndarray = None
    max_n_bins_proxy: int = 100

    def fit(self, X, y):
        """TODO: add documentation"""
        pass

    def transform_target(self, y):
        """TODO: add documentation"""
        pass

    def transform_proxy(self, X):
        """TODO: add documentation"""
        pass

@dataclass
class UnivariateBinning(Binning):
    n_cv: int = 10
    criterion: str = "dussap"
    preprocessor: object | None = None

    def fit(self, X, y):
        # Need to optimize targets aswell
        if self.target_bins is None:
           assert targets is not None, "Need to specify targets if initialized with no target bins!"
           ... 
        # TODO optimize bins for observations
        self.obs_bins = ...
        return self

    def transform_target(self, y):
        return self._transform(y, self.target_bins) # TODO: don't apply preprocessor

    def transform_proxy(self, X):
        return self._transform(X, self.obs_bins)

    def _transform(self, X, bins):
        """internally work with 2D arrays for now: e.g. (n_samples, 1) instead of (n_samples,)"""
        preprocessed_X = self._preprocess(X)
        if preprocessed_X.ndim == 2:
            return np.digitize(preprocessed_X, bins)
        if preprocessed_X.ndim == 3:
            digitized_vec = np.zeros_like(preprocessed_X)
            for batch in range(preprocessed_X.shape[0]):
                digitized_vec[batch, :, :] = np.digitize(preprocessed_vec[batch, :, :], bins)
            return digitized_vec

    def _preprocess(self, X):
        """Helper function to assure that vector has shape (n_samples, 1) or *optionally* (batch_size, n_samples, 1)"""
        preprocessed_X = X
        if X.ndim == 1:
            preprocessed_X = np.expand_dims(X, 1)
        elif X.ndim == 2:
            if X.shape[1] == self.preprocessor.n_features_in_:
                preprocessed_X = self.preprocessor.predict(X).reshape((-1, 1))
            elif X.shape[1] != 1:
                raise ValueError(
                    (
                        "Invalid input shape! Vector can have shape: "
                        f"(n_samples,), (n_samples, 1) or (n_samples, {self.preprocessor.n_features_in_}). "
                        f"Got shape (n_samples, {X.shape[1]}) instead."
                    )
                )
        elif X.ndim == 3: 
            """optional: batched data. For now batch_first assumed (batch, n_samples, n_features).
               might be useful for different configurations? """
            batch_size, n_samples, n_features = X.shape
            if n_features == self.preprocessor.n_features_in_:
                preprocessed_X = np.zeros((batch_size, n_samples, 1))
                for batch in range(batch_size):
                    preprocessed_X[batch, :, :] = self.preprocessor.predict(X[batch, :, :]).reshape((n_samples, 1))
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
            raise ValueError(f"Invalid input dim! Maximum valid vector dim is 3, got {X.ndim} instead.")

        return preprocessed_X

        
@dataclass
class TreeBinning(Binning):
    # TODO
    def fit(self, X, y):
        ...

    # TODO
    def transform_proxy(self, X):
        ...
