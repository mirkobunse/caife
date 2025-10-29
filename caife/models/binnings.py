import itertools
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable

@dataclass
class AbstractBinning(ABC):
    """Base class that all binnings in caife inherit from."""

    @abstractmethod
    def fit(self, X, y):
        """TODO: add documentation"""
        pass

    @abstractmethod
    def transform_target(self, y):
        """TODO: add documentation"""
        pass

    @abstractmethod
    def transform_proxy(self, X):
        """TODO: add documentation"""
        pass

    @property
    @abstractmethod
    def n_bins_target(self):
        pass

    @property
    @abstractmethod
    def n_bins_proxy(self):
        pass

@dataclass
class UnivariateBinning(AbstractBinning):
    """TODO: add documentation"""
    target_bins: np.ndarray | None = None
    proxy_bins: np.ndarray | None = None
    min_n_bins_target: int | None = None
    max_n_bins_target: int | None = None
    max_n_bins_proxy: int | None = None
    n_cv: int = 10
    criterion: str = "dussap"
    preprocessor: object | None = None

    def fit(self, X, y):
        if self.target_bins is None: # TODO let's ignore the target_bins for now
           raise NotImplementedError("target_bins must be specified")

        # TODO optimize proxy_bins
        if self.proxy_bins is None:
            self.proxy_bins = None

        return self

    def transform_target(self, y):
        # TODO check what shapes y might have and add validity checks
        return self._transform(y, self.target_bins)

    def transform_proxy(self, X):
        X = self._preprocess(X)
        return self._transform(X, self.proxy_bins)

    @property
    def n_bins_target(self):
        return len(self.target_bins) - 1

    @property
    def n_bins_proxy(self):
        return len(self.proxy_bins) - 1

    def _transform(self, X, bins):
        """internally work with 2D arrays for now: e.g. (n_samples, 1) instead of (n_samples,)"""
        if X.ndim == 2:
            return np.digitize(X, bins)
        if X.ndim == 3:
            digitized_X = np.zeros_like(X)
            for batch in range(X.shape[0]):
                digitized_X[batch, :, :] = np.digitize(X[batch, :, :], bins)
            return digitized_X

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
class TreeBinning(AbstractBinning):
    """TODO: add documentation

    TODO:
        Should the tree object be passed to the TreeBinning or should TreeBinning create it?
        If so, should this object expose all Tree hyperparams so the user can customize them?
        Should this object optimize Tree hyperparams? Perform cross validation?
        If we create it, do we need a param that specifies Regression or Classification?
        What about boosting? funfolding uses AdaBoost, do we want that aswell?

        - the tree object should be passed, such that the user can configure it without us having to expose any of its hyperparameters
        - the tree object should be un-typed ("tree: object"), such that our code does not need to import anything from sklearn and such that any class conforming with a certain API (e.g., "tree.apply(X)") can be provided. In general, I aim at only a loose coupling with sklearn, where we assume sklearn's APIs but never import it and, hence, don't need to have it as a regular dependency. Assuming sklearn's API can also mean to disregard checks like "check_is_fitted(tree)" because "tree.apply" is meant to raise an exception if the tree is not fitted.
        - the binning should be given by the leaf indices, as you already implemented with "tree.apply". However, any two trees would produce different indices that cannot be mapped to each other; hence, there is no way of performing cross validation, boosting, or any other kind of ensembling (unless one uses "tree.predict" instead of "tree.apply", which, however, does not expose the leaf indices).
    """
    tree: object
    target_bins: np.ndarray
    fit_tree: bool = True

    def fit(self, X, y):
        if self.fit_tree:
            self.tree.fit(X, y) # fit the tree

        # create a mapping from arbitrary leaf IDs to nice, consecutive IDs
        X_tree = self.tree.apply(X) # arbitrary leaf IDs
        self.bin_index_ = TreeBinning._create_bin_index(X_tree) # the mapping
        self.proxy_bins = np.arange(self.n_bins_proxy + 1)

        return self

    def transform_proxy(self, X):
        X_tree = self.tree.apply(X)
        return self.bin_index_[X_tree, 1] # return nice, consecutive IDs

    def transform_target(self, y):
        return np.digitize(y, self.target_bins, right=True)

    @property
    def n_bins_target(self):
        return len(self.target_bins) - 1

    @property
    def n_bins_proxy(self):
        return self.bin_index_[:, 1].max() + 1 # the highest bin ID

    @staticmethod
    def _create_bin_index(X_tree):
        """Create a mapping of leaf IDs to consecutive bin numbers."""
        keys = np.unique(X_tree) # arbitrary leaf IDs != { 0, 1, ..., n_bins }
        bin_index = -1 * np.ones((keys.max()+1, 2), dtype=int) # invalid default value -1
        bin_index[keys, 0] = keys # populate the 1st column with keys
        bin_index[keys, 1] = np.arange(len(keys)) # populate the 2nd column with values
        return bin_index # a vectorizable mapping; apply like "bin_index[X_tree, 1]"

@dataclass
class GridSearchBinning(AbstractBinning):
    """TODO: add documentation"""
    base_binning: AbstractBinning | Callable
    param_grid: dict[str, object]
    criterion: str = "dussap"
    n_jobs: int = None

    def fit(self, X, y):
        trials = itertools.product(*self.param_grid.values())
        def trial_fn(trial):
            params = dict(zip(self.param_grid.keys(), trial))
            if isinstance(self.base_binning, AbstractBinning):
                binning = self.base_binning.set_params(**params)
            else:
                binning = self.base_binning(**params)
            binning.fit(X, y)
            loss = -1. # TODO create and evaluate matrix A
            return loss, params, binning
        results = []
        with Pool(self.n_jobs) as pool:
            results.extend(pool.imap(trial_fn, trials))
        self.results_ = sorted(results, key=lambda result: result[0])
        return self

    def transform_target(self, y):
        self.results_[0][2].transform_target(y)

    def transform_proxy(self, X):
        self.results_[0][2].transform_proxy(X)

    @property
    def n_bins_target(self):
        self.results_[0][2].n_bins_target()

    @property
    def n_bins_proxy(self):
        self.results_[0][2].n_bins_proxy()
