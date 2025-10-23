from abc import ABC
from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

SKLEARN_TREE_LEAF_FEATURE = -2

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
           assert y is not None, "Need to specify targets if initialized with no target bins!"
           ... 
        # TODO optimize bins for observations
        self.obs_bins = ...
        return self

    def transform_target(self, y):
        # TODO check what shapes y might have and add validity checks
        return self._transform(y, self.target_bins)

    def transform_proxy(self, X):
        return self._transform(X, self.obs_bins, apply_preprocessor=True)

    def _transform(self, X, bins, apply_preprocessor=False):
        """internally work with 2D arrays for now: e.g. (n_samples, 1) instead of (n_samples,)"""
        preprocessed_X = X
        if apply_preprocessor:
            preprocessed_X = self._preprocess(X)
        if preprocessed_X.ndim == 2:
            return np.digitize(preprocessed_X, bins)
        if preprocessed_X.ndim == 3:
            digitized_X = np.zeros_like(preprocessed_X)
            for batch in range(preprocessed_X.shape[0]):
                digitized_X[batch, :, :] = np.digitize(preprocessed_X[batch, :, :], bins)
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
class TreeBinning(Binning):
    """ 
    TODO:
        Should the tree object be passed to the TreeBinning or should TreeBinning create it?
        If so, should this object expose all Tree hyperparams so the user can customize them?
        Should this object optimize Tree hyperparams? Perform cross validation?
        If we create it, do we need a param that specifies Regression or Classification?
        What about boosting? funfolding uses AdaBoost, do we want that aswell?
    """
    tree: DecisionTreeRegressor | DecisionTreeClassifier

    def fit(self, X, y):
        """TODO: add documentation"""
        if not self._tree_fitted() or self.tree.get_n_leaves() > self.max_n_bins_proxy:
            setattr(self.tree, "max_leaf_nodes", self.max_n_bins_proxy)
            self.tree.fit(X, y) # refit if more leaves than max_bins
        self._create_bin_index()
        return self

    def transform_proxy(self, X):
        """TODO: add documentation"""
        if not self._tree_fitted():
            raise NotFittedError("self.tree is not fitted!")
        
        if not hasattr(self, 'bin_index'):
            raise NotFittedError("This TreeBinning object has not been fitted!")
        
        leaf_indices = self.tree.apply(X)
        return np.array([self.bin_index[li] for li in leaf_indices])

    def transform_target(self, y):
        """TODO: add documentation"""
        if self.target_bins is None:
            raise ValueError("Unable to transform targets if no target bins were provided!")
        if y.ndim == 1:
            y = np.expand_dims(y, 1)
        return np.digitize(y, self.target_bins)

    def _create_bin_index(self):
        """create a mapping of tree nodes to bin numbers"""
        if not self._tree_fitted():
            raise NotFittedError("Unable to create bin index for unfitted tree!")
        
        self.bin_index = {}
        is_leaf = self.tree.tree_.feature == SKLEARN_TREE_LEAF_FEATURE  # <- I like this for readability instead of magic number -2
        # get all tree nodes that are leafs and map them to a counter
        # leaf1 => 0, leaf2 => 1, ...
        for leaf_index, tree_index in enumerate(np.nonzero(is_leaf)[0]):
            self.bin_index[tree_index] = leaf_index

    def _tree_fitted(self):
        """check whether self.tree is already fitted"""
        try:
            check_is_fitted(self.tree, 'tree_')
            return True
        except NotFittedError:
            return False
