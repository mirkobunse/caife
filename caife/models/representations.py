import itertools
import numpy as np
from dataclasses import dataclass
from multiprocessing import Pool
from qunfold import AbstractRepresentation
from typing import Callable

@dataclass
class TreeRepresentation(AbstractRepresentation):
    """TODO: add documentation"""
    tree: object
    fit_tree: bool = True

    def fit_transform(self, X, y, average=True, n_classes=None):
        if self.fit_tree:
            self.tree.fit(X, y) # fit the tree

        if n_classes is None:
            n_classes = np.unique(y).shape[0] # TODO: change this

        # create a mapping from arbitrary leaf IDs to nice, consecutive IDs
        X_tree = self.tree.apply(X) # arbitrary leaf IDs
        self.bin_index_ = TreeRepresentation._create_bin_index(X_tree) # the mapping

        # transform X
        X = self.bin_index_[X_tree, 1]
        
        # return (f(X), y) if average==False
        if not average:
            return X, y
        
        A = np.zeros((self.n_bins_, n_classes))
        for c in range(n_classes):
            # insert normalized bin counts for all classes
            A[:, c] = np.bincount(X[y==c], minlength=self.n_bins_) / X[y==c].shape[0]

        return A

    def transform(self, X, average=True):
        X_tree = self.tree.apply(X)
        i_tree = self.bin_index_[X_tree, 1] # nice, consecutive IDs

        # return f(X) if average==False
        if not average:
            return i_tree

        # TODO: return normalized bincount if average==True?
        return np.bincount(i_tree) / i_tree.shape[0]


    @property
    def n_bins_(self):
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
class GridSearchRepresentation(AbstractRepresentation):
    """TODO: add documentation"""
    base_representation: AbstractRepresentation | Callable
    param_grid: dict[str, object]
    criterion: str = "dussap"
    n_jobs: int = None

    def fit_transform(self, X, y, average=True, n_classes=None):
        if not average:
            raise ValueError("GridSearchRepresentation.fit requires average==True")

        # instantiate all configurations in the param_grid
        configurations = itertools.product(*self.param_grid.values())

        # define the evaluation of each configuration
        def trial_fn(configuration):
            # instantiate the current representation
            params = dict(zip(self.param_grid.keys(), configuration))
            if isinstance(self.base_representation, AbstractRepresentation):
                representation = self.base_representation.set_params(**params)
            else:
                representation = self.base_representation(**params)

            # fit and evaluate the current representation
            A = representation.fit_transform(X, y)
            loss = -1. # TODO evaluate matrix A

            return loss, params, representation, A # the results of this trial

        # evaluate all configurations in parallel
        results = []
        with Pool(self.n_jobs) as pool:
            results.extend(pool.imap(trial_fn, configurations))
        self.results_ = sorted(results, key=lambda result: result[0])

        # return the outcome of the best configuration
        return self.results_[0][3]

    def transform(self, X, average=True):
        return self.results_[0][2].transform(X, average=average)
