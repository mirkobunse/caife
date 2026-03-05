import itertools
import numpy as np
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from qunfold import AbstractRepresentation
from qunfold.methods import class_prevalences, check_y

@dataclass
class TreeBinning(AbstractRepresentation):
    """TODO: add documentation.

    Args:
        tree: A decision tree object that is in line with the API exposed by `sklearn.tree.DecisionTreeClassifier`.
        fit_tree (optional): Whether to clone and fit the given `tree`. Defaults to `True`.
    """
    tree: object
    fit_tree: bool = True

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        check_y(y, n_classes)
        self.p_trn = class_prevalences(y, n_classes)
        n_classes = len(self.p_trn) # not None anymore

        # fit the tree
        if self.fit_tree:
            self.tree = self.tree.__sklearn_clone__()
            self.tree.fit(X, y, sample_weight=sample_weight)

        # create a mapping from arbitrary leaf IDs to nice, consecutive IDs
        X_tree = self.tree.apply(X) # arbitrary leaf IDs
        self.bin_index_ = TreeBinning._create_bin_index(X_tree) # the mapping
        i_tree = self.bin_index_[X_tree, 1] # nice, consecutive IDs

        # return f(X) if average==False
        if not average:
            return np.eye(self.n_output_features)[i_tree] # one-hot encoding

        # create the transfer matrix
        A = np.bincount(
            n_classes * i_tree + y, # combined X*y bins
            weights=sample_weight,
            minlength=self.n_output_features * n_classes,
        ).reshape((self.n_output_features, n_classes))
        return A / A.sum(axis=0, keepdims=True)

    def transform(self, X, sample_weight=None, average=True):
        X_tree = self.tree.apply(X)
        i_tree = self.bin_index_[X_tree, 1] # nice, consecutive IDs

        # return f(X) if average==False
        if not average:
            i_tree_one_hot = np.eye(self.n_output_features)[i_tree]
            return i_tree_one_hot

        # create a histogram
        bin_count = np.bincount(
            i_tree,
            weights=sample_weight,
            minlength=self.n_output_features,
        )
        return bin_count / bin_count.sum()

    @property
    def n_output_features(self):
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
class UnivariateBinning(AbstractRepresentation):
    """TODO: add documentation"""
    proxy_bins: list[float]

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        check_y(y, n_classes)
        self.p_trn = class_prevalences(y, n_classes)
        n_classes = len(self.p_trn) # not None anymore
        if self.proxy_bins[0] > -np.inf or self.proxy_bins[-1] < np.inf:
            raise ValueError("proxy_bins are not defined from -inf to inf")

        # nothing to fit; immediately return the transformed data
        if not average:
            return self.transform(X, average=False)
        A = np.bincount(
            n_classes * self._digitize(X) + y, # combined X*y bins
            weights=sample_weight,
            minlength=self.n_output_features * n_classes,
        ).reshape((self.n_output_features, n_classes))
        return A / A.sum(axis=0, keepdims=True)

    def transform(self, X, sample_weight=None, average=True):
        X = self._digitize(X)
        if not average:
            return np.eye(self.n_output_features)[X] # one-hot encoding
        g = np.bincount(X, weights=sample_weight, minlength=self.n_output_features)
        return g / g.sum()

    def _digitize(self, X):
        return np.digitize(np.asarray(X)[:, 0], self.proxy_bins) - 1

    @property
    def n_output_features(self):
        return len(self.proxy_bins) - 1


@dataclass
class GridSearchRepresentation(AbstractRepresentation):
    """TODO: add documentation"""
    base_representation: AbstractRepresentation
    param_grid: dict[str, object]
    criterion: str = "dussap"
    n_jobs: int | None = None
    is_verbose: bool = False

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        if not average:
            raise ValueError("GridSearchRepresentation.fit_transform requires average==True")
        if self.criterion not in ["dussap", "blobel"]:
            raise ValueError(f"Unknown criterion==\"{self.criterion}\"")

        # instantiate all grid_cells in the param_grid
        grid_cells = list(itertools.product(*self.param_grid.values()))
        if len(grid_cells) == 0: # mark at least the base_binning for evaluation
            grid_cells = [ None ]

        # define the evaluation of each grid_cell
        grid_cell_fn = partial(
            GridSearchRepresentation._grid_cell_fn,
            X=X,
            y=y,
            sample_weight=sample_weight,
            n_classes=n_classes,
            param_grid_keys=list(self.param_grid.keys()),
            base_representation=self.base_representation,
        )

        # evaluate all grid_cells in parallel
        results = []
        if self.n_jobs == 1:
            for grid_cell in grid_cells:
                results.append(grid_cell_fn(grid_cell))
                if self.is_verbose:
                    print(f"{self} evaluated {len(results)}/{len(grid_cells)} cells")
        else:
            n_jobs = self.n_jobs
            if n_jobs is not None and n_jobs < 1:
                n_jobs = None
            with Pool(n_jobs) as pool:
                for cell_results in pool.imap(grid_cell_fn, grid_cells):
                    results.append(cell_results)
                    if self.is_verbose:
                        print(f"{self} evaluated {len(results)}/{len(grid_cells)} cells")
        self.results_ = sorted( # sort by loss (ascending)
            results,
            key=lambda result: result["losses"][self.criterion]
        )

        # return the outcome of the best grid_cell
        return self.results_[0]["A"]

    @staticmethod
    def _grid_cell_fn( # how to evaluate each grid_cell
            grid_cell,
            X,
            y,
            sample_weight,
            n_classes,
            param_grid_keys,
            base_representation,
        ):
        params = dict(zip(param_grid_keys, grid_cell))

        # instantiate and fit the current representation
        representation = base_representation.clone().set_params(**params)
        A = representation.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            n_classes=n_classes,
        )

        # evaluate all criteria (fairly cheap; enables extensive evaluation)
        losses = {}

        # [dussap2023label]: minimize the inverse of the 2nd-smallest eigenvalue
        cme = A - A.mean(axis=0, keepdims=True) # centered mean embedding
        cgm = cme @ cme.T # centered gram matrix, gcm[i, j] = cme[i] @ cme[j]
        eigvals, _ = np.linalg.eigh(cgm) # eigenvalues
        losses["dussap"] = 1 / np.sqrt(np.abs(eigvals[1]))

        # [blobel1985unfolding]: minimize the condition number
        losses["blobel"] = np.linalg.cond(A)

        return { # the results of this trial
            "losses": losses,
            "params": params,
            "representation": representation,
            "A": A,
        }

    def transform(self, X, sample_weight=None, average=True):
        representation = self.results_[0]["representation"]
        return representation.transform(
            X,
            sample_weight=sample_weight,
            average=average,
        )

    @property
    def n_output_features(self):
        representation = self.results_[0]["representation"]
        return representation.n_output_features
