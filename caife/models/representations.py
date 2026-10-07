"""Module containing representations of the proxy feature space."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import numpy as np

from .._qunfold import AbstractRepresentation, check_y, class_prevalences


@dataclass
class TreeBinning(AbstractRepresentation):
    """A proxy representation that partitions the feature space according to the leaves of a decision tree.

    A `TreeBinning` uses a (typically shallow) decision tree classifier to discretize the proxy features `X` into bins: each leaf of the tree becomes one proxy bin, and `fit_transform`/`transform` maps samples to these bins through `tree.apply(X)`. Since decision tree leaves are trained to be informative about the target `y`, this typically yields proxy bins that carry more information than naive, equidistant binnings. It also supports arbitrary numbers of features natively.

    Args:
        tree: A tree-based classifier with a scikit-learn-compatible `fit(X, y, sample_weight=None)` and `apply(X)` interface, such as an instance of `sklearn.tree.DecisionTreeClassifier`.
        fit_tree (optional): Whether `fit_transform` should call `tree.fit`. Set this to `False` to reuse an already-fitted `tree`, e.g., one that was fitted on a separate portion of the data. Defaults to `True`.

    Attributes:
        bin_index_: A mapping from the tree's (arbitrary) leaf IDs to consecutive proxy bin IDs, shape `(max_leaf_id + 1, 2)`; apply it as `bin_index_[tree.apply(X), 1]`. Set during `fit_transform`.
    """
    tree: object
    fit_tree: bool = True

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        check_y(y, n_classes)
        self.p_trn = class_prevalences(y, n_classes)
        n_classes = len(self.p_trn) # not None anymore

        # fit the tree
        if self.fit_tree:
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
class _SingleUnivariateBinning:
    """The `UnivariateBinning` of a single variable."""
    proxy_bins_i : list[float]

    def __post_init__(self):
        if self.proxy_bins_i[0] > -np.inf or self.proxy_bins_i[-1] < np.inf:
            raise ValueError("proxy_bins are not defined from -inf to inf")

    def create_A_i(self, X_i, y, sample_weight, n_classes):
        A_i = np.bincount( # nothing to fit; immediately return the transformed data
            n_classes * self._digitize(X_i) + y, # combined X_i*y bins
            weights=sample_weight,
            minlength=self.n_output_features_i * n_classes,
        ).reshape((self.n_output_features_i, n_classes))
        return A_i / A_i.sum(axis=0, keepdims=True)

    def transform_i(self, X_i, sample_weight=None, average=True):
        X_i = self._digitize(X_i)
        if not average:
            return np.eye(self.n_output_features_i)[X_i] # one-hot encoding
        g_i = np.bincount(X_i, weights=sample_weight, minlength=self.n_output_features_i)
        return g_i / g_i.sum()

    def _digitize(self, X_i):
        return np.digitize(X_i, self.proxy_bins_i) - 1

    @property
    def n_output_features_i(self):
        return len(self.proxy_bins_i) - 1


@dataclass
class UnivariateBinning(AbstractRepresentation):
    """A proxy representation that bins each proxy feature independently.

    Each proxy feature is discretized into its own bins according to `proxy_bins`, and the resulting per-feature representations are concatenated along the bin dimension. Unlike `GridBinning`, which forms the full Cartesian product of per-feature bins, `UnivariateBinning`'s total number of proxy bins is the *sum* (not the product) of the per-feature bin counts, which keeps it usable even with several proxy features.

    Args:
        proxy_bins: The bin boundaries of a single proxy feature, ranging from `-inf` to `inf`, or a list of such boundaries, one for each proxy feature.
    """
    proxy_bins: list[float] | list[list[float]]

    def __post_init__(self):
        if isinstance(self.proxy_bins[0], float):
            self._binnings = [_SingleUnivariateBinning(self.proxy_bins)]
        else:
            self._binnings = [_SingleUnivariateBinning(b) for b in self.proxy_bins]

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        X = np.array(X)
        if X.ndim != 2 or X.shape[1] != len(self._binnings):
            raise ValueError("X.shape != (n_samples, n_proxy_variables)")
        check_y(y, n_classes)
        self.p_trn = class_prevalences(y, n_classes)
        n_classes = len(self.p_trn) # not None anymore
        if not average:
            return self.transform(X, average=False)
        return np.concatenate([ # concatenate A matrices along the feature dimension
            b_i.create_A_i(X_i, y, sample_weight, n_classes)
            for X_i, b_i in zip(X.T, self._binnings)
        ])

    def transform(self, X, sample_weight=None, average=True):
        return np.concatenate(
            [
                b_i.transform_i(X_i, sample_weight, average)
                for X_i, b_i in zip(X.T, self._binnings)
            ],
            axis=0 if average else 1, # concatenate along the respective feature dimension
        )

    @property
    def n_output_features(self):
        return np.sum([b.n_output_features_i for b in self._binnings])


@dataclass
class GridBinning(AbstractRepresentation):
    """A proxy representation whose bins are the cells of a regular grid.

    Each proxy feature is discretized according to its own bin boundaries in `proxy_bins`, and the resulting per-feature bin indices are combined into a single joint bin index for each sample, one per cell of the resulting grid. Unlike `UnivariateBinning`, whose number of bins grows additively with the number of proxy features, `GridBinning`'s bin number grows multiplicatively (the product of the per-feature numbers of bins), which quickly becomes impractical for more than a few proxy features but is highly informative for a few relevant proxy features.

    Args:
        proxy_bins: A list of bin boundaries, one list per proxy feature, each ranging from `-inf` to `inf`.
    """
    proxy_bins: list[list[float]]

    def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
        X = np.array(X)
        if X.ndim != 2 or X.shape[1] != len(self.proxy_bins):
            raise ValueError("X.shape != (n_samples, n_proxy_variables)")
        check_y(y, n_classes)
        self.p_trn = class_prevalences(y, n_classes)
        n_classes = len(self.p_trn) # not None anymore
        if not average:
            return self.transform(X, average=False)
        A = np.bincount( # nothing to fit; immediately return the transformed data
            n_classes * self.represent_features(X) + y, # combined X*y bins
            weights=sample_weight,
            minlength=self.n_output_features * n_classes,
        ).reshape((self.n_output_features, n_classes))
        return A / A.sum(axis=0, keepdims=True)

    def transform(self, X, sample_weight=None, average=True):
        X = self.represent_features(X)
        if not average:
            return np.eye(self.n_output_features)[X] # one-hot encoding
        g = np.bincount(X, weights=sample_weight, minlength=self.n_output_features)
        return g / g.sum()

    def represent_features(self, X):
        """Represent all features jointly through binning."""
        X = np.array([
            np.digitize(X_i, b_i) - 1
            for X_i, b_i in zip(X.T, self.proxy_bins)
        ]).T # shape (n_samples, n_features)
        factors = np.concatenate(([1], np.cumprod(self.n_bins_per_feature[::-1][:-1])))
        return np.sum(X[:,::-1] * factors, axis=1)

    @property
    def n_output_features(self):
        return np.prod(self.n_bins_per_feature)

    @property
    def n_bins_per_feature(self):
        return np.array([len(x)-1 for x in self.proxy_bins])


@dataclass
class GridSearchRepresentation(AbstractRepresentation):
    """A representation that selects the hyper-parameters of its `base_representation` via grid search.

    For each combination of hyper-parameters in `param_grid`, a clone of `base_representation` is fitted and its resulting transfer matrix `A` is scored according to the given `criterion`. The clone with the lowest score is kept and used for all further calls to `transform`.

    Args:
        base_representation: The representation whose hyper-parameters are searched over. It must support scikit-learn's `clone` and `set_params` interface.
        param_grid: A `dict` that maps each hyper-parameter name (as accepted by `base_representation.set_params`) to a list of candidate values, in the same way as scikit-learn's `GridSearchCV`. An empty `dict` evaluates `base_representation` itself, unmodified.
        criterion (optional): The criterion used to score each candidate representation's transfer matrix `A`, either `"dussap"` (Dussap et al., 2023; minimizes the inverse of the second-smallest eigenvalue of the centered Gram matrix of `A`) or `"blobel"` (Blobel, 1985; minimizes the condition number of `A`). Defaults to `"dussap"`.
        n_jobs (optional): The number of parallel processes used to evaluate candidates. Use `1` to evaluate sequentially, without spawning subprocesses; any other value, including `None`, evaluates candidates through `multiprocessing.Pool` with that many processes, or with one process per available processor if `n_jobs` is `None` or less than `1`. Defaults to `None`.
        is_verbose (optional): Whether to print progress while evaluating candidates. Defaults to `False`.

    Attributes:
        results_: A list of all evaluated candidates, sorted ascending by `criterion`. Each candidate is represented as a `dict` with keys `"losses"`, `"params"`, `"representation"`, and `"A"`. This attribute is set during `fit_transform`.
    """
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
