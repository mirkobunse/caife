import numpy as np
import itertools
import jax
from . import AbstractModel
from dataclasses import dataclass, replace


@dataclass
class SystematicBinCollection(AbstractModel):
    """Abstract collection of separate models that correspond to systematic bins."""
    base_model: AbstractModel
    n_bins_per_systematic: int = 3

    @staticmethod
    def create_systematic_bins(systematics, n_bins_per_systematic=3):
        """Create systematic bin boundaries.

        Args:
            systematics: Per-sample systematic parameters, shape `(n_samples, n_systematic_parameters)`.
            n_bins_per_systematic (optional): The number of bins per systematic parameter. Must be greater than or equal to 2. Defaults to 3.

        Returns:
            A matrix with the bin boundaries for all systematic parameters, shape `(n_systematic_parameters, n_bins_per_systematic)`.
        """
        if n_bins_per_systematic < 2:
            raise ValueError("n_bins_per_systematic must be >= 2")
        systematic_bins = np.stack([
            np.linspace(s.min(), s.max(), n_bins_per_systematic+1) for s in systematics.T
        ])
        systematic_bins[:,0] = -np.inf
        systematic_bins[:,-1] = np.inf
        return systematic_bins

    @staticmethod
    def create_bin_indices(systematics, systematic_bins):
        """Create a mapping from bin keys to sample indices. The bin keys are tuples of the systematic-wise bin indices, e.g., `(0, 0), (0, 1), ... (2, 2)` for 2 systematic parameters and 3 bins per systematic. The sample indices are represented as boolean vectors of shape `(n_samples,)` indicating which samples are within the corresponding bin.

        Args:
            systematics: Per-sample systematic parameters, shape `(n_samples, n_systematic_parameters)`.
            systematic_bins: A matrix with the bin boundaries for all systematic parameters, shape `(n_systematic_parameters, n_bins_per_systematic)`. Can be instantiated with `SystematicBinCollection.create_systematic_bins`.

        Returns:
            A dict mapping from tuples of systematic-wise bin indices to boolean indicator vectors of shape `(n_samples,)`.
        """
        n_systematics = systematic_bins.shape[0]
        n_bins_per_systematic = systematic_bins.shape[1] - 1
        bin_indices = {}
        for bin_key in itertools.product(*[range(n_bins_per_systematic)]*n_systematics):
            is_in_bin = np.ones(len(systematics), dtype=bool)
            for i_systematic, i_bin in enumerate(bin_key):
                is_in_bin = np.logical_and(
                    is_in_bin,
                    systematics[:, i_systematic] >= systematic_bins[i_systematic, i_bin],
                )
                is_in_bin = np.logical_and(
                    is_in_bin,
                    systematics[:, i_systematic] < systematic_bins[i_systematic, i_bin+1],
                )
            bin_indices[bin_key] = is_in_bin
        return bin_indices

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if systematics is None:
            raise ValueError("No systematics given; use another model")

        # find sample indices that partition the training data into systematic bins
        self.systematic_bins_ = SystematicBinCollection.create_systematic_bins(
            systematics, self.n_bins_per_systematic)
        bin_indices = SystematicBinCollection.create_bin_indices(
            systematics, self.systematic_bins_)

        # fit one model for each systematic bin
        self.models = {}
        for bin_key, is_in_bin in bin_indices.items():
            self.models[bin_key] = self.base_model.clone().fit(
                X[is_in_bin],
                y[is_in_bin],
                sample_weight=None if sample_weight is None else sample_weight[is_in_bin],
                background=background,
            )
        return self

    def proxy_view(self, X, sample_weight=None):
        return jax.tree.map(lambda m: m.proxy_view(X), self.models)

    def target_view(self, y, sample_weight=None):
        return self.base_model.target_view(
            y, sample_weight) # assume that fitting is not necessary

    def clone(self):
        return replace(self)


@dataclass
class SeparateSystematicBinCollection(SystematicBinCollection):
    """A collection of separate models that correspond to systematic bins which fits as many separate spectra as there are models.

    Args:
        base_model: The model configuration that is to be replicated across systematic bins.
        n_bins_per_systematic (optional): The number of bins per systematic parameter. Defaults to 3.

    Note:
        The total number of bins will be `n_bins_per_systematic ** n_systematic_parameters`, which can become prohibitively large. A large number of total bins not only results in a large model collection that requires a long time for being solved but also in low training statistics for each individual model within the collection.
    """

    def create_latents(self, X):
        return jax.tree.map(lambda m: m.create_latents(X), self.models)

    def __call__(self, params):
        return jax.tree.map(lambda m, p: m(p), self.models, params)


@dataclass
class JointSystematicBinCollection(SystematicBinCollection):
    """A collection of separate models that correspond to systematic bins which fits only a single joint spectrum across all models.

    Args:
        base_model: The model configuration that is to be replicated across systematic bins.
        n_bins_per_systematic (optional): The number of bins per systematic parameter. Defaults to 3.

    Note:
        The total number of bins will be `n_bins_per_systematic ** n_systematic_parameters`, which can become prohibitively large. A large number of total bins not only results in a large model collection that requires a long time for being solved but also in low training statistics for each individual model within the collection.
    """

    def create_latents(self, X):
        return self.base_model.create_latents(X)

    def __call__(self, params):
        return jax.tree.map(lambda m: m(params), self.models)
