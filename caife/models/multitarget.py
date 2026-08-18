from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp
from qunfold import AbstractRepresentation

from . import AbstractModel
from .latents import LatentReshape, LatentSpectrum


@dataclass
class MultiTargetModel(AbstractModel):
    """TODO."""
    target_bins: list[list[float]]
    representation: AbstractRepresentation

    def fit(self, X, Y, sample_weight=None, systematics=None, background=None):
        for b_i in self.target_bins:
            if b_i[0] > -np.inf or b_i[-1] < np.inf:
                raise ValueError("Not all target_bins are defined from -inf to inf")
        y = self.represent_targets(Y) # compute a single, joint class label of all targets
        self._fit_transfer(X, y, sample_weight, systematics)

        # store the background distribution
        n_bins_proxy = self.representation.n_output_features
        g_background = np.zeros(n_bins_proxy)
        if background is not None:
            if isinstance(background, tuple): # background with weights
                g_background = self.proxy_view(
                    background[0],
                    sample_weight=background[1],
                )
            else:
                g_background = self.proxy_view(background)
        self.g_background_ = jnp.array(g_background)

        return self # sklearn convention; allows method chaining

    def _fit_transfer(self, X, y, sample_weight, systematics):
        """Fit the transfer models `A[i](s)`."""
        if systematics is not None:
            print("WARNING: MultiTargetModel does not support systematics; chose another model")
        self.A_ = jnp.array(self.representation.fit_transform(
            X, y, sample_weight=sample_weight, n_classes=self.n_bins_multitarget))

    def proxy_view(self, X, sample_weight=None):
        g = self.representation.transform(X, sample_weight=sample_weight)
        return g * len(X) # scale to counts; assume g is scaled to a unit sum

    def target_view(self, Y, sample_weight=None):
        f = np.bincount(
            self.represent_targets(Y),
            weights=sample_weight,
            minlength=self.n_bins_multitarget,
        )
        return f.reshape(self.n_bins_per_target) * (Y.shape[0] / f.sum()) # scale to counts

    def represent_targets(self, Y, separate=False):
        """Represent all targets jointly (or separately) through binning."""
        if not np.isfinite(Y).all():
            raise ValueError("Y contains nans or infs")
        Y = np.array([
            np.digitize(Y_i, b_i) - 1
            for Y_i, b_i in zip(Y.T, self.target_bins)
        ]).T # shape (n_samples, n_targets)
        if separate:
            return Y
        factors = np.concatenate(([1], np.cumprod(self.n_bins_per_target[::-1][:-1])))
        return np.sum(Y[:,::-1] * factors, axis=1)

    def create_latents(self, X): # create a single latent such that params = f
        return LatentReshape(
            LatentSpectrum(n_samples=len(X), n_bins_target=self.n_bins_multitarget),
            self.n_bins_per_target, # = shape
        )

    def __call__(self, f): # f = params with shape tuple(self.n_bins_per_target)
        g_pred = self.A_ @ f.reshape(-1) + self.g_background_
        return g_pred

    @property
    def n_background_samples(self):
        return self.g_background_.sum()

    @n_background_samples.setter
    def n_background_samples(self, value):
        self.g_background_ = self.g_background_ * value / self.g_background_.sum()

    @property
    def n_bins_multitarget(self):
        return np.prod(self.n_bins_per_target)

    @property
    def n_bins_per_target(self):
        return np.array([len(x)-1 for x in self.target_bins])
