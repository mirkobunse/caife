from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp
from qunfold import AbstractRepresentation

from . import AbstractModel
from .latents import LatentSpectrum


@dataclass
class MultiTargetModel(AbstractModel):
    """TODO."""
    target_bins: list[list[float]]
    representations: list[AbstractRepresentation]

    def fit(self, X, Y, sample_weight=None, systematics=None, background=None):
        for b_i in self.target_bins:
            if b_i[0] > -np.inf or b_i[-1] < np.inf:
                raise ValueError("Not all target_bins are defined from -inf to inf")
        y = self.represent_targets(Y)
        self._fit_transfers(X, y, sample_weight, systematics)

        # store the background distribution
        if background is not None:
            if isinstance(background, tuple): # background with weights
                self.gs_background_ = self.proxy_view(
                    background[0],
                    sample_weight=background[1],
                )
            else:
                self.gs_background_ = self.proxy_view(background)
        else:
            self.gs_background_ = [jnp.zeros(r.n_output_features) for r in self.representations]
        return self

    def _fit_transfers(self, X, y, sample_weight, systematics):
        """Fit the transfer models `A[i](s)`."""
        if systematics is not None:
            print("WARNING: MultiTargetModel does not support systematics; chose another model")
        As = []
        for i, representation in enumerate(self.representations):
            A = representation.fit_transform(
                X[:,[i]],
                y,
                sample_weight=sample_weight,
                n_classes=self.n_bins_multitarget,
            )
            As.append(jnp.array(A))
        self.As_ = As

    def proxy_view(self, X, sample_weight=None, separate=True):
        def create_view(representation):
            g = self.representation.transform(X, sample_weight=sample_weight)
            return jnp.array(g * len(X)) # scale to counts
        separate_views = [create_view(r) for r in self.representations]
        if separate:
            return separate_views
        return jnp.concatenate(separate_views)

    def target_view(self, Y, sample_weight=None):
        f = np.bincount(
            self.represent_targets(Y),
            weights=sample_weight,
            minlength=self.n_bins_multitarget,
        )
        return f * (Y.shape[0] / f.sum()) # scale to counts

    def represent_targets(self, Y, separate=False):
        """Represent each individual target through binning."""
        if not np.isfinite(Y).all():
            raise ValueError("Y contains nans or infs")
        Y = np.array([
            np.digitize(Y_i, b_i) - 1
            for Y_i, b_i in zip(Y.T, self.target_bins)
        ]).T # shape (n_samples, n_targets)
        if separate:
            return Y
        factors = np.concatenate(([1], np.cumprod(self.n_bins_per_target[:-1])))
        return np.sum(Y * factors, axis=1)

    def create_latents(self, X): # create a single latent such that params = f
        return LatentSpectrum(n_samples=len(X), n_bins_target=self.n_bins_multitarget)

    def __call__(self, f): # f = params with shape (self.n_bins_multitarget,)
        g_pred = [A @ f + g for A, g in zip(self.As_, self.gs_background_)]
        return g_pred

    @property
    def n_background_samples(self):
        return self.gs_background_[0].sum()

    @n_background_samples.setter
    def n_background_samples(self, value):
        self.gs_background_ = [g * value / g.sum() for g in self.gs_background_]

    @property
    def n_bins_multitarget(self):
        return np.prod(self.n_bins_per_target)

    @property
    def n_bins_per_target(self):
        return np.array([len(x)-1 for x in self.target_bins])
