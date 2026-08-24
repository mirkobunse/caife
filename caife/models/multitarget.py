import time
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy
from qunfold import AbstractRepresentation

from ..solvers.scipy import minimize
from . import AbstractModel
from .latents import LatentReshape, LatentSpectrum, LatentSystematics


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
        """Fit the transfer model `A(s)`."""
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


@dataclass
class MultiTargetSystematicsModel(MultiTargetModel):
    """TODO."""
    solver: str = "L-BFGS-B" # same as in sklearn's LogisticRegression
    solver_options: dict[str,object] = field(default_factory=lambda: {
        "gtol": 1e-8,
        "maxiter": 100,
    })
    max_rank: int | None = None
    min_samples_per_transfer_bin: int = 1
    seed: int | None = 0

    def _fit_transfer(self, X, y, sample_weight=None, systematics=None):
        if systematics is None:
            raise ValueError("No systematics given; use a MultiTargetModel instead")
        X = self.representation.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            n_classes=self.n_bins_multitarget,
            average=False,
        )
        is_finite = np.all(np.isfinite(X), axis=1)
        X = X[is_finite,:]
        y = y[is_finite]
        if sample_weight is not None:
            sample_weight = sample_weight[is_finite]

        # fit parameters of logistic regressions that estimate A for systematics
        # A[p,t] ~ softmax_t(<a, (s, 1)>) with a = (a', b)
        self.systematic_bounds = np.stack(
            (systematics.min(axis=0), systematics.max(axis=0))).T
        systematics = jnp.concatenate( # append a one-column for the bias term
            (systematics, jnp.ones((systematics.shape[0], 1))),
            axis=1,
        )
        target_mask = jax.nn.one_hot(y, self.n_bins_multitarget)

        # handle masking and weighting
        self.A_mask = jnp.einsum( # check where the full matrix is sufficiently populated
            "np,nt->pt", X, target_mask) >= self.min_samples_per_transfer_bin
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        class_weight = jnp.sum( # normalize weights per class to unit sum
            target_mask * sample_weight.reshape((-1, 1)),
            axis=0,
        )
        sample_weight = sample_weight / class_weight[y]

        # decompose coeffs = trunk @ heads
        rank = systematics.shape[1]
        if self.max_rank is not None and self.max_rank < rank:
            rank = self.max_rank
        trunk_shape = (
            rank, # = r
            systematics.shape[1], # = s = n_systematic_params
        )
        heads_shape = (
            X.shape[1], # = p = n_bins_proxy
            self.n_bins_multitarget, # = t = n_bins_multitarget
            rank, # = r
        )
        x0, unravel_fn = jax.flatten_util.ravel_pytree([
            np.random.default_rng(self.seed).normal( # initial guess for the trunk
                scale=1/systematics.shape[1], size=trunk_shape),
            np.zeros(heads_shape), # initial guess for the heads
        ])
        def loss_fn(params):
            trunk, heads = unravel_fn(params)
            loss = jnp.average(
                softmax_cross_entropy(
                    jnp.einsum( # compute logits for each sample
                        "rs,ptr,ns,nt->np", # n = n_samples
                        trunk, # shape (r, s), see above
                        heads, # shape (p, t, r), see above
                        systematics, # shape (n, s)
                        target_mask, # shape (n, t)
                    ),
                    X, # one-hot encoding of proxy bins
                ),
                weights=sample_weight,
            )
            return loss
        t_init = time.time()
        self.opt_ = minimize(
            loss_fn,
            x0=x0,
            solver=self.solver,
            solver_options=self.solver_options,
        )
        self.opt_.wallclock_time = time.time() - t_init
        self.trunk_, self.heads_ = unravel_fn(self.opt_.x) # optimized coefficients

    def A(self, s):
        return jax.nn.softmax( # p=n_bins_proxy, t=n_bins_multitarget, s=n_systematic_params
            jnp.einsum(
                "rs,ptr,s->pt",
                self.trunk_,
                self.heads_,
                jnp.concatenate((s, jnp.ones(1)))
            ),
            axis=0, # for each target bin, apply softmax over all proxy bins
            where=self.A_mask,
        )

    def create_latents(self, X):
        return ( # create a tuple of latents such that params = (f, s)
            LatentReshape(
                LatentSpectrum(n_samples=len(X), n_bins_target=self.n_bins_multitarget),
                self.n_bins_per_target, # = shape
            ),
            LatentSystematics(bounds=self.systematic_bounds),
        )

    def __call__(self, params):
        f, s = params # f has shape tuple(self.n_bins_per_target)
        g_pred = self.A(s) @ f.reshape(-1) + self.g_background_
        return g_pred
