"""Module containing linear models of the measurement process over multiple target quantities."""

from __future__ import annotations

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
    """Linear model for solving `g = A @ f + b`, where `f` is a multi-dimensional count histogram.

    The `MultiTargetModel` generalizes the `LinearCountModel` to several target quantities at once. Each target quantity is discretized independently, according to its own bin boundaries in `target_bins`, and the resulting per-quantity bins are combined into a single joint class label, in the same way as `GridBinning` combines several proxy features. The target spectrum `f` therefore has shape `tuple(n_bins_per_target)`, a multi-dimensional count histogram over all target quantities jointly.

    This model does not consider any systematic parameters; use `MultiTargetSystematicsModel` if `systematics` need to be modeled.

    Args:
        target_bins: A list of bin boundary vectors, one vector per target quantity, each ranging from `-inf` to `inf`.
        representation: The data representation that is computed from the proxy features.

    Attributes:
        A_: The fitted transfer matrix `A`, shape `(n_bins_proxy, n_bins_multitarget)`. Set during `fit`.
        g_background_: The fitted background proxy distribution `b`, shape `(n_bins_proxy,)`. Set during `fit`.
        n_background_samples: Number of background samples, set during `fit` but meant to be manually changed to the actual number of background samples after fitting.
        n_bins_multitarget: The total number of joint target bins, i.e., the product of `n_bins_per_target`.
        n_bins_per_target: The number of bins of each individual target quantity, as determined by `target_bins`.
    """
    target_bins: list[list[float]]
    representation: AbstractRepresentation

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        for b_i in self.target_bins:
            if b_i[0] > -np.inf or b_i[-1] < np.inf:
                raise ValueError("Not all target_bins are defined from -inf to inf")
        y = self.represent_targets(y) # compute a single, joint class label of all targets
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
        """Fit the constant transfer matrix `A`; `systematics`, if given, are ignored."""
        if systematics is not None:
            print("WARNING: MultiTargetModel does not support systematics; chose another model")
        self.A_ = jnp.array(self.representation.fit_transform(
            X, y, sample_weight=sample_weight, n_classes=self.n_bins_multitarget))

    def proxy_view(self, X, sample_weight=None):
        g = self.representation.transform(X, sample_weight=sample_weight)
        return g * len(X) # scale to counts; assume g is scaled to a unit sum

    def target_view(self, y, sample_weight=None):
        f = np.bincount(
            self.represent_targets(y),
            weights=sample_weight,
            minlength=self.n_bins_multitarget,
        )
        return f.reshape(self.n_bins_per_target) * (y.shape[0] / f.sum()) # scale to counts

    def represent_targets(self, y, separate=False):
        """Represent all targets jointly (or separately) through binning."""
        if not np.isfinite(y).all():
            raise ValueError("y contains nans or infs")
        y = np.array([
            np.digitize(y_i, b_i) - 1
            for y_i, b_i in zip(y.T, self.target_bins)
        ]).T # shape (n_samples, n_targets)
        if separate:
            return y
        factors = np.concatenate(([1], np.cumprod(self.n_bins_per_target[::-1][:-1])))
        return np.sum(y[:,::-1] * factors, axis=1)

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
    """Systematics-aware `MultiTargetModel`, solving `g = A(s) @ f + b` over a joint, multi-dimensional target spectrum `f` and a vector of systematic parameters `s`.

    As in `LinearSystematicsCountModel`, each column of `A(s)` (representing the mean feature embedding of one joint target bin) is modeled as one logistic regression that takes `s` as its input, so that each column stays properly normalized to a unit sum and varies monotonically with each entry of `s`. As in `MultiTargetModel`, the target quantities are first combined into a single joint class label, over which this logistic regression is fitted, so that `A(s)` has shape `(n_bins_proxy, n_bins_multitarget)` and `f` has shape `tuple(n_bins_per_target)`.

    Args:
        target_bins: A list of bin boundaries, one list per target quantity, each ranging from `-inf` to `inf`.
        representation: The data representation that is computed from the proxy features.
        C (optional): The regularization strength for each logistic regression model, with the behavior defined by scikit-learn. Defaults to `None` for no regularization.
        solver (optional): The `method` argument in `scipy.optimize.minimize`. Defaults to `"L-BFGS-B"`.
        solver_options (optional): The `options` argument in `scipy.optimize.minimize`. Defaults to `{"gtol": 1e-8, "maxiter": 10_000}`.

    Attributes:
        coeffs_: The fitted coefficients of the per-column logistic regressions, shape `(n_bins_proxy, n_bins_multitarget, n_systematic_parameters + 1)`. Set during `fit`.
        A_mask: A boolean mask of shape `(n_bins_proxy, n_bins_multitarget)`, marking the proxy/target bin combinations that were actually observed while fitting; `A(s)` is only meaningful for these entries. Set during `fit`.
        systematic_bounds: The per-parameter `(min, max)` bounds observed in the training `systematics`, shape `(n_systematic_parameters, 2)`. Set during `fit`.
        opt_: The `scipy.optimize.OptimizeResult` of fitting the logistic regressions, with an additional `wallclock_time` attribute. Set during `fit`.
        g_background_: The fitted background proxy distribution `b`, shape `(n_bins_proxy,)`. Set during `fit`.
        n_background_samples: Number of background samples, set during `fit` but meant to be manually changed to the actual number of background samples after fitting.
        n_bins_multitarget: The total number of joint target bins, i.e., the product of `n_bins_per_target`.
        n_bins_per_target: The number of bins of each individual target quantity, as determined by `target_bins`.
    """
    C: float | None = None
    solver: str = "L-BFGS-B" # same as in sklearn's LogisticRegression
    solver_options: dict[str,object] = field(default_factory=lambda: {
        "gtol": 1e-8,
        "maxiter": 10_000,
    })

    def _fit_transfer(self, X, y, sample_weight=None, systematics=None):
        """Fit the systematics-dependent transfer matrix `A(s)` via per-column logistic regression."""
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
        coeffs_shape = (
            X.shape[1], # = p = n_bins_proxy
            self.n_bins_multitarget, # = t = n_bins_multitarget
            systematics.shape[1], # = s = n_systematic_params
        )
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.A_mask = jnp.einsum( # check where the full matrix is > 0
            "np,nt,n->pt", X, target_mask, sample_weight) > 0
        class_weight = jnp.sum( # normalize weights per class to unit sum
            target_mask * sample_weight.reshape((-1, 1)),
            axis=0,
        )
        sample_weight = sample_weight / class_weight[y]
        def loss_fn(coeffs):
            loss = jnp.average(
                softmax_cross_entropy(
                    jnp.einsum( # compute logits for each sample
                        "pts,ns,nt->np", # n = n_samples; for others, see coeffs_shape
                        coeffs.reshape(coeffs_shape), # shape (p, t, s)
                        systematics, # shape (n, s)
                        target_mask, # shape (n, t)
                    ),
                    X, # one-hot encoding of proxy bins
                ),
                weights=sample_weight,
            )
            if self.C is not None: # regularize
                return loss + coeffs @ coeffs / (2 * self.C * len(X))
            return loss
        t_init = time.time()
        self.opt_ = minimize(
            loss_fn,
            x0=jnp.zeros(coeffs_shape).reshape(-1), # initial guess: all zeros
            solver=self.solver,
            solver_options=self.solver_options,
        )
        self.opt_.wallclock_time = time.time() - t_init
        self.coeffs_ = self.opt_.x.reshape(coeffs_shape) # optimized coefficients

    def A(self, s):
        return jax.nn.softmax( # p=n_bins_proxy, t=n_bins_multitarget, s=n_systematic_params
            jnp.einsum("pts,s->pt", self.coeffs_, jnp.concatenate((s, jnp.ones(1)))),
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
