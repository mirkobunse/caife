import numpy as np
import jax
from . import AbstractModel
from ..solvers.scipy import minimize
from dataclasses import dataclass, field
from jax import numpy as jnp
from optax.losses import softmax_cross_entropy
from qunfold import AbstractRepresentation

@dataclass
class LinearCountModel(AbstractModel):
    """Linear Model for solving `g = A @ f + b` over count histograms `g`, `f`, and `b`.

    Args:
        target_bins: Bin boundaries of the target quantity, shape (n_target_bins+1,).
        representation: The data representation that is computed from the proxy features.

    Attributes:
        n_background_samples: Number of background samples, set during `fit` but meant to be manually changed to the actual number of background samples after fitting.
        n_bins_target: Number of target bins, as determined by the `target_bins`.
    """
    target_bins: list[float]
    representation: AbstractRepresentation

    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        if self.target_bins[0] > -np.inf or self.target_bins[-1] < np.inf:
            raise ValueError("target_bins are not defined from -inf to inf")
        y = self.represent_target(y)
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
        A = self.representation.fit_transform( # constant; no systematics modeled
            X,
            y,
            sample_weight=sample_weight,
            n_classes=self.n_bins_target,
        )
        self.A_ = jnp.array(A) # cast A to a JAX array to make __call__ differentiable

    def A(self, s=None):
        """Estimate the transfer matrix `A` for some systematics vector `s`."""
        return self.A_ # constant if no systematics are modeled

    def proxy_view(self, X, sample_weight=None):
        g = self.representation.transform(X, sample_weight=sample_weight)
        return g * len(X) # scale to counts; assume g is scaled to a unit sum

    def target_view(self, y, sample_weight=None):
        f = np.bincount(
            self.represent_target(y),
            weights=sample_weight,
            minlength=self.n_bins_target,
        )
        return f * (len(y) / f.sum()) # scale to counts

    def represent_target(self, y):
        if not np.isfinite(y).all():
            raise ValueError("y contains nans or infs")
        return np.digitize(y, self.target_bins) - 1

    def __call__(self, f, s=None):
        g_pred = self.A(s) @ f + self.g_background_
        return g_pred

    @property
    def n_background_samples(self):
        return self.g_background_.sum()

    @n_background_samples.setter
    def n_background_samples(self, value):
        self.g_background_ = self.g_background_ * value / self.g_background_.sum()

    @property
    def n_bins_target(self):
        return len(self.target_bins) - 1


@dataclass
class LinearSystematicsCountModel(LinearCountModel):
    """TODO: document."""
    C: float | None = None
    solver: str = "L-BFGS-B" # same as in sklearn's LogisticRegression
    solver_options: dict[str,object] = field(default_factory=lambda: {
        "gtol": 1e-8,
        "maxiter": 100,
    })

    def _fit_transfer(self, X, y, sample_weight=None, systematics=None):
        if systematics is None:
            raise ValueError("No systematics given; use a LinearCountModel instead")
        X = self.representation.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            n_classes=self.n_bins_target,
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
        target_mask = jax.nn.one_hot(y, self.n_bins_target)
        coeffs_shape = (
            X.shape[1], # = p = n_bins_proxy
            self.n_bins_target, # = t = n_bins_target
            systematics.shape[1], # = s = n_systematic_params
        )
        self.A_mask = jnp.einsum( # check where the full matrix is > 0
            "np,nt,n->pt", X, target_mask, sample_weight) > 0
        if sample_weight is not None:
            class_weight = jnp.sum( # normalize weights per class to unit sum
                target_mask * sample_weight.reshape((-1, 1)),
                axis=0,
            )
            sample_weight = sample_weight / class_weight[y]
        else:
            sample_weight = np.ones(len(y)) / representation.p_trn
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
        self.opt_ = minimize(
            loss_fn,
            x0=jnp.zeros(coeffs_shape).reshape(-1), # initial guess: all zeros
            solver=self.solver,
            solver_options=self.solver_options,
        )
        self.coeffs_ = self.opt_.x.reshape(coeffs_shape) # optimized coefficients

    def A(self, s):
        return jax.nn.softmax( # p=n_bins_proxy, t=n_bins_target, s=n_systematic_params
            jnp.einsum("pts,s->pt", self.coeffs_, jnp.concatenate((s, jnp.ones(1)))),
            axis=0, # for each target bin, apply softmax over all proxy bins
            where=self.A_mask,
        )


@dataclass
class LinearSystematicsScaleCountModel(LinearCountModel):
    """The scaling vector that models systematics in funfolding. TODO: document."""

    def _fit_transfer(self, X, y, sample_weight=None, systematics=None):
        if systematics is None:
            raise ValueError("No systematics given; use a LinearCountModel instead")
        X = self.representation.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            n_classes=self.n_bins_target,
            average=False,
        )
        is_finite = np.all(np.isfinite(X), axis=1)
        X = X[is_finite,:]
        y = y[is_finite]
        if sample_weight is not None:
            sample_weight = sample_weight[is_finite]

        # compute the regular full matrix
        target_mask = jax.nn.one_hot(y, self.n_bins_target)
        self.A_ = jnp.einsum("np,nt,n->pt", X, target_mask, sample_weight)

        # do what funfolding does
        self.systematic_bounds = np.stack(
            (systematics.min(axis=0), systematics.max(axis=0))).T
        # TODO

    def A(self, s):
        return None # TODO
