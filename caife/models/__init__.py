"""Module containing models of the measurement process."""

from abc import ABC, abstractmethod

from jax import numpy as jnp


class AbstractModel(ABC):
    """Abstract base class that all caife models inherit from."""

    @abstractmethod
    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        """Fit the model to the given data.

        Args:
            X: Proxy samples, shape (n_samples,) or (n_samples, n_proxy_features).
            y: Target samples that correspond to the proxy samples, shape (n_samples,).
            sample_weight (optional): Per-sample weights, shape (n_samples,). Defaults to `None` for uniform weights.
            systematics (optional): Per-sample systematic parameters, shape (n_samples, n_systematic_parameters). Defaults to `None` for no modeling of systematics.
            background (optional): A separate set of background instances, shape (n_background_samples,) or (n_background_samples, n_proxy_features). If the background should be weighted, a tuple of this set and its weights. Defaults to `None` for no background consideration.

        Returns:
            This model itself.
        """

    @abstractmethod
    def proxy_view(self, X, sample_weight=None):
        """Return a view of the proxy distribution from the given proxy samples.

        Args:
            X: Proxy samples, shape (n_samples,) or (n_samples, n_proxy_features).
            sample_weight (optional): Per-sample weights, shape (n_samples,). Defaults to `None` for uniform weights.

        Returns:
            The model's view of the proxy distribution.
        """

    @abstractmethod
    def target_view(self, y, sample_weight=None):
        """Return a view of the target distribution from the given target samples.

        Args:
            y: Target samples, shape (n_samples,).
            sample_weight (optional): Per-sample weights, shape (n_samples,). Defaults to `None` for uniform weights.

        Returns:
            The model's view of the target distribution.
        """

    @abstractmethod
    def create_latents(self, X):
        """Create a JAX pytree of latent vectors for this model.

        Args:
            X: The observations that are to be reconstructed.

        Returns:
            A JAX pytree of `AbstractLatentVector` instances with the same structure that calling this model requires for the `params` argument.
        """

    @abstractmethod
    def __call__(self, params):
        """Apply this model to a candidate spectrum.

        Args:
            params: A JAX pytree of parameters for this model; could be a single candidate spectrum of shape `(n_bins_target,)` or a collection of such a spectrum and a vector of systematic parameter values of shape `(n_systematic_parameters,)`.

        Returns:
            The predicted proxy distribution `g` of shape `(n_bins_proxy,)` that the model predicts for the given `params`.
        """


def create_mixture_model_fn(models):
    """Combine a collection of models into a single model that returns a weighted average of the model-wise outputs.

    Args:
        models: The models to combine, each with the same callable interface `*args -> g_est`. In particular, `*args` needs to be identical across all models.

    Returns:
        A callable `(weights, *args) -> g_est` that serves as a single linear model. This model averages the outputs of the individual models with the given weights.
    """
    def mixture_model_fn(weights, *args):
        g_est = jnp.stack([m(*args) for m in models]) # shape (n_models, n_bins_proxy)
        return jnp.average(g_est, weights=weights, axis=0)
    return mixture_model_fn
