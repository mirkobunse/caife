"""Module containing models of the measurement process."""

from abc import ABC, abstractmethod

class AbstractModel(ABC):
    """Abstract Base Class all caife Models inherit from."""

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
            This model's itself.
        """
        pass

    @abstractmethod
    def proxy_view(self, X, sample_weight=None):
        """Return a view of the proxy distribution from the given proxy samples.

        Args:
            X: Proxy samples, shape (n_samples,) or (n_samples, n_proxy_features).
            sample_weight (optional): Per-sample weights, shape (n_samples,). Defaults to `None` for uniform weights.

        Returns:
            The model's view of the proxy distribution.
        """
        pass

    @abstractmethod
    def target_view(self, y, sample_weight=None):
        """Return a view of the target distribution from the given target samples.

        Args:
            y: Target samples, shape (n_samples,).
            sample_weight (optional): Per-sample weights, shape (n_samples,). Defaults to `None` for uniform weights.

        Returns:
            The model's view of the target distribution.
        """
        pass

    @abstractmethod
    def __call__(self, f, s):
        """Apply this model to a candidate spectrum.

        Args:
            f: A candidate spectrum, shape `(n_target_bins,)`.
            s: A vector of systematic parameter values, shape `(n_systematic_parameters,)`. For some models, `s` is optional.

        Returns:
            The predicted proxy distribution `g`, shape `(n_proxy_bins,)`, as modeled for `f`.
        """
        pass
