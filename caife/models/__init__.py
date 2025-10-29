from abc import ABC, abstractmethod

class AbstractModel(ABC):
    """Abstract Base Class all caife Models inherit from."""

    @abstractmethod
    def fit(self, X, y, sample_weight=None, systematics=None, background=None):
        """TODO: add documentation"""
        pass

    @abstractmethod
    def proxy_view(self, X):
        """Return a view of the proxy distribution from the given proxy samples.

        Args:
            X: Proxy samples, shape (n_samples,) or (n_samples, n_proxy_features).

        Returns:
            The model's view of the proxy distribution.
        """
        pass # TODO move views of distributions to a representation module?

    @abstractmethod
    def __call__(self, f):
        """Apply this model to a candidate spectrum.

        Args:
            f: A candidate spectrum, shape `(n_target_bins,)`.

        Returns:
            The predicted proxy distribution `g`, shape `(n_proxy_bins,)`, as modeled for `f`.
        """
        pass
