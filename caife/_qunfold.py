"""Private copy of those parts of qunfold (https://github.com/mirkobunse/qunfold) that caife uses.

The code is copied from qunfold's commit d046beb, reduced to the members that caife uses. This copy removes caife's dependency on qunfold until qunfold is fully integrated into caife.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

import jax.numpy as jnp
import numpy as np


class BaseMixin():
  """
  A mix-in for any configurable component. This mix-in defines `get_params` and `set_params` for hyper-parameter optimization, a `clone` method, and a concise string representation through `__str__`.
  """

  @classmethod
  def _get_param_names(cls): # utility for get_params
    return sorted([ p.name for p in inspect.signature(cls).parameters.values() ])

  def get_params(self, deep=True):
    """
    Get the parameters of this component, just like in scikit-learn estimators.
    """
    params = dict()
    for k in self._get_param_names():
      v = getattr(self, k)
      if deep and hasattr(v, "get_params") and not isinstance(v, type):
        params.update((k + "__" + _k, _v) for _k, _v in v.get_params(deep).items())
      params[k] = v
    return params

  def set_params(self, **params):
    """
    Set the parameters of this component, just like in scikit-learn estimators.
    """
    valid_params = self.get_params(deep=True)
    nested_params = defaultdict(dict)
    for k, v in params.items():
      k, delim, k_nested = k.partition("__")
      if k not in valid_params:
        raise ValueError(f"Invalid parameter \"{k}\" for {self}")
      if delim:
        nested_params[k][k_nested] = v
      else:
        setattr(self, k, v)
        valid_params[k] = v
    for k, nested in nested_params.items():
        valid_params[k].set_params(**nested)
    return self

  def clone(self, **params):
    """
    Create a clone of this object. If additional keyword arguments are provided, set these values in the newly created clone.
    """
    clone_params = self.get_params(deep=False)
    for name, param in clone_params.items():
      if isinstance(param, BaseMixin): # deeply clone BaseMixin types
        clone_params[name] = param.clone()
    clone_params = clone_params | params # update with additional arguments
    return self.__class__(**clone_params)

  def __str__(self): # logging sugar: a concise string representation
    params = []
    for param in inspect.signature(self.__class__).parameters.values():
      if getattr(self, param.name) != param.default:
        params.append(f"{param.name}={getattr(self, param.name)}")
    return f"{self.__class__.__name__}({', '.join(params)})"

  # TODO add the @dataclass annotation to all sub-classes of BaseMixin


def class_prevalences(y, n_classes=None):
  """Determine the prevalence of each class.

  Args:
      y: An array of labels, shape (n_samples,).
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `np.max(y)+1`.

  Returns:
      An array of class prevalences that sums to one, shape (n_classes,).
  """
  if n_classes is None:
    n_classes = np.max(y)+1
  n_samples_per_class = np.zeros(n_classes, dtype=int)
  i, n = np.unique(y, return_counts=True)
  n_samples_per_class[i] = n # non-existing classes maintain a zero entry
  return n_samples_per_class / n_samples_per_class.sum() # normalize to prevalences

def check_y(y, n_classes=None):
  """Emit a warning if the given labels are not sane."""
  if n_classes is not None:
    if n_classes != np.max(y)+1:
      warnings.warn(f"Classes are missing: n_classes != np.max(y)+1 = {np.max(y)+1}")


class AbstractLoss(ABC,BaseMixin):
  """Abstract base class for loss functions and for regularization terms."""
  @abstractmethod
  def instantiate(self, q, M, N):
    """This abstract method has to create a lambda expression `p -> loss` with JAX.

    In particular, your implementation of this abstract method should return a lambda expression

        >>> return lambda p: loss_value(q, M, p, N)

    where `loss_value` has to return the result of a JAX expression. The JAX requirement ensures that the loss function can be auto-differentiated. Hence, no derivatives of the loss function have to be provided manually. JAX expressions are easy to implement. Just import the numpy wrapper

        >>> import jax.numpy as jnp

    and use `jnp` just as if you would use numpy.

    Note:
        `p` is a vector of class-wise probabilities. This vector will already be the result of our soft-max trick, so that you don't have to worry about constraints or latent parameters.

    Args:
        q: A numpy array.
        M: A numpy matrix.
        N: The number of data items that `q` represents.

    Returns:
        A lambda expression `p -> loss`, implemented in JAX.

    Examples:
        The least squares loss, `(q - M*p)' * (q - M*p)`, is simply

            >>> jnp.dot(q - jnp.dot(M, p), q - jnp.dot(M, p))
    """
    pass

# helpers for TikhonovRegularization
def _tikhonov_matrix(C):
  return (
    jnp.diag(jnp.full(C, 2))
    + jnp.diag(jnp.full(C-1, -1), -1)
    + jnp.diag(jnp.full(C-1, -1), 1)
  )[1:C-1, :]
def _tikhonov(p, T):
  Tp = jnp.dot(T, p)
  return jnp.dot(Tp, Tp) / 2

class TikhonovRegularization(AbstractLoss):
  """Tikhonov regularization, as proposed by Blobel (1985).

  This regularization promotes smooth solutions. This behavior is often required in ordinal quantification and in unfolding problems.
  """
  def instantiate(self, q, M, N):
    T = _tikhonov_matrix(M.shape[1])
    return lambda p: _tikhonov(p, T)


class AbstractRepresentation(ABC,BaseMixin):
  """Abstract base class for representations."""
  @abstractmethod
  def fit_transform(self, X, y, sample_weight=None, average=True, n_classes=None):
    """This abstract method has to fit the representation and to return the transformed input data.

    Note:
        Implementations of this abstract method should check the sanity of labels by calling `check_y(y, n_classes)` and they must set the property `self.p_trn = class_prevalences(y, n_classes)`.

    Args:
        X: The feature matrix to which this representation will be fitted.
        y: The labels to which this representation will be fitted.
        sample_weight (optional): Importance weights for each (X[i], y[i]) pair to use during fitting. Defaults to `None`.
        average (optional): Whether to return a transfer matrix `M` or a transformation `f(X)`. Defaults to `True`.
        n_classes (optional): The number of expected classes. Defaults to `None`.

    Returns:
        A transfer matrix `M` if `average==True` or a transformation `f(X)` if `average==False`. `f(X)` might contain `NaN` entries if the fitting permits the estimation, e.g., in bootstrapped fitting procedures.
    """
    pass
  @abstractmethod
  def transform(self, X, sample_weight=None, average=True):
    """This abstract method has to transform the data `X`.

    Args:
        X: The feature matrix that will be transformed.
        sample_weight (optional): Importance weights for each X[i] to use during averaging if `average==True`. Defaults to `None`.
        average (optional): Whether to return a vector `q` or a transformation `f(X)`. Defaults to `True`.

    Returns:
        A vector `q = f(X).average(axis=0, weights=sample_weight)` if `average==True` or a transformation `f(X)` if `average==False`.
    """
    pass
  @property
  @abstractmethod
  def n_output_features(self):
    """The number of output features generated by this representation."""
    pass
