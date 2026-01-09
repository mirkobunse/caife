# API

TODO.


## Models

```{eval-rst}
.. autoclass:: caife.AbstractModel

.. autoclass:: caife.LinearCountModel

.. autoclass:: caife.LinearSystematicsCountModel
```

## Representations

```{eval-rst}
.. autoclass:: caife.TreeBinning

.. autoclass:: caife.UnivariateBinning

.. autoclass:: caife.GridSearchRepresentation
```

## Solvers

```{eval-rst}
.. autoclass:: caife.AbstractSolver

.. autoclass:: caife.ScipySolver

.. autoclass:: caife.AbstractLatentVector

.. autoclass:: caife.LatentSpectrum

.. autoclass:: caife.LatentSystematics
```

## Loss functions

```{eval-rst}
.. autofunction:: caife.poisson_nll

.. autofunction:: caife.tikhonov_regularization
```

## Uncertainty estimation

```{eval-rst}
.. autofunction:: caife.uncertainty_from_hessian
```
