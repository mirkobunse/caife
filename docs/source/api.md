# API

This page documents the public API of `caife`, organized by the kind of unfolding component that each symbol implements.


## Models and latent vectors

```{eval-rst}
.. autoclass:: caife.AbstractModel
   :members:

.. autoclass:: caife.LinearCountModel

.. autoclass:: caife.LinearSystematicsCountModel

.. autoclass:: caife.MultiTargetModel

.. autoclass:: caife.MultiTargetSystematicsModel

.. autofunction:: caife.create_mixture_model_fn

.. autoclass:: caife.AbstractLatentVector
   :members:

.. autoclass:: caife.LatentSpectrum

.. autoclass:: caife.LatentReshape

.. autoclass:: caife.LatentSystematics
```

## Representations

```{eval-rst}
.. autoclass:: caife.TreeBinning
   :members:

.. autoclass:: caife.UnivariateBinning
   :members:

.. autoclass:: caife.GridBinning
   :members:

.. autoclass:: caife.GridSearchRepresentation
   :members:
```

## Solvers

```{eval-rst}
.. autoclass:: caife.AbstractSolver
   :members:

.. autoclass:: caife.ScipySolver
```

## Loss functions

```{eval-rst}
.. autofunction:: caife.poisson_nll

.. autofunction:: caife.tikhonov_regularization
```

## Evaluation

```{eval-rst}
.. autofunction:: caife.global_correlation_coefficients

.. autofunction:: caife.pairwise_correlation_scores

.. autofunction:: caife.effective_number_of_degrees_of_freedom

.. autofunction:: caife.logarithmic_earth_movers_distance

.. autofunction:: caife.gaussian_nll_score
```

## Uncertainty estimation

```{eval-rst}
.. autofunction:: caife.uncertainty_from_hessian

.. autofunction:: caife.uncertainty_from_aux
```
