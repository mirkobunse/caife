"""caife: Composable and Auto-differentiable Inversion of Fredholm Equations.

caife is a Python package for unfolding, the inverse problem of recovering a target distribution from a smeared, indirectly observed proxy distribution. It provides composable models of the measurement process, representations of the proxy feature space, solvers, and utilities for evaluating and estimating the uncertainty of unfolding results.
"""
__version__ = "0.0.4-rc6"

from jax import config

# necessary for successful scipy.minimize
config.update("jax_enable_x64", True) # TODO can we enable 64 bits locally, through dtypes?


from .losses import (  # noqa: I001
    poisson_nll as poisson_nll,
    tikhonov_regularization as tikhonov_regularization,
)

from .models import (
    AbstractModel as AbstractModel,
    create_mixture_model_fn as create_mixture_model_fn,
)

from .models.latents import (
    AbstractLatentVector as AbstractLatentVector,
    LatentReshape as LatentReshape,
    LatentSpectrum as LatentSpectrum,
    LatentSystematics as LatentSystematics,
)

from .models.representations import (
    TreeBinning as TreeBinning,
    UnivariateBinning as UnivariateBinning,
    GridBinning as GridBinning,
    GridSearchRepresentation as GridSearchRepresentation,
)

from .models.linear import (
    LinearCountModel as LinearCountModel,
    LinearSystematicsCountModel as LinearSystematicsCountModel,
)

from .models.multitarget import (
    MultiTargetModel as MultiTargetModel,
    MultiTargetSystematicsModel as MultiTargetSystematicsModel,
)

from .solvers import (
    AbstractSolver as AbstractSolver,
)

from .solvers.scipy import (
    ScipySolver as ScipySolver,
)

from .evaluation import (
    global_correlation_coefficients as global_correlation_coefficients,
    pairwise_correlation_scores as pairwise_correlation_scores,
    effective_number_of_degrees_of_freedom as effective_number_of_degrees_of_freedom,
    logarithmic_earth_movers_distance as logarithmic_earth_movers_distance,
    gaussian_nll_score as gaussian_nll_score,
)

from .uncertainties import (
    uncertainty_from_hessian as uncertainty_from_hessian,
    uncertainty_from_aux as uncertainty_from_aux,
)
