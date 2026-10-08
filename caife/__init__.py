"""caife: Composable and Auto-differentiable Inversion of Fredholm Equations.

caife is a Python package for unfolding, the inverse problem of recovering a target distribution from a smeared, indirectly observed proxy distribution. It provides composable models of the measurement process, representations of the proxy feature space, solvers, and utilities for evaluating and estimating the uncertainty of unfolding results.

Importing caife enables 64-bit precision in JAX because unfolding requires this precision to produce meaningful results. Please note that this setting affects all JAX code in your Python process, not only caife, and that float64 computations can be considerably slower on some GPUs than float32 computations. To keep 32 bits, set the environment variable `JAX_ENABLE_X64=0` before importing caife; caife respects any explicit value of this variable.
"""
__version__ = "0.0.5.dev2"

import logging
import os

from jax import config

# enable 64 bits, which are necessary for meaningful unfolding results
if "JAX_ENABLE_X64" not in os.environ and not config.jax_enable_x64:
    config.update("jax_enable_x64", True)
    logging.getLogger(__name__).info(
        "caife has enabled 64-bit precision in JAX, which affects all JAX code in this process. To keep 32 bits, set the environment variable JAX_ENABLE_X64=0 before importing caife.")


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
    gcc_from_nll as gcc_from_nll,
    pcs_from_nll as pcs_from_nll,
    gcc_from_aux as gcc_from_aux,
    pcs_from_aux as pcs_from_aux,
    effective_number_of_degrees_of_freedom as effective_number_of_degrees_of_freedom,
    logarithmic_earth_movers_distance as logarithmic_earth_movers_distance,
    gaussian_nll_score as gaussian_nll_score,
)

from .uncertainties import (
    uncertainty_from_hessian as uncertainty_from_hessian,
    uncertainty_from_aux as uncertainty_from_aux,
)
