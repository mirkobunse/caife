__version__ = "0.0.4-rc2"

# necessary for successful scipy.minimize
from jax import config
config.update("jax_enable_x64", True) # TODO can we enable 64 bits locally, through dtypes?


from .losses import (
    poisson_nll,
    tikhonov_regularization,
)

from .models import (
    AbstractModel,
    create_mixture_model_fn,
)

from .models.latents import (
    AbstractLatentVector,
    LatentReshape,
    LatentSpectrum,
    LatentSystematics,
)

from .models.representations import (
    TreeBinning,
    UnivariateBinning,
    GridBinning,
    GridSearchRepresentation,
)

from .models.collections import (
    SeparateSystematicBinCollection,
    JointSystematicBinCollection,
)

from .models.linear import (
    LinearCountModel,
    LinearSystematicsCountModel,
)

from .models.multitarget import (
    MultiTargetModel,
)

from .solvers import (
    AbstractSolver,
)

from .solvers.scipy import (
    ScipySolver,
)

from .evaluation import (
    global_correlation_coefficients,
    pairwise_correlation_scores,
    effective_number_of_degrees_of_freedom,
    logarithmic_earth_movers_distance,
    gaussian_nll_score,
)

from .uncertainties import (
    uncertainty_from_hessian,
)
