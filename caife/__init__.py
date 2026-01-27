__version__ = "0.0.2"

# necessary for successful scipy.minimize
from jax import config
config.update("jax_enable_x64", True) # TODO can we enable 64 bits locally, through dtypes?


from .losses import (
    poisson_nll,
    tikhonov_regularization,
)

from .models.representations import (
    TreeBinning,
    UnivariateBinning,
    GridSearchRepresentation,
)

from .models.linear import (
    AbstractModel,
    LinearCountModel,
    LinearSystematicsCountModel,
)

from .models.model_selection import (
    total_correlation_score,
    logarithmic_earth_movers_distance,
)

from .solvers import (
    AbstractSolver,
    AbstractLatentVector,
    LatentSpectrum,
    LatentSystematics,
)

from .solvers.scipy import (
    ScipySolver,
)

from .uncertainties import (
    uncertainty_from_hessian,
)
