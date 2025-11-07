__version__ = "0.0.1"

# necessary for successful scipy.minimize
from jax import config
config.update("jax_enable_x64", True) # TODO can we enable 64 bits locally, through dtypes?


from .losses import (
    poisson_nll,
)

from .models.representations import (
    TreeBinning,
    UnivariateBinning,
    GridSearchRepresentation,
)

from .models.linear import (
    AbstractModel,
    LinearModel,
)

from .solvers import (
    AbstractSolver,
)

from .solvers.scipy import (
    ScipySolver,
)

from .losses import (
    poisson_nll,
    tikhonov_regularization,
)