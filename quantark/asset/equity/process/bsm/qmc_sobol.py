"""Back-compat shim. Canonical home: quantark.montecarlo.qmc_sobol."""
from quantark.montecarlo.qmc_sobol import *  # noqa: F401,F403
from quantark.montecarlo.qmc_sobol import (  # noqa: F401
    RandomStream,
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.montecarlo.qmc_sobol import __all__  # noqa: F401
