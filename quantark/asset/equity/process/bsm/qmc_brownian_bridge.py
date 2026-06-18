"""Back-compat shim. Canonical home: quantark.montecarlo.qmc_brownian_bridge."""
from quantark.montecarlo.qmc_brownian_bridge import *  # noqa: F401,F403
from quantark.montecarlo.qmc_brownian_bridge import (  # noqa: F401
    BrownianBridge,
    apply_brownian_bridge,
    apply_brownian_bridge_multi_asset,
    compute_step_crossing_probabilities,
    compute_barrier_crossing_probabilities,
)
from quantark.montecarlo.qmc_brownian_bridge import __all__  # noqa: F401
