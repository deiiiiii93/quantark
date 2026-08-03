"""Asset-agnostic Monte Carlo / quasi-MC primitives.

Relocated from quantark.asset.equity.process.bsm so both equity and FX engines
can depend on shared infrastructure. The legacy equity paths remain valid via
re-export shims.
"""

from .qmc_sobol import (
    RandomStream,
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from .qmc_brownian_bridge import (
    BrownianBridge,
    apply_brownian_bridge,
    apply_brownian_bridge_multi_asset,
    compute_step_crossing_probabilities,
    compute_barrier_crossing_probabilities,
)
from .qmc_variance_reduction import (
    VarianceReductionConfig,
    build_antithetic_pairs,
    apply_importance_sampling_shift,
    importance_sampling_weights,
    gbm_control_variate,
    apply_variance_reduction_to_normals,
)
from .conditional_snowball import (
    ConditionalSnowballMoments,
    conditional_standard_snowball_moments,
)
from .qmc_qe_coupling import (
    CoupledQESubstepDrawProvider,
    invert_brownian_bridge,
)
from .qmc_rqmc_driver import (
    PairedRQMCGreeksResult,
    RQMCCheckpoint,
    RQMCResult,
    RQMCRunSpec,
    run_rqmc,
    run_rqmc_traced,
    run_paired_rqmc_greeks,
)

__all__ = [
    "RandomStream",
    "PseudoRandomNormalGenerator",
    "SobolNormalGenerator",
    "BrownianBridge",
    "apply_brownian_bridge",
    "apply_brownian_bridge_multi_asset",
    "compute_step_crossing_probabilities",
    "compute_barrier_crossing_probabilities",
    "VarianceReductionConfig",
    "build_antithetic_pairs",
    "apply_importance_sampling_shift",
    "importance_sampling_weights",
    "gbm_control_variate",
    "apply_variance_reduction_to_normals",
    "ConditionalSnowballMoments",
    "conditional_standard_snowball_moments",
    "CoupledQESubstepDrawProvider",
    "invert_brownian_bridge",
    "RQMCCheckpoint",
    "PairedRQMCGreeksResult",
    "RQMCResult",
    "RQMCRunSpec",
    "run_rqmc",
    "run_rqmc_traced",
    "run_paired_rqmc_greeks",
]
