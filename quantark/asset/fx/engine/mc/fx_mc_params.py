"""Monte Carlo configuration for FX MC engines (barrier / sharkfin / TARN).

Standard-error semantics by method:
  - PSEUDO: pathwise sample standard error s/sqrt(N).
  - QUASI (deterministic Sobol): pathwise dispersion is NOT a valid error
    estimate; engines report std_error = None.
  - RANDOMIZED_QUASI: error from independent randomized batches (deferred in the
    barrier foundation phase).
"""

from dataclasses import dataclass

from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError


@dataclass
class FxMCParams:
    """Path count, RNG and method configuration for FX Monte Carlo engines.

    Attributes:
        num_paths: Total paths (PSEUDO/QUASI); paths-per-batch (RQMC).
        time_steps: Grid resolution for continuous monitoring.
        seed: RNG seed (fixed seed gives common random numbers across bumps).
        use_antithetic: Antithetic variates (MC only; ignored for QMC).
        method: PSEUDO / QUASI / RANDOMIZED_QUASI.
        rqmc_min_batches, rqmc_max_batches, target_std: RQMC controls.
    """

    num_paths: int = 200_000
    time_steps: int = 100
    seed: int = 42
    use_antithetic: bool = True
    method: MonteCarloMethod = MonteCarloMethod.PSEUDO
    rqmc_min_batches: int = 4
    rqmc_max_batches: int = 32
    target_std: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {self.num_paths}")
        if self.time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {self.time_steps}")
        if not isinstance(self.method, MonteCarloMethod):
            raise ValidationError(
                f"method must be a MonteCarloMethod, got {type(self.method).__name__}"
            )
