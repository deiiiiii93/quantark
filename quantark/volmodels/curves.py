"""Re-exports: grid samplers moved to quantark.param.term_sampling.

Kept so existing imports (engines, tests) continue to work.
"""

from quantark.param.term_sampling import (  # noqa: F401
    _validate_grid,
    forward_carry_on_grid,
    forward_rates_on_grid,
)
