"""Forward exposure time grid (spec §3.2).

Uniform exposure nodes unioned with product event dates (monitoring / coupon /
KO / final settlement), run to final settlement so maturity-zeroing and pending
receivables are representable. Times are year-fractions from valuation (t0=0).
"""

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class ExposureGrid:
    times: np.ndarray

    @staticmethod
    def build(horizon: float, n_steps: int, event_times) -> "ExposureGrid":
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
            raise ValidationError("horizon must be numeric")
        if not (horizon > 0) or not np.isfinite(horizon):
            raise ValidationError(f"horizon must be > 0, got {horizon}")
        if n_steps < 1:
            raise ValidationError("n_steps must be >= 1")
        base = np.linspace(0.0, float(horizon), n_steps + 1)
        ev = np.array([float(t) for t in event_times if 0.0 < float(t) <= horizon],
                      dtype=float)
        times = np.unique(np.concatenate([base, ev, [0.0]]))
        return ExposureGrid(times=times)
