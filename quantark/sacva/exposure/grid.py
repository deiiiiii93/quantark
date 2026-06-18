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

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float).copy()
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            raise ValidationError("times must be a finite non-empty 1-D array")
        if t[0] != 0.0:
            raise ValidationError("times must start at the valuation origin (t0=0)")
        if t.size > 1 and np.any(np.diff(t) <= 0):
            raise ValidationError("times must be strictly increasing")
        t.flags.writeable = False
        object.__setattr__(self, "times", t)

    @staticmethod
    def build(horizon: float, n_steps: int, event_times) -> "ExposureGrid":
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
            raise ValidationError("horizon must be numeric")
        if not (horizon > 0) or not np.isfinite(horizon):
            raise ValidationError(f"horizon must be > 0, got {horizon}")
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise ValidationError("n_steps must be an int")
        if n_steps < 1:
            raise ValidationError("n_steps must be >= 1")
        try:
            ev_raw = [float(t) for t in event_times]
        except (TypeError, ValueError):
            raise ValidationError("event_times must be an iterable of numbers")
        if any(not np.isfinite(t) for t in ev_raw):
            raise ValidationError("event_times must be finite")
        # fail fast on out-of-range events: silently dropping a coupon/KO/settlement
        # date past the horizon would understate exposure
        if any(t < 0.0 or t > horizon for t in ev_raw):
            raise ValidationError(
                f"event_times must lie in [0, horizon={horizon}]; widen the horizon")
        base = np.linspace(0.0, float(horizon), n_steps + 1)
        ev = np.array([t for t in ev_raw if 0.0 < t <= horizon], dtype=float)
        times = np.unique(np.concatenate([base, ev, [0.0]]))
        return ExposureGrid(times=times)
