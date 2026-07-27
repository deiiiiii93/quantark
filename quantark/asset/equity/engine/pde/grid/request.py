"""Declarative grid geometry — the ONLY product/market input the builders see.

``GridRequest`` is pure geometry (hashable, cacheable, freezable across bumps);
``MarketSnapshot`` is the minimal market state the spatial builder consumes.
Everything event-semantic (transforms, payoffs) lives in ``events.EventSchedule``,
built per solve — never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


def _dedup_sorted_interior(times, tau: float) -> Tuple[float, ...]:
    """Sorted, is_close-deduplicated event times, strictly inside (0, tau).

    Endpoint events are structurally excluded: t=0 semantics belong to the
    ``valuation_readout`` stage and t=tau to the ``terminal`` stage (spec §4.2).
    """
    out: list[float] = []
    for t in sorted(float(t) for t in times):
        if t <= 0.0 or t >= tau or is_close(t, 0.0) or is_close(t, tau):
            raise ValidationError(
                f"event_times must lie strictly inside (0, {tau}); got {t} "
                "(endpoint events belong to the terminal/valuation_readout stages)"
            )
        if out and is_close(out[-1], t):
            continue
        out.append(t)
    return tuple(out)


@dataclass(frozen=True)
class MarketSnapshot:
    """Minimal market state consumed by the spatial builder (spec §4.2)."""

    spot: float
    sigma_ref: float
    r_ref: float
    q_ref: float

    def __post_init__(self):
        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}")
        if self.sigma_ref <= 0.0:
            raise ValidationError(
                f"sigma_ref must be positive, got {self.sigma_ref}"
            )


@dataclass(frozen=True)
class GridRequest:
    """Frozen geometry declaration for one product (spec §4.2).

    Attributes:
        tau: Time to maturity in years (> 0).
        bound_anchors: Centers of the ±h auto-bounds envelope (spot, strike).
        critical_prices: Concentration targets (barriers, strike, spot).
        hard_lower: Absorbing lower domain edge (continuous KO/touch), or None.
        hard_upper: Absorbing upper domain edge, or None.
        event_times: ALL interior event-bearing dates (KO, coupon, discrete KI)
            — each becomes an exact grid node, indexed and damped.
    """

    tau: float
    bound_anchors: Tuple[float, ...]
    critical_prices: Tuple[float, ...]
    hard_lower: Optional[float]
    hard_upper: Optional[float]
    event_times: Tuple[float, ...]

    def __post_init__(self):
        if self.tau <= 0.0:
            raise ValidationError(f"tau must be positive, got {self.tau}")
        for name in ("bound_anchors", "critical_prices"):
            vals = tuple(float(p) for p in getattr(self, name))
            if any(p <= 0.0 for p in vals):
                raise ValidationError(f"{name} must be positive, got {vals}")
            object.__setattr__(self, name, vals)
        if not self.bound_anchors:
            raise ValidationError(
                "bound_anchors must contain at least one price (spot)"
            )
        for side in ("hard_lower", "hard_upper"):
            v = getattr(self, side)
            if v is not None:
                v = float(v)
                if v <= 0.0:
                    raise ValidationError(f"{side} must be positive, got {v}")
                object.__setattr__(self, side, v)
        if (
            self.hard_lower is not None
            and self.hard_upper is not None
            and self.hard_lower >= self.hard_upper
        ):
            raise ValidationError(
                f"hard_lower ({self.hard_lower}) must be < "
                f"hard_upper ({self.hard_upper})"
            )
        object.__setattr__(
            self, "event_times", _dedup_sorted_interior(self.event_times, self.tau)
        )
