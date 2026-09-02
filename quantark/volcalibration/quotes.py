"""Convention-neutral quote container: the point where every normalizer converges.

``listed.py`` (strike-quoted books) and ``fxdelta.py`` (delta-quoted books) do
different work but produce the same :class:`QuoteSet`, so everything downstream
-- smoothing, admission, model calibration -- never branches on convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError


def _finite_positive(value: float, what: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{what} must be positive and finite, got {out}")
    return out


@dataclass(frozen=True)
class IvNode:
    """One (strike, implied vol) observation plus its fit weight."""

    strike: float
    iv: float
    weight_hint: float

    def __post_init__(self) -> None:
        _finite_positive(self.strike, "IvNode.strike")
        _finite_positive(self.iv, "IvNode.iv")
        weight = float(self.weight_hint)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValidationError(
                f"IvNode.weight_hint must be non-negative and finite, got {weight}"
            )


@dataclass(frozen=True)
class ExpiryQuotes:
    """One expiry's carry pillars and IV nodes.

    ``expiry_label`` always identifies the slice (an expiry date for listed
    books, a tenor such as ``"3M"`` for FX).  ``expiry_date`` is None when the
    snapshot carries no calendar date; deriving one would require an expiry
    calendar and adjustment convention this module does not own.
    """

    expiry_label: str
    expiry_date: Optional[str]
    T: float
    forward: float
    discount_factor: float
    r: float
    q: float
    nodes: Tuple[IvNode, ...]
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not str(self.expiry_label).strip():
            raise ValidationError("ExpiryQuotes.expiry_label must be non-empty")
        _finite_positive(self.T, "ExpiryQuotes.T")
        _finite_positive(self.forward, "ExpiryQuotes.forward")
        _finite_positive(self.discount_factor, "ExpiryQuotes.discount_factor")
        for name in ("r", "q"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValidationError(f"ExpiryQuotes.{name} must be finite, got {value}")
        if not self.nodes:
            raise ValidationError(
                f"ExpiryQuotes {self.expiry_label}: at least one IV node required"
            )
        strikes = [n.strike for n in self.nodes]
        if any(a >= b for a, b in zip(strikes, strikes[1:])):
            raise ValidationError(
                f"ExpiryQuotes {self.expiry_label}: nodes must be strike-ordered "
                "and unique"
            )


@dataclass(frozen=True)
class QuoteSet:
    """One trading date's expiries, normalized out of any quoting convention.

    ``universe`` is the normalizer's audit record -- how many quotes it saw,
    what it filtered and why, which expiries it excluded.  It is carried here
    rather than returned separately because it describes exactly this QuoteSet,
    and the artifact writer persists it so an excluded quote is always
    accounted for rather than silently missing.
    """

    trade_date: date
    spot: float
    convention: str
    expiries: Tuple[ExpiryQuotes, ...]
    universe: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite_positive(self.spot, "QuoteSet.spot")
        if not self.expiries:
            raise ValidationError("QuoteSet requires at least one expiry")
        times = [e.T for e in self.expiries]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValidationError(
                "QuoteSet expiry maturities must be strictly increasing; got "
                f"{times}"
            )
