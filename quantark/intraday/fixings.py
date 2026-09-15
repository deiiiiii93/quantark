"""Fixing inputs and the records of what intraday mode had to assume."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Optional, Tuple

from quantark.intraday.timestamp import require_aware
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class Fixing:
    """An observed underlying level at a contractual event instant."""

    timestamp: datetime        # aware; must equal an event instant of the timeline
    value: float               # > 0
    source: str = "confirmed"

    def __post_init__(self):
        require_aware(self.timestamp, "fixing timestamp")
        try:
            v = float(self.value)
        except (TypeError, ValueError):
            raise ValidationError(f"fixing value must be numeric, got {self.value!r}") from None
        if not isfinite(v) or v <= 0.0:
            raise ValidationError(f"fixing value must be finite and positive, got {self.value!r}")
        object.__setattr__(self, "value", v)


@dataclass(frozen=True)
class AssumedFixing:
    """A due fixing that was not supplied: the latest spot stood in for it."""

    event_ids: Tuple[str, ...]
    scheduled_at: datetime
    assumed_value: float
    spot_timestamp: datetime
    reason: str = "missing_fixing"


@dataclass(frozen=True)
class ContinuousHistoryAssumption:
    """A continuously monitored barrier whose touch history over an interval is unknown.

    The rule (design decision 8): no unreported touch inside the interval; the
    latest spot at the valuation instant is tested with the contract's own
    inclusive rule and, if it breaches, the hit is assumed at that instant.
    """

    uncovered_from: datetime
    uncovered_to: datetime
    assumed_hit_at: Optional[datetime]
