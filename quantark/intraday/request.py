"""The typed intraday valuation request."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional, Tuple

from quantark.intraday.events import EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.session import TradingSessionCalendar
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

GREEK_CONVENTIONS = ("point", "desk_bump")
THETA_UNITS = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}


@dataclass(frozen=True)
class IntradayValuationRequest:
    """Everything one intraday valuation depends on.

    ``pricing_env.valuation_date`` IS the timezone-aware valuation timestamp and
    ``pricing_env.vol_surface`` is the INNER trading-quoted surface; the
    resolver wraps it with the variance profile's clock.
    """

    product: object
    pricing_env: PricingEnvironment
    session_calendar: TradingSessionCalendar
    variance_profile: VarianceProfile
    lifecycle_state: Optional[object] = None
    fixings: Tuple[Fixing, ...] = ()
    event_phase: EventPhase = EventPhase.BEFORE
    greeks: Tuple[str, ...] = ()
    greek_convention: Optional[str] = None     # "point" | "desk_bump"; required when greeks are requested
    theta_step: Optional[timedelta] = None
    theta_unit: str = "hour"
    fixing_time_of_day: Optional[time] = None
    schedule_origin: Optional[datetime] = None
    request_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "event_phase", EventPhase.parse(self.event_phase))
        object.__setattr__(self, "fixings", tuple(self.fixings))
        for f in self.fixings:
            if not isinstance(f, Fixing):
                raise ValidationError(f"fixings must be Fixing objects, got {type(f).__name__}")
        object.__setattr__(self, "greeks", tuple(str(g) for g in self.greeks))
        if self.greeks:
            if self.greek_convention not in GREEK_CONVENTIONS:
                raise ValidationError(f"greek_convention must be one of {GREEK_CONVENTIONS} when greeks are requested")
        elif self.greek_convention is not None and self.greek_convention not in GREEK_CONVENTIONS:
            raise ValidationError(f"unknown greek_convention {self.greek_convention!r}")
        if self.theta_unit not in THETA_UNITS:
            raise ValidationError(f"theta_unit must be one of {sorted(THETA_UNITS)}")
        if self.theta_step is not None and (not isinstance(self.theta_step, timedelta) or self.theta_step <= timedelta(0)):
            raise ValidationError("theta_step must be a positive timedelta")
