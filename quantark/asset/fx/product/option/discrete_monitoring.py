"""
Discrete barrier fixings of an FX option, on either clock.

A numeric (``maturity``) product books its fixings as ``observation_times``, year fractions from the valuation date.
A dated (``expiry_date``) product books them as ``observation_dates``, resolved against the pricing environment's
valuation date and day count like ``expiry_date`` is. Both given: the dates win and the floats are ignored
(clock rule 2026-10-26).

Mid-life (user, 2026-10-26): a dated product records the realised spot of each fixing on or before the valuation
date in ``observed_fixings`` (``{observation date: spot}``). ``monitoring_state`` decides the past: the first fixing
that breached the barrier knocked the contract (out or in) at that fixing's time; otherwise only the fixings ahead
are left to simulate. A past fixing without a record is decided at the current spot, with one
``MissingFixingWarning`` naming every STRICTLY past one (today's fixing is live, filled silently), as the FX range
accrual does (user ruling 2026-10-09). A numeric product keys its fixings by time and ages them with the clock
(``time_shift``), so theta steps over a fixing on either clock alike.
"""

import copy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import is_history, snap_time, warn_filled_fixings
from ..base_fx_product import BaseFxProduct

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


Key = Union[datetime, float]


class DiscreteMonitoring(BaseFxProduct):
    """An FX product with ``observation_times`` / ``observation_dates`` barrier fixings.

    ``observed_fixings`` keys a recorded fixing by its observation date (date clock) or its observation time (numeric
    clock). A numeric product ages its fixing times with the clock (``time_shift``); a time on or below zero is a
    fixing on or before today, and booking one needs its recorded fixing.
    """

    observation_times: Optional[List[float]]
    observation_dates: Optional[List[datetime]]
    observed_fixings: Optional[Dict[Key, float]]
    barrier: float
    is_up: bool

    def _validate_observation_times(self) -> None:
        self._validate_observed_fixings()
        if self.observation_dates:
            self._validate_dates(list(self.observation_dates))
            return
        times = self.observation_times
        if not times:
            raise ValidationError(
                "DISCRETE monitoring requires a non-empty observation_times (or observation_dates) list"
            )
        self._validate_ordering(list(times), "observation_times")
        T = self.maturity
        recorded = self.observed_fixings or {}
        for t in times:
            if t <= 0.0 and t not in recorded:
                raise ValidationError(
                    "observation_times must be strictly positive (a fixing on or before today needs its "
                    "observed_fixings entry)"
                )
            if T is not None and t > T:
                raise ValidationError("observation_times must not exceed maturity")

    def _validate_dates(self, dates: List[datetime]) -> None:
        self._validate_ordering(dates, "observation_dates")
        if self.expiry_date is not None and dates[-1] > self.expiry_date:
            raise ValidationError("observation_dates must not be after expiry_date")

    def _validate_observed_fixings(self) -> None:
        fixings = self.observed_fixings
        if not fixings:
            return
        keys = set(self.observation_dates or self.observation_times or ())
        noun = "date" if self.observation_dates else "time"
        for key, spot in fixings.items():
            if key not in keys:
                raise ValidationError(f"observed_fixings: {key!r} is not an observation {noun}")
            if not spot > 0.0:
                raise ValidationError(f"observed_fixings: the fixing at {key!r} must be positive, got {spot}")

    @staticmethod
    def _validate_ordering(values: list, name: str) -> None:
        if values != sorted(values):
            raise ValidationError(f"{name} must be ascending")
        if len(set(values)) != len(values):
            raise ValidationError(f"{name} must be unique")

    def time_shift(self, time_bump: float) -> None:
        """Age a numeric product for theta: maturity, delivery and every fixing time (and its recorded fixing's key)
        move with the clock, a time landing on today snapped to exactly 0.0. A fixing stepped over stays unrecorded
        until ``record_passed_fixings`` fixes it. A dated product ages by its environment's date (no change)."""
        numeric = self.maturity is not None and not self.observation_dates and self.observation_times
        super().time_shift(time_bump)
        if not numeric:
            return
        moved = {t: snap_time(t - time_bump) for t in self.observation_times}
        self.observation_times = [moved[t] for t in self.observation_times]
        if self.observed_fixings:
            self.observed_fixings = {moved[t]: spot for t, spot in self.observed_fixings.items()}

    def observation_times_at(self, fx_env: Optional["FxPricingEnvironment"] = None) -> List[float]:
        """The times, in years from ``fx_env``'s valuation date, of the fixings still ahead (strictly positive): a
        numeric product's ``observation_times``; a dated product's ``observation_dates`` resolved on the
        environment's day count. The past is ``monitoring_state``'s."""
        return [t for _, t in self._fixings(fx_env) if t > 0.0]

    def _fixings(self, fx_env: Optional["FxPricingEnvironment"]) -> List[Tuple[Key, float]]:
        """``(key, time)`` of every fixing, the time in years from the valuation date (<= 0 on or before it)."""
        if self.observation_dates:
            if fx_env is None:
                raise ValidationError("FxPricingEnvironment required to resolve observation_dates")
            return [(when, self._fixing_time(when, fx_env)) for when in self.observation_dates]
        return [(t, float(t)) for t in self.observation_times or ()]

    @staticmethod
    def _label(index: int, key: Key) -> str:
        return f"#{index} {key.date().isoformat()}" if isinstance(key, datetime) else f"#{index} t={key:.6g}"

    def _breached(self, spot: float) -> bool:
        return spot >= self.barrier if self.is_up else spot <= self.barrier

    def _fixing_time(self, when: datetime, fx_env: "FxPricingEnvironment") -> float:
        """Years from the valuation date to ``when``: negative before it, 0.0 on it."""
        if when >= fx_env.valuation_date:
            return self._year_fraction_to(when, fx_env)
        return -calculate_year_fraction(when, fx_env.valuation_date, fx_env.day_count_convention,
                                        fx_env.bus_days_in_year, calendar=fx_env.calendar)

    def monitoring_state(
        self, fx_env: "FxPricingEnvironment", *, warn_missing_fixings: bool = True
    ) -> Tuple[Optional[float], List[float]]:
        """``(hit_time, ahead)``: the time (<= 0) of the first fixing on or before the valuation date that breached
        the barrier, or ``None``; and the times of the fixings still ahead (empty once the past has decided).

        A past fixing is read from ``observed_fixings``, else decided at ``fx_env.spot`` (see the module docstring);
        a fixing ahead must not carry one. The product is not changed.
        """
        fixings = self.observed_fixings or {}
        filled: List[str] = []
        ahead: List[float] = []
        hit_time: Optional[float] = None
        for index, (key, t) in enumerate(self._fixings(fx_env)):
            if t > 0.0:
                if key in fixings:
                    raise ValidationError(f"observed_fixings: the future fixing at {key!r} must not carry an outcome")
                ahead.append(t)
                continue
            spot = fixings.get(key)
            if spot is None:
                spot = float(fx_env.spot)
                if is_history(t):
                    filled.append(self._label(index, key))
            if self._breached(spot):
                hit_time = t
                break
        if warn_missing_fixings:
            warn_filled_fixings(type(self).__name__, "observed_fixings", filled, stacklevel=3)
        return (hit_time, []) if hit_time is not None else (None, ahead)

    def record_passed_fixings(self, fx_env: "FxPricingEnvironment") -> None:
        """Record every unrecorded fixing on or before ``fx_env``'s valuation at its spot (a theta bump holds the
        spot, so the fixings it steps over fix there)."""
        fixings = dict(self.observed_fixings or {})
        for key, t in self._fixings(fx_env):
            if t <= 0.0 and key not in fixings:
                fixings[key] = float(fx_env.spot)
        if fixings != (self.observed_fixings or {}):
            self.observed_fixings = fixings

    def missing_fixing_fills(self, fx_env: "FxPricingEnvironment") -> Dict[Key, float]:
        """``{key: spot}`` for every STRICTLY past fixing without a record, at ``fx_env.spot`` (today's is live)."""
        fixings = self.observed_fixings or {}
        return {key: float(fx_env.spot) for key, t in self._fixings(fx_env) if is_history(t) and key not in fixings}

    def with_missing_fixings_filled(self, fx_env: "FxPricingEnvironment", *, warn: bool = True):
        """A COPY with every strictly past fixing lacking a record fixed at ``fx_env``'s spot, so bumped and shocked
        repricings keep the base fill (``quantark.util.fixings.freeze_missing_fixings``); ``self`` when none."""
        fills = self.missing_fixing_fills(fx_env)
        if not fills:
            return self
        if warn:
            keys = [key for key, _ in self._fixings(fx_env)]
            labels = [self._label(keys.index(key), key) for key in fills]
            warn_filled_fixings(type(self).__name__, "observed_fixings", labels, stacklevel=3)
        frozen = copy.deepcopy(self)
        frozen.observed_fixings = {**(self.observed_fixings or {}), **fills}
        return frozen
