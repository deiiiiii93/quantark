"""
FX range accrual product.

A range accrual pays a coupon proportional to the fraction of fixings on which
the FX spot stays inside a ``[lower, upper]`` band:

    coupon = notional * accrual_rate * (in-range weight / total weight) * yf

The coupon is a fixed cash amount in the domestic (quote) currency, so each
fixing contributes a *cash-or-nothing range digital* ``P(L < X_t < U)``. The
engine prices it analytically by linearity of expectation over the fixings
(see :class:`FxRangeAccrualAnalyticalEngine`).

Scope: domestic-currency coupon only. A foreign-currency or quanto coupon
turns each fixing into an asset-or-nothing / quanto exposure (it would need the
joint law of the fixing rate and the payment-date rate, or an FX/FX
correlation) and is intentionally left to a future extension rather than
approximated here.
"""

import copy
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label, warn_filled_fixings
from quantark.util.numerical import is_zero

from ..base_fx_product import BaseFxProduct
from ..currency_pair import CurrencyPair
from .fx_range_accrual_config import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
)

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


class FxRangeAccrualOption(BaseFxProduct):
    """
    FX range accrual paying a domestic-currency coupon on in-range fixings.

    Provide the fixing schedule in exactly one of three ways:

    - ``observation_records``: explicit records (supports weights, per-fixing
      barriers, and realised past outcomes);
    - ``observation_times``: a list of fixing times as year fractions
      (weight 1.0 each);
    - ``num_observations``: a uniform schedule of N fixings over the tenor,
      built lazily against the product maturity.

    Attributes:
        notional: Coupon notional in the domestic (quote) currency.
        range_config: Barriers, accrual rate, pay type and reverse flag.
        observation_records / observation_times / num_observations: schedule.
    """

    def __init__(
        self,
        notional: float,
        range_config: FxRangeAccrualConfig,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
        observation_records: Optional[List[FxRangeAccrualObservationRecord]] = None,
        observation_times: Optional[List[float]] = None,
        num_observations: Optional[int] = None,
        initial_date: Optional[datetime] = None,
        tenor: Optional[float] = None,
    ):
        # Set before the base constructor validates. An annualized coupon accrues over the whole contract: ``tenor``,
        # or ``initial_date`` -> ``expiry_date``; on a numeric contract ``maturity`` plus the time ``time_shift`` has
        # aged it by, banked in ``accrued_offset``.
        self.initial_date = initial_date
        self.tenor = tenor
        self.accrued_offset = 0.0
        super().__init__(
            currency_pair=currency_pair,
            maturity=maturity,
            expiry_date=expiry_date,
            delivery=delivery,
            delivery_date=delivery_date,
        )
        self.notional = notional
        self.range_config = range_config
        self.observation_records = observation_records
        self.observation_times = observation_times
        self.num_observations = num_observations

        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the range accrual specification."""
        if self.notional <= 0:
            raise ValidationError(f"notional must be positive, got {self.notional}")
        if self.range_config is None:
            raise ValidationError("range_config is required")
        self._validate_maturity_inputs()
        self._validate_accrual_period()
        self._validate_schedule()
        self._validate_mid_life_booking()
        self._validate_barrier_lengths()

    def _validate_accrual_period(self) -> None:
        if self.tenor is not None and self.tenor <= 0:
            raise ValidationError(f"tenor must be positive, got {self.tenor}")
        if (
            self.initial_date is not None
            and self.expiry_date is not None
            and self.initial_date >= self.expiry_date
        ):
            raise ValidationError(
                f"initial_date ({self.initial_date}) must be before expiry_date ({self.expiry_date})"
            )
        # A dated contract that names no start would have to start its accrual at the valuation date, so its
        # coupon would shrink every day it ages; refuse it instead.
        if (
            self.range_config.is_rate_annualized
            and self.maturity is None
            and self.tenor is None
            and self.initial_date is None
        ):
            raise ValidationError(
                "An annualized dated FX range accrual needs initial_date or tenor: its coupon accrues over the "
                "whole contract, not from the valuation date"
            )

    def _validate_mid_life_booking(self) -> None:
        # A numeric contract with a fixing already behind it was booked mid-life: its maturity is only what is left,
        # so the contract's length cannot be read off it. Ageing (``time_shift``) banks what it removes; a booking
        # has to say how long the contract is.
        if (
            self.range_config.is_rate_annualized
            and self.maturity is not None
            and self.tenor is None
            and is_zero(self.accrued_offset)
            and self.observation_records is not None
            and any(rec.observation_time is not None and rec.observation_time < 0 for rec in self.observation_records)
        ):
            raise ValidationError(
                "An annualized numeric FX range accrual with past fixings needs tenor: its maturity is only the "
                "remaining life, and the coupon accrues over the whole contract"
            )

    def _validate_schedule(self) -> None:
        has_records = self.observation_records is not None
        has_times = self.observation_times is not None
        has_num = self.num_observations is not None

        provided = sum((has_records, has_times, has_num))
        if provided == 0:
            raise ValidationError(
                "Provide one of observation_records, observation_times or "
                "num_observations"
            )
        if provided > 1:
            raise ValidationError(
                "observation_records, observation_times and num_observations "
                "are mutually exclusive"
            )

        if has_times:
            times = self.observation_times or []
            if len(times) == 0:
                raise ValidationError("observation_times cannot be empty")
            if any(t < 0 for t in times):
                raise ValidationError("observation_times must be non-negative")
            if list(times) != sorted(times):
                raise ValidationError("observation_times must be ascending")

        if has_records:
            records = self.observation_records or []
            if len(records) == 0:
                raise ValidationError("observation_records cannot be empty")
            for rec in records:
                rec.validate()

        if has_num:
            if self.num_observations is None or self.num_observations < 1:
                raise ValidationError(
                    f"num_observations must be >= 1, got {self.num_observations}"
                )
            if self.maturity is None:
                raise ValidationError(
                    "num_observations requires a year-fraction maturity to build "
                    "the uniform schedule"
                )

    def _validate_barrier_lengths(self) -> None:
        num_obs = self._expected_observation_count()
        if num_obs is None:
            return
        for name, barrier in (
            ("upper_barrier", self.range_config.upper_barrier),
            ("lower_barrier", self.range_config.lower_barrier),
        ):
            if isinstance(barrier, list) and len(barrier) != num_obs:
                raise ValidationError(
                    f"{name} list length ({len(barrier)}) must match the fixing "
                    f"count ({num_obs})"
                )

    def _expected_observation_count(self) -> Optional[int]:
        if self.observation_records is not None:
            return len(self.observation_records)
        if self.observation_times is not None:
            return len(self.observation_times)
        return self.num_observations

    # ------------------------------------------------------------------
    # Schedule resolution
    # ------------------------------------------------------------------

    def get_observation_records(self) -> List[FxRangeAccrualObservationRecord]:
        """Materialise the fixing schedule as observation records."""
        if self.observation_records is not None:
            return self.observation_records
        if self.observation_times is not None:
            return [
                FxRangeAccrualObservationRecord(observation_time=t, weight=1.0)
                for t in self.observation_times
            ]
        if self.num_observations is not None and self.maturity is not None:
            n = self.num_observations
            return [
                FxRangeAccrualObservationRecord(
                    observation_time=(i + 1) / n * self.maturity, weight=1.0
                )
                for i in range(n)
            ]
        raise ValidationError("Cannot materialise observation records")

    def resolve_observations(
        self, fx_env: "FxPricingEnvironment", *, warn_missing_fixings: bool = True
    ) -> Tuple[
        List[Tuple[float, bool]],
        List[Tuple[float, float, int]],
        float,
    ]:
        """
        Split fixings into past (settled) and future relative to valuation.

        Returns ``(past, future, total_weights)`` where ``past`` holds
        ``(weight, in_range)`` for fixings on/before the valuation date and
        ``future`` holds ``(weight, time, obs_idx)`` for fixings strictly after.

        A past fixing without ``observed_in_range`` is decided at ``fx_env.spot`` through ``is_in_range`` (its own
        barriers, reverse mode), as ``record_passed_fixings`` would fix it, and one ``MissingFixingWarning`` names
        every fixing so filled (``warn_missing_fixings=False`` skips it). The product is not changed. A future
        fixing must not carry an outcome.
        """
        records = self.get_observation_records()
        past: List[Tuple[float, bool]] = []
        future: List[Tuple[float, float, int]] = []
        total_weights = 0.0
        filled: List[str] = []

        for idx, rec in enumerate(records):
            t = rec.resolve_time(fx_env)
            total_weights += rec.weight
            if t <= 0:
                in_range = rec.observed_in_range
                if in_range is None:
                    # A missing past fixing is decided at the current spot (user ruling 2026-10-09)
                    in_range = self.is_in_range(fx_env.spot, idx, rec)
                    filled.append(fixing_label(idx, rec))
                past.append((rec.weight, in_range))
            else:
                if rec.observed_in_range is not None:
                    raise ValidationError(
                        f"Future fixing at index {idx} (t={t}) must not set "
                        "observed_in_range"
                    )
                future.append((rec.weight, t, idx))

        if warn_missing_fixings:
            warn_filled_fixings(type(self).__name__, "observed_in_range", filled)
        return past, future, total_weights

    def get_total_weights(self) -> float:
        """Sum of all fixing weights."""
        return sum(rec.weight for rec in self.get_observation_records())

    def get_past_accrual(self, fx_env: "FxPricingEnvironment") -> Tuple[float, float]:
        """Realised ``(in_range_weight, past_total_weight)`` from past fixings."""
        past, _, _ = self.resolve_observations(fx_env)
        in_range = sum(w for w, ok in past if ok)
        total = sum(w for w, _ in past)
        return in_range, total

    # ------------------------------------------------------------------
    # Barriers / payoff
    # ------------------------------------------------------------------

    def barriers_for(
        self,
        obs_idx: int,
        record: Optional[FxRangeAccrualObservationRecord] = None,
    ) -> Tuple[float, float]:
        """Resolved ``(lower, upper)`` for a fixing, preferring per-fixing values."""
        if record is not None and record.lower_barrier is not None:
            lower = record.lower_barrier
        else:
            lower = self.range_config.get_lower_barrier(obs_idx)
        if record is not None and record.upper_barrier is not None:
            upper = record.upper_barrier
        else:
            upper = self.range_config.get_upper_barrier(obs_idx)
        return lower, upper

    def is_in_range(
        self,
        spot: float,
        obs_idx: int,
        record: Optional[FxRangeAccrualObservationRecord] = None,
    ) -> bool:
        """True when ``spot`` accrues for this fixing (honours reverse mode)."""
        lower, upper = self.barriers_for(obs_idx, record)
        inside = lower <= spot <= upper
        return (not inside) if self.range_config.is_reverse else inside

    def get_year_fraction(
        self, fx_env: Optional["FxPricingEnvironment"] = None
    ) -> float:
        """
        Year fraction the annualized coupon accrues over: the WHOLE contract.

        The in-range ratio counts every fixing, past and future, so the year fraction it multiplies is the
        contract's too: ``tenor`` when set, else ``initial_date`` -> ``expiry_date`` in the config's day count (a
        BUSINESS_DAYS count reads the environment's calendar and year length), else a numeric contract's
        ``maturity`` plus the time ``time_shift`` has aged it by. 1.0 unless the rate is annualized.
        """
        if not self.range_config.is_rate_annualized:
            return 1.0
        if self.tenor is not None:
            return self.tenor
        if self.initial_date is not None and self.expiry_date is not None:
            return calculate_year_fraction(
                self.initial_date,
                self.expiry_date,
                self.range_config.day_count_convention,
                getattr(fx_env, "bus_days_in_year", 252),
                calendar=getattr(fx_env, "calendar", None),
            )
        if self.maturity is not None:
            return self.maturity + self.accrued_offset
        raise ValidationError(
            "An annualized dated FX range accrual needs initial_date or tenor to fix its accrual period"
        )

    def time_shift(self, time_bump: float) -> None:
        """
        Age a numeric contract for theta: maturity and fixing times move with the clock and the elapsed time is
        banked, so the coupon keeps the whole contract's year fraction. Fixings stepped over stay unrecorded until
        ``record_passed_fixings`` fixes them.
        """
        records = [replace(rec) for rec in self.get_observation_records()]
        super().time_shift(time_bump)
        if self.maturity is not None:
            self.accrued_offset += time_bump
            for rec in records:
                if rec.observation_time is not None:
                    rec.observation_time -= time_bump
        self._set_records(records)

    def record_passed_fixings(self, fx_env: "FxPricingEnvironment") -> None:
        """Fix every unrecorded fixing at or before ``fx_env``'s valuation at its spot (see the base class)."""
        records = [replace(rec) for rec in self.get_observation_records()]
        for idx, rec in enumerate(records):
            if rec.observed_in_range is None and rec.resolve_time(fx_env) <= 0:
                rec.observed_in_range = self.is_in_range(fx_env.spot, idx, rec)
        self._set_records(records)

    def missing_fixing_fills(self, fx_env: "FxPricingEnvironment") -> Dict[int, bool]:
        """``{index: outcome}`` for every past fixing without ``observed_in_range``, decided at ``fx_env.spot``
        (the decision ``resolve_observations`` and ``record_passed_fixings`` make, without the warning)."""
        return {
            idx: self.is_in_range(fx_env.spot, idx, rec)
            for idx, rec in enumerate(self.get_observation_records())
            if rec.observed_in_range is None and rec.resolve_time(fx_env) <= 0
        }

    def with_fixings_recorded(self, fills: Mapping[int, bool]) -> "FxRangeAccrualOption":
        """A COPY with ``observed_in_range`` recorded from ``fills`` on every listed fixing still without one.

        A fixing that already carries an outcome keeps it. Returns ``self`` when nothing changes; the schedule is
        written out as explicit records on the copy. The product is never changed.
        """
        records = [replace(rec) for rec in self.get_observation_records()]
        changed = False
        for idx, outcome in fills.items():
            if 0 <= idx < len(records) and records[idx].observed_in_range is None:
                records[idx].observed_in_range = bool(outcome)
                changed = True
        if not changed:
            return self
        frozen = copy.deepcopy(self)
        frozen._set_records(records)
        return frozen

    def with_missing_fixings_filled(
        self, fx_env: "FxPricingEnvironment", *, warn: bool = True
    ) -> "FxRangeAccrualOption":
        """
        This contract with every past fixing lacking ``observed_in_range`` decided at ``fx_env``'s spot.

        The fill ``resolve_observations`` makes, recorded once on a COPY so that bumped and shocked repricings keep
        it at the base spot (user decision 2026-10-09: a filled past fixing is history and does not move). One
        ``MissingFixingWarning`` names the fixings filled (``warn=False`` skips it). Returns ``self`` when nothing
        is missing; the product is never changed.
        """
        fills = self.missing_fixing_fills(fx_env)
        if not fills:
            return self
        if warn:
            records = self.get_observation_records()
            warn_filled_fixings(
                type(self).__name__, "observed_in_range", [fixing_label(idx, records[idx]) for idx in fills]
            )
        return self.with_fixings_recorded(fills)

    def _set_records(self, records: List[FxRangeAccrualObservationRecord]) -> None:
        self.observation_records = records
        self.observation_times = None
        self.num_observations = None

    def get_payoff(
        self,
        spot: float,
        in_range_weights: Optional[float] = None,
        total_weights: Optional[float] = None,
        fx_env: Optional["FxPricingEnvironment"] = None,
    ) -> float:
        """
        Terminal coupon in domestic currency given accumulated fixing weights.

        ``spot`` is accepted for API symmetry with the base product but is not
        used: the coupon depends on the fixing history, not the terminal rate.
        """
        _ = spot
        if in_range_weights is None or total_weights is None:
            raise ValidationError(
                "in_range_weights and total_weights are required for the payoff"
            )
        ratio = 0.0 if is_zero(total_weights) else in_range_weights / total_weights
        return (
            self.notional
            * self.range_config.accrual_rate
            * ratio
            * self.get_year_fraction(fx_env)
        )

    def __repr__(self) -> str:
        cfg = self.range_config
        maturity_str = (
            f"T={self.maturity:.4f}"
            if self.maturity is not None
            else f"exp={self.expiry_date}"
        )
        return (
            f"FxRangeAccrualOption({self.currency_pair}, "
            f"range=[{cfg.lower_barrier}, {cfg.upper_barrier}], "
            f"rate={cfg.accrual_rate:.2%}, notional={self.notional:,.0f}, "
            f"{maturity_str})"
        )
