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

from datetime import datetime
from typing import List, Optional, Tuple, TYPE_CHECKING

from quantark.util.exceptions import ValidationError
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
    ):
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
        self._validate_schedule()
        self._validate_barrier_lengths()

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
        self, fx_env: "FxPricingEnvironment"
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
        """
        records = self.get_observation_records()
        past: List[Tuple[float, bool]] = []
        future: List[Tuple[float, float, int]] = []
        total_weights = 0.0

        for idx, rec in enumerate(records):
            t = rec.resolve_time(fx_env)
            total_weights += rec.weight
            if t <= 0:
                if rec.observed_in_range is None:
                    raise ValidationError(
                        f"Past fixing at index {idx} (t={t}) must set "
                        "observed_in_range"
                    )
                past.append((rec.weight, rec.observed_in_range))
            else:
                if rec.observed_in_range is not None:
                    raise ValidationError(
                        f"Future fixing at index {idx} (t={t}) must not set "
                        "observed_in_range"
                    )
                future.append((rec.weight, t, idx))

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
        """Year fraction applied to the coupon (1.0 unless rate is annualized)."""
        if not self.range_config.is_rate_annualized:
            return 1.0
        if self.maturity is not None:
            return self.maturity
        if self.expiry_date is not None and fx_env is not None:
            return self.get_maturity(fx_env)
        return 1.0

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
