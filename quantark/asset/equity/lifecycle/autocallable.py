"""
Lifecycle tracking for autocallable products (Snowball, Phoenix).

Extracted from ``quantark/backtest/otc/_replay.py`` (``ProductReplay``); the
event-detection logic is unchanged. Differences from the original:

- Methods return ``LifecycleEvent`` objects instead of appending dict rows to
  sink lists; the caller decides how to record them.
- The market-calendar lookup is an injected ``date_resolver`` callable
  (``ProductReplay`` passes its market-data resolver; the dynamic scenario
  engine uses the default identity resolver because its simulation days are
  consecutive calendar days).

``start_date`` may be ``None`` at construction and assigned later (the OTC
backtest engine sets it at the top of ``run()``); it must be set before any
observation method is called.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.settlement_support import (
    resolve_terminal_timing,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.settlement import SettlementLagUnit
from quantark.priceenv import PricingEnvironment

from .events import LifecycleEvent, LifecycleEventType
from .cashflows import RealizedCashflow, ValuationPoint
from .state import AutocallableLifecycleState

DateResolver = Callable[[pd.Timestamp], pd.Timestamp]


def _identity_date_resolver(date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


class AutocallableLifecycleTracker:
    """
    Tracks realized lifecycle state of one autocallable product and detects
    KO / KI / coupon / maturity events from daily closes.
    """

    def __init__(
        self,
        *,
        product: Any,
        quantity: float,
        lifecycle: Optional[AutocallableLifecycleState] = None,
        start_date: Optional[pd.Timestamp] = None,
        date_resolver: Optional[DateResolver] = None,
        has_lifecycle: bool = True,
    ) -> None:
        self.product = product
        self.quantity = float(quantity)
        self.lifecycle = lifecycle if lifecycle is not None else AutocallableLifecycleState()
        self.start_date = start_date
        self.has_lifecycle = has_lifecycle
        self._date_resolver = date_resolver or _identity_date_resolver

    # ------------------------------------------------------------------
    # Pricing-product construction (mirrors ProductReplay)
    # ------------------------------------------------------------------

    def product_for_lifecycle(self) -> Any:
        """
        Return a copy of the product carrying the current knocked-in state.

        Used for event observation; the copy's ``_otc_lifecycle_knocked_in``
        attribute reflects the tracker's lifecycle state.
        """
        product = deepcopy(self.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        return product

    def product_for_pricing(self, date: pd.Timestamp, pricing_env: PricingEnvironment) -> Any:
        """
        Return a time-decayed copy of the product for pricing on ``date``.

        ``start_date`` must be set before calling this method; if it is
        ``None`` no time decay or barrier-schedule shift is applied (elapsed
        defaults to 0.0). The returned product's ``_otc_lifecycle_knocked_in``
        attribute reflects the current lifecycle state.
        """
        product = deepcopy(self.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        if (
            getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity", None) is not None
            and self.start_date is not None
        ):
            elapsed = max(
                0.0, (pd.Timestamp(date) - pd.Timestamp(self.start_date)).days / 365.0
            )
            product.maturity = max(float(product.maturity) - elapsed, 1e-8)
        elif self.start_date is not None:
            elapsed = max(
                0.0, (pd.Timestamp(date) - pd.Timestamp(self.start_date)).days / 365.0
            )
        else:
            elapsed = 0.0
        barrier_config = getattr(product, "barrier_config", None)
        if barrier_config is not None and hasattr(barrier_config, "time_shift"):
            shifted_config, dropped_all = barrier_config.time_shift(
                elapsed,
                pd.Timestamp(date).to_pydatetime(),
                pricing_env,
            )
            if shifted_config is not None and not dropped_all:
                product.barrier_config = shifted_config
        return product

    # ------------------------------------------------------------------
    # Event detection (mirrors ProductReplay.apply_lifecycle_events)
    # ------------------------------------------------------------------

    def observe(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Detect KO / KI / coupon events at this day's close."""
        timestamp = pd.Timestamp(date).to_pydatetime()
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive:
            return []
        if not self.has_lifecycle:
            return []
        events: List[LifecycleEvent] = []

        ko_records = self._scheduled_records(product, env, "ko")
        ko_disabled_after_ki = bool(
            self.lifecycle.knocked_in
            and getattr(product.barrier_config, "disable_ko_after_ki", False)
        )
        if not ko_disabled_after_ki:
            for idx, rec in enumerate(ko_records):
                if idx in self.lifecycle.observed_ko_indices:
                    continue
                if not self._record_is_due(date, valuation_point, rec):
                    continue
                self.lifecycle.observed_ko_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=True):
                    before = self._state_snapshot()
                    payoff = float(rec.get("payoff", 0.0))
                    cashflow = self.quantity * payoff
                    settlement_date = rec.get("settlement_date")
                    if settlement_date is not None:
                        settlement_date = pd.Timestamp(settlement_date).to_pydatetime()
                    realized = self._record_cashflow(
                        LifecycleEventType.KNOCK_OUT,
                        f"knock-out:{idx}",
                        cashflow,
                        rec,
                        valuation_point,
                    )
                    if self.lifecycle.mark_ko(
                        timestamp,
                        cashflow,
                        settlement_date=settlement_date,
                        realized_cashflow=realized,
                        valuation_point=valuation_point,
                    ):
                        events.append(
                            LifecycleEvent(
                                event_type=LifecycleEventType.KNOCK_OUT,
                                date=date,
                                spot=spot,
                                observation_index=idx,
                                barrier=rec["barrier"],
                                payoff=payoff,
                                cashflow=cashflow,
                                realized_cashflow=realized,
                                terminates_position=True,
                                state_before=before,
                                state_after=self._state_snapshot(),
                                metadata={
                                    "payoff": payoff,
                                    "settlement_date": settlement_date,
                                },
                            )
                        )
                    return events

        ki_observation_type = getattr(product.barrier_config, "ki_observation_type", None)
        ki_continuous = getattr(product, "has_ki_barrier", False) and (
            product.barrier_config.ki_continuous
            or getattr(ki_observation_type, "name", None) == "CONTINUOUS"
        )
        if ki_continuous:
            barrier = product.barrier_config.ki_barrier
            if isinstance(barrier, list):
                barrier = barrier[0]
            if self._barrier_hit(spot, float(barrier), product.is_reverse, is_ko=False):
                before = self._state_snapshot()
                if self.lifecycle.mark_ki(pd.Timestamp(date).to_pydatetime()):
                    events.append(
                        LifecycleEvent(
                            event_type=LifecycleEventType.KNOCK_IN,
                            date=date,
                            spot=spot,
                            observation_index=None,
                            barrier=float(barrier),
                            payoff=0.0,
                            cashflow=0.0,
                            terminates_position=False,
                            state_before=before,
                            state_after=self._state_snapshot(),
                            metadata={"monitoring": "daily_close"},
                        )
                    )
        else:
            ki_records = self._scheduled_records(product, env, "ki")
            for idx, rec in enumerate(ki_records):
                if idx in self.lifecycle.observed_ki_indices:
                    continue
                if not self._record_is_due(date, valuation_point, rec):
                    continue
                self.lifecycle.observed_ki_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=False):
                    before = self._state_snapshot()
                    if self.lifecycle.mark_ki(pd.Timestamp(date).to_pydatetime()):
                        events.append(
                            LifecycleEvent(
                                event_type=LifecycleEventType.KNOCK_IN,
                                date=date,
                                spot=spot,
                                observation_index=idx,
                                barrier=rec["barrier"],
                                payoff=0.0,
                                cashflow=0.0,
                                terminates_position=False,
                                state_before=before,
                                state_after=self._state_snapshot(),
                            )
                        )

        if isinstance(product, PhoenixOption):
            for idx, rec in enumerate(ko_records):
                if idx in self.lifecycle.observed_coupon_indices:
                    continue
                if not self._record_is_due(date, valuation_point, rec):
                    continue
                self.lifecycle.observed_coupon_indices.add(idx)
                if product.is_coupon_triggered(spot, idx):
                    before = self._state_snapshot()
                    payoff = float(product.get_coupon_payoff(idx))
                    coupon = self.quantity * payoff
                    realized = self._record_cashflow(
                        LifecycleEventType.COUPON,
                        f"coupon:{idx}",
                        coupon,
                        rec,
                        valuation_point,
                    )
                    self.lifecycle.add_cashflow(
                        coupon,
                        realized_cashflow=realized,
                        valuation_point=valuation_point,
                    )
                    self.lifecycle.coupon_memory_count = 0
                    events.append(
                        LifecycleEvent(
                            event_type=LifecycleEventType.COUPON,
                            date=date,
                            spot=spot,
                            observation_index=idx,
                            barrier=product.get_coupon_barrier_at(idx),
                            payoff=payoff,
                            cashflow=coupon,
                            realized_cashflow=realized,
                            terminates_position=False,
                            state_before=before,
                            state_after=self._state_snapshot(),
                        )
                    )
                elif product.has_memory_coupon:
                    self.lifecycle.coupon_memory_count += 1

        return events

    def settle_maturity_if_due(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> Optional[LifecycleEvent]:
        """Determine the terminal payoff once contractual maturity is reached."""
        timestamp = pd.Timestamp(date).to_pydatetime()
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive:
            return None
        schedule_env = self._schedule_resolution_env(product, env)
        timing = resolve_terminal_timing(product, schedule_env)
        if not self._timing_is_due(date, valuation_point, timing):
            return None

        before = self._state_snapshot()
        payoff = float(
            product.get_payoff(spot, env, knocked_in=self.lifecycle.knocked_in)
        )
        cashflow = self.quantity * payoff
        realized = self._timing_cashflow(
            LifecycleEventType.MATURITY,
            "maturity",
            cashflow,
            timing,
            valuation_point,
        )
        if self.lifecycle.mark_maturity(
            timestamp,
            cashflow,
            realized_cashflow=realized,
            valuation_point=valuation_point,
        ):
            return LifecycleEvent(
                event_type=LifecycleEventType.MATURITY,
                date=date,
                spot=spot,
                observation_index=None,
                barrier=None,
                payoff=payoff,
                cashflow=cashflow,
                realized_cashflow=realized,
                terminates_position=True,
                state_before=before,
                state_after=self._state_snapshot(),
                metadata={"payoff": payoff},
            )
        return None

    # ------------------------------------------------------------------
    # Schedule resolution helpers (mirrors ProductReplay)
    # ------------------------------------------------------------------

    def _state_snapshot(self) -> Dict[str, bool]:
        return {
            "alive": self.lifecycle.alive,
            "knocked_in": self.lifecycle.knocked_in,
            "knocked_out": self.lifecycle.knocked_out,
            "matured": self.lifecycle.matured,
        }

    def _schedule_resolution_env(
        self, product: Any, env: PricingEnvironment
    ) -> PricingEnvironment:
        """Resolve lifecycle schedules from the contract issue date."""
        base_date = getattr(product, "initial_date", None) or self.start_date
        if base_date is None:
            return env

        schedule_env = deepcopy(env)
        schedule_env.valuation_date = pd.Timestamp(base_date).to_pydatetime()
        return schedule_env

    def _scheduled_records(
        self, product: Any, env: PricingEnvironment, kind: str
    ) -> List[Dict[str, Any]]:
        schedule_env = self._schedule_resolution_env(product, env)
        if kind == "ko":
            resolved = product.resolve_ko_observations(schedule_env)
        else:
            if not getattr(product, "has_ki_barrier", False):
                return []
            resolved = product.resolve_ki_observations(schedule_env)

        records: List[Dict[str, Any]] = []
        # Replay clock: a numeric schedule still plays out on real calendar
        # days of this run, so dates absent from the schedule are derived from
        # the run's base date. The settlement RESOLVER never fabricates dates;
        # this is the replay layer's own calendar, and it is what date-based
        # due checks, pending settlement, and run termination key on.
        base = getattr(product, "initial_date", None) or self.start_date
        base_date = pd.Timestamp(base) if base is not None else None
        for rec in resolved:
            obs_time = float(rec.observation_time)
            settlement_time = float(rec.settlement_time)
            observation_date = getattr(rec, "observation_date", None)
            determination_date = observation_date
            if observation_date is not None:
                obs_date = self._date_resolver(
                    pd.Timestamp(observation_date).normalize()
                )
            elif base_date is not None:
                obs_date = self._date_resolver(
                    (
                        base_date + timedelta(days=int(round(obs_time * 365)))
                    ).normalize()
                )
                determination_date = obs_date
            else:
                obs_date = None
            settlement_date = getattr(rec, "settlement_date", None)
            if settlement_date is None and obs_date is not None:
                if settlement_time <= obs_time:
                    settlement_date = obs_date
                elif base_date is not None:
                    settlement_date = self._date_resolver(
                        (
                            base_date
                            + timedelta(days=int(round(settlement_time * 365)))
                        ).normalize()
                    )
            records.append(
                {
                    "date": obs_date,
                    "determination_date": determination_date,
                    "time": obs_time,
                    "settlement_date": settlement_date,
                    "settlement_time": settlement_time,
                    "barrier": (
                        float(rec.barrier)
                        if rec.barrier is not None
                        else None
                    ),
                    "payoff": float(rec.payoff),
                }
            )
        return records

    def _valuation_point(
        self, date: pd.Timestamp, product: Any
    ) -> ValuationPoint:
        if self._uses_date_timing(product):
            return ValuationPoint(date=pd.Timestamp(date).to_pydatetime())
        if self.start_date is None:
            raise ValueError(
                "start_date is required for numeric lifecycle schedules"
            )
        elapsed_days = (
            pd.Timestamp(date).normalize()
            - pd.Timestamp(self.start_date).normalize()
        ).days
        return ValuationPoint(time=max(0.0, elapsed_days / 365.0))

    @staticmethod
    def _uses_date_timing(product: Any) -> bool:
        convention = getattr(product, "settlement_convention", None)
        if (
            convention is not None
            and convention.lag_unit is SettlementLagUnit.YEAR_FRACTION
        ):
            return False
        if (
            getattr(product, "exercise_date", None) is not None
            or getattr(product, "maturity_date", None) is not None
        ):
            return True
        barrier_config = getattr(product, "barrier_config", None)
        has_date_schedule = False
        for name in ("ko_observation_schedule", "ki_observation_schedule"):
            schedule = getattr(barrier_config, name, None)
            if schedule is None or not schedule.uses_dates():
                continue
            has_date_schedule = True
            if any(
                record.settlement_time is not None
                and record.settlement_date is None
                for record in schedule.records
            ):
                return False
        return has_date_schedule

    @staticmethod
    def _record_is_due(
        date: pd.Timestamp,
        valuation_point: ValuationPoint,
        record: Dict[str, Any],
    ) -> bool:
        if record["date"] is not None:
            return pd.Timestamp(date).normalize() >= record["date"]
        elapsed_days = int(round(float(valuation_point.time) * 365.0))
        due_days = int(round(float(record["time"]) * 365.0))
        return elapsed_days >= due_days

    @staticmethod
    def _timing_is_due(date, valuation_point, timing) -> bool:
        if timing.determination_date is not None:
            return (
                pd.Timestamp(date).normalize()
                >= pd.Timestamp(timing.determination_date).normalize()
            )
        elapsed_days = int(round(float(valuation_point.time) * 365.0))
        due_days = int(round(float(timing.determination_time) * 365.0))
        return elapsed_days >= due_days

    @staticmethod
    def _record_cashflow(
        event_type: LifecycleEventType,
        cashflow_id: str,
        amount: float,
        record: Dict[str, Any],
        valuation_point: ValuationPoint,
    ) -> RealizedCashflow:
        # The entry's representation must match the product's timing (the
        # valuation point already encodes that choice): a numeric lifecycle
        # keeps time-based flows even when the record carries replay-clock
        # dates, so ledger queries never mix dates with times.
        if (
            valuation_point.date is not None
            and record["date"] is not None
            and record["settlement_date"] is not None
        ):
            determination_date = pd.Timestamp(
                record["determination_date"]
            ).to_pydatetime()
            payment_date = pd.Timestamp(
                record["settlement_date"]
            ).to_pydatetime()
            return RealizedCashflow(
                cashflow_id=cashflow_id,
                event_type=event_type,
                amount=amount,
                determination_date=determination_date,
                payment_date=payment_date,
            )

        delay = float(record["settlement_time"]) - float(record["time"])
        determination_time = float(valuation_point.time)
        return RealizedCashflow(
            cashflow_id=cashflow_id,
            event_type=event_type,
            amount=amount,
            determination_time=determination_time,
            payment_time=determination_time + delay,
        )

    @staticmethod
    def _timing_cashflow(
        event_type: LifecycleEventType,
        cashflow_id: str,
        amount: float,
        timing,
        valuation_point: ValuationPoint,
    ) -> RealizedCashflow:
        if (
            timing.determination_date is not None
            and timing.payment_date is not None
        ):
            return RealizedCashflow(
                cashflow_id=cashflow_id,
                event_type=event_type,
                amount=amount,
                determination_date=timing.determination_date,
                payment_date=timing.payment_date,
            )

        delay = float(timing.payment_time) - float(timing.determination_time)
        determination_time = float(valuation_point.time)
        return RealizedCashflow(
            cashflow_id=cashflow_id,
            event_type=event_type,
            amount=amount,
            determination_time=determination_time,
            payment_time=determination_time + delay,
        )

    @staticmethod
    def _barrier_hit(
        spot: float, barrier: Optional[float], is_reverse: bool, is_ko: bool
    ) -> bool:
        if barrier is None:
            return False
        if is_ko:
            return spot <= barrier if is_reverse else spot >= barrier
        return spot >= barrier if is_reverse else spot <= barrier
