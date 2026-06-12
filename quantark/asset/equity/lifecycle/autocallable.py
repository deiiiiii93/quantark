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

from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.priceenv import PricingEnvironment

from .events import LifecycleEvent, LifecycleEventType
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
                if date < rec["date"]:
                    continue
                self.lifecycle.observed_ko_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=True):
                    before = self._state_snapshot()
                    payoff = float(rec.get("payoff", 0.0))
                    cashflow = self.quantity * payoff
                    if self.lifecycle.mark_ko(pd.Timestamp(date).to_pydatetime(), cashflow):
                        events.append(
                            LifecycleEvent(
                                event_type=LifecycleEventType.KNOCK_OUT,
                                date=date,
                                spot=spot,
                                observation_index=idx,
                                barrier=rec["barrier"],
                                payoff=payoff,
                                cashflow=cashflow,
                                terminates_position=True,
                                state_before=before,
                                state_after=self._state_snapshot(),
                                metadata={"payoff": payoff},
                            )
                        )
                    return events

        ki_records = self._scheduled_records(product, env, "ki")
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
            for idx, rec in enumerate(ki_records):
                if idx in self.lifecycle.observed_ki_indices:
                    continue
                if date < rec["date"]:
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
                if date < rec["date"]:
                    continue
                self.lifecycle.observed_coupon_indices.add(idx)
                if product.is_coupon_triggered(spot, idx):
                    before = self._state_snapshot()
                    payoff = float(product.get_coupon_payoff(idx))
                    coupon = self.quantity * payoff
                    self.lifecycle.add_cashflow(coupon)
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
        """Settle the product at maturity if the settlement date has been reached."""
        if not self.lifecycle.alive:
            return None
        if pd.Timestamp(date) < self._maturity_settlement_date(product, env):
            return None

        before = self._state_snapshot()
        payoff = float(
            product.get_payoff(spot, env, knocked_in=self.lifecycle.knocked_in)
        )
        cashflow = self.quantity * payoff
        if self.lifecycle.mark_maturity(pd.Timestamp(date).to_pydatetime(), cashflow):
            return LifecycleEvent(
                event_type=LifecycleEventType.MATURITY,
                date=date,
                spot=spot,
                observation_index=None,
                barrier=None,
                payoff=payoff,
                cashflow=cashflow,
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
            profile = product.get_ko_observation_profile(schedule_env)
            schedule = getattr(product.barrier_config, "ko_observation_schedule", None)
        else:
            if not getattr(product, "has_ki_barrier", False):
                return []
            profile = product.get_ki_observation_profile(schedule_env)
            schedule = getattr(product.barrier_config, "ki_observation_schedule", None)

        times = list(profile.get("observation_times", []))
        barriers = list(profile.get("barriers", []))
        payoffs = list(profile.get("payoffs", [0.0] * len(times)))
        schedule_dates = []
        if schedule is not None:
            for rec in schedule.records:
                schedule_dates.append(getattr(rec, "observation_date", None))

        records = []
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self.start_date)
        for idx, obs_time in enumerate(times):
            if idx < len(schedule_dates) and schedule_dates[idx] is not None:
                obs_date = pd.Timestamp(schedule_dates[idx]).normalize()
            else:
                obs_date = (
                    base_date + timedelta(days=int(round(float(obs_time) * 365)))
                ).normalize()
            obs_date = self._date_resolver(obs_date)
            records.append(
                {
                    "date": obs_date,
                    "time": float(obs_time),
                    "barrier": float(barriers[idx]) if idx < len(barriers) and barriers[idx] is not None else None,
                    "payoff": float(payoffs[idx]) if idx < len(payoffs) and payoffs[idx] is not None else 0.0,
                }
            )
        return records

    def _maturity_settlement_date(
        self, product: Any, env: PricingEnvironment
    ) -> pd.Timestamp:
        explicit = (
            getattr(product, "maturity_date", None)
            or getattr(product, "exercise_date", None)
        )
        if explicit is not None:
            return self._date_resolver(pd.Timestamp(explicit).normalize())

        maturity = float(product.get_maturity(env))
        base_date = pd.Timestamp(getattr(product, "initial_date", None) or self.start_date)
        maturity_date = (
            base_date + timedelta(days=int(round(maturity * 365.0)))
        ).normalize()
        return self._date_resolver(maturity_date)

    @staticmethod
    def _barrier_hit(
        spot: float, barrier: Optional[float], is_reverse: bool, is_ko: bool
    ) -> bool:
        if barrier is None:
            return False
        if is_ko:
            return spot <= barrier if is_reverse else spot >= barrier
        return spot >= barrier if is_reverse else spot <= barrier
