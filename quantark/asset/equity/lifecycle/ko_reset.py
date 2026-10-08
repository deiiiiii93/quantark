"""Lifecycle tracking for the knock-out-reset snowball (spec 2026-10-08 KO-reset lifecycle).

The contract's rules, as the desk stated them and every engine applies them:

* R1 a knock-in during schedule 1 replaces it by schedule 2 for good, schedule 2 in force from the knock-in
  observation, that observation included;
* R2 a contract not knocked in when schedule 1 ends has matured there: it pays V0 on schedule 1's contract
  length, and a knock-in is never tested after that;
* R3 a schedule-1 knock-out and a knock-in at the same observation: the knock-out wins;
* R4 a carried knock-in happened.

A knocked-in REBASED contract is the ABSOLUTE contract ``with_absolute_schedule_2`` builds at the knock-in
time, so every engine prices an ordinary knocked-in contract and the tracker books from that same contract.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing_at
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .autocallable import AutocallableLifecycleTracker, DateResolver
from .events import LifecycleEvent, LifecycleEventType
from .state import AutocallableLifecycleState

DAYS_PER_YEAR = 365.0

__all__ = ["KOResetLifecycleTracker", "tracker_for"]


class KOResetLifecycleTracker(AutocallableLifecycleTracker):
    """Tracks one ``KnockOutResetSnowballOption`` under rules R1-R4 (see the module docstring).

    The knock-in close is the state's ``ki_date`` (set by ``mark_ki``); on the numeric clock the knock-in time
    is ``(ki_date - start_date).days / 365``. Without a ``lifecycle`` the starting state is the product's
    carried one: ``_otc_lifecycle_knocked_in`` and ``_otc_lifecycle_knock_in_date``.
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
        if lifecycle is None:
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            lifecycle = AutocallableLifecycleState(knocked_in=knocked_in)
            if knocked_in:
                lifecycle.ki_date = getattr(product, "_otc_lifecycle_knock_in_date", None)
        if (
            lifecycle.knocked_in
            and lifecycle.ki_date is None
            and product.post_ko_mode == PostKOScheduleMode.REBASED
        ):
            raise ValidationError(
                "KO-reset snowball: a carried knock-in on a REBASED contract needs its knock-in date "
                "(_otc_lifecycle_knock_in_date): schedule 2 holds offsets from it and cannot be placed without it"
            )
        super().__init__(
            product=product,
            quantity=quantity,
            lifecycle=lifecycle,
            start_date=start_date,
            date_resolver=date_resolver,
            has_lifecycle=has_lifecycle,
        )
        self._previous_close: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # The contract in force
    # ------------------------------------------------------------------

    def knock_in_time(self) -> Optional[float]:
        """Years from ``start_date`` to the knock-in close (numeric clock); ``None`` while not knocked in."""
        if not self.lifecycle.knocked_in or self.lifecycle.ki_date is None or self.start_date is None:
            return None
        days = (
            pd.Timestamp(self.lifecycle.ki_date).normalize() - pd.Timestamp(self.start_date).normalize()
        ).days
        return days / DAYS_PER_YEAR

    def contract(self) -> Any:
        """The snapped contract, or the ABSOLUTE contract a knocked-in REBASED one has become."""
        if self.lifecycle.knocked_in and self.product.post_ko_mode == PostKOScheduleMode.REBASED:
            knock_in_time = self.knock_in_time()
            if knock_in_time is None:
                raise ValidationError("a knocked-in REBASED KO-reset needs start_date and its knock-in date")
            return self.product.with_absolute_schedule_2(knock_in_time)
        return self.product

    def product_for_lifecycle(self) -> Any:
        product = super().product_for_lifecycle()
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        return product

    def _contract_for_pricing(self) -> Any:
        product = deepcopy(self.contract())
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        return product

    def _age_extra(self, product, elapsed, date, pricing_env, keep_landing) -> None:
        """Age schedule 2 with schedule 1; a REBASED one (offsets) stays until a knock-in converts it."""
        if product.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            product.post_barrier_config, _ = product.post_barrier_config.time_shift(
                elapsed, date, pricing_env, keep_landing=keep_landing
            )

    # ------------------------------------------------------------------
    # Records, tenors and terminal timings
    # ------------------------------------------------------------------

    def _schedule_records(self, product: Any, env: PricingEnvironment, config: Any) -> List[Dict[str, Any]]:
        """One schedule's knock-out records; the cash is the product's own ``_ko_cash`` (both schedules)."""
        schedule_env = self._schedule_resolution_env(product, env)
        resolved, rates, sources = product._resolve_ko_schedule(config, schedule_env)
        payoffs = [
            product._ko_cash(record.observation_time, rate, source, schedule_env)
            for record, rate, source in zip(resolved, rates, sources)
        ]
        return self._record_dicts(product, resolved, payoffs)

    def pre_contract_tenor(self, env: PricingEnvironment) -> float:
        """Schedule 1's contract length (V0's tenor)."""
        return float(self.product._resolve_pre_contract_tenor(self._schedule_resolution_env(self.product, env)))

    def final_contract_tenor(self, env: PricingEnvironment) -> float:
        """The knocked-in contract's length (V1's tenor)."""
        contract = self.contract()
        return float(contract._resolve_post_contract_tenor(self._schedule_resolution_env(contract, env)))

    def _last_pre_record(self, product: Any, env: PricingEnvironment) -> Dict[str, Any]:
        records = self._schedule_records(product, env, product.barrier_config)
        if not records:
            raise ValidationError(
                "KO-reset snowball: schedule 1 has no observation left, so a contract not knocked in has matured "
                "before this run; it cannot be tracked from here"
            )
        return records[-1]

    def _pre_timing(self, product: Any, env: PricingEnvironment, last: Dict[str, Any]):
        """V0's terminal timing: determined at schedule 1's last observation."""
        schedule_env = self._schedule_resolution_env(product, env)
        if self._uses_date_timing(product) and last["determination_date"] is not None:
            return resolve_terminal_timing_at(
                product, schedule_env, determination_date=pd.Timestamp(last["determination_date"]).to_pydatetime(),
                cashflow_id="ko_reset_v0_terminal",
            )
        return resolve_terminal_timing_at(
            product, schedule_env, determination_time=float(last["time"]), cashflow_id="ko_reset_v0_terminal"
        )

    def _final_timing(self, env: PricingEnvironment):
        """V1's terminal timing: determined at the knocked-in contract's end."""
        contract = self.contract()
        schedule_env = self._schedule_resolution_env(contract, env)
        end_date = contract._resolve_schedule_end_date(contract.post_barrier_config.ko_observation_schedule)
        if end_date is not None and self._uses_date_timing(contract):
            return resolve_terminal_timing_at(
                contract, schedule_env, determination_date=end_date, cashflow_id="ko_reset_v1_terminal"
            )
        return resolve_terminal_timing_at(
            contract, schedule_env, determination_time=float(contract.get_post_maturity_time(schedule_env)),
            cashflow_id="ko_reset_v1_terminal",
        )

    def _due_at_previous_close(self, product: Any, record: Dict[str, Any]) -> bool:
        """Whether ``record`` was already due at the previous observed close (none: nothing was)."""
        if self._previous_close is None:
            return False
        previous = self._previous_close
        return self._record_is_due(previous, self._valuation_point(previous, product), record)

    # ------------------------------------------------------------------
    # The day's close
    # ------------------------------------------------------------------

    def observe(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Schedule-1 knock-out, then the knock-in, then schedule 2 (the knock-in close included)."""
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive or not self.has_lifecycle:
            return []
        events: List[LifecycleEvent] = []
        try:
            if not self.lifecycle.knocked_in:
                knock_out = self._test_knock_outs(date, product, env, spot, valuation_point, schedule=1)
                if knock_out is not None:
                    return [knock_out]                      # R3: a knock-out ends the day
                knock_in = self._test_knock_in(date, product, env, spot, valuation_point)
                if knock_in is None:
                    return events
                events.append(knock_in)
                self._consume_schedule_two(env)
            contract = self.contract()
            if not contract.barrier_config.disable_ko_after_ki:
                knock_out = self._test_knock_outs(date, contract, env, spot, valuation_point, schedule=2)
                if knock_out is not None:
                    events.append(knock_out)
            return events
        finally:
            self._previous_close = pd.Timestamp(date).normalize()

    def _test_knock_outs(self, date, product, env, spot, valuation_point, *, schedule: int):
        config = product.barrier_config if schedule == 1 else product.post_barrier_config
        observed = (
            self.lifecycle.observed_ko_indices if schedule == 1 else self.lifecycle.observed_post_ko_indices
        )
        for idx, rec in enumerate(self._schedule_records(product, env, config)):
            if idx in observed or not self._record_is_due(date, valuation_point, rec):
                continue
            observed.add(idx)
            if not self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=True):
                continue
            before = self._state_snapshot()
            payoff = float(rec["payoff"])
            cashflow = self.quantity * payoff
            settlement_date = rec.get("settlement_date")
            if settlement_date is not None:
                settlement_date = pd.Timestamp(settlement_date).to_pydatetime()
            realized = self._record_cashflow(
                LifecycleEventType.KNOCK_OUT, f"knock-out:ko{schedule}:{idx}", cashflow, rec, valuation_point,
                metadata={"schedule": schedule},
            )
            if self.lifecycle.mark_ko(
                pd.Timestamp(date).to_pydatetime(),
                cashflow,
                settlement_date=settlement_date,
                realized_cashflow=realized,
                valuation_point=valuation_point,
            ):
                return LifecycleEvent(
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
                    metadata={"payoff": payoff, "settlement_date": settlement_date, "schedule": schedule},
                )
        return None

    def _test_knock_in(self, date, product, env, spot, valuation_point) -> Optional[LifecycleEvent]:
        """The knock-in, only while schedule 1 is live (R2): never once its last observation has passed."""
        if self._due_at_previous_close(product, self._last_pre_record(product, env)):
            return None
        config = product.barrier_config
        ki_observation_type = getattr(config, "ki_observation_type", None)
        continuous = getattr(product, "has_ki_barrier", False) and (
            config.ki_continuous or getattr(ki_observation_type, "name", None) == "CONTINUOUS"
        )
        if continuous:
            barrier = config.ki_barrier[0] if isinstance(config.ki_barrier, list) else config.ki_barrier
            hits = [(None, float(barrier))] if self._barrier_hit(
                spot, float(barrier), product.is_reverse, is_ko=False) else []
            metadata = {"monitoring": "daily_close", "schedule": 1}
        else:
            hits = []
            for idx, rec in enumerate(self._scheduled_records(product, env, "ki")):
                if idx in self.lifecycle.observed_ki_indices or not self._record_is_due(date, valuation_point, rec):
                    continue
                self.lifecycle.observed_ki_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=False):
                    hits.append((idx, rec["barrier"]))
            metadata = {"schedule": 1}
        if not hits:
            return None
        index, barrier = hits[0]
        before = self._state_snapshot()
        if not self.lifecycle.mark_ki(pd.Timestamp(date).normalize().to_pydatetime()):
            return None
        return LifecycleEvent(
            event_type=LifecycleEventType.KNOCK_IN,
            date=date,
            spot=spot,
            observation_index=index,
            barrier=barrier,
            payoff=0.0,
            cashflow=0.0,
            terminates_position=False,
            state_before=before,
            state_after=self._state_snapshot(),
            metadata=metadata,
        )

    def _consume_schedule_two(self, env: PricingEnvironment) -> None:
        """At the knock-in close: schedule-2 records that fell due on an EARLIER close were due while schedule 2
        was not in force, so the knock-in consumes them (observed, never tested)."""
        contract = self.contract()
        for idx, rec in enumerate(self._schedule_records(contract, env, contract.post_barrier_config)):
            if self._due_at_previous_close(contract, rec):
                self.lifecycle.observed_post_ko_indices.add(idx)

    def settle_maturity_if_due(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> Optional[LifecycleEvent]:
        """V0 at schedule 1's last observation (R2), or V1 at the knocked-in contract's end."""
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive:
            return None
        if self.lifecycle.knocked_in:
            timing = self._final_timing(env)
            if not self._timing_is_due(date, valuation_point, timing):
                return None
            contract = self.contract()
            payoff = float(contract.get_maturity_payoff_v1(
                spot, self._schedule_resolution_env(contract, env), contract_tenor=self.final_contract_tenor(env)
            ))
        else:
            last = self._last_pre_record(product, env)
            if not self._record_is_due(date, valuation_point, last):
                return None
            timing = self._pre_timing(product, env, last)
            payoff = float(product.get_maturity_payoff_v0(
                spot, self._schedule_resolution_env(product, env), contract_tenor=self.pre_contract_tenor(env)
            ))
        return self._book_maturity(date, spot, payoff, timing, valuation_point)


def tracker_for(product: Any, **kwargs) -> AutocallableLifecycleTracker:
    """The tracker a product's rules need: KO-reset or the snowball/Phoenix one."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    cls = KOResetLifecycleTracker if isinstance(product, KnockOutResetSnowballOption) else AutocallableLifecycleTracker
    return cls(product=product, **kwargs)
