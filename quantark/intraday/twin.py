"""Float-time numerical twin of a contract for a given valuation instant.

Engines never see dates in intraday mode. The twin carries seconds-exact
ACT/365 calendar fractions from the valuation timestamp, the contract's own
cash amounts (through positional accrual factors), and a TIME-based lifecycle
ledger. It is a deep copy: the caller's product is untouched. Every twin
re-resolves its own cash and must reproduce the contractual table, else it
fails closed — a twin that pays different amounts is a different contract.
"""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, LifecycleCashflowLedger, RealizedCashflow, ValuationPoint
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.execution.errors import CapabilityError
from quantark.intraday.events import ContractEvent, ContractTimeline, EventKind, EventPhase, phoenix_coupon_fractions
from quantark.intraday.provisional import LifecycleReconstruction
from quantark.intraday.timestamp import calendar_year_fraction
from quantark.param import FlatRateCurve, FlatVolSurface, NoDividend, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

_CASH_REL_TOL = 1e-12
_CASH_ABS_TOL = 1e-12
_TRACKER_IDS = {"knock-out": EventKind.KO, "coupon": EventKind.COUPON}


@dataclass(frozen=True)
class NumericalContract:
    product: object                       # float-time deep copy; engines see ONLY this (None once terminated)
    lifecycle_state: Optional[object]     # time-based AutocallableLifecycleState (ValuationPoint(time=0.0)); None without lifecycle
    remaining_events: Tuple[ContractEvent, ...]
    event_taus: Mapping[str, float]       # event_id -> tau (years from valuation); tau == 0.0 only under BEFORE
    maturity_tau: float
    knocked_in: bool
    terminated: bool                      # state is not alive (KO'd / matured): no contingent claim remains
    pending_cashflows: Tuple[Tuple[str, float, float], ...]   # (cashflow_id, amount, payment_tau) with payment_tau > 0
    paid_cash: float                      # sum of ledger amounts with payment_tau <= 0


def _settlement(lag: float) -> Optional[SettlementConvention]:
    if lag < 0.0:
        raise ValidationError("terminal payment before determination")
    return SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION) if lag > 0.0 else None


def _clear_dates(twin) -> None:
    for name in ("initial_date", "exercise_date", "settlement_date", "maturity_date"):
        if hasattr(twin, name):
            setattr(twin, name, None)


def _pick(value, indices):
    """A per-observation list restricted to ``indices``; scalars pass through."""
    return [value[i] for i in indices] if isinstance(value, (list, tuple)) else value


def _cash_env(ts: datetime, product) -> PricingEnvironment:
    """Market-free environment used only to re-resolve the twin's own cash table."""
    return PricingEnvironment(rate_curve=FlatRateCurve(0.0), valuation_date=ts,
                              spot_quote=SpotQuote(float(getattr(product, "initial_price", 1.0)) or 1.0),
                              vol_surface=FlatVolSurface(0.2), div_yield=NoDividend())


def _autocallable_twin(product, timeline: ContractTimeline, remaining, ts: datetime, state):
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    phoenix = type(product) is PhoenixOption
    twin = deepcopy(product)
    kos = [e for e in remaining if e.kind is EventKind.KO]
    kis = [e for e in remaining if e.kind is EventKind.KI]
    coupons = [e for e in remaining if e.kind is EventKind.COUPON]
    if not kos:
        raise ValidationError("a live autocallable needs at least one remaining KO observation")
    principal = twin.initial_price * twin.contract_multiplier if twin.payoff_config.include_principal else 0.0
    unit = twin.initial_price * twin.contract_multiplier
    src = list(product.barrier_config.ko_observation_schedule.records)
    records, ko_factors, rates = [], [], []
    for e in kos:
        rate = src[e.index].return_rate
        rate = float(product.get_ko_rate_at(e.index)) if rate is None else float(rate)
        ko_factors.append((e.cash - principal) / (unit * rate) if rate != 0.0 else None)
        rates.append(rate)
        records.append(ObservationRecord(observation_time=calendar_year_fraction(ts, e.timestamp), barrier=e.barrier,
                                         return_rate=rate, settlement_time=calendar_year_fraction(ts, e.payment_timestamp)))
    if phoenix:
        if [c.index for c in coupons] != [k.index for k in kos]:
            raise ValidationError("Phoenix coupon events must align with the remaining KO events")
        coupon_rate = twin.coupon_config.coupon_rate
        coupon_factors = [c.cash / (unit * coupon_rate) if coupon_rate != 0.0 else 0.0 for c in coupons]
        # One positional list serves both the KO accrual (x ko_rate) and the coupon fractions.
        for k, (kf, cf) in enumerate(zip(ko_factors, coupon_factors)):
            if kf is not None and not is_close(kf, cf, rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
                raise CapabilityError(
                    f"Phoenix KO accrual ({kf!r}) and coupon fraction ({cf!r}) differ at KO[{kos[k].index}]: the "
                    "float-time twin carries a single accrual_factors list; not in the intraday inventory")
        factors = coupon_factors
        memory = int(getattr(state, "coupon_memory_count", 0) or 0) if state is not None else 0
        if memory and twin.coupon_config.fixed_coupon_year_fraction is None:
            raise CapabilityError("aged Phoenix coupon memory needs fixed_coupon_year_fraction on the float-time twin")
    else:
        factors = [0.0 if f is None else f for f in ko_factors]
    indices = [e.index for e in kos]
    bc = twin.barrier_config
    ko_schedule = ObservationSchedule(records=records, aggregation_mode=bc.ko_observation_schedule.aggregation_mode,
                                      frequency=bc.ko_observation_schedule.frequency)
    kwargs = dict(ko_barrier=_pick(bc.ko_barrier, indices), ko_rate=_pick(bc.ko_rate, indices),
                  ko_observation_dates=None, ko_observation_schedule=ko_schedule)
    if product.has_ki_barrier and timeline.continuous_ki_barrier is None:
        ki_records = [ObservationRecord(observation_time=calendar_year_fraction(ts, e.timestamp), barrier=e.barrier) for e in kis]
        ki_indices = [e.index for e in kis]
        if ki_records:
            kwargs.update(ki_barrier=_pick(bc.ki_barrier, ki_indices), ki_observation_dates=None,
                          ki_observation_schedule=ObservationSchedule(records=ki_records))
        else:
            kwargs.update(ki_observation_dates=None, ki_observation_schedule=ObservationSchedule(records=[]))
    twin.barrier_config = replace(bc, **kwargs)
    twin.accrual_config = replace(twin.accrual_config, accrual_factors=[float(f) for f in factors],
                                  accrued_offset=0.0, accrual_dates=None)
    if phoenix:
        twin.coupon_config = replace(twin.coupon_config, coupon_barrier=_pick(twin.coupon_config.coupon_barrier, indices))
    terminal = timeline.terminal()
    twin.maturity = calendar_year_fraction(ts, terminal.timestamp)
    twin.tenor = timeline.contractual_tenor
    _clear_dates(twin)
    twin.settlement_convention = _settlement(calendar_year_fraction(ts, terminal.payment_timestamp) - twin.maturity)
    setattr(twin, "_otc_lifecycle_knocked_in", bool(state.knocked_in) if state is not None else False)
    _verify_autocallable_cash(twin, kos, coupons, ts, phoenix)
    return twin


def _verify_autocallable_cash(twin, kos, coupons, ts, phoenix) -> None:
    resolved = twin.resolve_ko_observations(_cash_env(ts, twin))
    if len(resolved) != len(kos):
        raise ValidationError(f"twin resolves {len(resolved)} KO observations for {len(kos)} remaining events")
    for r, e in zip(resolved, kos):
        if not is_close(float(r.payoff), e.cash, rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
            raise ValidationError(f"twin KO[{e.index}] pays {r.payoff!r}, contract pays {e.cash!r}")
    if phoenix:
        fractions = phoenix_coupon_fractions(twin, [r.observation_time for r in resolved])
        for i, (f, c) in enumerate(zip(fractions, coupons)):
            amount = float(twin.get_coupon_payoff(i, year_fraction=f))
            if not is_close(amount, c.cash, rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
                raise ValidationError(f"twin coupon[{c.index}] pays {amount!r}, contract pays {c.cash!r}")


def _digital_twin(product, timeline: ContractTimeline, remaining, ts: datetime):
    terminal = timeline.terminal()
    if terminal not in remaining:
        raise ValidationError("the digital's terminal fixing is already history; a product without a lifecycle "
                              "ledger has nothing left to value in intraday mode")
    twin = deepcopy(product)
    twin.maturity = calendar_year_fraction(ts, terminal.timestamp)
    _clear_dates(twin)
    twin.settlement_convention = _settlement(calendar_year_fraction(ts, terminal.payment_timestamp) - twin.maturity)
    return twin


def _event_for_cashflow(cf: RealizedCashflow, timeline: ContractTimeline) -> Optional[ContractEvent]:
    """The contract event a tracker cashflow id names ("knock-out:4", "coupon:2", "maturity")."""
    if cf.cashflow_id == "maturity":
        return timeline.terminal()
    match = re.fullmatch(r"(knock-out|coupon):(\d+)", cf.cashflow_id)
    if match is None:
        return None
    kind, index = _TRACKER_IDS[match.group(1)], int(match.group(2))
    return next((e for e in timeline.events if e.kind is kind and e.index == index), None)


def _time_based_state(state: AutocallableLifecycleState, timeline: ContractTimeline, ts: datetime, cal):
    flows = []
    for cf in state.ledger.cashflows:
        if cf.determination_date is None:
            raise ValidationError(f"cashflow {cf.cashflow_id!r} is time-based; intraday mode places dated ledgers only")
        event = _event_for_cashflow(cf, timeline)
        if event is not None:
            if event.payment_timestamp.astimezone(cal.tz).date() != cf.payment_date.date():
                raise ValidationError(f"cashflow {cf.cashflow_id!r} pays on {cf.payment_date.date()} but its contract event "
                                      f"pays on {event.payment_timestamp.astimezone(cal.tz).date()}")
            det_ts, pay_ts = event.timestamp, event.payment_timestamp
        else:
            d = cf.determination_date.date()
            det_ts = cal.close_at(d) if cal.is_trading_day(d) else cal.payment_at(d)
            pay_ts = cal.payment_at(cf.payment_date.date())
        flows.append(RealizedCashflow(cashflow_id=cf.cashflow_id, event_type=cf.event_type, amount=cf.amount,
                                      determination_time=calendar_year_fraction(ts, det_ts),
                                      payment_time=calendar_year_fraction(ts, pay_ts), metadata=dict(cf.metadata)))
    new = deepcopy(state)
    new.ledger = LifecycleCashflowLedger(flows)
    new.valuation_point = ValuationPoint(time=0.0)
    return new


def build_numerical_contract(product, timeline: ContractTimeline, reconstruction: LifecycleReconstruction, *,
                             valuation_timestamp: datetime, phase: EventPhase, session_calendar) -> NumericalContract:
    ts = valuation_timestamp
    state = reconstruction.state
    remaining = timeline.remaining(ts, phase)
    taus = {e.event_id: calendar_year_fraction(ts, e.timestamp) for e in remaining}
    if any(t < 0.0 for t in taus.values()) or (phase is EventPhase.AFTER and any(t == 0.0 for t in taus.values())):
        raise ValidationError("remaining events must be strictly future (or at valuation under BEFORE)")
    if state is None:
        if hasattr(product, "barrier_config"):
            raise ValidationError("an autocallable needs a reconstructed lifecycle state")
        twin, tstate, terminated = _digital_twin(product, timeline, remaining, ts), None, False
    else:
        tstate = _time_based_state(state, timeline, ts, session_calendar)
        terminated = not tstate.alive
        twin = None if terminated else _autocallable_twin(product, timeline, remaining, ts, state)
    pending, paid = [], 0.0
    if tstate is not None:
        for cf in tstate.ledger.cashflows:
            if cf.payment_time > 0.0:
                pending.append((cf.cashflow_id, float(cf.amount), float(cf.payment_time)))
            else:
                paid += float(cf.amount)
    return NumericalContract(product=twin, lifecycle_state=tstate, remaining_events=tuple(remaining),
                             event_taus=MappingProxyType(taus),
                             maturity_tau=calendar_year_fraction(ts, timeline.terminal().timestamp),
                             knocked_in=bool(state.knocked_in) if state is not None else False,
                             terminated=terminated, pending_cashflows=tuple(pending), paid_cash=paid)
