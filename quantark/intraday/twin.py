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
from quantark.intraday.provisional import ASSUMED, DETERMINATION_TIMESTAMP, PAYMENT_TIMESTAMP, LifecycleReconstruction
from quantark.intraday.timestamp import calendar_year_fraction, to_utc
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


def _ko_reset_twin(product, timeline: ContractTimeline, remaining, ts: datetime, state, schedule_env):
    """Float-time KO-reset snowball carrying both KO schedules.

    Its engines read the KO accrual through ``compute_ko_accrual_factor``, which
    for a dateless schedule is ``accrued_offset + tau`` and ignores positional
    factors. With every fixing at one time of day, (contractual accrual - tau)
    is the same for every record, so one ``accrued_offset`` reproduces the
    contract's cash and both contract tenors; that is verified, else fail closed.
    """
    from quantark.intraday.events import ko_reset_records
    twin = deepcopy(product)
    bc, pbc = twin.barrier_config, twin.post_barrier_config
    pre_all = [e for e in timeline.events if e.kind is EventKind.KO and e.regime == "pre"]
    if not pre_all:
        raise ValidationError("a KO-reset snowball needs a pre-KI KO schedule")
    offset = float(product._resolve_pre_contract_tenor(schedule_env)) - calendar_year_fraction(ts, pre_all[-1].timestamp)
    if offset < 0.0:
        raise CapabilityError("valuation before the contract's accrual origin: the KO-reset twin cannot carry a "
                              "negative accrued offset")

    def schedule(config, regime):
        events = [e for e in remaining if e.kind is EventKind.KO and e.regime == regime]
        src = list(config.ko_observation_schedule.records)
        records = []
        for e in events:
            rate = src[e.index].return_rate
            rate = float(product._get_barrier_at(config.ko_rate, e.index, "KO rate")) if rate is None else float(rate)
            records.append(ObservationRecord(observation_time=calendar_year_fraction(ts, e.timestamp), barrier=e.barrier,
                                             return_rate=rate, settlement_time=calendar_year_fraction(ts, e.payment_timestamp)))
        indices = [e.index for e in events]
        barrier = _pick(config.ko_barrier, indices) if indices else (
            config.ko_barrier[-1] if isinstance(config.ko_barrier, (list, tuple)) else config.ko_barrier)
        rate = _pick(config.ko_rate, indices) if indices else (
            config.ko_rate[-1] if isinstance(config.ko_rate, (list, tuple)) else config.ko_rate)
        new = replace(config, ko_barrier=barrier, ko_rate=rate, ko_observation_dates=None,
                      ko_observation_schedule=ObservationSchedule(records=records,
                                                                  aggregation_mode=config.ko_observation_schedule.aggregation_mode,
                                                                  frequency=config.ko_observation_schedule.frequency))
        return new, events

    pre_config, pre_events = schedule(bc, "pre")
    post_config, post_events = schedule(pbc, "post")
    if product.has_ki_barrier and timeline.continuous_ki_barrier is None:
        kis = [e for e in remaining if e.kind is EventKind.KI]
        ki_records = [ObservationRecord(observation_time=calendar_year_fraction(ts, e.timestamp), barrier=e.barrier) for e in kis]
        pre_config = replace(pre_config, ki_barrier=_pick(bc.ki_barrier, [e.index for e in kis]) if kis else bc.ki_barrier,
                             ki_observation_dates=None, ki_observation_schedule=ObservationSchedule(records=ki_records))
    twin.barrier_config, twin.post_barrier_config = pre_config, post_config
    twin.accrual_config = replace(twin.accrual_config, accrual_factors=None, accrual_dates=None, accrued_offset=offset)
    terminal = timeline.terminal()
    twin.maturity = calendar_year_fraction(ts, terminal.timestamp)
    _clear_dates(twin)
    twin.settlement_convention = _settlement(calendar_year_fraction(ts, terminal.payment_timestamp) - twin.maturity)
    setattr(twin, "_otc_lifecycle_knocked_in", bool(state.knocked_in))
    env = _cash_env(ts, twin)
    for config, events in ((pre_config, pre_events), (post_config, post_events)):
        records = ko_reset_records(twin, config, env)
        if len(records) != len(events):
            raise ValidationError("KO-reset twin resolves a different number of KO observations")
        for (_rec, _src, _rate, cash), e in zip(records, events):
            if not is_close(cash, e.cash, rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
                raise CapabilityError(f"KO-reset twin {e.event_id} pays {cash!r}, the contract pays {e.cash!r}: fixings at "
                                      "different times of day cannot share one accrued offset")
    for name in ("_resolve_pre_contract_tenor", "_resolve_post_contract_tenor"):
        got, want = float(getattr(twin, name)(env)), float(getattr(product, name)(schedule_env))
        if not is_close(got, want, rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
            raise CapabilityError(f"KO-reset twin {name} = {got!r}, contract {want!r}")
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


def event_for_cashflow(cf: RealizedCashflow, timeline: ContractTimeline, tz=None) -> Optional[ContractEvent]:
    """The contract event a tracker cashflow id names.

    Autocallable tracker ids carry the schedule index ("knock-out:4", "coupon:2",
    "maturity"); barrier tracker ids carry the determination date ("ko:<date>",
    "expiry:<date>"). A flow the intraday layer assumed itself names no event.
    """
    if cf.cashflow_id == "maturity" or cf.cashflow_id.startswith("expiry:"):
        return timeline.terminal()
    match = re.fullmatch(r"(knock-out|coupon):(\d+)", cf.cashflow_id)
    if match is not None:
        kind, index = _TRACKER_IDS[match.group(1)], int(match.group(2))
        return next((e for e in timeline.events if e.kind is kind and e.index == index), None)
    if cf.cashflow_id.startswith("ko:") and not cf.metadata.get(ASSUMED) and cf.determination_date is not None:
        day = cf.determination_date.date()
        same_day = [e for e in timeline.events if e.kind is EventKind.KO
                    and (e.timestamp.astimezone(tz) if tz is not None else e.timestamp).date() == day]
        return same_day[0] if len(same_day) == 1 else None
    return None


def place_cashflow(cf: RealizedCashflow, timeline: ContractTimeline, cal):
    """(determination instant, payment instant, contract event or None) of a dated ledger flow; cash-checked."""
    if cf.determination_date is None:
        raise ValidationError(f"cashflow {cf.cashflow_id!r} is time-based; intraday mode places dated ledgers only")
    if DETERMINATION_TIMESTAMP in cf.metadata:
        return (datetime.fromisoformat(cf.metadata[DETERMINATION_TIMESTAMP]),
                datetime.fromisoformat(cf.metadata[PAYMENT_TIMESTAMP]), None)
    event = event_for_cashflow(cf, timeline, cal.tz)
    expected = event.cash if event is not None else None
    if event is None and cf.cashflow_id.startswith("ko:") and timeline.continuous_barrier is not None:
        expected = timeline.continuous_barrier.hit_cash            # a continuous barrier touched at a close
    if expected is not None and not is_close(float(cf.amount), float(expected), rel_tol=_CASH_REL_TOL, abs_tol=_CASH_ABS_TOL):
        raise CapabilityError(f"ledger cash {cf.amount!r} for {cf.cashflow_id!r} differs from the contract's "
                              f"{expected!r} (the daily lifecycle tracker and the engines disagree on this amount); "
                              "not in the intraday inventory")
    if event is not None and event.kind is not EventKind.TERMINAL:
        if event.payment_timestamp.astimezone(cal.tz).date() != cf.payment_date.date():
            raise ValidationError(f"cashflow {cf.cashflow_id!r} pays on {cf.payment_date.date()} but its contract event "
                                  f"pays on {event.payment_timestamp.astimezone(cal.tz).date()}")
        return event.timestamp, event.payment_timestamp, event
    d = cf.determination_date.date()
    if event is not None:
        det_ts = event.timestamp
    else:
        det_ts = cal.close_at(d) if cal.is_trading_day(d) else cal.payment_at(d)
    pay_ts = max(cal.payment_at(cf.payment_date.date()), det_ts, key=to_utc)
    if event is not None and event.payment_timestamp.astimezone(cal.tz).date() == cf.payment_date.date():
        pay_ts = event.payment_timestamp
    return det_ts, pay_ts, event


def _time_based_state(state, timeline: ContractTimeline, ts: datetime, cal):
    flows = []
    for cf in state.ledger.cashflows:
        det_ts, pay_ts, _ = place_cashflow(cf, timeline, cal)
        flows.append(RealizedCashflow(cashflow_id=cf.cashflow_id, event_type=cf.event_type, amount=cf.amount,
                                      determination_time=calendar_year_fraction(ts, det_ts),
                                      payment_time=calendar_year_fraction(ts, pay_ts), metadata=dict(cf.metadata)))
    new = deepcopy(state)
    new.ledger = LifecycleCashflowLedger(flows)
    new.valuation_point = ValuationPoint(time=0.0)
    return new


def _barrier_twin(product, timeline: ContractTimeline, remaining, ts: datetime, state):
    """Float-time barrier / touch product; a knocked-in barrier option is its European vanilla."""
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    terminal = timeline.terminal()
    if terminal not in remaining:
        raise ValidationError("the terminal fixing is history but the lifecycle is still alive")
    maturity = calendar_year_fraction(ts, terminal.timestamp)
    convention = _settlement(calendar_year_fraction(ts, terminal.payment_timestamp) - maturity)
    if isinstance(product, BarrierOption) and state.knocked_in:
        return EuropeanVanillaOption(strike=float(product.strike), option_type=product.option_type, maturity=maturity,
                                     contract_multiplier=float(product.contract_multiplier), settlement_convention=convention)
    twin = deepcopy(product)
    marks = [e for e in remaining if e.kind in (EventKind.KO, EventKind.KI)]
    if product.observation_schedule is not None:
        records = [ObservationRecord(observation_time=calendar_year_fraction(ts, e.timestamp), barrier=e.barrier,
                                     payoff=e.cash, settlement_time=calendar_year_fraction(ts, e.payment_timestamp))
                   for e in marks]
        twin.observation_schedule = ObservationSchedule(records=records,
                                                        aggregation_mode=product.observation_schedule.aggregation_mode,
                                                        frequency=product.observation_schedule.frequency)
    if hasattr(twin, "observation_dates"):
        twin.observation_dates = None
    twin.maturity = maturity
    _clear_dates(twin)
    twin.settlement_convention = convention
    if marks and not twin.observation_schedule.records:
        raise ValidationError("discrete barrier twin lost its remaining observations")
    return twin


def build_numerical_contract(product, timeline: ContractTimeline, reconstruction: LifecycleReconstruction, *,
                             valuation_timestamp: datetime, phase: EventPhase, session_calendar,
                             schedule_env=None) -> NumericalContract:
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    ts = valuation_timestamp
    state = reconstruction.state
    remaining = timeline.remaining(ts, phase)
    taus = {e.event_id: calendar_year_fraction(ts, e.timestamp) for e in remaining}
    if any(t < 0.0 for t in taus.values()) or (phase is EventPhase.AFTER and any(t == 0.0 for t in taus.values())):
        raise ValidationError("remaining events must be strictly future (or at valuation under BEFORE)")
    if state is None:
        if hasattr(product, "barrier_config") or hasattr(product, "observation_type"):
            raise ValidationError("a product with a lifecycle needs a reconstructed lifecycle state")
        twin, tstate, terminated = _digital_twin(product, timeline, remaining, ts), None, False
    else:
        tstate = _time_based_state(state, timeline, ts, session_calendar)
        terminated = not tstate.alive
        if terminated:
            twin = None
        elif type(product) is KnockOutResetSnowballOption:
            if schedule_env is None:
                raise ValidationError("a KO-reset twin needs the contract's schedule environment")
            twin = _ko_reset_twin(product, timeline, remaining, ts, state, schedule_env)
        elif hasattr(product, "barrier_config"):
            twin = _autocallable_twin(product, timeline, remaining, ts, state)
        else:
            twin = _barrier_twin(product, timeline, remaining, ts, state)
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
