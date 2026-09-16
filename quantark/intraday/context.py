"""One resolved, immutable valuation context per request (design §Engine integration).

Order of resolution is the design's sequence: validate/snapshot -> contractual
instants -> confirmed + provisional history -> numerical twin and clock map ->
identity. Engines and Greek bumps consume this object; they never re-decide
whether an observation happened.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from quantark.execution.cache.fingerprint import Uncanonicalizable, canonical_tree, fingerprint
from quantark.intraday.events import ContractTimeline, EventPhase, _schedule_env, resolve_timeline
from quantark.intraday.profile import IntradayTimeMap
from quantark.intraday.provisional import LifecycleReconstruction, reconstruct_lifecycle
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.timestamp import require_aware, to_utc
from quantark.intraday.twin import NumericalContract, build_numerical_contract
from quantark.param.vol import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError

_HORIZON_PAD = timedelta(days=2)


def value_tree(obj, _seen=frozenset()):
    """Structural value of ``obj`` for identities: canonical leaves, containers and objects walked by field.

    Unlike ``canonical_tree`` it never gives up on a composite (sets, mapping
    proxies, non-dataclass objects) and never embeds a memory address.
    """
    composite = isinstance(obj, (list, tuple, set, frozenset, Mapping)) or (
        not isinstance(obj, (type, Enum)) and (dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__")))
    if not composite:
        try:
            return canonical_tree(obj)
        except Uncanonicalizable:
            return ("repr", repr(obj))
    if id(obj) in _seen:
        return ("cycle", type(obj).__qualname__)
    seen = _seen | {id(obj)}
    if isinstance(obj, (list, tuple)):
        return ("seq", tuple(value_tree(x, seen) for x in obj))
    if isinstance(obj, (set, frozenset)):
        return ("set", tuple(sorted((value_tree(x, seen) for x in obj), key=repr)))
    if isinstance(obj, Mapping):
        return ("map", tuple(sorted(((repr(k), value_tree(v, seen)) for k, v in obj.items()), key=lambda kv: kv[0])))
    names = [f.name for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else []
    if hasattr(obj, "__dict__"):
        names += sorted(set(vars(obj)) - set(names))
    return ("obj", f"{type(obj).__module__}.{type(obj).__qualname__}",
            tuple((n, value_tree(getattr(obj, n), seen)) for n in names))


def _iso_utc(ts):
    return None if ts is None else to_utc(ts).isoformat()


def market_snapshot_id(env: PricingEnvironment) -> str:
    """Value identity of the market inputs (spot, its timestamp, vol, rate, dividend, basis)."""
    from quantark.execution import greeks as summaries
    return fingerprint(value_tree((
        float(env.spot), _iso_utc(env.spot_quote.timestamp),
        summaries._vol_summary(env), summaries._rate_summary(env), summaries._div_summary(env), env.basis_yield,
    )))


@dataclass(frozen=True)
class IntradayValuationContext:
    request: IntradayValuationRequest
    valuation_timestamp: datetime
    phase: EventPhase
    time_map: IntradayTimeMap
    pricing_env: PricingEnvironment            # NUMERICAL env: aware valuation_date, TradingClockVolSurface(inner, time_map), CALENDAR_DAYS
    timeline: ContractTimeline
    reconstruction: LifecycleReconstruction
    numerical: NumericalContract
    market_snapshot_id: str
    identity: str                              # everything that changes the economics

    @property
    def provisional(self) -> bool:
        return self.reconstruction.provisional

    @property
    def spot(self) -> float:
        return float(self.pricing_env.spot)


def _validate_env(env: PricingEnvironment) -> datetime:
    ts = require_aware(env.valuation_date, "pricing_env.valuation_date (the valuation timestamp)")
    if env.day_count_convention is not DayCountConvention.CALENDAR_DAYS:
        raise ValidationError("intraday mode uses calendar time as the numerical axis: "
                              "pricing_env.day_count_convention must be CALENDAR_DAYS")
    if env.vol_surface is None or env.spot_quote is None:
        raise ValidationError("intraday mode needs a vol surface and a spot quote")
    if isinstance(env.vol_surface, TradingClockVolSurface):
        raise ValidationError("pass the inner trading-quoted vol surface; the intraday resolver wraps it with the profile's clock")
    sq_ts = env.spot_quote.timestamp
    if sq_ts is not None:
        require_aware(sq_ts, "spot_quote.timestamp")
        if to_utc(sq_ts) > to_utc(ts):
            raise ValidationError("spot_quote.timestamp is after the valuation timestamp")
    return ts


def _snapshot_request(request: IntradayValuationRequest) -> IntradayValuationRequest:
    """The request with its economics copied out of the caller's hands.

    A resolved context is a frozen price function: rolling it, bumping it or
    curving it must keep answering for the market that was resolved. The frozen
    dataclass only stops field REASSIGNMENT — the spot quote, curves, surface,
    product, checkpoint and the session calendar's holiday set are all mutable
    objects the caller still owns, and editing one of them after resolution
    silently changes the price while the identity says nothing moved.
    """
    env = request.pricing_env
    snapshot_env = PricingEnvironment(
        rate_curve=deepcopy(env.rate_curve), valuation_date=env.valuation_date,
        spot_quote=deepcopy(env.spot_quote), vol_surface=deepcopy(env.vol_surface),
        div_yield=deepcopy(env.div_yield), basis_yield=env.basis_yield,
        day_count_convention=env.day_count_convention, bus_days_in_year=env.bus_days_in_year,
        calendar=env.calendar)
    return dataclasses.replace(request, product=deepcopy(request.product), pricing_env=snapshot_env,
                               session_calendar=request.session_calendar.snapshot(),
                               lifecycle_state=deepcopy(request.lifecycle_state))


def resolve_context(request: IntradayValuationRequest) -> IntradayValuationContext:
    """Resolve the request once: snapshot, instants, history, twin, clock map and identity."""
    _validate_env(request.pricing_env)
    request = _snapshot_request(request)
    env = request.pricing_env
    ts = _validate_env(env)
    phase = request.event_phase
    cal, profile = request.session_calendar, request.variance_profile
    if profile.session_weights and len(profile.session_weights) != len(cal.sessions):
        raise ValidationError(f"profile declares {len(profile.session_weights)} sessions, calendar has {len(cal.sessions)}")
    timeline = resolve_timeline(request.product, cal, env, fixing_time_of_day=request.fixing_time_of_day,
                                schedule_origin=request.schedule_origin)
    reconstruction = reconstruct_lifecycle(request.product, timeline, request.lifecycle_state, request.fixings,
                                           valuation_timestamp=ts, phase=phase, spot=float(env.spot),
                                           spot_timestamp=env.spot_quote.timestamp,
                                           schedule_env=_schedule_env(request.product, env), session_calendar=cal)
    numerical = build_numerical_contract(request.product, timeline, reconstruction, valuation_timestamp=ts,
                                         phase=phase, session_calendar=cal,
                                         schedule_env=_schedule_env(request.product, env))
    last_payment = max((e.payment_timestamp for e in timeline.events), key=to_utc)
    horizon = max(last_payment, ts, key=to_utc) + _HORIZON_PAD
    time_map = IntradayTimeMap(cal, profile, ts, horizon)
    numerical_env = PricingEnvironment(rate_curve=env.rate_curve, valuation_date=ts, spot_quote=env.spot_quote,
                                       vol_surface=TradingClockVolSurface(env.vol_surface, time_map),
                                       div_yield=env.div_yield, basis_yield=env.basis_yield,
                                       day_count_convention=DayCountConvention.CALENDAR_DAYS,
                                       bus_days_in_year=profile.days_per_year, calendar=cal.calendar)
    snapshot_id = market_snapshot_id(env)
    identity = fingerprint(value_tree((
        _iso_utc(ts), phase.value, profile.identity(), cal.identity(), snapshot_id,
        fingerprint(value_tree(request.product)),
        tuple((_iso_utc(a.scheduled_at), a.assumed_value, _iso_utc(a.spot_timestamp), a.event_ids)
              for a in reconstruction.assumptions),
        None if reconstruction.continuous_assumption is None else value_tree(tuple(
            _iso_utc(t) for t in (reconstruction.continuous_assumption.uncovered_from,
                                  reconstruction.continuous_assumption.uncovered_to,
                                  reconstruction.continuous_assumption.assumed_hit_at))),
        tuple(sorted((_iso_utc(f.timestamp), f.value) for f in request.fixings)),
        reconstruction.checkpoint_fingerprint,
        None if request.fixing_time_of_day is None else request.fixing_time_of_day.isoformat(),
        _iso_utc(request.schedule_origin),
    )))
    return IntradayValuationContext(request=request, valuation_timestamp=ts, phase=phase, time_map=time_map,
                                    pricing_env=numerical_env, timeline=timeline, reconstruction=reconstruction,
                                    numerical=numerical, market_snapshot_id=snapshot_id, identity=identity)
