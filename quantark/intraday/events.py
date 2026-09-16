"""Contract events as timezone-aware instants.

The ORIGINAL product's own resolvers produce the contractual cash table
(Snowball/Phoenix: ``resolve_ko_observations``/``resolve_ki_observations`` on an
environment anchored at ``initial_date``, so nothing is filtered and every cash
amount uses the contract's day count). This module only attaches instants:
date-only records -> exchange close (or the contract's fixing time), explicit
``observation_timestamp`` -> as given, float schedules -> origin + tau.
Ordering is by instant (UTC) then kind priority — never by a float tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import List, Optional, Tuple

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.intraday.session import TradingSessionCalendar
from quantark.intraday.timestamp import add_year_fraction, require_aware, to_utc
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType
from quantark.util.exceptions import ValidationError


class EventKind(Enum):
    KO = "ko"
    KI = "ki"
    COUPON = "coupon"
    TERMINAL = "terminal"


#: Determination order of events sharing one instant.
KIND_PRIORITY = {EventKind.KI: 0, EventKind.KO: 1, EventKind.COUPON: 2, EventKind.TERMINAL: 3}


class EventPhase(Enum):
    """Whether an event exactly at the valuation instant is still undetermined (BEFORE) or history (AFTER)."""

    BEFORE = "before"
    AFTER = "after"

    @classmethod
    def parse(cls, value) -> "EventPhase":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            raise ValidationError("event_phase must be 'before' or 'after'") from None


@dataclass(frozen=True)
class ContractEvent:
    """One contractual determination: when, when paid, at which level, for how much cash."""

    event_id: str
    kind: EventKind
    index: int                    # position in the ORIGINAL schedule of that kind (KO index doubles as the Phoenix coupon index)
    timestamp: datetime           # aware determination instant
    payment_timestamp: datetime   # aware payment instant
    barrier: Optional[float]
    cash: Optional[float]         # contractual cash if determined in the money (KO redemption, coupon); None for KI/terminal
    date_only: bool               # resolved from a date by convention (True) or from an explicit timestamp (False)
    regime: str = ""              # KO-reset KO events: "pre" (before KI) or "post" (after KI); empty otherwise

    @staticmethod
    def make(kind, index, timestamp, payment_timestamp, barrier, cash, date_only, regime: str = "") -> "ContractEvent":
        require_aware(timestamp, "event timestamp")
        require_aware(payment_timestamp, "payment timestamp")
        if to_utc(payment_timestamp) < to_utc(timestamp):
            raise ValidationError(f"{kind.value}[{index}] pays before it is determined")
        label = f"{kind.value}_{regime}" if regime else kind.value
        return ContractEvent(f"{label}[{index}]@{timestamp.isoformat()}", kind, int(index), timestamp,
                             payment_timestamp, None if barrier is None else float(barrier),
                             None if cash is None else float(cash), bool(date_only), regime)


def monitoring_of(timeline: "ContractTimeline") -> str:
    """Which capability-matrix monitoring column this contract falls in.

    Lives here rather than in the service because the qualification gate needs it
    too: a certificate earned on discrete monitoring says nothing about the same
    product under a continuously observed barrier.
    """
    if timeline.continuous_ki_barrier is not None or timeline.continuous_barrier is not None:
        return "continuous"
    if all(e.kind is EventKind.TERMINAL for e in timeline.events):
        return "terminal"
    return "discrete"


def _order_key(e: ContractEvent):
    return (to_utc(e.timestamp), KIND_PRIORITY[e.kind], e.index)


@dataclass(frozen=True)
class ContinuousBarrier:
    """A single barrier monitored continuously (barrier options, one-touch / no-touch).

    ``hit_cash`` is what a hit fixes: the rebate of a knock-out or one-touch,
    0.0 for a no-touch, None for a knock-in (which switches to the vanilla).
    ``rebate`` is the never-hit amount of a knock-in paid at expiry.
    """

    level: float
    is_up: bool
    is_knock_out: bool
    pays_at_hit: bool
    rebate: float
    hit_cash: Optional[float]


@dataclass(frozen=True)
class ContractTimeline:
    """Every remaining-or-past event of one contract, ordered by (instant, kind priority)."""

    events: Tuple[ContractEvent, ...]
    initial_timestamp: Optional[datetime]
    contractual_tenor: float                # years, the ORIGINAL product's contract tenor
    continuous_ki_barrier: Optional[float]  # set when the product monitors KI continuously
    schedule_origin: Optional[datetime]     # for float-time products
    continuous_barrier: Optional[ContinuousBarrier] = None

    def __post_init__(self):
        object.__setattr__(self, "events", tuple(sorted(self.events, key=_order_key)))
        if sum(1 for e in self.events if e.kind is EventKind.TERMINAL) != 1:
            raise ValidationError("a contract timeline needs exactly one terminal event")
        ids = [e.event_id for e in self.events]
        if len(set(ids)) != len(ids):
            raise ValidationError("duplicate contract events at one instant of one kind")

    def terminal(self) -> ContractEvent:
        return next(e for e in self.events if e.kind is EventKind.TERMINAL)

    def due(self, ts: datetime, phase: EventPhase) -> Tuple[ContractEvent, ...]:
        """Events determined by ``ts``: strictly earlier, or exactly at ``ts`` under AFTER."""
        key = to_utc(ts, "ts")
        return tuple(e for e in self.events
                     if to_utc(e.timestamp) < key or (to_utc(e.timestamp) == key and phase is EventPhase.AFTER))

    def remaining(self, ts: datetime, phase: EventPhase) -> Tuple[ContractEvent, ...]:
        """Events still undetermined at ``ts``: strictly later, or exactly at ``ts`` under BEFORE."""
        key = to_utc(ts, "ts")
        return tuple(e for e in self.events
                     if to_utc(e.timestamp) > key or (to_utc(e.timestamp) == key and phase is EventPhase.BEFORE))

    def at(self, ts: datetime) -> Tuple[ContractEvent, ...]:
        key = to_utc(ts, "ts")
        return tuple(e for e in self.events if to_utc(e.timestamp) == key)

    def by_id(self, event_id: str) -> ContractEvent:
        for e in self.events:
            if e.event_id == event_id:
                return e
        raise ValidationError(f"unknown event id {event_id!r}")


# ---------------------------------------------------------------------------
def _schedule_env(product, template_env: PricingEnvironment) -> PricingEnvironment:
    """Naive environment anchored at the contract's inception (dates) or as-is (floats).

    Only the product's own resolvers read it, to produce the contractual cash
    table; nothing from the template's market enters the timeline. Without an
    ``initial_date`` the anchor must still precede every contract date (a
    same-day date-only expiry is midnight, before any intraday valuation).
    """
    initial = getattr(product, "initial_date", None)
    if initial is not None:
        anchor = initial
    else:
        dates = _contract_dates(product)
        if dates and hasattr(product, "barrier_config"):
            raise ValidationError("dated autocallables need initial_date in intraday mode: it anchors the "
                                  "contractual accrual, which must not move with the valuation timestamp")
        anchor = min(dates) - timedelta(days=1) if dates else template_env.valuation_date.replace(tzinfo=None)
    return PricingEnvironment(rate_curve=template_env.rate_curve, valuation_date=anchor,
                              spot_quote=template_env.spot_quote, vol_surface=template_env.vol_surface,
                              div_yield=template_env.div_yield, basis_yield=template_env.basis_yield)


def _contract_dates(product) -> List[datetime]:
    dates = [getattr(product, name, None) for name in ("exercise_date", "maturity_date", "settlement_date")]
    bc = getattr(product, "barrier_config", None)
    schedules = [getattr(bc, name, None) for name in ("ko_observation_schedule", "ki_observation_schedule")]
    schedules.append(getattr(product, "observation_schedule", None))
    for schedule in schedules:
        if schedule is not None:
            dates.extend(r.observation_date for r in schedule.records)
    return [d for d in dates if d is not None]


def _instant(cal: TradingSessionCalendar, *, timestamp, dt, tau, origin, fixing_time, what):
    """(determination instant, date_only) for one record."""
    if timestamp is not None:
        return require_aware(timestamp, f"{what} timestamp"), False
    if dt is not None:
        d = dt.date()
        if not cal.is_trading_day(d):
            raise ValidationError(f"{what} date {d} is not a trading day on {cal.name}; "
                                  "supply an explicit observation_timestamp")
        if fixing_time is not None:
            return cal.localize(d, fixing_time), True
        return cal.close_at(d), True
    if tau is None:
        raise ValidationError(f"{what} has neither a date nor a time")
    if origin is None:
        raise ValidationError("float schedules need schedule_origin in intraday mode")
    return add_year_fraction(origin, float(tau)), True


def _payment(cal: TradingSessionCalendar, *, timestamp, dt, tau, origin, fallback):
    """Payment instant; ``fallback`` is the determination instant (pay at determination)."""
    if timestamp is not None:
        return require_aware(timestamp, "payment timestamp")
    if dt is not None:
        # A date-only payment is deemed made at the calendar's payment time, but
        # never before its own determination (e.g. a fixing after the close).
        deemed = cal.payment_at(dt.date())
        return fallback if to_utc(deemed) < to_utc(fallback) else deemed
    if tau is not None:
        if origin is None:
            raise ValidationError("float schedules need schedule_origin in intraday mode")
        return add_year_fraction(origin, float(tau))
    return fallback


def _ki_is_continuous(product) -> bool:
    bc = product.barrier_config
    return bool(bc.ki_continuous or bc.ki_observation_type == ObservationType.CONTINUOUS)


def phoenix_coupon_fractions(product, ko_times: List[float]) -> List[float]:
    """Contractual per-period coupon year fractions over the FULL KO schedule.

    A rate that is not annualized is the period's amount, so every fraction is 1.
    Otherwise the precedence the QUAD V2 Phoenix adapter applies at inception:
    explicit ``coupon_year_fractions``, positional ``accrual_factors``, a fixed
    fraction, the coupon day count between consecutive contract dates (from
    ``initial_date``), else consecutive observation-time differences.
    """
    raw = product.barrier_config.ko_observation_schedule.records
    n = len(raw)
    if not product.is_coupon_rate_annualized:
        return [1.0] * n
    explicit = getattr(product.coupon_config, "coupon_year_fractions", None)
    if explicit is not None:
        if len(explicit) != n:
            raise ValidationError(f"coupon_year_fractions has {len(explicit)} entries for {n} KO records")
        return [float(f) for f in explicit]
    factors = product.accrual_config.accrual_factors
    if factors is not None:
        if len(factors) != n:
            raise ValidationError(f"accrual_factors has {len(factors)} entries for {n} KO records")
        return [float(f) for f in factors]
    fixed = product.coupon_config.fixed_coupon_year_fraction
    if fixed is not None:
        return [float(fixed)] * n
    if product.initial_date is not None and all(r.observation_date is not None for r in raw):
        dates = [product.initial_date] + [r.observation_date for r in raw]
        return [float(product.get_coupon_year_fraction(dates[i], dates[i + 1])) for i in range(n)]
    if len(ko_times) != n:
        raise ValidationError("Phoenix coupon fractions need one observation time per KO record")
    return [float(ko_times[0])] + [float(ko_times[i] - ko_times[i - 1]) for i in range(1, n)]


def ko_reset_records(product, config, env):
    """(resolved record, source record, rate, contractual cash) of one KO-reset schedule, as QUAD V2 composes it."""
    resolved, rates, sources = product._resolve_ko_schedule(config, env)
    principal = product.initial_price * product.contract_multiplier if product.payoff_config.include_principal else 0.0
    out = []
    for rec, rate, src in zip(resolved, rates, sources):
        accrual = product.compute_ko_accrual_factor(rec.observation_time, src, env)
        cash = product.initial_price * product.contract_multiplier * float(rate) * float(accrual) + principal
        out.append((rec, src, float(rate), float(cash)))
    return out


def _ko_reset_events(product, cal, env, origin, fixing_time):
    from quantark.execution.errors import CapabilityError
    from quantark.util.enum.option_enums import PostKOScheduleMode
    if product.post_ko_mode is not PostKOScheduleMode.ABSOLUTE:
        raise CapabilityError("relative-to-hit (REBASED) KO reset schedules are not in the intraday inventory: the post-KI "
                              "schedule would depend on the unknown hit instant")
    terminal = _terminal_event(product, cal, env, origin, fixing_time)
    events = [terminal]
    for regime, config in (("pre", product.barrier_config), ("post", product.post_barrier_config)):
        schedule = config.ko_observation_schedule
        records = ko_reset_records(product, config, env)
        if schedule is None or len(records) != len(schedule.records):
            raise ValidationError(f"schedule env must keep every {regime}-KI KO record active (anchor at initial_date)")
        for i, (rec, src, _rate, cash) in enumerate(records):
            ts, date_only = _instant(cal, timestamp=src.observation_timestamp, dt=rec.observation_date,
                                     tau=rec.observation_time, origin=origin, fixing_time=fixing_time, what=f"ko_{regime}[{i}]")
            if product.accrual_config.coupon_pay_type.name == "EXPIRY":
                pay = max(terminal.payment_timestamp, ts, key=to_utc)
            else:
                pay = _payment(cal, timestamp=src.settlement_timestamp, dt=rec.settlement_date, tau=rec.settlement_time,
                               origin=origin, fallback=ts)
            events.append(ContractEvent.make(EventKind.KO, i, ts, pay, rec.barrier, cash, date_only, regime))
    continuous = None
    if product.has_ki_barrier:
        if _ki_is_continuous(product):
            continuous = float(product.barrier_config.ki_barrier)
        else:
            ki_src = list(product.barrier_config.ki_observation_schedule.records)
            for i, rec in enumerate(product.resolve_ki_observations(env)):
                ts, date_only = _instant(cal, timestamp=ki_src[i].observation_timestamp, dt=rec.observation_date,
                                         tau=rec.observation_time, origin=origin, fixing_time=fixing_time, what=f"ki[{i}]")
                events.append(ContractEvent.make(EventKind.KI, i, ts, ts, rec.barrier, None, date_only))
    return events, continuous


def _terminal_event(product, cal, env, origin, fixing_time) -> ContractEvent:
    timing = resolve_terminal_timing(product, env)
    exercise = getattr(product, "exercise_date", None)
    ts, date_only = _instant(cal, timestamp=None, dt=exercise,
                             tau=None if exercise is not None else product.get_maturity(env),
                             origin=origin, fixing_time=fixing_time, what="terminal")
    payment_date = getattr(timing, "payment_date", None)
    pay = _payment(cal, timestamp=None, dt=payment_date,
                   tau=None if payment_date is not None else timing.payment_time, origin=origin, fallback=ts)
    return ContractEvent.make(EventKind.TERMINAL, 0, ts, pay, None, None, date_only)


def _autocallable_events(product, cal, env, origin, fixing_time, with_coupons: bool):
    ko_sched = product.barrier_config.ko_observation_schedule
    if ko_sched is None:
        raise ValidationError("intraday autocallables need a KO ObservationSchedule")
    ko_src = list(ko_sched.records)
    ko_res = product.resolve_ko_observations(env)
    if len(ko_src) != len(ko_res):
        raise ValidationError("schedule env must keep every KO record active (anchor at initial_date)")
    events = []
    coupon_cash = None
    if with_coupons:
        fractions = phoenix_coupon_fractions(product, [rec.observation_time for rec in ko_res])
        coupon_cash = [float(product.get_coupon_payoff(i, year_fraction=f)) for i, f in enumerate(fractions)]
    for i, rec in enumerate(ko_res):
        src = ko_src[i]
        ts, date_only = _instant(cal, timestamp=src.observation_timestamp, dt=rec.observation_date,
                                 tau=rec.observation_time, origin=origin, fixing_time=fixing_time, what=f"ko[{i}]")
        pay = _payment(cal, timestamp=src.settlement_timestamp, dt=rec.settlement_date,
                       tau=rec.settlement_time, origin=origin, fallback=ts)
        events.append(ContractEvent.make(EventKind.KO, i, ts, pay, rec.barrier, rec.payoff, date_only))
        if with_coupons:
            events.append(ContractEvent.make(EventKind.COUPON, i, ts, pay, product.get_coupon_barrier_at(i),
                                             coupon_cash[i], date_only))
    continuous = None
    if product.has_ki_barrier:
        if _ki_is_continuous(product):
            kb = product.barrier_config.ki_barrier
            if isinstance(kb, (list, tuple)):
                raise ValidationError("continuous KI needs a scalar barrier in intraday mode")
            continuous = float(kb)
        else:
            ki_sched = product.barrier_config.ki_observation_schedule
            ki_src = list(ki_sched.records) if ki_sched is not None else None
            for i, rec in enumerate(product.resolve_ki_observations(env)):
                src = ki_src[i] if ki_src is not None else None
                ts, date_only = _instant(cal, timestamp=getattr(src, "observation_timestamp", None),
                                         dt=rec.observation_date, tau=rec.observation_time, origin=origin,
                                         fixing_time=fixing_time, what=f"ki[{i}]")
                events.append(ContractEvent.make(EventKind.KI, i, ts, ts, rec.barrier, None, date_only))
    events.append(_terminal_event(product, cal, env, origin, fixing_time))
    return events, continuous


def barrier_terms(product) -> ContinuousBarrier:
    """What a hit of this single-barrier product means (engine semantics: rebates carry no multiplier)."""
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    if isinstance(product, OneTouchOption):
        return ContinuousBarrier(level=float(product.barrier), is_up=bool(product.is_up_barrier), is_knock_out=True,
                                 pays_at_hit=bool(product.payment_at_hit), rebate=float(product.rebate),
                                 hit_cash=float(product.get_payoff(float(product.barrier), touched=True)))
    knock_out = bool(product.is_knock_out)
    return ContinuousBarrier(level=float(product.barrier), is_up=bool(product.is_up_barrier), is_knock_out=knock_out,
                             pays_at_hit=bool(product.pay_at_hit), rebate=float(product.rebate),
                             hit_cash=float(product.rebate) if knock_out else None)


def _barrier_events(product, cal, env, origin, fixing_time):
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    terms = barrier_terms(product)
    terminal = _terminal_event(product, cal, env, origin, fixing_time)
    events = [terminal]
    obs = product.observation_type
    if obs == ObservationType.CONTINUOUS:
        return events, terms
    if obs == ObservationType.EXPIRY:
        return events, None
    schedule = product.observation_schedule
    if schedule is None or not schedule.records:
        raise ValidationError("discrete barrier monitoring needs an ObservationSchedule")
    no_touch = isinstance(product, OneTouchOption) and product.is_no_touch
    resolved = schedule.resolve(pricing_env=env, default_barrier=terms.level,
                                default_payoff=0.0 if no_touch else float(terms.hit_cash or 0.0),
                                require_single=True, product=product)
    kind = EventKind.KO if terms.is_knock_out else EventKind.KI
    for i, (src, rec) in enumerate(zip(schedule.records, resolved)):
        ts, date_only = _instant(cal, timestamp=src.observation_timestamp, dt=rec.observation_date, tau=rec.observation_time,
                                 origin=origin, fixing_time=fixing_time, what=f"{kind.value}[{i}]")
        if terms.pays_at_hit:
            pay = _payment(cal, timestamp=src.settlement_timestamp, dt=rec.settlement_date, tau=rec.settlement_time,
                           origin=origin, fallback=ts)
        else:
            pay = max(terminal.payment_timestamp, ts, key=to_utc)
        cash = (0.0 if no_touch else float(rec.payoff)) if kind is EventKind.KO else None
        events.append(ContractEvent.make(kind, i, ts, pay, rec.barrier, cash, date_only))
    return events, None


def resolve_timeline(product, session_calendar: TradingSessionCalendar, template_env: PricingEnvironment, *,
                     fixing_time_of_day: Optional[time] = None,
                     schedule_origin: Optional[datetime] = None) -> ContractTimeline:
    """Resolve every contractual event of ``product`` to aware instants (the product is not mutated)."""
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    if schedule_origin is not None:
        require_aware(schedule_origin, "schedule_origin")
    if fixing_time_of_day is not None and not isinstance(fixing_time_of_day, time):
        raise ValidationError("fixing_time_of_day must be a datetime.time")
    env = _schedule_env(product, template_env)
    continuous_barrier = None
    if type(product) is KnockOutResetSnowballOption:
        events, continuous = _ko_reset_events(product, session_calendar, env, schedule_origin, fixing_time_of_day)
    elif type(product) in (BarrierOption, OneTouchOption):
        events, continuous_barrier = _barrier_events(product, session_calendar, env, schedule_origin, fixing_time_of_day)
        continuous = None
    elif type(product) is PhoenixOption:
        events, continuous = _autocallable_events(product, session_calendar, env, schedule_origin, fixing_time_of_day, True)
    elif type(product) is SnowballOption:
        events, continuous = _autocallable_events(product, session_calendar, env, schedule_origin, fixing_time_of_day, False)
    elif type(product) is CashOrNothingDigitalOption:
        events = [_terminal_event(product, session_calendar, env, schedule_origin, fixing_time_of_day)]
        continuous = None
    else:
        raise ValidationError(f"resolve_timeline: {type(product).__name__} is not in the intraday timeline inventory")
    initial = getattr(product, "initial_date", None)
    if initial is not None:
        d = initial.date()
        initial_ts = session_calendar.close_at(d) if session_calendar.is_trading_day(d) else session_calendar.payment_at(d)
    else:
        initial_ts = schedule_origin
    if hasattr(product, "get_contract_tenor"):
        tenor = float(product.get_contract_tenor(env))
    else:
        tenor = float(product.get_maturity(env))
    return ContractTimeline(tuple(events), initial_ts, tenor, continuous, schedule_origin, continuous_barrier)
