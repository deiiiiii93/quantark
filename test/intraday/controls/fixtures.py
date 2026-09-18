"""Fixtures of the independent numerical controls: products, clocks, markets, confirmed history, cell contexts.

A ``Cell`` names one (product, engine, profile, time-to-fixing, spot offset, barrier) point; ``build_context``
resolves it. The controls in this package compare the runtime with independent implementations on these
points as regression tests. They license nothing: accuracy is certified offline by modelvalidation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from math import exp, sqrt

from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.intraday import EventKind, Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import BarrierDirection, BarrierType, ObservationType, OptionType, TouchType

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)
SHORT_EXPIRY = datetime(2026, 9, 16)          # the snowball's sixth fixing day: every product's ladder ends at its 15:00 close

#: The daily-KI fixture's close and its time-to-fixing ladder. Every rung sits inside the 24 h gap after the
#: previous close (2026-09-09 15:00): a one-day rung would value exactly at that close, itself an event.
DAILY_KI_DAY = datetime(2026, 9, 10)
DAILY_KI_HORIZONS = (timedelta(hours=6), timedelta(hours=1), timedelta(minutes=15), timedelta(minutes=5),
                     timedelta(minutes=1), timedelta(seconds=10), timedelta(seconds=1))
BARRIERS = {"snowball_discrete_ki": ("ko", "ki"), "digital": ("strike",), "barrier_uo_zero_carry": ("ko",),
            "one_touch_zero_carry": ("ko",), "snowball_long_gap": ("ko", "ki"), "snowball_daily_ki": ("ko", "ki")}
#: Capability-matrix monitoring column each product falls in.
MONITORING = {"snowball_discrete_ki": "discrete", "digital": "terminal",
              "barrier_uo_zero_carry": "continuous", "one_touch_zero_carry": "continuous",
              "snowball_long_gap": "discrete", "snowball_daily_ki": "discrete"}


@dataclass(frozen=True)
class Cell:
    product: str
    engine: str
    profile: str
    horizon: timedelta
    offset: str
    barrier: str

    @property
    def id(self) -> str:
        return f"{self.product}-{self.engine}-{self.profile}-{int(self.horizon.total_seconds())}s-{self.offset}-{self.barrier}"



@lru_cache(maxsize=None)
def sse():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                  sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))))


@lru_cache(maxsize=None)
def profile(name: str) -> VarianceProfile:
    if name == "uniform":
        return VarianceProfile.uniform(sse(), 244, reference_date=date(2026, 9, 15))
    if name == "desk":
        return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
    if name == "sessions_only":
        return VarianceProfile.sessions_only(sse(), 244)
    raise ValueError(name)


def product(name: str):
    from intraday.conftest import dated_snowball, digital
    if name in ("snowball_discrete_ki", "snowball_long_gap"):
        prod = dated_snowball(sse().calendar, T0)
        if name == "snowball_long_gap":
            # A long first observation period followed by monthly observations.
            # Same future cashflows as the sixth-fixing fixture, but no earlier
            # observations/history to cross when moving valuation >29 days back.
            # This is an explicit contract, never a future fixing passed as history.
            for schedule in (prod.barrier_config.ko_observation_schedule, prod.barrier_config.ki_observation_schedule):
                schedule.records = schedule.records[5:]
        return prod
    if name == "snowball_daily_ki":
        # The monthly fixture's terms with KI 75 observed at EVERY SSE close after the trade date: the contract
        # desks book. Its remaining events differ from the monthly fixture's, so it has its own study.
        prod = dated_snowball(sse().calendar, T0)
        cal, maturity = sse().calendar, prod.exercise_date
        closes = [d for d in (T0 + timedelta(days=k) for k in range(1, (maturity - T0).days + 1)) if cal.is_business_day(d)]
        prod.barrier_config.ki_observation_schedule.records = [ObservationRecord(observation_date=d, barrier=75.0)
                                                               for d in closes]
        return prod
    if name == "digital":
        return digital(SHORT_EXPIRY)
    if name == "barrier_uo_zero_carry":
        return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=103.0, barrier_type=BarrierType.UP_OUT,
                             exercise_date=SHORT_EXPIRY, observation_type=ObservationType.CONTINUOUS)
    if name == "one_touch_zero_carry":
        return OneTouchOption(barrier=103.0, barrier_direction=BarrierDirection.UP, exercise_date=SHORT_EXPIRY, rebate=1.0,
                              payment_at_hit=True, touch_type=TouchType.ONE_TOUCH, observation_type=ObservationType.CONTINUOUS)
    raise ValueError(name)


def notional(name: str) -> float:
    """What the price budget scales with: initial notional (snowball, barrier) or the cash amount (digital, touch)."""
    return {"snowball_discrete_ki": 100.0, "snowball_long_gap": 100.0, "snowball_daily_ki": 100.0, "digital": 1.0,
            "barrier_uo_zero_carry": 100.0, "one_touch_zero_carry": 1.0}[name]


def market(name: str):
    """(r, q): zero carry for the barrier family (its reference is the variance-time closed form)."""
    return (0.0, 0.0) if name in ("barrier_uo_zero_carry", "one_touch_zero_carry") else (0.03, 0.01)


@lru_cache(maxsize=None)
def fixing_and_history(name: str):
    """(fixing instant, confirmed fixings) of the cell's product: the snowball's sixth KO fixing, else the expiry close."""
    from intraday.conftest import flat_env
    prod = product(name)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse(), variance_profile=profile("desk")))
    if name == "snowball_daily_ki":
        fixing = next(e.timestamp for e in probe.timeline.events
                      if e.kind is EventKind.KI and e.timestamp.date() == DAILY_KI_DAY.date())
        history = sorted({e.timestamp for e in probe.timeline.events if e.timestamp < fixing})
        return fixing, tuple(Fixing(t, 100.0) for t in history)
    if name in ("snowball_discrete_ki", "snowball_long_gap"):
        kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
        if name == "snowball_long_gap":
            return kos[0].timestamp, ()
        return kos[5].timestamp, tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    return probe.timeline.terminal().timestamp, ()


def barrier_level(cell: Cell) -> float:
    return {"ko": 103.0, "ki": 75.0, "strike": 100.0}[cell.barrier]


def build_context(cell: Cell):
    """(resolved context, sqrt of the variance to the first event) of one cell, at its spot offset from its barrier."""
    from intraday.conftest import flat_env
    fixing_ts, fixings = fixing_and_history(cell.product)
    ts = fixing_ts - cell.horizon
    prod, prof, (r, q) = product(cell.product), profile(cell.profile), market(cell.product)
    barrier = barrier_level(cell)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=barrier, r=r, q=q),
                                                     session_calendar=sse(), variance_profile=prof, fixings=fixings))
    tau = probe.numerical.maturity_tau if not cell.product.startswith("snowball_") else min(
        t for t in probe.numerical.event_taus.values() if t > 0.0)
    sw = sqrt(max(float(probe.pricing_env.vol_surface.total_variance(100.0, tau, barrier)), 0.0))
    kind, _, amount = cell.offset.partition("+") if "+" in cell.offset else cell.offset.partition("-")
    sign = 1.0 if "+" in cell.offset else -1.0
    if cell.offset == "eq":
        spot = barrier
    elif kind == "bp":
        spot = barrier * (1.0 + sign * float(amount) * 1e-4)
    else:
        spot = barrier * exp(sign * float(amount) * sw)
    ctx = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=spot, r=r, q=q),
                                                   session_calendar=sse(), variance_profile=prof, fixings=fixings))
    return ctx, sw
