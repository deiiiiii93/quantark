"""Gate C cell catalogue: time-to-fixing ladder x spot offsets x profiles x (product, engine, barrier)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache

from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.intraday import EventKind, Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import BarrierDirection, BarrierType, ObservationType, OptionType, TouchType

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)
SHORT_EXPIRY = datetime(2026, 9, 16)          # the snowball's sixth fixing day: every product's ladder ends at its 15:00 close

HORIZONS = (timedelta(days=1), timedelta(hours=6), timedelta(hours=1), timedelta(minutes=15), timedelta(minutes=5),
            timedelta(minutes=1), timedelta(seconds=10), timedelta(seconds=1))
SPOT_OFFSETS = ("bp-10", "bp-1", "bp+1", "bp+10", "sd-2", "sd-1", "sd-0.5", "sd+0.5", "sd+1", "sd+2", "eq")
PROFILES = ("uniform", "desk", "sessions_only")
PRODUCTS = ("snowball_discrete_ki", "digital", "barrier_uo_zero_carry", "one_touch_zero_carry")
ENGINES = {
    "snowball_discrete_ki": ("quad_v2", "pde", "mc_rqmc"),
    "digital": ("analytical", "mc_rqmc"),
    "barrier_uo_zero_carry": ("analytical", "pde", "mc_rqmc"),
    "one_touch_zero_carry": ("analytical", "pde"),
}
BARRIERS = {"snowball_discrete_ki": ("ko", "ki"), "digital": ("strike",), "barrier_uo_zero_carry": ("ko",),
            "one_touch_zero_carry": ("ko",)}
#: Capability-matrix monitoring column each catalogued product falls in. A Greek
#: demonstration is scoped to it: a certificate earned on discrete fixings says
#: nothing about the same product under a continuously observed barrier.
MONITORING = {"snowball_discrete_ki": "discrete", "digital": "terminal",
              "barrier_uo_zero_carry": "continuous", "one_touch_zero_carry": "continuous"}
FAST_HORIZON, FAST_OFFSET, FAST_PROFILE = timedelta(hours=1), "sd+1", "desk"


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


def all_cells():
    return [Cell(p, e, prof, h, o, b) for p in PRODUCTS for e in ENGINES[p] for prof in PROFILES for h in HORIZONS
            for o in SPOT_OFFSETS for b in BARRIERS[p]]


def fast_cells():
    return [Cell("snowball_discrete_ki", e, FAST_PROFILE, FAST_HORIZON, FAST_OFFSET, "ko") for e in ENGINES["snowball_discrete_ki"]]


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
    if name == "snowball_discrete_ki":
        return dated_snowball(sse().calendar, T0)
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
    return {"snowball_discrete_ki": 100.0, "digital": 1.0, "barrier_uo_zero_carry": 100.0, "one_touch_zero_carry": 1.0}[name]


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
    if name == "snowball_discrete_ki":
        kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
        return kos[5].timestamp, tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    return probe.timeline.terminal().timestamp, ()


def barrier_level(cell: Cell) -> float:
    return {"ko": 103.0, "ki": 75.0, "strike": 100.0}[cell.barrier]
