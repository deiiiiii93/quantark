"""A dated equity option valued ON its exercise date prices its settled value (user ruling 2026-10-09, option (a)).

The standing rule: a numeric contract shows no difference from its dated twin. The numeric twin aged to exactly its
maturity (``maturity == 0.0``) already prices the settled value -- the contract payoff at the valuation spot with every
past fixing or event as recorded, and a fixing dated the valuation instant live (t = 0, fixed at the spot; ruling D2).
Until this change the dated contract refused the same instant (``Valuation date ... must be before exercise date``).

Every family below is written twice, on dates (``exercise_date`` == the valuation date, fixings and observations on
``INITIAL + round(k / 12 * 365)`` days) and numerically (``maturity`` 0.0, the same fixings at ``k / 12 - 1`` years), and
priced by every engine that routes it. The two must agree to rel 1e-12 and equal the hand settled value. A valuation
strictly AFTER the exercise date still raises.

``FAMILIES`` is also the evidence script's inventory (docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import (
    AccumulatorAnalyticalEngine,
    AmericanOptionAnalyticalEngine,
    AsianOptionAnalyticalEngine,
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    DigitalOptionAnalyticalEngine,
    DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine,
    OneTouchAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.equity.engine.mc import (
    AccumulatorMCEngine,
    HestonBarrierMCEngine,
    HestonMCEngine,
    HestonPhoenixMCEngine,
    HestonSLVBarrierMCEngine,
    HestonSLVMCEngine,
    HestonSnowballMCEngine,
    LocalVolBarrierMCEngine,
    LocalVolMCEngine,
    LocalVolPhoenixMCEngine,
    LocalVolSnowballMCEngine,
    AmericanOptionMCEngine,
    AsianOptionMCEngine,
    BarrierOptionMCEngine,
    DigitalOptionMCEngine,
    DoubleSharkfinOptionMCEngine,
    EuropeanMCEngine,
    PhoenixMCEngine,
    RangeAccrualMCEngine,
    SingleSharkfinOptionMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonBarrierPDESolver,
    HestonPDESolver,
    HestonPhoenixPDESolver,
    HestonSLVBarrierPDESolver,
    HestonSLVPDESolver,
    HestonSnowballPDESolver,
    LocalVolBarrierPDESolver,
    LocalVolPDESolver,
    LocalVolPhoenixPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.american_pde_solver import AmericanPDESolver
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.engine.pde.double_barrier_pde_solver import DoubleBarrierPDESolver
from quantark.asset.equity.engine.pde.double_one_touch_pde_solver import DoubleOneTouchPDESolver
from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.one_touch_pde_solver import OneTouchPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad import (
    BarrierQuadEngine,
    EuropeanQuadEngine,
    KOResetSnowballQuadEngine,
    KOResetSnowballQuadEngineV2,
    OneTouchQuadEngine,
    PhoenixQuadEngine,
    PhoenixQuadEngineV2,
    SnowballQuadEngine,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    AmericanOption,
    BarrierOption,
    CashOrNothingDigitalOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    DoubleSharkfinOption,
    EuropeanVanillaOption,
    OneTouchOption,
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    AsianStrikeType,
    AveragingType,
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.enum.option_enums import ObservationAggregation, PostKOScheduleMode
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

#: the valuation instant == the exercise date; the contract started a year (365 days) earlier
VALUATION = datetime(2026, 1, 5)
INITIAL = datetime(2025, 1, 5)
#: twelve monthly fixings: the contract's ``k / 12`` years on the snapped day ``round(k / 12 * 365)``; the last is today
K12 = range(1, 13)


def day(k: int) -> datetime:
    return INITIAL + timedelta(days=round(k / 12 * 365))


def numeric_time(k: int) -> float:
    """The fixing's time from the valuation instant: ``k / 12 - 1`` (exactly 0.0 for k = 12)."""
    return k / 12.0 - 1.0


def env(spot: float, valuation: datetime = VALUATION, grid: bool = False) -> PricingEnvironment:
    """Flat 20% vol, 3% rate, 1% yield; ``grid`` gives the local-vol routes the same flat vol as a market IV grid."""
    vol = (GridVolSurface(list(100.0 * np.exp(np.linspace(-1.5, 1.5, 13))), [0.25, 0.5, 1.0, 1.5, 2.0],
                          np.full((5, 13), 0.20)) if grid else FlatVolSurface(volatility=0.20))
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=0.03),
                              div_yield=ContinuousDividendYield(0.01), valuation_date=valuation)


def env_for(engine_name: str, spot: float, valuation: datetime = VALUATION) -> PricingEnvironment:
    """The local-vol routes and the SLV MC European (its Dupire leg) read the market IV grid."""
    return env(spot, valuation, grid=engine_name.startswith(("lv_", "slv_mc")))


def _age(product, years: float = 1.0):
    """The numeric contract aged to exactly its maturity, keeping the record on the landing instant (ruling D2)."""
    product.time_shift(years, VALUATION, env(100.0), keep_landing=True)
    return product


@dataclass(frozen=True)
class Family:
    """One product family: its two forms, the hand settled value and the engines that route it."""

    name: str
    build: Callable[[str], object]                      # form ("dated" | "numeric") -> product
    settled: Callable[[float], float]                   # spot -> the contract's settled value
    spots: Tuple[float, ...]
    engines: Tuple[Tuple[str, Callable[[], object]], ...]


def _mc():
    return MCParams(seed=7, num_paths=256)


#: a Heston model for the vol-model routes (the settled value cannot depend on it)
HESTON = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
_ADI = dict(n_x=64, n_v=16, n_t=16)
#: a flat (unit) leverage for the SLV routes that consume a calibrated one
SLV_LEVERAGE = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 6), strike_grid=np.array([40.0, 100.0, 250.0]),
                               leverage_grid=np.ones((6, 3)))


# -- vanillas ------------------------------------------------------------------------------------------------------

def _vanilla(cls, option_type, **kw):
    def build(form):
        if form == "dated":
            return cls(strike=100.0, option_type=option_type, exercise_date=VALUATION, initial_date=INITIAL, **kw)
        return _age(cls(strike=100.0, option_type=option_type, maturity=1.0, **kw))
    return build


EUROPEAN = Family(
    "european_call", _vanilla(EuropeanVanillaOption, OptionType.CALL), lambda s: max(s - 100.0, 0.0), (110.0, 90.0),
    (("analytical_bs", BlackScholesEngine), ("mc", lambda: EuropeanMCEngine(params=_mc())),
     ("pde", lambda: EuropeanPDESolver(PDEParams())), ("quad", lambda: EuropeanQuadEngine()),
     ("heston_analytical", lambda: HestonAnalyticalEngine(HESTON)), ("lv_mc", lambda: LocalVolMCEngine(params=_mc())),
     ("lv_pde", lambda: LocalVolPDESolver()), ("heston_mc", lambda: HestonMCEngine(HESTON, params=_mc())),
     ("heston_pde", lambda: HestonPDESolver(HESTON, **_ADI)),
     ("slv_mc", lambda: HestonSLVMCEngine(HESTON, params=_mc())),
     ("slv_pde", lambda: HestonSLVPDESolver(HESTON, SLV_LEVERAGE, **_ADI))))

AMERICAN = Family(
    "american_put", _vanilla(AmericanOption, OptionType.PUT), lambda s: max(100.0 - s, 0.0), (90.0, 110.0),
    (("analytical", AmericanOptionAnalyticalEngine), ("mc", lambda: AmericanOptionMCEngine(params=_mc())),
     ("pde", lambda: AmericanPDESolver(PDEParams()))))

DIGITAL = Family(
    "digital_call", _vanilla(CashOrNothingDigitalOption, OptionType.CALL, payout=10.0),
    lambda s: 10.0 if s > 100.0 else 0.0, (110.0, 90.0),
    (("analytical", DigitalOptionAnalyticalEngine), ("mc", lambda: DigitalOptionMCEngine(params=_mc()))))


# -- Asian: eleven recorded fixings 101..111, the twelfth (today) live at the spot ----------------------------------

def _asian(form):
    if form == "dated":
        records = [AsianObservationRecord(observation_date=day(k), observed_price=100.0 + k if k < 12 else None)
                   for k in K12]
        return AsianOption(strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
                           averaging_type=AveragingType.ARITHMETIC, exercise_date=VALUATION, initial_date=INITIAL,
                           observation_records=records, initial_price=100.0)
    records = [AsianObservationRecord(observation_time=numeric_time(k), observed_price=100.0 + k if k < 12 else None)
               for k in K12]
    product = AsianOption(strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
                          averaging_type=AveragingType.ARITHMETIC, maturity=1.0, observation_records=records,
                          initial_price=100.0)
    product.maturity = 0.0
    return product


ASIAN = Family(
    "asian_fixed_call", _asian, lambda s: max((sum(100.0 + k for k in range(1, 12)) + s) / 12.0 - 100.0, 0.0),
    (110.0, 90.0),
    (("analytical", AsianOptionAnalyticalEngine), ("mc", lambda: AsianOptionMCEngine(params=_mc()))))


# -- barrier family (continuous monitoring) ------------------------------------------------------------------------

def _barrier(form):
    kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT, rebate=5.0,
              observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return BarrierOption(exercise_date=VALUATION, initial_date=INITIAL, **kw)
    return _age(BarrierOption(maturity=1.0, **kw))


BARRIER = Family(
    "barrier_up_out_call", _barrier, lambda s: 5.0 if s >= 130.0 else max(s - 100.0, 0.0), (110.0, 90.0, 135.0),
    (("analytical", BarrierAnalyticalEngine), ("mc", lambda: BarrierOptionMCEngine(params=_mc())),
     ("pde", lambda: BarrierPDESolver(PDEParams())), ("quad", lambda: BarrierQuadEngine()),
     ("lv_mc", lambda: LocalVolBarrierMCEngine(params=_mc())), ("lv_pde", lambda: LocalVolBarrierPDESolver()),
     ("heston_mc", lambda: HestonBarrierMCEngine(HESTON, params=_mc())),
     ("heston_pde", lambda: HestonBarrierPDESolver(HESTON, **_ADI)),
     ("slv_mc", lambda: HestonSLVBarrierMCEngine(HESTON, SLV_LEVERAGE, params=_mc())),
     ("slv_pde", lambda: HestonSLVBarrierPDESolver(HESTON, SLV_LEVERAGE, **_ADI))))


def _double_barrier(form):
    kw = dict(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
              barrier_type=DoubleBarrierType.KNOCK_OUT, rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return DoubleBarrierOption(exercise_date=VALUATION, initial_date=INITIAL, **kw)
    return _age(DoubleBarrierOption(maturity=1.0, **kw))


DOUBLE_BARRIER = Family(
    "double_barrier_out_call", _double_barrier,
    lambda s: 5.0 if (s >= 130.0 or s <= 80.0) else max(s - 100.0, 0.0), (110.0, 90.0, 135.0),
    (("analytical", DoubleBarrierOptionAnalyticalEngine), ("pde", lambda: DoubleBarrierPDESolver(PDEParams()))))


def _one_touch(form):
    kw = dict(barrier=130.0, barrier_direction=BarrierDirection.UP, rebate=10.0, touch_type=TouchType.ONE_TOUCH,
              observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return OneTouchOption(exercise_date=VALUATION, **kw)
    return _age(OneTouchOption(maturity=1.0, **kw))


ONE_TOUCH = Family(
    "one_touch_up", _one_touch, lambda s: 10.0 if s >= 130.0 else 0.0, (110.0, 135.0),
    (("analytical", OneTouchAnalyticalEngine), ("pde", lambda: OneTouchPDESolver(PDEParams())),
     ("quad", lambda: OneTouchQuadEngine())))


def _double_one_touch(form):
    kw = dict(upper_barrier=130.0, lower_barrier=80.0, rebate=10.0, touch_type=TouchType.DOUBLE_ONE_TOUCH,
              observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return DoubleOneTouchOption(exercise_date=VALUATION, **kw)
    return _age(DoubleOneTouchOption(maturity=1.0, **kw))


DOUBLE_ONE_TOUCH = Family(
    "double_one_touch", _double_one_touch, lambda s: 10.0 if (s >= 130.0 or s <= 80.0) else 0.0, (110.0, 135.0),
    (("pde", lambda: DoubleOneTouchPDESolver(PDEParams())),))


def _sharkfin(form):
    kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=130.0, participation_rate=1.0,
              knock_out_rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return SingleSharkfinOption(exercise_date=VALUATION, initial_date=INITIAL, **kw)
    return _age(SingleSharkfinOption(maturity=1.0, **kw))


SHARKFIN = Family(
    "single_sharkfin_call", _sharkfin, lambda s: 5.0 if s >= 130.0 else max(s - 100.0, 0.0), (110.0, 90.0, 135.0),
    (("analytical", SingleSharkfinOptionAnalyticalEngine),
     ("mc", lambda: SingleSharkfinOptionMCEngine(params=_mc()))))


def _double_sharkfin(form):
    kw = dict(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
              participation_rate=1.0, knock_out_rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    if form == "dated":
        return DoubleSharkfinOption(exercise_date=VALUATION, initial_date=INITIAL, **kw)
    return _age(DoubleSharkfinOption(maturity=1.0, **kw))


DOUBLE_SHARKFIN = Family(
    "double_sharkfin_call", _double_sharkfin,
    lambda s: 5.0 if (s >= 130.0 or s <= 80.0) else max(s - 100.0, 0.0), (110.0, 90.0, 135.0),
    (("analytical", DoubleSharkfinOptionAnalyticalEngine),
     ("mc", lambda: DoubleSharkfinOptionMCEngine(params=_mc()))))


# -- range accrual: notional 100 at 10% over the one-year tenor; eleven recorded in range, today live --------------

def _range_accrual(form):
    config = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10)
    if form == "dated":
        records = [RangeAccrualObservationRecord(observation_date=day(k), observed_in_range=True if k < 12 else None)
                   for k in K12]
        return RangeAccrualOption(initial_price=100.0, range_config=config, exercise_date=VALUATION,
                                  initial_date=INITIAL, observation_records=records)
    records = [RangeAccrualObservationRecord(observation_time=numeric_time(k),
                                             observed_in_range=True if k < 12 else None) for k in K12]
    product = RangeAccrualOption(initial_price=100.0, range_config=config, maturity=1.0, tenor=1.0,
                                 observation_records=records)
    product.maturity = 0.0
    return product


RANGE_ACCRUAL = Family(
    "range_accrual", _range_accrual, lambda s: 10.0 * (12.0 if 90.0 <= s <= 110.0 else 11.0) / 12.0, (100.0, 120.0),
    (("analytical", RangeAccrualAnalyticalEngine), ("mc", lambda: RangeAccrualMCEngine(params=_mc()))))


# -- accumulator: one remaining observation, today ----------------------------------------------------------------

def _accumulator(form):
    kw = dict(strike=95.0, knock_out_barrier=108.0, option_type=OptionType.CALL, daily_share_accumulation=1.0,
              gearing=2.0, knock_out_type=AccumulatorKnockOutType.TERMINATION, knock_out_rebate_rate=0.02,
              notional=100.0, extra_shares_at_expiry=3.0, observation_type=ObservationType.DISCRETE)
    if form == "dated":
        schedule = ObservationSchedule(records=[ObservationRecord(observation_date=VALUATION, barrier=108.0)],
                                       aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
        return AccumulatorOption(exercise_date=VALUATION, initial_date=INITIAL, observation_schedule=schedule, **kw)
    return _age(AccumulatorOption(maturity=1.0, observation_dates=[1.0], **kw))


def _accumulator_settled(s):
    """Today's observation decided on the spot: at or above the 108 knock-out the TERMINATION rebate (2% of 100)
    and nothing else; otherwise one share's gain, or two shares' loss below 95, plus 3 extra shares' put at 95."""
    if s >= 108.0:
        return 2.0
    return (s - 95.0 if s >= 95.0 else 2.0 * (s - 95.0)) - 3.0 * max(95.0 - s, 0.0)


ACCUMULATOR = Family(
    "accumulator", _accumulator, _accumulator_settled, (100.0, 90.0, 110.0),
    (("analytical", AccumulatorAnalyticalEngine), ("mc", lambda: AccumulatorMCEngine(params=_mc()))))


# -- autocallables: the tracker-clock parity contracts (flat cash, so the two forms are the same contract) ----------

def _autocallable(kind, knocked_in=False):
    from test_tracker_clock_parity import KINDS, phoenix
    import pandas as pd

    builder = phoenix(annualized=False, memory=False) if kind == "phoenix_plain" else KINDS[kind]

    def build(form):
        product = builder(form, pd.Timestamp(INITIAL))
        if knocked_in:      # a knock-in carried into the valuation (schedule 1, contract day 152)
            product._otc_lifecycle_knocked_in = True
            product._otc_lifecycle_knock_in_date = INITIAL + timedelta(days=152)
        return product if form == "dated" else _age(product, 1.0)
    return build


def _snowball_settled(s):
    """KO 103 pays the flat 15% coupon; no KI (continuous 75) pays the 15% dividend coupon; KI pays s - 100."""
    return -30.0 if s <= 75.0 else 15.0


def _phoenix_settled(s):
    """KO 103 / coupon barrier 80 pays the 1% coupon (memory carries nothing: past coupons treated as paid); below
    the continuous KI 70 the principal loss s - 100."""
    return (s - 100.0) if s <= 70.0 else (1.0 if s >= 80.0 else 0.0)


_AUTOCALL_ENGINES = {
    "snowball": (("mc", lambda: SnowballMCEngine(params=_mc())), ("pde", lambda: SnowballPDESolver(PDEParams())),
                 ("quad_v1", lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001))),
                 ("quad_v2", lambda: SnowballQuadEngineV2()),
                 ("lv_mc", lambda: LocalVolSnowballMCEngine(params=_mc())), ("lv_pde", LocalVolSnowballPDESolver),
                 ("heston_mc", lambda: HestonSnowballMCEngine(HESTON, params=_mc())),
                 ("heston_pde", lambda: HestonSnowballPDESolver(HESTON, **_ADI))),
    "phoenix": (("mc", lambda: PhoenixMCEngine(params=_mc())), ("pde", lambda: PhoenixPDESolver(PDEParams())),
                ("quad_v1", lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001))),
                ("quad_v2", lambda: PhoenixQuadEngineV2()),
                ("lv_mc", lambda: LocalVolPhoenixMCEngine(params=_mc())), ("lv_pde", LocalVolPhoenixPDESolver),
                ("heston_mc", lambda: HestonPhoenixMCEngine(HESTON, params=_mc())),
                ("heston_pde", lambda: HestonPhoenixPDESolver(HESTON, **_ADI))),
    "ko_reset_absolute": (("mc", lambda: SnowballMCEngine(params=_mc())),
                          ("pde", lambda: KOResetSnowballPDESolver(PDEParams())),
                          ("quad_v1", lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))),
                          ("quad_v2", lambda: KOResetSnowballQuadEngineV2())),
}

SNOWBALL = Family("snowball", _autocallable("snowball"), _snowball_settled, (104.0, 90.0, 70.0),
                  _AUTOCALL_ENGINES["snowball"])
PHOENIX = Family("phoenix", _autocallable("phoenix"), _phoenix_settled, (104.0, 90.0, 75.0, 60.0),
                 _AUTOCALL_ENGINES["phoenix"])

PHOENIX_PLAIN = Family("phoenix_no_memory", _autocallable("phoenix_plain"), _phoenix_settled, (104.0, 90.0, 75.0, 60.0),
                       _AUTOCALL_ENGINES["phoenix"])


def _ko_reset_settled(s):
    """Knocked in on schedule 1, so schedule 2 (KO 95, flat 3%) runs to contract day 365: its last record knocks out
    at 95 and pays 3; below it the knocked-in terminal pays s - 100 (no principal)."""
    return 3.0 if s >= 95.0 else s - 100.0


KO_RESET = Family("ko_reset_absolute_knocked_in", _autocallable("ko_reset_absolute", knocked_in=True),
                  _ko_reset_settled, (104.0, 90.0), _AUTOCALL_ENGINES["ko_reset_absolute"])

FAMILIES: Tuple[Family, ...] = (
    EUROPEAN, AMERICAN, DIGITAL, ASIAN, BARRIER, DOUBLE_BARRIER, ONE_TOUCH, DOUBLE_ONE_TOUCH, SHARKFIN,
    DOUBLE_SHARKFIN, RANGE_ACCRUAL, ACCUMULATOR, SNOWBALL, PHOENIX, PHOENIX_PLAIN, KO_RESET,
)


def price(family: Family, name: str, engine_factory, form: str, spot: float,
          valuation: datetime = VALUATION) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(engine_factory().price(family.build(form), env_for(name, spot, valuation)))


#: Routes that price NEITHER form at maturity 0.0, kept refusing on both: (family, engine) -> why. The vol-model
#: European and barrier routes left this table in W9 (user ruling 2026-10-10: they settle at T = 0).
REFUSING: Dict[Tuple[str, str], str] = {
    ("phoenix", "heston_pde"): "HestonPhoenixPDESolver refuses memory coupons at any maturity (pre-existing)",
}

#: Routes where the numeric twin settles but the DATED contract is refused at every valuation date for a reason that
#: is not the exercise date (pre-existing, unchanged): (family, engine) -> why
DATED_REFUSED_ANY_DATE: Dict[Tuple[str, str], str] = {
    ("ko_reset_absolute_knocked_in", "quad_v2"): "QUAD V2 refuses a dated KO-reset whose schedule-1 records are in "
                                                 "the past (settlement resolution of a past record), at any date",
}

CASES = [(f, name, factory, spot) for f in FAMILIES for name, factory in f.engines for spot in f.spots]
SETTLING = [c for c in CASES if (c[0].name, c[1]) not in {**REFUSING, **DATED_REFUSED_ANY_DATE}]
REFUSED = [c for c in CASES if (c[0].name, c[1]) in REFUSING]
DATED_REFUSED = [c for c in CASES if (c[0].name, c[1]) in DATED_REFUSED_ANY_DATE]


def _ids(cases):
    return [f"{f.name}-{name}-{spot:g}" for f, name, _, spot in cases]


@pytest.mark.parametrize("family,name,factory,spot", SETTLING, ids=_ids(SETTLING))
def test_dated_contract_on_its_exercise_date_prices_its_numeric_twins_settled_value(family, name, factory, spot):
    dated = price(family, name, factory, "dated", spot)
    numeric = price(family, name, factory, "numeric", spot)
    assert dated == pytest.approx(numeric, rel=1e-12, abs=1e-12)
    assert numeric == pytest.approx(family.settled(spot), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("family,name,factory,spot", REFUSED, ids=_ids(REFUSED))
def test_a_route_without_a_settled_value_refuses_both_forms(family, name, factory, spot):
    for form in ("dated", "numeric"):
        with pytest.raises(Exception):
            price(family, name, factory, form, spot)


@pytest.mark.parametrize("family,name,factory,spot", DATED_REFUSED, ids=_ids(DATED_REFUSED))
def test_numeric_twin_settles_where_the_dated_contract_is_refused_at_any_date(family, name, factory, spot):
    """QUAD V2 prices the numeric KO-reset at maturity 0.0 (ruling Q1, 2026-10-10); the dated contract keeps its
    pre-existing refusal, which is not caused by the exercise date (it refuses mid-life too)."""
    assert price(family, name, factory, "numeric", spot) == pytest.approx(family.settled(spot), rel=1e-12, abs=1e-12)
    with pytest.raises(ValidationError, match="end_date .* must be after start_date"):
        price(family, name, factory, "dated", spot)
    with pytest.raises(ValidationError, match="end_date .* must be after start_date"):
        price(family, name, factory, "dated", spot, valuation=INITIAL + timedelta(days=292))


@pytest.mark.parametrize("family,engine", [
    (SNOWBALL, SnowballQuadEngineV2), (PHOENIX, PhoenixQuadEngineV2), (KO_RESET, KOResetSnowballQuadEngineV2),
], ids=["snowball", "phoenix", "ko_reset"])
def test_quad_v2_refuses_a_numeric_contract_aged_past_its_maturity(family, engine):
    """Maturity 0.0 settles on QUAD V2; a strictly negative maturity is still refused, and pricing does not
    touch the caller's product."""
    product = family.build("numeric")
    product.maturity = -1e-6
    with pytest.raises(ValidationError, match="Maturity must be positive"):
        engine().price(product, env(100.0))
    assert product.maturity == -1e-6


@pytest.mark.parametrize("cls_name", ["SnowballOption", "PhoenixOption"])
def test_constructing_a_contract_with_maturity_zero_is_still_refused(cls_name):
    """Only re-validating an existing contract aged to its maturity instant accepts 0.0 (QUAD V2's compile)."""
    product = (SNOWBALL if cls_name == "SnowballOption" else PHOENIX).build("numeric")
    assert product.maturity == 0.0
    with pytest.raises(ValidationError, match="Maturity must be positive"):
        product.validate()
    product.validate(allow_zero_maturity=True)


AFTER = VALUATION + timedelta(days=1)


@pytest.mark.parametrize("family,name,factory", [
    (EUROPEAN, "analytical_bs", BlackScholesEngine),
    (BARRIER, "pde", lambda: BarrierPDESolver(PDEParams())),
    (ONE_TOUCH, "analytical", OneTouchAnalyticalEngine),
    (DOUBLE_ONE_TOUCH, "pde", lambda: DoubleOneTouchPDESolver(PDEParams())),
    (RANGE_ACCRUAL, "analytical", RangeAccrualAnalyticalEngine),
    (SNOWBALL, "pde", lambda: SnowballPDESolver(PDEParams())),
], ids=["european", "barrier", "one_touch", "double_one_touch", "range_accrual", "snowball"])
def test_a_valuation_strictly_after_the_exercise_date_still_raises(family, name, factory):
    """Refused as before (the terminal settlement timing or ``get_maturity``, whichever the route reads first)."""
    with pytest.raises(ValidationError):
        price(family, name, factory, "dated", 110.0, valuation=AFTER)


def test_time_to_exercise_is_zero_on_the_instant_and_positive_before():
    from quantark.asset.equity.product.option.base_equity_option import time_to_exercise

    assert time_to_exercise(VALUATION, env(100.0)) == 0.0
    assert time_to_exercise(VALUATION, env(100.0, VALUATION - timedelta(days=73))) == pytest.approx(0.2, rel=1e-15)
    with pytest.raises(ValidationError, match="must be on or before exercise date"):
        time_to_exercise(VALUATION, env(100.0, AFTER))


def test_an_instant_later_on_the_exercise_date_is_after_it():
    """The rule is the same INSTANT: a valuation timestamp later on the exercise day is after the exercise."""
    with pytest.raises(ValidationError):
        price(EUROPEAN, "analytical_bs", BlackScholesEngine, "dated", 110.0,
              valuation=VALUATION + timedelta(hours=15))


# -- W9: the vol-model routes settle on the exercise date (user ruling 2026-10-10) -----------------------------------
# At T = 0 the vol model has no effect on the price: each vol-model European / barrier route short-circuits, before
# its model runs, to the expiry code of its Black-Scholes counterpart of the same method (analytical / MC / PDE). Each
# route is checked against that counterpart in both forms, on the products and spots below: in / at / beyond the
# money and the barrier, a knock-in fresh and carried as triggered, expiry monitoring and a two-day settlement lag.

_LAG2D = SettlementConvention(lag=2 / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)


def _barrier_variant(barrier_type, option_type, barrier, observation=ObservationType.CONTINUOUS, carried=False,
                     **extra):
    def build(form):
        kw = dict(strike=100.0, option_type=option_type, barrier=barrier, barrier_type=barrier_type, rebate=5.0,
                  observation_type=observation, **extra)
        product = (BarrierOption(exercise_date=VALUATION, initial_date=INITIAL, **kw) if form == "dated"
                   else _age(BarrierOption(maturity=1.0, **kw)))
        if carried:     # a knock-in carried into the valuation as triggered (spec 2026-10-07 R4)
            product._otc_lifecycle_knocked_in = True
        return product
    return build


VOL_EUROPEAN_PRODUCTS = (
    ("call", _vanilla(EuropeanVanillaOption, OptionType.CALL), (110.0, 90.0, 100.0)),
    ("put", _vanilla(EuropeanVanillaOption, OptionType.PUT), (90.0, 110.0)),
    ("call_lag2d", _vanilla(EuropeanVanillaOption, OptionType.CALL, settlement_convention=_LAG2D), (110.0,)),
)

VOL_BARRIER_PRODUCTS = (
    ("up_out_call", _barrier_variant(BarrierType.UP_OUT, OptionType.CALL, 130.0), (110.0, 90.0, 130.0, 135.0)),
    ("up_in_call", _barrier_variant(BarrierType.UP_IN, OptionType.CALL, 130.0), (110.0, 130.0, 135.0)),
    ("up_in_call_knocked_in", _barrier_variant(BarrierType.UP_IN, OptionType.CALL, 130.0, carried=True),
     (110.0, 135.0)),
    ("down_out_put", _barrier_variant(BarrierType.DOWN_OUT, OptionType.PUT, 80.0), (90.0, 110.0, 80.0, 75.0)),
    ("up_out_call_expiry", _barrier_variant(BarrierType.UP_OUT, OptionType.CALL, 130.0, ObservationType.EXPIRY),
     (110.0, 135.0)),
    ("up_out_call_lag2d", _barrier_variant(BarrierType.UP_OUT, OptionType.CALL, 130.0, settlement_convention=_LAG2D),
     (110.0, 135.0)),
)

#: (route, the vol-model engine, its Black-Scholes counterpart, the products it routes)
VOL_ROUTES = (
    ("heston_analytical", lambda: HestonAnalyticalEngine(HESTON), BlackScholesEngine, VOL_EUROPEAN_PRODUCTS),
    ("heston_mc", lambda: HestonMCEngine(HESTON, params=_mc()), lambda: EuropeanMCEngine(params=_mc()),
     VOL_EUROPEAN_PRODUCTS),
    ("lv_mc", lambda: LocalVolMCEngine(params=_mc()), lambda: EuropeanMCEngine(params=_mc()), VOL_EUROPEAN_PRODUCTS),
    ("slv_mc", lambda: HestonSLVMCEngine(HESTON, params=_mc()), lambda: EuropeanMCEngine(params=_mc()),
     VOL_EUROPEAN_PRODUCTS),
    ("heston_pde", lambda: HestonPDESolver(HESTON, **_ADI), lambda: EuropeanPDESolver(PDEParams()),
     VOL_EUROPEAN_PRODUCTS),
    ("slv_pde", lambda: HestonSLVPDESolver(HESTON, SLV_LEVERAGE, **_ADI), lambda: EuropeanPDESolver(PDEParams()),
     VOL_EUROPEAN_PRODUCTS),
    ("lv_barrier_mc", lambda: LocalVolBarrierMCEngine(params=_mc()), lambda: BarrierOptionMCEngine(params=_mc()),
     VOL_BARRIER_PRODUCTS),
    ("heston_barrier_mc", lambda: HestonBarrierMCEngine(HESTON, params=_mc()),
     lambda: BarrierOptionMCEngine(params=_mc()), VOL_BARRIER_PRODUCTS),
    ("slv_barrier_mc", lambda: HestonSLVBarrierMCEngine(HESTON, SLV_LEVERAGE, params=_mc()),
     lambda: BarrierOptionMCEngine(params=_mc()), VOL_BARRIER_PRODUCTS),
    ("heston_barrier_pde", lambda: HestonBarrierPDESolver(HESTON, **_ADI), lambda: BarrierPDESolver(PDEParams()),
     VOL_BARRIER_PRODUCTS),
    ("slv_barrier_pde", lambda: HestonSLVBarrierPDESolver(HESTON, SLV_LEVERAGE, **_ADI),
     lambda: BarrierPDESolver(PDEParams()), VOL_BARRIER_PRODUCTS),
)

VOL_CASES = [(route, vol, bsm, kind, build, spot) for route, vol, bsm, products in VOL_ROUTES
             for kind, build, spots in products for spot in spots]
_VOL_IDS = [f"{route}-{kind}-{spot:g}" for route, _, _, kind, _, spot in VOL_CASES]


def _vol_env(spot: float, valuation: datetime = VALUATION) -> PricingEnvironment:
    """The flat market IV grid for every engine of this section (the local-vol legs read it; nothing else does)."""
    return env(spot, valuation, grid=True)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


@pytest.mark.parametrize("route,vol,bsm,kind,build,spot", VOL_CASES, ids=_VOL_IDS)
def test_vol_model_route_on_its_exercise_date_prices_its_black_scholes_counterparts_settled_value(
        route, vol, bsm, kind, build, spot):
    """The dated contract on its exercise date == the numeric twin at maturity 0.0 == the Black-Scholes engine of
    the same method on the same contract (which never reads the vol model at T = 0)."""
    dated = _quiet(lambda: float(vol().price(build("dated"), _vol_env(spot))))
    numeric = _quiet(lambda: float(vol().price(build("numeric"), _vol_env(spot))))
    bsm_dated = _quiet(lambda: float(bsm().price(build("dated"), _vol_env(spot))))
    bsm_numeric = _quiet(lambda: float(bsm().price(build("numeric"), _vol_env(spot))))
    assert dated == pytest.approx(numeric, rel=1e-12, abs=1e-12)
    assert dated == pytest.approx(bsm_dated, rel=1e-12, abs=1e-12)
    assert numeric == pytest.approx(bsm_numeric, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("route,vol,bsm,kind,build,spot", VOL_CASES, ids=_VOL_IDS)
def test_vol_model_route_greeks_on_its_exercise_date_are_its_black_scholes_counterparts(
        route, vol, bsm, kind, build, spot):
    """``calculate_greeks`` on the exercise date: the counterpart's keys (price, delta, gamma) and values, in both
    forms."""
    dated = _quiet(lambda: vol().calculate_greeks(build("dated"), _vol_env(spot)))
    numeric = _quiet(lambda: vol().calculate_greeks(build("numeric"), _vol_env(spot)))
    reference = _quiet(lambda: bsm().calculate_greeks(build("dated"), _vol_env(spot)))
    assert set(dated) == set(numeric) == set(reference) == {"price", "delta", "gamma"}
    for key in reference:
        assert float(dated[key]) == pytest.approx(float(numeric[key]), rel=1e-12, abs=1e-12), key
        assert float(dated[key]) == pytest.approx(float(reference[key]), rel=1e-12, abs=1e-12), key


_VOL_FIRST = [(route, vol, products[0][1]) for route, vol, _, products in VOL_ROUTES]


@pytest.mark.parametrize("route,vol,build", _VOL_FIRST, ids=[r for r, _, _ in _VOL_FIRST])
def test_vol_model_route_still_refuses_a_valuation_after_the_exercise(route, vol, build):
    """Strictly after the exercise date, and later on the exercise day than the booked exercise instant (ruling Q4:
    the same instant), the dated contract is refused; so is a numeric contract aged past its maturity."""
    for valuation in (AFTER, VALUATION + timedelta(hours=15)):
        with pytest.raises(ValidationError):
            _quiet(lambda: vol().price(build("dated"), _vol_env(110.0, valuation)))
    product = build("numeric")
    product.maturity = -1e-6
    with pytest.raises(ValidationError):
        _quiet(lambda: vol().price(product, _vol_env(110.0)))
