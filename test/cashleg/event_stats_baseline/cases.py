"""Cases, legs and record computation of the cash-leg baseline (unified event statistics spec D9).

One module feeds both the generator (``generate.py``, which banks ``baseline.json``) and the
regression test (``test/cashleg/test_event_stats_baseline.py``, which recomputes every record), so
the two can never build different engines, products or legs.

A *case* is one engine on one product. Every case banks

* ``price``           -- ``engine.price(product, env)``;
* ``event_npv``       -- ``engine.price_with_events(product, env).npv`` (spec D14);
* ``event``           -- per leg, ``leg.value(price_with_events(...).event_distribution, env, UNIT)``;
* ``fold_total``      -- per leg, ``folded_trade_value(engine, ..., [leg], mode=TOTAL).total``
                         (``npv + leg PV`` at quantity 1);
* ``fold_drilldown``  -- per leg, ``per_leg[leg_id]`` of ONE ``folded_trade_value(..., mode=DRILLDOWN)``
                         over the case's banked drilldown leg set (the legs whose TOTAL folded at
                         base), plus that call's ``total`` and ``npv`` as case-level records.

A record's ``outcome`` is ``{"value": float}`` or ``{"raises": "<ExceptionClass>"}``. Everything is
built fresh per computation (fresh engine instance, fixed seeds), so a record does not depend on
which other records ran first in the same process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import HestonPhoenixMCEngine, LocalVolPhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import HestonSnowballMCEngine, LocalVolSnowballMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import (
    HestonKOResetSnowballPDESolver,
    LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver, LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection, DCNOption
from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import (
    AccrualBasis,
    AccrualLeg,
    AutocallableCashLeg,
    AutocallableLegType,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    FixedPayoffLeg,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentConvention,
    PaymentTrigger,
    PvFormula,
    SurvivalBasis,
)
from quantark.cashleg.folding import FoldMode, folded_trade_value
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

BASE_SHA = "79938a1c9cc0b1f6fa256ee2d933680b5ccbedde"
#: position notional handed to ``leg.value`` (every leg base here is ABSOLUTE, so it is not read)
UNIT = 1_000_000.0
VALUATION = datetime(2024, 1, 1)
KO_DATES = (0.25, 0.5, 0.75, 1.0)

# --- markets -----------------------------------------------------------------------------------


def flat_env(spot: float = 100.0) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION,
    )


def skewed_env(spot: float = 100.0) -> PricingEnvironment:
    """A sloped grid surface (local-vol engines need a ``GridVolSurface``)."""
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.4, 11)))
    maturities = list(np.linspace(0.25, 2.0, 8))
    grid = np.array([[0.34 - 0.22 * (np.log(k / 100.0) + 0.6) for k in strikes] for _ in maturities])
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=GridVolSurface(strikes, maturities, grid),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION,
    )


DCN_START = datetime(2023, 1, 3)
DCN_SIGMA, DCN_R, DCN_Q = 0.184, 0.0356, 0.1406


def dcn_env(grid: bool = False, spot: float = 6000.0) -> PricingEnvironment:
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = DCN_SIGMA + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        vol = FlatVolSurface(volatility=DCN_SIGMA)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=DCN_R),
        div_yield=ContinuousDividendYield(div_yield=DCN_Q), valuation_date=DCN_START,
        day_count_convention=DayCountConvention.ACT_365,
    )


# --- products ----------------------------------------------------------------------------------


def _snowball(ki: str) -> SnowballOption:
    if ki == "continuous":
        ki_kw = dict(ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(KO_DATES))
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(KO_DATES), **ki_kw),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )


def _phoenix(memory: bool, pay: CouponPayType) -> PhoenixOption:
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(KO_DATES), ki_barrier=70.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * len(KO_DATES), coupon_rate=0.02,
                                          coupon_pay_type=pay, day_count_convention=DayCountConvention.ACT_365,
                                          memory_coupon=memory),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


KO_RESET_PRE = (0.25, 0.5, 0.75, 1.0)
KO_RESET_POST_ABSOLUTE = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
KO_RESET_POST_REBASED = (0.25, 0.5, 0.75, 1.0)          # offsets from the knock-in


def _schedule(times, level) -> ObservationSchedule:
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def _ko_reset(mode: PostKOScheduleMode) -> KnockOutResetSnowballOption:
    """Schedule 1 quarterly to 1y (103, 15%), discrete KI at 80 on schedule 1's dates, schedule 2 (95, 3%)
    quarterly to 2y absolute, or quarterly offsets to 1y after the knock-in (REBASED)."""
    post = KO_RESET_POST_ABSOLUTE if mode is PostKOScheduleMode.ABSOLUTE else KO_RESET_POST_REBASED
    maturity = 2.0
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(KO_RESET_PRE, 103.0), ki_barrier=80.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(KO_RESET_PRE, 80.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=mode,
    )


DCN_CONTRACT = dict(
    initial_date=DCN_START, valuation_date=DCN_START, maturity_date=datetime(2025, 1, 3), tenor_months=24,
    lock_months=3, ko_lock_months=3, coupon_settlement_offset=2, ko_settlement_offset=2,
    settlement_date=datetime(2025, 1, 7),
)


def _dcn(direction: DCNDirection = DCNDirection.BUYER) -> DCNOption:
    """``test/dcn_fixtures.py`` DCN_A (24 monthly observations)."""
    schedule = build_dcn_schedule(calendar=create_calendar(CalendarType.CHINA_SSE), **DCN_CONTRACT)
    return DCNOption(
        notional=1_000_000.0, initial_price=6000.0, coupon_barrier_ratio=0.80, ko_barrier_ratio=1.00,
        ki_barrier_ratio=0.75, ki_put_strike_ratio=1.10, coupon_rate=0.12, ko_coupon_rate=0.12,
        participation=1.0, coupon_counted_days=30, coupon_days_denom=360,
        settlement_date=DCN_CONTRACT["settlement_date"], direction=direction, schedule=schedule,
        knocked_in_at_valuation=False,
    )


def _knocked_in(product):
    """The trade already knocked in before valuation (the OTC lifecycle flag the engines read)."""
    product._otc_lifecycle_knocked_in = True
    return product


@dataclass(frozen=True)
class ProductSpec:
    """A product and the facts a later task needs to tell which D9 cause could move its legs."""

    family: str                      # snowball | phoenix | ko_reset_absolute | ko_reset_rebased | dcn
    build: Callable[[], object]
    features: Dict[str, object]
    life: float                      # last payment horizon in years, for the accrual-leg schedule


PRODUCTS: Dict[str, ProductSpec] = {
    "snowball_cki": ProductSpec("snowball", lambda: _snowball("continuous"),
                                dict(ki="continuous", memory=None, coupon_pay=None, include_principal=True), 1.0),
    "snowball_dki": ProductSpec("snowball", lambda: _snowball("discrete"),
                                dict(ki="discrete", memory=None, coupon_pay=None, include_principal=True), 1.0),
    "phoenix_mem_instant": ProductSpec("phoenix", lambda: _phoenix(True, CouponPayType.INSTANT),
                                       dict(ki="continuous", memory=True, coupon_pay="INSTANT",
                                            include_principal=True), 1.0),
    "phoenix_nomem_instant": ProductSpec("phoenix", lambda: _phoenix(False, CouponPayType.INSTANT),
                                         dict(ki="continuous", memory=False, coupon_pay="INSTANT",
                                              include_principal=True), 1.0),
    "phoenix_mem_expiry": ProductSpec("phoenix", lambda: _phoenix(True, CouponPayType.EXPIRY),
                                      dict(ki="continuous", memory=True, coupon_pay="EXPIRY",
                                           include_principal=True), 1.0),
    "phoenix_nomem_expiry": ProductSpec("phoenix", lambda: _phoenix(False, CouponPayType.EXPIRY),
                                        dict(ki="continuous", memory=False, coupon_pay="EXPIRY",
                                             include_principal=True), 1.0),
    "ko_reset_absolute": ProductSpec("ko_reset_absolute", lambda: _ko_reset(PostKOScheduleMode.ABSOLUTE),
                                     dict(ki="discrete", memory=None, coupon_pay=None, include_principal=True,
                                          post_ko_mode="ABSOLUTE"), 2.0),
    "ko_reset_rebased": ProductSpec("ko_reset_rebased", lambda: _ko_reset(PostKOScheduleMode.REBASED),
                                    dict(ki="discrete", memory=None, coupon_pay=None, include_principal=True,
                                         post_ko_mode="REBASED"), 2.0),
    "dcn": ProductSpec("dcn", _dcn, dict(ki="discrete_daily_close", memory=None, coupon_pay="INSTANT",
                                         include_principal=None, direction="BUYER"), 2.0),
    "dcn_seller": ProductSpec("dcn", lambda: _dcn(DCNDirection.SELLER),
                              dict(ki="discrete_daily_close", memory=None, coupon_pay="INSTANT",
                                   include_principal=None, direction="SELLER"), 2.0),
    "snowball_cki_knocked_in": ProductSpec("snowball", lambda: _knocked_in(_snowball("continuous")),
                                           dict(ki="continuous", memory=None, coupon_pay=None,
                                                include_principal=True, knocked_in_at_valuation=True), 1.0),
    "phoenix_mem_instant_knocked_in": ProductSpec("phoenix",
                                                  lambda: _knocked_in(_phoenix(True, CouponPayType.INSTANT)),
                                                  dict(ki="continuous", memory=True, coupon_pay="INSTANT",
                                                       include_principal=True, knocked_in_at_valuation=True), 1.0),
}

# --- engines -----------------------------------------------------------------------------------

HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
_MC = dict(num_paths=4096, seed=7)
#: coarse 2D grids (``test/test_cashleg/test_fold_pde_2d.py``)
_ADI = dict(n_x=100, n_v=18, n_t=24)


def _leverage(value: float = 0.9, s0: float = 100.0) -> LeverageSurface:
    """A constant, non-unit leverage: the SLV march differs from the Heston one."""
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(time_grid=np.linspace(0.0, 2.0, 5), strike_grid=strikes,
                           leverage_grid=np.full((5, strikes.size), float(value)))


@dataclass(frozen=True)
class EngineSpec:
    kind: str                        # mc | quad_v1 | quad_v2 | pde_1d | pde_1d_lv | pde_adi | dcn_mc | dcn_pde
    build: Callable[[], object]
    env: Callable[[], PricingEnvironment]


ENGINES: Dict[str, EngineSpec] = {
    "mc_snowball": EngineSpec("mc", lambda: SnowballMCEngine(params=MCParams(**_MC)), flat_env),
    "mc_phoenix": EngineSpec("mc", lambda: PhoenixMCEngine(params=MCParams(**_MC)), flat_env),
    "mc_lv_snowball": EngineSpec("mc_lv", lambda: LocalVolSnowballMCEngine(MCParams(**_MC)), skewed_env),
    "mc_lv_phoenix": EngineSpec("mc_lv", lambda: LocalVolPhoenixMCEngine(MCParams(**_MC)), skewed_env),
    "mc_heston_snowball": EngineSpec("mc_heston", lambda: HestonSnowballMCEngine(HESTON, params=MCParams(**_MC)),
                                     flat_env),
    "mc_heston_phoenix": EngineSpec("mc_heston", lambda: HestonPhoenixMCEngine(HESTON, params=MCParams(**_MC)),
                                    flat_env),
    "mc_dcn": EngineSpec("dcn_mc", lambda: DCNMCEngine(**_MC), dcn_env),
    "mc_lv_dcn": EngineSpec("dcn_mc_lv", lambda: LocalVolDCNMCEngine(**_MC), lambda: dcn_env(grid=True)),
    "mc_heston_dcn": EngineSpec("dcn_mc_heston", lambda: HestonDCNMCEngine(model_params=HESTON, **_MC), dcn_env),
    "quad_v1_snowball": EngineSpec("quad_v1", lambda: SnowballQuadEngine(QuadParams(grid_points=401)), flat_env),
    "quad_v1_phoenix": EngineSpec("quad_v1", lambda: PhoenixQuadEngine(QuadParams(grid_points=401)), flat_env),
    "quad_v1_ko_reset": EngineSpec("quad_v1", lambda: KOResetSnowballQuadEngine(QuadParams(grid_points=401)),
                                   flat_env),
    "quad_v2_snowball": EngineSpec("quad_v2", SnowballQuadEngineV2, flat_env),
    "quad_v2_phoenix": EngineSpec("quad_v2", PhoenixQuadEngineV2, flat_env),
    "quad_v2_ko_reset": EngineSpec("quad_v2", KOResetSnowballQuadEngineV2, flat_env),
    "pde_snowball": EngineSpec("pde_1d", lambda: SnowballPDESolver(params=PDEParams(accuracy="fast")), flat_env),
    "pde_phoenix": EngineSpec("pde_1d", lambda: PhoenixPDESolver(params=PDEParams(accuracy="fast")), flat_env),
    "pde_ko_reset": EngineSpec("pde_1d", lambda: KOResetSnowballPDESolver(params=PDEParams(accuracy="fast")),
                               flat_env),
    "pde_lv_snowball": EngineSpec("pde_1d_lv", lambda: LocalVolSnowballPDESolver(params=PDEParams(accuracy="fast")),
                                  skewed_env),
    "pde_lv_phoenix": EngineSpec("pde_1d_lv", lambda: LocalVolPhoenixPDESolver(params=PDEParams(accuracy="fast")),
                                 skewed_env),
    "pde_lv_ko_reset": EngineSpec("pde_1d_lv",
                                  lambda: LocalVolKOResetSnowballPDESolver(params=PDEParams(accuracy="fast")),
                                  skewed_env),
    "adi_heston_snowball": EngineSpec("pde_adi", lambda: HestonSnowballPDESolver(HESTON, **_ADI), flat_env),
    "adi_heston_phoenix": EngineSpec("pde_adi", lambda: HestonPhoenixPDESolver(HESTON, **_ADI), flat_env),
    "adi_heston_ko_reset": EngineSpec("pde_adi", lambda: HestonKOResetSnowballPDESolver(HESTON, **_ADI), flat_env),
    "adi_slv_snowball": EngineSpec("pde_adi_slv", lambda: HestonSLVSnowballPDESolver(HESTON, _leverage(), **_ADI),
                                   flat_env),
    "pde_dcn": EngineSpec("dcn_pde", lambda: DCNPDEEngine(PDEParams(accuracy="fast")), dcn_env),
    "pde_lv_dcn": EngineSpec("dcn_pde_lv", lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")),
                             lambda: dcn_env(grid=True)),
    "adi_heston_dcn": EngineSpec("dcn_pde_adi", lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21), dcn_env),
}

_SNOWBALLS = ("snowball_cki", "snowball_dki")
_PHOENIXES = ("phoenix_mem_instant", "phoenix_nomem_instant", "phoenix_mem_expiry", "phoenix_nomem_expiry")
_KO_RESETS = ("ko_reset_absolute", "ko_reset_rebased")

#: (engine, product) pairs, in banking order
CASES: Tuple[Tuple[str, str], ...] = (
    *(("mc_snowball", p) for p in (*_SNOWBALLS, *_KO_RESETS)),
    *(("mc_phoenix", p) for p in _PHOENIXES),
    ("mc_lv_snowball", "snowball_cki"),
    ("mc_lv_phoenix", "phoenix_mem_instant"),
    ("mc_heston_snowball", "snowball_cki"),
    ("mc_heston_snowball", "snowball_dki"),
    ("mc_heston_phoenix", "phoenix_mem_expiry"),
    ("mc_dcn", "dcn"),
    ("mc_lv_dcn", "dcn"),
    ("mc_heston_dcn", "dcn"),
    *(("quad_v1_snowball", p) for p in _SNOWBALLS),
    *(("quad_v1_phoenix", p) for p in _PHOENIXES),
    *(("quad_v1_ko_reset", p) for p in _KO_RESETS),
    *(("quad_v2_snowball", p) for p in _SNOWBALLS),
    *(("quad_v2_phoenix", p) for p in _PHOENIXES),
    *(("quad_v2_ko_reset", p) for p in _KO_RESETS),
    *(("pde_snowball", p) for p in _SNOWBALLS),
    *(("pde_phoenix", p) for p in _PHOENIXES),
    *(("pde_ko_reset", p) for p in _KO_RESETS),
    ("pde_lv_snowball", "snowball_cki"),
    ("pde_lv_phoenix", "phoenix_mem_instant"),
    ("pde_lv_ko_reset", "ko_reset_absolute"),
    ("adi_heston_snowball", "snowball_cki"),
    ("adi_heston_snowball", "snowball_dki"),
    ("adi_heston_phoenix", "phoenix_mem_instant"),
    ("adi_heston_phoenix", "phoenix_nomem_expiry"),
    ("adi_heston_ko_reset", "ko_reset_absolute"),
    ("adi_slv_snowball", "snowball_cki"),
    ("pde_dcn", "dcn"),
    ("pde_lv_dcn", "dcn"),
    ("adi_heston_dcn", "dcn"),
    # already knocked in at valuation; seller DCN
    *((e, "snowball_cki_knocked_in") for e in ("mc_snowball", "quad_v1_snowball", "quad_v2_snowball",
                                                "pde_snowball", "adi_heston_snowball")),
    *((e, "phoenix_mem_instant_knocked_in") for e in ("mc_phoenix", "quad_v1_phoenix", "quad_v2_phoenix",
                                                       "pde_phoenix")),
    ("mc_dcn", "dcn_seller"),
    ("pde_dcn", "dcn_seller"),
)


def case_id(engine: str, product: str) -> str:
    return f"{engine}__{product}"


# --- legs --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LegSpec:
    name: str
    leg: object
    descriptor: Dict[str, object]
    scale: float                     # the leg's nominal cash size: the comparison's absolute scale


def _ko_axis(product_name: str, product) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The parent's knock-out event axis an ``AutocallableCashLeg`` must align to, and a leg settlement
    schedule on it (the leg settles 0.01y after each observation; DCN at the knock-out payment)."""
    if product_name.startswith("dcn"):
        ctx = build_dcn_grid_context(product)
        obs = tuple(float(t) for t in np.asarray(ctx.times[ctx.obs_cols], dtype=float))
        return obs, tuple(float(t) for t in ctx.ko_pay_times)
    if product_name == "ko_reset_absolute":
        obs = tuple(sorted(KO_RESET_PRE + KO_RESET_POST_ABSOLUTE))
    elif product_name == "ko_reset_rebased":
        obs = tuple(sorted(KO_RESET_PRE + KO_RESET_POST_REBASED))
    else:
        obs = KO_DATES
    return obs, tuple(t + 0.01 for t in obs)


def _accrual(life, *, behavior=KOBehavior.TRUNCATE_AT_KO, basis=SurvivalBasis.ENTER_PERIOD,
             convention=PaymentConvention.AT_PERIOD_END, pay_shift=0.0, base=200_000.0):
    n = int(round(life * 4))
    starts = np.arange(n) * 0.25
    ends = starts + 0.25
    return AccrualLeg(
        direction=LegDirection.BUYER_RECEIVES, rate=0.04,
        base=BaseAmount(value=base, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends + pay_shift),
        ko_behavior=behavior, survival_basis=basis, payment_convention=convention,
    )


def _autocallable(obs, settle, *, leg_type, formula, basis=AccrualBasis.KO_MATURITY,
                  terminal=frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}),
                  direction=LegDirection.BUYER_PAYS, notional=1_000_000.0):
    return AutocallableCashLeg(
        direction=direction, leg_type=leg_type, notional=notional, rate=0.05,
        observation_schedule=tuple(obs), accrual_factors=tuple(np.linspace(0.3, 1.0, len(obs))),
        settlement_schedule=tuple(settle), terminal_accrual_factor=1.1,
        terminal_settlement_time=float(settle[-1]) + 0.01, pv_formula=formula, accrual_basis=basis,
        terminal_events=terminal,
    )


def legs_for(product_name: str, product) -> List[LegSpec]:
    """Every leg type, every ``PaymentTrigger``, both ``KOBehavior`` values and survival bases, each
    ``AccrualBasis`` and terminal-bucket choice of ``AutocallableCashLeg``. ``leg_id`` is the leg name, so
    a DRILLDOWN's per-leg dictionary is keyed by it."""
    life = PRODUCTS[product_name].life
    obs, settle = _ko_axis(product_name, product)
    both = frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})
    specs = [
        ("deterministic", DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5_000.0, payment_time=0.1),
         dict(leg_type="DeterministicLeg"), 5_000.0),
        ("accrual_truncate_enter", _accrual(life),
         dict(leg_type="AccrualLeg", ko_behavior="TRUNCATE_AT_KO", survival_basis="ENTER_PERIOD",
              payment_convention="AT_PERIOD_END"), 200_000.0 * 0.04 * life),
        ("accrual_truncate_complete", _accrual(life, basis=SurvivalBasis.COMPLETE_PERIOD, pay_shift=0.02),
         dict(leg_type="AccrualLeg", ko_behavior="TRUNCATE_AT_KO", survival_basis="COMPLETE_PERIOD",
              payment_convention="AT_PERIOD_END"), 200_000.0 * 0.04 * life),
        ("accrual_truncate_at_maturity", _accrual(life, convention=PaymentConvention.AT_MATURITY),
         dict(leg_type="AccrualLeg", ko_behavior="TRUNCATE_AT_KO", survival_basis="ENTER_PERIOD",
              payment_convention="AT_MATURITY"), 200_000.0 * 0.04 * life),
        ("accrual_full_schedule", _accrual(life, behavior=KOBehavior.PAY_FULL_SCHEDULE),
         dict(leg_type="AccrualLeg", ko_behavior="PAY_FULL_SCHEDULE", survival_basis="ENTER_PERIOD",
              payment_convention="AT_PERIOD_END"), 200_000.0 * 0.04 * life),
    ]
    for trigger, offset, amount in (
        (PaymentTrigger.AT_KO, 2, 7_000.0),
        (PaymentTrigger.AT_KI, 0, 6_000.0),
        (PaymentTrigger.AT_MATURITY_NO_KO, 0, 2_000.0),
        (PaymentTrigger.AT_MATURITY_WITH_KI, 0, 4_000.0),
        (PaymentTrigger.AT_MATURITY_ANY, 3, 3_000.0),
    ):
        specs.append((
            f"fixed_{trigger.value}",
            FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=amount, trigger=trigger,
                           payment_offset_days=offset),
            dict(leg_type="FixedPayoffLeg", trigger=trigger.name, payment_offset_days=offset), amount,
        ))
    autocallables = (
        ("autocallable_margin_nmp_both", AutocallableLegType.MARGIN, PvFormula.NOTIONAL_MINUS_PAYOFF,
         AccrualBasis.KO_MATURITY, both),
        ("autocallable_backend_premium_no_ko", AutocallableLegType.BACKEND_PREMIUM, PvFormula.NORMAL,
         AccrualBasis.KO_MATURITY, frozenset({EventType.MATURITY_NO_KO})),
        ("autocallable_rebate_with_ki", AutocallableLegType.REBATE, PvFormula.NORMAL,
         AccrualBasis.KO_MATURITY, frozenset({EventType.MATURITY_WITH_KI})),
        ("autocallable_backend_interest_coupon", AutocallableLegType.BACKEND_INTEREST, PvFormula.NORMAL,
         AccrualBasis.COUPON, both),
    )
    for name, leg_type, formula, basis, terminal in autocallables:
        specs.append((
            name,
            _autocallable(obs, settle, leg_type=leg_type, formula=formula, basis=basis, terminal=terminal),
            dict(leg_type="AutocallableCashLeg", autocallable_leg_type=leg_type.name, pv_formula=formula.name,
                 accrual_basis=basis.name, terminal_events=sorted(e.name for e in terminal)),
            1_000_000.0,
        ))
    out = []
    for name, leg, descriptor, scale in specs:
        out.append(LegSpec(name, _with_id(leg, name), descriptor, float(scale)))
    return out


def _with_id(leg, name):
    from dataclasses import replace

    return replace(leg, leg_id=name, name=name)


# --- record computation ------------------------------------------------------------------------


def _outcome(fn) -> Dict[str, object]:
    try:
        return {"value": float(fn())}
    except Exception as exc:  # noqa: BLE001 -- the bank records which exception a combination raises
        return {"raises": type(exc).__name__, "message": str(exc)[:300]}


@dataclass
class Case:
    engine_name: str
    product_name: str
    engine: EngineSpec = field(init=False)
    product_spec: ProductSpec = field(init=False)

    def __post_init__(self):
        self.engine = ENGINES[self.engine_name]
        self.product_spec = PRODUCTS[self.product_name]

    @property
    def id(self) -> str:
        return case_id(self.engine_name, self.product_name)

    def fresh(self):
        """A fresh engine, product and environment (no state carried between computations)."""
        return self.engine.build(), self.product_spec.build(), self.engine.env()

    def legs(self) -> List[LegSpec]:
        _, product, _ = self.fresh()
        return legs_for(self.product_name, product)

    def meta(self) -> Dict[str, object]:
        engine, product, env = self.fresh()
        return dict(
            engine=self.engine_name, engine_class=type(engine).__name__, engine_kind=self.engine.kind,
            product=self.product_name, product_class=type(product).__name__, family=self.product_spec.family,
            features=dict(self.product_spec.features),
            supports_cash_leg_folding=bool(getattr(engine, "supports_cash_leg_folding", False)),
        )


def event_route(case: Case) -> Dict[str, object]:
    """``price``, ``event_npv`` and every leg's event-route value for one case (two solves)."""
    engine, product, env = case.fresh()
    price = _outcome(lambda: engine.price(product, env))
    engine, product, env = case.fresh()
    try:
        result = engine.price_with_events(product, env)
    except Exception as exc:  # noqa: BLE001
        raised = {"raises": type(exc).__name__, "message": str(exc)[:300]}
        return dict(price=price, event_npv=raised, legs={spec.name: dict(raised) for spec in case.legs()},
                    diagnostics=None)
    dist = result.event_distribution
    legs = {spec.name: _outcome(lambda spec=spec: spec.leg.value(dist, env, UNIT)) for spec in case.legs()}
    return dict(price=price, event_npv={"value": float(result.npv)}, legs=legs, diagnostics=_diagnostics(dist))


def _diagnostics(dist) -> Dict[str, object]:
    """What the event route's distribution looked like at base (banked for T23's explanations; not compared)."""
    def total(p):
        return float(np.sum(np.asarray(p, dtype=float)))

    return dict(
        n_event_times=int(np.asarray(dist.event_times).size),
        event_times=[float(t) for t in np.asarray(dist.event_times, dtype=float)],
        streams=sorted(e.name for e in dist.probabilities),
        probability_totals={e.name: total(p) for e, p in dist.probabilities.items()},
        path_dependent_payment_events=sorted(e.name for e in dist.path_dependent_payment_events),
    )


def fold_total(case: Case, leg_name: str) -> Dict[str, object]:
    """``folded_trade_value(engine, product, env, [leg], quantity=1, mode=TOTAL).total``."""
    engine, product, env = case.fresh()
    spec = next(s for s in legs_for(case.product_name, product) if s.name == leg_name)
    return _outcome(lambda: folded_trade_value(engine, product, env, [spec.leg], quantity=1.0,
                                               unit_notional=UNIT, mode=FoldMode.TOTAL).total)


def fold_drilldown(case: Case, leg_names: List[str]) -> Dict[str, object]:
    """One DRILLDOWN over ``leg_names``: ``{"total", "npv", "per_leg": {name: outcome}}`` or a raise for all.

    An empty leg set (no leg folded at base) banks the pseudo-raise ``NoFoldableLeg`` for total and npv.
    """
    if not leg_names:
        none = {"raises": "NoFoldableLeg", "message": "no leg folded at base"}
        return dict(total=none, npv=dict(none), per_leg={})
    engine, product, env = case.fresh()
    legs = [s.leg for s in legs_for(case.product_name, product) if s.name in set(leg_names)]
    try:
        value = folded_trade_value(engine, product, env, legs, quantity=1.0, unit_notional=UNIT,
                                   mode=FoldMode.DRILLDOWN)
    except Exception as exc:  # noqa: BLE001
        raised = {"raises": type(exc).__name__, "message": str(exc)[:300]}
        return dict(total=raised, npv=dict(raised), per_leg={name: dict(raised) for name in leg_names})
    return dict(total={"value": float(value.total)}, npv={"value": float(value.npv)},
                per_leg={name: {"value": float(value.per_leg[name])} for name in leg_names})


def all_cases() -> List[Case]:
    return [Case(e, p) for e, p in CASES]
