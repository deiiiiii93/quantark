"""The MC engine x product x route grid of the unified event statistics task T3 (spec AC2, AC4, D14).

One definition of every configuration whose ``price()`` the event statistics must reproduce bitwise: the
plain and quasi-random serial routes, the Dask batch route, adaptive and fixed-count RQMC, the
conditional-moment estimators (Heston exact affine conditioning, Heston-SLV conditional control and its
control-only variant), the local-vol / Heston / SLV subclasses, KO-reset ABSOLUTE and REBASED, DCN (Sobol,
RQMC batches, pseudo-random, antithetic, vol models), a carried knock-in, a knock-out decided at valuation
and zero time to maturity.

Imports only ``quantark`` and the shared test fixtures that already existed at the base commit, so the
byte-compare script (``price()`` before vs after) loads this file against either checkout.

``ROUTES``: name -> ``Route(engine, product, env, kind)``. Every factory builds a fresh object.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    QEDCNMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
    QEPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.dcn_option import DCNDirection, DCNOption
from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

PSEUDO = MonteCarloMethod.PSEUDO
QUASI = MonteCarloMethod.QUASI
RQMC = MonteCarloMethod.RANDOMIZED_QUASI

VALUATION = datetime(2026, 1, 1)
KO_DATES = (0.25, 0.5, 0.75, 1.0)
HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


# --- markets -----------------------------------------------------------------------------------

def flat_env(spot: float = 100.0) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION,
    )


def grid_env(spot: float = 100.0, skew: bool = True) -> PricingEnvironment:
    """A grid surface (local-vol and SLV engines need one); flat at 20 % when ``skew`` is False."""
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.4, 11)))
    maturities = list(np.linspace(0.25, 2.0, 8))
    if skew:
        grid = np.array([[0.34 - 0.22 * (np.log(k / 100.0) + 0.6) for k in strikes] for _ in maturities])
    else:
        grid = np.full((len(maturities), len(strikes)), 0.20)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=GridVolSurface(strikes, maturities, grid),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION,
    )


DCN_START = datetime(2023, 1, 3)


def dcn_env(grid: bool = False, valuation: datetime = DCN_START) -> PricingEnvironment:
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = 0.184 + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        vol = FlatVolSurface(volatility=0.184)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=6000.0), vol_surface=vol, rate_curve=FlatRateCurve(rate=0.0356),
        div_yield=ContinuousDividendYield(div_yield=0.1406), valuation_date=valuation,
        day_count_convention=DayCountConvention.ACT_365,
    )


def unit_leverage(s0: float = 100.0, horizon: float = 2.0) -> LeverageSurface:
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(time_grid=np.linspace(0.0, horizon, 5), strike_grid=strikes,
                           leverage_grid=np.ones((5, strikes.size)))


# --- products ----------------------------------------------------------------------------------

def snowball(ki: str = "continuous", ko_dates=KO_DATES, *, maturity: float = 1.0, reverse: bool = False,
             disable_ko_after_ki: bool = False, knocked_in: bool = False,
             pay: CouponPayType = CouponPayType.INSTANT) -> SnowballOption:
    extra = dict(disable_ko_after_ki=True) if disable_ko_after_ki else {}
    ko_barrier, ki_barrier = (97.0, 125.0) if reverse else (103.0, 75.0)
    if ki == "continuous":
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(ko_dates))
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity, contract_multiplier=1.0, is_reverse=reverse,
        barrier_config=BarrierConfig(ko_barrier=ko_barrier, ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), **ki_kw, **extra),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        accrual_config=AccrualConfig(coupon_pay_type=pay),
    )
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


def affine_snowball() -> SnowballOption:
    """Discrete knock-out and knock-in: the exact conditional-moment estimator's domain."""
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(KO_DATES), ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(KO_DATES),
        ),
    )


def phoenix(memory: bool = True, pay: CouponPayType = CouponPayType.INSTANT, ko_dates=KO_DATES, *,
            maturity: float = 1.0, ki: str = "continuous", knocked_in: bool = False) -> PhoenixOption:
    if ki == "continuous":
        ki_kw = dict(ki_barrier=70.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=70.0, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(ko_dates))
    product = PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=maturity, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), **ki_kw),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * len(ko_dates), coupon_rate=0.02,
                                          coupon_pay_type=pay, day_count_convention=DayCountConvention.ACT_365,
                                          memory_coupon=memory),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


def _schedule(times, level) -> ObservationSchedule:
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def ko_reset(mode: PostKOScheduleMode = PostKOScheduleMode.ABSOLUTE, ki: str = "discrete", *,
             knocked_in: bool = False, pay: CouponPayType = CouponPayType.INSTANT) -> KnockOutResetSnowballOption:
    pre = (0.25, 0.5, 0.75, 1.0)
    post = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0) if mode is PostKOScheduleMode.ABSOLUTE else pre
    if ki == "continuous":
        ki_kw = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=_schedule(pre, 80.0))
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=2.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(pre, 103.0), ki_barrier=80.0, **ki_kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True,
                                     coupon_pay_type=pay),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=mode,
    )
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


def dcn(direction: DCNDirection = DCNDirection.BUYER, valuation: datetime = DCN_START) -> DCNOption:
    schedule = build_dcn_schedule(
        calendar=create_calendar(CalendarType.CHINA_SSE), initial_date=DCN_START, valuation_date=valuation,
        maturity_date=datetime(2025, 1, 3), tenor_months=24, lock_months=3, ko_lock_months=3,
        coupon_settlement_offset=2, ko_settlement_offset=2, settlement_date=datetime(2025, 1, 7),
    )
    return DCNOption(
        notional=1_000_000.0, initial_price=6000.0, coupon_barrier_ratio=0.80, ko_barrier_ratio=1.00,
        ki_barrier_ratio=0.75, ki_put_strike_ratio=1.10, coupon_rate=0.12, ko_coupon_rate=0.12,
        participation=1.0, coupon_counted_days=30, coupon_days_denom=360,
        settlement_date=datetime(2025, 1, 7), direction=direction, schedule=schedule,
        knocked_in_at_valuation=False,
    )


# --- engines -----------------------------------------------------------------------------------

N = 2048


def mc(num_paths: int = N, **kw) -> MCParams:
    return MCParams(num_paths=num_paths, seed=kw.pop("seed", 7), **kw)


def rqmc_fixed(batches: int = 3, paths: int = 512) -> MCParams:
    """RQMC that runs exactly ``batches`` batches (the target is unreachable)."""
    return MCParams(num_paths=paths, seed=11, rqmc_min_batches=batches, rqmc_max_batches=batches,
                    rqmc_target_std=1e-12, rqmc_paths_mode="per_batch")


def rqmc_adaptive(paths: int = 512) -> MCParams:
    """Adaptive RQMC: a loose target stops it between the minimum and the maximum batch count."""
    return MCParams(num_paths=paths, seed=11, rqmc_min_batches=2, rqmc_max_batches=12,
                    rqmc_target_std=0.25, rqmc_paths_mode="per_batch")


@dataclass(frozen=True)
class Route:
    engine: Callable[[], object]
    product: Callable[[], object]
    env: Callable[[], PricingEnvironment]
    #: snowball | phoenix | ko_reset | dcn
    family: str
    #: the expected number of priced batches (RQMC adaptive: read from the engine's last result)
    batches: object = 1
    #: zero time to maturity: price() decides on the spot and the statistics are None (expired rule)
    expired: bool = False


def _sb(method=PSEUDO, params=None, **kw):
    """A fresh engine per call (``params`` the route's MCParams, default ``mc()``)."""
    return lambda: SnowballMCEngine(params=deepcopy(params) if params is not None else mc(), method=method, **kw)


def _ph(method=PSEUDO, params=None, **kw):
    return lambda: PhoenixMCEngine(params=deepcopy(params) if params is not None else mc(), method=method, **kw)


def _t0_snowball():
    """Zero time to maturity: the contract's last observation is today (constructors refuse a zero
    maturity, so it is aged in place, as an OTC book ages a live trade to its last day)."""
    product = snowball(ko_dates=(0.0,))
    product.maturity = 0.0
    return product


def _t0_phoenix():
    product = phoenix(ko_dates=(0.0,))
    product.maturity = 0.0
    return product


ROUTES: Dict[str, Route] = {
    # --- Snowball, GBM --------------------------------------------------------------------------
    "sb_cki_pseudo": Route(_sb(), snowball, flat_env, "snowball"),
    "sb_cki_quasi": Route(_sb(QUASI), snowball, flat_env, "snowball"),
    "sb_dki_pseudo": Route(_sb(), lambda: snowball("discrete"), flat_env, "snowball"),
    "sb_dki_quasi": Route(_sb(QUASI), lambda: snowball("discrete"), flat_env, "snowball"),
    "sb_cki_expiry_pseudo": Route(_sb(), lambda: snowball(pay=CouponPayType.EXPIRY), flat_env, "snowball"),
    "sb_cki_rqmc_fixed": Route(_sb(RQMC, params=rqmc_fixed()), snowball, flat_env, "snowball", batches=3),
    "sb_cki_rqmc_adaptive": Route(_sb(RQMC, params=rqmc_adaptive()), snowball, flat_env, "snowball",
                                  batches="adaptive"),
    "sb_dki_rqmc_adaptive": Route(_sb(RQMC, params=rqmc_adaptive()), lambda: snowball("discrete"), flat_env,
                                  "snowball", batches="adaptive"),
    "sb_cki_dask": Route(_sb(use_dask=True, num_batches=3), snowball, flat_env, "snowball", batches=3),
    "sb_dki_dask_quasi": Route(_sb(QUASI, use_dask=True, num_batches=3), lambda: snowball("discrete"),
                               flat_env, "snowball", batches=3),
    "sb_cki_knocked_in": Route(_sb(), lambda: snowball(knocked_in=True), flat_env, "snowball"),
    "sb_cki_knocked_in_no_ko_after_ki": Route(
        _sb(), lambda: snowball(knocked_in=True, disable_ko_after_ki=True), flat_env, "snowball"),
    "sb_cki_no_ko_after_ki": Route(_sb(), lambda: snowball(disable_ko_after_ki=True), flat_env, "snowball"),
    "sb_dki_no_ko_after_ki": Route(_sb(), lambda: snowball("discrete", disable_ko_after_ki=True), flat_env,
                                   "snowball"),
    "sb_ko_at_valuation": Route(_sb(), lambda: snowball(ko_dates=(0.0, *KO_DATES)), lambda: flat_env(110.0),
                                "snowball"),
    "sb_reverse": Route(_sb(), lambda: snowball(reverse=True), flat_env, "snowball"),
    "sb_t0_ko": Route(_sb(), _t0_snowball, lambda: flat_env(110.0), "snowball", batches=0, expired=True),
    "sb_t0_alive": Route(_sb(), _t0_snowball, flat_env, "snowball", batches=0, expired=True),
    "sb_t0_ki": Route(_sb(), _t0_snowball, lambda: flat_env(60.0), "snowball", batches=0, expired=True),
    # --- Snowball, vol models -------------------------------------------------------------------
    "lv_sb_cki_pseudo": Route(lambda: LocalVolSnowballMCEngine(mc()), snowball, grid_env, "snowball"),
    "lv_sb_cki_quasi": Route(lambda: LocalVolSnowballMCEngine(mc(), method=QUASI), snowball, grid_env, "snowball"),
    "lv_sb_dki_rqmc": Route(lambda: LocalVolSnowballMCEngine(rqmc_fixed(), method=RQMC),
                            lambda: snowball("discrete"), grid_env, "snowball", batches=3),
    "lv_sb_cki_dask": Route(lambda: LocalVolSnowballMCEngine(mc(), use_dask=True, num_batches=2), snowball,
                            grid_env, "snowball", batches=2),
    "heston_sb_cki_pseudo": Route(lambda: HestonSnowballMCEngine(HESTON, params=mc()), snowball, flat_env,
                                  "snowball"),
    "heston_sb_dki_antithetic": Route(lambda: HestonSnowballMCEngine(HESTON, params=mc(use_antithetic=True)),
                                      lambda: snowball("discrete"), flat_env, "snowball"),
    "heston_sb_dki_quasi": Route(lambda: HestonSnowballMCEngine(HESTON, params=mc(), method=QUASI),
                                 lambda: snowball("discrete"), flat_env, "snowball"),
    "heston_sb_dki_rqmc": Route(lambda: HestonSnowballMCEngine(HESTON, params=rqmc_fixed(), method=RQMC),
                                lambda: snowball("discrete"), flat_env, "snowball", batches=3),
    "qe_sb_affine_rqmc": Route(
        lambda: QESnowballMCEngine(HESTON, params=rqmc_fixed(2), method=RQMC, martingale_correction=True,
                                   substeps_per_interval=2, rqmc_affine_spot_factor=True),
        affine_snowball, flat_env, "snowball", batches=2),
    "qe_sb_affine_bridge_strata_rqmc": Route(
        lambda: QESnowballMCEngine(HESTON, params=rqmc_fixed(2), method=RQMC, martingale_correction=True,
                                   substeps_per_interval=2, rqmc_affine_spot_factor=True,
                                   rqmc_spot_bridge_strata=2),
        affine_snowball, flat_env, "snowball", batches=2),
    "qe_sb_affine_adaptive": Route(
        lambda: QESnowballMCEngine(HESTON, params=rqmc_adaptive(), method=RQMC, rqmc_affine_spot_factor=True),
        affine_snowball, flat_env, "snowball", batches="adaptive"),
    "slv_sb_dki_pseudo": Route(lambda: HestonSLVSnowballMCEngine(HESTON, params=mc(),
                                                                 leverage_surface=unit_leverage()),
                               lambda: snowball("discrete"), lambda: grid_env(skew=False), "snowball"),
    "slvqe_sb_conditional_control": Route(
        lambda: HestonSLVQESnowballMCEngine(HESTON, params=rqmc_fixed(2), method=RQMC,
                                            leverage_surface=unit_leverage(), martingale_correction=True,
                                            substeps_per_interval=2, rqmc_heston_conditional_control=True,
                                            rqmc_spot_strata=2),
        affine_snowball, lambda: grid_env(skew=False), "snowball", batches=2),
    "slvqe_sb_control_only": Route(
        lambda: HestonSLVQESnowballMCEngine(HESTON, params=rqmc_fixed(2), method=RQMC,
                                            leverage_surface=unit_leverage(), martingale_correction=True,
                                            substeps_per_interval=2, rqmc_heston_conditional_control=True,
                                            rqmc_frozen_leverage_conditional_control=True,
                                            rqmc_conditional_control_only=True),
        affine_snowball, lambda: grid_env(skew=False), "snowball", batches=2),
    # --- KO-reset -------------------------------------------------------------------------------
    "kr_abs_pseudo": Route(_sb(), ko_reset, flat_env, "ko_reset"),
    "kr_abs_quasi": Route(_sb(QUASI), ko_reset, flat_env, "ko_reset"),
    "kr_abs_expiry": Route(_sb(), lambda: ko_reset(pay=CouponPayType.EXPIRY), flat_env, "ko_reset"),
    "kr_abs_cki_pseudo": Route(_sb(), lambda: ko_reset(ki="continuous"), flat_env, "ko_reset"),
    "kr_abs_rqmc_adaptive": Route(_sb(RQMC, params=rqmc_adaptive()), ko_reset, flat_env, "ko_reset",
                                  batches="adaptive"),
    "kr_abs_dask": Route(_sb(use_dask=True, num_batches=3), ko_reset, flat_env, "ko_reset", batches=3),
    "kr_abs_knocked_in": Route(_sb(), lambda: ko_reset(knocked_in=True), flat_env, "ko_reset"),
    "kr_rebased_pseudo": Route(_sb(), lambda: ko_reset(PostKOScheduleMode.REBASED), flat_env, "ko_reset"),
    "kr_rebased_quasi": Route(_sb(QUASI), lambda: ko_reset(PostKOScheduleMode.REBASED), flat_env, "ko_reset"),
    "kr_rebased_rqmc": Route(_sb(RQMC, params=rqmc_fixed()), lambda: ko_reset(PostKOScheduleMode.REBASED),
                             flat_env, "ko_reset", batches=3),
    "kr_rebased_dask": Route(_sb(use_dask=True, num_batches=2), lambda: ko_reset(PostKOScheduleMode.REBASED),
                             flat_env, "ko_reset", batches=2),
    "kr_rebased_expiry": Route(_sb(), lambda: ko_reset(PostKOScheduleMode.REBASED, pay=CouponPayType.EXPIRY),
                               flat_env, "ko_reset"),
    "heston_kr_abs_pseudo": Route(lambda: HestonSnowballMCEngine(HESTON, params=mc()), ko_reset, flat_env,
                                  "ko_reset"),
    "lv_kr_abs_cki_pseudo": Route(lambda: LocalVolSnowballMCEngine(mc()), lambda: ko_reset(ki="continuous"),
                                  grid_env, "ko_reset"),
    # --- Phoenix --------------------------------------------------------------------------------
    "ph_mem_instant_pseudo": Route(_ph(), phoenix, flat_env, "phoenix"),
    "ph_nomem_instant_pseudo": Route(_ph(), lambda: phoenix(False), flat_env, "phoenix"),
    "ph_mem_expiry_pseudo": Route(_ph(), lambda: phoenix(True, CouponPayType.EXPIRY), flat_env, "phoenix"),
    "ph_nomem_expiry_quasi": Route(_ph(QUASI), lambda: phoenix(False, CouponPayType.EXPIRY), flat_env, "phoenix"),
    "ph_mem_instant_dki_quasi": Route(_ph(QUASI), lambda: phoenix(ki="discrete"), flat_env, "phoenix"),
    "ph_mem_instant_rqmc_fixed": Route(_ph(RQMC, params=rqmc_fixed()), phoenix, flat_env, "phoenix", batches=3),
    "ph_mem_expiry_rqmc_adaptive": Route(_ph(RQMC, params=rqmc_adaptive()),
                                         lambda: phoenix(True, CouponPayType.EXPIRY), flat_env, "phoenix",
                                         batches="adaptive"),
    "ph_mem_instant_dask": Route(_ph(use_dask=True, num_batches=3), phoenix, flat_env, "phoenix", batches=3),
    "ph_nomem_expiry_dask_quasi": Route(_ph(QUASI, use_dask=True, num_batches=2),
                                        lambda: phoenix(False, CouponPayType.EXPIRY), flat_env, "phoenix",
                                        batches=2),
    "ph_knocked_in": Route(_ph(), lambda: phoenix(knocked_in=True), flat_env, "phoenix"),
    "ph_ko_at_valuation": Route(_ph(), lambda: phoenix(ko_dates=(0.0, *KO_DATES)), lambda: flat_env(110.0),
                                "phoenix"),
    "ph_t0_ko": Route(_ph(), _t0_phoenix, lambda: flat_env(110.0), "phoenix", batches=0, expired=True),
    "ph_t0_alive": Route(_ph(), _t0_phoenix, flat_env, "phoenix", batches=0, expired=True),
    "lv_ph_pseudo": Route(lambda: LocalVolPhoenixMCEngine(mc()), phoenix, grid_env, "phoenix"),
    "lv_ph_rqmc": Route(lambda: LocalVolPhoenixMCEngine(rqmc_fixed(), method=RQMC), phoenix, grid_env, "phoenix",
                        batches=3),
    "heston_ph_pseudo": Route(lambda: HestonPhoenixMCEngine(HESTON, params=mc()),
                              lambda: phoenix(True, CouponPayType.EXPIRY), flat_env, "phoenix"),
    "heston_ph_quasi": Route(lambda: HestonPhoenixMCEngine(HESTON, params=mc(), method=QUASI),
                             lambda: phoenix(ki="discrete"), flat_env, "phoenix"),
    "qe_ph_rqmc": Route(lambda: QEPhoenixMCEngine(HESTON, params=rqmc_fixed(2), method=RQMC),
                        lambda: phoenix(ki="discrete"), flat_env, "phoenix", batches=2),
    "slv_ph_pseudo": Route(lambda: HestonSLVPhoenixMCEngine(HESTON, params=mc(), leverage_surface=unit_leverage()),
                           lambda: phoenix(ki="discrete"), lambda: grid_env(skew=False), "phoenix"),
    # --- DCN ------------------------------------------------------------------------------------
    "dcn_sobol": Route(lambda: DCNMCEngine(num_paths=N, seed=7), dcn, dcn_env, "dcn"),
    "dcn_sobol_batches": Route(lambda: DCNMCEngine(num_paths=N, seed=7, num_batches=4), dcn, dcn_env, "dcn",
                               batches=4),
    "dcn_sobol_batches_threads": Route(lambda: DCNMCEngine(num_paths=N, seed=7, num_batches=4, num_workers=3),
                                       dcn, dcn_env, "dcn", batches=4),
    "dcn_pseudo": Route(lambda: DCNMCEngine(num_paths=N, seed=7, use_sobol=False, num_batches=2), dcn, dcn_env,
                        "dcn", batches=2),
    "dcn_antithetic": Route(lambda: DCNMCEngine(num_paths=N, seed=7, use_sobol=False, use_antithetic=True),
                            dcn, dcn_env, "dcn"),
    "dcn_seller": Route(lambda: DCNMCEngine(num_paths=N, seed=7), lambda: dcn(DCNDirection.SELLER), dcn_env, "dcn"),
    "lv_dcn": Route(lambda: LocalVolDCNMCEngine(num_paths=N, seed=7), dcn, lambda: dcn_env(grid=True), "dcn"),
    "heston_dcn": Route(lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=N, seed=7), dcn, dcn_env, "dcn"),
    "qe_dcn_batches": Route(lambda: QEDCNMCEngine(model_params=HESTON, num_paths=N, seed=7, num_batches=2),
                            dcn, dcn_env, "dcn", batches=2),
}

#: routes whose price() runs the Dask batch reducer
DASK_ROUTES = tuple(name for name in ROUTES if "dask" in name)
