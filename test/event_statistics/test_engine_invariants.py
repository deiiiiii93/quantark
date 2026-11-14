"""Identities, partition, no knock-in after a knock-out and the valuation-date rules on EVERY statistics engine
(unified event statistics spec AC8, AC9, AC10; D16; T22).

One place, one parametrization over every engine class that produces ``EventStatistics`` -- MC (Snowball, Phoenix,
KO-reset, DCN and every vol-model MC subclass), PDE 1D (flat, local vol), Heston / SLV ADI, DCN PDE (flat, local vol,
Heston), QUAD V2, QUAD V1 -- each on the products of its family:

- AC8 (D16): ``|probability_residual|`` and ``|reconciliation_error| / max(1, |pv|)`` within the engine's own banked
  bound (``RESIDUAL_BOUNDS``, measured here on every case of this module) on every case; the bounds of the engines
  whose residual is a discretisation error (PDE 1D, DCN PDE, QUAD V1 EXPIRY) shrink under refinement
  (``test_cross_engine_smoke.test_banked_residuals_shrink_under_refinement`` and the per-engine suites);
- AC9: a knock-out certain before the knock-in barrier can be reached, the barrier crossed afterwards on a large
  share of paths: P(knock in) = 0 and expected loss 0 (the same note with an unreachable knock-out knocks in);
- AC10 (DESIGN §4.5): knocked out at valuation (no knock-in, nothing knocked in, an empty KI table, the expected time
  undefined), a carried knock-in and a knock-in at t = 0 (P = 1 at time 0, one row at 0), no KI barrier (computed
  zeros). A KO-reset and a DCN always carry a KI barrier, so "no KI barrier" has no case on them.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace
from datetime import datetime
from typing import Callable, Dict, NamedTuple

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc import (
    CoupledCoarseHestonDCNMCEngine,
    DCNMCEngine,
    HestonDCNMCEngine,
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolDCNMCEngine,
    LocalVolPhoenixMCEngine,
    LocalVolSnowballMCEngine,
    PhoenixMCEngine,
    QEDCNMCEngine,
    QEPhoenixMCEngine,
    QESnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    DCNPDEEngine,
    HestonDCNPDESolver,
    HestonKOResetSnowballPDESolver,
    HestonPhoenixPDESolver,
    HestonSLVKOResetSnowballPDESolver,
    HestonSLVPhoenixPDESolver,
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    KOResetSnowballPDESolver,
    LocalVolDCNPDEEngine,
    LocalVolKOResetSnowballPDESolver,
    LocalVolPhoenixPDESolver,
    LocalVolSnowballPDESolver,
    PhoenixPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, make_dcn
from test_asset.equity.engine_mc import mc_price_path_routes as routes

warnings.filterwarnings("ignore", category=DeprecationWarning)

HP = routes.HESTON
DCN_HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
#: a small MC run: the identities are path sums (round-off whatever the count), AC9/AC10 are exact zeros and ones
PATHS = 512
ADI_MESH = dict(n_x=64, n_v=12, n_t=48)
#: the Phoenix ADI on the 64 x 12 x 48 mesh refuses a knock-in at valuation (spot 65: its Craig-Sneyd splitting is
#: not monotone, a KO probability column reads -4.4e-6, beyond the type's slack, ValidationError); 128 x 24 x 96 reads
#: -4.4e-7, inside it (T14 open item: coarse ADI meshes can refuse statistics; default meshes within slack)
ADI_MESH_PHOENIX = dict(n_x=128, n_v=24, n_t=96)


def _mc(seed=7):
    return MCParams(num_paths=PATHS, seed=seed)


class Engine(NamedTuple):
    make: Callable[[], object]
    family: str          # snowball | phoenix | ko_reset | dcn
    market: str          # flat | grid | grid_flat (equity) ; dcn_flat | dcn_grid
    kind: str            # the RESIDUAL_BOUNDS key


#: every engine class that produces EventStatistics, on the family it prices (vol-model Snowball MC engines price
#: KO-reset too: one KO-reset entry each for the flat and the Heston MC)
ENGINES: Dict[str, Engine] = {
    # MC
    "mc_snowball": Engine(lambda: SnowballMCEngine(params=_mc()), "snowball", "flat", "mc"),
    "mc_ko_reset": Engine(lambda: SnowballMCEngine(params=_mc()), "ko_reset", "flat", "mc"),
    "mc_phoenix": Engine(lambda: PhoenixMCEngine(params=_mc()), "phoenix", "flat", "mc"),
    "mc_lv_snowball": Engine(lambda: LocalVolSnowballMCEngine(_mc()), "snowball", "grid", "mc"),
    "mc_heston_snowball": Engine(lambda: HestonSnowballMCEngine(HP, params=_mc()), "snowball", "flat", "mc"),
    "mc_heston_ko_reset": Engine(lambda: HestonSnowballMCEngine(HP, params=_mc()), "ko_reset", "flat", "mc"),
    "mc_qe_snowball": Engine(lambda: QESnowballMCEngine(HP, params=_mc()), "snowball", "flat", "mc"),
    "mc_slv_snowball": Engine(lambda: HestonSLVSnowballMCEngine(HP, params=_mc(),
                                                                leverage_surface=routes.unit_leverage()),
                              "snowball", "grid_flat", "mc"),
    "mc_slvqe_snowball": Engine(lambda: HestonSLVQESnowballMCEngine(HP, params=_mc(),
                                                                    leverage_surface=routes.skew_leverage()),
                                "snowball", "grid_flat", "mc"),
    "mc_lv_phoenix": Engine(lambda: LocalVolPhoenixMCEngine(_mc()), "phoenix", "grid", "mc"),
    "mc_heston_phoenix": Engine(lambda: HestonPhoenixMCEngine(HP, params=_mc()), "phoenix", "flat", "mc"),
    "mc_qe_phoenix": Engine(lambda: QEPhoenixMCEngine(HP, params=_mc()), "phoenix", "flat", "mc"),
    "mc_slv_phoenix": Engine(lambda: HestonSLVPhoenixMCEngine(HP, params=_mc(),
                                                              leverage_surface=routes.unit_leverage()),
                             "phoenix", "grid_flat", "mc"),
    "mc_slvqe_phoenix": Engine(lambda: HestonSLVQEPhoenixMCEngine(HP, params=_mc(),
                                                                  leverage_surface=routes.skew_leverage()),
                               "phoenix", "grid_flat", "mc"),
    "mc_dcn": Engine(lambda: DCNMCEngine(num_paths=PATHS, seed=7, use_sobol=False, num_workers=1), "dcn",
                     "dcn_flat", "mc"),
    "mc_lv_dcn": Engine(lambda: LocalVolDCNMCEngine(num_paths=PATHS, seed=7, use_sobol=False, num_workers=1), "dcn",
                        "dcn_grid", "mc"),
    "mc_heston_dcn": Engine(lambda: HestonDCNMCEngine(model_params=DCN_HESTON, num_paths=PATHS, seed=7,
                                                      use_sobol=False, num_workers=1), "dcn", "dcn_flat", "mc"),
    "mc_qe_dcn": Engine(lambda: QEDCNMCEngine(model_params=DCN_HESTON, num_paths=PATHS, seed=7, use_sobol=False,
                                              num_workers=1), "dcn", "dcn_flat", "mc"),
    "mc_coupled_heston_dcn": Engine(
        lambda: CoupledCoarseHestonDCNMCEngine(
            HestonDCNMCEngine(model_params=DCN_HESTON, substeps_per_interval=2, num_paths=PATHS, seed=7,
                              use_sobol=False, num_workers=1),
            model_params=DCN_HESTON, num_paths=PATHS, seed=7, use_sobol=False, num_workers=1),
        "dcn", "dcn_flat", "mc"),
    # PDE 1D
    "pde_snowball": Engine(lambda: SnowballPDESolver(PDEParams()), "snowball", "flat", "pde_1d"),
    "pde_phoenix": Engine(lambda: PhoenixPDESolver(PDEParams()), "phoenix", "flat", "pde_1d"),
    "pde_ko_reset": Engine(lambda: KOResetSnowballPDESolver(PDEParams()), "ko_reset", "flat", "pde_1d"),
    "pde_lv_snowball": Engine(lambda: LocalVolSnowballPDESolver(params=PDEParams()), "snowball", "grid", "pde_1d"),
    "pde_lv_phoenix": Engine(lambda: LocalVolPhoenixPDESolver(params=PDEParams()), "phoenix", "grid", "pde_1d"),
    "pde_lv_ko_reset": Engine(lambda: LocalVolKOResetSnowballPDESolver(params=PDEParams()), "ko_reset", "grid",
                              "pde_1d"),
    # Heston / SLV ADI
    "adi_heston_snowball": Engine(lambda: HestonSnowballPDESolver(HP, **ADI_MESH), "snowball", "flat", "adi"),
    "adi_heston_phoenix": Engine(lambda: HestonPhoenixPDESolver(HP, **ADI_MESH), "phoenix", "flat", "adi"),
    "adi_heston_ko_reset": Engine(lambda: HestonKOResetSnowballPDESolver(HP, **ADI_MESH), "ko_reset", "flat", "adi"),
    "adi_slv_snowball": Engine(lambda: HestonSLVSnowballPDESolver(HP, leverage_surface=routes.skew_leverage(),
                                                                  **ADI_MESH), "snowball", "flat", "adi"),
    "adi_slv_phoenix": Engine(lambda: HestonSLVPhoenixPDESolver(HP, leverage_surface=routes.unit_leverage(),
                                                                **ADI_MESH), "phoenix", "flat", "adi"),
    "adi_slv_ko_reset": Engine(lambda: HestonSLVKOResetSnowballPDESolver(HP, leverage_surface=routes.unit_leverage(),
                                                                         **ADI_MESH), "ko_reset", "flat", "adi"),
    # DCN PDE
    "dcn_pde": Engine(lambda: DCNPDEEngine(PDEParams(accuracy="fast")), "dcn", "dcn_flat", "dcn_pde"),
    "dcn_lv_pde": Engine(lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")), "dcn", "dcn_grid",
                         "dcn_pde"),
    "dcn_heston_adi": Engine(lambda: HestonDCNPDESolver(DCN_HESTON, n_x=61, n_v=21), "dcn", "dcn_flat",
                             "dcn_heston_adi"),
    # QUAD
    "quad_v2_snowball": Engine(SnowballQuadEngineV2, "snowball", "flat", "quad_v2"),
    "quad_v2_phoenix": Engine(PhoenixQuadEngineV2, "phoenix", "flat", "quad_v2"),
    "quad_v2_ko_reset": Engine(KOResetSnowballQuadEngineV2, "ko_reset", "flat", "quad_v2"),
    "quad_v1_snowball": Engine(lambda: SnowballQuadEngine(QuadParams()), "snowball", "flat", "quad_v1"),
    "quad_v1_phoenix": Engine(lambda: PhoenixQuadEngine(QuadParams()), "phoenix", "flat", "quad_v1"),
    "quad_v1_ko_reset": Engine(lambda: KOResetSnowballQuadEngine(QuadParams()), "ko_reset", "flat", "quad_v1"),
}
#: (engine, case) run on another build of the engine (see ADI_MESH_PHOENIX)
OVERRIDES = {
    ("adi_heston_phoenix", "ki_at_valuation"): lambda: HestonPhoenixPDESolver(HP, **ADI_MESH_PHOENIX),
    ("adi_slv_phoenix", "ki_at_valuation"): lambda: HestonSLVPhoenixPDESolver(
        HP, leverage_surface=routes.unit_leverage(), **ADI_MESH_PHOENIX),
}
NON_DCN = [name for name, e in ENGINES.items() if e.family != "dcn"]
NO_KI_BARRIER_FAMILIES = ("snowball", "phoenix")

#: AC8 bounds per engine kind, (|probability_residual|, |reconciliation_error| / max(1, |pv|)), banked over the max
#: measured on every case of this module (T22, ARM64; worst case in brackets):
#: - mc: every outcome its own path sum (D16: MC round-off) -- measured 0 / 2.6e-14 (mc_lv_dcn base);
#: - pde_1d: the march's discount on the unit columns, O(dt^2) -- 5.7e-9 (pde_ko_reset ki_at_valuation) / 1.3e-15;
#: - adi: every outcome its own exactly discounted column -- 1.5e-14 / 1.2e-15;
#: - dcn_pde: the march's own discounting of a unit claim against DF(t) -- 3.0e-8 (ki_at_valuation) / 9.4e-14;
#: - dcn_heston_adi: as dcn_pde on the DCN's own ADI loop -- 3.1e-7 (ki_at_valuation) / 4.6e-15;
#: - quad_v2: round-off -- 1.1e-15 / 4.1e-16;
#: - quad_v1: round-off -- 1.6e-15 -- but the EXPIRY Phoenix reconciliation (the price knocks out W by the nodal
#:   weight, the statistics by the joint projection): 1.08e-9 x pv at the default 1001 nodes (quad_v1_phoenix base;
#:   per-grid bound and its shrinkage in test_quad_v1_event_statistics).
#: The residuals of pde_1d, dcn_pde, dcn_heston_adi and the V1 EXPIRY gap shrink under refinement (the smoke bank's
#: residual evidence; test_pde_event_statistics, test_dcn_pde_event_statistics, test_quad_v1_event_statistics).
RESIDUAL_BOUNDS = {
    "mc": (1e-12, 1e-12),
    "pde_1d": (1e-8, 1e-12),
    "adi": (1e-12, 1e-12),
    "dcn_pde": (1e-7, 1e-12),
    "dcn_heston_adi": (1e-6, 1e-12),
    "quad_v2": (1e-12, 1e-12),
    "quad_v1": (1e-12, 2e-9),
}

#: AC9 on a truncated grid. The knock-out at the first row is certain on every path; a grid engine still reports the
#: mass its own price row treats as knocked in before that knock-out: (a) the far knock-in-side Dirichlet row (D23: the
#: statistics follow the price's own row) is the knocked-in surface's while a discrete KI observation remains and the
#: edge lies beyond the KI level -- a row that assumes the knock-out unreachable at the low-spot edge, which a KO
#: barrier of 1 (below the domain) breaks; the mass is the edge-reaching mass before the knock-out, vanishing with the
#: domain (test_the_ac9_knock_in_on_a_grid_is_far_edge_mass_that_vanishes); (b) on the DCN Heston ADI, numerical
#: diffusion of one 3-day step putting mass through 75% on the knock-out close, where the DCN tests KI first.
#: Measured (T22): pde_lv 1.07e-5 / 1.07e-5 / 2.1e-8 (snowball / phoenix / KO-reset; flat PDE 1.5e-15), ADI Heston
#: snowball / phoenix 2.1e-10 / 3.3e-10, SLV snowball (skew leverage) / phoenix 4.9e-9 / 3.3e-10, ADI KO-reset 6.9e-13,
#: dcn_heston_adi 4.6e-9, DCN 1D 4.9e-26; MC, QUAD V2 and V1 exactly 0 (V1 KO-reset 2.4e-19).
AC9_BOUNDS = {"pde_lv_snowball": 2e-5, "pde_lv_phoenix": 2e-5, "pde_lv_ko_reset": 5e-8, "adi_heston_snowball": 5e-10,
              "adi_heston_phoenix": 5e-10, "adi_slv_snowball": 1e-8, "adi_slv_phoenix": 5e-10, "dcn_heston_adi": 1e-8}
AC9_DEFAULT_BOUND = 2e-12


# --- markets --------------------------------------------------------------------------------------------------------

DCN_DAY = datetime(2024, 5, 15)                # a trading day that is not a monthly row
DCN_OBS = datetime(2024, 12, 3)                # a coupon + knock-out row
DCN_BEFORE_FIRST_KO = datetime(2023, 3, 31)    # the trading day before DCN_A's first knock-out row (2023-04-03)


def _dcn_env(day, spot=6000.0, grid=False, vol=None):
    sigma = FLAT["sigma"] if vol is None else vol
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = sigma + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        surface = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        surface = FlatVolSurface(volatility=sigma)
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=surface,
                              rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(FLAT["q"]),
                              valuation_date=day, day_count_convention=DayCountConvention.ACT_365)


def _env(engine: Engine, spot: float = None, vol: float = None, day=DCN_DAY):
    if engine.market.startswith("dcn"):
        return _dcn_env(day, 6000.0 if spot is None else spot, grid=engine.market == "dcn_grid", vol=vol)
    spot = 100.0 if spot is None else spot
    if engine.market == "flat":
        if vol is None:
            return routes.flat_env(spot)
        return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                                  rate_curve=FlatRateCurve(rate=0.03),
                                  div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=routes.VALUATION)
    if engine.market == "grid":
        return routes.grid_env(spot)
    return routes.grid_env(spot, skew=False)


# --- products -------------------------------------------------------------------------------------------------------

def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def _knock_out_first(family: str, ko: float):
    """KO barrier ``ko`` on every row from 0.25 on (1.0: certain at 0.25); a discrete KI at 99 observed only after it
    (0.5, 0.75, 1.0), which the market crosses on about half the paths."""
    ki = dict(ki_barrier=99.0, ki_observation_type=ObservationType.DISCRETE)
    if family == "snowball":
        return SnowballOption(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
            barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                         ko_observation_dates=list(routes.KO_DATES),
                                         ki_observation_dates=[0.5, 0.75, 1.0], **ki),
            payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))
    if family == "phoenix":
        return PhoenixOption(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
            barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                         ko_observation_dates=list(routes.KO_DATES),
                                         ki_observation_dates=[0.5, 0.75, 1.0], **ki),
            coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * 4, coupon_rate=0.02,
                                              coupon_pay_type=CouponPayType.INSTANT,
                                              day_count_convention=DayCountConvention.ACT_365, memory_coupon=False),
            payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=2.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(routes.KO_DATES, ko), ki_barrier=99.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule((0.5, 0.75, 1.0), 99.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule((0.5, 0.75, 1.0, 1.5, 2.0), 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)


def _ko_reset_with_ko_at_valuation():
    product = routes.ko_reset()
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=_schedule(
        (0.0, 0.25, 0.5, 0.75, 1.0), 103.0))
    product.validate()
    return product


def _no_ki_snowball():
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(routes.KO_DATES), ki_barrier=None),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def _no_ki_phoenix():
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(routes.KO_DATES), ki_barrier=None),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * 4, coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def case(engine: Engine, name: str):
    """``(product, env)`` of a named case on the engine's family and market."""
    family = engine.family
    if family == "dcn":
        if name == "base":
            return make_dcn(DCN_A, valuation_date=DCN_DAY), _env(engine, day=DCN_DAY)
        if name in ("ac9", "ac9_reachable"):
            # KO at 0.76 x S0 on the next row (one trading day on) is certain but for a 24% one-day fall; KI at 0.75
            # observed daily after it, crossed on most paths of the 21 months left (q - r = 10.5%); 0.76 -> 1.50
            # puts the knock-out out of reach
            ratio = 0.76 if name == "ac9" else 1.50
            return (make_dcn(DCN_A, valuation_date=DCN_BEFORE_FIRST_KO, ko_barrier_ratio=ratio),
                    _env(engine, day=DCN_BEFORE_FIRST_KO))
        if name == "ko_at_valuation":
            return make_dcn(DCN_A, valuation_date=DCN_OBS), _env(engine, 6300.0, day=DCN_OBS)
        if name == "carried":
            return (make_dcn(DCN_A, valuation_date=DCN_DAY, knocked_in_at_valuation=True),
                    _env(engine, 5000.0, day=DCN_DAY))
        if name == "ki_at_valuation":
            return make_dcn(DCN_A, valuation_date=DCN_DAY), _env(engine, 4400.0, day=DCN_DAY)
        raise KeyError(name)
    if name == "base":
        product = {"snowball": routes.snowball, "phoenix": lambda: routes.phoenix(pay=CouponPayType.EXPIRY),
                   "ko_reset": routes.ko_reset}[family]()
        return product, _env(engine)
    if name in ("ac9", "ac9_reachable"):
        return _knock_out_first(family, 1.0 if name == "ac9" else 103.0), _env(engine)
    if name == "ko_at_valuation":
        product = {"snowball": lambda: routes.snowball(ko_dates=(0.0, *routes.KO_DATES)),
                   "phoenix": lambda: routes.phoenix(ko_dates=(0.0, *routes.KO_DATES)),
                   "ko_reset": _ko_reset_with_ko_at_valuation}[family]()
        return product, _env(engine, 110.0)
    if name == "carried":
        product = {"snowball": lambda: routes.snowball(knocked_in=True),
                   "phoenix": lambda: routes.phoenix(knocked_in=True),
                   "ko_reset": lambda: routes.ko_reset(knocked_in=True)}[family]()
        return product, _env(engine)
    if name == "ki_at_valuation":
        # continuous KI with the spot already through it (snowball 75, phoenix 70, KO-reset 80)
        product, spot = {"snowball": (routes.snowball, 70.0), "phoenix": (routes.phoenix, 65.0),
                         "ko_reset": (lambda: routes.ko_reset(ki="continuous"), 75.0)}[family]
        return product(), _env(engine, spot)
    if name == "no_ki_barrier":
        return {"snowball": _no_ki_snowball, "phoenix": _no_ki_phoenix}[family](), _env(engine)
    raise KeyError(name)


def _stats(engine_name: str, case_name: str) -> EventStatistics:
    engine = ENGINES[engine_name]
    product, env = case(engine, case_name)
    make = OVERRIDES.get((engine_name, case_name), engine.make)
    stats = make().calculate_event_stats(product, env)
    assert isinstance(stats, EventStatistics), (engine_name, case_name)
    return stats


def _assert_partition(engine_name: str, stats: EventStatistics):
    probability_bound, reconciliation_bound = RESIDUAL_BOUNDS[ENGINES[engine_name].kind]
    assert abs(stats.probability_residual) <= probability_bound, stats.probability_residual
    assert abs(stats.reconciliation_error) <= reconciliation_bound * max(1.0, abs(stats.pv)), stats.reconciliation_error
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    assert math.isclose(stats.ledger.total_discounted_cash, stats.total_discounted_cash, rel_tol=1e-9,
                        abs_tol=1e-9 * max(1.0, abs(stats.pv)))


def _assert_no_knock_in(stats: EventStatistics):
    assert stats.knock_in.probability == 0.0
    assert stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert stats.ki_table is not None and stats.ki_table.size == 0
    assert stats.knock_in.expected_time is None
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


def _assert_knock_in_at_zero(stats: EventStatistics):
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert stats.ki_table.times[0] == 0.0 and stats.ki_table.probability[0] == 1.0
    assert np.all(stats.ki_table.probability[1:] == 0.0)
    assert stats.maturity.probability == pytest.approx(0.0, abs=1e-12)
    assert "knock_in.expected_time" not in stats.unavailable


CASES_PER_FAMILY = {
    "snowball": ("base", "ac9", "ko_at_valuation", "carried", "ki_at_valuation", "no_ki_barrier"),
    "phoenix": ("base", "ac9", "ko_at_valuation", "carried", "ki_at_valuation", "no_ki_barrier"),
    "ko_reset": ("base", "ac9", "ko_at_valuation", "carried", "ki_at_valuation"),
    "dcn": ("base", "ac9", "ko_at_valuation", "carried", "ki_at_valuation"),
}
ALL = [(e, c) for e, engine in ENGINES.items() for c in CASES_PER_FAMILY[engine.family]]


# --- AC8 ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine_name, case_name", ALL)
def test_partition_and_reconciliation_within_the_engine_bound(engine_name, case_name):
    _assert_partition(engine_name, _stats(engine_name, case_name))


# --- AC9 ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine_name", list(ENGINES))
def test_no_knock_in_after_a_certain_knock_out(engine_name):
    stats = _stats(engine_name, "ac9")
    first = int(np.argmax(stats.observations.probability > 1e-3))
    assert stats.observations.probability[first] == pytest.approx(1.0, abs=2e-5)
    bound = AC9_BOUNDS.get(engine_name, AC9_DEFAULT_BOUND)
    assert 0.0 <= stats.knock_in.probability <= bound, stats.knock_in.probability
    assert stats.ki_settlement.probability <= bound
    assert float(np.sum(stats.ki_table.probability)) <= bound * (1.0 + 1e-9)
    scale = 2.0 * (1e6 if ENGINES[engine_name].family == "dcn" else 100.0)        # above any knocked-in loss
    # expected loss = principal DF P(KIS) - KIS cash on the grid engines: a difference, -9.4e-16 on QUAD V1 KO-reset
    assert -1e-12 * scale <= stats.knock_in.expected_loss <= bound * scale, stats.knock_in.expected_loss
    if stats.knock_in.probability == 0.0:
        assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    # the barrier is really crossed: the same note with an ordinary knock-out knocks in on a large share of paths
    reachable = _stats(engine_name, "ac9_reachable")
    assert reachable.knock_in.probability > 0.25 and reachable.knock_in.expected_loss > 0.0


def test_the_ac9_knock_in_on_a_grid_is_far_edge_mass_that_vanishes():
    """The grid's AC9 knock-in is the mass on the truncated domain's far knock-in-side edge before the certain
    knock-out, not a knock-in after it: it equals 1 - Σ P(KO) and falls exponentially with the domain (local-vol PDE:
    num_std 4 / 6 / 8 -> 1.07e-5 / 4.9e-9 / 2.6e-13 measured) and with the ADI mesh (64x12x48 -> 128x24x96:
    2.1e-10 -> 4.8e-13 measured)."""
    from quantark.asset.equity.engine.pde.grid.config import GridConfig

    product, env = _knock_out_first("snowball", 1.0), routes.grid_env()
    by_width = []
    for width in (4.0, 6.0, 8.0):
        params = PDEParams(grid=GridConfig(num_std=width, points=int(100 * width)))
        stats = LocalVolSnowballPDESolver(params=params).calculate_event_stats(product, env)
        assert stats.knock_in.probability == pytest.approx(1.0 - float(np.sum(stats.observations.probability)),
                                                           abs=1e-9)
        by_width.append(stats.knock_in.probability)
    assert by_width[0] > 1e3 * by_width[1] > 1e6 * by_width[2]
    by_mesh = [HestonSnowballPDESolver(HP, **mesh).calculate_event_stats(product, routes.flat_env()).knock_in
               .probability for mesh in (ADI_MESH, dict(n_x=128, n_v=24, n_t=96))]
    assert by_mesh[0] > 100.0 * by_mesh[1]


# --- AC10 -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine_name", list(ENGINES))
def test_knocked_out_at_valuation_has_no_knock_in(engine_name):
    stats = _stats(engine_name, "ko_at_valuation")
    assert stats.observations.probability[0] == 1.0 and stats.observations.times[0] == 0.0
    assert np.all(stats.observations.probability[1:] == 0.0)
    assert stats.maturity.probability == 0.0
    assert stats.expected_lifetime == 0.0
    _assert_no_knock_in(stats)


@pytest.mark.parametrize("engine_name", list(ENGINES))
def test_a_carried_knock_in_is_a_knock_in_at_zero(engine_name):
    stats = _stats(engine_name, "carried")
    _assert_knock_in_at_zero(stats)
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert not np.any(stats.ki_table.interpolated)


@pytest.mark.parametrize("engine_name", list(ENGINES))
def test_a_knock_in_at_valuation_is_a_knock_in_at_zero(engine_name):
    _assert_knock_in_at_zero(_stats(engine_name, "ki_at_valuation"))


def _discrete_knock_in_observed_at_valuation(family: str):
    """A discrete KI with an observation at t = 0 and the spot below it (DESIGN §4.5: a knock-in at time 0), the
    knock-out out of reach at 0; the table must be the single row at 0 as for a continuous breach or a carried
    knock-in (T24 matrix finding: MC listed every schedule row with zeros after the first)."""
    ki_times = (0.0, *routes.KO_DATES)
    if family in ("snowball", "phoenix"):
        product = routes.snowball("discrete") if family == "snowball" else routes.phoenix(ki="discrete")
        old = product.barrier_config
        product.barrier_config = BarrierConfig(
            ko_barrier=old.ko_barrier, ko_rate=old.ko_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(routes.KO_DATES), ki_barrier=old.ki_barrier,
            ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times))
    else:
        product = routes.ko_reset()
        product.barrier_config = replace(product.barrier_config, ki_observation_schedule=_schedule(ki_times, 80.0))
    product.validate()
    # just below the KI level (snowball 75, phoenix 70, KO-reset 80): a far-down spot puts the coarse ADI mesh's
    # non-monotone splitting outside the type's slack (T14 open item), unrelated to the table
    return product, {"snowball": 73.0, "phoenix": 68.0, "ko_reset": 78.0}[family]


@pytest.mark.parametrize("engine_name", NON_DCN)
def test_a_discrete_knock_in_observed_at_valuation_is_one_row_at_zero(engine_name):
    engine = ENGINES[engine_name]
    product, spot = _discrete_knock_in_observed_at_valuation(engine.family)
    # the Phoenix ADI below its KI level needs the finer mesh, as for a continuous breach (ADI_MESH_PHOENIX)
    make = OVERRIDES.get((engine_name, "ki_at_valuation"), engine.make)
    stats = make().calculate_event_stats(product, _env(engine, spot))
    _assert_knock_in_at_zero(stats)
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.survival, [0.0])


@pytest.mark.parametrize("engine_name", [e for e in NON_DCN if ENGINES[e].family in NO_KI_BARRIER_FAMILIES])
def test_no_knock_in_barrier_reports_computed_zeros(engine_name):
    _assert_no_knock_in(_stats(engine_name, "no_ki_barrier"))


def test_the_mc_discrete_knock_in_row_at_valuation_carries_the_valuation_date():
    """A dated note valued on a discrete KI observation date with the spot through the barrier: the single row at 0
    carries that date (CP8 minor ii; the dates a dated table's rows carry elsewhere)."""
    from intraday.conftest import dated_snowball
    from quantark.util.calendar import CalendarType, create_calendar

    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    product = dated_snowball(calendar, datetime(2026, 3, 16))
    first = product.barrier_config.ki_observation_schedule.records[0].observation_date
    env = PricingEnvironment(spot_quote=SpotQuote(spot=70.0), vol_surface=FlatVolSurface(volatility=0.22),
                             rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                             valuation_date=first, calendar=calendar)
    stats = SnowballMCEngine(params=_mc()).calculate_event_stats(product, env)
    _assert_knock_in_at_zero(stats)
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.ki_table.dates == (first,)
