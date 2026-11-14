"""The full cross-engine matrix of the unified event statistics (spec AC20 full matrix, AC21; DESIGN §5; T24).

Every DESIGN §5 contract, priced in every model group that serves it, by every engine of that group, every
``EventStatistics`` number against the group's MC reference:

- a *contract* (``CONTRACTS``) is a product, a spot, optionally a lifecycle state, and the model groups that price it;
- a *model group* (``GROUPS``) is a market, the MC reference engine (``mc(seed)``: PSEUDO, one independent seed per
  run, spec AC21), the grid engines at three levels (``L0`` gated, ``L1``/``L2`` the refinement evidence of the banked
  allowance) and an *alternative MC* (RQMC or batched PSEUDO, its own batch-mean standard errors: the engine's AC21
  errors are what its gate uses);
- references refine their stepping where the defaults are biased (plan rulings): local-vol MC
  ``max_sde_step_days=1``, Heston / SLV MC ``substeps_per_interval=32`` (DCN Heston 4 per daily interval).

The DESIGN §5 list and where each item lives (``CONTRACTS`` key prefix):

==============================================  ====================================================================
continuous / discrete KI                        ``sb_cki`` / ``sb_dki`` (and every family)
memory / non-memory Phoenix                     ``ph_mem_*`` / ``ph_nomem_*``
INSTANT / EXPIRY coupon pay                     ``*_instant_*`` / ``*_expiry_*`` (Snowball, Phoenix, KO-reset)
principal included / excluded                   every note / ``sb_no_principal``
protection none / partial / full                every note / ``sb_protection_partial`` / ``sb_protection_full``
already knocked in                              ``*_carried``
knocked out at valuation                        ``*_ko_at_valuation``
KI observed at valuation                        ``sb_ki_observed_at_valuation`` (discrete row at t = 0, spot below);
                                                ``*_ki_at_valuation`` (continuous, spot through the barrier)
reverse (call-style) note                       ``sb_reverse``
KO-reset ABSOLUTE / REBASED (MC only)           ``kr_abs_*`` / ``kr_rebased`` (reference vs the RQMC alternative; an
                                                already-knocked-in REBASED note is refused by the MC: the REBASED
                                                schedule needs the knock-in time a carried knock-in does not hold)
DCN non-SSE run calendar                        ``dcn_us_calendar`` (US schedule), ``dcn_sse_holiday`` (valued on
                                                an SSE holiday)
DCN ``ko_lock_months > lock_months``            ``dcn_ko_lock``
Phoenix non-zero ``ko_rate``                    ``ph_ko_rate``
no KI barrier                                   ``sb_no_ki``, ``ph_no_ki``
numeric-clock contract                          every equity note (float observation times); DCNs are dated
seller DCN                                      ``dcn_seller``
aged trade with receivables / terminated        ``sb_receivable`` (QUAD V2), ``dcn_receivable``;
unpaid (engines accepting ``lifecycle_state``)  ``sb_terminated_unpaid`` (QUAD V2), ``dcn_terminated_unpaid``
==============================================  ====================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from quantark.asset.equity.engine.mc import (
    HestonDCNMCEngine,
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
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
    DCNMCEngine,
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
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, DCNLifecycleState, ValuationPoint
from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, RealizedCashflow
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams, QuadV2Params
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.enum.option_enums import PostKOScheduleMode, ProtectionType
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, DCN_B, FLAT, make_dcn
from event_statistics.cases import cdf_rows, fields_of
from test_asset.equity.engine_mc import mc_price_path_routes as routes

__all__ = ["CONTRACTS", "GROUPS", "ENTRIES", "LEVELS", "MC_PATHS", "SEEDS", "Contract", "Group", "Reference",
           "alt_statistics", "engine_statistics", "grid_engines", "has_alt", "matrix_fields_of", "mc_reference",
           "pool_runs"]

#: independent PSEUDO runs of the reference and their paths (as the smoke: 16 x 8192)
SEEDS = tuple(range(101, 117))
MC_PATHS = 8192
#: the alternative MC: batches x paths per batch (RQMC: distinct scrambles; PSEUDO DCN: distinct streams)
ALT_BATCHES = 8
ALT_PATHS = 8192
LEVELS = ("L0", "L1", "L2")
HP = routes.HESTON
DCN_HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
#: Heston / SLV MC references: QE substeps per contractual interval (one step per interval biases a quarterly
#: digital by ~8.5 SE: plan note); DCN Heston: substeps per daily interval
HESTON_SUBSTEPS = 32
DCN_HESTON_SUBSTEPS = 4
RQMC = MonteCarloMethod.RANDOMIZED_QUASI


# --- engines at levels ----------------------------------------------------------------------------------------------

def _pde(cls, lv=False):
    def at(points, steps):
        params = PDEParams() if points is None else PDEParams(grid=GridConfig(points=points, steps_per_day=steps))
        return (lambda: cls(params=params)) if lv else (lambda: cls(params))
    return {"L0": at(None, None), "L1": at(800, 8.0), "L2": at(1600, 16.0)}


def _quad_v2(cls):
    return {"L0": lambda: cls(QuadV2Params()), "L1": lambda: cls(QuadV2Params(order=10, cells_per_sd=3.0)),
            "L2": lambda: cls(QuadV2Params(order=12, cells_per_sd=4.0))}


def _quad_v1(cls):
    return {level: (lambda g=g: cls(QuadParams(grid_points=g)))
            for level, g in (("L0", 1001), ("L1", 4001), ("L2", 16001))}


def _adi(cls, leverage=None, phoenix=False):
    def at(n_x, n_v, n_t):
        extra = {} if leverage is None else {"leverage_surface": leverage()}
        return lambda: cls(HP, n_x=n_x, n_v=n_v, n_t=n_t, **extra)
    return {"L0": at(120, 32, 200), "L1": at(240, 64, 400), "L2": at(480, 128, 800)}


def _dcn_pde(lv=False):
    def at(accuracy):
        return (lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy=accuracy))) if lv else \
            (lambda: DCNPDEEngine(PDEParams(accuracy=accuracy)))
    return {"L0": at("fast"), "L1": at("standard"), "L2": at("high")}


def _dcn_adi():
    return {"L0": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=61, n_v=21),
            "L1": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=121, n_v=41, substeps_per_interval=2),
            "L2": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=241, n_v=81, substeps_per_interval=4)}


def _continuous_ki(product) -> bool:
    barrier = getattr(product, "barrier_config", None)
    return barrier is not None and bool(barrier.ki_continuous)


def _vol_alt_substeps(product) -> int:
    """The vol-model RQMC alternative's substeps per contractual interval: RQMC cannot cap the step in days
    (``max_sde_step_days`` is PSEUDO only) and Sobol caps the dimension at 21201, so a continuous knock-in (daily
    intervals) takes 4 substeps a day, a discrete one 32 per interval (about two trading days a step)."""
    return 4 if _continuous_ki(product) else HESTON_SUBSTEPS


def _pseudo(seed):
    return MCParams(num_paths=MC_PATHS, seed=seed)


def _rqmc(seed=11):
    """RQMC with exactly ``ALT_BATCHES`` independent scrambles (the target is unreachable)."""
    return MCParams(num_paths=ALT_PATHS, seed=seed, rqmc_min_batches=ALT_BATCHES, rqmc_max_batches=ALT_BATCHES,
                    rqmc_target_std=1e-12, rqmc_paths_mode="per_batch")


# --- model groups ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Group:
    """A market and its engines per family: ``mc[family](seed)`` the reference run, ``alt[family](product)`` the alternative
    MC (own batch errors), ``grid[family]`` ``{engine: {level: factory}}``."""
    env: Callable[..., PricingEnvironment]
    mc: Mapping[str, Callable[[int], object]]
    alt: Mapping[str, Callable[[], object]]
    grid: Mapping[str, Mapping[str, Mapping[str, Callable[[], object]]]]


def _dcn_env(day, spot=6000.0, grid=False, calendar=None):
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = FLAT["sigma"] + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        surface = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        surface = FlatVolSurface(volatility=FLAT["sigma"])
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=surface,
                              rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(FLAT["q"]),
                              valuation_date=day, day_count_convention=DayCountConvention.ACT_365, calendar=calendar)


def _dcn_mc(cls, **kw):
    return lambda seed: cls(num_paths=MC_PATHS, seed=seed, use_sobol=False, num_workers=1, **kw)


def _dcn_alt(cls, **kw):
    return lambda p, seed=11: cls(num_paths=ALT_PATHS * ALT_BATCHES, seed=seed, use_sobol=False, num_batches=ALT_BATCHES,
                       num_workers=1, **kw)


GROUPS: Dict[str, Group] = {
    "bs": Group(
        env=lambda spot=100.0, **_: routes.flat_env(spot),
        mc={"snowball": lambda s: SnowballMCEngine(params=_pseudo(s)),
            "ko_reset": lambda s: SnowballMCEngine(params=_pseudo(s)),
            "phoenix": lambda s: PhoenixMCEngine(params=_pseudo(s))},
        alt={"snowball": lambda p, seed=11: SnowballMCEngine(params=_rqmc(seed), method=RQMC),
             "ko_reset": lambda p, seed=11: SnowballMCEngine(params=_rqmc(seed), method=RQMC),
             "phoenix": lambda p, seed=11: PhoenixMCEngine(params=_rqmc(seed), method=RQMC)},
        grid={"snowball": {"pde_1d": _pde(SnowballPDESolver), "quad_v2": _quad_v2(SnowballQuadEngineV2),
                           "quad_v1": _quad_v1(SnowballQuadEngine)},
              "phoenix": {"pde_1d": _pde(PhoenixPDESolver), "quad_v2": _quad_v2(PhoenixQuadEngineV2),
                          "quad_v1": _quad_v1(PhoenixQuadEngine)},
              "ko_reset": {"pde_1d": _pde(KOResetSnowballPDESolver), "quad_v2": _quad_v2(KOResetSnowballQuadEngineV2),
                           "quad_v1": _quad_v1(KOResetSnowballQuadEngine)},
              "ko_reset_rebased": {}}),
    "lv": Group(
        env=lambda spot=100.0, **_: routes.grid_env(spot),
        mc={"snowball": lambda s: LocalVolSnowballMCEngine(_pseudo(s), max_sde_step_days=1.0),
            "ko_reset": lambda s: LocalVolSnowballMCEngine(_pseudo(s), max_sde_step_days=1.0),
            "phoenix": lambda s: LocalVolPhoenixMCEngine(_pseudo(s), max_sde_step_days=1.0)},
        alt={"snowball": lambda p, seed=11: LocalVolSnowballMCEngine(_rqmc(seed), method=RQMC,
                                                            substeps_per_interval=_vol_alt_substeps(p)),
             "ko_reset": lambda p, seed=11: LocalVolSnowballMCEngine(_rqmc(seed), method=RQMC,
                                                            substeps_per_interval=_vol_alt_substeps(p)),
             "phoenix": lambda p, seed=11: LocalVolPhoenixMCEngine(_rqmc(seed), method=RQMC,
                                                          substeps_per_interval=_vol_alt_substeps(p))},
        grid={"snowball": {"lv_pde_1d": _pde(LocalVolSnowballPDESolver, lv=True)},
              "phoenix": {"lv_pde_1d": _pde(LocalVolPhoenixPDESolver, lv=True)},
              "ko_reset": {"lv_pde_1d": _pde(LocalVolKOResetSnowballPDESolver, lv=True)}}),
    "heston": Group(
        env=lambda spot=100.0, **_: routes.flat_env(spot),
        mc={"snowball": lambda s: HestonSnowballMCEngine(HP, params=_pseudo(s), substeps_per_interval=HESTON_SUBSTEPS),
            "ko_reset": lambda s: HestonSnowballMCEngine(HP, params=_pseudo(s), substeps_per_interval=HESTON_SUBSTEPS),
            "phoenix": lambda s: HestonPhoenixMCEngine(HP, params=_pseudo(s), substeps_per_interval=HESTON_SUBSTEPS)},
        alt={"snowball": lambda p, seed=11: QESnowballMCEngine(HP, params=_rqmc(seed), method=RQMC,
                                                      substeps_per_interval=_vol_alt_substeps(p)),
             "ko_reset": lambda p, seed=11: QESnowballMCEngine(HP, params=_rqmc(seed), method=RQMC,
                                                      substeps_per_interval=_vol_alt_substeps(p)),
             "phoenix": lambda p, seed=11: QEPhoenixMCEngine(HP, params=_rqmc(seed), method=RQMC,
                                                    substeps_per_interval=_vol_alt_substeps(p))},
        grid={"snowball": {"heston_adi": _adi(HestonSnowballPDESolver)},
              "phoenix": {"heston_adi": _adi(HestonPhoenixPDESolver)},
              "ko_reset": {"heston_adi": _adi(HestonKOResetSnowballPDESolver)}}),
    "slv": Group(
        env=lambda spot=100.0, **_: routes.grid_env(spot, skew=False),
        mc={"snowball": lambda s: HestonSLVSnowballMCEngine(HP, params=_pseudo(s), leverage_surface=routes.skew_leverage(),
                                                            substeps_per_interval=HESTON_SUBSTEPS),
            "ko_reset": lambda s: HestonSLVSnowballMCEngine(HP, params=_pseudo(s), leverage_surface=routes.skew_leverage(),
                                                            substeps_per_interval=HESTON_SUBSTEPS),
            "phoenix": lambda s: HestonSLVPhoenixMCEngine(HP, params=_pseudo(s), leverage_surface=routes.skew_leverage(),
                                                          substeps_per_interval=HESTON_SUBSTEPS)},
        alt={},
        grid={"snowball": {"slv_adi": _adi(HestonSLVSnowballPDESolver, leverage=routes.skew_leverage)},
              "phoenix": {"slv_adi": _adi(HestonSLVPhoenixPDESolver, leverage=routes.skew_leverage)},
              "ko_reset": {"slv_adi": _adi(HestonSLVKOResetSnowballPDESolver, leverage=routes.skew_leverage)}}),
    "dcn_bs": Group(
        env=lambda spot=6000.0, day=None, calendar=None, **_: _dcn_env(day, spot, calendar=calendar),
        mc={"dcn": _dcn_mc(DCNMCEngine)},
        alt={"dcn": _dcn_alt(DCNMCEngine)},
        grid={"dcn": {"dcn_pde": _dcn_pde()}}),
    "dcn_lv": Group(
        env=lambda spot=6000.0, day=None, calendar=None, **_: _dcn_env(day, spot, grid=True, calendar=calendar),
        mc={"dcn": _dcn_mc(LocalVolDCNMCEngine)},
        alt={"dcn": _dcn_alt(LocalVolDCNMCEngine)},
        grid={"dcn": {"dcn_lv_pde": _dcn_pde(lv=True)}}),
    "dcn_heston": Group(
        env=lambda spot=6000.0, day=None, calendar=None, **_: _dcn_env(day, spot, calendar=calendar),
        mc={"dcn": _dcn_mc(HestonDCNMCEngine, model_params=DCN_HESTON, substeps_per_interval=DCN_HESTON_SUBSTEPS)},
        alt={"dcn": _dcn_alt(QEDCNMCEngine, model_params=DCN_HESTON, substeps_per_interval=DCN_HESTON_SUBSTEPS)},
        grid={"dcn": {"dcn_heston_adi": _dcn_adi()}}),
}


# --- contracts ------------------------------------------------------------------------------------------------------

def _schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def _with_payoff(product, **changes):
    product.payoff_config = replace(product.payoff_config, **changes)
    product.validate()
    return product


def _ki_observed_at_valuation():
    """Discrete KI at 75 observed at t = 0 and on every knock-out row; the spot (72) is below it."""
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(routes.KO_DATES), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.0, *routes.KO_DATES]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def _no_ki_snowball():
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(routes.KO_DATES), ki_barrier=None),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def _phoenix_ko_rate():
    product = routes.phoenix()
    # a fresh BarrierConfig: ``replace`` would carry the derived observation schedule (and its 0 KO rate) over
    product.barrier_config = BarrierConfig(
        ko_barrier=105.0, ko_rate=0.06, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(routes.KO_DATES), ki_barrier=70.0, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True)
    product.validate()
    return product


def _no_ki_phoenix():
    product = routes.phoenix()
    product.barrier_config = BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                           ko_observation_dates=list(routes.KO_DATES), ki_barrier=None)
    product.validate()
    return product


def _ko_reset_ko_at_valuation():
    product = routes.ko_reset()
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=_schedule(
        (0.0, 0.25, 0.5, 0.75, 1.0), 103.0))
    product.validate()
    return product


# DCN dates (DCN_A: 2023-01-03 .. 2025-01-03, monthly rows on the 3rd)
DCN_DAY = datetime(2024, 5, 15)           # a trading day that is not a monthly row
DCN_OBS = datetime(2024, 12, 3)           # a coupon + knock-out row
DCN_SSE_HOLIDAY = datetime(2024, 6, 10)   # SSE closed (Dragon Boat), a US / weekday run day
DCN_LOCK = datetime(2023, 3, 15)          # DCN_B inside its knock-out lock (ko_lock_months 6 > lock_months 3)
DCN_ROW, DCN_ROW_PAY, DCN_AGED = datetime(2024, 6, 3), datetime(2024, 6, 5), datetime(2024, 6, 4)
DCN_CASH = 10_000.0
US = create_calendar(CalendarType.US)


def _dcn_us_calendar():
    """DCN_A on a US schedule calendar (monthly rows rolled by US holidays, the daily KI on US trading days)."""
    from quantark.asset.equity.product.option.dcn_option import DCNOption
    from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
    from dcn_fixtures import PRODUCT_KEYS, SCHEDULE_KEYS

    c = dict(DCN_A, valuation_date=DCN_DAY)
    schedule = build_dcn_schedule(calendar=US, **{k: c[k] for k in SCHEDULE_KEYS})
    return DCNOption(direction=DCNDirection.BUYER, schedule=schedule, knocked_in_at_valuation=False,
                     **{k: c[k] for k in PRODUCT_KEYS})


def _dcn_flow(kind, determined, paid, amount=DCN_CASH):
    event = {"coupon": LifecycleEventType.COUPON, "ko": LifecycleEventType.KNOCK_OUT}[kind]
    return RealizedCashflow(cashflow_id=f"dcn:{kind}:{determined:%Y-%m-%d}", event_type=event, amount=amount,
                            determination_date=determined, payment_date=paid)


def _dcn_receivable_state():
    return DCNLifecycleState(valuation_point=ValuationPoint(date=DCN_AGED),
                             ledger=LifecycleCashflowLedger([_dcn_flow("coupon", DCN_ROW, DCN_ROW_PAY)]))


def _dcn_terminated_state():
    return DCNLifecycleState(valuation_point=ValuationPoint(date=DCN_AGED),
                             ledger=LifecycleCashflowLedger([_dcn_flow("ko", DCN_ROW, DCN_ROW_PAY)]),
                             knocked_out=True, ko_date=DCN_ROW)


def _equity_receivable_state():
    state = AutocallableLifecycleState(valuation_point=ValuationPoint(date=routes.VALUATION))
    state.ledger.register(RealizedCashflow(
        cashflow_id="coupon-pending", event_type=LifecycleEventType.COUPON, amount=2.0,
        determination_date=routes.VALUATION - timedelta(days=5), payment_date=routes.VALUATION + timedelta(days=10)))
    return state


def _equity_terminated_state():
    state = AutocallableLifecycleState()
    state.mark_ko(routes.VALUATION - timedelta(days=3), cashflow=105.0,
                  settlement_date=routes.VALUATION + timedelta(days=10),
                  valuation_point=ValuationPoint(date=routes.VALUATION))
    return state


@dataclass(frozen=True)
class Contract:
    """One DESIGN §5 contract: ``family`` keys the group's engines; ``env`` keyword arguments to the group market;
    ``lifecycle`` a lifecycle-state factory (passed only to engines that accept it: ``lifecycle_engines``)."""
    family: str
    product: Callable[[], object]
    groups: Tuple[str, ...]
    env: Mapping[str, object] = field(default_factory=dict)
    lifecycle: Optional[Callable[[], object]] = None
    lifecycle_engines: Tuple[str, ...] = ()
    terminated: bool = False


EQ_ALL = ("bs", "lv", "heston", "slv")
EQ_VOL = ("bs", "lv", "heston")
DCN_ALL = ("dcn_bs", "dcn_lv", "dcn_heston")
EXPIRY = CouponPayType.EXPIRY

CONTRACTS: Dict[str, Contract] = {
    # Snowball
    "sb_cki": Contract("snowball", routes.snowball, EQ_ALL),
    "sb_dki": Contract("snowball", lambda: routes.snowball("discrete"), EQ_ALL),
    "sb_cki_expiry": Contract("snowball", lambda: routes.snowball(pay=EXPIRY), ("bs",)),
    "sb_reverse": Contract("snowball", lambda: routes.snowball(reverse=True), EQ_VOL),
    "sb_no_principal": Contract("snowball", lambda: _with_payoff(routes.snowball(), include_principal=False), ("bs",)),
    "sb_protection_partial": Contract(
        "snowball", lambda: _with_payoff(routes.snowball(), protection_type=ProtectionType.PARTIAL,
                                         protection_rate=0.10), ("bs",)),
    "sb_protection_full": Contract(
        "snowball", lambda: _with_payoff(routes.snowball(), protection_type=ProtectionType.FULL), ("bs",)),
    "sb_carried": Contract("snowball", lambda: routes.snowball(knocked_in=True), EQ_VOL),
    "sb_ko_at_valuation": Contract("snowball", lambda: routes.snowball(ko_dates=(0.0, *routes.KO_DATES)), EQ_VOL,
                                   env={"spot": 110.0}),
    "sb_ki_at_valuation": Contract("snowball", routes.snowball, EQ_VOL, env={"spot": 70.0}),
    "sb_ki_observed_at_valuation": Contract("snowball", _ki_observed_at_valuation, EQ_VOL, env={"spot": 72.0}),
    "sb_no_ki": Contract("snowball", _no_ki_snowball, EQ_VOL),
    "sb_receivable": Contract("snowball", routes.snowball, ("bs",), lifecycle=_equity_receivable_state,
                              lifecycle_engines=("quad_v2",)),
    "sb_terminated_unpaid": Contract("snowball", routes.snowball, ("bs",), lifecycle=_equity_terminated_state,
                                     lifecycle_engines=("quad_v2",), terminated=True),
    # Phoenix
    "ph_mem_instant_cki": Contract("phoenix", routes.phoenix, EQ_ALL),
    "ph_mem_expiry_dki": Contract("phoenix", lambda: routes.phoenix(pay=EXPIRY, ki="discrete"), EQ_VOL),
    "ph_nomem_instant_cki": Contract("phoenix", lambda: routes.phoenix(memory=False), ("bs",)),
    "ph_nomem_expiry_dki": Contract("phoenix", lambda: routes.phoenix(memory=False, pay=EXPIRY, ki="discrete"),
                                    ("bs",)),
    "ph_ko_rate": Contract("phoenix", _phoenix_ko_rate, ("bs",)),
    "ph_carried": Contract("phoenix", lambda: routes.phoenix(knocked_in=True), EQ_VOL),
    "ph_ko_at_valuation": Contract("phoenix", lambda: routes.phoenix(ko_dates=(0.0, *routes.KO_DATES)), ("bs",),
                                   env={"spot": 110.0}),
    "ph_ki_at_valuation": Contract("phoenix", routes.phoenix, ("bs",), env={"spot": 65.0}),
    "ph_no_ki": Contract("phoenix", _no_ki_phoenix, ("bs",)),
    # KO-reset
    "kr_abs_dki": Contract("ko_reset", routes.ko_reset, EQ_ALL),
    "kr_abs_cki": Contract("ko_reset", lambda: routes.ko_reset(ki="continuous"), ("bs", "heston")),
    "kr_abs_expiry": Contract("ko_reset", lambda: routes.ko_reset(pay=EXPIRY), ("bs",)),
    "kr_carried": Contract("ko_reset", lambda: routes.ko_reset(knocked_in=True), ("bs",)),
    "kr_ko_at_valuation": Contract("ko_reset", _ko_reset_ko_at_valuation, ("bs",), env={"spot": 110.0}),
    "kr_ki_at_valuation": Contract("ko_reset", lambda: routes.ko_reset(ki="continuous"), ("bs",),
                                   env={"spot": 75.0}),
    "kr_rebased": Contract("ko_reset_rebased", lambda: routes.ko_reset(PostKOScheduleMode.REBASED), ("bs",)),
    # DCN
    "dcn_base": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_DAY), DCN_ALL, env={"day": DCN_DAY}),
    "dcn_us_calendar": Contract("dcn", _dcn_us_calendar, ("dcn_bs", "dcn_heston"),
                                env={"day": DCN_DAY, "calendar": US}),
    "dcn_sse_holiday": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_SSE_HOLIDAY), ("dcn_bs", "dcn_lv"),
                                env={"day": DCN_SSE_HOLIDAY, "spot": 5500.0}),
    "dcn_ko_lock": Contract("dcn", lambda: make_dcn(DCN_B, valuation_date=DCN_LOCK), ("dcn_bs", "dcn_heston"),
                            env={"day": DCN_LOCK}),
    "dcn_seller": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_DAY, direction=DCNDirection.SELLER),
                           ("dcn_bs", "dcn_heston"), env={"day": DCN_DAY}),
    "dcn_carried": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_DAY, knocked_in_at_valuation=True),
                            ("dcn_bs",), env={"day": DCN_DAY, "spot": 5000.0}),
    "dcn_ko_at_valuation": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_OBS), ("dcn_bs",),
                                    env={"day": DCN_OBS, "spot": 6300.0}),
    "dcn_ki_at_valuation": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_DAY), ("dcn_bs",),
                                    env={"day": DCN_DAY, "spot": 4400.0}),
    "dcn_receivable": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_AGED), ("dcn_bs", "dcn_heston"),
                               env={"day": DCN_AGED}, lifecycle=_dcn_receivable_state,
                               lifecycle_engines=("mc", "alt", "dcn_pde", "dcn_heston_adi")),
    "dcn_terminated_unpaid": Contract("dcn", lambda: make_dcn(DCN_A, valuation_date=DCN_AGED), ("dcn_bs",),
                                      env={"day": DCN_AGED}, lifecycle=_dcn_terminated_state,
                                      lifecycle_engines=("mc", "alt", "dcn_pde"), terminated=True),
}

def _one_knock_in_path_snowball():
    """A knock-in at 55 observed only at maturity: under 4 RQMC scrambles x 128 paths (seed 1) exactly one path knocks
    in before a knock-out (spec AC21: E[KI time] defined, its standard error undefined)."""
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(routes.KO_DATES), ki_barrier=55.0,
                                     ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=[1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


#: the engine-side AC21 case: product, paths per RQMC batch, seed
ONE_KNOCK_IN_PATH = {"product": _one_knock_in_path_snowball, "paths": 128, "seed": 1}

#: (contract, group) pairs of the matrix
ENTRIES = [(name, group) for name, contract in CONTRACTS.items() for group in contract.groups]


def market(name: str, group: str):
    contract = CONTRACTS[name]
    return contract.product(), GROUPS[group].env(**contract.env)


def _kwargs(contract: Contract, engine: str):
    if contract.lifecycle is not None and engine in contract.lifecycle_engines:
        return {"lifecycle_state": contract.lifecycle()}
    return {}


def engine_statistics(name: str, group: str, engine: str, level: str = "L0"):
    contract = CONTRACTS[name]
    product, env = market(name, group)
    make = GROUPS[group].grid[contract.family][engine][level]
    return make().calculate_event_stats(product, env, **_kwargs(contract, engine))


def alt_statistics(name: str, group: str):
    contract = CONTRACTS[name]
    product, env = market(name, group)
    family = "ko_reset" if contract.family == "ko_reset_rebased" else contract.family
    return GROUPS[group].alt[family](product).calculate_event_stats(product, env, **_kwargs(contract, "alt"))


def grid_engines(name: str, group: str) -> Tuple[str, ...]:
    """The group's grid engines for the contract; on a lifecycle contract only those receiving the state (the others
    would price the plain contract, another entry)."""
    contract = CONTRACTS[name]
    engines = tuple(GROUPS[group].grid.get(contract.family, {}))
    if contract.lifecycle is not None:
        engines = tuple(e for e in engines if e in contract.lifecycle_engines)
    return engines


def has_alt(name: str, group: str) -> bool:
    contract = CONTRACTS[name]
    family = "ko_reset" if contract.family == "ko_reset_rebased" else contract.family
    if contract.lifecycle is not None and "alt" not in contract.lifecycle_engines:
        return False
    return family in GROUPS[group].alt


# --- the numbers compared -------------------------------------------------------------------------------------------

def matrix_fields_of(stats) -> Dict[str, np.ndarray]:
    """The smoke's numbers (``cases.fields_of``) and every other number of the contract: the survival columns
    (observations, coupons, knock-in table at the compared rows), the cash totals, the receivables, and on REBASED
    KO-reset the post-KI offset table and the termination distribution."""
    out = fields_of(stats)
    out["observations.survival"] = np.asarray(stats.observations.survival, dtype=float)
    if stats.coupons is not None:
        out["coupons.survival"] = np.asarray(stats.coupons.survival, dtype=float)
    table = stats.ki_table
    out["ki_survival"] = np.asarray(table.survival, dtype=float)[cdf_rows(table.size)]
    out["total_discounted_cash"] = np.array([float(stats.total_discounted_cash)])
    out["total_cash_to_pay"] = np.array([float(stats.total_cash_to_pay)])
    out["knock_in.probability_no_ko"] = np.array([float(stats.knock_in.probability_no_ko)])
    out["receivables.discounted_cash"] = np.asarray(stats.receivables.discounted_cash, dtype=float)
    out["receivables.cash_to_pay"] = np.asarray(stats.receivables.cash_to_pay, dtype=float)
    if stats.post_ki_offsets is not None:
        offsets = stats.post_ki_offsets
        out["post_ki_offsets.offsets"] = np.asarray(offsets.offsets, dtype=float)
        for column in ("probability", "discounted_cash", "cash_to_pay"):
            out[f"post_ki_offsets.{column}"] = np.asarray(getattr(offsets, column), dtype=float)
    if stats.termination is not None:
        term = stats.termination
        out["termination.times"] = np.asarray(term.times, dtype=float)
        for column in ("probability", "discounted_cash", "cash_to_pay"):
            out[f"termination.{column}"] = np.asarray(getattr(term, column), dtype=float)
    return out


#: axes: equal across engines (asserted), never gated
AXES = ("ki_table.times", "post_ki_offsets.offsets", "termination.times")
#: deterministic numbers (no MC error): gated at round-off across engines
DETERMINISTIC = ("receivables.discounted_cash", "receivables.cash_to_pay")


@dataclass(frozen=True)
class Reference:
    """The MC reference: per number the mean over the independent runs and its standard error (run spread /
    sqrt(runs)); the expected knock-in time a pooled ratio with a delta-method error, ``None`` with an
    ``"undefined:"`` reason under two pooled knock-in paths (spec AC21)."""
    mean: Dict[str, np.ndarray]
    se: Dict[str, Optional[np.ndarray]]
    runs: int
    paths: int
    knock_in_paths: float
    unavailable: Dict[str, str]


def pool_runs(runs, paths_per_run: int) -> Reference:
    """The reference from independent runs' ``matrix_fields_of`` (equal path counts)."""
    k = len(runs)
    if k < 2:
        raise ValueError("an MC reference needs at least two independent runs")
    mean, se, unavailable = {}, {}, {}
    for path in runs[0]:
        if path == "knock_in.expected_time":
            continue
        if path in AXES:
            for r in runs[1:]:
                np.testing.assert_array_equal(r[path], runs[0][path])
            mean[path] = runs[0][path]
            continue
        stack = np.vstack([r[path] for r in runs])
        mean[path] = stack.mean(axis=0)
        se[path] = stack.std(axis=0, ddof=1) / math.sqrt(k)
    probability = np.array([float(r["knock_in.probability"][0]) for r in runs])
    times = np.array([float(r.get("knock_in.expected_time", [0.0])[0]) for r in runs])
    knock_in_paths = float(probability.sum() * paths_per_run)
    if probability.sum() > 0.0:
        weighted = probability * times
        ratio = float(weighted.sum() / probability.sum())
        mean["knock_in.expected_time"] = np.array([ratio])
        if knock_in_paths >= 2.0:
            # delta method on the ratio of the runs' sums: SE^2 = k / (k - 1) sum_b (N_b - R D_b)^2 / (sum_b D_b)^2
            residual = weighted - ratio * probability
            se["knock_in.expected_time"] = np.array(
                [math.sqrt(k / (k - 1) * float(np.sum(residual ** 2))) / float(probability.sum())])
        else:
            se["knock_in.expected_time"] = None
            unavailable["knock_in.expected_time"] = "undefined: fewer than two knock-in paths"
    else:
        unavailable["knock_in.expected_time"] = "undefined: P(knock in) = 0"
    return Reference(mean=mean, se=se, runs=k, paths=k * paths_per_run, knock_in_paths=knock_in_paths,
                     unavailable=unavailable)


def mc_reference(name: str, group: str, seeds: Tuple[int, ...] = SEEDS) -> Reference:
    contract = CONTRACTS[name]
    family = "ko_reset" if contract.family == "ko_reset_rebased" else contract.family
    runs = []
    for seed in seeds:
        product, env = market(name, group)
        stats = GROUPS[group].mc[family](seed).calculate_event_stats(product, env, **_kwargs(contract, "mc"))
        runs.append(matrix_fields_of(stats))
    return pool_runs(runs, MC_PATHS)
