"""The at-valuation knock-in row carries the valuation date on a dated note, on every engine (CP8 fix 2 ruling).

A knock-in decided at valuation -- a discrete KI observed at valuation with the spot through it, a carried knock-in,
a continuous barrier already breached -- is the KI table's single row at t = 0. On a dated note that row's ``dates``
(and ``payment_dates``) are ``(valuation_date,)`` on MC, QUAD V2, QUAD V1, PDE 1D (flat and local vol), Heston ADI
and every DCN engine; a numeric note keeps ``dates=None``. One unified table: dates never differ by engine.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import (
    DCNMCEngine,
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    PhoenixMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    DCNPDEEngine,
    HestonDCNPDESolver,
    HestonKOResetSnowballPDESolver,
    HestonPhoenixPDESolver,
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
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, make_dcn
from intraday.conftest import dated_ko_reset, dated_phoenix, dated_snowball
from test_asset.equity.engine_mc import mc_price_path_routes as routes

warnings.filterwarnings("ignore", category=DeprecationWarning)

SSE = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
START = datetime(2026, 3, 16)
LATER = datetime(2026, 3, 20)                 # a trading day that is not an observation date
HP = routes.HESTON
ADI = dict(n_x=128, n_v=24, n_t=96)
DCN_HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
DCN_DAY = datetime(2024, 5, 15)

EQUITY_ENGINES = {
    "snowball": {"mc": lambda: SnowballMCEngine(params=MCParams(num_paths=256, seed=3)),
                 "quad_v2": SnowballQuadEngineV2, "quad_v1": lambda: SnowballQuadEngine(QuadParams()),
                 "pde_1d": lambda: SnowballPDESolver(PDEParams()),
                 "lv_pde_1d": lambda: LocalVolSnowballPDESolver(params=PDEParams()),
                 "heston_adi": lambda: HestonSnowballPDESolver(HP, **ADI)},
    "phoenix": {"mc": lambda: PhoenixMCEngine(params=MCParams(num_paths=256, seed=3)),
                "quad_v2": PhoenixQuadEngineV2, "quad_v1": lambda: PhoenixQuadEngine(QuadParams()),
                "pde_1d": lambda: PhoenixPDESolver(PDEParams()),
                "lv_pde_1d": lambda: LocalVolPhoenixPDESolver(params=PDEParams()),
                "heston_adi": lambda: HestonPhoenixPDESolver(HP, **ADI)},
    "ko_reset": {"mc": lambda: SnowballMCEngine(params=MCParams(num_paths=256, seed=3)),
                 "quad_v2": KOResetSnowballQuadEngineV2, "quad_v1": lambda: KOResetSnowballQuadEngine(QuadParams()),
                 "pde_1d": lambda: KOResetSnowballPDESolver(PDEParams()),
                 "lv_pde_1d": lambda: LocalVolKOResetSnowballPDESolver(params=PDEParams()),
                 "heston_adi": lambda: HestonKOResetSnowballPDESolver(HP, **ADI)}}
FAMILIES = {"snowball": dated_snowball, "phoenix": dated_phoenix, "ko_reset": dated_ko_reset}
CASES = ("discrete_breach", "carried", "continuous_breach")


def _equity_env(valuation, spot, grid):
    if grid:
        strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.4, 11)))
        surface = GridVolSurface(strikes, list(np.linspace(0.25, 2.0, 8)), np.full((8, 11), 0.22))
    else:
        surface = FlatVolSurface(volatility=0.22)
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=surface,
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=valuation, calendar=SSE)


def _equity_case(family, case, engine):
    """``(product, env)``: a dated note with its knock-in decided at valuation."""
    product = FAMILIES[family](SSE, START)
    if case == "discrete_breach":
        valuation, spot = product.barrier_config.ki_observation_schedule.records[0].observation_date, 60.0
    elif case == "carried":
        product._otc_lifecycle_knocked_in = True
        valuation, spot = LATER, 95.0
    else:
        product.barrier_config = replace(product.barrier_config, ki_continuous=True,
                                         ki_observation_type=ObservationType.CONTINUOUS,
                                         ki_observation_schedule=None, ki_observation_dates=None)
        valuation, spot = LATER, 60.0
    return product, _equity_env(valuation, spot, grid=engine == "lv_pde_1d")


EQUITY = [(f, c, e) for f in FAMILIES for c in CASES for e in EQUITY_ENGINES[f]]


@pytest.mark.parametrize("family, case, engine", EQUITY)
def test_the_at_valuation_ki_row_carries_the_valuation_date(family, case, engine):
    product, env = _equity_case(family, case, engine)
    stats = EQUITY_ENGINES[family][engine]().calculate_event_stats(product, env)
    table = stats.ki_table
    np.testing.assert_array_equal(table.times, [0.0])
    assert stats.knock_in.probability == 1.0
    assert table.dates == (env.valuation_date,), (engine, table.dates)
    assert table.payment_dates == (env.valuation_date,)


def _dcn_env(spot, grid=False):
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        surface = GridVolSurface(strikes=strikes, maturities=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
                                 iv_grid=np.full((6, 5), 0.184))
    else:
        surface = FlatVolSurface(volatility=0.184)
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=surface,
                              rate_curve=FlatRateCurve(rate=0.0356), div_yield=ContinuousDividendYield(0.1406),
                              valuation_date=DCN_DAY, day_count_convention=DayCountConvention.ACT_365)


DCN_ENGINES = {
    "mc": (lambda: DCNMCEngine(num_paths=256, seed=3, use_sobol=False, num_workers=1), False),
    "lv_mc": (lambda: LocalVolDCNMCEngine(num_paths=256, seed=3, use_sobol=False, num_workers=1), True),
    "heston_mc": (lambda: HestonDCNMCEngine(model_params=DCN_HESTON, num_paths=256, seed=3, use_sobol=False,
                                            num_workers=1), False),
    "pde": (lambda: DCNPDEEngine(PDEParams(accuracy="fast")), False),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")), True),
    "heston_adi": (lambda: HestonDCNPDESolver(DCN_HESTON, n_x=61, n_v=21), False),
}


@pytest.mark.parametrize("case", ["breach", "carried"])
@pytest.mark.parametrize("engine", list(DCN_ENGINES))
def test_the_dcn_at_valuation_ki_row_carries_the_valuation_date(engine, case):
    make, grid = DCN_ENGINES[engine]
    product = make_dcn(DCN_A, valuation_date=DCN_DAY, knocked_in_at_valuation=case == "carried")
    stats = make().calculate_event_stats(product, _dcn_env(4400.0 if case == "breach" else 5000.0, grid))
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.ki_table.dates == (DCN_DAY,), (engine, stats.ki_table.dates)


@pytest.mark.parametrize("engine", ["mc", "quad_v2", "quad_v1", "pde_1d"])
def test_a_numeric_note_keeps_no_dates_on_the_at_valuation_row(engine):
    stats = EQUITY_ENGINES["snowball"][engine]().calculate_event_stats(routes.snowball(knocked_in=True),
                                                                      routes.flat_env())
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    assert stats.ki_table.dates is None
