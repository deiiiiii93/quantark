from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay import AutocallableEngineConfig
from quantark.backtest.replay.market import derive_implied_dividend_yield
from quantark.backtest.simulation.carry import carry_at, day_chain
from quantark.backtest.simulation.dividends import CurveTailPillars, dividend_yield_for_day, legacy_implied_q
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, make_market_path


def _chain_and_path():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    return mp, day_chain(mp, 0)


def test_legacy_implied_q_is_the_engine_channel():
    _, implied = derive_implied_dividend_yield(rate=RATE, spot=SPOT, futures_price=SPOT * 0.995, time_to_maturity=0.05)
    assert legacy_implied_q(SPOT, SPOT * 0.995, 0.05, RATE) == pytest.approx(implied)


def test_flat_active_source_returns_the_active_contract_yield():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig()  # dividend_source None = legacy
    div = dividend_yield_for_day(chain, 1, spot=mp.spot[1, 0], rate=RATE, engine_config=cfg,
                                 active_contract="IM2402", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[1, 0])
    j = chain.contracts.index("IM2402")
    expected = legacy_implied_q(mp.spot[1, 0], chain.prices[1, j], chain.tenors[j], RATE)
    assert div.get_yield(0.5) == pytest.approx(expected)
    assert div.get_yield(2.0) == pytest.approx(expected)


def test_term_sources_reprice_the_listed_contracts_beyond_the_minimum_tenor():
    mp, chain = _chain_and_path()
    for extrapolation in ("flat_q", "flat_forward_carry", "surface_forward_carry"):
        cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_extrapolation=extrapolation,
                                       futures_curve_min_tenor_days=7)
        div = dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                                     active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
        for j, t in enumerate(chain.tenors):
            if t * 365.0 >= 7:
                model_forward = mp.spot[0, 0] * np.exp((RATE - div.get_yield(t)) * t)
                assert model_forward == pytest.approx(chain.prices[0, j], rel=1e-12)


def test_surface_forward_carry_tail_follows_the_simulated_curve():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry",
                                   futures_curve_min_tenor_days=1)
    div = dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                                 active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
    # the fixture curve is flat -10% carry: beyond the last contract the yield is r + 10%
    t_last = float(chain.tenors[-1])
    for t in (1.0, 1.5, 2.5):
        b_model = (RATE - div.get_yield(t)) * t
        b_curve = carry_at(mp.carry[0, 0], mp.tenor_grid, np.array([t]))[0]
        assert b_model == pytest.approx(b_curve, abs=1e-12)
    assert t_last < 1.0


def test_curve_tail_pillars_expose_only_tenors_beyond_the_last_contract():
    grid = np.array([0.25, 0.5, 1.0, 1.5])
    pillars = CurveTailPillars(tenors=grid, carry=-0.1 * grid)
    times, yields = pillars.implied_q_pillars(rate=RATE)
    assert times == [0.25, 0.5, 1.0, 1.5]
    assert yields == pytest.approx([RATE + 0.1] * 4)


def test_no_eligible_contract_fails_closed():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_min_tenor_days=400)
    with pytest.raises(ValidationError):
        dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                               active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
