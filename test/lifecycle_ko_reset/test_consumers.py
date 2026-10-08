"""KO-reset in the portfolio manager and the replay, and the REBASED provider matrix (spec 2026-10-08 5, 6a)."""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, PortfolioLifecycleManager
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.backtest.replay import (AutocallableEngineConfig, HedgeSpec, ReplayBacktestConfig,
                                      ReplayBacktestEngine, ReplayProduct)
from quantark.backtest.replay.engine_factory import create_autocallable_engine
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .test_tracker_rules import kr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_golden import fixtures  # noqa: E402

VAL_DATE = pd.Timestamp("2026-01-05")


def _env(spot, date):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.03), valuation_date=pd.Timestamp(date).to_pydatetime())


def _portfolio(product, engine):
    from quantark.portfolio import Portfolio

    portfolio = Portfolio(portfolio_name="ko-reset", pricing_environments={"IDX": _env(100.0, VAL_DATE)})
    portfolio.add_position(product=product, quantity=1.0, entry_price=0.0, underlying="IDX", engine=engine)
    return portfolio


def test_the_manager_tracks_a_ko_reset_and_closes_it_at_schedule_ones_end():
    """Before: a warning, no tracker, and the day after schedule 1's end every engine raised."""
    portfolio = _portfolio(kr(), KOResetSnowballPDESolver(PDEParams()))
    manager = PortfolioLifecycleManager(VAL_DATE.to_pydatetime())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        manager.register_positions(portfolio)
    assert manager.num_tracked == 1
    events = []
    for n, spot in ((91, 90.0), (182, 90.0)):
        date = VAL_DATE + pd.Timedelta(days=n)
        portfolio.pricing_environments["IDX"] = _env(spot, date)
        events = manager.process_day(portfolio, n, date.to_pydatetime())
    assert [e.event.event_type.name for e in events] == ["MATURITY"]
    assert not portfolio.positions
    date = VAL_DATE + pd.Timedelta(days=183)
    portfolio.pricing_environments["IDX"] = _env(90.0, date)
    assert manager.process_day(portfolio, 183, date.to_pydatetime()) == []


def test_the_managers_pricing_copy_of_a_knocked_in_rebased_contract_prices_off_mc():
    portfolio = _portfolio(kr(M.REBASED, post=(91, 365)), SnowballMCEngine(params=MCParams(seed=3, num_paths=2048)))
    manager = PortfolioLifecycleManager(VAL_DATE.to_pydatetime())
    manager.register_positions(portfolio)
    date = VAL_DATE + pd.Timedelta(days=91)
    portfolio.pricing_environments["IDX"] = _env(70.0, date)
    manager.process_day(portfolio, 91, date.to_pydatetime())
    copy = portfolio.positions[next(iter(portfolio.positions))].product
    assert copy.post_ko_mode == M.ABSOLUTE
    assert math.isfinite(KOResetSnowballPDESolver(PDEParams()).price(copy, _env(70.0, date)))


def test_the_manager_refuses_an_unknocked_rebased_contract_off_mc():
    portfolio = _portfolio(kr(M.REBASED, post=(91, 365)), PDEEngine())
    with pytest.raises(ValidationError, match="REBASED.*Monte Carlo"):
        PortfolioLifecycleManager(VAL_DATE.to_pydatetime()).register_positions(portfolio)


def test_replay_quad_prices_a_ko_reset_on_its_own_engine():
    """SnowballQuadEngine ignored schedule 2: a knocked-in contract priced 4.195 where its own engine prices -1.744."""
    engine = create_autocallable_engine(product=kr(), engine_type=EngineType.QUADRATURE,
                                        config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE))
    assert isinstance(engine, KOResetSnowballQuadEngine)


def _day_scale_ko_reset(mode=M.ABSOLUTE):
    """On the replay fixture's calendar (spots 100, 96, 104, 105, 106): KI 97 on days 1 and 3, schedule 1 (103)
    on days 2 and 5, schedule 2 (103) on days 2 and 5 (ABSOLUTE) or offsets 1 and 4 (REBASED)."""
    post = (2, 5) if mode == M.ABSOLUTE else (1, 4)
    product = kr(mode, post=post, pre=(2, 5), ki=(1, 3), ki_level=97.0, post_level=103.0)
    product.initial_price = product.strike = 100.0
    return product


def _replay_config(product, engine_type, **overrides):
    kwargs = dict(
        products=[ReplayProduct(product=product, quantity=-1.0, position_id=1, has_lifecycle=True)],
        market_data=fixtures._market_data(), hedge=HedgeSpec(kind="futures", multiplier=300.0),
        engine_config=AutocallableEngineConfig(pricing_engine_type=engine_type,
                                               mc_params=MCParams(seed=3, num_paths=2048),
                                               quad_params=QuadParams(grid_points=801)),
        transaction_cost_model=ZeroCostModel(), calculate_surfaces=False, calculate_event_probabilities=False,
    )
    kwargs.update(overrides)
    return ReplayBacktestConfig(**kwargs)


def test_replay_selects_the_ko_reset_tracker():
    engine = ReplayBacktestEngine(_replay_config(_day_scale_ko_reset(), EngineType.PDE))
    assert isinstance(engine._replays[0]._tracker, KOResetLifecycleTracker)


@pytest.mark.parametrize("engine_type", [EngineType.PDE, EngineType.QUADRATURE])
def test_the_replay_config_refuses_an_unknocked_rebased_contract_off_mc(engine_type):
    with pytest.raises(ValidationError, match="REBASED"):
        _replay_config(_day_scale_ko_reset(M.REBASED), engine_type)


def test_the_replay_config_refuses_rebased_event_stats_off_mc():
    with pytest.raises(ValidationError, match="event_stats"):
        _replay_config(_day_scale_ko_reset(M.REBASED), EngineType.MONTE_CARLO, calculate_event_probabilities=True,
                       engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO,
                                                              event_stats_engine_type=EngineType.PDE))


def test_the_replay_config_refuses_a_ko_reset_on_a_vol_model_pde_solver(tmp_path):
    from quantark.volcalibration.config import VolModelCalibrationConfig
    engine_config = AutocallableEngineConfig(pricing_engine_type=EngineType.PDE, vol_source="surface",
                                             vol_model="localvol", vol_model_solver="pde",
                                             vol_model_calibration=VolModelCalibrationConfig(cache_dir=None))
    with pytest.raises(ValidationError, match="KO-reset"):
        _replay_config(_day_scale_ko_reset(), EngineType.PDE, engine_config=engine_config)


@pytest.mark.parametrize("mode, engine_type", [(M.ABSOLUTE, EngineType.PDE), (M.ABSOLUTE, EngineType.QUADRATURE),
                                               (M.REBASED, EngineType.MONTE_CARLO)])
def test_a_replay_runs_a_ko_reset_through_its_knock_in(mode, engine_type):
    results = ReplayBacktestEngine(_replay_config(_day_scale_ko_reset(mode), engine_type)).run()
    actions = results.actions_df() if callable(results.actions_df) else results.actions_df
    states = results.states_df() if callable(results.states_df) else results.states_df
    kinds = [str(k).upper() for k in actions[[c for c in actions.columns if "type" in c][0]]]
    assert any("KNOCK_IN" in k or k == "KI" for k in kinds), kinds
    assert any("KNOCK_OUT" in k or k == "KO" for k in kinds), kinds
    assert np.isfinite(states["product_mtm"]).all()
