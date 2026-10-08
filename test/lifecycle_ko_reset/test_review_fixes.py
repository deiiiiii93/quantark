"""Final-review findings on the KO-reset lifecycle branch, each pinned by the case that exposed it."""
import math
from datetime import timedelta

import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, LifecycleEventType as E
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env
from .test_tracker_rules import D, day, kr, run, tracker


# -- C2: a knocked-in REBASED copy keeps a just-observed schedule-2 record ~1e-17 years ahead ----------------------

@pytest.mark.parametrize("ki_day", [91, 182])
def test_a_knocked_in_rebased_copy_drops_the_record_on_its_due_close(ki_day):
    kept = []
    for offset in sorted(set(range(1, 400, 3)) | {5}):
        t = tracker(kr(M.REBASED, post=(offset,)))
        run(t, {91: 70.0} if ki_day == 91 else {91: 90.0, 182: 70.0})
        copy = t.product_for_pricing(day(ki_day + offset), env(90.0, day(ki_day + offset)))
        left = copy.post_barrier_config.ko_observation_schedule.records
        if left:
            kept.append((offset, left[0].observation_time))
    assert kept == []


def test_quad_prices_a_knocked_in_rebased_copy_on_a_schedule_two_close():
    """The review's case: knock-in day 91, offset 5, spot 94: QUAD raised on a 5.55e-17-year record."""
    t = tracker(kr(M.REBASED, post=(5, 365)))
    run(t, {91: 70.0, 96: 94.0})
    copy = t.product_for_pricing(day(96), env(94.0, day(96)))
    assert math.isfinite(KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)).price(copy, env(94.0, day(96))))


# -- C1: a REBASED contract on the date clock, and a mid-life run -------------------------------------------------

def dated_rebased(offsets=(30, 60)):
    """Schedule 1 and the KI on DATES (91 and 182 days after START), schedule 2 as REBASED offsets in days."""
    dated = lambda days, level: ObservationSchedule(records=[  # noqa: E731
        ObservationRecord(observation_date=(START + pd.Timedelta(days=d)).to_pydatetime(), barrier=level) for d in days])
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=(182 + max(offsets)) * D, initial_date=START.to_pydatetime(),
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=dated((91, 182), 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=dated((91, 182), 75.0)),
        post_barrier_config=BarrierConfig(
            ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[
                ObservationRecord(observation_time=o * D, barrier=95.0) for o in offsets])),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=M.REBASED)


def test_a_dated_rebased_contract_knocks_out_on_schedule_two():
    assert run(tracker(dated_rebased()), {91: 70.0, 121: 96.0}) == [
        (91, E.KNOCK_IN, None), (121, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_a_dated_rebased_contract_matures_at_its_knocked_in_end():
    assert run(tracker(dated_rebased()), {91: 70.0, 121: 90.0, 150: 80.0, 151: 60.0})[-1] == (
        151, E.MATURITY, "maturity")


@pytest.mark.parametrize("engine", [
    lambda: KOResetSnowballPDESolver(PDEParams()),
    lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    lambda: SnowballMCEngine(params=MCParams(seed=3, num_paths=4096)),
], ids=["pde", "quad", "mc"])
def test_a_dated_rebased_knocked_in_copy_prices(engine):
    t = tracker(dated_rebased())
    run(t, {91: 70.0})
    copy = t.product_for_pricing(day(100), env(90.0, day(100)))
    assert math.isfinite(engine().price(copy, env(90.0, day(100))))


def test_a_mid_life_run_places_schedule_two_after_the_knock_in():
    """initial_date 100 days before the run: the record clock's origin; schedule 2 hangs 30 days after the knock-in
    close, not 30 days after the run's own day 60 measured from initial_date (which lay before the knock-in)."""
    product = kr(M.REBASED, post=(30, 60), pre=(160, 250), ki=(160, 250))
    product.initial_date = (START - pd.Timedelta(days=100)).to_pydatetime()
    assert run(tracker(product), {60: 70.0, 90: 96.0}) == [(60, E.KNOCK_IN, None), (90, E.KNOCK_OUT, "knock-out:ko2:0")]


# -- I1: PnL explain fingerprints the new bookkeeping field ---------------------------------------------------------

def test_observed_schedule_two_indices_are_bookkeeping():
    from quantark.pnlexplain.equity.fingerprints import lifecycle_fingerprint
    a, b = AutocallableLifecycleState(), AutocallableLifecycleState()
    b.observed_post_ko_indices.add(0)
    assert lifecycle_fingerprint(a) == lifecycle_fingerprint(b)


# -- I2: the baseline is the tracker's initial copy ------------------------------------------------------------------

def _carried_knocked_in_rebased():
    """Knocked in 10 days before the run on its first knock-in record (accrual time spec §5.1: carried with its
    anchor); schedule 2 at offsets 91 and 365 days from that close."""
    from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
    product = kr(M.REBASED, post=(91, 365))
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", (START - pd.Timedelta(days=10)).to_pydatetime())
    setattr(product, "_otc_lifecycle_knock_in_anchor",
            knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[0], 0))
    return product


def _portfolio(product, engine):
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.portfolio import Portfolio
    from quantark.priceenv import PricingEnvironment
    e = PricingEnvironment(spot_quote=SpotQuote(spot=90.0, asset_name="IDX"), vol_surface=FlatVolSurface(0.2),
                           rate_curve=FlatRateCurve(0.03), valuation_date=START.to_pydatetime())
    portfolio = Portfolio(portfolio_name="carried", pricing_environments={"IDX": e})
    portfolio.add_position(product=product, quantity=1.0, entry_price=0.0, underlying="IDX", engine=engine)
    return portfolio, e


def test_a_dynamic_scenario_values_a_carried_knocked_in_rebased_position_from_day_zero():
    """The baseline was priced from the booked product before the trackers were registered: a carried knocked-in
    REBASED contract was valued as the un-converted contract (PDE refuses it), not as the ABSOLUTE contract the
    tracker prices from day 0."""
    from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine, PathBuilder

    engine = KOResetSnowballPDESolver(PDEParams())
    portfolio, e = _portfolio(_carried_knocked_in_rebased(), engine)
    path = PathBuilder(num_days=1, name="Flat").build()
    path.start_date = START.to_pydatetime()
    results = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        portfolio, path)
    initial = tracker(_carried_knocked_in_rebased()).product_for_pricing(START, e, initial=True)
    assert results.baseline_value == pytest.approx(float(engine.price(initial, e)), rel=1e-12)


def test_the_manager_hands_each_tracked_position_its_initial_copy():
    from quantark.asset.equity.lifecycle import PortfolioLifecycleManager

    engine = KOResetSnowballPDESolver(PDEParams())
    portfolio, e = _portfolio(_carried_knocked_in_rebased(), engine)
    PortfolioLifecycleManager(base_date=START.to_pydatetime()).register_positions(portfolio)
    initial = tracker(_carried_knocked_in_rebased()).product_for_pricing(START, e, initial=True)
    assert portfolio.get_portfolio_value() == pytest.approx(float(engine.price(initial, e)), rel=1e-12)


def test_an_equity_backtest_values_a_carried_knocked_in_rebased_position_from_day_zero():
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position
    from test_backtest_lifecycle import UNDERLYING, RampAdapter

    adapter = RampAdapter(start_spot=90.0, end_spot=90.0, num_days=5)
    start = adapter.dates[0].to_pydatetime()
    product = _carried_knocked_in_rebased()
    setattr(product, "_otc_lifecycle_knock_in_date", start - timedelta(days=10))
    position = Position(product=product, quantity=1.0, entry_price=0.0, underlying=UNDERLYING,
                        engine=KOResetSnowballPDESolver(PDEParams()), entry_timestamp=start)
    config = BacktestConfig(strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=start,
                            end_date=adapter.dates[-1].to_pydatetime(), underlying=UNDERLYING,
                            initial_positions=[position], market_data_adapter=adapter,
                            transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=True,
                            calculate_greeks=False)
    engine = BacktestEngine(config)
    engine._initialize()                                        # the run's set-up: portfolio, trackers, baseline
    held = next(iter(engine.portfolio.positions.values())).product
    assert held.post_ko_mode == M.ABSOLUTE                      # the knocked-in contract it has become
    assert engine._initial_portfolio_value == pytest.approx(engine.portfolio.get_portfolio_value(), rel=1e-12)


# -- I3: the simulation refuses a carried knock-in ------------------------------------------------------------------

def test_the_simulation_refuses_a_carried_knock_in():
    from simulation.conftest import ensemble_config
    from quantark.backtest.replay import ReplayProduct
    product = kr()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    with pytest.raises(ValidationError, match="carried knock-in"):
        ensemble_config(products=[ReplayProduct(product=product, quantity=-1.0, position_id=1, has_lifecycle=True)])
