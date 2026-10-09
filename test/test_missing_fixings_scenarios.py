"""A filled missing past fixing stays at the UNSHOCKED base spot under every scenario revaluation.

User decision 2026-10-09: "history doesn't move" applies to scenario shocks too (VaR, stress tests, PnL explain,
dynamic scenarios, backtests). Multi-day runs and snapshot pairs fill a fixing already missing at the FIRST
valuation once, at that valuation's spot, and keep it. Parity per path: the shocked values of a contract with
missing past fixings equal those of the same contract with the fixings recorded explicitly at the base spot.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
)
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.fx.engine.analytical import FxRangeAccrualAnalyticalEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import (
    FxRangeAccrualConfig,
    FxRangeAccrualObservationRecord,
    FxRangeAccrualOption,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.fixings import MissingFixingWarning

VAL = datetime(2025, 1, 1)
EQ_SPOT = 109.5  # inside the [90, 110] band; most shocks cross 110


def _count_missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, sum(issubclass(w.category, MissingFixingWarning) for w in caught)


# =============================================================================
# Equity book: a numeric and a dated range accrual plus a dated Asian, each with one missing past fixing
# =============================================================================


def _eq_env(spot=EQ_SPOT, date=VAL):
    return PricingEnvironment(
        valuation_date=date,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
    )


def _ra_numeric(outcome=None):
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        maturity=1.0,
        tenor=1.25,
        observation_records=[
            RangeAccrualObservationRecord(observation_time=-0.2, observed_in_range=True),
            RangeAccrualObservationRecord(observation_time=-0.1, weight=2.0, observed_in_range=outcome),
            RangeAccrualObservationRecord(observation_time=0.5),
            RangeAccrualObservationRecord(observation_time=1.0),
        ],
    )


def _ra_dated(outcome=None, start=VAL):
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.05),
        exercise_date=start + timedelta(days=365),
        initial_date=start - timedelta(days=92),
        observation_records=[
            RangeAccrualObservationRecord(observation_date=start - timedelta(days=61), observed_in_range=True),
            RangeAccrualObservationRecord(
                observation_date=start - timedelta(days=31), weight=2.0, observed_in_range=outcome
            ),
            RangeAccrualObservationRecord(observation_date=start + timedelta(days=181)),
            RangeAccrualObservationRecord(observation_date=start + timedelta(days=365)),
        ],
    )


def _asian_dated(price=None, start=VAL):
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        exercise_date=start + timedelta(days=365),
        observation_records=[
            AsianObservationRecord(observation_date=start - timedelta(days=61), observed_price=95.0),
            AsianObservationRecord(observation_date=start - timedelta(days=31), observed_price=price),
            AsianObservationRecord(observation_date=start + timedelta(days=181)),
            AsianObservationRecord(observation_date=start + timedelta(days=365)),
        ],
    )


def _eq_book(explicit, spot=EQ_SPOT):
    """The book; ``explicit`` records the missing fixings at the base spot instead."""
    pf = EquityPortfolio(portfolio_name="book", pricing_environments={"X": _eq_env(spot)}, creation_date=VAL)
    outcome = True if explicit else None
    for product, engine, qty in (
        (_ra_numeric(outcome), RangeAccrualAnalyticalEngine(), 10.0),
        (_ra_dated(outcome), RangeAccrualAnalyticalEngine(), -4.0),
        (_asian_dated(spot if explicit else None), AsianOptionAnalyticalEngine(), 3.0),
    ):
        pf.add_position(
            product=product, quantity=qty, entry_price=0.0, underlying="X", engine=engine, entry_timestamp=VAL
        )
    return pf


def _spot_history(days=300, sigma=0.02, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "spot_return": rng.normal(0.0, sigma, days),
            "vol_change": rng.normal(0.0, 0.002, days),
            "rate_shift": rng.normal(0.0, 0.0002, days),
        },
        index=pd.date_range("2023-01-01", periods=days, freq="D"),
    )


def _var_engines():
    from quantark.var import VaRConfig, VaRMethod
    from quantark.var.engines import HistoricalVaREngine, MonteCarloVaREngine, ParametricVaREngine

    return {
        "historical": lambda: HistoricalVaREngine(
            VaRConfig(confidence_level=0.95, var_method=VaRMethod.HISTORICAL, lookback_days=250)
        ),
        "mc": lambda: MonteCarloVaREngine(
            VaRConfig(
                confidence_level=0.95,
                var_method=VaRMethod.MONTE_CARLO,
                lookback_days=250,
                mc_num_simulations=300,
                mc_seed=5,
            )
        ),
        "parametric": lambda: ParametricVaREngine(
            VaRConfig(confidence_level=0.95, var_method=VaRMethod.PARAMETRIC, lookback_days=250)
        ),
    }


@pytest.mark.parametrize("kind", ["historical", "mc", "parametric"])
def test_equity_var_keeps_the_base_fill(kind):
    make = _var_engines()[kind]
    got, count = _count_missing(make().calculate_var, _eq_book(False), _spot_history())
    assert count == 3  # one per contract, at the base valuation; none per scenario
    want = make().calculate_var(_eq_book(True), _spot_history())
    assert got.var == want.var and got.cvar == want.cvar
    if got.scenarios is not None:
        assert np.array_equal(got.scenarios["pnl"].to_numpy(), want.scenarios["pnl"].to_numpy())


def test_equity_incremental_var_keeps_the_base_fill():
    make = _var_engines()["historical"]
    got, _ = _count_missing(make().calculate_incremental_var, _eq_book(False), _spot_history())
    want = make().calculate_incremental_var(_eq_book(True), _spot_history())
    assert got.portfolio_var == want.portfolio_var
    assert list(got.position_ivari.values()) == list(want.position_ivari.values())


def _stress_values(results):
    return [
        (r.portfolio_value, r.portfolio_pnl, tuple(sorted((r.greeks or {}).items()))) for r in results.scenario_results
    ]


def test_equity_stress_keeps_the_base_fill():
    from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine

    scenarios = [ScenarioBuilder().name(f"spot {s:+.2f}").spot_stress(s).build() for s in (-0.05, 0.02, 0.05)]
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True))
    got, count = _count_missing(engine.run_static_scenarios, _eq_book(False), scenarios)
    assert count == 3
    want = StressTestEngine(StressTestConfig(calculate_greeks=True)).run_static_scenarios(_eq_book(True), scenarios)
    assert got.baseline_value == want.baseline_value
    assert _stress_values(got) == _stress_values(want)


def test_equity_dynamic_scenario_fills_once_at_the_first_valuation():
    from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine, PathLibrary

    def run(explicit):
        path = PathLibrary.consecutive_rally(days=4, daily_pct=0.01, vol_change_pct=0.0)
        path.start_date = VAL
        config = DynamicScenarioConfig(calculate_greeks=True, handle_lifecycle_events=False)
        return DynamicScenarioEngine(config).run(_eq_book(explicit), path)

    got, count = _count_missing(run, False)
    assert count == 3
    want = run(True)
    assert got.baseline_value == want.baseline_value
    assert [(d.portfolio_value, d.greeks) for d in got.day_results] == [
        (d.portfolio_value, d.greeks) for d in want.day_results
    ]


def test_equity_backtest_fills_once_at_the_first_valuation():
    from test_backtest_lifecycle import START, RampAdapter

    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    def run(explicit):
        adapter = RampAdapter(start_spot=EQ_SPOT, end_spot=112.0, num_days=8)
        positions = [
            Position(
                product=product,
                quantity=5.0,
                entry_price=0.0,
                underlying="EQUITY_INDEX",
                engine=engine,
                entry_timestamp=START,
            )
            for product, engine in (
                (_ra_dated(True if explicit else None, start=START), RangeAccrualAnalyticalEngine()),
                (_asian_dated(EQ_SPOT if explicit else None, start=START), AsianOptionAnalyticalEngine()),
            )
        ]
        config = BacktestConfig(
            strategy=DeltaNeutralStrategy(delta_threshold=1e12),
            start_date=START,
            end_date=adapter.dates[-1].to_pydatetime(),
            underlying="EQUITY_INDEX",
            initial_positions=positions,
            market_data_adapter=adapter,
            transaction_cost_model=ZeroCostModel(),
            handle_lifecycle_events=False,
        )
        return BacktestEngine(config).run()

    got, _ = _count_missing(run, False)
    want = run(True)
    assert got.get_value_series().tolist() == want.get_value_series().tolist()
    assert got.get_delta_series().tolist() == want.get_delta_series().tolist()


# =============================================================================
# PnL explain: one snapshot pair, the fill made at t0 and kept at t1
# =============================================================================


def test_pnl_explain_keeps_the_t0_fill_across_the_pair():
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    t1 = VAL + timedelta(days=3)
    engine = RangeAccrualAnalyticalEngine()
    config = PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR))

    def run(explicit):
        outcome = True if explicit else None
        s0 = ValuationSnapshot(_ra_dated(outcome), engine, _eq_env(EQ_SPOT, VAL), date=VAL, quantity=2.0)
        s1 = ValuationSnapshot(_ra_dated(outcome), engine, _eq_env(111.0, t1), date=t1, quantity=2.0)
        return explain(s0, s1, config=config)

    got, count = _count_missing(run, False)
    assert count == 1
    want = run(True)
    assert got.pv_t0 == want.pv_t0 and got.pv_alive_t1 == want.pv_alive_t1 and got.pv_t1 == want.pv_t1
    assert [(r.factor, r.method, r.pnl) for r in got.rows] == [(r.factor, r.method, r.pnl) for r in want.rows]


# =============================================================================
# FX book
# =============================================================================

FX_SPOT = 1.2995
FX_VAL = datetime(2026, 6, 15)


def _fx_env(spot=FX_SPOT, date=FX_VAL):
    return FxPricingEnvironment(
        valuation_date=date,
        spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )


def _fx_dated(outcome=None, start=FX_VAL):
    return FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"),
        expiry_date=start + timedelta(days=365),
        initial_date=start - timedelta(days=92),
        observation_records=[
            FxRangeAccrualObservationRecord(observation_date=start - timedelta(days=61), observed_in_range=True),
            FxRangeAccrualObservationRecord(
                observation_date=start - timedelta(days=31), weight=2.0, observed_in_range=outcome
            ),
            FxRangeAccrualObservationRecord(observation_date=start + timedelta(days=181)),
            FxRangeAccrualObservationRecord(observation_date=start + timedelta(days=365)),
        ],
    )


def _fx_numeric(outcome=None):
    return FxRangeAccrualOption(
        notional=1_000_000.0,
        range_config=FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04),
        currency_pair=CurrencyPair("EUR", "USD"),
        maturity=1.0,
        observation_records=[
            FxRangeAccrualObservationRecord(observation_time=-0.25, observed_in_range=True),
            FxRangeAccrualObservationRecord(observation_time=-0.1, weight=2.0, observed_in_range=outcome),
            FxRangeAccrualObservationRecord(observation_time=0.5),
            FxRangeAccrualObservationRecord(observation_time=1.0),
        ],
    )


def _fx_book(explicit, start=FX_VAL):
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": _fx_env(date=start)})
    outcome = True if explicit else None
    for product, qty in ((_fx_numeric(outcome), 1.0), (_fx_dated(outcome, start=start), -2.0)):
        pf.add_position(
            product=product, quantity=qty, entry_price=0.0, underlying="EURUSD", engine=FxRangeAccrualAnalyticalEngine()
        )
    return pf


def _fx_history(days=300, sigma=0.006, seed=7):
    rng = np.random.default_rng(seed)
    spot = FX_SPOT * np.exp(np.cumsum(rng.normal(0.0, sigma, days)))
    return pd.DataFrame(
        {
            "EURUSD_spot": spot,
            "EURUSD_vol": np.full(days, 0.10),
            "EURUSD_dom_rate": np.full(days, 0.05),
            "EURUSD_for_rate": np.full(days, 0.03),
        },
        index=pd.date_range("2025-01-01", periods=days, freq="B"),
    )


@pytest.mark.parametrize("kind", ["parametric", "historical", "mc"])
def test_fx_var_keeps_the_base_fill(kind):
    from quantark.var import VaRConfig
    from quantark.var.fx.engine import FXHistoricalVaREngine, FXMonteCarloVaREngine, FXParametricVaREngine

    cls = {"parametric": FXParametricVaREngine, "historical": FXHistoricalVaREngine, "mc": FXMonteCarloVaREngine}[kind]
    config = VaRConfig(confidence_level=0.95, lookback_days=250, mc_num_simulations=300, mc_seed=5)
    got, count = _count_missing(cls(config).calculate_var, _fx_book(False), _fx_history())
    assert count == 2
    want = cls(config).calculate_var(_fx_book(True), _fx_history())
    assert got.var == want.var and got.cvar == want.cvar


def test_fx_stress_keeps_the_base_fill():
    from quantark.stresstest import FXStressConfig, FXStressEngine
    from quantark.stresstest.scenario.scenario import Scenario, Stress
    from quantark.stresstest.stress.stress_types import StressLevel, StressType

    scenarios = [
        Scenario(name=f"{pct:+.3f}", stresses=[Stress("spot", StressType.PERCENTAGE, pct, StressLevel.PORTFOLIO)])
        for pct in (-0.01, 0.002, 0.01)
    ]
    got, count = _count_missing(
        FXStressEngine(FXStressConfig(calculate_greeks=True)).run_static_scenarios, _fx_book(False), scenarios
    )
    assert count == 2
    want = FXStressEngine(FXStressConfig(calculate_greeks=True)).run_static_scenarios(_fx_book(True), scenarios)
    assert got.baseline_value == want.baseline_value
    assert _stress_values(got) == _stress_values(want)


def test_fx_dynamic_scenario_fills_once_at_the_first_valuation():
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine, FXPathLibrary

    def run(explicit):
        path = FXPathLibrary.spot_trend(days=4, daily_pct=0.001)
        return FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True)).run(_fx_book(explicit), path)

    got, count = _count_missing(run, False)
    assert count == 2
    want = run(True)
    assert got.baseline_value == want.baseline_value
    assert [(d.portfolio_value, d.greeks) for d in got.day_results] == [
        (d.portfolio_value, d.greeks) for d in want.day_results
    ]


def test_fx_backtest_fills_once_at_the_first_valuation():
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy
    from quantark.backtest.transaction_costs import ZeroCostModel

    start = FX_VAL
    idx = pd.bdate_range(start, periods=6)
    path = pd.DataFrame(
        {
            "EURUSD_spot": np.linspace(FX_SPOT, 1.305, len(idx)),
            "EURUSD_vol": np.full(len(idx), 0.10),
            "EURUSD_dom_rate": np.full(len(idx), 0.05),
            "EURUSD_for_rate": np.full(len(idx), 0.03),
        },
        index=idx,
    )

    def run(explicit):
        config = FXBacktestConfig(
            portfolio=_fx_book(explicit, start=start),
            market_path=path,
            strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
            transaction_cost_model=ZeroCostModel(),
        )
        return FXBacktestEngine(config).run().to_dataframe()

    got, _ = _count_missing(run, False)
    want = run(True)
    pd.testing.assert_frame_equal(got, want, check_exact=True)


# =============================================================================
# Execution-framework greek cells and SA-CVA
# =============================================================================


@pytest.mark.parametrize("make_ra", [_ra_numeric, _ra_dated], ids=["numeric", "dated"])
def test_execution_greek_cells_keep_the_base_fill(make_ra):
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    from quantark.execution.greeks import TradeState, greek_bump_cells, greek_bump_transform, run_greek_bump

    gc = GreeksCalculator()
    greeks = ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]

    def cells(product):
        state = TradeState(
            product=product, pricing_env=_eq_env(), cash_legs=(), engine=RangeAccrualAnalyticalEngine(),
            streams=None, quantity=1.0, greeks_params=gc.params,
        )
        return {
            cell.bump_id: run_greek_bump(
                cell.bump_id, state, greek_bump_transform(state, {"bump_id": cell.bump_id}), gc
            )
            for cell in greek_bump_cells(greeks)
        }

    got, _ = _count_missing(cells, make_ra())
    assert got == cells(make_ra(True))


def test_sacva_hedge_sensitivities_keep_the_base_fill():
    """SA-CVA's exposure engine cannot price a range accrual or an Asian (its payoff needs the fixing history), so
    such a contract only reaches SA-CVA as a hedge, whose market sensitivity is a spot-bumped reprice."""
    from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    from quantark.sacva.exposure.simulator import MonteCarloExposureConfig, MonteCarloExposureEngine
    from quantark.sacva.models.enums import CreditQuality
    from quantark.sacva.portfolio.counterparty import Counterparty
    from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
    from quantark.sacva.portfolio.netting import NettingSet
    from quantark.sacva.portfolio.trade import CVAHedge, CVATrade
    from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
    from quantark.sacva.sacva_engine import SACVAEngine
    from quantark.util.calendar import DayCountConvention

    def env():
        return PricingEnvironment(
            rate_curve=FlatRateCurve(0.03), valuation_date=VAL,
            spot_quote=SpotQuote(spot=EQ_SPOT, asset_name="ACME"), vol_surface=FlatVolSurface(0.25),
            div_yield=ContinuousDividendYield(0.0), day_count_convention=DayCountConvention.CALENDAR_DAYS,
        )

    def run(explicit):
        call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=datetime(2027, 1, 1))
        trade = CVATrade(trade_id="t1", product=call, engine=BlackScholesEngine(), env=env(), quantity=10.0,
                         trade_currency="USD", equity_bucket=5)
        hedge = CVAHedge(trade_id="h1", product=_ra_dated(True if explicit else None),
                         engine=RangeAccrualAnalyticalEngine(), env=env(), quantity=-100.0, trade_currency="USD",
                         equity_bucket=5)
        curve = PillarHazardCurve(tenors=[0.5, 1.0, 3.0, 5.0, 10.0], hazards=[0.02] * 5, recovery_rate=0.4)
        cp = Counterparty(name="CP", netting_sets=[NettingSet("n1", [trade])], credit_curve=curve, bucket=2,
                          credit_quality=CreditQuality.IG)
        engine = SACVAEngine(
            exposure_engine=MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=500, n_steps=6, seed=3)),
            include_ir_delta=False,
        )
        return engine.compute(CVATradePortfolio(counterparties=[cp], hedges=[hedge]))

    got, count = _count_missing(run, False)
    assert count == 1
    want = run(True)
    assert got.total_capital == want.total_capital
    assert [vars(s) for s in got.sensitivities] == [vars(s) for s in want.sensitivities]


def test_replay_book_fills_once_at_the_first_valuation():
    """The replay book (an Asian on the analytical route) fills once, at the first day's spot (100.0)."""
    from replay_golden import fixtures

    from quantark.backtest.replay.config import AutocallableEngineConfig
    from quantark.backtest.replay.engine import ReplayBacktestEngine
    from quantark.util.enum.engine_enums import EngineType

    start = datetime(2024, 1, 2)

    def run(price):
        cfg = fixtures.make_book_config()
        cfg.products = [
            fixtures.BookProduct(product=_asian_dated(price, start=start), quantity=1.0, position_id=1,
                                 has_lifecycle=False)
        ]
        cfg.calculate_event_probabilities = False
        cfg.engine_config = AutocallableEngineConfig(pricing_engine_type=EngineType.ANALYTICAL)
        return ReplayBacktestEngine(cfg).run()

    got, count = _count_missing(run, None)
    assert count == 1
    want = run(100.0)
    assert got.get_value_series().tolist() == want.get_value_series().tolist()
    pd.testing.assert_frame_equal(got.greeks_df(), want.greeks_df(), check_exact=True)
