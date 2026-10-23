"""Day-0 parity across every consumer of a trade-value portfolio (ship spec 2026-10-23 portfolio-trade-value AC3).

One book with cash legs, ``trade_value=True``, whole-day schedule, legs anchored on the run start: the portfolio
value, the static-stress baseline, the historical-VaR base value, the dynamic scenario's baseline and the
backtest's first recorded value are one number (bitwise), and the portfolio Greeks equal the stress baseline
Greeks.
"""
import copy

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest import BacktestEngine
from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine
from quantark.var import HistoricalVaREngine, VaRConfig, VaRMethod
from test_backtest import test_trade_value_runs as runs
from test_var.test_var_trade_value import _factor_history


def _book():
    return runs._dynamic_portfolio(True)            # the backtest's market on underlying "IDX", with legs


def test_every_consumer_sees_the_same_day_zero_trade_value():
    value = copy.deepcopy(_book()).get_portfolio_value()

    stress = StressTestEngine(StressTestConfig(calculate_greeks=True)).run_static_scenarios(
        _book(), [ScenarioBuilder().name("spot -5%").spot_stress(-0.05).build()])
    assert stress.baseline_value == value
    assert stress.baseline_greeks == copy.deepcopy(_book()).get_portfolio_greeks(GreeksCalculator())

    var = HistoricalVaREngine(VaRConfig(var_method=VaRMethod.HISTORICAL, lookback_days=6, confidence_level=0.9,
                                        calculate_marginal_var=False)).calculate_var(_book(), _factor_history(6))
    assert var.portfolio_value == value

    dynamic = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        _book(), runs._rally())
    assert dynamic.baseline_value == value

    backtest = BacktestEngine(runs._config()).run()
    assert backtest.states_df["portfolio_value"].iloc[0] == runs._static_value(trade_value=True)
    assert runs._static_value(trade_value=True, underlying="IDX") == value
