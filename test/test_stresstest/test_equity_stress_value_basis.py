"""Static equity stress follows ``EquityPortfolio.trade_value`` (ship spec 2026-10-23 portfolio-trade-value D5; AC3, AC4).

Flag off: values ``get_market_value``, Greeks ``get_greeks(use_analytical=...)``, product aggregate keys,
``value_basis == "product"``; ``native_batch`` warns and runs scalar. Flag on: trade value, and the baseline
equals the portfolio's own trade-mode value and Greeks.
"""
import copy
import warnings

import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.portfolio.equity import valuation as shared
from quantark.stresstest import StressTestConfig, StressTestEngine
from quantark.stresstest.stress.stress_applicator import StressApplicator
from test_portfolio.test_equity_portfolio_trade_value import _book
from test_stresstest.test_equity_stress_trade_value import _scenarios


def _product_rows(pf, env_by_underlying, gc, use_analytical):
    rows = {}
    for pid, p in pf.positions.items():
        q = copy.deepcopy(p)
        env = env_by_underlying[p.underlying]
        rows[pid] = (q.get_market_value(env), copy.deepcopy(p).get_greeks(env, gc, use_analytical=use_analytical))
    return rows


@pytest.mark.parametrize("greeks_method", ["analytical", "numerical"])
def test_flag_off_stress_is_the_product_path(greeks_method):
    pf = _book(trade_value=False)
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True, greeks_method=greeks_method))
    results = engine.run_static_scenarios(pf, _scenarios())
    gc = engine.greeks_calculator
    use_analytical = greeks_method == "analytical"

    base = _product_rows(pf, pf.pricing_environments, gc, use_analytical)
    assert results.baseline_value == sum(v for v, _ in base.values())
    assert list(results.baseline_greeks) == list(shared.PRODUCT_AGGREGATE_KEYS)
    assert results.metadata["value_basis"] == "product"
    assert results.metadata["valuation_mode"] == "scalar"
    for scenario, result in zip(_scenarios(), results.scenario_results):
        envs = StressApplicator.apply_scenario_to_portfolio(pf, scenario)
        want = _product_rows(pf, envs, gc, use_analytical)
        assert result.portfolio_value == sum(v for v, _ in want.values())
        for row in result.position_results:
            value, greeks = want[row["position_id"]]
            assert row["stressed_value"] == value
            assert row["greeks"] == greeks
        assert list(result.greeks) == list(shared.PRODUCT_AGGREGATE_KEYS)
        assert result.metadata["value_basis"] == "product"


def test_flag_off_native_batch_warns_and_runs_scalar():
    pf = _book(trade_value=False)
    scalar = StressTestEngine(StressTestConfig(calculate_greeks=False)).run_static_scenarios(pf, _scenarios())
    with pytest.warns(UserWarning, match="native_batch"):
        batch = StressTestEngine(StressTestConfig(calculate_greeks=False, native_batch=True)).run_static_scenarios(
            pf, _scenarios())
    assert batch.metadata["valuation_mode"] == "scalar"
    assert [r.portfolio_value for r in batch.scenario_results] == [r.portfolio_value for r in scalar.scenario_results]


def test_flag_on_stress_baseline_equals_the_portfolio_trade_mode():
    """AC3, stress half: the stress baseline is the portfolio's own trade-mode value and Greeks, bitwise."""
    pf = _book(trade_value=True)
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True))
    results = engine.run_static_scenarios(pf, _scenarios()[:1])
    ref = copy.deepcopy(pf)
    assert results.metadata["value_basis"] == "trade"
    assert results.baseline_value == ref.get_portfolio_value()
    assert results.baseline_greeks == copy.deepcopy(pf).get_portfolio_greeks(GreeksCalculator())


@pytest.mark.parametrize("flag", [True, False])
def test_pool_honors_the_mode(flag):
    pf = _book(trade_value=flag)
    serial = StressTestEngine(StressTestConfig(calculate_greeks=True)).run_static_scenarios(pf, _scenarios())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pooled = StressTestEngine(StressTestConfig(calculate_greeks=True, parallel_execution=True, max_workers=2)
                                  ).run_static_scenarios(pf, _scenarios())
    assert pooled.metadata["execution"] == "process_pool"
    assert pooled.metadata["value_basis"] == ("trade" if flag else "product")
    for a, b in zip(serial.scenario_results, pooled.scenario_results):
        assert a.portfolio_value == b.portfolio_value
        assert a.greeks == b.greeks
        assert [r["greeks"] for r in a.position_results] == [r["greeks"] for r in b.position_results]
