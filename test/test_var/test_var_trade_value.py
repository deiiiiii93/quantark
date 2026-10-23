"""VaR honours ``EquityPortfolio.trade_value`` (ship spec 2026-10-23 portfolio-trade-value D7, D20; AC1, AC3, AC6).

Historical and Monte Carlo VaR reprice every position, and every per-position component P&L, through the
portfolio's valuation switch (trade value in trade mode); parametric VaR reads trade-risk Greeks and replaces the
Black–Scholes psi proxy by the trade ``dividend_rho`` in psi's units ($ per unit of q). Flag off: the product
formulas, bitwise.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import ContinuousDividendYield
from quantark.portfolio.equity import valuation as shared
from quantark.var import HistoricalVaREngine, MonteCarloVaREngine, ParametricVaREngine, VaRConfig
from quantark.var.config import EquityRiskFactorConfig, VaRMethod
from test_portfolio.test_equity_portfolio_trade_value import _book

ALL_FACTORS = EquityRiskFactorConfig(include_div_yield=True)


def _factor_history(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "spot_return": rng.normal(0.0, 0.015, n),
            "vol_change": rng.normal(0.0, 0.004, n),
            "rate_shift": rng.normal(0.0, 0.0003, n),
            "div_yield_shift": rng.normal(0.0, 0.0002, n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )


def _value(position, env, trade):
    """The reference value of one position: the old formula (product) or the trade value."""
    if trade and shared.has_trade_api(position):
        return position.get_trade_value(env)
    return position.get_market_value(env)


def _expected_scenario_pnl(engine, ref, scenarios, trade):
    envs = ref.pricing_environments
    base = 0.0
    for p in ref.positions.values():
        base += _value(p, envs[p.underlying], trade)
    out = []
    for _, scenario in scenarios.iterrows():
        total = 0.0
        for p in ref.positions.values():
            total += _value(p, engine._create_stressed_environment(envs[p.underlying], scenario), trade)
        out.append(total - base)
    return base, np.array(out)


def _old_component_pnls(engine, ref, scenarios):
    """Per-position scenario P&L exactly as the engines computed it before the switch (price × quantity)."""
    envs = ref.pricing_environments
    out = {}
    for pos_id, p in ref.positions.items():
        current_price = p.get_current_price(envs[p.underlying])
        pnls = []
        for _, scenario in scenarios.iterrows():
            stressed_price = p.get_current_price(engine._create_stressed_environment(envs[p.underlying], scenario))
            pnls.append(stressed_price * p.quantity - (p.quantity * current_price))
        out[pos_id] = np.array(pnls)
    return out


def _trade_component_pnls(engine, ref, scenarios):
    envs = ref.pricing_environments
    out = {}
    for pos_id, p in ref.positions.items():
        base = _value(p, envs[p.underlying], True)
        out[pos_id] = np.array([
            _value(p, engine._create_stressed_environment(envs[p.underlying], s), True) - base
            for _, s in scenarios.iterrows()
        ])
    return out


def _euler_components(position_pnls, pnl, confidence):
    portfolio_var = np.var(pnl, ddof=1)
    var_result = -np.percentile(pnl, (1 - confidence) * 100)
    return {
        pos_id: abs(np.cov(pos_pnls, pnl, ddof=1)[0, 1] / portfolio_var * var_result)
        for pos_id, pos_pnls in position_pnls.items()
    }


# --- historical ------------------------------------------------------------------------------------------------

def _historical(**kw):
    config = VaRConfig(var_method=VaRMethod.HISTORICAL, lookback_days=6, confidence_level=0.9,
                       calculate_marginal_var=False, **kw)
    return HistoricalVaREngine(config)


def test_historical_flag_off_is_the_product_path_bitwise():
    pf = _book()
    ref = copy.deepcopy(pf)
    engine = _historical()
    history = _factor_history(6)
    result = engine.calculate_var(pf, history)
    base, pnl = _expected_scenario_pnl(engine, ref, history, trade=False)
    assert result.portfolio_value == base
    assert np.array_equal(result.scenarios["pnl"].to_numpy(), pnl)
    want = _euler_components(_old_component_pnls(engine, copy.deepcopy(pf), history), pnl, 0.9)
    assert result.component_var == want


def test_historical_flag_on_reprices_trade_values():
    pf = _book(trade_value=True)
    ref = copy.deepcopy(pf)
    engine = _historical()
    history = _factor_history(6)
    result = engine.calculate_var(pf, history)
    base, pnl = _expected_scenario_pnl(engine, ref, history, trade=True)
    # AC3 (VaR half): the base value is the portfolio's own trade value, bitwise
    assert result.portfolio_value == copy.deepcopy(pf).get_portfolio_value() == base
    assert np.array_equal(result.scenarios["pnl"].to_numpy(), pnl)
    _, product_pnl = _expected_scenario_pnl(engine, copy.deepcopy(pf), history, trade=False)
    assert not np.array_equal(pnl, product_pnl)  # the legs move with the scenario


def test_historical_flag_on_components_are_trade_valued_and_sum_to_the_total():
    pf = _book(trade_value=True)
    engine = _historical()
    history = _factor_history(6)
    position_pnls = engine._position_scenario_pnls(pf, history)
    want = _trade_component_pnls(engine, copy.deepcopy(pf), history)
    assert set(position_pnls) == set(want)
    for pos_id in want:
        assert np.array_equal(position_pnls[pos_id], want[pos_id]), pos_id
    total = engine._compute_scenario_pnl(pf, history, pf.get_portfolio_value())
    summed = sum(position_pnls.values())
    scale = np.max(np.abs(total))
    assert np.allclose(summed, total, rtol=0.0, atol=1e-12 * scale)
    result = engine.calculate_var(pf, history)
    assert result.component_var == _euler_components(want, result.scenarios["pnl"].to_numpy(), 0.9)


# --- Monte Carlo -----------------------------------------------------------------------------------------------

def _monte_carlo():
    config = VaRConfig(var_method=VaRMethod.MONTE_CARLO, lookback_days=30, confidence_level=0.9,
                       mc_num_simulations=100, mc_seed=11, calculate_marginal_var=False)
    return MonteCarloVaREngine(config)


def _mc_case(trade_value):
    pf = _book(trade_value=trade_value)
    ref = copy.deepcopy(pf)
    engine = _monte_carlo()
    history = _factor_history(30)
    result = engine.calculate_var(pf, history)
    scenarios = engine._generate_scenarios(history.tail(30))
    return pf, ref, engine, result, scenarios


def _standalone(position_pnls, confidence):
    return {pos_id: abs(-np.percentile(p, (1 - confidence) * 100)) for pos_id, p in position_pnls.items()}


def test_monte_carlo_flag_off_is_the_product_path_bitwise():
    pf, ref, engine, result, scenarios = _mc_case(trade_value=False)
    base, pnl = _expected_scenario_pnl(engine, ref, scenarios, trade=False)
    assert result.portfolio_value == base
    assert np.array_equal(result.scenarios["pnl"].to_numpy(), pnl)
    assert result.component_var == _standalone(_old_component_pnls(engine, copy.deepcopy(pf), scenarios), 0.9)


def test_monte_carlo_flag_on_reprices_trade_values_and_components():
    pf, ref, engine, result, scenarios = _mc_case(trade_value=True)
    base, pnl = _expected_scenario_pnl(engine, ref, scenarios, trade=True)
    assert result.portfolio_value == copy.deepcopy(pf).get_portfolio_value() == base
    assert np.array_equal(result.scenarios["pnl"].to_numpy(), pnl)
    want = _trade_component_pnls(engine, copy.deepcopy(pf), scenarios)
    position_pnls = engine._position_scenario_pnls(pf, scenarios)
    for pos_id in want:
        assert np.array_equal(position_pnls[pos_id], want[pos_id]), pos_id
    assert np.allclose(sum(position_pnls.values()), pnl, rtol=0.0, atol=1e-12 * np.max(np.abs(pnl)))
    assert result.component_var == _standalone(want, 0.9)


# --- parametric ------------------------------------------------------------------------------------------------

def _parametric(factors=None, **kw):
    config = VaRConfig(var_method=VaRMethod.PARAMETRIC, lookback_days=30, confidence_level=0.95,
                       equity_factors=factors, **kw)
    return ParametricVaREngine(config)


def test_parametric_flag_off_sensitivities_are_the_product_path_bitwise():
    pf = _book()
    ref = copy.deepcopy(pf)
    envs = ref.pricing_environments
    engine = _parametric()
    got = engine._compute_equity_sensitivities(pf)
    risk = {p.position_id: p.get_risk_measures(envs[p.underlying]) for p in ref.positions.values()}
    delta = 0.0
    for p in ref.positions.values():
        delta += risk[p.position_id]["delta"] * envs[p.underlying].spot
    assert got == {
        "spot_return": delta,
        "vol_change": sum(risk[p.position_id].get("vega", 0.0) for p in ref.positions.values()),
        "rate_shift": sum(risk[p.position_id].get("rho", 0.0) for p in ref.positions.values()),
    }
    result = engine.calculate_var(pf, _factor_history(30))
    total = sum(got.values())
    for pos_id, p in ref.positions.items():
        pos_sens = (risk[pos_id]["delta"] / p.quantity) * envs[p.underlying].spot
        assert result.marginal_var[pos_id] == pytest.approx(abs(pos_sens / total) * result.var, rel=1e-12)


@pytest.mark.parametrize("trade_value", [False, True])
def test_parametric_psi_proxy_only_in_product_mode(monkeypatch, trade_value):
    # The BS proxy itself cannot run today (it imports the nonexistent quantark.param.div_yield); stub it to
    # prove the routing: product mode sums proxy × quantity over option positions, trade mode never calls it.
    calls = []

    def proxy(product, env, bump_size=0.0001):
        calls.append(product)
        return 3.0 if product is pf.positions["vanilla"].product else 5.0

    pf = _book(trade_value=trade_value)
    engine = _parametric(ALL_FACTORS)
    monkeypatch.setattr(engine, "_calculate_div_yield_sensitivity", proxy)
    got = engine._compute_equity_sensitivities(pf)["div_yield_shift"]
    if trade_value:
        assert calls == []
    else:
        assert calls == [pf.positions["vanilla"].product, pf.positions["note"].product]
        assert got == 0.0 + 3.0 * 10 + 5.0 * -2.0


def test_parametric_flag_on_uses_trade_delta_vega_rho():
    pf = _book(trade_value=True)
    ref = copy.deepcopy(pf)
    envs = ref.pricing_environments
    gc = GreeksCalculator()
    engine = _parametric()
    got = engine._compute_equity_sensitivities(pf)
    greeks = {p.position_id: ref.position_greeks(p, gc) for p in ref.positions.values()}
    assert "dividend_rho" in greeks["vanilla"] and "dividend_rho" in greeks["note"]
    delta = 0.0
    for p in ref.positions.values():
        delta += greeks[p.position_id]["delta"] * envs[p.underlying].spot
    assert got == {
        "spot_return": delta,
        "vol_change": sum(greeks[p.position_id].get("vega", 0.0) for p in ref.positions.values()),
        "rate_shift": sum(greeks[p.position_id].get("rho", 0.0) for p in ref.positions.values()),
    }
    # the vanilla's delta is the bump delta, not the analytical one
    analytical = ref.positions["vanilla"].get_risk_measures(envs["X"])["delta"]
    assert greeks["vanilla"]["delta"] != analytical


def test_parametric_flag_on_component_and_marginal_use_trade_delta():
    pf = _book(trade_value=True)
    ref = copy.deepcopy(pf)
    envs = ref.pricing_environments
    gc = GreeksCalculator()
    engine = _parametric()
    result = engine.calculate_var(pf, _factor_history(30))
    sens = engine._compute_equity_sensitivities(copy.deepcopy(pf))
    total = sum(sens.values())
    for pos_id, p in ref.positions.items():
        delta = ref.position_greeks(p, gc)["delta"]
        dollar = delta * envs[p.underlying].spot
        assert result.component_var[pos_id] == pytest.approx(dollar / total * result.var, rel=1e-12)
        per_unit = (delta / p.quantity) * envs[p.underlying].spot
        assert result.marginal_var[pos_id] == pytest.approx(abs(per_unit / total) * result.var, rel=1e-12)


def _trade_value_at_q(position, env, q):
    bumped = copy.deepcopy(env)
    bumped.div_yield = ContinuousDividendYield(q)
    return copy.deepcopy(position).get_trade_value(bumped)


def test_parametric_flag_on_psi_is_trade_dividend_rho_per_unit_of_q():
    pf = _book(trade_value=True)
    ref = copy.deepcopy(pf)
    engine = _parametric(ALL_FACTORS)
    got = engine._compute_equity_sensitivities(pf)["div_yield_shift"]
    # $ per unit change of q: central bump of each option position's trade value (the TRS has no q exposure)
    h = 1e-4
    env = ref.pricing_environments["X"]
    q0 = env.get_div_yield(1.0)
    want = 0.0
    for pos_id in ("vanilla", "note"):
        p = ref.positions[pos_id]
        want += (_trade_value_at_q(p, env, q0 + h) - _trade_value_at_q(p, env, q0 - h)) / (2.0 * h)
    assert got == pytest.approx(want, rel=2e-3)
    # and it is the summed trade dividend_rho (per 1% of q) rescaled to per unit of q
    gc = GreeksCalculator()
    rho_q = sum(ref.position_greeks(ref.positions[i], gc)["dividend_rho"] for i in ("vanilla", "note"))
    assert got == pytest.approx(rho_q / 0.01, rel=1e-12)
