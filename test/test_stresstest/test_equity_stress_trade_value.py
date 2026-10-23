"""Static equity stress at full trade value (ship spec 2026-10-23 D1–D5b, AC1–AC3, AC9a, AC9b).

On an ``EquityPortfolio(trade_value=True)`` (opt-in since the portfolio-trade-value run) every EquityPosition is valued at ``get_trade_value`` (product × signed quantity + absolute cash legs) and
its Greeks come from ``get_trade_risk(...).total``; each (position, environment) is priced once per scenario.
"""
import copy
import pickle
from datetime import datetime

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine, StressType
from quantark.stresstest.equity.valuation import AGGREGATE_GREEK_KEYS, POSITION_GREEK_KEYS
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.util.enum import OptionType

VAL = datetime(2026, 3, 16)
#: a coarse V2 discretisation: these tests check the stress plumbing, not QUAD accuracy
FAST_V2 = QuadV2Params(order=4, cells_per_sd=0.5)


def _env(spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[i / 12 for i in range(1, 13)],
                                                       ki_barrier=75.0, ki_continuous=True))


def _rebate_leg():
    return AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                               notional=1_000.0, rate=0.05, terminal_accrual_factor=1.0,
                               terminal_settlement_time=1.0, accrual_basis=AccrualBasis.KO_MATURITY,
                               terminal_events=frozenset({EventType.MATURITY_NO_KO}))


def _portfolio(snowball_engine=None):
    pf = Portfolio(portfolio_name="book", pricing_environments={"IDX": _env()}, creation_date=VAL, trade_value=True)
    vanilla = EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL)
    pf.add_position(product=vanilla, quantity=10, entry_price=0.0, underlying="IDX", engine=BlackScholesEngine(),
                    entry_timestamp=VAL)
    pf.positions["note"] = EquityPosition(
        product=_snowball(), quantity=-2.0, entry_price=0.0, underlying="IDX",
        engine=snowball_engine or SnowballQuadEngine(params=QuadParams()), entry_timestamp=VAL,
        position_id="note", cash_legs=[_rebate_leg()])
    return pf


def _scenarios():
    return [ScenarioBuilder().name("spot -10%").spot_stress(-0.10).build(),
            ScenarioBuilder().name("vol +5").vol_stress(0.05, stress_type=StressType.ABSOLUTE).build()]


def _expected_position(position, env, gc):
    """What one position contributes under the trade-value definition, on a fresh copy."""
    pos = copy.deepcopy(position)
    value = pos.get_trade_value(env)
    total = copy.deepcopy(position).get_trade_risk(env, gc).total
    T = pos.product.get_maturity(env)
    comps = gc.estimate_theta_components(total["theta"], total["rho"], total["dividend_rho"],
                                         env.get_rate(T), env.get_div_yield(T), T)
    greeks = {"market_value": total["price"], **{k: total[k] for k in
              ("delta", "gamma", "vega", "theta", "rho", "dividend_rho")}, **comps}
    return value, greeks


def test_scalar_stress_values_every_position_at_trade_value():
    pf = _portfolio()
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True))
    results = engine.run_static_scenarios(pf, _scenarios())
    gc = engine.greeks_calculator

    base = [_expected_position(p, pf.pricing_environments["IDX"], gc) for p in pf.positions.values()]
    assert results.baseline_value == sum(v for v, _ in base)
    for scenario, result in zip(_scenarios(), results.scenario_results):
        env = StressApplicator.apply_scenario_to_portfolio(pf, scenario)["IDX"]
        want = [_expected_position(p, env, gc) for p in pf.positions.values()]
        assert result.portfolio_value == sum(v for v, _ in want)
        for row, (value, greeks) in zip(result.position_results, want):
            assert row["stressed_value"] == value
            assert list(row["greeks"]) == list(POSITION_GREEK_KEYS)
            assert row["greeks"] == greeks
        assert list(result.greeks) == list(AGGREGATE_GREEK_KEYS)
        for key in AGGREGATE_GREEK_KEYS:
            assert result.greeks[key] == sum(g[key] for _, g in want)
        assert result.underlying_results["IDX"]["total_value"] == result.portfolio_value
        assert result.underlying_results["IDX"]["greeks"] == result.greeks


def test_cash_leg_is_absolute_in_stress_pv():
    """A short note: the product scales by -2, the rebate leg keeps its own direction and size."""
    pf = _portfolio()
    results = StressTestEngine(StressTestConfig(calculate_greeks=False)).run_static_scenarios(pf, _scenarios()[:1])
    note = next(r for r in results.scenario_results[0].position_results if r["position_id"] == "note")
    env = StressApplicator.apply_scenario_to_portfolio(pf, _scenarios()[0])["IDX"]
    pos = pf.positions["note"]
    product_only = copy.deepcopy(pos).get_market_value(env)
    assert note["stressed_value"] == copy.deepcopy(pos).get_trade_value(env)
    assert note["stressed_value"] != product_only


def test_each_position_is_risked_once_per_scenario(monkeypatch):
    calls = []
    original = EquityPosition.get_trade_risk

    def counting(self, *args, **kwargs):
        calls.append(self.position_id)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EquityPosition, "get_trade_risk", counting)
    pf = _portfolio()
    StressTestEngine(StressTestConfig(calculate_greeks=True)).run_static_scenarios(pf, _scenarios())
    n = len(pf.positions)
    assert len(calls) == n * (1 + len(_scenarios()))      # the baseline + one risk per position per scenario


def test_config_native_batch_is_opt_in_and_reported():
    config = StressTestConfig()
    assert config.native_batch is False
    summary = config.get_summary()
    assert {"native_batch", "parallel_execution", "max_workers"} <= set(summary)


def test_standalone_evaluate_scenario_equals_the_run():
    pf = _portfolio()
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True))
    run = engine.run_static_scenarios(pf, _scenarios())
    alone = StressTestEngine(StressTestConfig(calculate_greeks=True)).evaluate_scenario(
        _portfolio(), _scenarios()[1], run.baseline_value)
    in_run = run.scenario_results[1]
    assert alone.portfolio_value == in_run.portfolio_value
    assert alone.greeks == in_run.greeks
    assert [r["original_value"] for r in alone.position_results] == [r["original_value"] for r in in_run.position_results]


def test_priced_quad_v2_engine_copies_and_pickles_cold():
    engine = SnowballQuadEngineV2(FAST_V2)
    product, env = _snowball(), _env()
    engine.price(product, env)
    fresh = SnowballQuadEngineV2(FAST_V2).price(product, _env(101.0))
    for clone in (copy.deepcopy(engine), pickle.loads(pickle.dumps(engine))):
        assert clone._prepared is None and clone._prepared_key is None
        assert clone.price(product, _env(101.0)) == fresh


def test_scalar_stress_runs_on_a_quad_v2_book():
    pf = _portfolio(snowball_engine=SnowballQuadEngineV2(FAST_V2))
    results = StressTestEngine(StressTestConfig(calculate_greeks=True)).run_static_scenarios(pf, _scenarios())
    assert len(results.scenario_results) == 2
    assert all(r.greeks is not None for r in results.scenario_results)
