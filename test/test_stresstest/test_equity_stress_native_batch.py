"""Static equity stress with native batching vs the scalar path (ship spec 2026-10-23 D6, D9, D14, AC6–AC11).

QUAD V2 positions share prepared contexts across scenarios (equal to the scalar path within the pinned V2
ceilings); every other position is valued exactly as in scalar mode (bitwise).
"""
import copy
from datetime import datetime

import pytest

import quantark.asset.equity.engine.quad.v2.engine as v2_engine
from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import FoldMode
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine, StressType
from quantark.stresstest.equity.valuation import (
    AGGREGATE_GREEK_KEYS, POSITION_GREEK_KEYS, supports_native_batch,
)
from quantark.util.enum import OptionType

from test_asset.equity.product_other.test_equity_position_batch import (
    FIRST_REL, GAMMA_REL, PV_REL, _phoenix, _rebate_leg, _snowball,
)
from test_stresstest.test_trs_stress import _make_trs

VAL = datetime(2026, 3, 16)


def _env(spot=100.0, vol=0.25):
    return PricingEnvironment(spot_quote=SpotQuote(spot, asset_name="X"), vol_surface=FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _trs_env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=4000.0, asset_name="IDX"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.048), valuation_date=datetime(2024, 1, 30))


def _plain_snowball():
    """A continuous-KI snowball valued between observation dates."""
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=0.5, contract_multiplier=10_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[1 / 6, 2 / 6, 0.5],
                                                       ki_barrier=75.0, ki_continuous=True))


def _add(pf, position_id, product, engine, quantity=-1.0, legs=(), underlying="X"):
    pf.positions[position_id] = EquityPosition(product=product, quantity=quantity, entry_price=0.0,
                                               underlying=underlying, engine=engine, entry_timestamp=VAL,
                                               position_id=position_id, cash_legs=list(legs))


def _book():
    pf = Portfolio(portfolio_name="book", pricing_environments={"X": _env(), "IDX": _trs_env()}, creation_date=VAL)
    knocked = _snowball()
    knocked._otc_lifecycle_knocked_in = True
    _add(pf, "v2_snowball_obs", _snowball(), SnowballQuadEngineV2(), quantity=-2.0)
    _add(pf, "v2_phoenix_obs_leg", _phoenix(), PhoenixQuadEngineV2(), legs=(_rebate_leg(),))
    _add(pf, "v2_knocked_in", knocked, SnowballQuadEngineV2(), quantity=1.0)
    _add(pf, "pde_snowball", _plain_snowball(), SnowballPDESolver(PDEParams()))
    _add(pf, "v1_snowball", _plain_snowball(), SnowballQuadEngine(params=QuadParams()), quantity=3.0)
    _add(pf, "v1_phoenix_obs", _phoenix(), PhoenixQuadEngine(params=QuadParams()))
    _add(pf, "vanilla", EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL),
         BlackScholesEngine(), quantity=10.0)
    swap = pf.positions.pop(pf.add_swap_position(product=_make_trs(), quantity=1.0).position_id)
    swap.position_id = "trs"                              # a fixed id, so two books' rows line up
    pf.positions["trs"] = swap
    return pf


def _scenarios():
    """Spot shocks across KI (-30%), the coupon level (-12%) and KO at t0 (+5%); vol, rate and a crash."""
    return [ScenarioBuilder().name("spot -30%").spot_stress(-0.30).build(),
            ScenarioBuilder().name("spot -12%").spot_stress(-0.12).build(),
            ScenarioBuilder().name("spot +5%").spot_stress(0.05).build(),
            ScenarioBuilder().name("vol +5").vol_stress(0.05, stress_type=StressType.ABSOLUTE).build(),
            ScenarioBuilder().name("rate +100bp").rate_stress(0.01, stress_type=StressType.ABSOLUTE).build(),
            ScenarioBuilder().name("crash").spot_stress(-0.20).vol_stress(0.10, stress_type=StressType.ABSOLUTE).build()]


def _run(book, scenarios, *, native_batch, greeks=True):
    config = StressTestConfig(calculate_greeks=greeks, native_batch=native_batch)
    return StressTestEngine(config).run_static_scenarios(book, scenarios)


def _greek_tol(name, want, notional, spot):
    rel = {"market_value": PV_REL, "gamma": GAMMA_REL}.get(name, FIRST_REL)
    unit = {"delta": spot, "gamma": spot ** 2}.get(name, 1.0)
    return rel * abs(want) + 1e-10 * notional / unit


def _compare(scalar, batch, book):
    """Non-V2 rows bitwise; V2 rows within the pinned ceilings; aggregates within the sum of row ceilings."""
    assert batch.baseline_value == scalar.baseline_value
    assert batch.baseline_greeks == scalar.baseline_greeks
    assert [r.scenario.name for r in batch.scenario_results] == [r.scenario.name for r in scalar.scenario_results]
    for s_res, b_res in zip(scalar.scenario_results, batch.scenario_results):
        _compare_scenario(s_res, b_res, book)


def _compare_scenario(s_res, b_res, book):
    """One scenario: non-V2 rows bitwise, V2 rows within the ceilings, totals within their sum."""
    budget = {key: 0.0 for key in ("value",) + AGGREGATE_GREEK_KEYS}
    for s_row, b_row in zip(s_res.position_results, b_res.position_results):
        assert s_row["position_id"] == b_row["position_id"]
        position = book.positions[s_row["position_id"]]
        if not supports_native_batch(position):
            assert b_row == s_row, s_row["position_id"]
            continue
        env = book.pricing_environments[position.underlying]
        notional = abs(position.get_actual_notional(env))
        spot = env.spot
        tol = PV_REL * max(abs(s_row["stressed_value"]), notional)
        assert abs(b_row["stressed_value"] - s_row["stressed_value"]) <= tol, s_row["position_id"]
        budget["value"] += tol
        assert list(b_row["greeks"]) == list(POSITION_GREEK_KEYS)
        for name, want in s_row["greeks"].items():
            t = _greek_tol(name, want, notional, spot)
            assert abs(b_row["greeks"][name] - want) <= t, (s_row["position_id"], s_res.scenario.name, name)
            if name in budget:
                budget[name] += t
    assert abs(b_res.portfolio_value - s_res.portfolio_value) <= budget["value"] + 1e-12 * abs(s_res.portfolio_value)
    for name in AGGREGATE_GREEK_KEYS:
        assert abs(b_res.greeks[name] - s_res.greeks[name]) <= budget[name] + 1e-12 * abs(s_res.greeks[name])
    for underlying, s_u in s_res.underlying_results.items():
        b_u = b_res.underlying_results[underlying]
        assert b_u["num_positions"] == s_u["num_positions"]
        assert abs(b_u["total_value"] - s_u["total_value"]) <= budget["value"] + 1e-12 * abs(s_u["total_value"])


@pytest.fixture(scope="module")
def runs():
    return _run(_book(), _scenarios(), native_batch=False), _run(_book(), _scenarios(), native_batch=True)


def test_native_batch_matches_scalar_on_a_mixed_book(runs):
    scalar, batch = runs
    _compare(scalar, batch, _book())


def test_result_schemas_and_order_are_stable(runs):
    scalar, batch = runs
    assert [r.scenario.name for r in batch.scenario_results] == [s.name for s in _scenarios()]
    assert scalar.metadata["valuation_mode"] == "scalar" and batch.metadata["valuation_mode"] == "native_batch"
    assert list(batch.to_summary_dataframe().columns) == list(scalar.to_summary_dataframe().columns)
    for s_res, b_res in zip(scalar.scenario_results, batch.scenario_results):
        assert set(b_res.to_dict()) == set(s_res.to_dict())
        assert list(b_res.greeks) == list(s_res.greeks) == list(AGGREGATE_GREEK_KEYS)
        assert [list(r) for r in b_res.position_results] == [list(r) for r in s_res.position_results]
        assert list(b_res.underlying_results) == list(s_res.underlying_results)
        assert b_res.metadata["valuation_mode"] == "native_batch"
        assert b_res.execution_time >= 0.0


def test_native_batch_without_greeks_matches_scalar():
    scalar = _run(_book(), _scenarios()[:3], native_batch=False, greeks=False)
    batch = _run(_book(), _scenarios()[:3], native_batch=True, greeks=False)
    for s_res, b_res in zip(scalar.scenario_results, batch.scenario_results):
        assert b_res.greeks is None and s_res.greeks is None
        for s_row, b_row in zip(s_res.position_results, b_res.position_results):
            assert abs(b_row["stressed_value"] - s_row["stressed_value"]) <= PV_REL * max(abs(s_row["stressed_value"]), 1e6)


def test_standalone_evaluate_scenario_honours_native_batch(runs):
    _, batch = runs
    engine = StressTestEngine(StressTestConfig(calculate_greeks=True, native_batch=True))
    for k in (1, 3):
        alone = engine.evaluate_scenario(_book(), _scenarios()[k], batch.baseline_value)
        assert alone.metadata["valuation_mode"] == "native_batch"
        _compare_scenario(batch.scenario_results[k], alone, _book())   # two session histories: V2 within ceilings


@pytest.mark.parametrize("mode", [FoldMode.TOTAL, FoldMode.DRILLDOWN])
def test_folded_cash_legs_batch_matches_scalar(mode):
    from test_cashleg._autocallable_helpers import make_env, make_snowball
    from test_cashleg.test_folding import _leg_set, _route_dist

    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    legs = _leg_set(_route_dist(SnowballPDESolver(PDEParams()), product, env).event_times)

    def book():
        pf = Portfolio(portfolio_name="folded", pricing_environments={"X": copy.deepcopy(env)}, creation_date=VAL)
        _add(pf, "folded", copy.deepcopy(product), SnowballQuadEngineV2(), quantity=2.0, legs=copy.deepcopy(legs))
        pf.positions["folded"].cash_leg_folding = mode
        return pf

    scenarios = _scenarios()[1:3]
    _compare(_run(book(), scenarios, native_batch=False), _run(book(), scenarios, native_batch=True), book())


def test_spot_vol_grid_prepares_scale_with_vol_points(monkeypatch):
    """User 2026-10-23: a spot × vol grid needs a context set per vol level, not per (spot, vol) cell."""
    calls = []
    original = v2_engine.prepare_compiled

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_engine, "prepare_compiled", counting)

    def grid(spots):
        return [ScenarioBuilder().name(f"s{s:+.2f} v{v:+.2f}").spot_stress(s)
                .vol_stress(v, stress_type=StressType.ABSOLUTE).build()
                for v in (-0.05, 0.0, 0.05) for s in spots]

    def book():
        pf = Portfolio(portfolio_name="grid", pricing_environments={"X": _env()}, creation_date=VAL)
        _add(pf, "v2", _plain_snowball(), SnowballQuadEngineV2())
        return pf

    def prepares(spots, native_batch=True):
        calls.clear()
        result = _run(book(), grid(spots), native_batch=native_batch)
        return len(calls), result

    five_spots = [-0.10, -0.05, 0.0, 0.05, 0.10]
    five, batched = prepares(five_spots)
    nine, _ = prepares([-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10])
    scalar_count, scalar = prepares(five_spots, native_batch=False)
    assert nine == five                                     # more spots: no more solves
    assert five <= 3 * 12 + 12                              # per vol level a bounded context set (+ the baseline)
    assert five < scalar_count
    _compare(scalar, batched, book())                       # the shared wide contexts price every cell


def test_unsupported_market_raises_the_same_in_both_modes():
    from quantark.param.vol.vol_surface import GridVolSurface
    import numpy as np

    def book():
        env = _env()
        env.vol_surface = GridVolSurface(strikes=[60.0, 100.0, 140.0], maturities=[0.25, 1.0],
                                         iv_grid=np.array([[0.3, 0.25, 0.24]] * 2))
        pf = Portfolio(portfolio_name="smile", pricing_environments={"X": env}, creation_date=VAL)
        _add(pf, "v2", _plain_snowball(), SnowballQuadEngineV2())
        return pf

    with pytest.raises(NotImplementedError) as scalar:
        _run(book(), _scenarios()[:1], native_batch=False)
    with pytest.raises(NotImplementedError) as batch:
        _run(book(), _scenarios()[:1], native_batch=True)
    assert str(batch.value) == str(scalar.value)


def test_wide_prepare_over_the_limit_falls_back_and_matches_scalar():
    probe = SnowballQuadEngineV2()
    narrow = probe.prepare(_plain_snowball(), _env()).diagnostics["nodes"]
    wide = probe.prepare(_plain_snowball(), _env(), [60.0, 120.0]).diagnostics["nodes"]
    params = QuadV2Params(max_nodes=(narrow + wide) // 2)

    def book():
        pf = Portfolio(portfolio_name="limit", pricing_environments={"X": _env()}, creation_date=VAL)
        _add(pf, "v2", _plain_snowball(), SnowballQuadEngineV2(params))
        return pf

    scenarios = _scenarios()[:3]
    _compare(_run(book(), scenarios, native_batch=False), _run(book(), scenarios, native_batch=True), book())
