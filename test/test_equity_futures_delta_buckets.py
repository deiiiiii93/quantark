"""Futures-tenor bucket Greeks (spec tests 3-8) + futures rhoq (5, 6)."""
import math
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.deltaone_engine import DeltaOneEngine
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def _env(spot=5000.0, r=0.03, q=0.01, vol=0.20):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


# --- spec test 5: theoretical futures rhoq ---

def test_futures_theoretical_dividend_rho():
    env = _env()
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5)
    greeks = DeltaOneEngine().calculate_greeks(fut, env)
    S, T, r, q = 5000.0, 0.5, 0.03, 0.01
    expected = -S * T * math.exp((r - q) * T) * 0.01
    assert greeks["dividend_rho"] == pytest.approx(expected, rel=1e-12)
    assert greeks["dividend_rho"] < 0.0  # long theoretical futures: rhoq < 0


# --- spec test 6: market-price mode rhoq at the mark's implied carry ---

def test_futures_market_price_dividend_rho_from_implied_carry():
    env = _env()  # spot 5000, r 3%
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.5, market_price=5100.0)
    greeks = DeltaOneEngine(use_market_price=True).calculate_greeks(fut, env)
    S, T, r, F = 5000.0, 0.5, 0.03, 5100.0
    implied_q = r - math.log(F / S) / T
    expected = -S * T * math.exp((r - implied_q) * T) * 0.01
    assert greeks["dividend_rho"] == pytest.approx(expected, rel=1e-12)
    # at the implied carry the formula collapses to -T * F_mkt * 0.01
    assert greeks["dividend_rho"] == pytest.approx(-T * F * 0.01, rel=1e-12)
    assert greeks["dividend_rho"] < 0.0


from copy import deepcopy

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.util.enum import FuturesCarryRiskMode, OptionType


def _ic_curve(spot=5000.0):
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
            IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
            IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
        ],
    )


# --- spec test 3: vanilla futures-delta bucket matches manual finite difference ---

def test_futures_delta_bucket_matches_manual_fd():
    env = _env()
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)

    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, engine, curve, price_bump=1.0
    )
    row = next(r for r in rows if r["contract"] == "IC01")

    base_env = deepcopy(env)
    base_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)
    bumped_env = deepcopy(env)
    bumped_env.div_yield = curve.bump_contract("IC01", 1.0).to_dividend_yield_curve(
        env.rate_curve
    )
    manual = engine.price(option, bumped_env) - engine.price(option, base_env)
    assert row["delta_bucket"] == pytest.approx(manual / 1.0, rel=1e-12)
    assert row["hedge_hands"] == pytest.approx(-row["delta_bucket"] / 200.0, rel=1e-12)
    assert row["delta_per_hand"] == 200.0
    assert row["extrapolated_tail"] is False


def test_futures_delta_buckets_row_shape_and_signs():
    env = _env()
    curve = _ic_curve()
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10),
        env,
        BlackScholesEngine(),
        curve,
    )
    assert [r["contract"] for r in rows] == ["IC00", "IC01", "IC02", "IC03"]
    # long call: positive futures exposure => short-futures hedge on active buckets
    active = [r for r in rows if abs(r["delta_bucket"]) > 1e-10]
    assert active and all(r["hedge_hands"] < 0 for r in active)


def test_futures_delta_buckets_mode_rejection():
    env = _env()
    curve = _ic_curve()
    calc = GreeksCalculator()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.MARKET_PRICE,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.THEORETICAL_CARRY,
        )
    with pytest.raises(ValidationError):
        calc.calculate_futures_delta_buckets(
            option, env, BlackScholesEngine(), curve, price_bump=0.0
        )


# --- rhoq buckets: implied node bump / theoretical BucketedDividendYield ---

def test_rhoq_buckets_implied_mode_matches_manual_node_bump():
    from quantark.asset.equity.market import bump_term_yield_node

    env = _env()
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    calc = GreeksCalculator()

    rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve, div_bump=0.0001
    )
    row = next(r for r in rows if r["contract"] == "IC01")

    base_div = curve.to_dividend_yield_curve(env.rate_curve)
    base_env = deepcopy(env)
    base_env.div_yield = base_div
    bumped_env = deepcopy(env)
    bumped_env.div_yield = bump_term_yield_node(base_div, 1, 0.0001)
    manual = (
        engine.price(option, bumped_env) - engine.price(option, base_env)
    ) * (0.01 / 0.0001)
    assert row["rhoq_bucket"] == pytest.approx(manual, rel=1e-12)
    assert row["rhoq_bucket"] < 0.0  # call: higher carry lowers forward


def test_rhoq_buckets_theoretical_mode_uses_bucketed_dividend():
    env = _env(q=0.01)
    curve = _ic_curve()
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10)
    calc = GreeksCalculator()

    rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve,
        mode=FuturesCarryRiskMode.THEORETICAL_CARRY, div_bump=0.0001,
    )
    # option matures at 0.10 = IC01 node: spot-yield q(0.10) sits in the
    # (0.03, 0.10] bucket, so only IC01's interval bump moves the PV
    by_contract = {r["contract"]: r["rhoq_bucket"] for r in rows}
    assert by_contract["IC01"] != pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC00"] == pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC02"] == pytest.approx(0.0, abs=1e-9)
    assert by_contract["IC03"] == pytest.approx(0.0, abs=1e-9)
    # bucket rows decompose the scalar rhoq: sum == scalar dividend_rho
    scalar = calc.calculate_numerical_dividend_rho(
        option, env, engine, div_bump=0.0001
    )
    assert sum(by_contract.values()) == pytest.approx(scalar, rel=1e-6)


def test_rhoq_buckets_theoretical_tail_beyond_last_node_not_dropped():
    # codex code-review regression: a product maturing beyond the last futures
    # tenor must keep its tail dividend exposure in the last bucket, not lose it
    env = _env(spot=100.0, q=0.01)
    curve = _short_curve()  # nodes [0.1, 0.3, 0.6]
    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    calc = GreeksCalculator()

    rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve,
        mode=FuturesCarryRiskMode.THEORETICAL_CARRY, div_bump=0.0001,
    )
    scalar = calc.calculate_numerical_dividend_rho(
        option, env, engine, div_bump=0.0001
    )
    assert abs(scalar) > 1e-6
    assert sum(r["rhoq_bucket"] for r in rows) == pytest.approx(scalar, rel=1e-6)
    assert abs(rows[-1]["rhoq_bucket"]) > 1e-6  # tail lands in the last bucket
    assert [r["extrapolated_tail"] for r in rows] == [False, False, True]


def test_rhoq_buckets_market_price_mode_rejected():
    env = _env()
    curve = _ic_curve()
    with pytest.raises(ValidationError):
        GreeksCalculator().calculate_futures_rhoq_buckets(
            EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.10),
            env, BlackScholesEngine(), curve,
            mode=FuturesCarryRiskMode.MARKET_PRICE,
        )


# --- spec test 8: extrapolated tail concentrates in the last contract ---

def _short_curve(spot=100.0, r=0.03):
    times, qs = [0.1, 0.3, 0.6], [0.01, 0.015, 0.012]
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote(
                f"IC{i:02d}", maturity=t, price=spot * math.exp((r - q) * t),
                multiplier=200.0,
            )
            for i, (t, q) in enumerate(zip(times, qs))
        ],
    )


def test_european_beyond_last_node_all_delta_in_last_bucket():
    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    assert rows[0]["delta_bucket"] == pytest.approx(0.0, abs=1e-12)
    assert rows[1]["delta_bucket"] == pytest.approx(0.0, abs=1e-12)
    assert abs(rows[2]["delta_bucket"]) > 1e-4
    assert [r["extrapolated_tail"] for r in rows] == [False, False, True]


def test_snowball_keeps_interior_node_sensitivity_beyond_last_node():
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    # barriers are ABSOLUTE levels (103/75 on spot 100), not ratios — a
    # ko_barrier below spot would mean an already-knocked-out product with a
    # deterministic PV and zero carry sensitivity
    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    env = _env(spot=100.0)
    curve = _short_curve()  # T_last = 0.6 < snowball maturity 1.0
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        snowball, env, SnowballQuadEngine(), curve
    )
    # KO observations at 0.25/0.5 sit inside [0.1, 0.6]: interior nodes
    # carry genuine sensitivity through the term-aware engine
    interior = [r for r in rows if not r["extrapolated_tail"]]
    assert any(abs(r["delta_bucket"]) > 1e-6 for r in interior)
    assert rows[-1]["extrapolated_tail"] is True


def test_first_bucket_flagged_when_maturity_before_first_node():
    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.05)
    rows = GreeksCalculator().calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    assert rows[0]["extrapolated_tail"] is True
    assert rows[1]["extrapolated_tail"] is False


# --- MC common random numbers ---

def test_mc_delta_buckets_deterministic_and_near_analytic():
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams

    env = _env(spot=100.0)
    curve = _short_curve()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.3)
    calc = GreeksCalculator()

    mc_engine = EuropeanMCEngine(MCParams(seed=42, num_paths=100_000))
    rows_a = calc.calculate_futures_delta_buckets(option, env, mc_engine, curve)
    rows_b = calc.calculate_futures_delta_buckets(option, env, mc_engine, curve)
    # fixed seed => common random numbers => bit-identical reruns
    assert [r["delta_bucket"] for r in rows_a] == [
        r["delta_bucket"] for r in rows_b
    ]

    analytic = calc.calculate_futures_delta_buckets(
        option, env, BlackScholesEngine(), curve
    )
    mc_mid, bs_mid = rows_a[1]["delta_bucket"], analytic[1]["delta_bucket"]
    assert mc_mid == pytest.approx(bs_mid, rel=0.05)


# --- portfolio aggregation ---

def test_aggregate_futures_delta_buckets():
    from quantark.portfolio.equity import aggregate_futures_delta_buckets

    row = {
        "contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
        "price_bump": 1.0, "delta_bucket": 600.0, "delta_per_hand": 300.0,
        "hedge_hands": -2.0, "extrapolated_tail": False,
    }
    other = dict(row, delta_bucket=300.0, hedge_hands=-1.0)
    out = aggregate_futures_delta_buckets({"pos_a": [row], "pos_b": [other]})
    assert len(out) == 1
    agg = out[0]
    assert agg["contract"] == "IC01"
    assert agg["delta_bucket"] == pytest.approx(900.0)
    assert agg["hedge_hands"] == pytest.approx(-3.0)
    assert agg["delta_per_hand"] == 300.0
    assert agg["maturity"] == 0.10


def test_aggregate_rejects_incompatible_metadata():
    from quantark.portfolio.equity import aggregate_futures_delta_buckets

    row = {
        "contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
        "delta_bucket": 600.0, "delta_per_hand": 300.0, "hedge_hands": -2.0,
    }
    clash = dict(row, future_price=5021.0)  # different curve mark
    with pytest.raises(ValidationError):
        aggregate_futures_delta_buckets({"a": [row], "b": [clash]})


def test_aggregate_rhoq_buckets_multiple_contracts_keeps_order():
    from quantark.portfolio.equity import aggregate_futures_rhoq_buckets

    r1 = {"contract": "IC00", "maturity": 0.03, "future_price": 5008.0,
          "div_bump": 1e-4, "rhoq_bucket": -1.0}
    r2 = {"contract": "IC01", "maturity": 0.10, "future_price": 5020.0,
          "div_bump": 1e-4, "rhoq_bucket": -2.0}
    out = aggregate_futures_rhoq_buckets({"a": [r1, r2], "b": [dict(r2, rhoq_bucket=-3.0)]})
    assert [r["contract"] for r in out] == ["IC00", "IC01"]
    assert out[1]["rhoq_bucket"] == pytest.approx(-5.0)


# --- spec test 7: portfolio net rhoq offset by futures hedge ---

def test_portfolio_net_rhoq_offset_by_futures_hedge():
    env = _env()  # spot 5000, r 3%
    curve = _ic_curve()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    # maturity 0.18 == IC02 node: risk concentrates on one bucket
    option = EuropeanVanillaOption(5000.0, OptionType.CALL, maturity=0.18)

    delta_rows = calc.calculate_futures_delta_buckets(option, env, engine, curve)
    rhoq_rows = calc.calculate_futures_rhoq_buckets(
        option, env, engine, curve, div_bump=0.0001
    )
    row_d = next(r for r in delta_rows if r["contract"] == "IC02")
    row_q = next(r for r in rhoq_rows if r["contract"] == "IC02")
    assert abs(row_q["rhoq_bucket"]) > 1e-6  # standalone option rhoq non-zero

    # hedge futures: theoretical carry under the implied curve
    hedge_env = deepcopy(env)
    hedge_env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)
    fut = Futures(underlying="IC", multiplier=1.0, maturity=0.18)
    fut_rhoq_per_hand = (
        DeltaOneEngine().calculate_greeks(fut, hedge_env)["dividend_rho"] * 200.0
    )
    assert fut_rhoq_per_hand < 0.0  # long futures rhoq negative

    hands = row_d["hedge_hands"]
    # net bucket delta ~ 0 by construction of hedge_hands
    net_delta = row_d["delta_bucket"] + hands * row_d["delta_per_hand"]
    assert net_delta == pytest.approx(0.0, abs=1e-9)
    # net rhoq reduced relative to standalone option rhoq
    net_rhoq = row_q["rhoq_bucket"] + hands * fut_rhoq_per_hand
    assert abs(net_rhoq) < abs(row_q["rhoq_bucket"])
