"""Trade-mode replay hedges the whole trade (user decision 2026-10-23; ship spec portfolio-trade-value D9, AC11).

- The single-contract hedge's leg delta / gamma are the trade-risk total less the product's, bumped with the
  engine's own spot and gamma bumps, and agree with a central difference of the legs' value.
- A futures-bucket hedge measures the book's carry risk on the trade value (product + legs), so the bucket plan
  hedges the legs too; the book audit prices the same trade value.
"""
import copy
from dataclasses import replace

import pandas as pd
import pytest

import bucket_hedge_fixtures as fixtures
from quantark.backtest.replay import ReplayBacktestEngine, ReplayBacktestConfig, ReplayProduct
from quantark.backtest.replay.engine import _book_price_callback
from quantark.cashleg import LegDirection
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from test_backtest import test_replay_trade_value as replay_fx
from test_backtest.test_replay_futures_buckets import make_engine, run_capturing_risk


def _ko_payoff():
    return FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=50_000.0, trigger=PaymentTrigger.AT_KO)


# --- single-contract hedge: bump basis and a finite-difference check ----------------------------------------------

def test_leg_greeks_use_the_engines_own_spot_and_gamma_bumps():
    engine = ReplayBacktestEngine(replace(replay_fx._config(), delta_bump_size=0.003, gamma_bump_size=0.02))
    engine.run()
    pricing = engine._pricing_engines[0].params.get_effective_bump_config()
    delta_gc, gamma_gc = engine._leg_greeks_calculators(engine._pricing_engines[0])
    assert delta_gc._bump_config.spot_bump == pricing.spot_bump == 0.003
    assert gamma_gc._bump_config.spot_bump == pricing.gamma_spot_bump == 0.02


def test_leg_delta_matches_a_central_difference_of_the_legs_value():
    engine = ReplayBacktestEngine(replay_fx._config())
    captured = {}
    replay = engine._replays[0]
    original = replay.calculate_greeks

    def hooked(product, env, *, engine=None):
        if "env" not in captured and 0 in engine_obj._leg_positions:
            captured.update(product=product, env=copy.deepcopy(env),
                            legs=list(engine_obj._leg_positions[0].cash_legs), engine=engine)
        return original(product, env, engine=engine)

    engine_obj = engine
    replay.calculate_greeks = hooked
    states = engine.run().states_df().set_index("date")
    env, product, legs, pricing = captured["env"], captured["product"], captured["legs"], captured["engine"]
    position = copy.deepcopy(engine._leg_positions[0])
    position.product, position.engine, position.cash_legs = product, pricing, legs

    def legs_value(spot):
        bumped = copy.deepcopy(env)
        bumped.spot_quote.spot = spot
        return position.get_trade_value(bumped) - position.quantity * pricing.price_with_events(
            product, bumped, emit_distribution=False).npv

    h = pricing.params.get_effective_bump_config().spot_bump * env.spot
    central = (legs_value(env.spot + h) - legs_value(env.spot - h)) / (2.0 * h)
    first = pd.Timestamp(env.valuation_date).normalize()
    assert states.loc[first, "leg_delta"] == pytest.approx(central, rel=1e-6, abs=1e-9)


# --- futures-bucket hedge: carry risk on the trade value ----------------------------------------------------------

def _bucket_engine(legs, trade_value=True, quantities=(-1.0,)):
    engine = make_engine(quantities=quantities, trade_value=trade_value)
    engine.config.products[:] = [replace(bp, cash_legs=list(legs)) for bp in engine.config.products]
    return ReplayBacktestEngine(engine.config)


def test_bucket_hedge_measures_the_trade_values_carry_risk():
    _, with_legs, _ = run_capturing_risk(_bucket_engine([_ko_payoff()]))
    _, without, _ = run_capturing_risk(_bucket_engine([]))
    common = sorted(set(with_legs) & set(without))
    assert common
    assert any(abs(with_legs[d].delta_q - without[d].delta_q) > 1e-9 * max(1.0, abs(without[d].delta_q))
               for d in common)


def test_bucket_book_audit_prices_the_trade_value():
    engine = _bucket_engine([_ko_payoff()])
    seen = {}
    original = engine._measure_book_carry_risk

    def measure(date, env, context, alive_specs):
        price_at = _book_price_callback(alive_specs, env)
        seen[pd.Timestamp(date)] = (price_at(float(env.spot), env.div_yield), list(alive_specs))
        return original(date, env, context, alive_specs)

    engine._measure_book_carry_risk = measure
    states = engine.run().states_df().set_index("date")
    assert seen
    for date, (audited, _specs) in seen.items():
        assert audited == pytest.approx(states.loc[date, "product_mtm"], rel=1e-12)


def test_bucket_hedge_without_trade_value_is_unchanged_by_legs():
    _, a, _ = run_capturing_risk(_bucket_engine([_ko_payoff()], trade_value=False))
    _, b, _ = run_capturing_risk(_bucket_engine([], trade_value=False))
    assert set(a) == set(b)
    for d in a:
        assert a[d].delta_q == b[d].delta_q
