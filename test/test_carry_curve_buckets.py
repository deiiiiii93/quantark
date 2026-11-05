"""Pillar-bucket delta / rho_q on a CarryCurve (spec .ship/2026-11-05-snowball-2y-carry-curve C2, AC4, AC6).

A European call maturing exactly at a node of a LOG_LINEAR_DF carry curve
depends on the curve only through that node's forward, so every other pillar
bucket is zero and the node bucket is the Black forward delta DF·N(d1).
"""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime

import pytest
from scipy.stats import norm

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.carry_curve_buckets import (
    CarryCurveBucketReport,
    CarryPillarBucket,
    calculate_carry_curve_buckets,
)
from quantark.param import CarryCurve, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.extrapolation import CarryExtrapolation
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

VD = datetime(2026, 11, 5)
S0, R, SIGMA, K = 100.0, 0.014, 0.22, 98.0
# negative carry (index futures at a discount), like the CSI 1000 chain
NODES = [(0.05, -0.0005), (0.25, -0.012), (0.6, -0.045), (0.85, -0.060)]
LABELS = ["IM1", "IM2", "IM3", "MO1"]


def _env(curve=None):
    curve = curve or CarryCurve(NODES)
    rate = FlatRateCurve(R)
    return PricingEnvironment(
        rate_curve=rate,
        valuation_date=VD,
        spot_quote=SpotQuote(spot=S0),
        vol_surface=FlatVolSurface(volatility=SIGMA),
        div_yield=curve.to_dividend_yield(rate),
    )


def _call_at_node(i):
    return EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=NODES[i][0])


def _analytic_forward_delta(i):
    T, B = NODES[i]
    F = S0 * math.exp(B)
    d1 = (math.log(F / K) + 0.5 * SIGMA**2 * T) / (SIGMA * math.sqrt(T))
    return math.exp(-R * T) * norm.cdf(d1)


def test_european_at_node_tight_bumps_is_analytic_and_local():
    curve = CarryCurve(NODES)
    rep = calculate_carry_curve_buckets(
        _call_at_node(1), _env(curve), BlackScholesEngine(), curve,
        labels=LABELS, spot_bump_rel=1e-4, forward_bump_rel=1e-4, q_bump=1e-4,
    )
    assert isinstance(rep, CarryCurveBucketReport)
    assert all(isinstance(b, CarryPillarBucket) for b in rep.buckets)
    assert [b.label for b in rep.buckets] == LABELS
    assert rep.buckets[1].delta == pytest.approx(_analytic_forward_delta(1), rel=1e-6)
    for j, b in enumerate(rep.buckets):
        assert b.forward == pytest.approx(S0 * math.exp(NODES[j][1]), rel=1e-14)
        assert b.tenor == NODES[j][0]
        assert b.hedge_lots is None
        assert b.extrapolated_tail is False          # product matures inside the curve
        if j != 1:
            assert abs(b.delta) <= 1e-10
            assert abs(b.rho_q) <= 1e-10
    assert abs(rep.pinned_forward_delta) <= 1e-10
    assert abs(rep.delta_identity_residual) <= 1e-6 * abs(rep.spot_delta)
    assert abs(rep.rho_sum_residual) <= 1e-6 * abs(rep.parallel_rho_q)
    # rho_q per +1% q at node 2: dPV/dF · dF/dq = delta · (−F·T) · 1%
    T, B = NODES[1]
    assert rep.buckets[1].rho_q == pytest.approx(
        _analytic_forward_delta(1) * (-S0 * math.exp(B) * T) * 0.01, rel=1e-4
    )


def test_european_at_node_default_bumps_h2_tolerances():
    curve = CarryCurve(NODES)
    rep = calculate_carry_curve_buckets(_call_at_node(1), _env(curve), BlackScholesEngine(), curve)
    assert rep.buckets[1].delta == pytest.approx(_analytic_forward_delta(1), rel=1e-5)
    assert abs(rep.delta_identity_residual) <= 1e-3 * abs(rep.spot_delta)


def test_hedge_lots_and_tail_flags_per_long_end_rule():
    # product beyond the last node: the flag follows the nodes the long-end rule reads
    option = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=2.0)
    expected = {
        CarryExtrapolation.FLAT_FORWARD_CARRY: [False, False, True, True],
        CarryExtrapolation.FLAT_CARRY_YIELD: [False, False, False, True],
        CarryExtrapolation.ZERO_FORWARD_CARRY: [False, False, False, True],
    }
    for ext, flags in expected.items():
        curve = CarryCurve(NODES, extrapolation=ext)
        rep = calculate_carry_curve_buckets(
            option, _env(curve), BlackScholesEngine(), curve, lot_multipliers=[200.0, 200.0, 200.0, 100.0]
        )
        assert [b.extrapolated_tail for b in rep.buckets] == flags
        for b, m in zip(rep.buckets, [200.0, 200.0, 200.0, 100.0]):
            assert b.hedge_lots == pytest.approx(-b.delta / m, rel=1e-15)
        # chain rule holds for every tail
        assert abs(rep.delta_identity_residual) <= 1e-3 * abs(rep.spot_delta)


def test_rho_sum_equals_parallel_on_node_shift_tails():
    option = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=2.0)
    for ext in (CarryExtrapolation.FLAT_FORWARD_CARRY, CarryExtrapolation.FLAT_CARRY_YIELD):
        curve = CarryCurve(NODES, extrapolation=ext)
        # exact identity: tight step; the beyond-T_n leverage (B(T) reads the last
        # nodes with weights ~(T−T_n)/ΔT) makes the O(ε²) term visible at the default
        tight = calculate_carry_curve_buckets(option, _env(curve), BlackScholesEngine(), curve, q_bump=1e-5)
        assert abs(tight.rho_sum_residual) <= 1e-6 * abs(tight.parallel_rho_q)
        rep = calculate_carry_curve_buckets(option, _env(curve), BlackScholesEngine(), curve)
        assert abs(rep.rho_sum_residual) <= 1e-3 * abs(rep.parallel_rho_q)


def test_validation_errors():
    curve = CarryCurve(NODES)
    env, eng, opt = _env(curve), BlackScholesEngine(), _call_at_node(1)
    bad = [
        dict(spot_bump_rel=0.0), dict(forward_bump_rel=-1e-3), dict(q_bump=0.0),
        dict(labels=["a", "b"]), dict(lot_multipliers=[200.0] * 3),
        dict(lot_multipliers=[200.0, 200.0, 0.0, 100.0]),
    ]
    for kw in bad:
        with pytest.raises(ValidationError):
            calculate_carry_curve_buckets(opt, env, eng, curve, **kw)
    with pytest.raises(ValidationError):
        calculate_carry_curve_buckets(opt, env, eng, object())
    dated = CarryCurve(NODES, valuation_date=datetime(2026, 11, 6))
    with pytest.raises(ValidationError):
        calculate_carry_curve_buckets(opt, env, eng, dated)


def test_caller_env_not_mutated():
    curve = CarryCurve(NODES)
    env = _env(curve)
    before_spot, before_div = env.spot_quote.spot, env.div_yield
    snapshot = deepcopy(env)
    calculate_carry_curve_buckets(_call_at_node(2), env, BlackScholesEngine(), curve)
    assert env.spot_quote.spot == before_spot
    assert env.div_yield is before_div
    assert env.get_div_yield(0.6) == snapshot.get_div_yield(0.6)


def test_pde_engine_runs_through_bump_context():
    from quantark.asset.equity.engine import PDEEngine

    curve = CarryCurve(NODES)
    rep = calculate_carry_curve_buckets(_call_at_node(1), _env(curve), PDEEngine(), curve, labels=LABELS)
    assert rep.buckets[1].delta == pytest.approx(_analytic_forward_delta(1), rel=1e-3)
    assert abs(rep.buckets[0].delta) <= 1e-3 * abs(rep.buckets[1].delta)
