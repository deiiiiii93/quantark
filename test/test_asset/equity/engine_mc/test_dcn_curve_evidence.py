"""Q2 acceptance evidence (spec WP3.4): canonical flat mapping, flat
recovery, and 'the engine really uses the curve'."""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.priceenv.flat_builders import build_flat_curves, build_flat_env

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

PATHS = 2 ** 14
TENORS = [0.5, 1.0, 1.5, 2.1]  # last node beyond DCN-A's 2.0027y grid end


def _curve_env():
    return build_flat_env(
        spot=6000.0, tenors=TENORS,
        valuation_date=datetime(2023, 1, 3), **FLAT,
    )


def test_canonical_flat_mapping():
    rate, carry, vol = build_flat_curves(tenors=TENORS, **FLAT)
    r, q, sig = FLAT["r"], FLAT["q"], FLAT["sigma"]
    for t in TENORS:
        assert rate.get_discount_factor(t) == pytest.approx(np.exp(-r * t))
        assert carry.carry(t) == pytest.approx((r - q) * t)
        assert vol.get_vol(6000.0, t, 6000.0) == pytest.approx(sig)


def test_flat_recovery_same_seed_near_exact():
    p = make_dcn(DCN_A)
    e = DCNMCEngine(num_paths=PATHS, seed=42)
    pv_flat = e.price(p, flat_env(**FLAT))
    pv_curves = e.price(p, _curve_env())
    assert abs(pv_flat - pv_curves) < 1e-8 * DCN_A["notional"]


def test_discount_bump_is_exact_cashflow_reweighting():
    # CRN + carry-invariant q re-derivation: paths are unchanged, so every
    # per-period leg PV must scale by EXACTLY its payment date's DF ratio
    # (spec: exact reweighting identity, not a tolerance guess).
    from copy import deepcopy

    from quantark.asset.equity.product.option.dcn_grid import (
        build_dcn_grid_context,
    )
    from quantark.param.div.dividend_yield import CarryInvariantDividendYield
    from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve
    from quantark.priceenv.term_sampling import make_df_fn

    p = make_dcn(DCN_A)
    env = _curve_env()
    e = DCNMCEngine(num_paths=PATHS, seed=42)
    base = e.price_detailed(p, env)

    bumped_curve = key_rate_bumped_zero_curve(env.rate_curve, 1.0, 1e-3)
    env_up = deepcopy(env)
    env_up.rate_curve = bumped_curve
    env_up.div_yield = CarryInvariantDividendYield(
        base=env.div_yield,
        base_rate_curve=env.rate_curve,
        bumped_rate_curve=bumped_curve,
    )
    up = e.price_detailed(p, env_up)

    ctx = build_dcn_grid_context(p)
    df0, df1 = make_df_fn(env), make_df_fn(env_up)
    checked = 0
    for j, (c0, c1) in enumerate(zip(base.pv_fixed_coupons_by_period,
                                     up.pv_fixed_coupons_by_period)):
        if c0 != 0.0:
            ratio = df1(ctx.coupon_pay_times[j]) / df0(ctx.coupon_pay_times[j])
            assert c1 / c0 == pytest.approx(ratio, rel=1e-10)
            checked += 1
    for j, (k0, k1) in enumerate(zip(base.pv_ko_coupons_by_period,
                                     up.pv_ko_coupons_by_period)):
        if k0 != 0.0:
            ratio = df1(ctx.ko_pay_times[j]) / df0(ctx.ko_pay_times[j])
            assert k1 / k0 == pytest.approx(ratio, rel=1e-10)
            checked += 1
    assert base.pv_loss_leg != 0.0
    ratio = df1(ctx.loss_pay_time) / df0(ctx.loss_pay_time)
    assert up.pv_loss_leg / base.pv_loss_leg == pytest.approx(ratio, rel=1e-10)
    assert checked > 10  # the identity was exercised across many periods


def test_discount_bump_beyond_last_payment_is_inert():
    # a pillar bump beyond every payment date must not move PV at all
    from copy import deepcopy

    from quantark.param.div.dividend_yield import CarryInvariantDividendYield
    from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve

    p = make_dcn(DCN_A)
    tenors = TENORS + [3.0]
    env = build_flat_env(spot=6000.0, tenors=tenors,
                         valuation_date=datetime(2023, 1, 3), **FLAT)
    e = DCNMCEngine(num_paths=PATHS, seed=42)
    base = e.price(p, env)
    bumped_curve = key_rate_bumped_zero_curve(env.rate_curve, 3.0, 1e-3)
    env_up = deepcopy(env)
    env_up.rate_curve = bumped_curve
    env_up.div_yield = CarryInvariantDividendYield(
        base=env.div_yield,
        base_rate_curve=env.rate_curve,
        bumped_rate_curve=bumped_curve,
    )
    # the 3.0y pillar triangle reaches down to the 2.1y node; DCN-A's last
    # payment is at ~2.0137y < 2.1y where the bump weight is exactly zero
    assert e.price(p, env_up) == base


def test_carry_bump_touches_only_overlapping_steps():
    # unit-level isolation evidence for B(T): bumping one CARRY NODE changes
    # only the per-step forward carries whose intervals overlap the node's
    # segments (spec WP3.4: unit-level assertion IS the isolation evidence).
    # The bump is applied on the carry curve itself and re-derived through
    # the pointwise q(T) = (r(T)T - B(T))/T adapter.
    from copy import deepcopy

    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
    from quantark.asset.equity.product.option.dcn_grid import (
        build_dcn_grid_context,
    )
    from quantark.param.div.forward_carry_curve import ForwardCarryCurve

    p = make_dcn(DCN_A)
    env = _curve_env()
    ctx = build_dcn_grid_context(p)
    dt = np.diff(ctx.times)
    base = build_mc_term_inputs(env, 6000.0, float(ctx.times[-1]), dt.size, dt)

    nodes = env.carry_curve.nodes
    nodes[1] = (nodes[1][0], nodes[1][1] + 0.01)  # bump B at the 1.0y node
    bumped_carry = ForwardCarryCurve(nodes)
    env_b = deepcopy(env)
    env_b.div_yield = bumped_carry.to_dividend_yield(env.rate_curve)
    bumped = build_mc_term_inputs(env_b, 6000.0, float(ctx.times[-1]), dt.size, dt)

    changed = ~np.isclose(base.div, bumped.div, rtol=0.0, atol=1e-14)
    mid = (ctx.times[:-1] + ctx.times[1:]) / 2.0
    assert changed.any()
    # the 1.0y node's influence spans (0.5, 1.5) via linear-in-B interpolation
    assert not changed[mid < 0.5].any()
    assert not changed[mid > 1.5].any()


def test_vol_bump_touches_only_overlapping_steps():
    from copy import deepcopy

    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
    from quantark.asset.equity.product.option.dcn_grid import (
        build_dcn_grid_context,
    )
    from quantark.asset.equity.riskmeasures.greeks_calculator import (
        GreeksCalculator,
    )

    p = make_dcn(DCN_A)
    env = _curve_env()
    ctx = build_dcn_grid_context(p)
    dt = np.diff(ctx.times)
    base = build_mc_term_inputs(env, 6000.0, float(ctx.times[-1]), dt.size, dt)

    env_b = deepcopy(env)
    env_b.vol_surface = GreeksCalculator._bump_term_vol_node(
        env.vol_surface, 1, 0.01
    )
    bumped = build_mc_term_inputs(env_b, 6000.0, float(ctx.times[-1]), dt.size, dt)

    changed = ~np.isclose(base.vol, bumped.vol, rtol=0.0, atol=1e-14)
    mid = (ctx.times[:-1] + ctx.times[1:]) / 2.0
    assert changed.any()
    # total-variance interpolation localizes the 1.0y node to (0.5, 1.5)
    assert not changed[mid < 0.5].any()
    assert not changed[mid > 1.5].any()
