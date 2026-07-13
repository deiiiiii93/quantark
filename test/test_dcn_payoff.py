"""Deterministic hand-built-path tests for the DCN payoff kernel
(spec cross-cutting acceptance criterion #2)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_payoff import (
    build_dcn_grid_context,
    compute_dcn_cashflows,
)

from dcn_fixtures import DCN_A, FLAT, make_dcn

R = FLAT["r"]
A = 30.0 / 360.0  # accrual per period
N = 1_000_000.0
PART = 1.0


def _df(t):
    return float(np.exp(-R * t))


def _flat_path(ctx, level):
    return np.full((1, len(ctx.times)), float(level))


def _run(product, path):
    ctx = build_dcn_grid_context(product)
    return ctx, compute_dcn_cashflows(np.atleast_2d(path), product, ctx, _df)


def test_ko_first_obs_pays_only_ko_coupon():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)          # above coupon(4800), below KO
    path[0, ctx.obs_cols[0]] = 6100.0        # KO at first obs (>= 6000)
    ctx, cf = _run(p, path[0])
    assert cf.ko_obs_row[0] == 0
    expected = PART * 0.12 * A * N * _df(ctx.ko_pay_times[0])
    assert cf.ko_pv[0] == pytest.approx(expected)
    assert cf.fixed_coupon_pv[0] == 0.0      # KO wins: no fixed coupon that day
    assert cf.loss_pv[0] == 0.0


def test_ki_kills_future_coupons_even_above_coupon_barrier():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)           # coupon barrier satisfied everywhere
    path[0, 5] = 4400.0                       # daily KI before first obs (<= 4500)
    ctx, cf = _run(p, path[0])
    assert bool(cf.knocked_in[0])
    assert cf.fixed_coupon_pv[0] == 0.0
    # S_T = 5000 < K_loss = 6600 -> loss leg at settlement_date discount
    expected_loss = (
        -(N / 6000.0) * PART * max(6600.0 - 5000.0, 0.0) * _df(ctx.loss_pay_time)
    )
    assert cf.loss_pv[0] == pytest.approx(expected_loss)


def test_ki_then_ko_still_terminates_with_ko_coupon():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)
    path[0, 5] = 4400.0                        # KI early
    path[0, ctx.obs_cols[3]] = 6200.0          # KO at 4th obs
    ctx, cf = _run(p, path[0])
    assert cf.ko_obs_row[0] == 3
    assert cf.loss_pv[0] == 0.0                # KO extinguishes the loss leg
    assert cf.fixed_coupon_pv[0] == 0.0        # KI'd before: coupons dead


def test_never_ki_never_ko_pays_all_coupons_no_principal_leg():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)             # 4800 <= 5000 < 6000 always
    ctx, cf = _run(p, path[0])
    n_obs = len(ctx.obs_cols)
    expected = sum(
        PART * 0.12 * A * N * _df(ctx.coupon_pay_times[j]) for j in range(n_obs)
    )
    assert cf.fixed_coupon_pv[0] == pytest.approx(expected)
    assert cf.loss_pv[0] == 0.0 and cf.ko_pv[0] == 0.0
    assert cf.coupon_paid[0].all()


def test_coupon_condition_checked_per_obs():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)
    path[0, ctx.obs_cols[1]] = 4700.0          # below coupon barrier at obs 1 only
    ctx, cf = _run(p, path[0])
    assert not cf.coupon_paid[0, 1] and cf.coupon_paid[0, 0]


def test_knocked_in_at_valuation_pays_no_coupons_ever():
    p = make_dcn(DCN_A, knocked_in_at_valuation=True)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)             # never breaches KI in-simulation
    ctx, cf = _run(p, path[0])
    assert cf.fixed_coupon_pv[0] == 0.0
    assert cf.loss_pv[0] < 0.0


def test_valuation_date_breach_not_monitored():
    # column 0 (valuation date) at/below KI must NOT knock in (seed authoritative)
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)
    path[0, 0] = 4000.0
    ctx, cf = _run(p, path[0])
    assert not bool(cf.knocked_in[0])


def test_ki_at_maturity_settles_at_settlement_date_df():
    p = make_dcn(DCN_A)
    ctx = build_dcn_grid_context(p)
    path = _flat_path(ctx, 5000.0)
    path[0, -1] = 4400.0                       # KI on the final trading day
    ctx, cf = _run(p, path[0])
    expected_loss = (
        -(N / 6000.0) * max(6600.0 - 4400.0, 0.0) * _df(ctx.loss_pay_time)
    )
    assert cf.loss_pv[0] == pytest.approx(expected_loss)
