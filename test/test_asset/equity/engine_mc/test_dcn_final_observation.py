"""The DCN on its final observation date (W10, user rulings 2026-10-10 on the proposal
docs/superpowers/specs/2026-10-10-dcn-final-observation-proposal.md).

At T = 0 every route settles the final row from ``dcn_valuation.settle_final_observation``: knock-in = carried OR
spot <= KI; knock-out (spot >= KO, either state) pays the KO coupon and ends the contract; the fixed coupon pays when
not knocked out, not knocked in and spot >= the coupon barrier; the loss leg pays when knocked in and not knocked out.
Each flow is discounted from the final date to its own payment date (DF = 1 when paid on the final date). Pointwise at
barrier ties on every route (Q2); greeks and result fields per Q6; sessions answer natively (Q8); the penultimate-day
date-rolled theta = settled - V(day before) (Q9).
"""
from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from dcn_fixtures import DCN_A, DCN_B, FLAT, SSE, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    QEDCNMCEngine,
    coupled_heston_ladder_pair,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

R, Q, SIGMA = FLAT["r"], FLAT["q"], FLAT["sigma"]
HESTON = HestonParams(v0=SIGMA ** 2, kappa=2.0, theta=SIGMA ** 2, sigma=0.3, rho=-0.5)
CONTRACTS = {"DCN_A": DCN_A, "DCN_B": DCN_B}
# both contracts: KI 4500, coupon 4800, KO 6000 (K_loss 6600 / 6900)
SPOTS = (7000.0, 6300.0, 6000.0, 5999.0, 5500.0, 4800.0, 4650.0, 4500.0, 4000.0)
DIRECTIONS = (DCNDirection.BUYER, DCNDirection.SELLER)


def env(spot, valuation_date):
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes=strikes, maturities=maturities,
                                   iv_grid=np.full((len(maturities), len(strikes)), SIGMA)),
        rate_curve=FlatRateCurve(rate=R), div_yield=ContinuousDividendYield(Q),
        valuation_date=valuation_date, day_count_convention=DayCountConvention.ACT_365)


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not SSE.is_business_day(d):
        d -= timedelta(days=1)
    return d


def final_product(c, carried, direction=DCNDirection.BUYER):
    return make_dcn(c, valuation_date=c["maturity_date"], knocked_in_at_valuation=carried, direction=direction)


def hand(c, spot, carried, direction=DCNDirection.BUYER):
    """The proposal's section 2, by hand: (pv, delta)."""
    p = final_product(c, carried, direction)
    row = p.schedule.monthly[-1]
    v = c["maturity_date"]

    def df(d):
        return math.exp(-R * (d - v).days / 365.0)

    sign = 1.0 if direction is DCNDirection.BUYER else -1.0
    a = c["coupon_counted_days"] / c["coupon_days_denom"]
    n, s0, part = c["notional"], c["initial_price"], c["participation"]
    k_loss = c["ki_put_strike_ratio"] * s0
    ki = carried or spot <= c["ki_barrier_ratio"] * s0
    ko = spot >= c["ko_barrier_ratio"] * s0
    cpn = not ko and not ki and spot >= c["coupon_barrier_ratio"] * s0
    pv = 0.0
    if ko:
        pv += part * c["ko_coupon_rate"] * a * n * df(row.ko_payment_date)
    if cpn:
        pv += part * c["coupon_rate"] * a * n * df(row.coupon_payment_date)
    delta = 0.0
    if ki and not ko:
        pv += -(n / s0) * part * max(k_loss - spot, 0.0) * df(c["settlement_date"])
        if spot < k_loss:
            delta = (n / s0) * part * df(c["settlement_date"])
    ties = [c["ko_barrier_ratio"] * s0]
    if not ki:
        ties.append(c["coupon_barrier_ratio"] * s0)
    if not carried:
        ties.append(c["ki_barrier_ratio"] * s0)
    if ki and not ko:
        ties.append(k_loss)
    if any(spot == level for level in ties):
        delta = 0.0
    return sign * pv, sign * delta


ROUTES = {
    "DCNMCEngine": lambda: DCNMCEngine(num_paths=256),
    "LocalVolDCNMCEngine": lambda: LocalVolDCNMCEngine(num_paths=256),
    "HestonDCNMCEngine": lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=256),
    "QEDCNMCEngine": lambda: QEDCNMCEngine(model_params=HESTON, num_paths=256),
    "CoupledCoarseHestonDCNMCEngine": lambda: coupled_heston_ladder_pair(HESTON, 1, "full_truncation_euler", num_paths=256)[0],
    "DCNPDEEngine": lambda: DCNPDEEngine(),
    "LocalVolDCNPDEEngine": lambda: LocalVolDCNPDEEngine(),
    "HestonDCNPDESolver": lambda: HestonDCNPDESolver(model_params=HESTON),
}


# --------------------------------------------------------------------------- hand values, every route

@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("direction", DIRECTIONS, ids=lambda d: d.name)
@pytest.mark.parametrize("carried", (False, True), ids=("not_carried", "carried"))
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_every_route_settles_the_final_row_at_the_hand_value(name, carried, direction, route):
    c = CONTRACTS[name]
    engine = ROUTES[route]()
    for spot in SPOTS:
        expected, _ = hand(c, spot, carried, direction)
        got = engine.price(final_product(c, carried, direction), env(spot, c["maturity_date"]))
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-9), (spot, got, expected)


@pytest.mark.parametrize("name, carried, spot, expected", [
    ("DCN_A", False, 5500.0, 9996.099391),     # coupon, paid 2025-01-07 (4 calendar days)
    ("DCN_A", True, 6300.0, 9996.099391),      # knock-out beats a carried knock-in
    ("DCN_A", True, 4000.0, -433164.306946),
    ("DCN_A", True, 5500.0, -183261.822170),
    ("DCN_A", False, 4650.0, 0.0),
    ("DCN_B", False, 5500.0, 10000.0),         # coupon paid on the final date itself: DF = 1
    ("DCN_B", False, 4000.0, -483239.059424),  # knocked in today
    ("DCN_B", True, 5500.0, -233287.821791),
])
def test_anchor_values(name, carried, spot, expected):
    c = CONTRACTS[name]
    product, e = final_product(c, carried), env(spot, c["maturity_date"])
    assert DCNMCEngine(num_paths=256).price(product, e) == pytest.approx(expected, abs=1e-6)
    assert DCNPDEEngine().price(product, e) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("carried", (False, True))
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_every_route_gives_the_same_number_bitwise(name, carried):
    c = CONTRACTS[name]
    for spot in SPOTS:
        product, e = final_product(c, carried), env(spot, c["maturity_date"])
        values = {route: make().price(product, e) for route, make in ROUTES.items()}
        assert len(set(values.values())) == 1, (spot, values)


@pytest.mark.parametrize("carried", (False, True))
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_helper_is_the_mc_kernel_on_a_one_column_path_bitwise(name, carried):
    from quantark.asset.equity.engine.dcn_valuation import settle_final_observation
    from quantark.asset.equity.engine.mc.dcn_payoff import compute_dcn_cashflows
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
    from quantark.priceenv.term_sampling import make_df_fn

    c = CONTRACTS[name]
    for spot in SPOTS:
        product, e = final_product(c, carried), env(spot, c["maturity_date"])
        cf = compute_dcn_cashflows(np.array([[spot]]), product, build_dcn_grid_context(product), make_df_fn(e))
        row = settle_final_observation(product, e)
        assert row.coupon_pv == float(cf.fixed_coupon_pv[0])
        assert row.ko_pv == float(cf.ko_pv[0])
        assert row.loss_pv == float(cf.loss_pv[0])
        assert row.knocked_in == bool(cf.knocked_in[0])
        assert row.knocked_out == bool(cf.ko_obs_row[0] == 0)
        assert row.coupon_paid == bool(cf.coupon_paid[0, 0])


# --------------------------------------------------------------------------- result fields and greeks (Q6)

@pytest.mark.parametrize("carried, spot, ki, ko, coupon", [
    (False, 6300.0, False, True, False),
    (False, 5500.0, False, False, True),
    (False, 4650.0, False, False, False),
    (False, 4000.0, True, False, False),
    (True, 6300.0, True, True, False),
    (True, 5500.0, True, False, False),
])
def test_mc_result_fields_are_the_decisions(carried, spot, ki, ko, coupon):
    product, e = final_product(DCN_A, carried), env(spot, DCN_A["maturity_date"])
    res = DCNMCEngine(num_paths=256).price_detailed(product, e)
    expected, _ = hand(DCN_A, spot, carried)
    assert res.pv == pytest.approx(expected, rel=1e-12, abs=1e-9)
    assert res.pv == res.pv_fixed_coupons + res.pv_ko_coupons + res.pv_loss_leg
    assert res.std_error == 0.0
    assert res.ki_probability == float(ki)
    assert res.ko_probability == float(ko)
    assert res.ko_timing_distribution == (float(ko),)
    assert res.coupon_probability == (float(coupon),)
    assert res.expected_life_years == 0.0
    assert res.prob_survive_ki == float(ki and not ko)
    assert res.prob_survive_no_ki == float(not ki and not ko)
    assert res.event_stats is not None and res.event_stats.pv == res.pv


def test_pde_price_detailed_on_the_final_date():
    for engine in (DCNPDEEngine(), LocalVolDCNPDEEngine(), HestonDCNPDESolver(model_params=HESTON)):
        res = engine.price_detailed(final_product(DCN_A, True), env(5500.0, DCN_A["maturity_date"]))
        assert res.pv == pytest.approx(-183261.822170, abs=1e-6)
        assert (res.num_space_nodes, res.num_time_steps) == (0, 0)


@pytest.mark.parametrize("direction", DIRECTIONS, ids=lambda d: d.name)
@pytest.mark.parametrize("carried", (False, True), ids=("not_carried", "carried"))
@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_pde_greeks_are_the_slope_of_the_settled_row(name, carried, direction):
    c = CONTRACTS[name]
    for engine in (DCNPDEEngine(), LocalVolDCNPDEEngine()):
        for spot in SPOTS:
            pv, delta = hand(c, spot, carried, direction)
            g = engine.calculate_greeks(final_product(c, carried, direction), env(spot, c["maturity_date"]))
            assert g["price"] == pytest.approx(pv, rel=1e-12, abs=1e-9)
            assert g["delta"] == pytest.approx(delta, rel=1e-12, abs=1e-12), spot
            assert g["gamma"] == 0.0


# --------------------------------------------------------------------------- sessions (Q8)

@pytest.mark.parametrize("route", ["DCNMCEngine", "LocalVolDCNMCEngine", "HestonDCNMCEngine", "QEDCNMCEngine",
                                   "CoupledCoarseHestonDCNMCEngine", "DCNPDEEngine", "LocalVolDCNPDEEngine"])
@pytest.mark.parametrize("carried", (False, True))
def test_session_answers_natively_and_equals_direct_bitwise(route, carried):
    from quantark.execution import PricingSession
    from quantark.execution.contracts import PricingOperation, PricingRequest

    product = final_product(DCN_A, carried)
    for spot in (6300.0, 5500.0, 4000.0):
        e = env(spot, DCN_A["maturity_date"])
        direct = ROUTES[route]().price(product, e)
        with PricingSession() as session:
            outcome = session.execute(ROUTES[route](), PricingRequest(product=product, pricing_env=e))
            assert outcome.value == direct
            assert outcome.manifest.plan_fingerprint is None
            if route.endswith("MCEngine"):
                detailed = session.execute(ROUTES[route](), PricingRequest(
                    product=product, pricing_env=e, operation=PricingOperation.PRICE_DETAILED)).value
                assert detailed.pv == direct and detailed.std_error == 0.0


# --------------------------------------------------------------------------- continuity and limit

@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_day_before_differs_by_the_last_day_only(name):
    """Spots at least 3 one-day standard deviations from every barrier: the coupon / knock-out cells move by one day
    of discounting exactly, the loss cells by one day of carry."""
    c = CONTRACTS[name]
    final, vday = c["maturity_date"], prev_trading_day(c["maturity_date"])
    days = (final - vday).days
    for carried, spot in ((False, 7000.0), (False, 5500.0), (True, 7000.0), (True, 5500.0), (True, 4000.0),
                          (False, 4000.0)):
        settled, _ = hand(c, spot, carried)
        before = make_dcn(c, valuation_date=vday, knocked_in_at_valuation=carried)
        for engine in (DCNMCEngine(num_paths=1 << 14), DCNPDEEngine()):
            v = engine.price(before, env(spot, vday))
            if settled > 0.0:
                assert v == pytest.approx(settled * math.exp(-R * days / 365.0), rel=1e-6)
            else:
                # measured: 149..246 (DCN_A, 1 calendar day), 722..1199 (DCN_B, 5 calendar days)
                assert abs(v - settled) < 5e-4 * c["notional"] * days


@pytest.mark.parametrize("carried, spot", [(False, 5500.0), (True, 4000.0), (True, 5500.0)])
def test_shrinking_the_last_step_converges_to_the_settled_value(monkeypatch, carried, spot):
    """The day-before contract with its last step shrunk to one minute (payment gaps after the final date held)."""
    from quantark.asset.equity.engine.mc import dcn_mc_engine
    from quantark.asset.equity.engine.pde import dcn_pde_solver
    from quantark.asset.equity.engine.pde.core.adapters import dcn as dcn_adapter
    from quantark.asset.equity.product.option import dcn_grid

    original = dcn_grid.build_dcn_grid_context
    h = 1.0 / 365.0 / 1440.0

    def shrunk(product):
        ctx = original(product)
        shift = float(ctx.times[-1]) - h
        return dataclasses.replace(ctx, times=np.array([0.0, h]), coupon_pay_times=ctx.coupon_pay_times - shift,
                                   ko_pay_times=ctx.ko_pay_times - shift, loss_pay_time=ctx.loss_pay_time - shift)

    for module in (dcn_mc_engine, dcn_pde_solver, dcn_adapter):
        monkeypatch.setattr(module, "build_dcn_grid_context", shrunk)
    vday = prev_trading_day(DCN_A["maturity_date"])
    product = make_dcn(DCN_A, valuation_date=vday, knocked_in_at_valuation=carried)
    settled, _ = hand(DCN_A, spot, carried)
    for engine in (DCNMCEngine(num_paths=1 << 14), DCNPDEEngine()):
        assert abs(engine.price(product, env(spot, vday)) - settled) < 0.2


# --------------------------------------------------------------------------- theta (Q9) and the kept refusal

@pytest.mark.parametrize("carried", (False, True))
def test_penultimate_day_theta_is_settled_minus_today(carried):
    from quantark.asset.equity.engine.dcn_valuation import settle_final_observation
    from quantark.asset.equity.riskmeasures.greek_conventions_report import build_cash_greeks_report

    vday = prev_trading_day(DCN_A["maturity_date"])
    product = make_dcn(DCN_A, valuation_date=vday, knocked_in_at_valuation=carried)
    e = env(5500.0, vday)
    settled = settle_final_observation(final_product(DCN_A, carried), e).pv
    for engine in (DCNMCEngine(num_paths=1 << 12), DCNPDEEngine()):
        report = build_cash_greeks_report(product, e, engine, calendar=SSE)
        assert report.theta_1d == settled - engine.price(product, e)


def test_valuation_after_the_final_observation_still_raises():
    with pytest.raises(ValidationError, match="valuation_date after maturity_date"):
        make_dcn(DCN_A, valuation_date=datetime(2025, 1, 6))
