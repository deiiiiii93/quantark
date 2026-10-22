"""A DCN monthly observation that falls on the valuation date (W10b, user ruling Q5 2026-10-10).

The MC path kernel pays the valuation-date row (column 0); the three PDE routes (GBM, local-vol, Heston) now apply it
pointwise at t = 0 through the W10 helper's row logic (``dcn_valuation.observe_valuation_row``): a knock-out today
settles the contract at the knock-out coupon (greeks 0), a coupon today is added to the smooth continuation, a knock-in
today reads the knocked-in surface. Before, the PDE routes dropped the row (measured on 0d1e4133, DCN_A 2024-12-03:
coupon today 5500 MC 19801.74 vs PDE 9779.74; carried KO today 6300 MC 9998.05 vs PDE -23212.00).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.volmodels.heston import HestonParams

R, Q, SIGMA = FLAT["r"], FLAT["q"], FLAT["sigma"]
HESTON = HestonParams(v0=SIGMA ** 2, kappa=2.0, theta=SIGMA ** 2, sigma=0.3, rho=-0.5)
OBS = datetime(2024, 12, 3)          # DCN_A month 23; coupon / KO paid 2024-12-05
CASH_TODAY = 10000.0 * math.exp(-R * 2 / 365.0)   # 9998.049505...
N = DCN_A["notional"]


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


PDE_ROUTES = {
    "DCNPDEEngine": lambda: DCNPDEEngine(),
    "LocalVolDCNPDEEngine": lambda: LocalVolDCNPDEEngine(),
    "HestonDCNPDESolver": lambda: HestonDCNPDESolver(model_params=HESTON),
}
MC_TWIN = {
    "DCNPDEEngine": lambda: DCNMCEngine(num_paths=1 << 15),
    "LocalVolDCNPDEEngine": lambda: LocalVolDCNMCEngine(num_paths=1 << 15),
    "HestonDCNPDESolver": lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=1 << 15),
}


def test_the_row_today_by_hand():
    from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row

    coupon = observe_valuation_row(make_dcn(DCN_A, valuation_date=OBS), env(5500.0, OBS))
    assert (coupon.is_final, coupon.knocked_in, coupon.knocked_out, coupon.coupon_paid) == (False, False, False, True)
    assert coupon.coupon_pv == pytest.approx(CASH_TODAY, rel=1e-14) and coupon.loss_pv == 0.0
    ko = observe_valuation_row(make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=True), env(6300.0, OBS))
    assert (ko.knocked_in, ko.knocked_out, ko.coupon_paid, ko.ends_contract) == (True, True, False, True)
    assert ko.ko_pv == pytest.approx(CASH_TODAY, rel=1e-14) and ko.pv == ko.ko_pv
    assert observe_valuation_row(make_dcn(DCN_A, valuation_date=prev_trading_day(OBS)), env(5500.0, OBS)) is None


@pytest.mark.parametrize("route", sorted(PDE_ROUTES))
@pytest.mark.parametrize("carried, spot", [(False, 6300.0), (True, 6300.0), (False, 6000.0), (True, 6000.0)])
def test_knock_out_today_settles_at_the_knock_out_coupon(route, carried, spot):
    product = make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=carried)
    assert PDE_ROUTES[route]().price(product, env(spot, OBS)) == pytest.approx(CASH_TODAY, rel=1e-12)
    assert DCNMCEngine(num_paths=256).price(product, env(spot, OBS)) == pytest.approx(CASH_TODAY, rel=1e-12)


@pytest.mark.parametrize("route", ["DCNPDEEngine", "LocalVolDCNPDEEngine"])
def test_knock_out_today_greeks(route):
    product = make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=True)
    g = PDE_ROUTES[route]().calculate_greeks(product, env(6300.0, OBS))
    assert g["price"] == pytest.approx(CASH_TODAY, rel=1e-12)
    assert (g["delta"], g["gamma"]) == (0.0, 0.0)


@pytest.mark.parametrize("route", sorted(PDE_ROUTES))
@pytest.mark.parametrize("carried, spot", [(False, 5500.0), (False, 4800.0), (False, 4650.0), (True, 5500.0),
                                           (False, 4000.0)])
def test_pde_equals_mc_on_an_observation_date(route, carried, spot):
    product = make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=carried)
    e = env(spot, OBS)
    mc = MC_TWIN[route]().price_detailed(product, e)
    pde = PDE_ROUTES[route]().price(product, e)
    assert abs(pde - mc.pv) < max(4.0 * mc.std_error, 5e-4 * N), (pde, mc.pv)


@pytest.mark.parametrize("route", sorted(PDE_ROUTES))
def test_coupon_today_is_no_overnight_jump(route):
    """The day before, the PDE already prices the 2024-12-03 coupon; on the day it must not vanish."""
    before = make_dcn(DCN_A, valuation_date=prev_trading_day(OBS))
    today = make_dcn(DCN_A, valuation_date=OBS)
    v_before = PDE_ROUTES[route]().price(before, env(5500.0, prev_trading_day(OBS)))
    v_today = PDE_ROUTES[route]().price(today, env(5500.0, OBS))
    assert abs(v_today - v_before) < 5e-4 * N, (v_before, v_today)


@pytest.mark.parametrize("route", ["DCNPDEEngine", "LocalVolDCNPDEEngine"])
def test_greeks_and_spot_curve_carry_the_row_today(route):
    product = make_dcn(DCN_A, valuation_date=OBS)
    e = env(5500.0, OBS)
    engine = PDE_ROUTES[route]()
    price = engine.price(product, e)
    g = engine.calculate_greeks(product, e)
    assert g["price"] == price
    rows = {row["spot"]: row for row in engine.calculate_spot_greeks_curve(product, e, [6300.0, 5500.0, 4650.0])}
    assert rows[6300.0]["price"] == pytest.approx(CASH_TODAY, rel=1e-12)
    assert (rows[6300.0]["delta"], rows[6300.0]["gamma"]) == (0.0, 0.0)
    # the curve projects the grid (_project_rows), the price reads it (readout): measured 6.6e-8 relative here
    assert rows[5500.0]["price"] == pytest.approx(price, rel=1e-6)
    assert rows[4650.0]["price"] == pytest.approx(engine.price(product, env(4650.0, OBS)), rel=1e-3)
