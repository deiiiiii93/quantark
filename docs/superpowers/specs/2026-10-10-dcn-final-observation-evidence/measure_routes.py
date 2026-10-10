"""W10 evidence: every DCN route on its final observation date (today: raises) and the day before; the vol-model PDE
routes on a non-final monthly observation date; the date-rolled theta of the cash-Greeks report on the day before.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/measure_routes.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine, LocalVolDCNMCEngine, QEDCNMCEngine,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.riskmeasures.greek_conventions_report import build_cash_greeks_report
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.volmodels.heston import HestonParams

warnings.simplefilter("ignore")
PATHS = 1 << 14
HESTON = HestonParams(v0=FLAT["sigma"] ** 2, kappa=2.0, theta=FLAT["sigma"] ** 2, sigma=0.3, rho=-0.5)


def env(spot, valuation_date):
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes=strikes, maturities=maturities,
                                   iv_grid=np.full((len(maturities), len(strikes)), FLAT["sigma"])),
        rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(FLAT["q"]),
        valuation_date=valuation_date, day_count_convention=DayCountConvention.ACT_365)


ROUTES = {
    "DCNMCEngine": lambda: DCNMCEngine(num_paths=PATHS),
    "DCNPDEEngine": lambda: DCNPDEEngine(),
    "LocalVolDCNMCEngine": lambda: LocalVolDCNMCEngine(num_paths=PATHS),
    "LocalVolDCNPDEEngine": lambda: LocalVolDCNPDEEngine(),
    "HestonDCNMCEngine": lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=PATHS),
    "QEDCNMCEngine": lambda: QEDCNMCEngine(model_params=HESTON, num_paths=PATHS),
    "HestonDCNPDESolver": lambda: HestonDCNPDESolver(model_params=HESTON),
}


def attempt(fn):
    try:
        return f"{float(fn()):.6f}"
    except Exception as exc:  # noqa: BLE001 - evidence
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not SSE.is_business_day(d):
        d -= timedelta(days=1)
    return d


def main():
    final = DCN_A["maturity_date"]
    vday = prev_trading_day(final)
    print(f"=== DCN_A routes: final {final:%Y-%m-%d} | day before {vday:%Y-%m-%d}")
    for name, make in ROUTES.items():
        for carried in (False, True):
            for spot in (6300.0, 5500.0, 4000.0):
                pf = make_dcn(DCN_A, valuation_date=final, knocked_in_at_valuation=carried)
                pb = make_dcn(DCN_A, valuation_date=vday, knocked_in_at_valuation=carried)
                print(f"{name:22} carried={carried!s:5} spot={spot:6g} | final: {attempt(lambda: make().price(pf, env(spot, final)))}"
                      f" | day before: {attempt(lambda: make().price(pb, env(spot, vday)))}")

    vobs = datetime(2024, 12, 3)
    print(f"\n=== vol-model PDE routes on a non-final monthly observation ({vobs:%Y-%m-%d}) vs their MC twin")
    for pde_name, mc_name in (("LocalVolDCNPDEEngine", "LocalVolDCNMCEngine"), ("HestonDCNPDESolver", "HestonDCNMCEngine")):
        for spot in (6300.0, 5500.0):
            p = make_dcn(DCN_A, valuation_date=vobs)
            print(f"{pde_name:22} spot={spot:6g} | PDE {attempt(lambda: ROUTES[pde_name]().price(p, env(spot, vobs)))}"
                  f" | {mc_name} {attempt(lambda: ROUTES[mc_name]().price(p, env(spot, vobs)))}")

    print(f"\n=== cash-Greeks report date-rolled theta (rolls the schedule one SSE day; day before -> final)")
    for carried in (False, True):
        p = make_dcn(DCN_A, valuation_date=vday, knocked_in_at_valuation=carried)
        for name in ("DCNMCEngine", "DCNPDEEngine"):
            def theta():
                return build_cash_greeks_report(p, env(5500.0, vday), ROUTES[name](), calendar=SSE).theta_1d
            print(f"{name:22} carried={carried!s:5} spot=5500 | theta_1d {attempt(theta)}")
        p2 = make_dcn(DCN_A, valuation_date=prev_trading_day(vday), knocked_in_at_valuation=carried)
        print(f"{'DCNMCEngine (2 days before)':22} carried={carried!s:5} spot=5500 | theta_1d "
              f"{attempt(lambda: build_cash_greeks_report(p2, env(5500.0, prev_trading_day(vday)), ROUTES['DCNMCEngine'](), calendar=SSE).theta_1d)}")


if __name__ == "__main__":
    main()
