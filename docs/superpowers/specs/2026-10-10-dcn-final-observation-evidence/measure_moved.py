"""W10 / W10b moved values: one value per line ``section | key | value-or-RAISES``, run on the base tree and on the
branch, then diffed by ``summarize.py``. Works on both trees (public engine APIs only).

    env PYTHONPATH=<tree>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/measure_moved.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np

from dcn_fixtures import DCN_A, DCN_B, FLAT, SSE, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine, LocalVolDCNMCEngine, QEDCNMCEngine, coupled_heston_ladder_pair,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.riskmeasures.greek_conventions_report import build_cash_greeks_report
from quantark.execution import PricingSession
from quantark.execution.contracts import PricingOperation, PricingRequest
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.volmodels.heston import HestonParams

warnings.simplefilter("ignore")
HESTON = HestonParams(v0=FLAT["sigma"] ** 2, kappa=2.0, theta=FLAT["sigma"] ** 2, sigma=0.3, rho=-0.5)
SPOTS = (7000.0, 6300.0, 6000.0, 5999.0, 5500.0, 4800.0, 4650.0, 4500.0, 4000.0)
ROUTES = {
    "DCNMCEngine": lambda: DCNMCEngine(num_paths=4096),
    "LocalVolDCNMCEngine": lambda: LocalVolDCNMCEngine(num_paths=4096),
    "HestonDCNMCEngine": lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=4096),
    "QEDCNMCEngine": lambda: QEDCNMCEngine(model_params=HESTON, num_paths=4096),
    "CoupledCoarseHestonDCNMCEngine": lambda: coupled_heston_ladder_pair(
        HESTON, 1, "full_truncation_euler", num_paths=4096)[0],
    "DCNPDEEngine": lambda: DCNPDEEngine(),
    "LocalVolDCNPDEEngine": lambda: LocalVolDCNPDEEngine(),
    "HestonDCNPDESolver": lambda: HestonDCNPDESolver(model_params=HESTON),
}


def env(spot, valuation_date):
    strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes=strikes, maturities=maturities,
                                   iv_grid=np.full((len(maturities), len(strikes)), FLAT["sigma"])),
        rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(FLAT["q"]),
        valuation_date=valuation_date, day_count_convention=DayCountConvention.ACT_365)


def prev_trading_day(d):
    d -= timedelta(days=1)
    while not SSE.is_business_day(d):
        d -= timedelta(days=1)
    return d


def show(section, key, fn):
    try:
        value = fn()
        if isinstance(value, dict):
            text = " ".join(f"{k}={float(v)!r}" for k, v in sorted(value.items()))
        elif isinstance(value, tuple):
            text = repr(value)
        else:
            text = repr(float(value))
    except Exception as exc:  # noqa: BLE001 - evidence
        text = f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    print(f"{section} | {key} | {text}", flush=True)


def mc_fields(res):
    return (res.pv, res.std_error, res.ki_probability, res.ko_probability, res.coupon_probability,
            res.expected_life_years)


def main():
    for name, c in (("DCN_A", DCN_A), ("DCN_B", DCN_B)):
        final, vday = c["maturity_date"], prev_trading_day(c["maturity_date"])
        for carried in (False, True):
            pf = make_dcn(c, valuation_date=final, knocked_in_at_valuation=carried)
            pb = make_dcn(c, valuation_date=vday, knocked_in_at_valuation=carried)
            for spot in SPOTS:
                tag = f"{name} carried={carried} spot={spot:g}"
                for route, make in ROUTES.items():
                    show("final", f"{tag} {route}.price", lambda: make().price(pf, env(spot, final)))
                show("final", f"{tag} DCNMCEngine.price_detailed",
                     lambda: mc_fields(DCNMCEngine(num_paths=4096).price_detailed(pf, env(spot, final))))
                for route in ("DCNPDEEngine", "LocalVolDCNPDEEngine"):
                    show("final", f"{tag} {route}.greeks", lambda: ROUTES[route]().calculate_greeks(pf, env(spot, final)))
                for route in ("DCNMCEngine", "DCNPDEEngine"):
                    def session_price():
                        with PricingSession() as s:
                            return s.execute(ROUTES[route](), PricingRequest(product=pf, pricing_env=env(spot, final))).value
                    show("final", f"{tag} session[{route}].price", session_price)
                # controls: the day before must not move
                for route in ("DCNMCEngine", "DCNPDEEngine"):
                    show("control", f"{tag} day-before {route}.price", lambda: ROUTES[route]().price(pb, env(spot, vday)))
            for spot in (6300.0, 5500.0, 4000.0):
                for route in ("DCNMCEngine", "DCNPDEEngine"):
                    show("theta", f"{name} carried={carried} spot={spot:g} penultimate {route}.theta_1d",
                         lambda: build_cash_greeks_report(pb, env(spot, vday), ROUTES[route](), calendar=SSE).theta_1d)

    # W10b: valuation dates that are (non-final) monthly observations, and controls one trading day before them
    for vobs in (datetime(2024, 12, 3), datetime(2024, 6, 3), datetime(2023, 4, 3)):
        vprev = prev_trading_day(vobs)
        for carried in (False, True):
            p = make_dcn(DCN_A, valuation_date=vobs, knocked_in_at_valuation=carried)
            q = make_dcn(DCN_A, valuation_date=vprev, knocked_in_at_valuation=carried)
            for spot in (6300.0, 6000.0, 5500.0, 4800.0, 4650.0, 4000.0):
                tag = f"DCN_A {vobs:%Y-%m-%d} carried={carried} spot={spot:g}"
                for route in ("DCNPDEEngine", "LocalVolDCNPDEEngine", "HestonDCNPDESolver", "DCNMCEngine"):
                    show("obsdate", f"{tag} {route}.price", lambda: ROUTES[route]().price(p, env(spot, vobs)))
                for route in ("DCNPDEEngine", "LocalVolDCNPDEEngine"):
                    show("obsdate", f"{tag} {route}.greeks", lambda: ROUTES[route]().calculate_greeks(p, env(spot, vobs)))
                show("control", f"DCN_A {vprev:%Y-%m-%d} carried={carried} spot={spot:g} day-before-obs DCNPDEEngine.price",
                     lambda: DCNPDEEngine().price(q, env(spot, vprev)))


if __name__ == "__main__":
    main()
