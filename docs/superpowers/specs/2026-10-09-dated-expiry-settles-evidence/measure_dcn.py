"""DCN on its final observation date (the DCN has no numeric twin: its schedule is always dated and carries its own
valuation date, so it never reached ``BaseEquityOption.get_maturity``). Records what each DCN engine does there.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/measure_dcn.py
"""
from __future__ import annotations

import warnings
from datetime import datetime

from dcn_fixtures import DCN_A, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

warnings.simplefilter("ignore")
MATURITY = DCN_A["maturity_date"]


def env(spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.184),
                              rate_curve=FlatRateCurve(rate=0.0356), div_yield=ContinuousDividendYield(0.1406),
                              valuation_date=MATURITY)


def attempt(fn):
    try:
        return f"{float(fn()):.10g}"
    except Exception as exc:
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"


for knocked_in in (False, True):
    for spot in (6600.0, 5000.0):
        print(f"knocked_in={knocked_in} spot={spot:g}",
              "| build:", attempt(lambda: make_dcn(DCN_A, valuation_date=MATURITY, knocked_in_at_valuation=knocked_in).get_maturity()),
              "| mc:", attempt(lambda: DCNMCEngine(num_paths=256).price(
                  make_dcn(DCN_A, valuation_date=MATURITY, knocked_in_at_valuation=knocked_in), env(spot))),
              "| pde:", attempt(lambda: DCNPDEEngine().price(
                  make_dcn(DCN_A, valuation_date=MATURITY, knocked_in_at_valuation=knocked_in), env(spot))))
