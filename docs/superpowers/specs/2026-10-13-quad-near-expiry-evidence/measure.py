"""W16 / signed W13 follow-up 2: EuropeanQuadEngine near-expiry error against its 1e-4 absolute lower-bound check.

Run (from this directory):
    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python measure.py > outputs/measure.txt

For each cell (method, spot scale, contract multiplier, moneyness, vol, maturity, call/put) price with the QUAD engine
and with the Black-Scholes closed form, and report the error, the bound margin (price - (bound - 1e-4)), refusals, and
FALSE refusals (QUAD refuses while the true BS price respects the bound). Serial, single process.
"""
from __future__ import annotations

import itertools
from datetime import datetime
import math

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import QuadratureMethod
from quantark.util.exceptions import NumericalError

HOUR = 1.0 / (365.0 * 24.0)
MATURITIES = [("1d", 1.0 / 365.0), ("1h", HOUR), ("1m", HOUR / 60.0), ("1s", HOUR / 3600.0), ("1e-9y", 1e-9)]
MONEYNESS = [0.8, 0.95, 0.999, 1.0, 1.001, 1.05, 1.2]
VOLS = [0.1, 0.3, 1.0]
SPOTS = [100.0, 10_000.0]
MULTIPLIERS = [1.0, 300.0]


def env(spot, sigma):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=sigma),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=datetime(2026, 1, 5))


def main():
    rows, refusals, false_refusals, worst = 0, 0, 0, {}
    bs = BlackScholesEngine()
    print("method spot mult K/S vol T type quad bs err margin status")
    for method, spot, mult, m, sigma, (tlab, T), ot in itertools.product(
            [QuadratureMethod.SIMPSON, QuadratureMethod.GAUSS_LEGENDRE], SPOTS, MULTIPLIERS, MONEYNESS, VOLS,
            MATURITIES, [OptionType.CALL, OptionType.PUT]):
        e = env(spot, sigma)
        product = EuropeanVanillaOption(strike=spot * m, option_type=ot, maturity=T, contract_multiplier=mult)
        ref = bs.price(product, e)
        engine = EuropeanQuadEngine(method=method)
        bound = engine._european_lower_bound(product, spot, spot * m, T, 0.01, e.get_discount_factor(T), 1.0)
        rows += 1
        try:
            quad = engine.price(product, e)
            status = "ok"
        except NumericalError:
            quad, status = float("nan"), "REFUSED"
            refusals += 1
            if ref >= bound - 1e-12 * max(1.0, bound):
                false_refusals += 1
                status = "FALSE_REFUSAL"
        err = quad - ref
        rel = abs(err) / max(abs(ref), 1e-300)
        key = (method.name, tlab)
        if not math.isnan(err) and abs(err) > worst.get(key, (0.0,))[0]:
            worst[key] = (abs(err), rel, spot, mult, m, sigma, ot.name)
        margin = quad - (bound - 1e-4)
        if status != "ok" or abs(err) > 1e-6:
            print(f"{method.name} {spot:g} {mult:g} {m:g} {sigma:g} {tlab} {ot.name} {quad:.10g} {ref:.10g} "
                  f"{err:.3e} {margin:.3e} {status}")
    print(f"\ncells {rows}, refusals {refusals}, false refusals {false_refusals}")
    print("worst |err| per (method, maturity): |err|, rel, spot, mult, K/S, vol, type")
    for key in sorted(worst):
        print(key, tuple(f"{v:.3e}" if isinstance(v, float) else v for v in worst[key]))


if __name__ == "__main__":
    main()
