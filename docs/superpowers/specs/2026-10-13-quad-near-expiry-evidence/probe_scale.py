"""Follow-ups to measure.py: (1) is the Gauss-Legendre ~0.4% ATM error near-expiry specific (compare 1y)?
(2) does the ABSOLUTE 1e-4 lower-bound tolerance refuse a correct price once the notional scale grows (deep ITM,
rounding noise ~1e-16 of the price)? Serial.

Run (from this directory): env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python probe_scale.py > outputs/probe_scale.txt
"""
from __future__ import annotations

from measure import env

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import QuadratureMethod
from quantark.util.exceptions import NumericalError

bs = BlackScholesEngine()
print("(1) ATM relative error by maturity, spot 100, vol 0.3")
for method in (QuadratureMethod.SIMPSON, QuadratureMethod.GAUSS_LEGENDRE):
    for label, T in (("1e-6y", 1e-6), ("1d", 1 / 365), ("1m", 1 / 12), ("1y", 1.0), ("5y", 5.0)):
        p = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=T)
        e = env(100.0, 0.3)
        q, r = EuropeanQuadEngine(method=method).price(p, e), bs.price(p, e)
        print(f"  {method.name:15s} {label:6s} quad {q:.10g} bs {r:.10g} rel {abs(q - r) / r:.3e}")

print("\n(2) deep ITM, tiny maturity, growing notional scale (absolute tolerance 1e-4)")
for method in (QuadratureMethod.SIMPSON, QuadratureMethod.GAUSS_LEGENDRE):
    for spot in (1e4, 1e5, 1e6):
        for mult in (300.0, 1e4, 1e6):
            for ot, m in ((OptionType.PUT, 1.2), (OptionType.CALL, 0.8)):
                for label, T in (("1e-9y", 1e-9), ("1s", 1 / (365 * 24 * 3600)), ("1d", 1 / 365)):
                    p = EuropeanVanillaOption(strike=spot * m, option_type=ot, maturity=T, contract_multiplier=mult)
                    e = env(spot, 0.3)
                    ref = bs.price(p, e)
                    try:
                        q = EuropeanQuadEngine(method=method).price(p, e)
                        status = f"ok err {q - ref:.3e} rel {abs(q - ref) / ref:.1e}"
                    except NumericalError as exc:
                        status = f"REFUSED ({exc}) true price {ref:.6f}"
                    if "REFUSED" in status or abs(q - ref) > 1e-5:
                        print(f"  {method.name:15s} S {spot:g} mult {mult:g} {ot.name} K/S {m} {label}: {status}")
print("done")
