"""Greeks ON the exercise date: ``engine.calculate_greeks`` on the dated contract vs its numeric twin at maturity 0.0.

Not in the branch's scope (the ruling is about the value); recorded so the follow-up can see where the two forms'
greeks agree and where they do not (theta bumps a day PAST the dated exercise, which is refused).

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/measure_greeks.py
"""
from __future__ import annotations

import math
import warnings

from test_dated_expiry_settles import BARRIER, EUROPEAN, RANGE_ACCRUAL, SNOWBALL, env_for

warnings.simplefilter("ignore")
KEYS = ("price", "delta", "gamma", "vega", "theta", "rho")


def greeks(family, name, factory, form, spot):
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

    try:
        engine, product, env = factory(), family.build(form), env_for(name, spot)
        out = {k: float(v) for k, v in engine.calculate_greeks(product, env).items() if k in KEYS}
        calc = GreeksCalculator()
        out["theta"] = float(calc.calculate_numerical_theta(product, env, engine))
        out["vega"] = float(calc.calculate_numerical_vega(product, env, engine))
        out["rho"] = float(calc.calculate_numerical_rho(product, env, engine))
        return out
    except Exception as exc:
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:100]}"


def same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return "n/a"
    return "yes" if all(math.isclose(a[k], b.get(k, math.nan), rel_tol=1e-9, abs_tol=1e-9) for k in a) else "NO"


print("family | engine | spot | dated | numeric | equal")
for family, spot in ((EUROPEAN, 110.0), (BARRIER, 110.0), (RANGE_ACCRUAL, 100.0), (SNOWBALL, 90.0)):
    for name, factory in family.engines:
        if name.startswith(("heston", "lv_", "quad_v2")):
            continue
        dated = greeks(family, name, factory, "dated", spot)
        numeric = greeks(family, name, factory, "numeric", spot)
        print(f"{family.name} | {name} | {spot:g} | {dated} | {numeric} | {same(dated, numeric)}")
