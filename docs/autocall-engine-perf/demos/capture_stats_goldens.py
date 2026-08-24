"""Bitwise stats goldens for the QUAD event-stats refactors (same-machine tool).

Capture BEFORE an engine change, `--check` after: every AutocallableEventStats /
PhoenixEventStats field, price(), and price_with_events npv, hex-exact, over a
product matrix. Dev-time gate only — never wire into CI (cross-arch rule).

Run:  .venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py [--check]
"""
import json
import sys
from dataclasses import fields as dc_fields
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

GOLDEN_PATH = "docs/autocall-engine-perf/demos/.stats_goldens.local.json"  # gitignored-by-location


def env(spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )


def cases():
    std = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    yield "snowball_cont_ki", SnowballQuadEngine, std, env()
    yield "snowball_ki_low_spot", SnowballQuadEngine, std, env(spot=74.0)  # knocked-in at valuation
    disc = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[(d + 1) * 1.9 / 96 for d in range(96)],
    )
    yield "snowball_discrete_ki", SnowballQuadEngine, disc, env()
    dko = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, disable_ko_after_ki=True,
    )
    yield "snowball_disable_ko_after_ki", SnowballQuadEngine, dko, env()
    phx = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01,
        num_observations=23,
    )
    yield "phoenix", PhoenixQuadEngine, phx, env()


def stats_hex(stats):
    out = {}
    for f in dc_fields(stats):
        v = getattr(stats, f.name)
        if isinstance(v, np.ndarray):
            out[f.name] = [float(x).hex() for x in np.asarray(v, dtype=float).ravel()]
        elif isinstance(v, float):
            out[f.name] = v.hex()
        else:
            out[f.name] = repr(v)
    return out


def collect():
    data = {}
    for name, cls, product, e in cases():
        engine = cls(params=QuadParams(grid_points=1001))
        price = float(engine.price(product, e))
        engine = cls(params=QuadParams(grid_points=1001))
        npv = float(engine.price_with_events(product, e).npv)
        engine = cls(params=QuadParams(grid_points=1001))
        stats = engine.calculate_event_stats(product, e)
        data[name] = {"price": price.hex(), "npv": npv.hex(),
                      "stats": stats_hex(stats) if stats is not None else None}
    return data


def main():
    data = collect()
    if "--check" in sys.argv:
        with open(GOLDEN_PATH) as fh:
            golden = json.load(fh)
        diffs = []

        def walk(path, a, b):
            if isinstance(a, dict):
                for k in a:
                    walk(f"{path}.{k}", a[k], b.get(k, "<missing>") if isinstance(b, dict) else "<missing>")
            elif a != b:
                diffs.append((path, a, b))

        walk("", golden, data)
        if diffs:
            print(f"BITWISE FAIL: {len(diffs)} diffs")
            for p, a, b in diffs[:20]:
                print(f"  {p}: {a} -> {b}")
            sys.exit(1)
        print("BITWISE OK")
    else:
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"captured -> {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
