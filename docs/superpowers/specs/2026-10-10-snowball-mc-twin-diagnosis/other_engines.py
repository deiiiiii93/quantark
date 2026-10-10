"""Step 4: the other autocallable MC engines on the same contracts and days. For each route and day: the node count
of each form's time grid (the engine's own builder), and the price of each form (W12 / test configuration:
MCParams(seed=7, num_paths=256), seed 11 x 4096 for the BS routes).

Contracts: the W12 continuous-KI snowball and non-memory Phoenix (common.py, numeric aged WITHOUT keep_landing as
W12 did); the W11 ABSOLUTE KO-reset of test/test_ko_reset_past_records.py (discrete KI, numeric aged WITH
keep_landing) and the same KO-reset with a CONTINUOUS KI 75 (built here)."""
from dataclasses import replace

import numpy as np

import common as c
import test_ko_reset_past_records as w11
from test_dated_expiry_settles import HESTON, SLV_LEVERAGE

from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.util.enum import ObservationType

DAYS = (14, 41, 45, 53, 92, 170, 200)


def ko_reset_continuous(form, day):
    """The W11 KO-reset with its knock-in made CONTINUOUS (level 75, monitored over schedule 1)."""
    import test_ko_reset_past_records as m
    orig = m._carry
    m._carry = lambda product, d, k: product          # build unaged, switch the knock-in, then age
    try:
        p = m.ko_reset(form, day)
    finally:
        m._carry = orig
    p.barrier_config = replace(p.barrier_config, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True,
                               ki_observation_schedule=None, ki_observation_dates=None)
    p.validate()
    return m._carry(p, day, None)


PRODUCTS = {
    "snowball": lambda f, d: c.build("snowball", f, d),
    "phoenix": lambda f, d: c.build("phoenix", f, d),
    "ko_reset_discrete_ki": lambda f, d: w11.ko_reset(f, d),
    "ko_reset_continuous_ki": ko_reset_continuous,
}


def mc(seed=7, paths=256):
    return MCParams(seed=seed, num_paths=paths)


#: (route, product kind, engine factory, reads the market IV grid)
ROUTES = [
    ("bs_mc", "snowball", lambda: SnowballMCEngine(params=mc(11, 4096)), False),
    ("bs_mc", "phoenix", lambda: PhoenixMCEngine(params=mc(11, 4096)), False),
    ("bs_mc", "ko_reset_discrete_ki", lambda: SnowballMCEngine(params=mc(11, 4096)), False),
    ("bs_mc", "ko_reset_continuous_ki", lambda: SnowballMCEngine(params=mc(11, 4096)), False),
    ("lv_mc", "snowball", lambda: LocalVolSnowballMCEngine(params=mc()), True),
    ("heston_mc", "snowball", lambda: HestonSnowballMCEngine(HESTON, params=mc()), False),
    ("slv_mc", "snowball", lambda: HestonSLVSnowballMCEngine(HESTON, params=mc(), leverage_surface=SLV_LEVERAGE),
     True),
    ("lv_mc", "phoenix", lambda: LocalVolPhoenixMCEngine(params=mc()), True),
    ("heston_mc", "phoenix", lambda: HestonPhoenixMCEngine(HESTON, params=mc()), False),
    ("slv_mc", "phoenix", lambda: HestonSLVPhoenixMCEngine(HESTON, params=mc(), leverage_surface=SLV_LEVERAGE),
     True),
    ("lv_mc", "ko_reset_continuous_ki", lambda: LocalVolSnowballMCEngine(params=mc()), True),
    ("heston_mc", "ko_reset_continuous_ki", lambda: HestonSnowballMCEngine(HESTON, params=mc()), False),
]


def nodes(eng, product, e):
    eng._term_ctx = (e, product.strike)
    eng._df = make_df_fn(e)
    if isinstance(product, KnockOutResetSnowballOption):
        T = product.get_max_maturity_time(e)
        g = eng._build_time_grid_ko_reset(product, e, T)
        at = g["all_times"]
    else:
        T = product.get_maturity(e)
        at = eng._build_time_grid(product, e, T)[0]
    return len(at), float(np.diff(np.concatenate([[0.0], at])).min())


if __name__ == "__main__":
    import sys
    routes, days = ROUTES, DAYS
    if sys.argv[1:] == ["ko_reset"]:      # the KO-reset days grid_scan.py found (26 both forms; 65/78/82 numeric)
        routes = [r for r in ROUTES if r[0] == "bs_mc" and r[1] == "ko_reset_continuous_ki"]
        days = (26, 45, 65, 78, 82)
    for route, kind, factory, grid in routes:
        for day in days:
            e = c.env(100.0, c.on(day), grid=grid)
            out = {}
            for form in ("dated", "numeric"):
                try:
                    n, mdt = nodes(factory(), PRODUCTS[kind](form, day), e)
                    v = float(factory().price(PRODUCTS[kind](form, day), e))
                    out[form] = (v, n, mdt)
                except Exception as exc:  # noqa: BLE001
                    out[form] = None
                    print(f"{route:9s} {kind:22s} day={day:3d} {form} RAISES {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:120]}")
            if out["dated"] and out["numeric"]:
                (dv, dn, dm), (nv, nn, nm) = out["dated"], out["numeric"]
                flag = "" if abs(dv - nv) <= 1e-10 * max(1.0, abs(dv)) else "  <-- TWIN GAP"
                print(f"{route:9s} {kind:22s} day={day:3d} dated={dv:.6f} numeric={nv:.6f} gap={dv - nv:+.3e} "
                      f"nodes {dn}/{nn} min_dt {dm:.1e}/{nm:.1e}{flag}", flush=True)
