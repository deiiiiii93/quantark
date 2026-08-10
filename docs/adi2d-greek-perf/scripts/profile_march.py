"""cProfile one production-grade Heston ADI march (near_ko target grid).

The Greek solve IS one march (delta/gamma read off the same surface), so
profiling price() at the certified target grid ranks the true hotspots.
"""
from __future__ import annotations

import cProfile
import importlib.util
import io
import pstats
import sys
import time

spec = importlib.util.spec_from_file_location(
    "cert16",
    "/private/tmp/quant-ark-adi-greek-certification/example/mo_volmodels/16_adi_greek_certification.py",
)
cert = importlib.util.module_from_spec(spec)
sys.modules["cert16"] = cert
spec.loader.exec_module(cert)

from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

CASE = [c for c in cert.certification_cases(quick=False) if c.name == "near_ko"][0]
product = cert.make_snowball(CASE)
env = cert.make_environment(CASE.spot, 0.20)
TARGET = cert.grid_ladders(CASE.maturity, quick=False)["target"]

eng = HestonSnowballPDESolver(
    CASE.params,
    n_x=TARGET.n_x, n_v=TARGET.n_v, n_t=TARGET.n_t,
    grid_style="concentrated",
    v0_boundary="degenerate_pde",
    variance_grid_mode="auto",
    v_drift_scheme="adaptive_upwind",
    params=PDEParams(cache_enabled=False),
    barrier_greek_steps_per_tick=0,
    greek_min_n_x=0, greek_min_n_v=0,
    greek_min_steps_per_year=0, barrier_greek_min_n_x=0,
)

t0 = time.time()
px_warm = eng.price(product, env)   # unprofiled wall-time baseline
wall = time.time() - t0
print(f"grid n_x={TARGET.n_x} n_v={TARGET.n_v} n_t={TARGET.n_t}")
print(f"unprofiled price={px_warm:+.6f}  wall={wall:.1f}s\n", flush=True)

pr = cProfile.Profile()
pr.enable()
px = eng.price(product, env)
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
ps.print_stats(22)
out = buf.getvalue()
# keep the header + rows, drop the long path prefixes for readability
out = out.replace("/private/tmp/quant-ark-adi-greek-certification/", "")
print(out)
