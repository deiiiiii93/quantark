"""RQMC-QE reference PV for the synthetic sigma-collapse cell, vs the PDE.

Reduced batch count relative to the production profile (8192 x 256) so this
does not compete with the in-flight certification run, but the SAME estimator:
QE-M paired randomized Sobol with exact affine spot-factor conditioning,
strata/dimensions = 1/1 as the harness assigns to this case.
"""
from __future__ import annotations

import importlib.util
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

PATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 64

CASE = [c for c in cert.certification_cases(quick=False) if c.name == "sigma_collapse"][0]
product = cert.make_snowball(CASE)
env = cert.make_environment(CASE.spot, 0.20)
profile = cert.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[CASE.name]

print(f"case={CASE.name}  T={CASE.maturity}  spot={CASE.spot}")
p = CASE.params
print(f"heston v0={p.v0} kappa={p.kappa} theta={p.theta} sigma={p.sigma}")
print(f"RQMC: {PATHS} paths x {BATCHES} scrambles, substeps=8 (fine leg), "
      f"strata={profile['strata']} dims={profile['dimensions']}")
print(f"(production profile is {cert.PRODUCTION_HESTON_PATHS_PER_BATCH} x "
      f"{cert.PRODUCTION_HESTON_BATCHES_BY_CASE[CASE.name]})\n", flush=True)

t0 = time.time()
ref = cert.paired_mc_reference(
    "heston",
    CASE,
    product,
    env,
    None,
    paths_per_batch=PATHS,
    batches=BATCHES,
    seed=cert.SEED,
    substeps=8,
    bump=cert.SPOT_BUMP,
    heston_spot_bridge_strata=profile["strata"],
    heston_spot_bridge_dimensions=profile["dimensions"],
    rqmc_batch_workers=2,
)
mc_secs = time.time() - t0
print(f"RQMC-QE reference   ({mc_secs:.0f}s)")
print(f"    price = {ref.price:+.6f}  +/- {ref.price_std_error:.6f} (SE)")
print(f"    delta = {ref.delta:+.6f}  +/- {ref.delta_std_error:.6f}")
print(f"    gamma = {ref.gamma:+.6f}  +/- {ref.gamma_std_error:.6f}")
print(f"    unique paths = {ref.total_unique_paths:,}\n", flush=True)

TARGET = cert.grid_ladders(CASE.maturity, quick=False)["target"]
SCHEMES = {
    "shipped (auto->path_focused + adaptive_upwind)": dict(
        variance_grid_mode="auto", v_drift_scheme="adaptive_upwind"
    ),
    "legacy grid + centered": dict(
        variance_grid_mode="legacy", v_drift_scheme="centered", v_grid_power=0.0
    ),
    "power grid 2.5 + centered": dict(
        variance_grid_mode="power", v_drift_scheme="centered"
    ),
}
print(f"PDE at the certification target grid {TARGET.as_dict()}")
for label, controls in SCHEMES.items():
    eng = HestonSnowballPDESolver(
        CASE.params,
        n_x=TARGET.n_x,
        n_v=TARGET.n_v,
        n_t=TARGET.n_t,
        grid_style="concentrated",
        v0_boundary="degenerate_pde",
        params=PDEParams(cache_enabled=False),
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0,
        greek_min_n_v=0,
        greek_min_steps_per_year=0,
        barrier_greek_min_n_x=0,
        **controls,
    )
    t0 = time.time()
    price = eng.price(product, env)
    err = price - ref.price
    sigmas = err / ref.price_std_error if ref.price_std_error > 0 else float("nan")
    print(
        f"    {label:48s} price={price:+.6f}  "
        f"err={err:+.6f}  ({sigmas:+7.1f} SE)   ({time.time()-t0:.0f}s)",
        flush=True,
    )
