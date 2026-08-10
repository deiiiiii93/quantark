"""Independent n_t-refinement check on the sigma-collapse cell.

Section 5.9 of the rebaseline design says the centered v-stencil DIVERGES
under time refinement (Rannacher damping dilutes), while a monotone stencil
converges.  This reruns that experiment against the SHIPPED adaptive_upwind
+ path_focused implementation, on a reduced but self-consistent ladder.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/private/tmp/quant-ark-adi-greek-certification/example/mo_volmodels")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "cert16",
    "/private/tmp/quant-ark-adi-greek-certification/example/mo_volmodels/16_adi_greek_certification.py",
)
cert = importlib.util.module_from_spec(spec)
sys.modules["cert16"] = cert  # dataclass() resolves annotations via sys.modules
spec.loader.exec_module(cert)

from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams

CASE = [c for c in cert.certification_cases(quick=False) if c.name == "sigma_collapse"][0]
product = cert.make_snowball(CASE)
env = cert.make_environment(CASE.spot, 0.20)

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
    "path_focused + centered (isolates the stencil)": dict(
        variance_grid_mode="path_focused", v_drift_scheme="centered"
    ),
}

N_X, N_V = 200, 60
N_T_LADDER = [int(x) for x in sys.argv[1:]] or [600, 1200, 2400]

print(f"case={CASE.name} T={CASE.maturity} spot={CASE.spot}")
p = CASE.params
print(
    f"heston v0={p.v0} kappa={p.kappa} theta={p.theta} sigma={p.sigma} "
    f"feller={2*p.kappa*p.theta/p.sigma**2:.1f}"
)
print(f"grid n_x={N_X} n_v={N_V}, n_t ladder {N_T_LADDER}\n")

results = {}
for label, controls in SCHEMES.items():
    print(f"--- {label} ---", flush=True)
    row = {}
    for n_t in N_T_LADDER:
        eng = HestonSnowballPDESolver(
            CASE.params,
            n_x=N_X,
            n_v=N_V,
            n_t=n_t,
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
        if n_t == N_T_LADDER[0]:
            core = eng._make_core(product, env, float(CASE.maturity))
            d = core.variance_operator_diagnostics()
            print(
                f"    diagnostics: grid={d['variance_grid_mode']:13s} "
                f"scheme={d['scheme']:16s} monotone={str(d['monotone']):5s} "
                f"non_monotone_centered_rows={d['centered_non_monotone_nodes']:3d} "
                f"fallback_rows={d['fallback_nodes']:3d} "
                f"max_Peclet={d['max_local_peclet']:.4g} "
                f"theta_node={d['theta_is_node']} v0_node={d['v0_is_node']}",
                flush=True,
            )
        t0 = time.time()
        price = eng.price(product, env)
        row[n_t] = price
        print(f"    n_t={n_t:5d}  price={price:+14.6f}   ({time.time()-t0:6.1f}s)", flush=True)
    results[label] = row
    print(flush=True)

print("=" * 78)
print("REFINEMENT BEHAVIOUR (successive |change| under n_t doubling)")
print("=" * 78)
for label, row in results.items():
    ks = sorted(row)
    deltas = [abs(row[ks[i + 1]] - row[ks[i]]) for i in range(len(ks) - 1)]
    trend = (
        "CONTRACTS" if len(deltas) >= 2 and deltas[-1] < deltas[0]
        else "DIVERGES/FLAT" if len(deltas) >= 2
        else "n/a"
    )
    print(f"{label}")
    print(
        "    "
        + "  ".join(f"n_t={k}:{row[k]:+.4f}" for k in ks)
        + f"   |steps|={[round(d,4) for d in deltas]}  -> {trend}"
    )
