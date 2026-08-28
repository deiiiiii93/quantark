"""Pilot controls 3 and 5 for the snowball-localvol-1d certification.

(5) ECONOMIC SCALE. A known raw delta must convert to the intended contract
count on BOTH surfaces. Uncorrected, the calm surface's errors would be
overstated by 6207.268 / 4993.105 = 1.243 -- which inflates a measured error and
so risks a false REJECTED, not a merely conservative pass.

(3) FLAT-SURFACE CONTROL. Flatten a surface to a constant vol and the local-vol
PDE must collapse onto the flat-BSM PDE. This separates "the input is wrong"
from "the formula is wrong", and it is the control that settled the original
diagnosis in FINDING-2026-08-26.

Run:
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py
"""

import numpy as np

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.modelvalidation.builders.equity_snowball_localvol import (
    REFERENCE_SPOT,
    load_surface,
    make_localvol_environment,
    resolve_product_spec,
)
from quantark.modelvalidation.study import HedgeContractScale
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    GridVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.localvol import build_dupire_local_vol

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
CALM = "example/modelvalidation/data/iv_surface_20231115.json"
RATE = 0.02
NOTIONAL = 998621.0
HEDGE_MULTIPLIER = 200.0

PRODUCT = {
    "strike_moneyness": 1.0,
    "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85,
    "ko_rate": 0.15,
    "rebate_rate": 0.15,
    "months": 12,
    "maturity": 1.0,
}

SCALE = HedgeContractScale(
    hedge_multiplier=HEDGE_MULTIPLIER,
    hedge_inception_spot=REFERENCE_SPOT,
    notional=NOTIONAL,
)

print("=" * 78)
print("CONTROL 5 -- economic scale on both surfaces")
print("=" * 78)
print(f"delta_quantum = {SCALE.delta_quantum:.9f}   (must be 1.0)")
print()
print(f"{'surface':>18} {'s0':>10} {'contract_mult':>14} "
      f"{'reported':>10} {'true':>10} {'ratio':>9}")
print("-" * 78)

scale_ok = True
for tag, path in (("crash 2024-02-08", CRASH), ("calm  2023-11-15", CALM)):
    surface = load_surface(path, RATE)
    s0 = float(surface.artifact.s0)
    spec = resolve_product_spec({"surface": path, "rate": RATE}, PRODUCT)
    cm = float(spec["contract_multiplier"])

    # A desk holding NOTIONAL of exposure at this index level is hedged by
    # NOTIONAL / (multiplier * s0) futures contracts per unit of raw delta.
    true_contracts = NOTIONAL / (HEDGE_MULTIPLIER * s0)
    reported = SCALE.to_economic("delta", 1.0 * cm)
    ratio = reported / true_contracts
    scale_ok &= abs(ratio - 1.0) < 1e-6
    print(f"{tag:>18} {s0:10.3f} {cm:14.6f} "
          f"{reported:10.6f} {true_contracts:10.6f} {ratio:9.6f}")

print()
print(f"CONTROL 5: {'PASS' if scale_ok else 'FAIL'}")

print()
print("=" * 78)
print("CONTROL 3 -- flat-surface collapse (LV PDE must equal flat-BSM PDE)")
print("=" * 78)

surface = load_surface(CRASH, RATE)
s0 = float(surface.artifact.s0)
flat_vol = float(max(surface.artifact.atm_pillars, key=lambda p: p["T"])["atm_vol"])
valuation_date = make_localvol_environment({"surface": CRASH, "rate": RATE}).valuation_date

flat_grid = GridVolSurface(
    list(surface.artifact.strikes),
    list(surface.artifact.maturities),
    np.full(
        (len(surface.artifact.maturities), len(surface.artifact.strikes)), flat_vol
    ),
)
zero_div = ContinuousDividendYield(0.0)
flat_env = PricingEnvironment(
    rate_curve=FlatRateCurve(RATE),
    valuation_date=valuation_date,
    spot_quote=SpotQuote(s0),
    vol_surface=flat_grid,
    div_yield=zero_div,
)
lv_flat = build_dupire_local_vol(
    flat_grid, spot=s0, rate_curve=flat_env.rate_curve, div_yield=zero_div.get_yield
)

spec = resolve_product_spec({"surface": CRASH, "rate": RATE}, PRODUCT)
product = make_snowball(spec)

lv_greeks = LocalVolSnowballPDESolver(
    params=PDEParams(accuracy="standard"), local_vol_surface=lv_flat
).calculate_greeks(product, flat_env)

bsm_env = PricingEnvironment(
    rate_curve=FlatRateCurve(RATE),
    valuation_date=valuation_date,
    spot_quote=SpotQuote(s0),
    vol_surface=FlatVolSurface(flat_vol),
    div_yield=zero_div,
)
bsm_greeks = SnowballPDESolver(
    params=PDEParams(accuracy="standard")
).calculate_greeks(product, bsm_env)

print(f"flat vol = {flat_vol:.6f}   (longest ATM pillar)")
print()
print(f"{'quantity':>8} {'LV':>16} {'BSM':>16} {'rel':>12}")
print("-" * 56)
flat_ok = True
for key in ("price", "delta", "gamma"):
    lv_v, bsm_v = float(lv_greeks[key]), float(bsm_greeks[key])
    rel = abs(lv_v - bsm_v) / max(abs(bsm_v), 1e-12)
    if key == "delta":
        flat_ok = rel < 1e-3
    print(f"{key:>8} {lv_v:16.8f} {bsm_v:16.8f} {rel:12.3e}")

print()
print(f"CONTROL 3: {'PASS' if flat_ok else 'FAIL'}")
print()
print("PASS CRITERIA")
print("  control 5: ratio == 1.000000 on BOTH surfaces")
print("  control 3: delta rel < 1e-3 (grid alignment differs between the two")
print("             solvers; the NUMBER must agree, not the discretization)")
