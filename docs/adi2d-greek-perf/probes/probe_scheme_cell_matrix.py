"""Cross-scheme PDE cell matrix: what would P1.4 certify at each v_drift_scheme?

The plan of record (STATE-AND-PLAN-2026-08-10.md D-0) says the v-axis scheme must
be fixed *before* MC references are regenerated, because certifying a knowingly
first-order-biased PDE would land INCONCLUSIVE again. But the scheme lives in a
single hardcoded production constant that three stages pin to each other
(stage-11 gate, stage-12 backtest, stage-16 certification), so changing it is a
production default change -- decision C-G6, which the plan defers to *after*
P1.4. That circularity has to be resolved with data, not sequencing.

This probe supplies the data. For every certification cell it solves the target
grid under each candidate scheme and reports the Greek in the same economic unit
the certification bounds use (futures contracts), plus wall time. It touches no
engine source: the scheme is injected by patching the stage-16 production
constant at runtime, exactly as the standalone-demo rule requires.

What it cannot do: say which scheme is *right*. Only the regenerated MC
references can do that. What it can do: show how far apart the two schemes are
per cell relative to the +/-0.10-contract aggregate bound, prove semi_lagrangian
runs clean on all 14 cells at production grids before hours of MC compute depend
on it, and price the switch.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_scheme_cell_matrix.py
    ... --cases near_expiry near_ko          # cheap subset first
    ... --schemes adaptive_upwind semi_lagrangian centered
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
# Long-lived artifacts live under /Users in the repo output tree, never
# /private/tmp -- see RECOVERY.md for what the 2026-08-10 crash cost.
OUTPUT_DIR = ROOT / "output" / "scheme_cell_matrix"

SCHEMES = ("adaptive_upwind", "semi_lagrangian", "centered")


def load_stage16():
    """Import the stage-16 harness by path, registered so dataclasses resolve."""
    spec = importlib.util.spec_from_file_location("stage16_cert", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_cert"] = module
    spec.loader.exec_module(module)
    return module


def backend_identity() -> dict:
    """Which accelerators produced these numbers."""
    from quantark.montecarlo import qe_kernels
    from quantark.util.numerical import tridiag

    return {
        "tridiag_backend": tridiag.tridiag_backend(),
        "qe_backend": qe_kernels.qe_backend(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def solve_cell(s16, variant: str, case, scheme: str) -> dict:
    """One target-grid central-bump solve for a cell under one v-axis scheme."""
    product = s16.make_snowball(case, dense_ki=True)
    env = s16.make_environment(
        case.spot, np.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = (
        s16.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    )
    ladders = s16.grid_ladders(
        case.maturity, quick=False, dense_ki_stencil=(case.name == "near_ki")
    )
    target = ladders["target"]

    # Runtime injection: make_pde_engine copies PRODUCTION_ENGINE_CONTROLS, so
    # patching the constant is enough. Restored by the caller's finally block.
    engine = s16.make_pde_engine(variant, case, target, leverage)

    started = time.perf_counter()
    greeks = s16.central_bump_greeks(engine, product, env, s16.SPOT_BUMP)
    elapsed = time.perf_counter() - started

    scale = s16.EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=s16.DEFAULT_HEDGE_INCEPTION_SPOT,
        study_notional=s16.STUDY_NOTIONAL,
        hedge_multiplier=s16.HEDGE_MULTIPLIER,
    )
    return {
        "variant": variant,
        "case": case.name,
        "scheme": scheme,
        "grid": target.as_dict(),
        "seconds": elapsed,
        "price": greeks["price"],
        "delta_raw": greeks["delta"],
        "gamma_raw": greeks["gamma"],
        "delta_contracts": s16._economic_value(scale, "delta", greeks["delta"]),
        "gamma_contracts": s16._economic_value(scale, "gamma", greeks["gamma"]),
    }


def run(
    *,
    variants: Sequence[str],
    case_names: Optional[Sequence[str]],
    schemes: Sequence[str],
    output_path: Path,
) -> dict:
    s16 = load_stage16()
    cases = [
        case
        for case in s16.certification_cases(quick=False)
        if case_names is None or case.name in set(case_names)
    ]
    if not cases:
        raise ValueError(f"no certification case matched {case_names}")

    baseline_controls = dict(s16.PRODUCTION_ENGINE_CONTROLS)
    payload = {
        "purpose": (
            "per-cell PDE Greeks at the certification target grid under each "
            "candidate v_drift_scheme, in certification economic units"
        ),
        "backends": backend_identity(),
        "baseline_production_engine_controls": baseline_controls,
        "schemes": list(schemes),
        "rows": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cheapest cells first: an early failure should cost seconds, not an hour.
    ordered = sorted(cases, key=lambda case: case.maturity)
    for case in ordered:
        for variant in variants:
            for scheme in schemes:
                s16.PRODUCTION_ENGINE_CONTROLS = dict(
                    baseline_controls, v_drift_scheme=scheme
                )
                label = f"{variant}/{case.name}/{scheme}"
                try:
                    row = solve_cell(s16, variant, case, scheme)
                except Exception as exc:  # a scheme failing a cell IS the result
                    row = {
                        "variant": variant,
                        "case": case.name,
                        "scheme": scheme,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"  {label:52s} FAILED {row['error']}", flush=True)
                else:
                    print(
                        f"  {label:52s} delta={row['delta_contracts']:+9.4f} "
                        f"gamma={row['gamma_contracts']:+9.4f} "
                        f"{row['seconds']:7.1f}s",
                        flush=True,
                    )
                finally:
                    s16.PRODUCTION_ENGINE_CONTROLS = dict(baseline_controls)
                payload["rows"].append(row)
                # Checkpoint every row; the crash lesson was that unwritten
                # results are lost results.
                output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def report(payload: dict) -> str:
    """Scheme gaps per cell, in the units the certification bounds use."""
    rows = {
        (row["variant"], row["case"], row["scheme"]): row
        for row in payload["rows"]
        if "error" not in row
    }
    schemes = [s for s in payload["schemes"] if s != "adaptive_upwind"]
    keys = sorted({(v, c) for v, c, _ in rows})

    lines = [
        "",
        "Delta gap vs adaptive_upwind, futures contracts "
        "(aggregate bound is +/-0.10)",
        "",
        f"| {'cell':28s} | {'upwind':>9s} | "
        + " | ".join(f"{s[:14]:>14s}" for s in schemes)
        + " | cost |",
        f"|{'-' * 30}|{'-' * 11}|"
        + "|".join("-" * 16 for _ in schemes)
        + f"|{'-' * 7}|",
    ]
    for variant, case in keys:
        base = rows.get((variant, case, "adaptive_upwind"))
        if base is None:
            continue
        cells = []
        for scheme in schemes:
            other = rows.get((variant, case, scheme))
            cells.append(
                "        --      "
                if other is None
                else f"{other['delta_contracts'] - base['delta_contracts']:+14.4f}"
            )
        sl = rows.get((variant, case, "semi_lagrangian"))
        cost = "--" if sl is None else f"{sl['seconds'] / base['seconds']:.2f}x"
        lines.append(
            f"| {variant + '/' + case:28s} | {base['delta_contracts']:+9.4f} | "
            + " | ".join(cells)
            + f" | {cost:>5s} |"
        )
    failures = [row for row in payload["rows"] if "error" in row]
    if failures:
        lines += ["", "Failures:"] + [
            f"  {row['variant']}/{row['case']}/{row['scheme']}: {row['error']}"
            for row in failures
        ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+", default=("heston", "heston_slv"),
        choices=("heston", "heston_slv"),
    )
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--schemes", nargs="+", default=("adaptive_upwind", "semi_lagrangian"),
        choices=SCHEMES,
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "scheme_cell_matrix.json"
    )
    args = parser.parse_args(argv)

    payload = run(
        variants=args.variants,
        case_names=args.cases,
        schemes=args.schemes,
        output_path=args.output,
    )
    print(report(payload))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
