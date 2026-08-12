"""Can a cell close its gate under the sequential loop? Measured, not projected.

Written for `low_feller`, the cell whose PDE-vs-MC gap has no v-axis fix: it has
zero non-monotone variance rows, so `v_drift_scheme="auto"` correctly leaves it on
`adaptive_upwind` and the scheme work cannot move its bias. The question is purely
quantitative — the gap is ~0.16 contracts against a 0.5 bound, so there is room in
principle, and whether the loop closes it depends on how fast the two shrinking
terms fall.

Projections from the 32-batch pilot say ~84-102 batches. This program has been
burned twice by exactly that extrapolation (near_ki went 1024 -> 2048 when the
deep pilot actually measured it), so this measures instead.

**Prefix invariance makes this cheap.** Since batch k is the same batch however
the run is segmented (verified bitwise), one run at the cap can be evaluated at
every prefix — no need to re-run per candidate batch count.

The PDE side is reused from the pilot checkpoint rather than recomputed, which is
sound here because `auto` resolves to `adaptive_upwind` on this cell and is
bitwise identical to it (pinned by `test_adi_auto_v_drift_scheme`).

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_gate_driven_cell.py \
        --variant heston --case low_feller --batches 256
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
PILOT = ROOT / "output" / "allocation_pilot" / "checkpoints"
OUTPUT_DIR = ROOT / "output" / "gate_driven_cell"

from quantark.validation import (  # noqa: E402
    EconomicGreekScale,
    SequentialAdmissionPolicy,
    SequentialAdmissionStatus,
    scan_admission_stream,
)

# The full production matrix: 7 regimes x 2 variants x 2 greeks.
PRODUCTION_TESTS = 28
FAMILY_ALPHA = 0.05
COHORT_FLOOR_BATCHES = 128


def load_stage16():
    spec = importlib.util.spec_from_file_location("stage16_cert", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_cert"] = module
    spec.loader.exec_module(module)
    return module


def _to_contracts(scale: EconomicGreekScale, greek: str, values):
    array = np.asarray(values, dtype=float)
    if greek == "delta":
        return np.asarray([scale.delta_contracts(float(v)) for v in array])
    return np.asarray(
        [scale.gamma_hedge_contract_change(float(v)) for v in array]
    )


def run_cell(s16, variant: str, case_name: str, *, batches: int) -> dict:
    case = {c.name: c for c in s16.certification_cases(quick=False)}[case_name]
    checkpoint = json.loads(
        (PILOT / f"{variant}__{case_name}.json").read_text()
    )["evidence"]
    recorded_scale = checkpoint["economic_scale"]
    scale = EconomicGreekScale(
        model_spot=float(recorded_scale["model_spot"]),
        hedge_inception_spot=float(recorded_scale["hedge_inception_spot"]),
        study_notional=float(recorded_scale["study_notional"]),
        hedge_multiplier=float(recorded_scale["hedge_multiplier"]),
    )

    product = s16.make_snowball(case, dense_ki=True)
    env = s16.make_environment(
        case.spot, float(np.sqrt(max(case.params.v0, case.params.theta)))
    )
    leverage = (
        s16.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    )
    substeps = s16.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][case_name]
    if variant == "heston":
        bridge = s16.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
        extra = {
            "heston_spot_bridge_strata": bridge["strata"],
            "heston_spot_bridge_dimensions": bridge["dimensions"],
        }
        paths_per_batch = s16.PRODUCTION_HESTON_PATHS_PER_BATCH
        seed = s16.HESTON_REFERENCE_SEED
    else:
        bridge = s16.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
        extra = {
            "slv_spot_bridge_strata": bridge["strata"],
            "slv_spot_bridge_dimensions": bridge["dimensions"],
        }
        paths_per_batch = s16.PRODUCTION_SLV_PATHS_PER_BATCH
        seed = s16.SLV_PRIMARY_SEED

    # The MC is the expensive part and the analysis below is where bugs live, so
    # the raw per-batch series is cached the moment it exists. A crash in the
    # gate arithmetic must never cost the simulation again.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT_DIR / f"raw__{variant}__{case_name}__B{int(batches)}.json"
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text())
        print(f"  reusing cached raw series from {cache_path.name}")
    else:
        raw = {}
        for level, level_substeps in (
            ("target", substeps["target"]),
            ("fine", substeps["fine"]),
        ):
            started = time.perf_counter()
            result = s16.paired_mc_reference(
                variant,
                case,
                product,
                env,
                leverage,
                paths_per_batch=paths_per_batch,
                batches=int(batches),
                seed=seed,
                substeps=int(level_substeps),
                bump=s16.SPOT_BUMP,
                rqmc_batch_workers=(
                    s16.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE[variant][
                        case_name
                    ]
                ),
                **extra,
            )
            raw[level] = {
                "substeps": int(level_substeps),
                "batch_delta": [float(v) for v in result.batch_delta],
                "batch_gamma": [float(v) for v in result.batch_gamma],
            }
            cache_path.write_text(json.dumps(raw, indent=1, sort_keys=True))
            print(
                f"  {level:<6} substeps={level_substeps:<3} "
                f"{time.perf_counter() - started:7.1f}s  (cached)",
                flush=True,
            )

    results = {}
    for greek in ("delta", "gamma"):
        certification = checkpoint["certifications"][greek]
        pde_contracts = float(
            _to_contracts(scale, greek, [certification["pde"]])[0]
        )
        # pde_envelope_contracts is a per-axis breakdown; the gate consumes its
        # total, which the verdict records directly.
        pde_envelope = float(certification["verdict"]["pde_discretization_envelope"])
        bound = float(certification["verdict"]["economic_bound"])

        key = "batch_delta" if greek == "delta" else "batch_gamma"
        fine = _to_contracts(scale, greek, raw["fine"][key])
        target = _to_contracts(scale, greek, raw["target"][key])
        # The fine level is the oracle; fine - target is the substep bias proxy.
        substep = fine - target

        policy = SequentialAdmissionPolicy(
            family_alpha=FAMILY_ALPHA,
            tests=PRODUCTION_TESTS,
            min_batches=16,
            aggregate_floor_batches=COHORT_FLOOR_BATCHES,
            planned_batches=int(batches),
            max_batches=int(batches),
        )
        decision = scan_admission_stream(
            policy=policy,
            pde_value=pde_contracts,
            greek_series=fine,
            substep_series=substep,
            pde_discretization_envelope=pde_envelope,
            economic_bound=bound,
        )
        results[greek] = {
            "decision": decision.as_dict(),
            "pde_contracts": pde_contracts,
            "pde_envelope": pde_envelope,
            "policy_sha256": policy.sha256()[:12],
            "mc_mean_contracts": float(np.mean(fine)),
            "substep_mean_contracts": float(np.mean(substep)),
        }
        status = decision.status
        verdict = (
            f"CLOSED at {decision.batches_used} batches"
            if status is SequentialAdmissionStatus.ADMIT
            else f"{status.value} at {decision.batches_used}"
        )
        print(
            f"  {greek:<6} gap {decision.reference_gap:.3f}  "
            f"w_greek {decision.greek_half_width:.3f}  "
            f"pde_env {decision.pde_discretization_envelope:.3f}  "
            f"bias_env {decision.bias_envelope:.3f}  "
            f"total {decision.reference_gap + decision.total_uncertainty:.3f}"
            f" / {bound:.2f}   -> {verdict}"
        )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="heston", choices=("heston", "heston_slv"))
    parser.add_argument("--case", default="low_feller")
    parser.add_argument("--batches", type=int, default=256)
    args = parser.parse_args(argv)

    s16 = load_stage16()
    print(
        f"gate-driven scan: {args.variant}/{args.case} at up to {args.batches} "
        f"batches, floor {COHORT_FLOOR_BATCHES}, K={PRODUCTION_TESTS}\n"
    )
    results = run_cell(s16, args.variant, args.case, batches=args.batches)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{args.variant}__{args.case}.json"
    destination.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "case": args.case,
                "batches_run": args.batches,
                "cohort_floor": COHORT_FLOOR_BATCHES,
                "tests": PRODUCTION_TESTS,
                "results": results,
            },
            indent=1,
            sort_keys=True,
        )
    )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
