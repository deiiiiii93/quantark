"""Joint-refinement attribution probe for the schema-11 signed delta gap.

Targets the two structural-bias cells (sigma_collapse, low_feller) flagged by
the schema-11 INCONCLUSIVE heston_slv verdict. Re-solves ONLY the PDE side at
refined grids / alternative v-drift schemes and compares against the BANKED
schema-11 MC references — no Monte Carlo is run.

Semantics fidelity: product, environment, leverage surface, engine controls,
and the 1% central-bump greek readout are all imported from the certification
runner itself (16_adi_greek_certification.py). The `target` row of every cell
must reproduce the banked `certifications.delta.pde` value exactly; any
mismatch aborts the probe run for that cell.

Usage (one process per variant/case, rows run serially inside a process):
  PYTHONPATH=<repo> python probe_delta_attribution.py \
      --variant heston --case sigma_collapse \
      --rows target:300:135:4800:adaptive_upwind,centered:300:135:4800:centered \
      --out output/delta_bias_attribution/heston_sigma_collapse.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "cert16", ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
)
assert _spec is not None and _spec.loader is not None
cert = importlib.util.module_from_spec(_spec)
sys.modules["cert16"] = cert
_spec.loader.exec_module(cert)

EVIDENCE_PATH = (
    ROOT / "output" / "adi_greek_certification_schema11" / "adi_greek_certification.json"
)


def banked_cell(evidence: dict, variant: str, case_name: str) -> dict:
    for cell in evidence["cells"]:
        if cell["variant"] == variant and cell["case"]["name"] == case_name:
            return cell
    raise KeyError(f"no banked cell for {variant}/{case_name}")


def make_engine(variant: str, case, n_x: int, n_v: int, n_t: int, scheme: str, leverage):
    controls = dict(cert.PRODUCTION_ENGINE_CONTROLS)
    controls.update(
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0,
        greek_min_n_v=0,
        greek_min_steps_per_year=0,
        barrier_greek_min_n_x=0,
        v_drift_scheme=scheme,
    )
    kwargs = dict(
        n_x=n_x,
        n_v=n_v,
        n_t=n_t,
        params=cert.PDEParams(cache_enabled=False),
        **controls,
    )
    if variant == "heston":
        return cert.HestonSnowballPDESolver(case.params, **kwargs)
    if variant == "heston_slv":
        return cert.HestonSLVSnowballPDESolver(
            case.params, leverage_surface=leverage, **kwargs
        )
    raise ValueError(f"unsupported variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["heston", "heston_slv"])
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--rows", required=True, help="comma list of name:n_x:n_v:n_t:scheme"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    evidence = json.loads(EVIDENCE_PATH.read_text())
    cell = banked_cell(evidence, args.variant, args.case)
    banked_delta = float(cell["certifications"]["delta"]["pde"])
    banked_gamma = float(cell["certifications"]["gamma"]["pde"])
    reference_delta = float(cell["certifications"]["delta"]["reference"])
    reference_se = float(cell["certifications"]["delta"]["reference_standard_error"])

    case = next(
        c for c in cert.certification_cases(quick=False) if c.name == args.case
    )
    product = cert.make_snowball(case, dense_ki=True)
    env = cert.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = (
        cert.make_leverage_surface(case.maturity)
        if args.variant == "heston_slv"
        else None
    )
    scale = cert.EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=float(evidence["run_configuration"]["hedge_inception_spot"]),
        study_notional=cert.STUDY_NOTIONAL,
        hedge_multiplier=cert.HEDGE_MULTIPLIER,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for row_spec in args.rows.split(","):
        name, n_x, n_v, n_t, scheme = row_spec.split(":")
        engine = make_engine(
            args.variant, case, int(n_x), int(n_v), int(n_t), scheme, leverage
        )
        started = time.perf_counter()
        values = cert.central_bump_greeks(engine, product, env, cert.SPOT_BUMP)
        seconds = time.perf_counter() - started
        record = {
            "variant": args.variant,
            "case": args.case,
            "row": name,
            "n_x": int(n_x),
            "n_v": int(n_v),
            "n_t": int(n_t),
            "scheme": scheme,
            "delta": values["delta"],
            "gamma": values["gamma"],
            "price": values["price"],
            "seconds": round(seconds, 2),
            "banked_pde_delta": banked_delta,
            "banked_pde_gamma": banked_gamma,
            "reference_delta": reference_delta,
            "reference_se_delta": reference_se,
            "delta_gap_contracts": scale.delta_contracts(values["delta"] - reference_delta),
            "delta_minus_banked": values["delta"] - banked_delta,
        }
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if name == "target" and record["delta_minus_banked"] != 0.0:
            drift = abs(record["delta_minus_banked"])
            if drift > 1e-12:
                raise SystemExit(
                    f"FIDELITY FAILURE {args.variant}/{args.case}: probe target delta "
                    f"differs from banked value by {record['delta_minus_banked']:.3e}"
                )
            print(f"fidelity note: sub-1e-12 drift {drift:.3e}", flush=True)


if __name__ == "__main__":
    main()
