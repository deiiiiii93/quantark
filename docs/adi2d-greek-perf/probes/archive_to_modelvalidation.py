"""Archive the ADI 2D Greek certification into the modelvalidation module.

The certification predates ``quantark/modelvalidation`` and was produced by its
own harness (stage 16, schema 13; stage 17, schema 12).  Re-running it to obtain
a native certificate would cost tens of hours of held-out sampling and would
break the evidence chain, so this tool TRANSLATES the banked evidence instead.

What is translated and what is not:

* **Candidate values** are carried verbatim -- ``certifications[g]["pde"]`` in
  the banked payload is exactly what the study's candidate builders recompute,
  which is what makes the anchors a real tripwire rather than a restatement.
* **Gate numbers** are carried verbatim, NOT re-derived.  The two frameworks do
  not gate identically: the certification converts gamma through each case's own
  model spot (``HedgeContractScale`` uses one hedge inception spot for every
  case), and its interval is a Student-t interval with a Bonferroni split across
  two stochastic components plus a PDE refinement envelope and a reference
  substep-bias envelope.  Re-gating the numbers with this module's simpler
  arithmetic would produce verdicts nobody ever earned.
* **Booleans** are either exact (``passed`` is the banked status) or an honest
  recomputation on banked numbers (``envelope_within_bound``).  None are
  invented; the mapping is recorded in the certificate's ``imported`` block.
* **Decisions and aggregates** are carried from the banked decision files.  The
  heston aggregate is the strided-pooled signed bias; the heston_slv aggregate
  is the stage-17 multilevel telescope on held-out seeds.

Run from the worktree root:

    PYTHONPATH=$PWD .venv/bin/python \\
        docs/adi2d-greek-perf/probes/archive_to_modelvalidation.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from quantark.modelvalidation import load_study, validate_payload
from quantark.modelvalidation.anchors import ANCHOR_SCHEMA, machine_fingerprint
from quantark.modelvalidation.evidence import atomic_write_json, atomic_write_text
from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.report import render_markdown

ROOT = Path(__file__).resolve().parents[3]
STUDY_YAML = ROOT / "example/modelvalidation/adi2d_snowball_greeks.yaml"

STAGE16 = ROOT / "output/p18_strided"
STAGE17 = ROOT / "output/p18_slv_amendment"

DEFAULT_BANK = ROOT / "docs/modelvalidation/certificates/adi2d-snowball-greeks"

#: Banked variant name -> the study's candidate name.
CANDIDATE_BY_VARIANT = {
    "heston": "equity.snowball.heston_pde",
    "heston_slv": "equity.snowball.heston_slv_pde",
}

QUANTITIES = ("delta", "gamma")

#: The commit the stage-16 evidence was produced at, as the stage-17 amendment
#: pinned it (``PARENT_SOURCE_COMMIT``) before validating against that parent.
STAGE16_SOURCE_COMMIT = "258fd7ec39416335a00e7fad70822c15c8c1294a"

#: Per-cell bound the certification declared, in hedge contracts.
CELL_BOUND_CONTRACTS = 0.5
#: Aggregate signed-bias bound, in hedge contracts.
BIAS_BOUND_CONTRACTS = 0.1
#: Share of the cell bound the candidate's own discretization envelope may use.
ENVELOPE_FRACTION = 0.5


def _interval_edge(verdict: dict) -> float:
    """The conservative edge of the banked comparison interval, in contracts."""
    low, high = verdict["interval"]
    return max(abs(float(low)), abs(float(high)))


def build_cells(stage16: dict, stage17: dict) -> list[dict]:
    """One module cell per (candidate, case, quantity), carrying banked numbers.

    The amendment re-measured the heston_slv aggregate only; its per-cell
    evidence is the parent's, carried by byte-linked reference. So cells come
    from the parent for both variants, and the amendment's own cells are used
    only to confirm the two agree.
    """
    amended = {
        (cell["variant"], cell["case"]["name"]): cell for cell in stage17["cells"]
    }

    cells: list[dict] = []
    for cell in stage16["cells"]:
        variant = cell["variant"]
        case_name = cell["case"]["name"]
        candidate = CANDIDATE_BY_VARIANT[variant]

        twin = amended.get((variant, case_name))
        for quantity in QUANTITIES:
            banked = cell["certifications"][quantity]
            verdict = banked["verdict"]

            if twin is not None:
                other = twin["certifications"][quantity]["pde"]
                if other != banked["pde"]:
                    raise ValueError(
                        f"{variant}/{case_name}/{quantity}: the amendment's "
                        "candidate value differs from its parent's; the chain "
                        "is not byte-linked"
                    )

            envelope_c = float(verdict["pde_discretization_envelope"])
            status = str(verdict["status"])
            cells.append(
                {
                    "candidate": candidate,
                    "case": case_name,
                    "quantity": quantity,
                    "candidate_value": float(banked["pde"]),
                    "reference": {
                        "value": float(banked["reference"]),
                        "se": float(banked["reference_standard_error"]),
                    },
                    "gate": {
                        "signed_err_c": float(verdict["estimate_difference"]),
                        "se_c": float(verdict["reference_standard_error"]),
                        "interval_c": _interval_edge(verdict),
                        "se_budget_met": status == "PASS",
                        "interval_within_bound": status == "PASS",
                        "envelope_c": envelope_c,
                        "envelope_within_bound": (
                            envelope_c <= ENVELOPE_FRACTION * CELL_BOUND_CONTRACTS
                        ),
                        "passed": status == "PASS",
                    },
                    "verdict": status,
                    "error": None,
                    "identity_hash": stage16["cell_identities"][f"{variant}/{case_name}"],
                    # The banked verdict, whole. Every number this module cannot
                    # represent -- the reference substep-bias envelope, the
                    # Bonferroni component confidence, the degrees of freedom --
                    # stays attached to the cell it belongs to.
                    "source_verdict": verdict,
                }
            )
    return cells


def build_aggregates(stage16: dict, stage17: dict) -> list[dict]:
    """The signed-bias aggregates, one per candidate, on delta.

    Gamma has no aggregate bias gate: the certification declared one only for
    delta, and inventing a gamma aggregate here would claim a test that was
    never run.
    """
    aggregates = []
    for variant, source in (
        # heston is final at stage 16; heston_slv was re-decided by the
        # stage-17 amendment on held-out seeds, so its aggregate comes from there.
        ("heston", stage16["decisions"]["heston"]),
        ("heston_slv", stage17["decisions"]["heston_slv"]),
    ):
        bias = source["delta_bias"]
        low, high = bias["interval"]
        centre = 0.5 * (float(low) + float(high))
        half_width = 0.5 * (float(high) - float(low))
        aggregates.append(
            {
                "candidate": CANDIDATE_BY_VARIANT[variant],
                "quantity": "delta",
                "mean_signed_bias_c": centre,
                "se_of_mean_c": half_width,
                "within_bound": bias["status"] == "PASS",
                "se_adequate": bias["status"] == "PASS",
                "passed": bias["status"] == "PASS",
                "cells": 7,
                # se_of_mean_c above is a HALF-WIDTH, not a standard error: the
                # certification's aggregate is an interval, built by Welch-t over
                # independent cohorts (heston_slv) or from a strided-pooled
                # variance with cross-cell CRN coupling (heston). Recording the
                # half-width keeps the reported edge equal to the banked edge.
                "source_bias": bias,
            }
        )
    return aggregates


def build_payload(study, stage16: dict, stage17: dict, s16_dec: dict, s17_dec: dict) -> dict:
    cells = build_cells(stage16, stage17)
    decisions = {
        CANDIDATE_BY_VARIANT[variant]: (
            "ADMITTED"
            if stage17["decisions"][variant]["route"] == "pde"
            else "INCONCLUSIVE"
        )
        for variant in CANDIDATE_BY_VARIANT
    }

    payload = {
        "schema": 1,
        "study": {
            "name": study.name,
            "source_text": study.source_text,
            "quantities": list(study.quantities),
            "bounds": {
                "cell": CELL_BOUND_CONTRACTS,
                "mean_signed_bias": BIAS_BOUND_CONTRACTS,
                "se_budget_fraction": 0.25,
                "interval_k": 2.0,
                "envelope_fraction": ENVELOPE_FRACTION,
            },
            "sampling": {
                "paths_per_batch": study.sampling.paths_per_batch,
                "min_batches": study.sampling.min_batches,
                "max_batches": study.sampling.max_batches,
                "seed": study.sampling.seed,
                "bump": study.sampling.bump,
            },
            "quick": False,
            "cases": [
                {
                    "name": case.name,
                    "environment_params": dict(case.environment_params),
                    "product_params": dict(case.product_params),
                }
                for case in study.cases
            ],
            "candidates": [
                {"name": c.name(), "params": dict(c.params())} for c in study.candidates
            ],
        },
        "runtime": _runtime(stage16),
        "reference_config": dict(study.reference.config()),
        "references": _references(stage16),
        "cells": cells,
        "aggregates": build_aggregates(stage16, stage17),
        "decisions": decisions,
        "wall_clock_seconds": float(stage16.get("elapsed_seconds", 0.0))
        + float(stage17.get("elapsed_seconds", 0.0)),
        "imported": _imported_block(stage16, stage17, s16_dec, s17_dec),
    }
    payload["projected_sha256"] = _digest(payload)
    return payload


def _runtime(stage16: dict) -> dict:
    """The banking machine, renamed into this module's runtime vocabulary.

    Not cosmetic: ``assert_anchors`` compares the anchor file's fingerprint to
    the running machine to decide exact-vs-tolerance, and the report prints this
    block so a reviewer can check the run happened where it claims.
    """
    env = stage16["runtime_environment"]
    return {
        "platform": env["platform"],
        "machine": env["machine"],
        "python": env["python_version"],
        "numpy": env["numpy_version"],
        # The stage-16 payload does not record its own commit; the stage-17
        # amendment pins it as the parent it validated against, which is the
        # same fact from the side that had to check it.
        "quantark_git_sha": STAGE16_SOURCE_COMMIT,
        "scipy": env.get("scipy_version"),
        "python_implementation": env.get("python_implementation"),
    }


def _references(stage16: dict) -> dict:
    """One reference block per case, pooled over both variants.

    The two variants have separate benchmarks (separate seed families and, for
    SLV, a different estimator), so a per-case block records both rather than
    pretending to one shared benchmark.
    """
    by_case: dict[str, dict] = {}
    for cell in stage16["cells"]:
        entry = by_case.setdefault(
            cell["case"]["name"],
            {
                "seeds": [],
                "batches": 0,
                "stopped_reason": "declared_allocation_exhausted",
                "values": {},
                "std_errors": {},
                "identity_hash": "",
                "by_variant": {},
            },
        )
        variant = cell["variant"]
        entry["by_variant"][variant] = {
            "values": {
                q: float(cell["certifications"][q]["reference"]) for q in QUANTITIES
            },
            "std_errors": {
                q: float(cell["certifications"][q]["reference_standard_error"])
                for q in QUANTITIES
            },
            "identity_hash": stage16["cell_identities"][
                f"{variant}/{cell['case']['name']}"
            ],
        }
        # The top-level block reports the heston arm, the study's base variant;
        # by_variant carries both so nothing is lost.
        if variant == "heston":
            entry["values"] = entry["by_variant"][variant]["values"]
            entry["std_errors"] = entry["by_variant"][variant]["std_errors"]
            entry["identity_hash"] = entry["by_variant"][variant]["identity_hash"]
            entry["batches"] = int(
                stage16["sampling_by_variant"]["heston"]["batches_by_case"][
                    cell["case"]["name"]
                ]
            )
            entry["seeds"] = [int(stage16["reference_seeds"]["heston"])]
    return by_case


def _imported_block(stage16: dict, stage17: dict, s16_dec: dict, s17_dec: dict) -> dict:
    return {
        "reason": (
            "The ADI 2D Greek certification predates quantark/modelvalidation. "
            "Its evidence is translated from the producing harness rather than "
            "re-measured: the benchmark is a multilevel control-variate "
            "telescope that cost 28.6 hours of held-out production sampling, "
            "and re-running it would break the evidence chain it is banked on."
        ),
        "harness": [
            "example/mo_volmodels/16_adi_greek_certification.py",
            "example/mo_volmodels/17_adi_slv_aggregate_certification.py",
        ],
        "evidence_files": {
            "stage16_certification": "evidence/stage16_greek_certification.json",
            "stage16_decision": "evidence/stage16_decision.json",
            "stage17_slv_amendment": "evidence/stage17_slv_aggregate_amendment.json",
            "stage17_decision": "evidence/stage17_decision.json",
        },
        "source_digests": {
            "stage16_schema": stage16["schema_version"],
            "stage16_evidence_sha256": stage16["evidence_sha256"],
            "stage16_implementation_sha256": stage16["implementation_sha256"],
            "stage16_numerical_implementation_sha256": stage16[
                "numerical_implementation_sha256"
            ],
            "stage16_run_configuration_sha256": stage16["run_configuration_sha256"],
            "stage16_decision_sha256": s16_dec.get("decision_sha256"),
            "stage17_schema": stage17["schema_version"],
            "stage17_evidence_sha256": stage17["evidence_sha256"],
            "stage17_parent_evidence_sha256": stage17["parent_certificate"][
                "evidence_sha256"
            ]
            if isinstance(stage17.get("parent_certificate"), dict)
            else None,
            "stage17_decision_sha256": s17_dec.get("decision_sha256"),
        },
        "gate_differences": [
            "Gamma is converted to hedge contracts through each CASE's own model "
            "spot, not one hedge inception spot; HedgeContractScale cannot "
            "express that, so the economic numbers here are the harness's own "
            "and were not re-derived.",
            "interval_c is the conservative edge of the harness's comparison "
            "interval: a Student-t interval on the reference, Bonferroni-split "
            "at 97.5% across two stochastic components, widened by the PDE "
            "refinement envelope and the reference substep-bias envelope. This "
            "module's |err| + 2*SE would be narrower.",
            "se_budget_met and interval_within_bound restate the harness's own "
            "PASS, whose declared reason is that the comparison interval is "
            "wholly inside the economic bound. envelope_within_bound is "
            "recomputed here from the banked envelope against half the cell "
            "bound.",
            "The delta aggregate is an interval, not a mean +/- SE: "
            "mean_signed_bias_c is its centre and se_of_mean_c its HALF-WIDTH, "
            "so the reported edge equals the banked edge. heston pools strided "
            "across common scrambles; heston_slv is the stage-17 multilevel "
            "telescope on held-out seeds 20260811/12 with Welch-t across "
            "independent cohorts.",
            "Gamma carries no aggregate bias gate, because the certification "
            "declared one only for delta.",
        ],
        "not_runnable": (
            "python -m quantark.modelvalidation run will refuse: the reference "
            "builder declares the external benchmark and raises rather than "
            "standing in for it. The candidate arm IS live, which is what "
            "assert_anchors exercises."
        ),
    }


def _digest(payload: dict) -> str:
    from quantark.modelvalidation.evidence import projected_sha256

    return projected_sha256(payload)


def build_anchors(payload: dict, study) -> dict:
    """Anchor the banked candidate values, so CI re-runs and compares to them."""
    grouped: dict[tuple, dict] = {}
    for cell in payload["cells"]:
        if cell["verdict"] == "ERROR" or cell["candidate_value"] is None:
            continue
        grouped.setdefault((cell["candidate"], cell["case"]), {})[
            cell["quantity"]
        ] = cell["candidate_value"]

    return {
        "schema": ANCHOR_SCHEMA,
        "study_source_text": study.source_text,
        "fingerprint": machine_fingerprint(),
        "rel_tol": 1e-9,
        "abs_tol": 1e-12,
        "certificate_sha256": payload["projected_sha256"],
        "anchors": [
            {"candidate": candidate, "case": case, "values": values}
            for (candidate, case), values in sorted(grouped.items())
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_BANK / "2026-08-19")
    args = parser.parse_args(argv)

    study = load_study(STUDY_YAML)
    stage16 = json.loads((STAGE16 / "adi_greek_certification.json").read_text())
    s16_dec = json.loads(
        (STAGE16 / "adi_greek_certification_decision.json").read_text()
    )
    stage17 = json.loads((STAGE17 / "adi_greek_certification.json").read_text())
    s17_dec = json.loads(
        (STAGE17 / "adi_greek_certification_decision.json").read_text()
    )

    payload = build_payload(study, stage16, stage17, s16_dec, s17_dec)
    validate_payload(payload)

    out = args.out
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "certificate.json", payload)
    atomic_write_text(out / "report.md", render_markdown(payload))
    atomic_write_text(out / "report.html", render_html(payload))
    atomic_write_json(out / "anchors.json", build_anchors(payload, study))

    for source, name in (
        (STAGE16 / "adi_greek_certification.json", "stage16_greek_certification.json"),
        (STAGE16 / "adi_greek_certification_decision.json", "stage16_decision.json"),
        (
            STAGE17 / "adi_greek_certification.json",
            "stage17_slv_aggregate_amendment.json",
        ),
        (STAGE17 / "adi_greek_certification_decision.json", "stage17_decision.json"),
    ):
        shutil.copyfile(source, out / "evidence" / name)

    print(f"banked {len(payload['cells'])} cells to {out}")
    print(f"  certificate digest {payload['projected_sha256']}")
    for name, decision in sorted(payload["decisions"].items()):
        print(f"  {name:38s} {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
