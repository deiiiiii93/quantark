"""The framework-owned certification pipeline.

This is the loop no study can customize away: sample the benchmark, evaluate
each candidate, gate the two against each other, decide, and bank the evidence.
Studies choose *what* is certified; this file fixes *how*, which is what makes
one certification comparable to another.

An error in one cell does not end the run -- an hours-scale certification must
not be destroyed by a single engine raising on a single scenario. The cell is
recorded as ERROR (with its traceback), the run continues, and the verdict
lattice makes sure the resulting decision can never be ADMITTED.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import (
    CandidateResult,
    candidate_identity,
    deserialize_candidate_result,
    envelope_from_ladders,
    serialize_candidate_result,
)
from quantark.modelvalidation.candidate import CHECKPOINT_KIND as CANDIDATE_KIND
from quantark.modelvalidation.decisions import Decision, Verdict, decide_candidate, decide_cell
from quantark.modelvalidation.evidence import (
    SCHEMA_VERSION,
    CheckpointStore,
    atomic_write_json,
    atomic_write_text,
    identity_hash,
    projected_sha256,
    validate_durable_root,
)
from quantark.modelvalidation.gates import evaluate_aggregate_gate, evaluate_cell_gate
from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.reference import ReferenceEstimate, run_reference
from quantark.modelvalidation.report import render_markdown
from quantark.modelvalidation.study import CertificationStudy, SamplingPolicy

CERTIFICATE_NAME = "certificate.json"
REPORT_NAME = "report.md"
HTML_REPORT_NAME = "report.html"


@dataclass(frozen=True)
class Certificate:
    """A written certification result."""

    payload: dict
    path: Path


def runtime_environment() -> dict:
    """Describe the machine that produced the evidence.

    Recorded so a reviewer can tell whether an anchor comparison should be exact
    (same machine) or tolerance-based (a different architecture).
    """
    try:
        git_sha: Optional[str] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or None
        )
    except (subprocess.SubprocessError, OSError):
        git_sha = None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "quantark_git_sha": git_sha,
    }


def quick_policy(policy: SamplingPolicy) -> SamplingPolicy:
    """Shrink a sampling policy for a wiring check.

    Quick mode exists to prove the plumbing runs, never to produce bankable
    evidence: the standard errors it reaches will usually leave cells
    UNRESOLVED, which is the correct outcome rather than a bug.
    """
    max_batches = min(4, policy.max_batches)
    return SamplingPolicy(
        paths_per_batch=max(128, policy.paths_per_batch // 8),
        min_batches=min(2, max_batches),
        max_batches=max_batches,
        seed=policy.seed,
        bump=policy.bump,
    )


def reference_block(estimate: ReferenceEstimate, identity: Mapping[str, Any]) -> dict:
    return {
        "values": dict(estimate.values),
        "std_errors": dict(estimate.std_errors),
        "batches": estimate.batches,
        "seeds": list(estimate.seeds),
        "stopped_reason": estimate.stopped_reason,
        "identity_hash": identity_hash(identity),
    }


def evaluate_candidate(
    candidate,
    case,
    store: Optional[CheckpointStore],
    resume: bool,
) -> CandidateResult:
    """Evaluate one candidate for one case, reusing a matching checkpoint."""
    identity = candidate_identity(candidate, case)
    key = f"{candidate.name()}-{case.name}".replace("/", "_")

    if resume and store is not None:
        banked = store.load(CANDIDATE_KIND, key, identity)
        if banked is not None:
            return deserialize_candidate_result(banked)

    result = candidate.evaluate(case)
    if store is not None:
        store.save(CANDIDATE_KIND, key, identity, serialize_candidate_result(result))
    return result


def certify(
    study: CertificationStudy,
    out_dir: str | Path,
    quick: bool = False,
    resume: bool = False,
) -> Certificate:
    """Run a certification and bank its evidence.

    Args:
        study: The study to certify.
        out_dir: Durable output root; results land in ``<out_dir>/<study.name>``.
        quick: Shrink sampling for a wiring check (never bankable evidence).
        resume: Reuse checkpoints whose identity still matches.

    Returns:
        The written :class:`Certificate`.

    Raises:
        ValidationError: an output root in temp storage, or a payload that fails
            its own validation before writing.
    """
    started = time.time()
    root = validate_durable_root(out_dir) / study.name
    store = CheckpointStore(root / "checkpoints")
    sampling = quick_policy(study.sampling) if quick else study.sampling

    references: Dict[str, dict] = {}
    estimates: Dict[str, Optional[ReferenceEstimate]] = {}
    reference_errors: Dict[str, str] = {}

    for case in study.cases:
        try:
            estimate = run_reference(
                builder=study.reference,
                case=case,
                quantities=study.quantities,
                scale=study.scale,
                bounds=study.bounds,
                policy=sampling,
                store=store,
                resume=resume,
            )
            estimates[case.name] = estimate
            references[case.name] = reference_block(
                estimate, study.reference.identity(case)
            )
        except Exception:  # noqa: BLE001 - recorded, not swallowed
            estimates[case.name] = None
            reference_errors[case.name] = traceback.format_exc()
            references[case.name] = {"error": reference_errors[case.name]}

    cells: List[dict] = []

    for candidate in study.candidates:
        name = candidate.name()

        for case in study.cases:
            estimate = estimates[case.name]
            result: Optional[CandidateResult] = None
            error: Optional[str] = None

            if estimate is None:
                error = reference_errors[case.name]
            else:
                try:
                    result = evaluate_candidate(candidate, case, store, resume)
                except Exception:  # noqa: BLE001 - recorded, not swallowed
                    error = traceback.format_exc()

            cells.extend(
                build_cells(
                    study=study,
                    candidate=candidate,
                    case=case,
                    estimate=estimate,
                    result=result,
                    error=error,
                )
            )

    aggregates, decisions = aggregate_and_decide(study, cells)
    payload = assemble_payload(
        study=study,
        sampling=sampling,
        quick=quick,
        references=references,
        cells=cells,
        aggregates=aggregates,
        decisions=decisions,
        started=started,
    )
    return write_certificate(payload, root)


def build_cells(
    study: CertificationStudy,
    candidate,
    case,
    estimate: Optional[ReferenceEstimate],
    result: Optional[CandidateResult],
    error: Optional[str],
) -> List[dict]:
    """Gate one candidate against the benchmark for one case, per quantity."""
    cells: List[dict] = []
    name = candidate.name()
    cell_identity = identity_hash(candidate_identity(candidate, case))

    for quantity in study.quantities:
        if error is not None or result is None or estimate is None:
            cells.append(
                {
                    "candidate": name,
                    "case": case.name,
                    "quantity": quantity,
                    "reference": None,
                    "candidate_value": None,
                    "gate": None,
                    "verdict": decide_cell(None, error=True).value,
                    "error": error,
                    "identity_hash": cell_identity,
                }
            )
            continue

        gate = evaluate_cell_gate(
            candidate_raw=result.values[quantity],
            reference_raw=estimate.values[quantity],
            reference_se_raw=estimate.std_errors[quantity],
            quantity=quantity,
            scale=study.scale,
            bounds=study.bounds,
            envelope_raw=envelope_from_ladders(result.ladders, quantity),
        )
        cells.append(
            {
                "candidate": name,
                "case": case.name,
                "quantity": quantity,
                "reference": {
                    "value": estimate.values[quantity],
                    "se": estimate.std_errors[quantity],
                },
                "candidate_value": result.values[quantity],
                "gate": asdict(gate),
                "verdict": decide_cell(gate, error=False).value,
                "error": None,
                "identity_hash": cell_identity,
            }
        )
    return cells


def aggregate_and_decide(
    study: CertificationStudy, cells: List[dict]
) -> tuple[List[dict], Dict[str, str]]:
    """Roll cells up into aggregate bias gates and one decision per candidate.

    Operates on cell *dicts* so it works identically for freshly computed cells
    and for cells carried forward from a parent certificate.
    """
    aggregates: List[dict] = []
    decisions: Dict[str, str] = {}

    for candidate in study.candidates:
        name = candidate.name()
        own = [cell for cell in cells if cell["candidate"] == name]
        verdicts = [Verdict(cell["verdict"]) for cell in own]

        candidate_aggregates = []
        for quantity in study.quantities:
            gated = [
                cell
                for cell in own
                if cell["quantity"] == quantity and cell["gate"] is not None
            ]
            if not gated:
                continue
            aggregate = evaluate_aggregate_gate(
                [cell["gate"]["signed_err_c"] for cell in gated],
                [cell["gate"]["se_c"] for cell in gated],
                study.bounds,
            )
            candidate_aggregates.append(aggregate)
            aggregates.append(
                {"candidate": name, "quantity": quantity, **asdict(aggregate)}
            )

        decisions[name] = decide_candidate(verdicts, candidate_aggregates).value

    return aggregates, decisions


def assemble_payload(
    study: CertificationStudy,
    sampling: SamplingPolicy,
    quick: bool,
    references: Dict[str, dict],
    cells: List[dict],
    aggregates: List[dict],
    decisions: Dict[str, str],
    started: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build the certificate payload and stamp its digest."""
    payload = {
        "schema": SCHEMA_VERSION,
        "study": {
            "name": study.name,
            "source_text": study.source_text,
            "quantities": list(study.quantities),
            "bounds": asdict(study.bounds),
            "sampling": asdict(sampling),
            "quick": quick,
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
        "runtime": runtime_environment(),
        "references": references,
        "cells": cells,
        "aggregates": aggregates,
        "decisions": decisions,
        "wall_clock_seconds": time.time() - started,
    }
    if extra:
        payload.update(extra)
    payload["projected_sha256"] = projected_sha256(payload)
    return payload


def write_certificate(payload: dict, root: Path) -> Certificate:
    """Validate, then write the certificate and both reports.

    Three artifacts, one act: the machine record, the markdown report for
    terminals and diffs, and the self-contained HTML report for review and
    circulation. They are written together so a banked directory can never
    contain a certificate whose reports describe something else.
    """
    validate_payload(payload)
    path = root / CERTIFICATE_NAME
    atomic_write_json(path, payload)
    atomic_write_text(root / REPORT_NAME, render_markdown(payload))
    atomic_write_text(root / HTML_REPORT_NAME, render_html(payload))
    return Certificate(payload=payload, path=path)


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Check a certificate's structure, enums, and digest.

    Run before writing and after loading, so a tampered or truncated
    certificate is caught at the boundary rather than believed.

    Raises:
        ValidationError: wrong schema, unknown enum value, dangling reference,
            or a digest that does not match the content.
    """
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValidationError(
            f"Certificate schema must be {SCHEMA_VERSION}, got {payload.get('schema')}"
        )

    for key in ("study", "runtime", "references", "cells", "aggregates", "decisions"):
        if key not in payload:
            raise ValidationError(f"Certificate is missing required key {key!r}")

    known_cases = {case["name"] for case in payload["study"]["cases"]}
    known_candidates = {c["name"] for c in payload["study"]["candidates"]}
    known_quantities = set(payload["study"]["quantities"])
    valid_verdicts = {v.value for v in Verdict}
    valid_decisions = {d.value for d in Decision}

    for cell in payload["cells"]:
        if cell["case"] not in known_cases:
            raise ValidationError(f"Cell references unknown case {cell['case']!r}")
        if cell["candidate"] not in known_candidates:
            raise ValidationError(
                f"Cell references unknown candidate {cell['candidate']!r}"
            )
        if cell["quantity"] not in known_quantities:
            raise ValidationError(
                f"Cell references unknown quantity {cell['quantity']!r}"
            )
        if cell["verdict"] not in valid_verdicts:
            raise ValidationError(f"Cell has unknown verdict {cell['verdict']!r}")

    for name, decision in payload["decisions"].items():
        if name not in known_candidates:
            raise ValidationError(f"Decision names unknown candidate {name!r}")
        if decision not in valid_decisions:
            raise ValidationError(f"Unknown decision {decision!r} for candidate {name!r}")

    stored = payload.get("projected_sha256")
    if not stored:
        raise ValidationError("Certificate is missing its projected_sha256")
    recomputed = projected_sha256(payload)
    if stored != recomputed:
        raise ValidationError(
            "Certificate digest does not match its content: stored "
            f"{stored}, recomputed {recomputed}"
        )
