"""Amendments: re-certify what changed, carry forward what did not.

Re-running an hours-scale certification because one engine's grid changed wastes
most of the run. An amendment re-runs only the affected work and carries the
rest forward by reference to the parent's digest, so the result is a hash chain
rather than a fresh, unrelated claim.

Three rules keep that chain honest:

* the parent is validated (schema, structure, digest) before any pricing;
* a cell is carried forward only when *both* its candidate identity and its
  benchmark identity still match -- a changed benchmark moves the comparison
  target, so even an untouched engine must be re-gated against it; and
* scope may grow but never silently shrink. Dropping a case or a candidate is a
  new certification, not an amendment, because a shrunken amendment would read
  as though the missing coverage had passed.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import CandidateResult, candidate_identity
from quantark.modelvalidation.evidence import (
    CheckpointStore,
    identity_hash,
    read_json,
    validate_durable_root,
)
from quantark.modelvalidation.pipeline import (
    Certificate,
    evaluate_candidate,
    reference_block,
    aggregate_and_decide,
    assemble_payload,
    build_cells,
    quick_policy,
    validate_payload,
    write_certificate,
)
from quantark.modelvalidation.reference import ReferenceEstimate, run_reference
from quantark.modelvalidation.study import CertificationStudy


def validate_parent(parent_path: str | Path) -> dict:
    """Load and fully validate a parent certificate.

    Raises:
        ValidationError: unreadable, malformed, or digest-mismatched.
    """
    path = Path(parent_path)
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Cannot read parent certificate {path}: {exc}") from exc

    validate_payload(payload)
    return payload


def _check_coverage(study: CertificationStudy, parent: Mapping[str, Any]) -> None:
    """An amendment may add coverage; it may never quietly remove any."""
    study_cases = {case.name for case in study.cases}
    parent_cases = {case["name"] for case in parent["study"]["cases"]}
    missing_cases = sorted(parent_cases - study_cases)
    if missing_cases:
        raise ValidationError(
            f"Amendment drops cases certified by the parent: {missing_cases}. "
            "Narrowing scope requires a new certification, not an amendment."
        )

    study_candidates = {candidate.name() for candidate in study.candidates}
    parent_candidates = {c["name"] for c in parent["study"]["candidates"]}
    missing_candidates = sorted(parent_candidates - study_candidates)
    if missing_candidates:
        raise ValidationError(
            f"Amendment drops candidates certified by the parent: {missing_candidates}. "
            "Narrowing scope requires a new certification, not an amendment."
        )


def amend(
    study: CertificationStudy,
    parent: str | Path,
    out_dir: str | Path,
    reason: str,
    quick: bool = False,
    resume: bool = False,
) -> Certificate:
    """Re-certify the parts of ``study`` that changed since ``parent``.

    Args:
        study: The current study.
        parent: Path to the parent certificate.
        out_dir: Durable output root for the amended certificate.
        reason: Why this amendment exists; recorded in the payload.
        quick: Shrink sampling (wiring check only).
        resume: Reuse checkpoints for the work that does re-run.

    Raises:
        ValidationError: no reason, an invalid parent, or an amendment that
            would drop parent coverage.
    """
    if not reason or not reason.strip():
        raise ValidationError("An amendment requires a reason; it is part of the record")

    started = time.time()
    parent_payload = validate_parent(parent)
    _check_coverage(study, parent_payload)

    root = validate_durable_root(out_dir) / study.name
    store = CheckpointStore(root / "checkpoints")
    sampling = quick_policy(study.sampling) if quick else study.sampling
    parent_digest = parent_payload["projected_sha256"]

    parent_references: Dict[str, dict] = dict(parent_payload["references"])
    parent_cells: Dict[tuple, dict] = {
        (cell["candidate"], cell["case"], cell["quantity"]): cell
        for cell in parent_payload["cells"]
    }

    references: Dict[str, dict] = {}
    estimates: Dict[str, Optional[ReferenceEstimate]] = {}
    reference_errors: Dict[str, str] = {}
    reference_carried: Dict[str, bool] = {}

    for case in study.cases:
        current_identity = identity_hash(study.reference.identity(case))
        banked = parent_references.get(case.name)
        if banked is not None and banked.get("identity_hash") == current_identity:
            references[case.name] = dict(banked)
            estimates[case.name] = None
            reference_carried[case.name] = True
            continue

        reference_carried[case.name] = False
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
    replaced: List[dict] = []
    carried: List[dict] = []

    for candidate in study.candidates:
        name = candidate.name()
        for case in study.cases:
            current_identity = identity_hash(candidate_identity(candidate, case))
            keys = [(name, case.name, quantity) for quantity in study.quantities]
            banked_cells = [parent_cells.get(key) for key in keys]

            can_carry = (
                reference_carried[case.name]
                and all(cell is not None for cell in banked_cells)
                and all(cell["identity_hash"] == current_identity for cell in banked_cells)
            )

            if can_carry:
                for cell in banked_cells:
                    carried_cell = dict(cell)
                    carried_cell["carried_from"] = parent_digest
                    cells.append(carried_cell)
                    carried.append(
                        {
                            "candidate": name,
                            "case": case.name,
                            "quantity": carried_cell["quantity"],
                        }
                    )
                continue

            estimate = estimates[case.name]
            result: Optional[CandidateResult] = None
            error: Optional[str] = None

            if estimate is None and reference_carried[case.name]:
                # The benchmark was carried forward but this candidate changed:
                # re-gating needs the parent's reference numbers.
                estimate = _estimate_from_block(references[case.name], study.quantities)

            if estimate is None:
                error = reference_errors.get(
                    case.name, "reference unavailable for this case"
                )
            else:
                try:
                    result = evaluate_candidate(candidate, case, store, resume)
                except Exception:  # noqa: BLE001 - recorded, not swallowed
                    error = traceback.format_exc()

            fresh = build_cells(
                study=study,
                candidate=candidate,
                case=case,
                estimate=estimate,
                result=result,
                error=error,
            )
            cells.extend(fresh)
            replaced.extend(
                {"candidate": name, "case": case.name, "quantity": cell["quantity"]}
                for cell in fresh
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
        extra={
            "amendment": {
                "parent": str(Path(parent)),
                "parent_projected_sha256": parent_digest,
                "reason": reason,
                "replaced_cells": replaced,
                "carried_cells": carried,
            }
        },
    )
    return write_certificate(payload, root)


def _estimate_from_block(block: Mapping[str, Any], quantities) -> Optional[ReferenceEstimate]:
    """Rebuild a reference estimate from a carried-forward reference block."""
    if "error" in block:
        return None
    return ReferenceEstimate(
        values={q: block["values"][q] for q in quantities},
        std_errors={q: block["std_errors"][q] for q in quantities},
        batches=block["batches"],
        seeds=tuple(block["seeds"]),
        stopped_reason=block["stopped_reason"],
    )
