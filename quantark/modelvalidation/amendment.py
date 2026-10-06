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

Schema 2 adds a fourth: the parent's *contract* (estimands, budgets, gate
policy, scale, reference targets, seed scheme, convergence rule) must be
identical. Case context and semantic expectations are not in the contract; they
live in the cell identity, so changing them re-evaluates exactly those cells.
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
    _reference_config,
    evaluate_candidate,
    reference_block,
    aggregate_and_decide,
    assemble_payload,
    build_cells,
    qualify_case,
    quick_policy,
    stop_quantities,
    study_contract,
    validate_payload,
    write_certificate,
)
from quantark.modelvalidation.reference import ReferenceEstimate, bound_reference, run_reference
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


def _check_contract(study: CertificationStudy, reference, parent: Mapping[str, Any], qualifier=None) -> None:
    """A schema-2 amendment keeps its parent's contract; anything else is a new certification."""
    if parent.get("schema") != study.schema:
        raise ValidationError(
            f"Amendment schema {study.schema} does not match the parent's {parent.get('schema')}; "
            "the first certification under a new schema is a full run")
    if study.schema != 2:
        return
    current, banked = study_contract(study, reference, qualifier), parent.get("contract") or {}
    changed = sorted(key for key in set(current) | set(banked) if current.get(key) != banked.get(key))
    if changed:
        raise ValidationError(
            f"Amendment changes the certification contract ({', '.join(changed)}). Estimands, budgets, gate policy, "
            "scale, reference targets, seed scheme, the convergence rule and a deterministic reference's error model "
            "and qualification policy are fixed by the parent; changing one is a new certification, not an amendment")


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
    sampling = quick_policy(study.sampling) if quick else study.sampling
    reference = bound_reference(study.reference, sampling, schema=study.schema, quick=quick)
    qualifier = (None if study.qualification is None
                 else bound_reference(study.qualification.builder, sampling, schema=study.schema))
    # The schema and contract are the more fundamental refusal: a study offered against a parent of
    # another schema must be told so, not told it dropped a case.
    _check_contract(study, reference, parent_payload, qualifier)
    _check_coverage(study, parent_payload)
    stopping = stop_quantities(study, reference)

    root = validate_durable_root(out_dir) / study.name
    store = CheckpointStore(root / "checkpoints")
    parent_digest = parent_payload["projected_sha256"]

    if _reference_config(reference).get("external"):
        return _amend_external(study, parent, parent_payload, reference, sampling, quick, resume, reason,
                               started, root, store)

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
        current_identity = identity_hash(reference.identity(case))
        banked = parent_references.get(case.name)
        if banked is not None and banked.get("identity_hash") == current_identity:
            references[case.name] = dict(banked)
            estimates[case.name] = None
            reference_carried[case.name] = True
            continue

        reference_carried[case.name] = False
        try:
            estimate = run_reference(
                builder=reference,
                case=case,
                quantities=study.quantities,
                scale=study.scale,
                bounds=study.bounds,
                policy=sampling,
                store=store,
                resume=resume,
                quantity_bounds=study.quantity_bounds if study.schema == 2 else None,
                stop_quantities=stopping,
            )
            estimates[case.name] = estimate
            references[case.name] = reference_block(
                estimate, reference.identity(case)
            )
        except Exception:  # noqa: BLE001 - recorded, not swallowed
            estimates[case.name] = None
            reference_errors[case.name] = traceback.format_exc()
            references[case.name] = {"error": reference_errors[case.name]}

    # The qualifying arm is carried with its reference, or sampled again; cells judged under a
    # qualification that did not carry are re-gated, never carried.
    qualification: Dict[str, dict] = {}
    qualification_carried: Dict[str, bool] = {}
    parent_qualification: Dict[str, dict] = dict(parent_payload.get("qualification", {}))
    for case in study.cases:
        qualification_carried[case.name] = True
        if qualifier is None:
            continue
        banked = parent_qualification.get(case.name)
        if (reference_carried[case.name] and banked is not None
                and banked.get("identity_hash") == identity_hash(qualifier.identity(case))):
            qualification[case.name] = dict(banked)
            continue
        qualification_carried[case.name] = False
        estimate = estimates[case.name]
        if estimate is None and reference_carried[case.name]:
            estimate = _estimate_from_block(references[case.name], study.quantities)
        if estimate is not None:
            qualification[case.name] = qualify_case(study, qualifier, reference, case, estimate, sampling, store, resume)

    cells: List[dict] = []
    replaced: List[dict] = []
    carried: List[dict] = []

    for candidate in study.candidates:
        name = candidate.name()
        for case in study.cases:
            current_identity = identity_hash(candidate_identity(candidate, case, schema=study.schema))
            keys = [(name, case.name, quantity) for quantity in study.quantities]
            banked_cells = [parent_cells.get(key) for key in keys]

            can_carry = (
                reference_carried[case.name]
                and qualification_carried[case.name]
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
                    result = evaluate_candidate(candidate, case, store, resume, schema=study.schema)
                except Exception:  # noqa: BLE001 - recorded, not swallowed
                    error = traceback.format_exc()

            fresh = build_cells(
                study=study,
                candidate=candidate,
                case=case,
                estimate=estimate,
                result=result,
                error=error,
                reference=reference,
                qualification=qualification.get(case.name),
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
        reference=reference,
        qualifier=qualifier,
        extra={
            **({"qualification": qualification} if qualifier is not None else {}),
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


def _amend_external(study, parent, parent_payload, reference, sampling, quick, resume, reason, started, root,
                    store) -> Certificate:
    """Amend an imported certification, whose benchmark is external: banked, never run, never re-gated.

    Every candidate re-runs, because an imported cell's identity is the producing harness's, never this
    module's. A value that comes back bit for bit carries its cell verbatim. A value that moved is
    TRANSLATED by the harness's own arithmetic (RELEASE_PROCEDURE section 10): its comparison interval
    shifts by the move in contracts, ``k = reference_standard_error / reference.se``, with the harness's
    envelopes held at their banked values. A translation that would change a verdict -- a cell's, or an
    aggregate's widened by its largest move -- is refused: only the harness can re-certify that.
    """
    banked_config, current_config = parent_payload.get("reference_config") or {}, _reference_config(reference)
    if "imported" not in parent_payload or banked_config != current_config:
        raise ValidationError(
            "An external benchmark is amended only against the imported certificate that banks it, under the "
            f"benchmark it declared: the parent declares {banked_config!r}, the study {current_config!r}")

    parent_digest = parent_payload["projected_sha256"]
    parent_cells = {(c["candidate"], c["case"], c["quantity"]): c for c in parent_payload["cells"]}
    cells: List[dict] = []
    carried: List[dict] = []
    translated: List[dict] = []
    moves: Dict[tuple, float] = {}

    for candidate in study.candidates:
        name = candidate.name()
        for case in study.cases:
            banked = [parent_cells.get((name, case.name, quantity)) for quantity in study.quantities]
            if any(cell is None for cell in banked):
                raise ValidationError(
                    f"{name} on {case.name!r} has no banked cell: coverage added under an external benchmark has "
                    "no reference to be judged against; re-run the harness to certify it")
            result = evaluate_candidate(candidate, case, store, resume, schema=study.schema)
            for quantity, cell in zip(study.quantities, banked):
                key = {"candidate": name, "case": case.name, "quantity": quantity}
                value = result.values[quantity]
                if value == cell["candidate_value"]:
                    cells.append({**cell, "carried_from": parent_digest})
                    carried.append(key)
                    continue
                moved, shift_c = _translate_imported_cell(cell, value)
                cells.append({**moved, "translated_from": parent_digest})
                translated.append({**key, "parent_candidate_value": cell["candidate_value"],
                                   "candidate_value": value, "shift_c": shift_c})
                moves[(name, quantity)] = max(moves.get((name, quantity), 0.0), abs(shift_c))

    bounded: List[dict] = []
    for aggregate in parent_payload["aggregates"]:
        largest = moves.get((aggregate["candidate"], aggregate["quantity"]))
        if largest is None:
            continue
        source = aggregate.get("source_bias") or {}
        lo, hi = source.get("interval") or (float("nan"), float("nan"))
        if source.get("status") != "PASS" or not max(abs(lo), abs(hi)) + largest < source.get("economic_bound", 0.0):
            raise ValidationError(
                f"The {aggregate['candidate']} {aggregate['quantity']} aggregate cannot be translated: its banked "
                f"interval [{lo}, {hi}] widened by the largest candidate move {largest!r} contracts does not stay "
                "wholly inside its economic bound; re-run the harness to re-certify")
        bounded.append({"candidate": aggregate["candidate"], "quantity": aggregate["quantity"],
                        "shift_bound_c": largest})

    payload = assemble_payload(
        study=study,
        sampling=sampling,
        quick=quick,
        references=dict(parent_payload["references"]),
        cells=cells,
        aggregates=[dict(aggregate) for aggregate in parent_payload["aggregates"]],
        decisions=dict(parent_payload["decisions"]),
        started=started,
        reference=reference,
        extra={
            "imported": parent_payload["imported"],
            "amendment": {
                "parent": str(Path(parent)),
                "parent_projected_sha256": parent_digest,
                "reason": reason,
                "external_benchmark": (
                    "banked by the parent and carried; never run. Every candidate re-ran: a bit-identical value "
                    "carries its cell, a moved one is translated by the harness's interval arithmetic with its "
                    "envelopes held at their banked values, and an aggregate keeps its banked numbers within "
                    "shift_bound_c of its translated centre"),
                "replaced_cells": [{k: t[k] for k in ("candidate", "case", "quantity")} for t in translated],
                "carried_cells": carried,
                "translated_cells": translated,
                "aggregate_shift_bounds": bounded,
            },
        },
    )
    return write_certificate(payload, root)


def _translate_imported_cell(cell: Mapping[str, Any], value: float) -> tuple:
    """Shift an imported cell's harness interval by its candidate's move; refuse if its verdict would change."""
    verdict, reference = cell.get("source_verdict") or {}, cell.get("reference") or {}
    if verdict.get("status") != "PASS" or not reference.get("se"):
        raise ValidationError(
            f"{cell['candidate']}/{cell['case']}/{cell['quantity']} moved, and only a PASS cell with a banked "
            "reference standard error can be translated; re-run the harness to re-certify it")
    shift_c = (value - cell["candidate_value"]) * (verdict["reference_standard_error"] / reference["se"])
    lo, hi = (edge + shift_c for edge in verdict["interval"])
    bound = verdict["economic_bound"]
    if not -bound < lo <= hi < bound:
        raise ValidationError(
            f"{cell['candidate']}/{cell['case']}/{cell['quantity']} moved by {shift_c!r} contracts, which takes "
            f"its harness interval to [{lo}, {hi}] against a bound of {bound}: the verdict would change, and "
            "only the harness can re-certify it")
    difference = verdict["estimate_difference"] + shift_c
    moved = {
        **cell,
        "candidate_value": value,
        "gate": {**cell["gate"], "signed_err_c": difference, "interval_c": max(abs(lo), abs(hi))},
        "source_verdict": {**verdict, "estimate_difference": difference, "interval": [lo, hi]},
    }
    return moved, shift_c


def _estimate_from_block(block: Mapping[str, Any], quantities) -> Optional[ReferenceEstimate]:
    """Rebuild a reference estimate from a carried-forward reference block."""
    if "error" in block:
        return None
    if block.get("kind") == "deterministic":
        return ReferenceEstimate(values=dict(block["values"]), std_errors={}, batches=0, seeds=(),
                                 stopped_reason="deterministic", kind="deterministic", radii=dict(block["radii"]),
                                 evidence=dict(block["evidence"]), undefined=dict(block["undefined"]),
                                 radius_basis=dict(block["radius_basis"]))
    return ReferenceEstimate(
        values={q: block["values"][q] for q in quantities},
        std_errors={q: block["std_errors"][q] for q in quantities},
        batches=block["batches"],
        seeds=tuple(block["seeds"]),
        stopped_reason=block["stopped_reason"],
    )
