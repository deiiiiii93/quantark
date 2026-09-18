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

import math
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import (
    MIN_CONVERGENCE_LEVELS,
    CandidateResult,
    candidate_identity,
    convergence_evidence,
    deserialize_candidate_result,
    envelope_from_ladders,
    serialize_candidate_result,
)
from quantark.modelvalidation.candidate import CHECKPOINT_KIND as CANDIDATE_KIND
from quantark.modelvalidation.decisions import Decision, Verdict, decide_candidate, decide_cell
from quantark.modelvalidation.evidence import (
    SUPPORTED_SCHEMA_VERSIONS,
    CheckpointStore,
    atomic_write_json,
    atomic_write_text,
    identity_hash,
    projected_sha256,
    validate_durable_root,
)
from quantark.modelvalidation.gates import (
    aggregate_gate_wire,
    cell_gate_wire,
    evaluate_aggregate_gate,
    evaluate_cell_gate,
)
from quantark.modelvalidation.html_report import render_html
from quantark.modelvalidation.reference import (
    ReferenceEstimate,
    bound_reference,
    reference_targets,
    run_reference,
)
from quantark.modelvalidation.report import render_markdown
from quantark.modelvalidation.study import QUANTITY_CATALOGUE, CertificationStudy, SamplingPolicy, reference_kind

CERTIFICATE_NAME = "certificate.json"
#: Checkpoint directory of the qualifying arm, apart from the reference's own bank.
QUALIFIER_KIND = "qualifier"
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
        seed_scheme=policy.seed_scheme,
    )


def _reference_config(reference) -> dict:
    """The benchmark's own configuration, when it declares one.

    Optional on the protocol: a reference builder that does not describe itself
    records nothing rather than having something invented on its behalf.
    """
    describe = getattr(reference, "config", None)
    if not callable(describe):
        return {}
    return dict(describe())


def stop_quantities(study: CertificationStudy, reference) -> Optional[List[str]]:
    """Schema 2: the quantities whose standard error drives sampling -- the ones the reference targets.

    An untargeted proxy is recorded, never gated, so it must never keep a reference sampling.
    ``None`` keeps schema 1's rule, where every quantity counts.
    """
    if study.schema != 2:
        return None
    targets = reference_targets(reference, study.quantities)
    return [q for q in study.quantities if targets[q] is not None]


def reference_block(estimate: ReferenceEstimate, identity: Mapping[str, Any]) -> dict:
    if estimate.kind == "deterministic":
        # typed: a deterministic solve has no batches, seeds or standard error to report (spec section 5)
        return {
            "kind": "deterministic",
            "values": dict(estimate.values),
            "radii": dict(estimate.radii),
            "undefined": dict(estimate.undefined),
            "evidence": dict(estimate.evidence),
            "identity_hash": identity_hash(identity),
        }
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
    schema: int = 1,
) -> CandidateResult:
    """Evaluate one candidate for one case, reusing a matching checkpoint."""
    identity = candidate_identity(candidate, case, schema=schema)
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
    # The builder that samples is the one bound to the EFFECTIVE policy, so the path count the
    # payload records is the one that ran (schema-1 builders pass through unchanged).
    reference = bound_reference(study.reference, sampling, schema=study.schema, quick=quick)
    stopping = stop_quantities(study, reference)
    qualifier = (None if study.qualification is None
                 else bound_reference(study.qualification.builder, sampling, schema=study.schema))

    references: Dict[str, dict] = {}
    estimates: Dict[str, Optional[ReferenceEstimate]] = {}
    reference_errors: Dict[str, str] = {}

    for case in study.cases:
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

    qualification = qualify_references(study, qualifier, reference, estimates, sampling, store, resume)
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
                    result = evaluate_candidate(candidate, case, store, resume, schema=study.schema)
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
                    reference=reference,
                    qualification=qualification.get(case.name),
                )
            )

    aggregates, decisions = aggregate_and_decide(study, cells)
    extra = {"qualification": qualification} if qualifier is not None else None
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
        extra=extra,
        qualifier=qualifier,
    )
    return write_certificate(payload, root)


#: Two exact computations of one number agree to floating point, not to zero.
_QUALIFICATION_FLOAT_TOLERANCE = 1e-12


def qualification_targets(study, qualifier, reference) -> List[str]:
    """The quantities both arms target: the only ones a qualifier can check."""
    theirs = reference_targets(qualifier, study.quantities)
    ours = reference_targets(reference, study.quantities)
    return [q for q in study.quantities if theirs[q] is not None and ours[q] is not None]


def qualify_case(study, qualifier, reference, case, estimate, sampling, store, resume) -> dict:
    """Sample the qualifying arm on one case and compare it with the deterministic reference.

    The check covers the quantities both arms target and the reference defines. The allowance is
    ``max_z`` qualifier standard errors plus the reference's radius (plus floating point for two
    exact values). A case that fails is not qualified: nobody knows which arm is wrong, so its
    cells are unresolved whatever the candidate says.
    """
    max_z = study.qualification.max_z
    shared = qualification_targets(study, qualifier, reference)
    try:
        sampled = run_reference(
            builder=qualifier, case=case, quantities=study.quantities, scale=study.scale, bounds=study.bounds,
            policy=sampling, store=store, resume=resume, quantity_bounds=study.quantity_bounds,
            stop_quantities=shared, checkpoint_kind=QUALIFIER_KIND,
        )
    except Exception:  # noqa: BLE001 - recorded, not swallowed
        return {"qualified": False, "error": traceback.format_exc(), "checks": {},
                "identity_hash": identity_hash(qualifier.identity(case))}
    checks = {}
    for quantity in shared:
        if quantity in estimate.undefined or quantity in case.expected:
            continue                                    # no number to compare on a semantic cell
        difference = estimate.values[quantity] - sampled.values[quantity]
        se, radius = sampled.std_errors[quantity], estimate.radii[quantity]
        allowed = max_z * se + radius + _QUALIFICATION_FLOAT_TOLERANCE * max(1.0, abs(estimate.values[quantity]))
        checks[quantity] = {"reference": estimate.values[quantity], "qualifier": sampled.values[quantity],
                            "qualifier_se": se, "difference": difference, "z": difference / se if se > 0.0 else None,
                            "allowed": allowed, "within": abs(difference) <= allowed}
    block = reference_block(sampled, qualifier.identity(case))
    return {**block, "checks": checks, "qualified": all(c["within"] for c in checks.values())}


def qualify_references(study, qualifier, reference, estimates, sampling, store, resume) -> Dict[str, dict]:
    """The qualifying arm on every case whose reference solved (a failed reference already errors its cells)."""
    if qualifier is None:
        return {}
    return {case.name: qualify_case(study, qualifier, reference, case, estimates[case.name], sampling, store, resume)
            for case in study.cases if estimates[case.name] is not None}


def build_cells(
    study: CertificationStudy,
    candidate,
    case,
    estimate: Optional[ReferenceEstimate],
    result: Optional[CandidateResult],
    error: Optional[str],
    reference=None,
    qualification: Optional[Mapping[str, Any]] = None,
) -> List[dict]:
    """Gate one candidate against the benchmark for one case, per quantity.

    ``reference`` is the builder that sampled (bound to the effective policy); schema 2 reads its
    declared targets. Schema-1 cells keep exactly their original keys.
    """
    cells: List[dict] = []
    name = candidate.name()
    cell_identity = identity_hash(candidate_identity(candidate, case, schema=study.schema))
    targets = reference_targets(reference or study.reference, study.quantities) if study.schema == 2 else {}

    for quantity in study.quantities:
        base = {
            "candidate": name,
            "case": case.name,
            "quantity": quantity,
            "reference": None,
            "candidate_value": None,
            "gate": None,
            "error": None,
            "identity_hash": cell_identity,
        }
        if study.schema == 2:
            base.update(kind="numeric", bound_c=None, reason=None, convergence=None)
        if error is not None or result is None or estimate is None:
            cells.append({**base, "verdict": decide_cell(None, error=True).value, "error": error})
            continue
        if study.schema == 2:
            cells.append(_schema2_cell(study, base, case, quantity, estimate, result, targets[quantity], qualification))
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
                **base,
                "reference": {
                    "value": estimate.values[quantity],
                    "se": estimate.std_errors[quantity],
                },
                "candidate_value": result.values[quantity],
                "gate": cell_gate_wire(gate, 1),
                "verdict": decide_cell(gate, error=False).value,
            }
        )
    return cells


def _reference_record(estimate: ReferenceEstimate, quantity: str) -> dict:
    """A cell's reference: value with its standard error, or a typed deterministic value with its radius."""
    if estimate.kind == "deterministic":
        return {"kind": "deterministic", "value": estimate.values.get(quantity), "radius": estimate.radii.get(quantity)}
    return {"value": estimate.values[quantity], "se": estimate.std_errors[quantity]}


def _schema2_cell(study, base, case, quantity, estimate, result, target, qualification=None) -> dict:
    """One schema-2 cell: a semantic assertion, an untargeted proxy, or a gated number."""
    reference = _reference_record(estimate, quantity)
    deterministic = estimate.kind == "deterministic"
    status = result.status(quantity)
    expected = case.expected.get(quantity)
    if expected is not None:
        held = status == expected
        reason = f"expected {expected}, candidate reported {status}"
        if not held and result.reasons.get(quantity):
            reason += f": {result.reasons[quantity]}"
        return {**base, "kind": "semantic", "reference": reference,
                "verdict": (Verdict.PASS if held else Verdict.FAIL).value, "reason": reason}
    if status != "ok":
        return {**base, "reference": reference, "verdict": Verdict.ERROR.value,
                "error": f"unexpected {status} in a numeric cell: {result.reasons.get(quantity, '')}"}
    value = result.values[quantity]
    if deterministic and quantity in estimate.undefined:
        return {**base, "reference": reference, "candidate_value": value, "verdict": Verdict.ERROR.value,
                "error": f"the reference is undefined here ({estimate.undefined[quantity]}) and the case declares no "
                         f"expected status for {quantity}"}
    if target is None:
        return {**base, "kind": "untargeted", "reference": reference, "candidate_value": value,
                "verdict": Verdict.UNRESOLVED.value,
                "reason": (f"the reference declares no estimator for {quantity}; its recorded value is a "
                           "finite-bump proxy whose bias is unassessed, and the quantity is uncertified")}
    evidence = convergence_evidence(result, quantity)
    bound_c = study.quantity_bounds[quantity].budget(study.scale.to_economic(quantity, estimate.values[quantity]))
    if deterministic and not math.isfinite(estimate.radii[quantity]):
        return {**base, "reference": reference, "candidate_value": value, "bound_c": bound_c,
                "verdict": Verdict.UNRESOLVED.value,
                "reason": f"the reference's refinement ladder could not bound its error in {quantity}"}
    gate = evaluate_cell_gate(
        candidate_raw=value, reference_raw=estimate.values[quantity],
        reference_se_raw=None if deterministic else estimate.std_errors[quantity],
        reference_radius_raw=estimate.radii[quantity] if deterministic else None,
        quantity=quantity, scale=study.scale, bounds=study.bounds, envelope_raw=evidence.envelope, bound_c=bound_c,
    )
    cell = {**base, "reference": reference, "candidate_value": value, "gate": cell_gate_wire(gate, 2), "bound_c": bound_c,
            "convergence": {"complete": evidence.complete, "missing": list(evidence.missing),
                            "envelope_raw": evidence.envelope, "observed_orders": dict(evidence.observed_orders),
                            "non_monotone": list(evidence.non_monotone)}}
    if not evidence.complete:
        # Missing required convergence evidence is unresolved, never implicitly acceptable (spec 6.2).
        return {**cell, "verdict": Verdict.UNRESOLVED.value,
                "reason": f"convergence evidence has fewer than {MIN_CONVERGENCE_LEVELS} levels on: "
                          f"{', '.join(evidence.missing)}"}
    if qualification is not None and not qualification["qualified"]:
        failed = sorted(q for q, check in qualification["checks"].items() if not check["within"])
        why = "its qualifying arm errored" if "error" in qualification else f"it disagrees with its qualifying arm on {failed}"
        return {**cell, "verdict": Verdict.UNRESOLVED.value,
                "reason": f"the reference is not qualified on this case: {why}"}
    return {**cell, "verdict": decide_cell(gate, error=False, schema=2).value}


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
        # An untargeted cell is reported, never decided on: it is outside the certified scope (spec 7.2).
        verdicts = [Verdict(cell["verdict"]) for cell in own if cell.get("kind", "numeric") != "untargeted"]

        candidate_aggregates = []
        for quantity in study.quantities:
            gated = [
                cell
                for cell in own
                if cell["quantity"] == quantity
                and cell["gate"] is not None
                # an ungated verdict (incomplete convergence evidence) never feeds the mean
                and not (cell.get("convergence") is not None and not cell["convergence"]["complete"])
            ]
            if not gated:
                continue
            if any(cell["gate"].get("radius_c") is not None for cell in gated):
                aggregate = evaluate_aggregate_gate(
                    [cell["gate"]["signed_err_c"] for cell in gated], None, study.bounds,
                    radii_c=[cell["gate"]["radius_c"] for cell in gated],
                )
            else:
                aggregate = evaluate_aggregate_gate(
                    [cell["gate"]["signed_err_c"] for cell in gated],
                    [cell["gate"]["se_c"] for cell in gated],
                    study.bounds,
                )
            candidate_aggregates.append(aggregate)
            aggregates.append(
                {"candidate": name, "quantity": quantity, **aggregate_gate_wire(aggregate, study.schema)}
            )

        decisions[name] = decide_candidate(verdicts, candidate_aggregates, schema=study.schema).value

    return aggregates, decisions


def bounds_wire(bounds, schema: int) -> dict:
    """The serialized gate bounds. Schema 1 keeps exactly its original keys; a stochastic schema-2 study has no radius."""
    data = asdict(bounds)
    if schema == 1 or data["radius_budget_fraction"] is None:
        data.pop("radius_budget_fraction")
    return data


def sampling_wire(policy: SamplingPolicy, schema: int) -> dict:
    """The serialized sampling policy. Schema 1 keeps exactly its original keys."""
    data = asdict(policy)
    if schema == 1:
        data.pop("seed_scheme")            # not part of the schema-1 format
    return data


def study_contract(study: CertificationStudy, reference, qualifier=None) -> dict:
    """What an amendment may not change: estimands, budgets, gate policy, scale, targets, seeds, convergence rule,
    and -- under a deterministic reference -- its kind, error model and qualification policy."""
    contract = _base_contract(study, reference)
    if reference_kind(reference) == "deterministic":
        contract["reference_kind"] = "deterministic"
        contract["reference_error_model"] = dict(reference.error_model())
        if study.qualification is not None:
            arm = study.qualification.builder if qualifier is None else qualifier
            contract["qualification"] = {"max_z": study.qualification.max_z,
                                         "targets": reference_targets(arm, study.quantities)}
    return contract


def _base_contract(study: CertificationStudy, reference) -> dict:
    return {
        "quantities": list(study.quantities),
        "quantity_definitions": {q: asdict(QUANTITY_CATALOGUE[q]) for q in study.quantities},
        "quantity_bounds": {q: asdict(b) for q, b in study.quantity_bounds.items()},
        "gate_policy": bounds_wire(study.bounds, study.schema),
        "scale": asdict(study.scale) if is_dataclass(study.scale) else repr(study.scale),
        "reference_targets": reference_targets(reference, study.quantities),
        "seed_scheme": study.sampling.seed_scheme,
        "min_convergence_levels": MIN_CONVERGENCE_LEVELS,
    }


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
    reference=None,
    qualifier=None,
) -> dict:
    """Build the certificate payload and stamp its digest.

    ``reference`` is the builder that sampled (bound to the effective policy); it
    defaults to the study's own for callers that predate binding.
    """
    reference = study.reference if reference is None else reference
    payload = {
        "schema": study.schema,
        "study": {
            "name": study.name,
            "source_text": study.source_text,
            "quantities": list(study.quantities),
            "bounds": bounds_wire(study.bounds, study.schema),
            "sampling": sampling_wire(sampling, study.schema),
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
        "reference_config": _reference_config(reference),
        "references": references,
        "cells": cells,
        "aggregates": aggregates,
        "decisions": decisions,
        "wall_clock_seconds": time.time() - started,
    }
    if study.schema == 2:
        contract = study_contract(study, reference, qualifier)
        payload["study"]["quantity_bounds"] = contract["quantity_bounds"]
        payload["study"]["quantity_definitions"] = contract["quantity_definitions"]
        payload["study"]["scale"] = contract["scale"]
        payload["study"]["uncertified_quantities"] = [
            q for q, target in contract["reference_targets"].items() if target is None
        ]
        for case_block, case in zip(payload["study"]["cases"], study.cases):
            case_block["context_params"] = dict(case.context_params)
            case_block["expected"] = dict(case.expected)
        payload["reference_targets"] = contract["reference_targets"]
        payload["contract"] = contract
        payload["contract_sha256"] = identity_hash(contract)
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
    if payload.get("schema") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValidationError(
            f"Certificate schema must be one of {SUPPORTED_SCHEMA_VERSIONS}, got {payload.get('schema')}"
        )

    for key in ("study", "runtime", "references", "cells", "aggregates", "decisions"):
        if key not in payload:
            raise ValidationError(f"Certificate is missing required key {key!r}")

    if payload["schema"] == 2:
        contract = payload.get("contract")
        if not isinstance(contract, Mapping):
            raise ValidationError("A schema-2 certificate must carry its contract block")
        if payload.get("contract_sha256") != identity_hash(contract):
            raise ValidationError("The schema-2 contract digest does not match its contract block")
        stray = sorted(set(payload["study"].get("uncertified_quantities", [])) - set(payload["study"]["quantities"]))
        if stray:
            raise ValidationError(f"uncertified_quantities names quantities the study does not certify: {stray}")
        for cell in payload["cells"]:
            if cell.get("kind") not in ("numeric", "semantic", "untargeted"):
                raise ValidationError(f"Schema-2 cell has unknown kind {cell.get('kind')!r}")

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
