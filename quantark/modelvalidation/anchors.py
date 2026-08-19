"""Deterministic anchors: the cheap CI residue of an expensive certification.

A full certification runs for hours because the stochastic benchmark is
expensive. Anchors extract what stays true afterwards: the deterministic
engine's own outputs at pinned configurations. Re-running only the deterministic
side takes seconds, so every commit can check that the certified engine still
produces the numbers it was certified on -- without re-certifying anything.

An anchor is not a second opinion about correctness. It is a tripwire: it says
"this engine no longer computes what the evidence describes", which means the
banked certificate no longer applies.

**Tolerance policy.** On the machine that produced the evidence the comparison
is exact -- any drift there is a real change. On a different architecture the
comparison uses a relative tolerance, because IEEE results legitimately differ
in the last ULP or two across instruction sets (this repo's CI is x86_64 Linux
while evidence is typically banked on ARM64 macOS).
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, List, Mapping

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.candidate import CandidateEvaluator
from quantark.modelvalidation.evidence import read_json
from quantark.modelvalidation.study import CertificationStudy
from quantark.modelvalidation.yaml_loader import load_study_text

ANCHOR_SCHEMA = 1

#: Cross-architecture slack. Tight enough that any real numerical change fails.
#:
#: Calibrated 2026-08-19 against the measurement, not guessed. The first CI run
#: of the banked certificates on x86_64 Linux -- the evidence having been frozen
#: on ARM64 -- produced 326 mismatches against the original 1e-12, spread over
#: 1e-14 to 2.2e-11 relative with a median of 2.0e-12 and no outlier of a
#: different order. That is architecture noise, not a behaviour difference:
#: x86-64 and ARM64 differ in FMA contraction, libm transcendentals and BLAS
#: kernels by the last ULP, and an autocallable PDE/QUAD solve marching hundreds
#: of steps through barrier events amplifies 1e-16 to the 1e-11 seen here. The
#: suite's own cross-arch standard (test/golden_compare.py) is 1e-9, which keeps
#: ~44x margin over that noise while still failing on any real numerics change
#: -- those move these engines by 1e-3 or more, six orders above this bound.
DEFAULT_REL_TOL = 1e-9

#: Absolute floor, so a value of exactly zero remains comparable, and so that
#: near-zero outputs (deep-OTM greeks, tail probabilities) where a relative
#: tolerance has no purchase stay comparable too.
DEFAULT_ABS_TOL = 1e-12

#: Floor applied to a banked file's OWN tolerances when comparing off the
#: banking machine. Cross-architecture agreement is bounded by what the
#: architectures can deliver, so a certificate cannot ask for more precision
#: than that -- and every certificate banked before this floor existed carries
#: the old, un-validated 1e-12. Raising the floor rather than rewriting banked
#: evidence keeps the anchored VALUES untouched, which is the whole point of
#: them.
CROSS_ARCH_REL_TOL = DEFAULT_REL_TOL
CROSS_ARCH_ABS_TOL = DEFAULT_ABS_TOL


def machine_fingerprint() -> Dict[str, str]:
    """Identify the architecture, to decide exact vs tolerance comparison."""
    return {"machine": platform.machine(), "system": platform.system()}


def extract_anchors(
    payload: Mapping[str, Any],
    study: CertificationStudy,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict:
    """Extract anchor values from a certificate.

    Only candidate/case pairs whose every quantity produced a real verdict are
    anchored: an errored cell has no value worth pinning.

    Raises:
        ValidationError: the study has no source text, so the anchors could not
            be re-run from the file alone.
    """
    source_text = payload["study"].get("source_text") or study.source_text
    if not source_text:
        raise ValidationError(
            "Anchors require the study's source text so they can be re-run from the "
            "anchor file alone; certify a study loaded from YAML (or set source_text)"
        )

    grouped: Dict[tuple, Dict[str, float]] = {}
    errored: set = set()
    for cell in payload["cells"]:
        key = (cell["candidate"], cell["case"])
        if cell["verdict"] == "ERROR" or cell["candidate_value"] is None:
            errored.add(key)
            continue
        grouped.setdefault(key, {})[cell["quantity"]] = cell["candidate_value"]

    anchors = [
        {"candidate": candidate, "case": case, "values": values}
        for (candidate, case), values in sorted(grouped.items())
        if (candidate, case) not in errored
    ]

    return {
        "schema": ANCHOR_SCHEMA,
        "study_source_text": source_text,
        "fingerprint": machine_fingerprint(),
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "certificate_sha256": payload.get("projected_sha256"),
        "anchors": anchors,
    }


def _candidates_by_name(study: CertificationStudy) -> Dict[str, CandidateEvaluator]:
    return {candidate.name(): candidate for candidate in study.candidates}


def _matches(
    expected: float, actual: float, exact: bool, rel_tol: float, abs_tol: float
) -> bool:
    """Compare one anchored value.

    The tolerance is genuinely relative to the anchored magnitude, with a tiny
    absolute floor so a value of exactly zero is still comparable. A fixed
    absolute floor of order 1 would be far too generous for the small
    magnitudes engine outputs often carry (a delta quantum is ~1e-4).
    """
    if exact:
        return actual == expected
    return abs(actual - expected) <= rel_tol * abs(expected) + abs_tol


def assert_anchors(anchor_path: str | Path) -> None:
    """Re-run the anchored engines and compare against the banked values.

    Designed to be one line in a pytest case.

    Raises:
        ValidationError: the anchor file is malformed, or names a candidate or
            case the study no longer defines.
        AssertionError: an engine no longer reproduces its anchored values. The
            message lists every mismatch, not just the first.
    """
    anchors = read_json(anchor_path)
    if anchors.get("schema") != ANCHOR_SCHEMA:
        raise ValidationError(
            f"Anchor schema must be {ANCHOR_SCHEMA}, got {anchors.get('schema')}"
        )

    study = load_study_text(anchors["study_source_text"])
    candidates = _candidates_by_name(study)
    cases = {case.name: case for case in study.cases}

    exact = anchors["fingerprint"] == machine_fingerprint()
    rel_tol = float(anchors.get("rel_tol", DEFAULT_REL_TOL))
    abs_tol = float(anchors.get("abs_tol", DEFAULT_ABS_TOL))
    if not exact:
        # Off the banking machine the comparison cannot be tighter than the
        # architectures agree; see CROSS_ARCH_REL_TOL.
        rel_tol = max(rel_tol, CROSS_ARCH_REL_TOL)
        abs_tol = max(abs_tol, CROSS_ARCH_ABS_TOL)

    failures: List[str] = []
    for entry in anchors["anchors"]:
        name, case_name = entry["candidate"], entry["case"]
        if name not in candidates:
            raise ValidationError(
                f"Anchor names candidate {name!r}, which the study no longer defines"
            )
        if case_name not in cases:
            raise ValidationError(
                f"Anchor names case {case_name!r}, which the study no longer defines"
            )

        result = candidates[name].evaluate(cases[case_name])
        for quantity, expected in sorted(entry["values"].items()):
            actual = result.values.get(quantity)
            if actual is None:
                failures.append(
                    f"{name}/{case_name}/{quantity}: engine no longer produces this quantity"
                )
                continue
            if not _matches(expected, actual, exact, rel_tol, abs_tol):
                failures.append(
                    f"{name}/{case_name}/{quantity}: expected {expected!r}, got "
                    f"{actual!r} (delta {actual - expected:.3e})"
                )

    if failures:
        mode = (
            "exact (same machine)"
            if exact
            else f"rel_tol={rel_tol:g}, abs_tol={abs_tol:g} (cross-arch)"
        )
        raise AssertionError(
            f"{len(failures)} anchor(s) no longer reproduce, comparison {mode}:\n  "
            + "\n  ".join(failures)
            + "\n\nThe banked certificate no longer describes this engine; re-certify "
            "or amend before releasing."
        )
