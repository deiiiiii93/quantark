"""Schema-1 identities, checkpoints and payloads keep their pre-migration shape.

The golden was written by the unmodified framework (Task 2 of the intraday plan). Running
the new code twice and comparing the two runs proves nothing about the old format; this
compares against what the old code actually produced.
"""
import json
import math
from pathlib import Path

import pytest

from quantark.modelvalidation.candidate import candidate_identity, serialize_candidate_result
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.study import CaseSpec

from conftest import make_study

GOLDEN = Path(__file__).parent / "golden" / "schema1_wire.json"


def _current(out_dir) -> dict:
    study = make_study()
    case = CaseSpec(name="ordinary")
    candidate = study.candidates[0]
    payload = certify(study, out_dir=out_dir).payload
    return {
        "candidate_identity": candidate_identity(candidate, case),
        "candidate_identity_hash": identity_hash(candidate_identity(candidate, case)),
        "candidate_checkpoint": serialize_candidate_result(candidate.evaluate(case)),
        "reference_identity_hash": identity_hash(study.reference.identity(case)),
        "schema": payload["schema"],
        "study": payload["study"],
        "references": payload["references"],
        "cells": payload["cells"],
        "aggregates": payload["aggregates"],
        "decisions": payload["decisions"],
        "payload_keys": sorted(payload),
    }


def _assert_same(expected, actual, path="$"):
    """Exact structure, exact strings/ints/bools, floats to 1e-12 relative (cross-architecture last-ULP slack)."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and sorted(expected) == sorted(actual), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), f"{path}: length"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_same(e, a, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-300), f"{path}: {actual!r} != {expected!r}"
    else:
        assert expected == actual, f"{path}: {actual!r} != {expected!r}"


def test_schema_1_wire_format_is_the_pre_migration_one(tmp_path):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    golden.pop("pinned_at_revision")                    # provenance for the reader, not part of the format
    _assert_same(golden, json.loads(json.dumps(_current(tmp_path))))


def test_the_identity_keeps_its_original_keys():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden["candidate_identity"]) == ["candidate", "case", "params"]
    assert sorted(golden["candidate_identity"]["case"]) == ["environment_params", "name", "product_params"]
    assert sorted(golden["candidate_checkpoint"]) == ["ladders", "values"]
    assert sorted(golden["cells"][0]["gate"]) == [
        "envelope_c", "envelope_within_bound", "interval_c", "interval_within_bound", "passed", "se_budget_met",
        "se_c", "signed_err_c"]
    assert sorted(golden["study"]["sampling"]) == ["bump", "max_batches", "min_batches", "paths_per_batch", "seed"]
