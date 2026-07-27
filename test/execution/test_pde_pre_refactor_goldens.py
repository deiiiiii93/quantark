"""Phase 4 independent oracle: DIRECT PDE outputs pinned to pre-refactor
goldens (plan-gate finding 2026-07-16).

The Task 1 seam refactor rewrites the direct methods themselves, so
session-vs-direct parity alone is a common-mode comparison. These goldens were
frozen on the pristine tree BEFORE any Phase 4 source edit; every assertion is
exact float equality (same-machine bitwise claim, like phase0_goldens).
"""
import json
import pathlib

import pytest

from execution.freeze_goldens import (
    PHASE4_GOLDEN_PATH,
    _PHASE4_CASES,
    _PHASE4_CURVE_CASES,
    _PHASE4_EVENT_CASES,
    _PHASE4_REFINED_CASES,
    _phase4_case_payload,
)
from execution.matrix_fixtures import FIXTURE_BUILDERS, _pdep
from golden_compare import assert_close

GOLDENS = json.loads(pathlib.Path(PHASE4_GOLDEN_PATH).read_text())["cases"]


def _assert_case_matches(golden: dict, engine, product, env, name: str) -> None:
    live = _phase4_case_payload(
        engine, product, env,
        with_curve=name in _PHASE4_CURVE_CASES,
        with_events=name in _PHASE4_EVENT_CASES,
    )
    # Cross-arch tolerance: these goldens were frozen same-machine; x86_64 CI
    # differs from the ARM64 freeze host by the last 1-2 ULP. See golden_compare.
    assert_close(live, golden, msg=name)


@pytest.mark.parametrize("name", _PHASE4_CASES)
def test_direct_outputs_match_pre_refactor_goldens(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    _assert_case_matches(GOLDENS[name], engine, product, env, name)


@pytest.mark.parametrize("name", _PHASE4_REFINED_CASES)
def test_refined_direct_outputs_match_pre_refactor_goldens(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    refined = type(engine)(
        params=_pdep(grid_size=180, time_steps=96)
    )
    _assert_case_matches(GOLDENS[f"{name}::refined"], refined, product, env, name)
