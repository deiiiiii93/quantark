"""Phase 4 convergence gate (spec section 21 exit).

Session == direct bitwise at BOTH the production and the refined resolution,
AND both equal the Task 0 pre-refactor goldens bitwise — the independent
oracle that keeps this gate non-tautological after the seam refactor
(plan-gate finding 2026-07-16).
"""
import json
import pathlib

import pytest

from quantark.execution import PricingRequest, PricingSession

from execution.freeze_goldens import PHASE4_GOLDEN_PATH
from execution.matrix_fixtures import FIXTURE_BUILDERS, _pdep

GOLDENS = json.loads(pathlib.Path(PHASE4_GOLDEN_PATH).read_text())["cases"]


def _session_price(engine, product, env) -> float:
    with PricingSession() as session:
        return session.execute(
            engine, PricingRequest(product=product, pricing_env=env)
        ).value


@pytest.mark.parametrize("name", ["EuropeanPDESolver", "SnowballPDESolver"])
def test_convergence_gate_production_and_refined(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    refined = type(engine)(
        params=_pdep(grid_size=180, time_steps=96, auto_grid=False)
    )

    direct_prod = engine.price(product, env)
    direct_ref = refined.price(product, env)

    # Independent oracle: pre-refactor goldens pin the direct values.
    assert direct_prod == GOLDENS[name]["price"]
    assert direct_ref == GOLDENS[f"{name}::refined"]["price"]

    # Session bitwise at both resolutions -> the refinement delta is the
    # pre-refactor refinement delta exactly.
    assert _session_price(engine, product, env) == direct_prod
    assert _session_price(refined, product, env) == direct_ref
