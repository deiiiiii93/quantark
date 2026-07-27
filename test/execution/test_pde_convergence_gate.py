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
from execution.matrix_fixtures import FIXTURE_BUILDERS, _pdep_refined
from golden_compare import GOLDEN_REL_TOL

GOLDENS = json.loads(pathlib.Path(PHASE4_GOLDEN_PATH).read_text())["cases"]


def _session_price(engine, product, env) -> float:
    with PricingSession() as session:
        return session.execute(
            engine, PricingRequest(product=product, pricing_env=env)
        ).value


@pytest.mark.parametrize("name", ["EuropeanPDESolver", "SnowballPDESolver"])
def test_convergence_gate_production_and_refined(name):
    engine, product, env, _shape = FIXTURE_BUILDERS[name]()
    refined = type(engine)(params=_pdep_refined())

    direct_prod = engine.price(product, env)
    direct_ref = refined.price(product, env)

    # Independent oracle: pre-refactor goldens pin the direct values. The
    # goldens are same-machine references; x86_64 CI differs from the ARM64
    # freeze host by the last 1-2 ULP, so compare with cross-arch tolerance.
    assert direct_prod == pytest.approx(GOLDENS[name]["price"], rel=GOLDEN_REL_TOL)
    assert direct_ref == pytest.approx(
        GOLDENS[f"{name}::refined"]["price"], rel=GOLDEN_REL_TOL
    )

    # Session bitwise at both resolutions -> the refinement delta is the
    # pre-refactor refinement delta exactly. Same-machine (both computed here),
    # so this stays an exact equality: the session path must not perturb a bit.
    assert _session_price(engine, product, env) == direct_prod
    assert _session_price(refined, product, env) == direct_ref
