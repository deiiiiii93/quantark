"""Phase 1 exit gate: direct-vs-session parity for EVERY concrete inventory
row (spec section 21 Phase 1)."""
import pytest

from quantark.execution import PricingRequest, PricingSession
from quantark.execution.inventory import ENGINE_INVENTORY

from execution.matrix_fixtures import FIXTURE_BUILDERS

CONCRETE = [r for r in ENGINE_INVENTORY if r.role != "abstract"]


def test_every_concrete_row_has_a_fixture():
    missing = [r.name for r in CONCRETE if r.name not in FIXTURE_BUILDERS]
    assert not missing, f"inventory rows without executable fixtures: {missing}"


@pytest.mark.parametrize("record", CONCRETE, ids=lambda r: r.name)
def test_direct_equals_session(record):
    engine, product, env, call_shape = FIXTURE_BUILDERS[record.name]()
    assert call_shape == record.call_shape
    if call_shape == "env_bound":
        direct = engine.price(product)
    else:
        direct = engine.price(product, env)
    with PricingSession() as session:
        outcome = session.execute(
            engine, PricingRequest(product=product, pricing_env=env)
        )
    assert outcome.value == direct, record.name
    assert type(outcome.value) is type(direct), record.name
