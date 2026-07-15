"""Immutable run context (spec section 5.3)."""
import dataclasses

import pytest

from quantark.execution.context import PricingRunContext, default_context


def test_default_context_is_serial_and_frozen():
    ctx = default_context()
    assert ctx.execution_policy.batch.backend == "serial"
    assert ctx.parent_run_id is None
    assert isinstance(ctx.run_id, str) and ctx.run_id
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_id = "x"


def test_child_shares_services_and_links_parent():
    ctx = default_context()
    child = ctx.child()
    assert child.parent_run_id == ctx.run_id
    assert child.run_id != ctx.run_id
    assert child.diagnostics_sink is ctx.diagnostics_sink
    assert child.execution_policy is ctx.execution_policy
