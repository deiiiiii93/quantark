"""Backend-independent scenario plan execution (spec sections 13.3, 15).

Built-in runner ``request/v1`` dispatches a transformed ``PricingRequest``
through the existing kernel — it is ``value_kind="native"`` and therefore
serial/threads-only (native value objects do not cross process
boundaries; plan-gate finding 4).
"""
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.scenario import registries

__all__ = ["ResolvedCellInputs", "run_request_cell"]


class ResolvedCellInputs:
    """What a runner receives: the base inputs, the transformer output,
    and the engine for this cell (None for workflow runners that build
    their own engines from payload parameters)."""

    __slots__ = ("base_inputs", "transformed", "engine")

    def __init__(self, base_inputs, transformed, engine=None):
        self.base_inputs = base_inputs
        self.transformed = transformed
        self.engine = engine


def run_request_cell(cell, resolved, child_context):
    """Dispatch the transformed request through the kernel (spec 13.3)."""
    from quantark.execution.kernel import ExecutionKernel

    request = resolved.transformed
    outcome = ExecutionKernel.dispatch(resolved.engine, request, child_context)
    manifest_fp = try_fingerprint(outcome.manifest)
    return outcome.value, outcome.normalized_economics, manifest_fp


registries.register_runner("request/v1", run_request_cell, value_kind="native")
