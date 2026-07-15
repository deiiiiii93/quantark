"""PricingSession: the explicit framework entry point (spec section 5.5)."""
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    PricingRequest,
)
from quantark.execution.context import PricingRunContext, default_context
from quantark.execution.errors import CapabilityError
from quantark.execution.kernel import ExecutionKernel
from quantark.execution.registry import build_default_registry

__all__ = ["PricingSession"]


class PricingSession:
    """Serial pricing session. Owns only services it creates; idempotently
    closable. ``PricingSession()`` with no context resolves a safe serial
    default exactly once at construction (spec section 11.1)."""

    def __init__(self, context: PricingRunContext | None = None):
        if context is None:
            context = default_context()
        if context.adapter_registry is None:
            import dataclasses

            registry = build_default_registry()
            registry.freeze()
            context = dataclasses.replace(context, adapter_registry=registry)
        self._context = context
        self._closed = False

    @property
    def context(self) -> PricingRunContext:
        return self._context

    def execute(self, engine, request: PricingRequest):
        self._ensure_open()
        return ExecutionKernel.dispatch(engine, request, self._context)

    def price(self, engine, product, pricing_env=None):
        outcome = self.execute(
            engine, PricingRequest(product=product, pricing_env=pricing_env)
        )
        return outcome.value

    def price_many(self, items, *, collect_errors: bool = False) -> list:
        """Serial, caller-ordered pricing of (engine, PricingRequest) pairs.

        Fail-fast by default; ``collect_errors=True`` returns a
        ``PricingFailure`` in place of each failed item (spec section 15).
        """
        self._ensure_open()
        results: list = []
        for index, (engine, request) in enumerate(items):
            try:
                results.append(self.execute(engine, request).value)
            except Exception as exc:  # noqa: BLE001 - typed into the failure record
                if not collect_errors:
                    raise
                item_id = request.request_id or str(index)
                from quantark.execution.diagnostics import RunDiagnostics

                results.append(
                    PricingFailure(
                        item_id=item_id,
                        error=FrameworkErrorInfo(
                            error_type=type(exc).__name__, message=str(exc)
                        ),
                        diagnostics=RunDiagnostics(adapter_id="unresolved"),
                    )
                )
        return results

    def run_scenarios(self, base_request, scenario_specs, engine_factory):
        raise CapabilityError(
            "scenario execution is not available in framework Phase 0; "
            "it arrives with the scenario planner in Phase 5"
        )

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise CapabilityError("PricingSession is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
