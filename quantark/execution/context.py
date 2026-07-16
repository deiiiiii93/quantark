"""Immutable pricing run context (spec section 5.3).

The context itself never mutates; repositories, sinks, and registries are
mutable services behind stable handles. ``child`` returns a new context in a
child scope. No active run context is ever stored globally or thread-locally
(spec section 3.3).
"""
import uuid
from dataclasses import dataclass, field, replace

from quantark.execution.diagnostics import InMemoryDiagnosticsSink
from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ResourceBudget,
    resolve_execution_policy,
    resolve_resource_budget,
)

__all__ = ["PricingRunContext", "default_context"]


def _new_run_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class PricingRunContext:
    execution_policy: ExecutionPolicy
    resource_budget: ResourceBudget
    determinism_policy: DeterminismPolicy
    diagnostics_sink: object
    adapter_registry: object | None = None
    cancellation_token: object | None = None
    artifact_cache: object | None = None
    lease_manager: object | None = None
    draw_repository: object | None = None
    run_id: str = field(default_factory=_new_run_id)
    parent_run_id: str | None = None
    config_snapshot: tuple = ()

    def child(self) -> "PricingRunContext":
        """New context in a child scope sharing service handles."""
        return replace(
            self, run_id=_new_run_id(), parent_run_id=self.run_id
        )


def default_context(environ=None) -> PricingRunContext:
    """Serial default context; resolves policy and budget exactly once."""
    policy, policy_sources = resolve_execution_policy(environ=environ)
    budget, budget_sources = resolve_resource_budget(environ=environ)
    return PricingRunContext(
        execution_policy=policy,
        resource_budget=budget,
        determinism_policy=DeterminismPolicy(),
        diagnostics_sink=InMemoryDiagnosticsSink(),
        config_snapshot=tuple(policy_sources) + tuple(budget_sources),
    )
