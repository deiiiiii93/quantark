"""QuantArk composable execution kernel (framework contract v1, Phases 0-5).

Additive public surface (spec section 5). Direct legacy engine calls are
unchanged; this package is reached only through explicit sessions or
``BaseEngine.execute``.
"""
from quantark.execution.api import PricingSession
from quantark.execution.cache.draws import DrawRepository
from quantark.execution.context import PricingRunContext, default_context
from quantark.execution.contracts import (
    DEFAULT_OUTPUTS,
    AdaptivePlan,
    BatchOutcome,
    BatchPlan,
    BatchTask,
    EngineCapabilities,
    FrameworkErrorInfo,
    NormalizedPricingRequest,
    OutputKind,
    PricingFailure,
    PricingOperation,
    PricingOutcome,
    PricingRequest,
    ScenarioOutcome,
    ScenarioSpec,
)
from quantark.execution.errors import (
    CapabilityError,
    DeterminismViolation,
    PreparationError,
    ResourceBudgetExceeded,
    TaskExecutionError,
    ValidationGateError,
)
from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
)
from quantark.execution.scenario.contracts import (
    BaseInputsRef,
    CallableRef,
    ScenarioCell,
    ScenarioPlan,
    WorkerSpec,
)
from quantark.execution.scenario.planner import plan_scenarios
from quantark.execution.scenario.validate import (
    ScenarioComparisonReport,
    compare_scenario_outcomes,
)

__all__ = [
    "AdaptivePlan",
    "BaseInputsRef",
    "BatchOutcome",
    "BatchPlan",
    "BatchTask",
    "CallableRef",
    "CapabilityError",
    "DEFAULT_OUTPUTS",
    "DeterminismPolicy",
    "DeterminismViolation",
    "DrawRepository",
    "EngineCapabilities",
    "ExecutionPolicy",
    "ExecutorSelection",
    "FrameworkErrorInfo",
    "NormalizedPricingRequest",
    "OutputKind",
    "PreparationError",
    "PricingFailure",
    "PricingOperation",
    "PricingOutcome",
    "PricingRequest",
    "PricingRunContext",
    "PricingSession",
    "ResourceBudget",
    "ResourceBudgetExceeded",
    "ScenarioCell",
    "ScenarioComparisonReport",
    "ScenarioOutcome",
    "ScenarioPlan",
    "ScenarioSpec",
    "TaskExecutionError",
    "ValidationGateError",
    "WorkerSpec",
    "compare_scenario_outcomes",
    "default_context",
    "plan_scenarios",
]
