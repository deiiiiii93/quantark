"""Typed scenario planning and multi-backend execution (spec section 13).

Phase 5 subpackage: contracts, importable registries, the normalizing
planner, the backend-independent runner, spawn worker reconstruction, and
the complete-payload comparison validator.
"""
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
    CallableRef,
    ScenarioCell,
    ScenarioPlan,
    WorkerSpec,
)

__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "BaseInputsRef",
    "CallableRef",
    "ScenarioCell",
    "ScenarioPlan",
    "WorkerSpec",
]
