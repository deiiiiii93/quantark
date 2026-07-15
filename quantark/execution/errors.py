"""Typed framework exceptions (spec section 15).

All framework errors derive from the existing ``QuantArkException`` root.
Direct legacy engine methods re-raise their historical exceptions without
framework wrapping; these types appear only on explicit framework APIs.
"""
from quantark.util.exceptions import QuantArkException

__all__ = [
    "CapabilityError",
    "ResourceBudgetExceeded",
    "PreparationError",
    "TaskExecutionError",
    "DeterminismViolation",
    "ValidationGateError",
]


class CapabilityError(QuantArkException):
    """An engine/adapter does not support the requested operation, output,
    or backend. Explicit requests never silently fall back."""


class ResourceBudgetExceeded(QuantArkException):
    """A resource lease could not be acquired within the admitted budget."""


class PreparationError(QuantArkException):
    """Immutable prepared-state construction failed."""


class TaskExecutionError(QuantArkException):
    """A submitted execution task failed."""


class DeterminismViolation(QuantArkException):
    """Input mutated during execution, or a reproducibility check failed."""


class ValidationGateError(QuantArkException):
    """A declared numerical or scenario validation gate failed."""
