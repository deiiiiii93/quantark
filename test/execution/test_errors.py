"""Framework exception hierarchy (spec section 15)."""
import pytest

from quantark.util.exceptions import QuantArkException


FRAMEWORK_ERRORS = [
    "CapabilityError",
    "ResourceBudgetExceeded",
    "PreparationError",
    "TaskExecutionError",
    "DeterminismViolation",
    "ValidationGateError",
]


@pytest.mark.parametrize("name", FRAMEWORK_ERRORS)
def test_framework_error_derives_quantark_root(name):
    import quantark.execution.errors as errors

    exc_type = getattr(errors, name)
    assert issubclass(exc_type, QuantArkException)
    with pytest.raises(QuantArkException):
        raise exc_type("boom")


def test_errors_module_all_is_exact():
    import quantark.execution.errors as errors

    assert sorted(errors.__all__) == sorted(FRAMEWORK_ERRORS)
