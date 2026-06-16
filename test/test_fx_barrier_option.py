import pytest

from quantark.util.enum import FxBarrierType


def test_fx_barrier_type_values():
    assert FxBarrierType.KNOCK_OUT.value == "knock_out"
    assert FxBarrierType.KNOCK_IN.value == "knock_in"
    assert str(FxBarrierType.KNOCK_OUT) == "knock_out"
