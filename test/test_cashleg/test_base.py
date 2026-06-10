import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from quantark.cashleg.base import CashLeg, LegDirection


def test_leg_direction_signs():
    assert LegDirection.BUYER_RECEIVES.value == +1
    assert LegDirection.BUYER_PAYS.value == -1


def test_cashleg_is_abstract():
    with pytest.raises(TypeError):
        CashLeg(direction=LegDirection.BUYER_RECEIVES)


def test_cashleg_subclass_requires_value_method():
    class IncompleteLeg(CashLeg):
        pass

    with pytest.raises(TypeError):
        IncompleteLeg(direction=LegDirection.BUYER_RECEIVES)
