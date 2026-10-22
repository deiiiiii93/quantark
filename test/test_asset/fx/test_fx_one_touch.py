import pytest

from quantark.asset.fx.product.option import FxOneTouchOption
from quantark.util.exceptions import ValidationError


def test_one_touch_construction_and_payoff():
    ot = FxOneTouchOption(barrier=1.30, is_up=True, payout=1.0, maturity=0.5)
    assert ot.barrier == 1.30
    assert ot.is_up is True
    assert ot.payout == 1.0
    # up-touch terminal payoff: pays at/above the barrier, else 0
    assert ot.get_payoff(1.31) == 1.0
    assert ot.get_payoff(1.29) == 0.0


def test_one_touch_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        FxOneTouchOption(barrier=-1.0, is_up=True, maturity=0.5)
    with pytest.raises(ValidationError):
        FxOneTouchOption(barrier=1.30, is_up=True, payout=0.0, maturity=0.5)
