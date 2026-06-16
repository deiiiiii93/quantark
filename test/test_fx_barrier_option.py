import pytest

from quantark.asset.fx.product.option import FxBarrierOption
from quantark.util.enum import OptionType, FxBarrierType
from quantark.util.exceptions import ValidationError


def test_fx_barrier_type_values():
    assert FxBarrierType.KNOCK_OUT.value == "knock_out"
    assert FxBarrierType.KNOCK_IN.value == "knock_in"
    assert str(FxBarrierType.KNOCK_OUT) == "knock_out"


def test_barrier_option_construction():
    opt = FxBarrierOption(
        strike=1.20, barrier=1.30, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=0.5,
    )
    assert opt.strike == 1.20
    assert opt.barrier == 1.30
    assert opt.knock_type == FxBarrierType.KNOCK_OUT
    # unconditional vanilla terminal payoff (barrier handled by engine)
    assert opt.get_payoff(1.25) == pytest.approx(0.05)
    assert opt.get_payoff(1.15) == 0.0


def test_barrier_rejects_rebate_at_hit_for_knock_in():
    with pytest.raises(ValidationError):
        FxBarrierOption(
            strike=1.20, barrier=1.30, is_up=True,
            knock_type=FxBarrierType.KNOCK_IN, option_type=OptionType.CALL,
            maturity=0.5, rebate=0.01, rebate_at_hit=True,
        )


def test_barrier_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        FxBarrierOption(
            strike=-1.0, barrier=1.30, is_up=True,
            knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
            maturity=0.5,
        )
