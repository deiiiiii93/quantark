import pytest
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError


def test_fx_mc_params_defaults():
    p = FxMCParams()
    assert p.num_paths == 200_000
    assert p.seed == 42
    assert p.use_antithetic is True
    assert p.method == MonteCarloMethod.PSEUDO


def test_fx_mc_params_rejects_nonpositive_paths():
    with pytest.raises(ValidationError):
        FxMCParams(num_paths=0)


# ---------------------------------------------------------------------------
# Task 4: FxBarrierOption monitoring fields
# ---------------------------------------------------------------------------

from quantark.asset.fx.product.option import FxBarrierOption
from quantark.util.enum import OptionType, FxBarrierType, ObservationType


def _barrier(**kw):
    base = dict(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        maturity=1.0,
    )
    base.update(kw)
    return FxBarrierOption(**base)


def test_barrier_defaults_to_continuous():
    opt = _barrier()
    assert opt.monitoring == ObservationType.CONTINUOUS
    assert opt.observation_times is None


def test_discrete_requires_sorted_unique_in_range_times():
    ok = _barrier(monitoring=ObservationType.DISCRETE,
                  observation_times=[0.25, 0.5, 0.75, 1.0])
    assert ok.observation_times == [0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=None)
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 0.25])
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 0.5])
    with pytest.raises(ValidationError):
        _barrier(monitoring=ObservationType.DISCRETE, observation_times=[0.5, 1.5])
