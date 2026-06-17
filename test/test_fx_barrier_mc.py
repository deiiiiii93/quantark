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
