import pytest
from quantark.util.enum.engine_enums import LeverageCalibrationMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.risk.contracts import SlvCalibrationSpec
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig


def test_default_method_is_ffp_with_fp_config():
    spec = SlvCalibrationSpec()
    assert spec.method is LeverageCalibrationMethod.FORWARD_FOKKER_PLANCK


def test_mc_method_accepts_mc_fields():
    spec = SlvCalibrationSpec(method=LeverageCalibrationMethod.MC_BINNING, num_paths=10_000)
    assert spec.num_paths == 10_000


def test_fp_config_rejected_for_mc():
    with pytest.raises(ValidationError):
        SlvCalibrationSpec(method=LeverageCalibrationMethod.MC_BINNING,
                           fp_config=FpCalibrationConfig())
