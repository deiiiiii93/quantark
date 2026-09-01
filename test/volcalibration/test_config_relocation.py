import pytest

from quantark.util.exceptions import ValidationError


def test_config_is_importable_from_the_new_home():
    from quantark.volcalibration.config import HESTON_PRESETS, VolModelCalibrationConfig

    cfg = VolModelCalibrationConfig()
    assert cfg.heston_preset == "mo_frozen"
    assert "mo_frozen" in HESTON_PRESETS
    assert cfg.slv_n_x == 161


def test_all_legacy_import_paths_are_the_same_class():
    from quantark.backtest.otc import VolModelCalibrationConfig as FromOtc
    from quantark.backtest.replay import VolModelCalibrationConfig as FromReplay
    from quantark.backtest.replay.config import VolModelCalibrationConfig as FromModule
    from quantark.volcalibration.config import VolModelCalibrationConfig as Canonical

    assert FromOtc is Canonical
    assert FromReplay is Canonical
    assert FromModule is Canonical


def test_validation_still_fails_closed():
    from quantark.volcalibration.config import VolModelCalibrationConfig

    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(heston_preset="nope")
    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(slv_n_x=2)
    with pytest.raises(ValidationError):
        VolModelCalibrationConfig(heston_temporal_regularization=1.0)  # no reference
