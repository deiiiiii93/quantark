"""The relocation must not change the disk cache key."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_golden import fixtures  # noqa: E402

from quantark.param.vol.surface_history import VolSurfaceHistory  # noqa: E402


def _artifact(tmp_path):
    history_dir = fixtures.write_localvol_history(tmp_path)
    return VolSurfaceHistory(history_dir).surface_for(fixtures.DATE_A)


def test_entry_written_via_the_old_path_is_a_hit_on_the_new_one(tmp_path):
    from quantark.volcalibration.calibrate import VolModelCalibrator as New
    from quantark.volcalibration.config import VolModelCalibrationConfig
    from quantark.volmodels.calibration import VolModelCalibrator as Old

    artifact = _artifact(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first = Old(VolModelCalibrationConfig(cache_dir=str(cache_dir))).calibrate(
        "localvol", artifact
    )
    assert first.record["cache_hit"] is False

    second = New(VolModelCalibrationConfig(cache_dir=str(cache_dir))).calibrate(
        "localvol", artifact
    )
    assert second.record["cache_hit"] is True
    assert second.surface_sha == first.surface_sha


def test_old_module_re_exports_the_new_class():
    from quantark.volcalibration.calibrate import VolModelCalibrator as New
    from quantark.volmodels.calibration import VolModelCalibrator as Old

    assert Old is New
