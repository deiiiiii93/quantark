"""Relocation invariants for quantark.volmodels.calibration (plan Task 4).

Runs everywhere: it uses the committed synthetic surface history from the
replay goldens, not the uncommitted real-data cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.param.vol.surface_history import VolSurfaceHistory  # noqa: E402


def _synthetic_artifact(tmp_path: Path):
    history_dir = fixtures.write_localvol_history(tmp_path)
    return VolSurfaceHistory(history_dir).surface_for(fixtures.DATE_A)


def test_cache_key_survives_module_relocation(tmp_path):
    """The disk cache key is sha256(surface_sha|variant|fingerprint) with no
    module paths — an entry written via the deprecated otc path must be a
    cache hit for the canonical quantark.volmodels.calibration class."""
    from quantark.backtest.otc.vol_calibrators import (
        VolModelCalibrator as OldPathCalibrator,
    )
    from quantark.backtest.otc import VolModelCalibrationConfig
    from quantark.volmodels.calibration import (
        VolModelCalibrator as NewPathCalibrator,
    )

    artifact = _synthetic_artifact(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first = OldPathCalibrator(
        VolModelCalibrationConfig(cache_dir=str(cache_dir))
    ).calibrate("localvol", artifact)
    assert first.record["cache_hit"] is False

    second = NewPathCalibrator(
        VolModelCalibrationConfig(cache_dir=str(cache_dir))
    ).calibrate("localvol", artifact)
    assert second.record["cache_hit"] is True
    assert OldPathCalibrator is NewPathCalibrator
