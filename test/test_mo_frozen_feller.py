"""The ``mo_frozen`` Heston preset must not produce Feller-violating fits.

Spec provenance: the 0.4.0 re-baseline design §7A traced a 2.5%-of-notional
disagreement between the 2D Heston PDE and the QE-M MC reference to Heston fits
whose ``kappa``/``sigma`` sit pinned on the preset's bounds, violating
``2*kappa*theta >= sigma**2``.  Every CFFEX MO settlement surface sampled did so
under the previous soft-penalty-only policy (``regularize_feller=0.05`` with
``enforce_feller=False``), landing at ratios ~0.29-0.48.

The fix is a hard constraint, and it changes the model being tested: fits on
those dates now trade smile accuracy for feasibility.  That is a deliberate
decision (owner, 2026-07-30), which is why the shipped flag is locked here
rather than left as an incidental default.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from quantark.backtest.replay.config import VolModelCalibrationConfig
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.volmodels.calibration import HESTON_PRESETS, VolModelCalibrator

REAL_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "mo_volmodels"
    / "data"
    / "history"
    / "iv_surface"
    / "mo_iv_surface_20230504.json"
)

pytestmark = pytest.mark.skipif(
    not REAL_ARTIFACT.exists(), reason="real IV surface artifact not present"
)


def _artifact() -> IvSurfaceArtifact:
    return IvSurfaceArtifact.from_file(REAL_ARTIFACT)


def test_mo_frozen_preset_enforces_feller():
    """Policy lock: a silent revert to the soft penalty would reopen §7A's gap."""
    assert HESTON_PRESETS["mo_frozen"]["enforce_feller"] is True


def test_real_mo_surface_calibrates_inside_the_feller_region():
    calibrator = VolModelCalibrator(VolModelCalibrationConfig())

    model = calibrator.calibrate("heston", _artifact())

    p = model.heston_params
    assert 2.0 * p.kappa * p.theta >= p.sigma**2
    assert model.record["feller_satisfied"] is True
    assert model.record["feller_margin"] >= 0.0


def test_heston_record_carries_the_scale_free_feller_ratio():
    """§7A.4(3) conditions the G2 verdict on ``2*kappa*theta/sigma**2``.

    ``feller_margin`` alone cannot support that: it is a difference, so a
    margin of 1e-3 is comfortable at sigma=0.03 and vanishing at sigma=0.6.
    The ratio has to be in the record, or the re-run produces evidence that
    cannot be bucketed by regime after the fact.
    """
    model = VolModelCalibrator(VolModelCalibrationConfig()).calibrate(
        "heston", _artifact()
    )

    p = model.heston_params
    assert model.record["feller_ratio"] == pytest.approx(
        2.0 * p.kappa * p.theta / p.sigma**2
    )
    assert model.record["feller_ratio"] >= 1.0


def test_feller_enforcement_invalidates_cached_heston_entries(tmp_path, monkeypatch):
    """§7A.4(1)'s cache-safety claim, exercised rather than asserted.

    The heston fingerprint embeds the whole preset, so flipping
    ``enforce_feller`` must land in a *different* cache file — otherwise the
    re-baseline would silently reuse the Feller-violating fits it was run to
    replace.
    """
    calibrator = VolModelCalibrator(
        VolModelCalibrationConfig(cache_dir=str(tmp_path))
    )
    calibrator.calibrate("heston", _artifact())
    enforced = {p.name for p in tmp_path.glob("heston-*.json")}

    soft = copy.deepcopy(HESTON_PRESETS["mo_frozen"])
    soft["enforce_feller"] = False
    monkeypatch.setitem(HESTON_PRESETS, "mo_frozen", soft)
    VolModelCalibrator(
        VolModelCalibrationConfig(cache_dir=str(tmp_path))
    ).calibrate("heston", _artifact())

    all_entries = {p.name for p in tmp_path.glob("heston-*.json")}
    assert len(enforced) == 1
    assert len(all_entries) == 2
