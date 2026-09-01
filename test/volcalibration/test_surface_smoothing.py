import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import (
    build_raw_surface,
    prepare_model_surface,
    sabr_smoothed_surface,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _raw():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return build_raw_surface(ListedNormalizer().normalize(snap))


def test_smoothing_matches_the_legacy_helper_bit_for_bit():
    """The promoted function must be numerically identical to the suite's."""
    sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
    import _mo_common as mc

    raw = _raw()
    lib = sabr_smoothed_surface(raw, beta=1.0)
    legacy = mc.sabr_smoothed_surface(raw, beta=1.0)
    assert np.array_equal(np.asarray(lib["iv_grid"]), np.asarray(legacy["iv_grid"]))


def test_total_variance_is_non_decreasing_in_maturity():
    out = sabr_smoothed_surface(_raw(), beta=1.0)
    grid = np.asarray(out["iv_grid"])
    T = np.asarray(out["maturities"])
    w = grid * grid * T[:, None]
    assert np.all(np.diff(w, axis=0) >= -1e-12)


def test_raw_points_are_retained_beside_the_model_points():
    out = sabr_smoothed_surface(_raw(), beta=1.0)
    pe = out["per_expiry"][0]
    assert len(pe["raw_points"]) == len(pe["points"])
    assert pe["raw_points"] != pe["points"]  # smoothing actually moved them
    assert out["target_smoothing"]["method"] == "sabr_calendar_projected"
    assert "sabr_params" in pe


def test_prepare_model_surface_none_is_a_passthrough():
    raw = _raw()
    out = prepare_model_surface(raw, iv_smoothing="none")
    assert out["target_smoothing"] == {"method": "none"}
    assert out["iv_grid"] == raw["iv_grid"]


def test_prepare_model_surface_rejects_unknown_mode():
    with pytest.raises(ValidationError, match="iv_smoothing"):
        prepare_model_surface(_raw(), iv_smoothing="bogus")
