"""The example's legacy _mo_common helper matches the promoted library function (split from test/volcalibration/test_surface_smoothing.py)."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import build_raw_surface, sabr_smoothed_surface

ROOT = Path(__file__).resolve().parents[4]
LIVE = ROOT / "test/volcalibration/data/mo_snapshot_sample.json"


def _raw():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return build_raw_surface(ListedNormalizer().normalize(snap))


def test_smoothing_matches_the_legacy_helper_bit_for_bit():
    """The promoted function must be numerically identical to the suite's."""
    sys.path.insert(0, str(ROOT / "example/case_studies/mo_volmodels"))
    import _mo_common as mc

    raw = _raw()
    lib = sabr_smoothed_surface(raw, beta=1.0)
    legacy = mc.sabr_smoothed_surface(raw, beta=1.0)
    assert np.array_equal(np.asarray(lib["iv_grid"]), np.asarray(legacy["iv_grid"]))
