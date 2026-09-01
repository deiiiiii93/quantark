import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import NumericalError
from quantark.volcalibration.admission import (
    AdmissionError,
    AdmissionReason,
    validate_static_arbitrage,
)
from quantark.volcalibration.normalize.listed import ListedNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.surface import build_raw_surface, sabr_smoothed_surface

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"


def _smoothed():
    snap = QuoteSnapshot.from_legacy_live(
        json.loads(LIVE.read_text()), trade_date=date(2026, 7, 6), symbol="000852.SH"
    )
    return sabr_smoothed_surface(build_raw_surface(ListedNormalizer().normalize(snap)))


def test_clean_surface_admits_via_the_dupire_path():
    label = validate_static_arbitrage(_smoothed())
    assert label == "build_dupire_local_vol(validate_arbitrage=True)"


def test_two_maturity_surface_uses_the_reduced_form_checks():
    s = _smoothed()
    s["maturities"] = s["maturities"][:2]
    s["iv_grid"] = s["iv_grid"][:2]
    s["per_expiry"] = s["per_expiry"][:2]
    assert validate_static_arbitrage(s) == "reduced_form_dupire_checks_2_maturities"


def test_calendar_arbitrage_is_rejected_not_repaired():
    s = _smoothed()
    grid = np.asarray(s["iv_grid"])
    grid[-1] *= 0.3  # collapse the far maturity: total variance now falls with T
    s["iv_grid"] = grid.tolist()
    with pytest.raises(NumericalError):
        validate_static_arbitrage(s)


def test_admission_error_carries_a_machine_readable_reason():
    err = AdmissionError(AdmissionReason.STATIC_ARBITRAGE, "calendar arbitrage at T=0.5")
    assert err.reason is AdmissionReason.STATIC_ARBITRAGE
    assert err.reason.value == "static_arbitrage"
    assert "calendar arbitrage" in err.detail
    assert "static_arbitrage" in str(err)
