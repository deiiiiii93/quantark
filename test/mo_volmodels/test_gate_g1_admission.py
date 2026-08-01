import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    path = REPO / "example" / "mo_volmodels" / "13_gate_g1_surface_admission.py"
    spec = importlib.util.spec_from_file_location("g1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1"] = mod          # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def _artifact(iv_dir: Path, day: date, n_maturities: int) -> None:
    iv_dir.mkdir(parents=True, exist_ok=True)
    tag = day.strftime("%Y%m%d")
    (iv_dir / f"mo_iv_surface_{tag}.json").write_text(
        json.dumps(
            {
                "trade_date": day.isoformat(),
                "maturities": [0.1 * (i + 1) for i in range(n_maturities)],
                # The real block: criteria, not a verdict.  Present here so the
                # test proves G1 does not read a verdict out of it.
                "admission": {"min_expiries": 2, "sabr_beta": 1.0},
            }
        )
    )


def test_surface_with_three_expiries_passes(tmp_path):
    g1 = _load()
    _artifact(tmp_path, date(2026, 7, 31), 3)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is True
    assert reason == ""


def test_thin_surface_fails_the_dupire_rule(tmp_path):
    """The Phase-1 builder admits 2-expiry surfaces; Dupire needs 3."""
    g1 = _load()
    _artifact(tmp_path, date(2026, 7, 31), 2)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is False
    assert "expiries" in reason


def test_missing_artifact_for_an_admitted_date_fails_closed(tmp_path):
    """The failure G1 exists to catch: the manifest admits it, disk lacks it."""
    g1 = _load()
    tmp_path.mkdir(parents=True, exist_ok=True)
    ok, reason = g1.verify_surface(date(2026, 7, 31), tmp_path)
    assert ok is False
    assert "no artifact" in reason


def test_scan_never_walks_the_directory(tmp_path, monkeypatch):
    """768 artifacts on disk, 766 admitted — the two extra must not be scanned.

    This is the regression that the first draft of this gate would have hit:
    a glob over iv_surface/ picks up the excluded thin surfaces, fails them on
    the 3-expiry rule, and halts Phase A on a false alarm.
    """
    g1 = _load()
    admitted = date(2026, 7, 30)
    excluded = date(2026, 7, 31)
    _artifact(tmp_path, admitted, 3)
    _artifact(tmp_path, excluded, 2)      # on disk, but NOT admitted
    monkeypatch.setattr(g1.cohort, "admitted_dates", lambda *a, **k: [admitted])

    summary = g1.scan_cohort(iv_dir=tmp_path)
    assert summary["n_admitted"] == 1
    assert summary["n_verified"] == 1
    assert summary["failures"] == []
    assert summary["min_expiries_seen"] == 3
