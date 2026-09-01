"""Rebuild real historical surfaces and compare against the artifacts on disk.

The 789 admitted MO artifacts are deliberately untracked (``.git/info/exclude``)
and live in the main checkout, so they are invisible from a worktree.  The
history root is therefore resolved from ``QUANTARK_MO_HISTORY``, defaulting to
the main repo's path; without it this test would glob nothing and skip in
silence, quietly disabling the one check that protects the warm calibration
cache and the cohort pins.

Byte equality is a LOCAL property only: CI is x86_64 while these artifacts were
frozen on ARM64, and SABR goes through scipy.optimize.  So the tolerance
comparison always runs, and the byte comparison runs only when
``QUANTARK_VOLCALIB_BYTES=1`` is set on the machine that froze them.
"""
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quantark.volcalibration.normalize.settlement import SettlementNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import serialize_artifact
from quantark.volcalibration.surface import build_artifact

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "example/mo_volmodels/data"
DEFAULT_HISTORY = Path(
    "/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history"
)
HISTORY = Path(os.environ.get("QUANTARK_MO_HISTORY", DEFAULT_HISTORY))
IV_TOL = 5e-9


def _cases():
    """(tag, settlement snapshot, stored artifact) for every locally available date."""
    iv_dir = HISTORY / "iv_surface"
    if not iv_dir.is_dir():
        return []
    out = []
    for snap_path in sorted(SAMPLE_DIR.glob("mo_settlement_snapshot_*.json")):
        tag = snap_path.stem.rsplit("_", 1)[-1]
        art = iv_dir / f"mo_iv_surface_{tag}.json"
        if art.is_file():
            out.append((tag, snap_path, art))
    return out


CASES = _cases()


@pytest.mark.skipif(
    not CASES, reason=f"no MO surface history at {HISTORY} (set QUANTARK_MO_HISTORY)"
)
@pytest.mark.parametrize("tag,snap_path,art_path", CASES)
def test_rebuilt_surface_matches_the_stored_artifact(tag, snap_path, art_path):
    stored = json.loads(art_path.read_text())
    snap = QuoteSnapshot.from_legacy_settlement(
        json.loads(snap_path.read_text()),
        trade_date=date.fromisoformat(stored["trade_date"]),
        spot=float(stored["s0"]),
        symbol="000852.SH",
        source_sha256=stored.get("source_sha256"),
        source_url=stored.get("source_url"),
    )
    rebuilt = build_artifact(SettlementNormalizer().normalize(snap), snap)

    assert rebuilt["strikes"] == stored["strikes"]
    assert rebuilt["maturities"] == pytest.approx(stored["maturities"], rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(rebuilt["iv_grid"]), np.asarray(stored["iv_grid"]), atol=IV_TOL
    )
    assert sorted(rebuilt) == sorted(stored), "artifact key set drifted"

    if os.environ.get("QUANTARK_VOLCALIB_BYTES") == "1":
        assert serialize_artifact(rebuilt) == art_path.read_bytes(), (
            f"{tag}: artifact bytes changed — the calibration cache and cohort "
            "pins key off this sha (spec 5.3)"
        )
