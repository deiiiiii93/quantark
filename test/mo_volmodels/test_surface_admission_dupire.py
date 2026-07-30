"""Every admitted IV surface must be usable by every model the study runs.

Gate G1 admitted surfaces on arbitrage grounds alone, and the Phase-1 builder's
own threshold is ``min_expiries = 2``.  Dupire local volatility needs at least
three maturity pillars to form dC/dT, so two admitted dates (2024-09-30,
2025-04-08) were arbitrage-free yet unpriceable - they killed 20 of 27
``localvol`` runs partway through a fleet replay.  They are now excluded via
``example/mo_volmodels/exclude_thin_surfaces.py`` and covered by the manifest's
carry-forward policy.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "example/mo_volmodels/data/history"
MODULE_PATH = ROOT / "example/mo_volmodels/exclude_thin_surfaces.py"

spec = importlib.util.spec_from_file_location("exclude_thin_surfaces", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

pytestmark = pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists(),
    reason="IV surface history not built",
)


def _manifest():
    return json.loads((HISTORY_DIR / "surface_manifest.json").read_text())


def test_no_admitted_surface_is_too_thin_for_dupire():
    """The invariant the fleet depends on."""
    thin = mod.thin_records(_manifest()["records"])
    assert thin == [], (
        "admitted surfaces that Dupire cannot consume: "
        + ", ".join(f"{r['date']}(n_expiries={r.get('n_expiries')})" for r in thin)
    )


def test_the_two_known_thin_dates_are_excluded_with_a_reason():
    by_date = {r["date"]: r for r in _manifest()["records"]}
    for date in ("20240930", "20250408"):
        rec = by_date[date]
        assert rec["status"] == "excluded"
        assert rec["reason"] == mod.EXCLUSION_REASON
        assert "carry forward" in rec["detail"]


def test_exclusion_is_recorded_in_the_manifest_for_provenance():
    admission = _manifest().get("study_admission", {}).get("vol_model_backtest")
    assert admission is not None, "the tightened criterion must be recorded"
    assert admission["min_expiries"] == mod.MIN_EXPIRIES_FOR_DUPIRE
    assert set(admission["excluded_dates"]) == {"20240930", "20250408"}


def test_history_loader_admits_only_the_usable_surfaces():
    from quantark.backtest.otc.vol_history import VolSurfaceHistory

    history = VolSurfaceHistory(HISTORY_DIR)
    assert len(history.admitted_dates) == 760


def test_carry_forward_covers_an_excluded_date():
    """An excluded date must resolve to the previous admitted surface."""
    from datetime import date

    from quantark.backtest.otc.vol_history import VolSurfaceHistory

    history = VolSurfaceHistory(HISTORY_DIR)
    artifact = history.surface_for(date(2024, 9, 30))
    assert artifact.trade_date < date(2024, 9, 30), "must carry forward, not use the thin surface"
    assert len(artifact.maturities) >= mod.MIN_EXPIRIES_FOR_DUPIRE


def test_apply_exclusions_is_idempotent():
    manifest = _manifest()
    assert mod.apply_exclusions(manifest) == [], "already applied; must be a no-op"


def test_detector_flags_a_synthetic_thin_record():
    records = [
        {"date": "20200101", "status": "ok", "n_expiries": 2},
        {"date": "20200102", "status": "ok", "n_expiries": 5},
        {"date": "20200103", "status": "excluded", "n_expiries": 1},
    ]
    thin = mod.thin_records(records)
    assert [r["date"] for r in thin] == ["20200101"], "only ADMITTED thin records"
