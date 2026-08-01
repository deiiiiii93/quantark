import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE = PROJECT_ROOT / "example/mo_volmodels/cohort.py"


def _load():
    spec = importlib.util.spec_from_file_location("mo_cohort", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_cohort"] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, records, study_admission=None) -> Path:
    history = tmp_path / "history"
    history.mkdir()
    payload = {"records": records}
    if study_admission is not None:
        payload["study_admission"] = study_admission
    (history / "surface_manifest.json").write_text(json.dumps(payload))
    return history


def test_asof_is_the_frozen_pin():
    assert _load().COHORT_ASOF == date(2026, 7, 31)


def test_admitted_dates_drop_records_after_the_asof(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20260730", "status": "ok"},
            {"date": "20260731", "status": "ok"},
            {"date": "20260803", "status": "ok"},   # next scheduler tick
        ],
    )
    assert mod.admitted_dates(history) == [date(2026, 7, 30), date(2026, 7, 31)]


def test_admitted_dates_drop_excluded_records(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "excluded",
             "reason": "insufficient_expiries_for_dupire", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
    )
    assert mod.admitted_dates(history) == [date(2024, 10, 8)]


def test_study_admission_exclusions_are_enforced_even_if_status_says_ok(tmp_path):
    """A rebuild resets per-record status but preserves study_admission."""
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "ok", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
        study_admission={
            "vol_model_backtest": {"excluded_dates": ["20240930"], "min_expiries": 3}
        },
    )
    assert mod.admitted_dates(history) == [date(2024, 10, 8)]


def test_excluded_records_are_reported_with_reasons(tmp_path):
    mod = _load()
    history = _write_manifest(
        tmp_path,
        [
            {"date": "20240930", "status": "excluded",
             "reason": "insufficient_expiries_for_dupire", "n_expiries": 2},
            {"date": "20241008", "status": "ok"},
        ],
    )
    assert mod.excluded_records(history) == [
        {"date": date(2024, 9, 30),
         "reason": "insufficient_expiries_for_dupire", "n_expiries": 2}
    ]


def test_real_history_matches_the_pinned_counts():
    """Regression pin: the numbers §7A.12 froze."""
    mod = _load()
    admitted = mod.admitted_dates()
    assert len(admitted) == 766
    assert admitted[0] == date(2023, 5, 4)
    assert admitted[-1] == date(2026, 7, 31)
    assert date(2024, 9, 30) not in admitted
    assert date(2025, 4, 8) not in admitted


import importlib.util as _ilu

SEED_MODULE = PROJECT_ROOT / "example/mo_volmodels/seed_calibration_cache.py"


def _load_seed():
    spec = _ilu.spec_from_file_location("mo_seed", SEED_MODULE)
    module = _ilu.module_from_spec(spec)
    sys.modules["mo_seed"] = module
    spec.loader.exec_module(module)
    return module


def _entry(path: Path, variant: str, key: str, fingerprint: str) -> None:
    (path / f"{variant}-{key}.json").write_text(
        json.dumps({"variant": variant, "config_fingerprint": fingerprint,
                    "surface_date": "2026-07-31", "schema_version": 1})
    )


def test_seed_copies_entries_and_reports_by_variant(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    _entry(src, "localvol", "bbb", "fp2")
    summary = mod.seed(src, dst)
    assert summary["n_source"] == 2
    assert summary["n_copied"] == 2
    assert summary["by_variant"] == {"heston": 1, "localvol": 1}
    assert sorted(summary["fingerprints"]) == ["fp1", "fp2"]
    assert (dst / "heston-aaa.json").is_file()


def test_seed_never_overwrites_an_existing_entry(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    (dst / "heston-aaa.json").write_text('{"mine": true}')
    summary = mod.seed(src, dst)
    assert summary["n_copied"] == 0
    assert summary["n_skipped_existing"] == 1
    assert json.loads((dst / "heston-aaa.json").read_text()) == {"mine": True}


def test_seed_dry_run_writes_nothing(tmp_path):
    mod = _load_seed()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir(); dst.mkdir()
    _entry(src, "heston", "aaa", "fp1")
    summary = mod.seed(src, dst, dry_run=True)
    assert summary["n_copied"] == 1
    assert not any(dst.iterdir())
