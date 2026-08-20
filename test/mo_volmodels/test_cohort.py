import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Three tests below pin regression counts against the REAL surface and spot
#: history. That cache is unversioned market data kept outside git, so those
#: tests are evidence wherever it exists and simply unrunnable everywhere
#: else -- CI included. They skip rather than fail, which is the honest
#: reading: absent data is not a broken engine.
HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"
requires_real_history = pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").exists()
    or not (HISTORY_DIR / "csi1000_spot.csv").exists(),
    reason="needs the unversioned mo_volmodels history cache",
)
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


@requires_real_history
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


def _load_stage12_for_cohort():
    path = PROJECT_ROOT / "example/mo_volmodels/12_snowball_volmodel_backtest.py"
    spec = importlib.util.spec_from_file_location("s12_cohort", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["s12_cohort"] = module
    spec.loader.exec_module(module)
    return module


@requires_real_history
def test_data_end_pin_governs_the_inception_count():
    """The pin must reach schedule_inceptions, not only the task windows.

    prepare_inceptions used to re-derive data_end from the spot CSV, so a pin
    that only touched run_fleet left the fleet size floating -- invisible while
    the pin and the cache end coincide, and wrong the next weekday.
    """
    import pandas as pd
    s12 = _load_stage12_for_cohort()
    history_dir = PROJECT_ROOT / "example/mo_volmodels/data/history"
    spot = pd.read_csv(history_dir / "csi1000_spot.csv")
    calendar = s12.stage11().TradingCalendar.from_spot_csv(
        history_dir / "csi1000_spot.csv"
    )
    kwargs = dict(
        calendar=calendar,
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        first_admitted_surface=date(2023, 5, 4),
    )
    assert len(s12.schedule_inceptions(data_end=date(2026, 7, 31), **kwargs)) == 27
    assert len(s12.schedule_inceptions(data_end=date(2026, 6, 30), **kwargs)) == 26


@requires_real_history
def test_prepare_inceptions_forwards_the_data_end_pin(monkeypatch):
    """prepare_inceptions must schedule against the pinned data_end, not the
    spot cache's last row -- this is the exact wiring the pin exists for.

    Pins to 2024-06-01, which admits only the first two monthly inceptions
    (2023-05-04, 2023-06-01) under the 12-month observable-horizon default --
    a value the un-pinned default (2026-07-31, 27 inceptions) could never
    produce by accident, so a pass here is real evidence of forwarding.

    ``solve_fair_coupon`` is stubbed out: each real solve is ~3 full PDE
    prices (~14s each, per its own docstring), which would make this
    regression test take a minute-plus for a fact that has nothing to do
    with coupon solving. Everything upstream of the solve -- scheduling,
    surface lookup, pricing-env construction -- still runs for real; only
    the root-find over the coupon is replaced with a fixed answer.
    """
    s12 = _load_stage12_for_cohort()
    history_dir = PROJECT_ROOT / "example/mo_volmodels/data/history"
    history = s12.surface_history(history_dir)
    spot = s12.load_spot_frame(history_dir)
    futures = s12.load_futures_frame(history_dir)
    calendar = s12.stage11().TradingCalendar.from_spot_csv(
        history_dir / "csi1000_spot.csv"
    )
    rate = s12.stage11().FLAT_RATE
    pinned_end = date(2024, 6, 1)

    def _stub_solve_fair_coupon(**kwargs):
        return s12.CouponSolution(
            coupon=0.2, pv=0.0, iterations=1,
            bracket_low=0.0, bracket_high=0.4, pv_tolerance=1.0, solved=True,
        )

    monkeypatch.setattr(s12, "solve_fair_coupon", _stub_solve_fair_coupon)

    prepared = s12.prepare_inceptions(
        history=history,
        spot=spot,
        futures=futures,
        calendar=calendar,
        rate=rate,
        notional=1.0,
        min_observable_months=12,
        data_end=pinned_end,
    )
    assert [p["inception"] for p in prepared] == ["2023-05-04", "2023-06-01"]
