"""Contracts for the market cache refresh tooling (offline, no akshare)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))


def _load_numbered(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh = _load_numbered("01_refresh_market_cache.py", "mo_refresh_market_cache_contract")


def _spot_frame(pairs: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([day for day, _ in pairs]),
            "spot": [value for _, value in pairs],
        }
    )


def test_extend_frame_overlap_day_keeps_last_without_duplicates() -> None:
    seed = _spot_frame([("2026-07-20", 1.0), ("2026-07-21", 2.0), ("2026-07-22", 3.0)])
    fetched = _spot_frame([("2026-07-22", 99.0), ("2026-07-23", 4.0)])

    merged = refresh.extend_frame(
        seed,
        fetched,
        ["date"],
        ["date"],
        pd.Timestamp("2026-07-20"),
        pd.Timestamp("2026-07-22"),
    )

    assert [str(day.date()) for day in merged["date"]] == [
        "2026-07-20",
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
    ]
    assert list(merged["spot"]) == [1.0, 2.0, 99.0, 4.0]
    assert not merged["date"].duplicated().any()
    assert merged["date"].is_monotonic_increasing


def test_extend_frame_with_empty_seed_returns_fetched_sorted() -> None:
    seed = _spot_frame([])
    fetched = _spot_frame([("2026-07-23", 4.0), ("2026-07-22", 3.0)])

    merged = refresh.extend_frame(
        seed,
        fetched,
        ["date"],
        ["date"],
        pd.Timestamp("2026-07-22"),
        pd.Timestamp("2026-07-22"),
    )

    assert list(merged["spot"]) == [3.0, 4.0]
    assert merged["date"].is_monotonic_increasing


def test_plan_fetch_from_covers_incremental_cold_start_and_empty_seed() -> None:
    seed_max = pd.Timestamp("2026-04-29")
    # incremental: the day after the newest seeded date
    assert refresh.plan_fetch_from(pd.Timestamp("2023-05-04"), seed_max) == pd.Timestamp(
        "2026-04-30"
    )
    assert refresh.plan_fetch_from(pd.Timestamp("2026-04-29"), seed_max) == pd.Timestamp(
        "2026-04-30"
    )
    # cold start: --start past the seeded range fetches the whole window
    assert refresh.plan_fetch_from(pd.Timestamp("2026-05-04"), seed_max) == pd.Timestamp(
        "2026-05-04"
    )
    # empty seed (no max date): everything must come from the source
    assert refresh.plan_fetch_from(pd.Timestamp("2023-05-04"), pd.NaT) == pd.Timestamp(
        "2023-05-04"
    )


def test_third_friday_and_contract_expiry_on_known_months() -> None:
    assert refresh.third_friday(2026, 7) == pd.Timestamp("2026-07-17")
    assert refresh.contract_expiry("IM2607") == pd.Timestamp("2026-07-17")
    # 2026-06-19 is the raw third Friday; unlike the MO fetcher, the case-study
    # futures convention does NOT roll the Dragon Boat holiday, and the frozen
    # case-study cache CSVs record the IM2606 expiry as 2026-06-19.
    assert refresh.contract_expiry("IM2606") == pd.Timestamp("2026-06-19")


def test_contract_expiry_rejects_invalid_month_and_symbol() -> None:
    with pytest.raises(refresh.RefreshError, match="invalid contract month"):
        refresh.contract_expiry("IM2513")
    with pytest.raises(refresh.RefreshError, match="Cannot infer expiry"):
        refresh.contract_expiry("IM256")


def test_futures_contract_symbols_enumerates_across_year_boundary() -> None:
    symbols = refresh.futures_contract_symbols(
        pd.Timestamp("2025-11-15"), pd.Timestamp("2026-02-10")
    )
    assert symbols == [
        "IM2511",
        "IM2512",
        "IM2601",
        "IM2602",
        "IM2603",
        "IM2604",
        "IM2605",
    ]


def test_fetch_im_futures_fails_closed_on_current_month_but_warns_on_future(
    capsys,
) -> None:
    today = pd.Timestamp(date.today()).normalize()
    current = f"IM{today:%y%m}"
    future = f"IM{refresh.add_months(today, 2):%y%m}"

    class FakeAk:
        def futures_zh_daily_sina(self, symbol):
            if symbol in {current, future}:
                raise RuntimeError("sina unreachable")
            return pd.DataFrame(
                {"date": [today.strftime("%Y-%m-%d")], "close": [7000.0]}
            )

    with pytest.raises(refresh.RefreshError) as excinfo:
        refresh.fetch_im_futures(FakeAk(), today, today)
    message = str(excinfo.value)
    assert current in message
    assert future not in message
    assert "[warn] Skipping not-yet-served future contract" in capsys.readouterr().out


def test_fetch_im_futures_returns_empty_frame_when_window_has_no_rows() -> None:
    today = pd.Timestamp(date.today()).normalize()

    class FakeAk:
        def futures_zh_daily_sina(self, symbol):
            return pd.DataFrame({"date": ["2020-01-02"], "close": [4000.0]})

    frame = refresh.fetch_im_futures(FakeAk(), today, today)

    assert frame.empty
    assert list(frame.columns) == refresh.FUTURES_COLUMNS


def test_main_refuses_to_write_when_merged_frames_are_empty(
    tmp_path, monkeypatch, capsys
) -> None:
    spot_csv = tmp_path / "csi1000_spot.csv"
    futures_csv = tmp_path / "im_futures.csv"
    spot_csv.write_text(
        "date,spot\n2026-07-21,7259.863\n2026-07-22,7158.762\n", encoding="utf-8"
    )
    futures_csv.write_text(
        "date,contract,futures_price,expiry_date,multiplier\n"
        "2026-07-22,IM2608,7091.2,2026-08-21,200.0\n",
        encoding="utf-8",
    )
    before_spot = spot_csv.read_bytes()
    before_futures = futures_csv.read_bytes()

    monkeypatch.setattr(refresh, "load_akshare", lambda: object())
    empty_spot = pd.DataFrame(
        {"date": pd.to_datetime([]), "spot": pd.Series(dtype=float)}
    )
    monkeypatch.setattr(refresh, "fetch_csi1000_spot", lambda ak: empty_spot)
    monkeypatch.setattr(
        refresh,
        "fetch_im_futures",
        lambda ak, start, end: pd.DataFrame(columns=refresh.FUTURES_COLUMNS),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "01_refresh_market_cache.py",
            "--output-dir",
            str(tmp_path),
            "--start",
            "2026-08-03",
            "--end",
            "2026-08-31",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        refresh.main()

    assert excinfo.value.code != 0
    assert "[error]" in capsys.readouterr().out
    assert spot_csv.read_bytes() == before_spot
    assert futures_csv.read_bytes() == before_futures
