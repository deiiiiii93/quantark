"""Refresh the CSI1000 spot and IM futures caches for the MO cross-date study.

This is the market-data network boundary for the multi-year snowball backtest
data build.  It seeds from the frozen case-study caches
(``output/case_study/ppp_dki_snowball_backtest_20260430/cache``), extends them
via AKShare to the latest available trading day, and writes the refreshed
caches to ``example/mo_volmodels/data/history`` without touching the
case-study output directory.

The fetch/normalize logic mirrors
``example/ppp_dki_snowball_backtest_case_study.py`` (same schemas, column
names, and date formats as the seeded CSVs: ``date,spot`` and
``date,contract,futures_price,expiry_date,multiplier``).  That module cannot
be imported here because it imports ``quantark`` at module level and the
AKShare interpreter (``/opt/anaconda3/bin/python``) does not have ``quantark``
installed.  One deliberate divergence: the case study raises when a fetch
window yields no IM futures rows at all, while here an empty in-window result
simply means "nothing new published yet" and the seeded cache is kept.

The script is idempotent: the fetch window starts the day after the newest
seeded date, so re-running only fetches genuinely new dates and merges them
with ``keep="last"`` deduplication (no duplicate dates).  On re-run the
previously refreshed output is used as the seed, so the tail extends
incrementally.  ``--start`` sets the lower bound of the output window
(default: first date of the seeded cache) and never triggers a re-download of
history the seed already covers.  ``--end`` bounds the FETCH window only:
seeded rows after ``--end`` are preserved in the output, never dropped.

Fail-closed policy:

- If ANY in-window IM contract month fails to fetch, the script prints
  ``[error]`` naming the failed contract months and exits nonzero — a skipped
  month would leave a permanent hole because later incremental runs start
  after the cache max date and never backfill.  The single exception: a
  contract whose month is after the current month may genuinely not be served
  yet (e.g. a quarterly that has not been listed, like IM2610 in July 2026)
  and degrades to a ``[warn]`` skip instead.
- If the merged spot OR futures frame is empty, the script prints ``[error]``
  and exits nonzero BEFORE any write, so a bad run can never poison the
  caches with an empty file.
- Both output CSVs are written atomically (temp file, fsync, ``os.replace``),
  so a crash cannot leave a torn cache behind.

Must run under the AKShare interpreter::

    /opt/anaconda3/bin/python example/mo_volmodels/01_refresh_market_cache.py
    /opt/anaconda3/bin/python example/mo_volmodels/01_refresh_market_cache.py \
        --end 2026-07-22
"""

from __future__ import annotations

import argparse
import os
import tempfile
from calendar import FRIDAY, monthcalendar
from datetime import date
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE_STUDY_CACHE = (
    ROOT
    / "output"
    / "case_study"
    / "ppp_dki_snowball_backtest_20260430"
    / "cache"
)
DEFAULT_OUTPUT_DIR = HERE / "data" / "history"
SPOT_FILENAME = "csi1000_spot.csv"
FUTURES_FILENAME = "im_futures.csv"

# Mirrored from example/ppp_dki_snowball_backtest_case_study.py.
UNDERLYING_SYMBOL = "000852"
FUTURES_PREFIX = "IM"
FUTURES_MULTIPLIER = 200.0
FUTURES_COLUMNS = ["date", "contract", "futures_price", "expiry_date", "multiplier"]


class RefreshError(Exception):
    """Raised when the market cache refresh cannot proceed."""


def load_akshare():
    """Import AKShare lazily so the module stays importable without it."""
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RefreshError(
            "AKShare is not installed. Run with /opt/anaconda3/bin/python."
        ) from exc
    return ak


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RefreshError(f"Missing {label} column. Available columns: {list(df.columns)}")


def normalize_index_spot(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_column(raw, ["date", "日期"], "index date")
    close_col = _pick_column(raw, ["close", "收盘", "收盘价"], "index close")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]).dt.normalize(),
            "spot": pd.to_numeric(raw[close_col], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates("date", keep="last")
    if out.empty:
        raise RefreshError("Index data is empty after normalization")
    return out.reset_index(drop=True)


def fetch_csi1000_spot(ak) -> pd.DataFrame:
    try:
        raw = ak.stock_zh_index_daily(symbol=f"sh{UNDERLYING_SYMBOL}")
        return normalize_index_spot(raw)
    except Exception as exc:
        print(
            f"[warn] stock_zh_index_daily failed ({exc}); "
            "falling back to index_zh_a_hist"
        )
        raw = ak.index_zh_a_hist(symbol=UNDERLYING_SYMBOL, period="daily")
        return normalize_index_spot(raw)


def third_friday(year: int, month: int) -> pd.Timestamp:
    fridays = [week[FRIDAY] for week in monthcalendar(year, month) if week[FRIDAY] != 0]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


def add_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1)


def _contract_year_month(contract: str) -> tuple[int, int]:
    """Parse and validate the (year, month) coded into an IM contract symbol."""
    suffix = str(contract).replace(FUTURES_PREFIX, "")
    if len(suffix) != 4 or not suffix.isdigit():
        raise RefreshError(f"Cannot infer expiry from futures contract {contract!r}")
    year = 2000 + int(suffix[:2])
    month = int(suffix[2:])
    if not 1 <= month <= 12:
        raise RefreshError(f"invalid contract month in futures contract {contract!r}")
    return year, month


def contract_expiry(contract: str) -> pd.Timestamp:
    year, month = _contract_year_month(contract)
    return third_friday(year, month)


def futures_contract_symbols(
    start_date: pd.Timestamp, end_date: pd.Timestamp, extra_months: int = 3
) -> list[str]:
    cursor = pd.Timestamp(start_date).normalize().replace(day=1)
    end = add_months(pd.Timestamp(end_date).normalize().replace(day=1), extra_months)
    symbols = []
    while cursor <= end:
        symbols.append(f"{FUTURES_PREFIX}{cursor:%y%m}")
        cursor = add_months(cursor, 1)
    return symbols


def normalize_im_futures(raw: pd.DataFrame, contract: str) -> pd.DataFrame:
    date_col = _pick_column(raw, ["date", "日期"], "futures date")
    close_col = _pick_column(raw, ["close", "收盘", "收盘价"], "futures close")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]).dt.normalize(),
            "contract": contract,
            "futures_price": pd.to_numeric(raw[close_col], errors="coerce"),
            "expiry_date": contract_expiry(contract),
            "multiplier": FUTURES_MULTIPLIER,
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates(
        ["date", "contract"], keep="last"
    )
    return out[out["futures_price"] > 0].reset_index(drop=True)


def fetch_im_futures(
    ak, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Fetch every contract month in the window; fail closed on real holes.

    A contract month that fails to fetch is fatal when its month is not after
    the current month: such a contract is (or was) trading, so skipping it
    would leave a permanent hole that later incremental runs never backfill.
    Months after the current month may genuinely not be served yet (e.g. a
    quarterly that has not been listed) and remain a skippable ``[warn]``.
    Returns an empty frame (with the canonical columns) when no contract has
    rows inside the window, e.g. when today's data is not published yet.
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    current_month = pd.Timestamp(date.today()).normalize().replace(day=1)
    for contract in futures_contract_symbols(start_date, end_date):
        try:
            raw = ak.futures_zh_daily_sina(symbol=contract)
            frame = normalize_im_futures(raw, contract)
        except Exception as exc:
            year, month = _contract_year_month(contract)
            if pd.Timestamp(year=year, month=month, day=1) > current_month:
                print(f"[warn] Skipping not-yet-served future contract {contract}: {exc}")
            else:
                print(f"[error] Failed to fetch in-window contract {contract}: {exc}")
                failed.append(contract)
            continue
        mask = (frame["date"] >= start_date) & (frame["date"] <= end_date)
        if mask.any():
            frames.append(frame.loc[mask])
    if failed:
        raise RefreshError(
            "failed to fetch in-window contract months: "
            + ", ".join(failed)
            + "; refusing to extend the cache with a hole, rerun to retry"
        )
    if not frames:
        return pd.DataFrame(columns=FUTURES_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "expiry_date", "contract"]
    )


def read_seed_cache(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Read the seed caches, preferring a previous refresh over the case study."""
    refreshed_spot = output_dir / SPOT_FILENAME
    refreshed_futures = output_dir / FUTURES_FILENAME
    if refreshed_spot.is_file() and refreshed_futures.is_file():
        seed_dir = output_dir
    else:
        seed_dir = CASE_STUDY_CACHE
    spot_path = seed_dir / SPOT_FILENAME
    futures_path = seed_dir / FUTURES_FILENAME
    if not spot_path.is_file() or not futures_path.is_file():
        raise RefreshError(f"Seed cache is missing under {seed_dir}")
    spot = pd.read_csv(spot_path, parse_dates=["date"])
    futures = pd.read_csv(futures_path, parse_dates=["date", "expiry_date"])
    if spot.empty or futures.empty:
        raise RefreshError(f"Seed cache under {seed_dir} is empty")
    return spot, futures, seed_dir


def plan_fetch_from(start: pd.Timestamp, seed_max: pd.Timestamp) -> pd.Timestamp:
    """Return the first date that must come from the source, not the seed.

    Dates up to ``seed_max`` are copied from the seed; anything newer (or the
    whole window when ``start`` is past ``seed_max``) is fetched via AKShare.
    """
    if seed_max >= start:
        return seed_max + pd.Timedelta(days=1)
    return start


def extend_frame(
    seed: pd.DataFrame,
    fetched: pd.DataFrame,
    key_cols: list[str],
    sort_cols: list[str],
    start: pd.Timestamp,
    fetch_from: pd.Timestamp,
) -> pd.DataFrame:
    """Keep seeded rows in [start, fetch_from) and append fetched rows after it.

    ``fetched`` is appended after the kept seed rows and deduplicated with
    ``keep="last"``, so re-runs can never produce duplicate dates and fresh
    values win wherever both frames cover a date.
    """
    kept = seed.loc[(seed["date"] >= start) & (seed["date"] < fetch_from)]
    merged = pd.concat([kept, fetched], ignore_index=True)
    merged = merged.drop_duplicates(key_cols, keep="last")
    return merged.sort_values(sort_cols).reset_index(drop=True)


def parse_iso_date(value: str, *, label: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).normalize()
    except ValueError as exc:
        raise RefreshError(f"invalid {label} date {value!r}; expected YYYY-MM-DD") from exc


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temp file, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        help="lower bound of the output window, YYYY-MM-DD "
        "(default: first date of the seeded cache)",
    )
    parser.add_argument(
        "--end",
        help="last date to fetch, YYYY-MM-DD (default: today); "
        "seeded rows after --end are preserved in the output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"cache output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    seed_spot, seed_futures, seed_dir = read_seed_cache(args.output_dir)
    seed_spot_max = pd.Timestamp(seed_spot["date"].max()).normalize()
    seed_futures_max = pd.Timestamp(seed_futures["date"].max()).normalize()

    start = (
        parse_iso_date(args.start, label="--start")
        if args.start is not None
        else pd.Timestamp(seed_spot["date"].min()).normalize()
    )
    end = (
        parse_iso_date(args.end, label="--end")
        if args.end is not None
        else pd.Timestamp(date.today())
    )
    if start > end:
        raise SystemExit(f"--start {start.date()} is after --end {end.date()}")

    print(
        f"seed: {seed_dir} | spot {seed_spot['date'].min().date()}..{seed_spot_max.date()} "
        f"({len(seed_spot)} rows) | futures {seed_futures['date'].min().date()}.."
        f"{seed_futures_max.date()} ({len(seed_futures)} rows)"
    )

    ak = load_akshare()

    # Only dates not already covered by the seed are fetched; seeded history is
    # copied, never re-downloaded.
    spot_from = plan_fetch_from(start, seed_spot_max)
    if spot_from <= end:
        print(f"fetching CSI1000 spot {spot_from.date()}..{end.date()} via AKShare")
        fetched_spot = fetch_csi1000_spot(ak)
        fetched_spot = fetched_spot.loc[
            (fetched_spot["date"] >= spot_from) & (fetched_spot["date"] <= end)
        ]
    else:
        print("spot cache already covers the requested window; nothing to fetch")
        fetched_spot = seed_spot.iloc[0:0]
    spot = extend_frame(
        seed_spot, fetched_spot, ["date"], ["date"], start, spot_from
    )

    futures_from = plan_fetch_from(start, seed_futures_max)
    if futures_from <= end:
        print(
            f"fetching IM futures {futures_from.date()}..{end.date()} via AKShare "
            f"({len(futures_contract_symbols(futures_from, end))} contract months)"
        )
        try:
            fetched_futures = fetch_im_futures(ak, futures_from, end)
        except RefreshError as exc:
            print(f"[error] {exc}")
            raise SystemExit(1) from exc
        if fetched_futures.empty:
            print("no new in-window IM futures rows; cache tail unchanged")
            fetched_futures = seed_futures.iloc[0:0]
    else:
        print("futures cache already covers the requested window; nothing to fetch")
        fetched_futures = seed_futures.iloc[0:0]
    futures = extend_frame(
        seed_futures,
        fetched_futures,
        ["date", "contract"],
        ["date", "expiry_date", "contract"],
        start,
        futures_from,
    )

    if spot.empty or futures.empty:
        print(
            "[error] merged cache frame is empty "
            f"(spot rows={len(spot)}, futures rows={len(futures)}); "
            "refusing to overwrite the caches"
        )
        raise SystemExit(1)

    spot_path = args.output_dir / SPOT_FILENAME
    futures_path = args.output_dir / FUTURES_FILENAME
    _atomic_write_bytes(spot_path, spot.to_csv(index=False).encode("utf-8"))
    _atomic_write_bytes(futures_path, futures.to_csv(index=False).encode("utf-8"))

    print(
        f"wrote {spot_path}: {spot['date'].min().date()}..{spot['date'].max().date()} "
        f"({len(spot)} rows, +{len(fetched_spot)} fetched)"
    )
    print(
        f"wrote {futures_path}: {futures['date'].min().date()}..{futures['date'].max().date()} "
        f"({len(futures)} rows, +{len(fetched_futures)} fetched)"
    )


if __name__ == "__main__":
    main()
