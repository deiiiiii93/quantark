"""Stage 1: extend the CFFEX/spot cache to the latest published day and build the daily chain.

Reads the shared ``mo_volmodels/data/history`` cache (READ-ONLY).  Days after its
last CFFEX file are downloaded to ``OUTPUT_DIR/cffex_extension/``; spot rows after
its last date come from the index dashboard DB (read-only) into
``OUTPUT_DIR/spot_extension.csv``.  Writes ``OUTPUT_DIR/daily_chain.pkl``.
"""

from __future__ import annotations

import csv
import pickle
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2ts import (  # noqa: E402
    CFFEX_EXTENSION_DIR, CFFEX_URL, CHAIN_PATH, DASHBOARD_DB, HISTORY_DIR, INDEX_CODE,
    OUTPUT_DIR, SPOT_EXTENSION_PATH, StudyDataError,
)
from s2ts.chain import build_all_chains, parse_cffex_csv, sse_trading_days  # noqa: E402

SLEEP_S = 0.5


def last_cached_csv_day() -> date:
    names = sorted(p.name for p in (HISTORY_DIR / "settlement_csv").glob("*_1.csv"))
    if not names:
        raise StudyDataError(f"no CFFEX CSVs under {HISTORY_DIR}/settlement_csv")
    return datetime.strptime(names[-1][:8], "%Y%m%d").date()


def fetch(day: date) -> bytes | None:
    """The day's CSV payload, or None on HTTP 404 (not published)."""
    url = CFFEX_URL.format(yyyymm=day.strftime("%Y%m"), dd=day.strftime("%d"), yyyymmdd=day.strftime("%Y%m%d"))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def extend_cffex(today: date) -> date:
    """Download missing trading days up to the latest published; returns that day."""
    CFFEX_EXTENSION_DIR.mkdir(parents=True, exist_ok=True)
    last = last_cached_csv_day()
    cal = sse_trading_days(2023, 2030)
    wanted = [d.date() for d in cal if last < d.date() <= today]
    latest, pending = last, []
    for d in wanted:
        path = CFFEX_EXTENSION_DIR / f"{d:%Y%m%d}_1.csv"
        if path.exists():
            payload = path.read_bytes()
        else:
            payload = fetch(d)
            time.sleep(SLEEP_S)
            if payload is None:
                pending.append(d)
                continue
            parse_cffex_csv(payload)  # fail closed on a non-CSV body
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
        if pending:
            raise StudyDataError(f"CFFEX CSV for trading day {pending[0]} is missing but {d} is published")
        latest = d
    if pending:
        print(f"CFFEX not yet published for {', '.join(map(str, pending))}; range ends at {latest}")
    return latest


def extend_spot(through: date) -> None:
    """Spot rows after the shared cache's last date, up to ``through``, from the dashboard DB."""
    rows = (HISTORY_DIR / "csi1000_spot.csv").read_text().splitlines()[1:]
    last_spot = date.fromisoformat(rows[-1].split(",")[0])
    con = sqlite3.connect(f"file:{DASHBOARD_DB}?mode=ro", uri=True)
    try:
        found = con.execute(
            "SELECT trading_date, spot FROM latest_daily_basis WHERE index_code = ? AND futures_family = 'IM' "
            "AND slot = 1 AND trading_date > ? AND trading_date <= ? ORDER BY trading_date",
            (INDEX_CODE, last_spot.isoformat(), through.isoformat()),
        ).fetchall()
    finally:
        con.close()
    SPOT_EXTENSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SPOT_EXTENSION_PATH.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "spot"])
        w.writerows(found)
    tmp.replace(SPOT_EXTENSION_PATH)
    print(f"spot extension: {len(found)} rows after {last_spot}")


def verify_reaches(chains, through: date) -> None:
    """The chain must end on the latest published CFFEX day; a lagging spot DB would silently drop days."""
    if chains[-1].day != through:
        raise StudyDataError(
            f"daily chain ends {chains[-1].day} but the latest published CFFEX day is {through}: the spot source "
            f"(dashboard DB) lags; update it and rerun 01_data.py")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    through = extend_cffex(today)
    print(f"latest published CFFEX day: {through}")
    extend_spot(through)
    cal = sse_trading_days(2023, 2030)
    chains = build_all_chains(HISTORY_DIR, CFFEX_EXTENSION_DIR, cal, SPOT_EXTENSION_PATH)
    verify_reaches(chains, through)
    tmp = CHAIN_PATH.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(chains, fh)
    tmp.replace(CHAIN_PATH)
    rolls = sum(1 for a, b in zip(chains, chains[1:]) if a.mo.series != b.mo.series)
    switches = sum(1 for a, b in zip(chains, chains[1:])
                   if a.mo.series == b.mo.series and a.mo.strike != b.mo.strike)
    print(f"daily chain: {chains[0].day} .. {chains[-1].day}, {len(chains)} days -> {CHAIN_PATH}")
    print(f"MO-roll days: {rolls}; ATM-switch days (same series): {switches}")


if __name__ == "__main__":
    main()
