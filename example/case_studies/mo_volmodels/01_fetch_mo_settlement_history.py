"""Freeze official CFFEX end-of-day MO settlement cross sections.

This is the network boundary for the MO cross-date study.  It reads the official
CFFEX daily statistics CSV either from ``--input-dir`` (the reproducible/offline
path) or from CFFEX, preserves both close and settlement prices, and writes one
``mo_settlement_snapshot_YYYYMMDD.json`` artifact per requested trade date.

Examples::

    /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_settlement_history.py \
        --dates 20260430 20260515 --input-dir /tmp/cffex-mo-history

    /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_settlement_history.py \
        --dates 20260720

The saved source is an exchange settlement cross section, not executable bid/ask
history.  Downstream code must not mix it with the intraday Sina midpoint snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import urllib.request
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_CLASS = "official_cffex_eod_settlement"
PRICE_FIELD = "settlement"
SOURCE_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{yyyymm}/{dd}/{trade_date}_1.csv"
# CFFEX rolls an option expiry that falls on a statutory holiday to the next
# trading day.  The frozen cohort currently needs this one published exception:
# 2026-06-19 through 2026-06-21 were closed for the Dragon Boat holiday.
# Product rule: https://www.cffex.com.cn/zz1000gzqq/
EXPIRY_DATE_OVERRIDES = {"2606": date(2026, 6, 22)}
CONTRACT_RE = re.compile(
    r"^MO(?P<year_month>\d{4})-(?P<option_type>[CP])-(?P<strike>\d+(?:\.\d+)?)$"
)

COL_SYMBOL = "合约代码"
COL_CLOSE = "今收盘"
COL_SETTLEMENT = "今结算"
COL_VOLUME = "成交量"
COL_OPEN_INTEREST = "持仓量"
COL_DELTA = "Delta"
REQUIRED_COLUMNS = {
    COL_SYMBOL,
    COL_CLOSE,
    COL_SETTLEMENT,
    COL_VOLUME,
    COL_OPEN_INTEREST,
}


def _parse_trade_date(value: str) -> date:
    """Parse the canonical YYYYMMDD tag used by CFFEX archives."""
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"trade date must be YYYYMMDD, got {value!r}")
    parsed = datetime.strptime(value, "%Y%m%d").date()
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"invalid trade date {value!r}")
    return parsed


def _third_friday(year_month: str) -> date:
    """Return the holiday-adjusted CFFEX expiry for a YYMM contract code."""
    if not re.fullmatch(r"\d{4}", year_month):
        raise ValueError(f"expiry month must be YYMM, got {year_month!r}")
    year = 2000 + int(year_month[:2])
    month = int(year_month[2:])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid expiry month {year_month!r}")
    cursor = date(year, month, 1)
    fridays: list[date] = []
    while cursor.month == month:
        if cursor.weekday() == 4:
            fridays.append(cursor)
        cursor += timedelta(days=1)
    return EXPIRY_DATE_OVERRIDES.get(year_month, fridays[2])


def _optional_float(value: Any, *, field: str, contract: str) -> float | None:
    """Parse one optional finite numeric CSV cell."""
    if value is None or str(value).strip() in {"", "--", "-"}:
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(f"{contract}: invalid {field} value {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{contract}: non-finite {field} value {value!r}")
    return parsed


def _non_negative_int(value: Any, *, field: str, contract: str) -> int:
    parsed = _optional_float(value, field=field, contract=contract)
    if parsed is None:
        return 0
    if parsed < 0.0 or not parsed.is_integer():
        raise ValueError(f"{contract}: {field} must be a non-negative integer")
    return int(parsed)


def official_url(trade_date: str) -> str:
    """Return the stable official daily-statistics URL for ``trade_date``."""
    _parse_trade_date(trade_date)
    return SOURCE_URL.format(
        yyyymm=trade_date[:6], dd=trade_date[6:], trade_date=trade_date
    )


def parse_cffex_csv(
    payload: bytes,
    trade_date: str,
    *,
    source_url: str | None = None,
) -> dict:
    """Parse one official GB18030 CFFEX CSV into a validated MO snapshot.

    Non-MO rows and CFFEX subtotal rows are ignored by the anchored contract
    regex.  Valid MO contracts are retained even when volume is zero; liquidity
    filtering belongs to the diagnostics stage and must be auditable there.
    """
    valuation_date = _parse_trade_date(trade_date)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("CFFEX CSV payload must be non-empty bytes")
    try:
        text = payload.decode("gb18030")
    except UnicodeDecodeError as exc:
        raise ValueError("CFFEX CSV is not valid GB18030 text") from exc

    reader = csv.DictReader(StringIO(text.lstrip("\ufeff")))
    columns = set(reader.fieldnames or ())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"CFFEX CSV missing required columns: {sorted(missing)}")

    by_expiry: dict[str, list[dict]] = {}
    ignored_rows = 0
    seen_contracts: set[str] = set()
    for row in reader:
        contract = str(row.get(COL_SYMBOL, "")).strip()
        match = CONTRACT_RE.fullmatch(contract)
        if match is None:
            ignored_rows += 1
            continue
        if contract in seen_contracts:
            raise ValueError(f"duplicate MO contract row {contract}")
        seen_contracts.add(contract)
        strike = float(match.group("strike"))
        if not math.isfinite(strike) or strike <= 0.0:
            raise ValueError(f"{contract}: strike must be finite and positive")
        year_month = match.group("year_month")
        quote = {
            "contract": contract,
            "type": match.group("option_type"),
            "strike": strike,
            "close": _optional_float(row.get(COL_CLOSE), field="close", contract=contract),
            "settlement": _optional_float(
                row.get(COL_SETTLEMENT), field="settlement", contract=contract
            ),
            "volume": _non_negative_int(
                row.get(COL_VOLUME), field="volume", contract=contract
            ),
            "oi": _non_negative_int(
                row.get(COL_OPEN_INTEREST), field="open interest", contract=contract
            ),
        }
        if COL_DELTA in columns:
            quote["exchange_delta"] = _optional_float(
                row.get(COL_DELTA), field="Delta", contract=contract
            )
        by_expiry.setdefault(year_month, []).append(quote)

    if not by_expiry:
        raise ValueError(f"CFFEX CSV for {trade_date} contains no MO option contracts")

    expiries = []
    for year_month in sorted(by_expiry):
        expiry_date = _third_friday(year_month)
        quotes = sorted(
            by_expiry[year_month], key=lambda quote: (quote["strike"], quote["type"])
        )
        expiries.append(
            {
                "contract_month": year_month,
                "expiry_date": expiry_date.isoformat(),
                "calendar_days": (expiry_date - valuation_date).days,
                "T_years": (expiry_date - valuation_date).days / 365.0,
                "quotes": quotes,
            }
        )

    url = source_url or official_url(trade_date)
    return {
        "schema_version": 1,
        "trade_date": valuation_date.isoformat(),
        "source_class": SOURCE_CLASS,
        "source_url": url,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "price_field": PRICE_FIELD,
        "underlying": {"code": "000852.SH", "option_product": "MO"},
        "expiry_calendar": {
            "rule": "third_Friday_rolled_to_next_trading_day",
            "frozen_overrides": {
                key: value.isoformat() for key, value in EXPIRY_DATE_OVERRIDES.items()
            },
        },
        "record_count": len(seen_contracts),
        "ignored_row_count": ignored_rows,
        "expiries": expiries,
        "limitations": [
            "Official CFFEX end-of-day settlement cross section; not executable bid/ask history.",
            "Close and settlement are retained separately; calibration uses settlement only.",
            "Expiry dates use third Friday plus the frozen official holiday overrides recorded in expiry_calendar.",
        ],
    }


def _download(trade_date: str) -> tuple[bytes, str]:
    """Download one official CSV; callers remain responsible for freezing it."""
    url = official_url(trade_date)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = response.read()
    return payload, url


def write_snapshot(snapshot: dict, output_dir: Path) -> Path:
    """Write a deterministic tagged snapshot JSON."""
    trade_date = str(snapshot["trade_date"]).replace("-", "")
    _parse_trade_date(trade_date)
    serialized = json.dumps(
        snapshot,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"mo_settlement_snapshot_{trade_date}.json"
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="CFFEX dates in YYYYMMDD")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="offline directory containing YYYYMMDD_1.csv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "data", help="snapshot output directory"
    )
    args = parser.parse_args()

    if len(set(args.dates)) != len(args.dates):
        raise SystemExit("--dates contains duplicates")
    for trade_date in args.dates:
        _parse_trade_date(trade_date)
        if args.input_dir is None:
            payload, url = _download(trade_date)
        else:
            source_path = args.input_dir / f"{trade_date}_1.csv"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            payload = source_path.read_bytes()
            url = official_url(trade_date)
        snapshot = parse_cffex_csv(payload, trade_date, source_url=url)
        path = write_snapshot(snapshot, args.output_dir)
        print(
            f"{trade_date}: {snapshot['record_count']} MO rows, "
            f"{len(snapshot['expiries'])} expiries -> {path}"
        )


if __name__ == "__main__":
    main()
