"""Shared pieces of the 2Y snowball flat-q vs CarryCurve study.

Question: how much does a single flat dividend yield misprice and mis-hedge a
2Y CSI 1000 snowball, against the term structure the IM futures + MO options
chain actually quotes (2026-11-05)?

Everything the stages share lives here:

- paths, the valuation date and the locked term sheet;
- market-data loading: the index-basis dashboard DB (read-only) and the CFFEX
  daily statistics CSV, frozen into ``data/market_20261105.json``;
- (T4+) term schedule, product, curves per tail convention, environments,
  engines, the coupon solve, and the study-local forward-interval yield bump.

Stage scripts load this module by path under the unique name
``snowball_2y_cc_common`` (several studies own a ``_common.py``).

Spec: ``.ship/2026-11-05-snowball-2y-carry-curve/spec.md`` (run directory, untracked).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from quantark.param.vol.marketquotes import implied_vol_black
from quantark.util.exceptions import MarketDataError
from quantark.util.numerical import safe_exp

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

STUDY_DIR = Path(__file__).resolve().parent
DATA_DIR = STUDY_DIR / "data"
RESULTS_DIR = STUDY_DIR / "results"
SNAPSHOT_PATH = DATA_DIR / "market_20261105.json"

VALUATION_DATE = date(2026, 11, 5)
TRADING_DATE = VALUATION_DATE.isoformat()
INDEX_CODE = "000852.SH"
DEFAULT_DB = Path("/Users/fuxinyao/index-dashboard-codex/data/basis.sqlite3")
CFFEX_URL = "http://www.cffex.com.cn/sj/hqsj/rtj/{yyyymm}/{dd}/{yyyymmdd}_1.csv"

#: ACT/365F year fraction from the valuation date (the study's one clock).
ACT = 365.0
#: Flat continuously-compounded rate, the dashboard's own synthetic-forward rate.
RATE = 0.014
IM_MULTIPLIER = 200.0  # CNY per index point per IM futures lot
MO_MULTIPLIER = 100.0  # CNY per index point per MO option

LISTED_SLOTS = (1, 2, 3, 4)
SYNTHETIC_SLOT = 5

# CFFEX daily statistics CSV (GBK): code, open, high, low, volume, turnover,
# open interest, OI change, close (今收盘), settlement (今结算), …
_CSV_CODE, _CSV_CLOSE, _CSV_SETTLE = 0, 8, 9
_CSV_HEADER_CODE = "合约代码"


class StudyDataError(MarketDataError):
    """A market-data row the study needs is missing or malformed (fail closed)."""


def year_fraction(d: date) -> float:
    """ACT/365F from the valuation date."""
    return (d - VALUATION_DATE).days / ACT


# ---------------------------------------------------------------------------
# Market data: dashboard DB + CFFEX CSV → frozen snapshot
# ---------------------------------------------------------------------------


def parse_cffex_csv(payload: bytes) -> Dict[str, Dict[str, float]]:
    """``{contract code: {"close", "settle"}}`` from a CFFEX daily statistics CSV.

    Rows with a non-numeric close or settlement (``null``/``--``) are skipped;
    a missing header raises ``StudyDataError``.
    """
    text = payload.decode("gbk")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or header[_CSV_CODE].strip() != _CSV_HEADER_CODE:
        raise StudyDataError("not a CFFEX daily statistics CSV (header missing)")
    out: Dict[str, Dict[str, float]] = {}
    for row in reader:
        if len(row) <= _CSV_SETTLE:
            continue
        try:
            close, settle = float(row[_CSV_CLOSE]), float(row[_CSV_SETTLE])
        except ValueError:
            continue
        out[row[_CSV_CODE].strip()] = {"close": close, "settle": settle}
    return out


def read_dashboard_chain(db_path: Path, trading_date: str = TRADING_DATE) -> Dict[str, Any]:
    """Spot, IM T1–T4 and the MO synthetic row of one day, from the dashboard DB.

    Opened read-only (``mode=ro``). Fails closed when any slot is missing or the
    rows disagree on spot.
    """
    uri = f"file:{Path(db_path)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT revision_id, slot, contract_code, futures, expiry_date, spot, content_hash, "
            "call_option_contract, put_option_contract FROM latest_daily_basis "
            "WHERE index_code = ? AND trading_date = ? ORDER BY slot",
            (INDEX_CODE, trading_date),
        ).fetchall()
    finally:
        con.close()
    by_slot = {int(r[1]): r for r in rows}
    missing = [s for s in LISTED_SLOTS + (SYNTHETIC_SLOT,) if s not in by_slot]
    if missing:
        raise StudyDataError(f"dashboard DB has no {INDEX_CODE} rows for slots {missing} on {trading_date}")
    spots = {float(r[5]) for r in by_slot.values()}
    if len(spots) != 1:
        raise StudyDataError(f"rows on {trading_date} disagree on spot: {sorted(spots)}")
    futures = []
    for s in LISTED_SLOTS:
        rid, _, code, price, expiry, _, chash, _, _ = by_slot[s]
        if not str(code).startswith("IM"):
            raise StudyDataError(f"slot {s} is {code}, expected an IM contract")
        futures.append({"contract": code, "price": float(price), "expiry": expiry,
                        "revision_id": int(rid), "content_hash": chash})
    rid, _, code, price, expiry, _, chash, call_c, put_c = by_slot[SYNTHETIC_SLOT]
    if not (call_c and put_c):
        raise StudyDataError(f"slot {SYNTHETIC_SLOT} ({code}) carries no option pair")
    return {
        "trading_date": trading_date,
        "spot": spots.pop(),
        "futures": futures,
        "synthetic": {"contract": code, "price": float(price), "expiry": expiry,
                      "call_contract": call_c, "put_contract": put_c,
                      "revision_id": int(rid), "content_hash": chash},
    }


def _cffex_option_code(dashboard_code: str) -> tuple:
    """``MO2709C6600`` → (``MO2709``, ``C``, 6600.0)."""
    code = dashboard_code.strip()
    for i in range(len(code) - 1, 0, -1):
        if code[i] in "CP" and code[:i][-1].isdigit():
            return code[:i], code[i], float(code[i + 1:])
    raise StudyDataError(f"cannot parse option code {dashboard_code!r}")


def _row(rows: Dict[str, Dict[str, float]], code: str) -> Dict[str, float]:
    if code not in rows:
        raise StudyDataError(f"CFFEX CSV has no row {code}")
    return rows[code]


def build_snapshot(chain: Dict[str, Any], csv_payload: bytes, *, source_url: str) -> Dict[str, Any]:
    """Freeze the study's market data (decisions 1, 2, 10).

    The MO pillar is the dashboard's pair (same strike, same expiry) priced at
    the CFFEX settlement; the flat vol is the Black-76 implied vol of the MO
    settlement pair on the grid strike nearest the pillar forward, with the
    pillar's own forward and ``DF = exp(−r·T)``.
    """
    rows = parse_cffex_csv(csv_payload)
    syn = chain["synthetic"]
    series, _, strike = _cffex_option_code(syn["call_contract"])
    put_series, _, put_strike = _cffex_option_code(syn["put_contract"])
    if (series, strike) != (put_series, put_strike):
        raise StudyDataError(f"dashboard pair {syn['call_contract']}/{syn['put_contract']} is not one strike")
    k_label = f"{strike:g}"
    call = _row(rows, f"{series}-C-{k_label}")
    put = _row(rows, f"{series}-P-{k_label}")
    expiry = date.fromisoformat(syn["expiry"])
    T = year_fraction(expiry)
    df = float(safe_exp(-RATE * T))
    mo_forward = strike + (call["settle"] - put["settle"]) / df

    futures = []
    for f in chain["futures"]:
        cf = _row(rows, f["contract"])
        futures.append({**f, "close_cffex": cf["close"], "settle": cf["settle"]})

    strikes = sorted({float(c.rsplit("-", 1)[1]) for c in rows if c.startswith(f"{series}-C-")})
    k_atm = min(strikes, key=lambda k: abs(k - mo_forward))
    atm_label = f"{k_atm:g}"
    c_atm = _row(rows, f"{series}-C-{atm_label}")["settle"]
    p_atm = _row(rows, f"{series}-P-{atm_label}")["settle"]
    call_iv = implied_vol_black(c_atm, mo_forward, k_atm, T, df, True)
    put_iv = implied_vol_black(p_atm, mo_forward, k_atm, T, df, False)

    return {
        "valuation_date": VALUATION_DATE.isoformat(),
        "index_code": INDEX_CODE,
        "spot": float(chain["spot"]),
        "rate": RATE,
        "futures": futures,
        "option_pair": {
            "series": series, "strike": strike, "expiry": syn["expiry"],
            "call_contract": f"{series}-C-{k_label}", "put_contract": f"{series}-P-{k_label}",
            "call_settle": call["settle"], "put_settle": put["settle"],
            "call_close": call["close"], "put_close": put["close"],
        },
        "mo_forward": mo_forward,
        "vol": {"value": 0.5 * (call_iv + put_iv), "strike": k_atm,
                "call_iv": call_iv, "put_iv": put_iv, "forward": mo_forward,
                "method": "Black-76 on MO settlement, pillar forward, DF=exp(-rT)"},
        "dashboard_syn_fut": {"price": syn["price"], "contract": syn["contract"],
                              "revision_id": syn["revision_id"], "content_hash": syn["content_hash"],
                              "note": "dashboard synthetic forward from live mids, r=1.4%; cross-check only"},
        "cffex_csv": {"url": source_url, "sha256": hashlib.sha256(csv_payload).hexdigest(),
                      "bytes": len(csv_payload)},
        "sources": {
            "spot_and_im": "index-dashboard basis.sqlite3 latest_daily_basis (closes)",
            "mo_pair": "CFFEX daily statistics CSV, settlement (今结算)",
        },
    }


def cffex_url(d: date = VALUATION_DATE) -> str:
    return CFFEX_URL.format(yyyymm=d.strftime("%Y%m"), dd=d.strftime("%d"), yyyymmdd=d.strftime("%Y%m%d"))


def write_json(path: Path, payload: Any) -> None:
    """Deterministic JSON (sorted keys, 2-space indent, no NaN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise StudyDataError(f"{path} missing; run the stage that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot(path: Optional[Path] = None) -> Dict[str, Any]:
    """The frozen market snapshot (``01_snapshot.py`` writes it)."""
    return read_json(SNAPSHOT_PATH if path is None else path)
