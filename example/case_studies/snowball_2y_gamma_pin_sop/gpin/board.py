"""The full CFFEX board: every IM future and every MO option with open / close / settle / VWAP / volume / open interest.

The tail-stability study's ``s2ts.chain`` keeps one MO series and five strikes; the ledger of this study needs the
whole board (spec C1): an overlay leg is a single listed MO option, tradable only on a day with a settlement,
volume and open interest. Marks: IM close and MO settlement; ``vwap`` = turnover (10k CNY) x 1e4 /
(volume x multiplier), ``None`` at zero volume.

Seller-neutral: this module holds market data only.
"""

from __future__ import annotations

import csv
import io
import os
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantark.util.numerical import is_valid_number, is_zero, safe_exp

from s2ts import ACT, IM_MULT, MO_MULT
from s2ts.chain import DayChain, contract_expiry, sse_trading_days

from . import BOARD_PATH, StudyDataError, load_prereg

#: The CFFEX daily statistics header, exactly; read by NAME, refused when any name differs.
CFFEX_HEADER: Tuple[str, ...] = (
    "合约代码", "今开盘", "最高价", "最低价", "成交量", "成交金额", "持仓量", "持仓变化",
    "今收盘", "今结算", "前结算", "涨跌1", "涨跌2", "Delta",
)

_IM_RE = re.compile(r"^IM\d{4}$")
_MO_OPT_RE = re.compile(r"^(MO\d{4})-([CP])-(\d+(?:\.\d+)?)$")

#: Calendar years kept past the last chain day when resolving contract expiries.
_CALENDAR_PAD_YEARS = 3


@dataclass(frozen=True)
class CffexRow:
    """One parsed contract row; a field that was not numeric is ``None`` (the row needs a settlement)."""

    code: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    settle: float
    volume: int
    turnover: Optional[float]
    open_interest: Optional[int]
    vwap: Optional[float]


@dataclass(frozen=True)
class OptionQuote:
    """One MO option on one day."""

    settle: float
    close: Optional[float]
    open: Optional[float]
    vwap: Optional[float]
    volume: int
    open_interest: Optional[int]


@dataclass(frozen=True)
class ImBar:
    """One IM contract on one day."""

    open: Optional[float]
    close: Optional[float]
    settle: float
    vwap: Optional[float]
    volume: int


@dataclass(frozen=True)
class SeriesInfo:
    """One MO series on one day: ``tenor`` ACT/365 to the last trading day, put-call-parity ``forward``."""

    last_trading_day: date
    tenor: float
    forward: float
    discount: float


@dataclass(frozen=True)
class DayBoard:
    """One day's board. ``mo[series][(cp, strike)]``; ``series`` only for series with a strike carrying both legs."""

    day: date
    spot: float
    im: Dict[str, ImBar]
    mo: Dict[str, Dict[Tuple[str, float], OptionQuote]]
    series: Dict[str, SeriesInfo]


def _number(text: str) -> Optional[float]:
    """The finite float in ``text``, else ``None`` ("null", "--", "")."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if is_valid_number(value) else None


def _count(text: str) -> Optional[int]:
    value = _number(text)
    return None if value is None else int(round(value))


def parse_cffex_full(payload: bytes) -> Dict[str, CffexRow]:
    """``{code: CffexRow}`` for every IM future and MO option row with a numeric settlement.

    Columns are read by header name; the header must be the CFFEX layout exactly (``StudyDataError`` otherwise).
    A non-numeric field becomes ``None``; a row without a numeric settlement is dropped; other product families
    (HO, IO) are ignored.
    """
    reader = csv.reader(io.StringIO(payload.decode("gbk")))
    header = next(reader, None)
    if header is None or tuple(h.strip() for h in header) != CFFEX_HEADER:
        raise StudyDataError(f"not a CFFEX daily statistics CSV in the expected layout: {CFFEX_HEADER}")
    col = {name: i for i, name in enumerate(h.strip() for h in header)}
    out: Dict[str, CffexRow] = {}
    for row in reader:
        if len(row) != len(CFFEX_HEADER):
            continue
        code = row[col["合约代码"]].strip()
        if _IM_RE.match(code):
            multiplier = IM_MULT
        elif _MO_OPT_RE.match(code):
            multiplier = MO_MULT
        else:
            continue
        settle = _number(row[col["今结算"]])
        if settle is None:
            continue
        volume = _count(row[col["成交量"]]) or 0
        turnover = _number(row[col["成交金额"]])
        denominator = volume * multiplier
        vwap = None if turnover is None or is_zero(denominator) else turnover * 1e4 / denominator
        out[code] = CffexRow(
            code=code,
            open=_number(row[col["今开盘"]]),
            high=_number(row[col["最高价"]]),
            low=_number(row[col["最低价"]]),
            close=_number(row[col["今收盘"]]),
            settle=settle,
            volume=volume,
            turnover=turnover,
            open_interest=_count(row[col["持仓量"]]),
            vwap=vwap,
        )
    return out


def build_day_board(day: date, spot: float, rows: Mapping[str, CffexRow], calendar: pd.DatetimeIndex,
                    rate: float) -> DayBoard:
    """One day's board from parsed rows; ``rate`` is the flat discount rate of the pre-registration."""
    im: Dict[str, ImBar] = {}
    legs: Dict[str, Dict[Tuple[str, float], OptionQuote]] = {}
    for code, r in rows.items():
        if _IM_RE.match(code):
            im[code] = ImBar(r.open, r.close, r.settle, r.vwap, r.volume)
            continue
        m = _MO_OPT_RE.match(code)
        if m:
            legs.setdefault(m.group(1), {})[(m.group(2), float(m.group(3)))] = OptionQuote(
                r.settle, r.close, r.open, r.vwap, r.volume, r.open_interest)
    series: Dict[str, SeriesInfo] = {}
    for name, quotes in legs.items():
        strikes = sorted(k for cp, k in quotes if cp == "C" and ("P", k) in quotes)
        if not strikes:
            continue  # no strike with both legs: no forward is invented
        last = contract_expiry(name, calendar).date()
        days = (last - day).days
        if days < 0:
            raise StudyDataError(f"{day}: series {name} listed after its last trading day {last}")
        tenor = days / ACT
        discount = float(safe_exp(-rate * tenor))
        k_star = min(strikes, key=lambda k: (abs(quotes[("C", k)].settle - quotes[("P", k)].settle), k))
        forward = k_star + (quotes[("C", k_star)].settle - quotes[("P", k_star)].settle) / discount
        series[name] = SeriesInfo(last, tenor, float(forward), discount)
    return DayBoard(day, float(spot), im, legs, series)


def _day_payload(day: date, history_dir: Path, extension_dir: Path) -> bytes:
    """The day's CSV bytes: shared cache first, then the extension; never writes, never downloads."""
    name = f"{day:%Y%m%d}_1.csv"
    for path in (Path(history_dir) / "settlement_csv" / name, Path(extension_dir) / name):
        if path.exists():
            return path.read_bytes()
    raise StudyDataError(f"no CFFEX CSV for {day} under {history_dir} or {extension_dir}")


def build_boards(chains: Sequence[DayChain], history_dir: Path, extension_dir: Path,
                 calendar: Optional[pd.DatetimeIndex] = None) -> Dict[date, DayBoard]:
    """``{day: DayBoard}`` for exactly the chain's days (a day without a CSV fails closed); spot from the chain."""
    if calendar is None:
        calendar = sse_trading_days(chains[0].day.year, chains[-1].day.year + _CALENDAR_PAD_YEARS)
    rate = float(load_prereg()["note"]["rate"])
    boards: Dict[date, DayBoard] = {}
    for dc in chains:
        rows = parse_cffex_full(_day_payload(dc.day, history_dir, extension_dir))
        boards[dc.day] = build_day_board(dc.day, dc.spot, rows, calendar, rate)
    return boards


def write_pickle_atomic(obj: object, path: Path) -> None:
    """Pickle ``obj`` to ``path`` atomically (tmp file, fsync, ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_boards(boards: Mapping[date, DayBoard], path: Path = BOARD_PATH) -> None:
    """Pickle the boards atomically."""
    write_pickle_atomic(dict(boards), path)


# The pickle is this study's own output (01_data.py), never untrusted input.
def load_boards(path: Path = BOARD_PATH) -> Dict[date, DayBoard]:
    """The pickle written by ``01_data.py`` (fails closed)."""
    path = Path(path)
    if not path.exists():
        raise StudyDataError(f"{path} missing; run 01_data.py")
    with path.open("rb") as fh:
        return pickle.load(fh)
