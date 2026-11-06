"""Daily CFFEX chain: IM futures + the longest MO series' ATM call/put pair.

Marks: spot = CSI 1000 close, IM = CFFEX close (col 8), MO = CFFEX settlement (col 9).
"""

from __future__ import annotations

import csv
import io
import math
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from quantark.backtest.simulation.carry import im_expiry
from quantark.param.vol.marketquotes import implied_vol_black
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, safe_log

from . import ACT, CHAIN_PATH, MIN_TENOR_DAYS, RATE, StudyDataError

# CFFEX daily statistics CSV (GBK): code, ..., close (今收盘, col 8), settlement (今结算, col 9).
_CSV_CODE, _CSV_CLOSE, _CSV_SETTLE = 0, 8, 9
_CSV_HEADER = {_CSV_CODE: "合约代码", _CSV_CLOSE: "今收盘", _CSV_SETTLE: "今结算"}

_FUT_RE = re.compile(r"^(IM|MO)(\d{2})(\d{2})$")
_OPT_RE = re.compile(r"^(MO\d{4})-([CP])-(\d+(?:\.\d+)?)$")
_N_NEAR = 5


def parse_cffex_csv(payload: bytes) -> Dict[str, Dict[str, float]]:
    """``{code: {"close", "settle"}}``; rows with a non-numeric close/settle are skipped."""
    reader = csv.reader(io.StringIO(payload.decode("gbk")))
    header = next(reader, None)
    if not header or len(header) <= _CSV_SETTLE or any(
        header[i].strip() != name for i, name in _CSV_HEADER.items()
    ):
        raise StudyDataError(f"not a CFFEX daily statistics CSV in the expected layout: {_CSV_HEADER}")
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


def sse_trading_days(first_year: int, last_year: int) -> pd.DatetimeIndex:
    """SSE trading days in ``[first_year, last_year]`` from the repo's China-SSE calendar."""
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(first_year, last_year))
    days = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31")
    return pd.DatetimeIndex([d for d in days if cal.is_business_day(d.to_pydatetime())])


def contract_expiry(code: str, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Expiry of ``IMyymm`` / ``MOyymm`` (third Friday, else next trading day)."""
    m = _FUT_RE.match(code)
    if not m:
        raise StudyDataError(f"cannot parse contract code {code!r}")
    return im_expiry(2000 + int(m.group(2)), int(m.group(3)), calendar)


@dataclass(frozen=True)
class ImQuote:
    code: str
    expiry: date
    tenor: float
    close: float
    settle: float


@dataclass(frozen=True)
class MoPillar:
    series: str
    expiry: date
    tenor: float
    strike: float
    call: float
    put: float
    forward: float
    vol: float
    #: The 5 strikes nearest the ATM strike as ``(K, C_settle, P_settle)``, sorted by K.
    strikes_near: Tuple[Tuple[float, float, float], ...]


@dataclass(frozen=True)
class DayChain:
    day: date
    spot: float
    im: Tuple[ImQuote, ...]
    mo: MoPillar
    mo_all: Dict[float, Tuple[float, float]]
    far_carry: float


def _tenor(expiry: pd.Timestamp, day: date) -> float:
    return (expiry.date() - day).days / ACT


def build_day_chain(day: date, spot: float, rows: Dict[str, Dict[str, float]],
                    calendar: pd.DatetimeIndex) -> DayChain:
    """One day's chain from parsed CFFEX rows."""
    im: List[ImQuote] = []
    for code, r in rows.items():
        m = _FUT_RE.match(code)
        if not m or m.group(1) != "IM":
            continue
        exp = contract_expiry(code, calendar)
        if (exp.date() - day).days < MIN_TENOR_DAYS:
            continue
        im.append(ImQuote(code, exp.date(), _tenor(exp, day), r["close"], r["settle"]))
    if not im:
        raise StudyDataError(f"{day}: no IM contract with >= {MIN_TENOR_DAYS} days to expiry")
    im.sort(key=lambda q: q.tenor)

    calls: Dict[str, Dict[float, float]] = {}
    puts: Dict[str, Dict[float, float]] = {}
    for code, r in rows.items():
        m = _OPT_RE.match(code)
        if m:
            (calls if m.group(2) == "C" else puts).setdefault(m.group(1), {})[float(m.group(3))] = r["settle"]
    series_list = [s for s in calls if s in puts]
    if not series_list:
        raise StudyDataError(f"{day}: no MO series with both calls and puts")
    series = max(series_list, key=lambda s: contract_expiry(s, calendar))
    mo_all = {k: (calls[series][k], puts[series][k]) for k in sorted(calls[series]) if k in puts[series]}
    if not mo_all:
        raise StudyDataError(f"{day}: no {series} strike has both a call and a put settlement")
    k_atm = min(mo_all, key=lambda k: (abs(mo_all[k][0] - mo_all[k][1]), k))
    c, p = mo_all[k_atm]
    exp = contract_expiry(series, calendar)
    T = _tenor(exp, day)
    df = float(safe_exp(-RATE * T))
    forward = k_atm + (c - p) / df
    k_vol = min(mo_all, key=lambda k: (abs(k - forward), k))
    cv, pv = mo_all[k_vol]
    try:
        vol = 0.5 * (implied_vol_black(cv, forward, k_vol, T, df, True)
                     + implied_vol_black(pv, forward, k_vol, T, df, False))
    except ValidationError as err:
        raise StudyDataError(f"{day}: cannot invert Black-76 at {series} K={k_vol:g}: {err}") from err
    near = sorted(sorted(mo_all, key=lambda k: (abs(k - k_atm), k))[:_N_NEAR])
    pillar = MoPillar(series, exp.date(), T, k_atm, c, p, forward, vol,
                      tuple((k, mo_all[k][0], mo_all[k][1]) for k in near))
    far = im[-1]
    return DayChain(day, float(spot), tuple(im), pillar, mo_all,
                    float(safe_log(far.close / spot)) / far.tenor)


def load_day_rows(day: date, history_dir: Path, extension_dir: Path) -> Dict[str, Dict[str, float]]:
    """Parsed CSV of ``day``: shared cache first, then the extension; never writes."""
    name = f"{day:%Y%m%d}_1.csv"
    for path in (Path(history_dir) / "settlement_csv" / name, Path(extension_dir) / name):
        if path.exists():
            return parse_cffex_csv(path.read_bytes())
    raise StudyDataError(f"no CFFEX CSV for {day}; run 01_data.py to download it")


def load_spot_series(history_dir: Path, spot_extension: Path | None = None) -> Dict[date, float]:
    """``{date: spot}`` from the shared cache plus the optional extension file."""
    out: Dict[date, float] = {}
    for path in (Path(history_dir) / "csi1000_spot.csv", spot_extension):
        if path is None or not Path(path).exists():
            continue
        for rec in csv.DictReader(Path(path).read_text().splitlines()):
            out[date.fromisoformat(rec["date"])] = float(rec["spot"])
    if not out:
        raise StudyDataError(f"no spot rows under {history_dir}; run 01_data.py")
    return out


def build_all_chains(history_dir: Path, extension_dir: Path, calendar: pd.DatetimeIndex,
                     spot_extension: Path | None = None) -> List[DayChain]:
    """A DayChain for every spot day; a calendar trading day with no spot row fails closed."""
    spots = load_spot_series(history_dir, spot_extension)
    days = sorted(spots)
    have = set(pd.Timestamp(d) for d in days)
    expected = calendar[(calendar >= pd.Timestamp(days[0])) & (calendar <= pd.Timestamp(days[-1]))]
    missing = [d for d in expected if d not in have]
    if missing:
        raise StudyDataError(
            f"spot series has no row for trading day {missing[0].date()} "
            f"({len(missing)} missing); run 01_data.py"
        )
    return [build_day_chain(d, spots[d], load_day_rows(d, history_dir, extension_dir), calendar) for d in days]


# The pickle is this study's own output (01_data.py), never untrusted input.
def load_chain(path: Path = CHAIN_PATH) -> List[DayChain]:
    """The pickle written by ``01_data.py`` (fails closed)."""
    path = Path(path)
    if not path.exists():
        raise StudyDataError(f"{path} missing; run 01_data.py")
    with path.open("rb") as fh:
        return pickle.load(fh)
