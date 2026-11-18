"""
Data layer for the real-data CSI study: basket, cache location, exchange
calendar and price alignment.

Raw vendor prices never enter the repo.  ``01_fetch.py`` (the only script that
imports akshare) writes them to ``cache_dir()``, outside the repository; every
other stage reads them back through ``load_prices`` / ``load_calendar``.

The exchange calendar is the one calendar of the study: prices are reindexed
onto its sessions, and a session a name did not trade (suspension) carries the
previous close.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Tuple

import pandas as pd

from quantark.util.exceptions import MarketDataError, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = Path("~/.cache/quantark/csi_real_data")
FETCH_START = "2023-01-01"


@dataclass(frozen=True)
class RealName:
    """One basket constituent."""

    code: str
    name: str
    kind: Literal["stock", "etf"]
    weight: float


# Public stand-in for a CSI: same shape and weights as the synthetic study's basket.
BASKET: Tuple[RealName, ...] = (
    RealName("600519", "Kweichow Moutai", "stock", 0.25),
    RealName("600036", "China Merchants Bank", "stock", 0.20),
    RealName("002230", "iFlytek", "stock", 0.15),
    RealName("510300", "CSI 300 ETF", "etf", 0.30),
    RealName("518880", "Gold ETF", "etf", 0.10),
)


def codes() -> list:
    return [n.code for n in BASKET]


def target_weights() -> "pd.Series":
    return pd.Series({n.code: n.weight for n in BASKET}, dtype=float)


def cache_dir() -> Path:
    """Cache location, resolved at call time; must lie outside the repository."""
    path = Path(os.environ.get("CSI_REAL_DATA_CACHE", str(DEFAULT_CACHE))).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValidationError(f"cache dir {path} is inside the repo; raw prices must stay outside it")
    return path


def _require(path: Path) -> Path:
    if not path.exists():
        raise MarketDataError(f"missing cached data {path}; run 01_fetch.py first")
    return path


CLOSE_FINAL = dtime(15, 30)   # Beijing time; a bar fetched earlier is intraday, not a close


def last_closed_session(sessions: pd.DatetimeIndex, now_cn: pd.Timestamp) -> pd.Timestamp:
    """Last session whose close is final: today's bar counts only after 15:30 Beijing time."""
    today = now_cn.normalize()
    cutoff = today if now_cn.time() < CLOSE_FINAL else today + pd.Timedelta(days=1)
    return pd.DatetimeIndex(sessions)[pd.DatetimeIndex(sessions) < cutoff].max()


def align_prices(raw: Mapping[str, pd.Series], sessions: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Reindex each name's closes onto the exchange sessions.

    The range starts at the first session on which every name has traded and
    ends at the last session any name has a close.  A session a name did not
    trade is forward-filled; the fill counts are returned per name.

    Raises:
        MarketDataError: a close on a non-session day, or a name with no closes.
    """
    sessions = pd.DatetimeIndex(sessions)
    firsts, lasts = [], []
    for code, series in raw.items():
        if series.empty:
            raise MarketDataError(f"{code}: no closes")
        idx = pd.DatetimeIndex(series.index)
        if idx.has_duplicates:
            raise MarketDataError(f"{code}: duplicate dates {list(idx[idx.duplicated()][:3].date)}")
        off = idx.difference(sessions)
        if len(off):
            raise MarketDataError(f"{code}: closes on non-session days {list(off[:3].date)}")
        firsts.append(idx.min())
        lasts.append(idx.max())
    span = sessions[(sessions >= max(firsts)) & (sessions <= max(lasts))]
    frame, fills = {}, {}
    for code, series in raw.items():
        s = pd.Series(series.values, index=pd.DatetimeIndex(series.index), dtype=float).reindex(span)
        fills[code] = int(s.isna().sum())
        frame[code] = s.ffill()
    return pd.DataFrame(frame, index=span), fills


def save_cache(raw: Mapping[str, pd.Series], sessions: pd.DatetimeIndex, meta: Mapping[str, Any]) -> Path:
    """Write raw closes, the session calendar and metadata to ``cache_dir()``."""
    root = cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    for code, series in raw.items():
        pd.DataFrame({"date": pd.DatetimeIndex(series.index).strftime("%Y-%m-%d"),
                      "close": series.values}).to_csv(root / f"raw_{code}.csv", index=False)
    pd.DataFrame({"date": pd.DatetimeIndex(sessions).strftime("%Y-%m-%d")}).to_csv(root / "calendar.csv", index=False)
    (root / "meta.json").write_text(json.dumps(dict(meta), indent=1, ensure_ascii=False), encoding="utf-8")
    return root


def load_calendar() -> pd.DatetimeIndex:
    """Exchange sessions (may extend past the price data)."""
    frame = pd.read_csv(_require(cache_dir() / "calendar.csv"))
    return pd.DatetimeIndex(pd.to_datetime(frame["date"]))


def calendar_horizon() -> pd.Timestamp:
    return load_calendar().max()


def load_meta() -> Dict[str, Any]:
    return json.loads(_require(cache_dir() / "meta.json").read_text(encoding="utf-8"))


def load_raw() -> Dict[str, pd.Series]:
    root = cache_dir()
    out = {}
    for code in codes():
        frame = pd.read_csv(_require(root / f"raw_{code}.csv"))
        out[code] = pd.Series(frame["close"].astype(float).values, index=pd.to_datetime(frame["date"]))
    return out


def load_prices() -> pd.DataFrame:
    """Aligned back-adjusted closes, one column per basket code (basket order)."""
    prices, _ = align_prices(load_raw(), load_calendar())
    return prices[codes()]


def load_fill_counts() -> Dict[str, int]:
    return align_prices(load_raw(), load_calendar())[1]
