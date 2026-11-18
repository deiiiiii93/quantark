"""
Stage 1 — fetch daily closes and the exchange calendar with akshare.

Writes raw back-adjusted (hfq) closes for the five basket names, the SSE/SZSE
session calendar (including announced future sessions) and a metadata file to
``_data.cache_dir()`` — outside the repository.  Nothing is written into the
repo: raw vendor prices are not committed.

Requires ``pip install akshare`` (not a project dependency).  akshare renamed
several functions in 1.20.4x; this script uses the current names:

* stocks: ``stock_zh_a_hist_em(adjust="hfq")``
* ETFs:   ``fund_etf_hist_em(adjust="hfq")``; fallback ``fund_etf_hist_sina``
          (unadjusted — flagged in the metadata) only if EastMoney fails
* calendar: ``calendar_trade_date_hist_sina``
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _data  # noqa: E402

try:
    import akshare as ak
except ImportError:  # pragma: no cover - network stage only
    sys.exit("01_fetch.py needs akshare: pip install akshare")

PAUSE_S = 1.0
RETRIES = 3


def _retry(fn, **kwargs):
    last = None
    for attempt in range(RETRIES):
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor errors are heterogeneous
            last = exc
            time.sleep(PAUSE_S * (attempt + 2))
    raise last


def _em_frame(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["收盘"].astype(float).values, index=pd.to_datetime(frame["日期"]))


def fetch_name(name: _data.RealName, start: str, end: str):
    """Return (closes, source label)."""
    kw = dict(symbol=name.code, period="daily", start_date=start, end_date=end, adjust="hfq")
    if name.kind == "stock":
        return _em_frame(_retry(ak.stock_zh_a_hist_em, **kw)), "stock_zh_a_hist_em(hfq)"
    try:
        return _em_frame(_retry(ak.fund_etf_hist_em, **kw)), "fund_etf_hist_em(hfq)"
    except Exception:  # noqa: BLE001 - fall back to Sina, flagged unadjusted
        prefix = "sh" if name.code.startswith("5") else "sz"
        frame = _retry(ak.fund_etf_hist_sina, symbol=f"{prefix}{name.code}")
        s = pd.Series(frame["close"].astype(float).values, index=pd.to_datetime(frame["date"]))
        s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return s, "fund_etf_hist_sina(UNADJUSTED)"


def main() -> None:
    start = _data.FETCH_START.replace("-", "")
    cal = _retry(ak.calendar_trade_date_hist_sina)
    sessions = pd.DatetimeIndex(pd.to_datetime(cal["trade_date"]))
    sessions = sessions[sessions >= pd.Timestamp(_data.FETCH_START)]
    last = _data.last_closed_session(sessions, pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None))
    end = last.strftime("%Y%m%d")
    raw, sources = {}, {}
    for name in _data.BASKET:
        series, source = fetch_name(name, start, end)
        raw[name.code] = series.sort_index()
        sources[name.code] = source
        print(f"  {name.code} {name.name:22s} {len(series):4d} closes "
              f"{series.index.min().date()} → {series.index.max().date()}  [{source}]")
        time.sleep(PAUSE_S)
    meta = {
        "akshare": ak.__version__,
        "sources": sources,
        "fetched_on": date.today().isoformat(),
        "requested": [_data.FETCH_START, last.date().isoformat()],
        "calendar_horizon": sessions.max().date().isoformat(),
    }
    root = _data.save_cache(raw, sessions, meta)
    prices, fills = _data.align_prices(raw, sessions)
    print(f"wrote {root}  ({len(prices)} sessions {prices.index[0].date()} → {prices.index[-1].date()}; "
          f"calendar to {meta['calendar_horizon']}; forward-filled sessions {fills})")


if __name__ == "__main__":
    main()
