from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.carry import carry_at
from quantark.backtest.simulation.paths.history import PathHistory
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

RATE = 0.02


def _frames(n_days: int = 10):
    dates = trading_calendar(date(2024, 1, 2), n_days)
    spot = pd.DataFrame({"date": dates, "spot": 6000.0 + 10.0 * np.arange(n_days)})
    vol = pd.DataFrame({"date": dates, "volatility": 0.20 + 0.001 * np.arange(n_days)})
    rows = []
    chain = [("IM2401", pd.Timestamp("2024-01-19")), ("IM2402", pd.Timestamp("2024-02-16")),
             ("IM2403", pd.Timestamp("2024-03-15")), ("IM2406", pd.Timestamp("2024-06-21"))]
    for k, d in enumerate(dates):
        s = float(spot["spot"].iloc[k])
        for c, e in chain:
            t = (e - d).days / 365.0
            rows.append({"date": d, "contract": c, "futures_price": s * np.exp(-0.10 * t),
                         "expiry_date": e, "multiplier": 200.0})
    return spot, vol, pd.DataFrame(rows)


def test_from_frames_builds_levels_changes_and_snapshot():
    spot, vol, futures = _frames()
    h = PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    assert h.spot.shape == (10,) and h.carry.shape == (10, DEFAULT_TENOR_GRID.size)
    assert h.levels().shape == (10, 3 + DEFAULT_TENOR_GRID.size)
    ch = h.changes()
    assert ch.shape == (9, 3 + DEFAULT_TENOR_GRID.size)
    assert ch[0, 0] == pytest.approx(np.log(6010.0 / 6000.0))
    assert ch[0, 1] == pytest.approx(0.001)
    assert ch[:, 2] == pytest.approx(0.0)
    # a flat -10% carry chain gives a flat -10% * T curve on every day
    for k in range(10):
        assert h.carry[k] == pytest.approx(-0.10 * DEFAULT_TENOR_GRID, abs=1e-12)
    snap = h.snapshot()
    assert isinstance(snap, StartState)
    assert snap.spot == pytest.approx(6090.0) and snap.atm_vol == pytest.approx(0.209)
    assert len(h.source_fingerprint) == 64


def test_from_frames_intersects_dates_and_accepts_a_rate_frame():
    spot, vol, futures = _frames()
    vol = vol.iloc[2:]                       # vol history starts later
    rate = pd.DataFrame({"date": spot["date"], "rate": [RATE] * len(spot)})
    h = PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=rate, tenor_grid=DEFAULT_TENOR_GRID)
    assert h.dates[0] == vol["date"].iloc[0] and len(h.dates) == 8


def test_from_frames_rejects_a_day_without_a_live_contract():
    spot, vol, futures = _frames()
    day3 = spot["date"].iloc[3]
    futures.loc[futures["date"] == day3, "expiry_date"] = day3 - pd.Timedelta(days=1)  # all expired that day
    with pytest.raises(ValidationError):
        PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
