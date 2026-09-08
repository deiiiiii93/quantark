from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import (
    DEFAULT_TENOR_GRID,
    MarketPath,
    StartState,
    trading_calendar,
)
from quantark.util.exceptions import ValidationError

from .conftest import SPOT, make_market_path


def test_trading_calendar_skips_weekends_and_holidays():
    cal = trading_calendar(date(2024, 1, 4), 5, holidays=[date(2024, 1, 8)])
    # Thu 4, Fri 5, (Sat, Sun), Mon 8 holiday, Tue 9, Wed 10, Thu 11
    assert list(cal.date) == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]


def test_trading_calendar_starts_on_the_next_trading_day_when_start_is_a_weekend():
    cal = trading_calendar(date(2024, 1, 6), 2)  # Saturday
    assert list(cal.date) == [date(2024, 1, 8), date(2024, 1, 9)]


def test_market_path_shapes_and_properties():
    mp = make_market_path(n_paths=3, n_days=30)
    assert (mp.n_paths, mp.n_days, mp.n_tenors) == (3, 30, DEFAULT_TENOR_GRID.size)
    one = mp.path(1)
    assert one.n_paths == 1
    assert one.spot[0, 0] == pytest.approx(SPOT * 1.01)
    assert one.dates.equals(mp.dates)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("spot", lambda a: -a),                       # non-positive spot
        ("atm_vol", lambda a: np.zeros_like(a)),      # non-positive vol
        ("spot", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.nan, a)),  # NaN
    ],
)
def test_market_path_rejects_invalid_arrays(field, bad):
    mp = make_market_path()
    kwargs = {k: getattr(mp, k) for k in ("dates", "spot", "atm_vol", "rate", "carry", "tenor_grid", "meta")}
    kwargs[field] = bad(kwargs[field])
    with pytest.raises(ValidationError):
        MarketPath(**kwargs)


def test_market_path_rejects_shape_mismatch_and_bad_tenor_grid():
    mp = make_market_path()
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates, spot=mp.spot[:, :-1], atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=mp.tenor_grid, meta={})
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates, spot=mp.spot, atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=np.array([0.5, 0.25, 1.0]), meta={})
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates[::-1], spot=mp.spot, atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=mp.tenor_grid, meta={})


def test_fingerprint_changes_with_data_and_not_with_meta():
    a = make_market_path()
    b = MarketPath(dates=a.dates, spot=a.spot, atm_vol=a.atm_vol, rate=a.rate, carry=a.carry,
                   tenor_grid=a.tenor_grid, meta={"generator": "other"})
    c = MarketPath(dates=a.dates, spot=a.spot * 1.0001, atm_vol=a.atm_vol, rate=a.rate, carry=a.carry,
                   tenor_grid=a.tenor_grid, meta=a.meta)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 64


def test_start_state_validates():
    with pytest.raises(ValidationError):
        StartState(spot=0.0, atm_vol=0.2, rate=0.02, carry=np.zeros(3))
    with pytest.raises(ValidationError):
        StartState(spot=100.0, atm_vol=0.2, rate=0.02, carry=np.array([[0.0]]))


def test_npz_round_trip_keeps_arrays_dates_meta_and_fingerprint(tmp_path):
    mp = make_market_path()
    file = tmp_path / "paths.npz"
    mp.to_npz(file)
    back = MarketPath.from_npz(file)
    assert back.fingerprint() == mp.fingerprint()
    assert back.dates.equals(mp.dates) and back.meta == mp.meta
    assert np.array_equal(back.carry, mp.carry)
