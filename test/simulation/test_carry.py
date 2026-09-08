from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.carry import carry_at, curve_from_chain
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

GRID = np.array([0.25, 0.5, 1.0])
B = np.array([-0.02, -0.045, -0.10])  # B(T) at the grid


def test_carry_at_matches_forward_carry_curve_inside_and_beyond_the_grid():
    ref = ForwardCarryCurve(list(zip(GRID, B)))
    tenors = np.array([0.1, 0.25, 0.375, 0.5, 0.8, 1.0, 1.5, 2.0])
    out = carry_at(B, GRID, tenors)
    assert out.shape == (8,)
    for t, b in zip(tenors, out):
        assert b == pytest.approx(ref.carry(float(t)), abs=1e-15)


def test_carry_at_is_zero_at_zero_tenor_and_rejects_negative():
    assert carry_at(B, GRID, np.array([0.0]))[0] == 0.0
    with pytest.raises(ValidationError):
        carry_at(B, GRID, np.array([-0.01]))


def test_carry_at_broadcasts_over_paths_and_days():
    carry = np.stack([B, 2.0 * B])[:, None, :].repeat(4, axis=1)  # (2 paths, 4 days, 3 tenors)
    out = carry_at(carry, GRID, np.array([0.375, 1.5]))
    assert out.shape == (2, 4, 2)
    assert out[1, 3, 0] == pytest.approx(2.0 * (-0.02 - 0.045) / 2.0)
    assert out[0, 0, 1] == pytest.approx(-0.10 + (-0.10 + 0.045) / 0.5 * 0.5)


def test_curve_from_chain_round_trips_a_curve_and_drops_expired_contracts():
    spot = 6000.0
    listed_tenors = [0.0, 0.05, 0.3, 0.55, 0.9]         # first one expires today
    prices = [spot * np.exp(carry_at(B, GRID, np.array([t]))[0]) for t in listed_tenors]
    grid_b = curve_from_chain(spot, listed_tenors, prices, DEFAULT_TENOR_GRID)
    assert grid_b.shape == DEFAULT_TENOR_GRID.shape
    ref = ForwardCarryCurve([(t, np.log(p / spot)) for t, p in zip(listed_tenors[1:], prices[1:])])
    for t, b in zip(DEFAULT_TENOR_GRID, grid_b):
        assert b == pytest.approx(ref.carry(float(t)), abs=1e-15)
    with pytest.raises(ValidationError):
        curve_from_chain(spot, [0.0], [spot], DEFAULT_TENOR_GRID)


from quantark.backtest.simulation.carry import im_expiry, listed_im_contracts, third_friday
from quantark.backtest.simulation.paths.market_path import trading_calendar


def test_third_friday():
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert third_friday(2024, 2) == date(2024, 2, 16)
    assert third_friday(2024, 3) == date(2024, 3, 15)
    assert third_friday(2024, 6) == date(2024, 6, 21)
    assert third_friday(2024, 9) == date(2024, 9, 20)


def test_im_expiry_rolls_a_holiday_friday_to_the_next_trading_day():
    cal = trading_calendar(date(2024, 1, 2), 300, holidays=[date(2024, 2, 16)])
    assert im_expiry(2024, 1, cal) == pd.Timestamp("2024-01-19")
    assert im_expiry(2024, 2, cal) == pd.Timestamp("2024-02-19")  # Monday after the holiday Friday


def test_listed_im_contracts_follow_the_cffex_cycle():
    cal = trading_calendar(date(2024, 1, 2), 400)
    listed = listed_im_contracts(pd.Timestamp("2024-01-02"), cal)
    assert [c for c, _ in listed] == ["IM2401", "IM2402", "IM2403", "IM2406"]
    assert [e.date() for _, e in listed] == [date(2024, 1, 19), date(2024, 2, 16), date(2024, 3, 15), date(2024, 6, 21)]
    # the expiring contract is still listed on its expiry day ...
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-01-19"), cal)][0] == "IM2401"
    # ... and gone the next trading day, when a new quarterly is listed
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-01-22"), cal)] == ["IM2402", "IM2403", "IM2406", "IM2409"]
    # after the March expiry the two nearest months are April and May
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-03-18"), cal)] == ["IM2404", "IM2405", "IM2406", "IM2409"]
    # December wraps the year
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-11-11"), cal)] == ["IM2411", "IM2412", "IM2503", "IM2506"]
    # after the November expiry (15th) the December contract is nearest and January follows
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-11-18"), cal)] == ["IM2412", "IM2501", "IM2503", "IM2506"]


from quantark.backtest.simulation.carry import FUTURES_MULTIPLIER, DayChain, day_chain
from .conftest import SPOT, make_market_path


def test_day_chain_prices_each_path_off_its_own_curve():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    chain = day_chain(mp, 0)
    assert isinstance(chain, DayChain)
    assert chain.contracts == ("IM2401", "IM2402", "IM2403", "IM2406")
    assert chain.prices.shape == (2, 4)
    assert chain.multiplier == FUTURES_MULTIPLIER
    for i in range(2):
        for j, t in enumerate(chain.tenors):
            expected = mp.spot[i, 0] * np.exp(carry_at(mp.carry[i, 0], mp.tenor_grid, np.array([t]))[0])
            assert chain.prices[i, j] == pytest.approx(expected, rel=1e-14)
    assert chain.tenors[0] == pytest.approx((date(2024, 1, 19) - date(2024, 1, 2)).days / 365.0)


def test_day_chain_frame_has_the_replay_columns():
    mp = make_market_path(n_paths=2, n_days=30)
    frame = day_chain(mp, 3).frame(1)
    assert list(frame.columns) == ["date", "contract", "futures_price", "expiry_date", "multiplier"]
    assert len(frame) == 4
    assert (frame["date"] == mp.dates[3]).all()
    assert frame["futures_price"].iloc[2] == pytest.approx(day_chain(mp, 3).prices[1, 2])


def test_expiring_contract_prices_at_spot_on_its_expiry_day():
    mp = make_market_path(n_paths=1, n_days=30, start=date(2024, 1, 2))
    day = list(mp.dates).index(pd.Timestamp("2024-01-19"))
    chain = day_chain(mp, day)
    assert chain.contracts[0] == "IM2401"
    assert chain.tenors[0] == 0.0
    assert chain.prices[0, 0] == pytest.approx(mp.spot[0, day])
