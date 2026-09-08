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
