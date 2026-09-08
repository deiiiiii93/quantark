from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths, StickyRealisedVol
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

from .conftest import flat_carry


def _start():
    return StartState(spot=6000.0, atm_vol=0.22, rate=0.02, carry=flat_carry(DEFAULT_TENOR_GRID))


def _gen(**overrides):
    kwargs = dict(start=_start(), calendar=trading_calendar(date(2025, 1, 2), 260), mu=0.05, sigma=0.25,
                  vol_rule=ConstantVol(0.22), carry_schedule=None, rate=0.02, tenor_grid=DEFAULT_TENOR_GRID)
    kwargs.update(overrides)
    return GBMPaths(**kwargs)


def test_log_return_moments_match_mu_and_sigma():
    mp = _gen().generate(4000, 253, seed=1)
    r = np.log(mp.spot[:, 1:] / mp.spot[:, :-1])
    assert r.mean() * 252 == pytest.approx(0.05 - 0.5 * 0.25**2, abs=0.01)
    assert r.std() * np.sqrt(252) == pytest.approx(0.25, abs=0.01)
    assert mp.spot[:, 0] == pytest.approx(6000.0)
    assert mp.meta["generator"] == "gbm" and mp.meta["seed"] == 1


def test_constant_vol_and_held_carry():
    mp = _gen().generate(3, 20, seed=2)
    assert np.all(mp.atm_vol == 0.22)
    assert mp.carry[1, 7] == pytest.approx(flat_carry(DEFAULT_TENOR_GRID))
    assert np.all(mp.rate == 0.02)


def test_sticky_realised_vol_rule():
    rule = StickyRealisedVol(a=0.05, b=1.0, window=5)
    r = np.full((1, 9), 0.01)
    out = rule.apply(r, start_vol=0.3)
    assert out.shape == (1, 10)
    assert np.all(out[0, :5] == 0.3)                       # not enough returns yet
    assert out[0, 5] == pytest.approx(0.05 + 1.0 * 0.0)     # identical returns: realised vol 0
    mp = _gen(vol_rule=rule).generate(2, 30, seed=3)
    assert np.all(mp.atm_vol > 0.0)


def test_carry_schedule_is_used_when_given():
    schedule = np.outer(np.linspace(-0.05, -0.15, 20), DEFAULT_TENOR_GRID)
    mp = _gen(carry_schedule=schedule).generate(2, 20, seed=4)
    assert mp.carry[0] == pytest.approx(schedule) and mp.carry[1] == pytest.approx(schedule)
    with pytest.raises(ValidationError):
        _gen(carry_schedule=schedule).generate(2, 25, seed=4)


def test_parameters_are_validated():
    with pytest.raises(ValidationError):
        _gen(sigma=-0.1)
    with pytest.raises(ValidationError):
        ConstantVol(0.0)
    with pytest.raises(ValidationError):
        StickyRealisedVol(a=0.05, b=1.0, window=0)
