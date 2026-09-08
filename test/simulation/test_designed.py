from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.paths.designed import SnowballStressLibrary, market_path_from_day_path, stress_set
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.exceptions import ValidationError

from .conftest import flat_carry

START = StartState(spot=6000.0, atm_vol=0.22, rate=0.02, carry=flat_carry(DEFAULT_TENOR_GRID))
CAL = trading_calendar(date(2025, 1, 2), 60)


def test_day_path_changes_are_applied_cumulatively():
    dp = DayPath(name="t", steps=[
        DayStep(0, [ParameterChange("spot", StressType.PERCENTAGE, -0.10)]),
        DayStep(1, [ParameterChange("volatility", StressType.ABSOLUTE, 0.05), ParameterChange("basis", StressType.ABSOLUTE, -0.02)]),
        DayStep(2, [ParameterChange("rate", StressType.VALUE, 0.03)]),
        DayStep(3, []),
    ])
    mp = market_path_from_day_path(dp, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert mp.n_paths == 1 and mp.n_days == 4
    assert mp.spot[0] == pytest.approx([5400.0, 5400.0, 5400.0, 5400.0])
    assert mp.atm_vol[0] == pytest.approx([0.22, 0.27, 0.27, 0.27])
    assert mp.rate[0] == pytest.approx([0.02, 0.02, 0.03, 0.03])
    assert mp.carry[0, 0] == pytest.approx(START.carry)
    assert mp.carry[0, 1] == pytest.approx(START.carry - 0.02 * DEFAULT_TENOR_GRID)
    assert mp.meta["generator"] == "designed" and mp.meta["scenario_name"] == "t"


def test_unknown_parameter_and_percentage_basis_are_rejected():
    bad = DayPath(name="x", steps=[DayStep(0, [ParameterChange("dividend", StressType.ABSOLUTE, 0.01)])])
    with pytest.raises(ValidationError):
        market_path_from_day_path(bad, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    bad = DayPath(name="y", steps=[DayStep(0, [ParameterChange("basis", StressType.PERCENTAGE, 0.1)])])
    with pytest.raises(ValidationError):
        market_path_from_day_path(bad, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)


def test_stress_library_shapes():
    crash = market_path_from_day_path(SnowballStressLibrary.crash_into_ki(depth=0.30, days=10, total_days=40),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert crash.n_days == 40
    assert crash.spot[0, 9] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    assert crash.spot[0, 39] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    v = market_path_from_day_path(SnowballStressLibrary.v_shape(depth=0.30, days_down=10, days_up=10, total_days=40),
                                  start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert v.spot[0, 9] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    assert v.spot[0, 19] == pytest.approx(6000.0, rel=1e-9)
    spike = market_path_from_day_path(SnowballStressLibrary.vol_spike(vol_up=0.15, decay_days=5, total_days=20),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert spike.atm_vol[0, 0] == pytest.approx(0.37) and spike.atm_vol[0, 5] == pytest.approx(0.22, abs=1e-9)
    blow = market_path_from_day_path(SnowballStressLibrary.basis_blowout(carry_shift=-0.10, days=5, total_days=20),
                                     start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert blow.carry[0, 4] == pytest.approx(START.carry - 0.10 * DEFAULT_TENOR_GRID)
    grind = market_path_from_day_path(SnowballStressLibrary.grind_up_to_ko(pct_per_day=0.005, days=10, total_days=20),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert grind.spot[0, 9] == pytest.approx(6000.0 * 1.005**10)


def test_stress_set_stacks_equal_length_paths():
    a = SnowballStressLibrary.crash_into_ki(depth=0.3, days=10, total_days=30)
    b = SnowballStressLibrary.vol_spike(vol_up=0.1, decay_days=5, total_days=30)
    mp = stress_set([a, b], start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert mp.n_paths == 2 and mp.n_days == 30
    assert mp.meta["scenario_names"] == [a.name, b.name]
    with pytest.raises(ValidationError):
        stress_set([a, SnowballStressLibrary.vol_spike(vol_up=0.1, decay_days=5, total_days=31)],
                   start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
