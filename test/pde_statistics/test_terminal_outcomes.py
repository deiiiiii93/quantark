"""Valuation-date outcomes (spec 3B §5.5): one rule set from every entry point."""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.calendar import CalendarType, create_calendar
from intraday.conftest import dated_phoenix, dated_snowball, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _first_obs(p):
    return p.barrier_config.ko_observation_schedule.records[0].observation_date


@pytest.mark.parametrize("cls,make", [(SnowballPDESolver, dated_snowball), (PhoenixPDESolver, dated_phoenix)])
def test_knocked_out_event_stats_is_a_terminal_outcome(cls, make, monkeypatch):
    product = make(CAL, T0)
    env = flat_env(_first_obs(product), spot=130.0)
    solver = cls(PDEParams())
    p = prepare(solver, product, env, OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True))
    assert p.terminal is not None and p.terminal.reason == "knocked_out_at_valuation"
    assert solver.explain(product, env).terminal["event_stats"] == "knocked_out_at_valuation"
    from quantark.asset.equity.engine.pde.grid.binder import GridBinder
    calls = []
    monkeypatch.setattr(GridBinder, "bind", lambda *a, **k: calls.append(1))
    stats = solver.calculate_event_stats(product, env)
    assert not calls                                      # no layout was bound
    assert stats.ko_probability[0] == 1.0 and not np.any(stats.ko_probability[1:])


def test_nodal_still_marches_until_stage_d():
    product = dated_snowball(CAL, T0)
    env = flat_env(_first_obs(product), spot=130.0)
    p = prepare(SnowballPDESolver(PDEParams(event_projection="nodal")), product, env,
                OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True))
    assert p.terminal is None
