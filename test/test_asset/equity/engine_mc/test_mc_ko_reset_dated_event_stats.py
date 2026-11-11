"""MC event statistics on a dated KO-reset: post-schedule payment times come from the resolved schedule.

The raw schedule records of a dated contract carry dates, not times, so reading their
``settlement_time`` / ``observation_time`` crashed (``float(None)``).
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.cashleg import EventDistribution, EventType
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.util.calendar import CalendarType, create_calendar

from intraday.conftest import dated_ko_reset, flat_env
from pde_3a.products import KO_RESET, env as numeric_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def _cases():
    return {
        "dated": (dated_ko_reset(CAL, datetime(2026, 3, 16)), flat_env(datetime(2026, 4, 1))),
        "numeric": (make_ko_reset(KO_RESET), numeric_env()),
    }


@pytest.mark.parametrize("case", list(_cases()))
def test_post_ko_payment_times_match_the_pde_route(case):
    product, env = _cases()[case]
    mc = SnowballMCEngine(params=MCParams(num_paths=4000, seed=7)).calculate_event_stats(product, env)
    pde = KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, env)
    # MC: EventStatistics since T9, both schedules on one observation table tagged pre / post
    obs = mc.observations
    tags = np.asarray(obs.schedule)
    # PDE 1D: EventStatistics since T12, the same pre / post tagging
    ref = pde.observations
    ref_tags = np.asarray(ref.schedule)
    np.testing.assert_allclose(obs.times[tags == "post"], ref.times[ref_tags == "post"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(obs.payment_times[tags == "post"], ref.payment_times[ref_tags == "post"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(obs.payment_times[tags == "pre"], ref.payment_times[ref_tags == "pre"],
                               rtol=0, atol=1e-12)
    if case == "dated":
        assert obs.dates is not None        # determination dates; the time-based settlement resolver gives no date
    dist = EventDistribution.from_statistics(mc)
    total = sum(float(np.sum(dist.probabilities[e]))
                for e in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI))
    assert total == pytest.approx(1.0, abs=1e-9)
