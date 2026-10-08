"""The replay dashboard ages a KO-reset the way the product's own ``time_shift`` does (review I-1, 2026-10-09):
an ABSOLUTE schedule 2 moves with the days elapsed and drops what has passed, its explicit factors with it; a
REBASED schedule 2 holds offsets from the knock-in and stays."""
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from quantark.asset.equity.product.option.accrual import ko_accrual_factor
from quantark.backtest.replay.dashboard import AutocallableBacktestDashboard
from quantark.util.enum.option_enums import PostKOScheduleMode as M

from .conftest import START, env
from .test_ko_reset_product import D, ko_reset

DAYS = 100


def _dashboard_copy(product, day):
    dash = object.__new__(AutocallableBacktestDashboard)
    dash.results = SimpleNamespace(config=SimpleNamespace(product=product, start_date=START.to_pydatetime()))
    states = pd.DataFrame({"knocked_in": [True]}, index=[day])
    return dash._product_for_date(day, env(date=day), states)


def _post(product):
    return product.post_barrier_config.ko_observation_schedule.records


def test_an_absolute_schedule_two_ages_with_the_dashboard_copy():
    product = ko_reset(M.ABSOLUTE, (91 * D, 182 * D, 273 * D))
    product.accrual_config = replace(product.accrual_config, post_accrual_factors=[0.1, 0.2, 0.3])
    product.validate()
    day = START + pd.Timedelta(days=DAYS)
    aged = _dashboard_copy(product, day)
    reference = ko_reset(M.ABSOLUTE, (91 * D, 182 * D, 273 * D))
    reference.accrual_config = replace(reference.accrual_config, post_accrual_factors=[0.1, 0.2, 0.3])
    reference.validate()
    reference.time_shift(DAYS / 365.0, day.to_pydatetime(), env(date=day))
    assert [r.observation_time for r in _post(aged)] == [r.observation_time for r in _post(reference)]
    assert len(_post(aged)) == 2
    assert [ko_accrual_factor(aged, r) for r in _post(aged)] == [0.2, 0.3]


def test_a_rebased_schedule_two_keeps_its_offsets_on_the_dashboard_copy():
    product = ko_reset(M.REBASED, (91 * D, 182 * D))
    before = [r.observation_time for r in _post(product)]
    aged = _dashboard_copy(product, START + pd.Timedelta(days=DAYS))
    assert [r.observation_time for r in _post(aged)] == before
