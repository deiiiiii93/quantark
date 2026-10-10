"""Shared contract builders for the W17 diagnosis: the W12 snowball twins (test/test_vol_dated_schedules.py on
branch w12-vol-dated-schedules), copied verbatim so the scripts run on main 489cac69."""
from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime, timedelta

warnings.simplefilter("ignore")

from test_dated_expiry_settles import env as _env  # noqa: E402

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule  # noqa
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix  # noqa: E402
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402

INITIAL = datetime(2025, 1, 5)
EXERCISE = INITIAL + timedelta(days=365)
DAYS = [round(k / 12 * 365) for k in range(1, 13)]


def env(spot, valuation, grid=False):
    return _env(spot, valuation, grid=grid)


def on(day):
    return INITIAL + timedelta(days=day)


def records(form, level, days=DAYS, **kw):
    if form == "dated":
        return [ObservationRecord(observation_date=on(d), barrier=level, **kw) for d in days]
    return [ObservationRecord(observation_time=d / 365.0, barrier=level, **kw) for d in days]


def set_expiry(product, form):
    product.initial_date = INITIAL
    if form == "dated":
        product.maturity = None
        product.exercise_date = EXERCISE


def aged(product, form, day, age_dated=False, keep_landing=False):
    """The W12 aging: the numeric twin is aged with ``time_shift``; the dated form is priced as booked."""
    if form == "numeric" or age_dated:
        valuation = on(day)
        product.time_shift(day / 365.0, valuation, env(100.0, valuation), keep_landing=keep_landing)
    return product


def snowball(form, discrete_ki=False):
    """KO 103 monthly (flat 15%, not annualized), KI 75 continuous or monthly-discrete."""
    p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                 is_annualized=False)
    set_expiry(p, form)
    cfg = replace(p.barrier_config, ko_observation_dates=None,
                  ko_observation_schedule=ObservationSchedule(records=records(form, 103.0)))
    if discrete_ki:
        cfg = replace(cfg, ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
                      ki_observation_dates=None,
                      ki_observation_schedule=ObservationSchedule(records=records(form, 75.0)))
    p.barrier_config = cfg
    p.validate()
    return p


def phoenix(form, memory=False):
    """Non-memory Phoenix: KO 103, coupon 1% per period at 80, KI 70 continuous."""
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=12, memory_coupon=memory, is_annualized=False)
    set_expiry(p, form)
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=ObservationSchedule(records=records(form, 103.0)))
    p.validate()
    return p


def build(kind, form, day, **kw):
    if kind == "snowball":
        return aged(snowball(form), form, day, **kw)
    if kind == "snowball_discrete_ki":
        return aged(snowball(form, discrete_ki=True), form, day, age_dated=True, **kw)
    if kind == "phoenix":
        return aged(phoenix(form), form, day, **kw)
    if kind == "phoenix_memory":
        return aged(phoenix(form, memory=True), form, day, **kw)
    raise KeyError(kind)
