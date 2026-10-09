"""Checked for a moved value: an accumulator's theta on the day BEFORE its last observation (result: unchanged).

Theta (spec 2026-10-07 theta boundary R1) ages the contract one day keeping the observation that lands on the bumped
date; landing on the maturity, the aged copy would have maturity 0.0 and go through the engines' expiry shortcut, which
dropped the landing observation before this branch. Measured: theta is 0 in both trees, because ``numerical_theta``
returns 0.0 whenever ``current_maturity <= time_bump`` (the shortcut is never reached from theta on the last day).
Run once per tree:

    env PYTHONPATH=<tree>:<wt>/test <venv>/bin/python docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/measure_accumulator_theta.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta

from quantark.asset.equity.engine.analytical import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.mc import AccumulatorMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import AccumulatorOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AccumulatorKnockOutType, ObservationType, OptionType
from quantark.util.enum.option_enums import ObservationAggregation

warnings.simplefilter("ignore")
EXPIRY = datetime(2026, 1, 5)
DAY_BEFORE = EXPIRY - timedelta(days=1)
KW = dict(strike=95.0, knock_out_barrier=108.0, option_type=OptionType.CALL, daily_share_accumulation=1.0,
          gearing=2.0, knock_out_type=AccumulatorKnockOutType.TERMINATION, observation_type=ObservationType.DISCRETE)


def env(spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=DAY_BEFORE)


def numeric():
    return AccumulatorOption(maturity=1.0 / 365.0, observation_dates=[1.0 / 365.0], **KW)


def dated():
    schedule = ObservationSchedule(records=[ObservationRecord(observation_date=EXPIRY, barrier=108.0)],
                                   aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
    return AccumulatorOption(exercise_date=EXPIRY, observation_schedule=schedule, **KW)


print("form | engine | spot | price (day before) | theta (1 day)")
for form, build in (("numeric", numeric), ("dated", dated)):
    for name, engine in (("analytical", AccumulatorAnalyticalEngine), ("mc", lambda: AccumulatorMCEngine(params=MCParams(seed=7, num_paths=4096)))):
        for spot in (100.0, 90.0):
            try:
                p = float(engine().price(build(), env(spot)))
                theta = float(GreeksCalculator().calculate_numerical_theta(build(), env(spot), engine()))
                print(f"{form} | {name} | {spot:g} | {p:.10g} | {theta:.10g}")
            except Exception as exc:
                print(f"{form} | {name} | {spot:g} | RAISES {type(exc).__name__}: {str(exc)[:100]}")
