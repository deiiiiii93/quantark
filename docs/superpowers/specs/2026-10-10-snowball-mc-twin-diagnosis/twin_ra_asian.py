"""W17a step 4: the range-accrual and Asian MC grids (``set(future_times) | {T}``,
range_accrual_mc_engine.py:325, asian_option_mc_engine.py:337). The only collision those grids can hold is an
observation time against maturity T.

Twins: monthly records on contract days d_k = round(k/12*365) (the last on day 365 = the exercise day) and a
schedule whose last record is one day BEFORE the exercise day; dated (records on INITIAL + d_k, exercise INITIAL +
365) and numeric (records at d_k/365, maturity 1.0) aged with ``time_shift(day/365, keep_landing=True)``. Every
valuation day 1..364: node counts and the smallest step of each form's grid; prices (seed 11 x 4096) on every
day where either grid holds a step below 1e-10 and on days 45 / 92 / 200."""
import numpy as np

import common as c
from quantark.asset.equity.engine.mc.asian_option_mc_engine import AsianOptionMCEngine
from quantark.asset.equity.engine.mc.range_accrual_mc_engine import RangeAccrualMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.asset.equity.product.option.range_accrual_config import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
)
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.util.enum import OptionType

TOL = 1e-10
SCHEDULES = {"monthly_to_expiry": c.DAYS, "last_record_day_364": c.DAYS[:-1] + [364]}


def range_accrual(form, days):
    cfg = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10)
    if form == "dated":
        recs = [RangeAccrualObservationRecord(observation_date=c.on(d)) for d in days]
        return RangeAccrualOption(initial_price=100.0, range_config=cfg, exercise_date=c.EXERCISE,
                                  initial_date=c.INITIAL, observation_records=recs)
    recs = [RangeAccrualObservationRecord(observation_time=d / 365.0) for d in days]
    return RangeAccrualOption(initial_price=100.0, range_config=cfg, maturity=1.0, tenor=1.0,
                              observation_records=recs)


def asian(form, days):
    if form == "dated":
        recs = [AsianObservationRecord(observation_date=c.on(d)) for d in days]
        return AsianOption(strike=100.0, option_type=OptionType.CALL, exercise_date=c.EXERCISE,
                           initial_date=c.INITIAL, observation_records=recs, num_observations=None)
    recs = [AsianObservationRecord(observation_time=d / 365.0) for d in days]
    return AsianOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0, observation_records=recs,
                       num_observations=None)


PRODUCTS = {"range_accrual": (range_accrual, RangeAccrualMCEngine), "asian": (asian, AsianOptionMCEngine)}


def build(kind, form, days, day):
    p = PRODUCTS[kind][0](form, days)
    if form == "numeric":
        p.time_shift(day / 365.0, c.on(day), c.env(100.0, c.on(day)), keep_landing=True)
    return p


def grid(kind, form, days, day):
    p = build(kind, form, days, day)
    e = c.env(100.0, c.on(day))
    eng = PRODUCTS[kind][1](params=MCParams(seed=11, num_paths=16))
    at = eng._build_observation_grid(p, e, p.get_maturity(e))[0]
    return np.asarray(at, float)


def price(kind, form, days, day):
    e = c.env(100.0, c.on(day))
    return float(PRODUCTS[kind][1](params=MCParams(seed=11, num_paths=4096)).price(build(kind, form, days, day), e))


if __name__ == "__main__":
    for kind in PRODUCTS:
        for sname, days in SCHEDULES.items():
            flagged, node_diff, errors = [], [], []
            for day in range(1, 365):
                g = {}
                for form in ("dated", "numeric"):
                    try:
                        at = grid(kind, form, days, day)
                    except Exception as exc:  # noqa: BLE001
                        errors.append((day, form, f"{type(exc).__name__}: {str(exc)[:80]}"))
                        continue
                    steps = np.diff(np.concatenate([[0.0], at]))
                    g[form] = (at.size, float(steps.min()))
                    if steps.min() < TOL:
                        flagged.append((day, form))
                if len(g) == 2 and g["dated"][0] != g["numeric"][0]:
                    node_diff.append((day, g["dated"][0], g["numeric"][0]))
            print(f"### {kind} / {sname}: sub-1e-10 steps {flagged}; node counts differ {node_diff}; "
                  f"errors {len(errors)} {errors[:3]}")
            for day in sorted({d for d, _ in flagged} | {45, 92, 200}):
                d, n = price(kind, "dated", days, day), price(kind, "numeric", days, day)
                print(f"   day {day:3d}: dated={d:.10f} numeric={n:.10f} gap={d - n:+.3e}", flush=True)
