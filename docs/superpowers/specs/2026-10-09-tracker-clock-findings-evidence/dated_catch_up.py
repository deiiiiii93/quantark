"""Control for the findings 2026-10-09: the same contract on DATES, issued N days before the run.

The dated contract has one clock (its record dates), so its copy and its events agree. It shows what option A
would make a numeric contract do: records dated before ``start_date`` are all booked on the run's first close,
tested against that close (no history filter in the tracker; a caller that wants them as history must carry them
in ``observed_*_indices``). Read-only.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule

from measure_clock_mismatch import START, env_on, snowball


def dated_snowball(initial):
    p = snowball(initial)
    days = [round(r.observation_time * 365.0) for r in p.barrier_config.ko_observation_schedule.records]
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None, ko_observation_schedule=ObservationSchedule(
        records=[ObservationRecord(observation_date=(initial + timedelta(days=d)).to_pydatetime(), barrier=103.0)
                 for d in days]))
    p.maturity = None
    p.exercise_date = (initial + timedelta(days=365)).to_pydatetime()
    p.validate()
    return p


def main():
    for n in (0, 30, 100):
        initial = START - timedelta(days=n)
        t = AutocallableLifecycleTracker(product=dated_snowball(initial), quantity=1.0, start_date=START)
        booked = {}
        for d in range(400):
            day = START + timedelta(days=d)
            e = env_on(day)
            before = set(t.lifecycle.observed_ko_indices)
            t.observe(day, t.product_for_lifecycle(), e, 100.0)
            for k in set(t.lifecycle.observed_ko_indices) - before:
                booked[k] = d
            if t.lifecycle.alive and t.settle_maturity_if_due(day, t.product_for_lifecycle(), e, 100.0) is not None:
                booked["maturity"] = d
            if not t.lifecycle.alive:
                break
        copy0 = t.product_for_pricing(START, env_on(START), initial=True)
        print(f"dated N={n}: booked run days {booked}; day-0 copy records={copy0.barrier_config.ko_observation_count}")


if __name__ == "__main__":
    main()
