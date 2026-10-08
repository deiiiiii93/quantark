"""Measure the tracker clock mismatch (findings 2026-10-09). Read-only: changes no library behaviour.

A numeric snowball and a numeric Phoenix (12 records at k/12, 1y), ``initial_date`` N days before the
tracker's ``start_date`` (N = 0, 30, 100). For each record k:

* ``booked``   -- the run day (offset from start_date) on which ``AutocallableLifecycleTracker.observe`` first
                  marks record k observed (and, for the Phoenix, books coupon k), spot held at 100 (no KO, coupon
                  paid every period).
* ``copy_due`` -- the run day on which ``product_for_pricing`` stops holding record k (the copy's own placement
                  of the record), and ``copy_left`` the record's remaining time in the copy, in days, on ``booked``.

Then the PV effect: the tracker's pricing copy (aged from start_date) priced with the PDE solver against the copy
the tracker's EVENT clock implies (aged from initial_date), on a few dates.

Run:
  env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python measure_clock_mismatch.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

START = pd.Timestamp("2026-01-05")
NS = (0, 30, 100)
HORIZON = 400


def env_on(day, spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
                              vol_surface=FlatVolSurface(volatility=0.20), rate_curve=FlatRateCurve(rate=0.03),
                              valuation_date=pd.Timestamp(day).to_pydatetime())


def snowball(initial_date):
    p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15)
    p.initial_date = None if initial_date is None else initial_date.to_pydatetime()
    p.validate()
    return p


def phoenix(initial_date):
    p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                coupon_rate=0.01, num_observations=12, memory_coupon=True)
    p.initial_date = None if initial_date is None else initial_date.to_pydatetime()
    p.validate()
    return p


BUILD = {"snowball": snowball, "phoenix": phoenix}
SOLVER = {"snowball": SnowballPDESolver, "phoenix": PhoenixPDESolver}


def ko_times(product):
    return [float(r.observation_time) for r in product.barrier_config.ko_observation_schedule.records]


def walk(kind, n, spot=100.0):
    """Day-by-day observe; per record: booked day, copy's last-holding day, copy remaining days on booked day."""
    initial = START - timedelta(days=n)
    tracker = AutocallableLifecycleTracker(product=BUILD[kind](initial), quantity=1.0, start_date=START)
    times = ko_times(tracker.product)
    booked = {}
    coupons = {}
    ko_day = None
    maturity_day = None
    copy_has = {k: [] for k in range(len(times))}
    copy_left_on = {}
    for d in range(HORIZON):
        day = START + timedelta(days=d)
        env = env_on(day, spot)
        before = set(tracker.lifecycle.observed_ko_indices)
        events = tracker.observe(day, tracker.product_for_lifecycle(), env, spot)
        for e in events:
            if e.event_type.value == "COUPON":
                coupons[e.observation_index] = d
            if e.event_type.value == "KO" and ko_day is None:
                ko_day = (d, e.observation_index, e.payoff, e.realized_cashflow.determination_time)
        for k in set(tracker.lifecycle.observed_ko_indices) - before:
            booked[k] = d
        if tracker.lifecycle.alive:
            m = tracker.settle_maturity_if_due(day, tracker.product_for_lifecycle(), env, spot)
            if m is not None:
                maturity_day = d
        # the post-close copy (initial=True on the run's first day, as register_positions does)
        copy = tracker.product_for_pricing(day, env, initial=(d == 0))
        left = [float(r.observation_time) for r in copy.barrier_config.ko_observation_schedule.records]
        dropped = len(times) - len(left)
        for k in range(len(times)):
            if k >= dropped:
                copy_has[k].append(d)
                if booked.get(k) == d:
                    copy_left_on[k] = round(left[k - dropped] * 365.0, 6)
        if not tracker.lifecycle.alive and maturity_day is None and ko_day is not None:
            break
    rows = []
    for k, t in enumerate(times):
        last_held = max(copy_has[k]) if copy_has[k] else None
        rows.append({
            "record": k,
            "contract_day": round(t * 365.0),
            "booked_day": booked.get(k),
            "coupon_day": coupons.get(k),
            "copy_last_holding_day": last_held,
            "copy_remaining_days_on_booked_day": copy_left_on.get(k, 0.0 if booked.get(k) is not None else None),
            "lead_days": (None if booked.get(k) is None or last_held is None else last_held - booked[k]),
        })
    return {"kind": kind, "N": n, "rows": rows, "maturity_settled_day": maturity_day, "ko": ko_day}


def aged_from(tracker, day, origin, env):
    """The copy the event clock implies: ``product_for_pricing`` with ``start_date`` temporarily = ``origin``."""
    saved = tracker.start_date
    try:
        tracker.start_date = origin
        return tracker.product_for_pricing(day, env)
    finally:
        tracker.start_date = saved


def pv_effect(kind, n, offsets=(0, 15, 45, 180, 300)):
    initial = START - timedelta(days=n)
    tracker = AutocallableLifecycleTracker(product=BUILD[kind](initial), quantity=1.0, start_date=START)
    solver = SOLVER[kind](PDEParams())
    out = []
    for off in offsets:
        day = START + timedelta(days=off)
        env = env_on(day)
        copy = tracker.product_for_pricing(day, env, initial=(off == 0))
        event_copy = aged_from(tracker, day, initial, env)
        pv_copy = float(solver.price(copy, env))
        pv_event = float(solver.price(event_copy, env))
        out.append({
            "offset_days": off,
            "copy_maturity_days": round(float(copy.maturity) * 365.0, 4),
            "event_clock_maturity_days": round(float(event_copy.maturity) * 365.0, 4),
            "copy_records": copy.barrier_config.ko_observation_count,
            "event_clock_records": event_copy.barrier_config.ko_observation_count,
            "pv_copy": pv_copy,
            "pv_event_clock": pv_event,
            "diff": pv_copy - pv_event,
            "diff_bp_of_notional": (pv_copy - pv_event) / 100.0 * 1e4,
        })
    return {"kind": kind, "N": n, "rows": out}


def ko_path(kind, n):
    """Spot 104 every day (above KO 103): (run day, record, payoff per unit, ledger determination_time) of the KO,
    against record 0's day on the copy's clock (its contract day)."""
    res = walk(kind, n, spot=104.0)
    return {"kind": kind, "N": n, "ko_day_record_payoff_dettime": res["ko"],
            "copy_first_record_day": res["rows"][0]["contract_day"]}


def guarded(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # option C refuses; record it as the arm's outcome
        return {"kind": args[0], "N": args[1], "refused": f"{type(exc).__name__}: {exc}", "rows": []}


def main(out_dir: Path):
    mode = os.environ.get("TRACKER_CLOCK_OPTION", "")
    if mode:
        import tracker_clock_options  # noqa: F401  -- patches the tracker for this process (A / B / C)
    walks = [guarded(walk, k, n) for k in BUILD for n in NS]
    pvs = [pv_effect(k, n) for k in BUILD for n in NS] if not mode or mode == "NONE" else []
    kos = [guarded(ko_path, k, n) for k in BUILD for n in NS]
    data = {"mode": mode or "current", "start_date": str(START.date()), "walks": walks, "pv": pvs, "ko_paths": kos}
    name = f"measure_clock_mismatch{'_' + mode if mode and mode != 'NONE' else ''}.json"
    (out_dir / name).write_text(json.dumps(data, indent=1, default=str))
    for w in walks:
        if "refused" in w:
            print(f"\n## {w['kind']} N={w['N']}  REFUSED: {w['refused']}")
            continue
        print(f"\n## {w['kind']} N={w['N']}  maturity settled on run day {w['maturity_settled_day']}")
        print("rec contract_day booked_day coupon_day copy_last_day copy_left_on_booked lead")
        for r in w["rows"]:
            print(r["record"], r["contract_day"], r["booked_day"], r["coupon_day"], r["copy_last_holding_day"],
                  r["copy_remaining_days_on_booked_day"], r["lead_days"])
    for p in pvs:
        print(f"\n## PV {p['kind']} N={p['N']}")
        for r in p["rows"]:
            print(r)
    for k in kos:
        print(k)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
