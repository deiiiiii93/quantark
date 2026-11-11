"""Same-path references for the MC Snowball / KO-reset event statistics (spec T9; the cash-leg bank's checks).

Each reference runs a plain ``price()`` of a single-batch route with a spy on the payoff kernel, takes the exact
paths (and bridge seed) the price used, and decides every outcome here from the definitions:

- Snowball: the first knock-out row, the first knock-in (discrete: the product's KI rows; continuous: the
  engine's own Brownian-bridge crossing step, a level breached at valuation is a knock-in at 0), and whether the
  knock-in came before any knock-out (a knock-out wins a tie with a DISCRETE knock-in at the same instant; a
  continuous touch inside a step precedes the observation closing it, except at valuation);
- KO-reset: the kernel's own per-path outcome (its payoffs, settlements and flags, the price's arithmetic).

The continuous knock-in table is rebuilt by brute force: each step's mass split over the days it overlaps,
proportionally to the overlap (spec D18), on a trading-day axis computed here.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from quantark.util.enum import ObservationType

__all__ = ["day_overlap", "ko_reset_paths", "numeric_trading_days", "snowball_paths"]


def _spy(engine, name):
    """Record every call of ``engine.<name>`` (an instance attribute shadowing the method; delete it after)."""
    calls = []
    original = getattr(engine, name)

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    setattr(engine, name, spy)
    return calls


def _first(hit: np.ndarray):
    any_hit = hit.any(axis=1)
    return any_hit, np.where(any_hit, np.argmax(hit, axis=1), -1)


def snowball_paths(engine, product, env) -> Dict[str, np.ndarray]:
    """Per path of a single-batch Snowball price: KO row / time, first KI (row or step) / time, KI-before-KO."""
    calls = _spy(engine, "_compute_payoffs")
    try:
        engine.price(product, env)
    finally:
        del engine.__dict__["_compute_payoffs"]
    if len(calls) != 1:
        raise AssertionError(f"a single-batch route prices one kernel call, got {len(calls)}")
    args, kwargs, _ = calls[0]
    names = ("product", "pricing_env", "paths", "all_times", "ko_indices", "ki_indices", "r", "T", "sigma",
             "rng_seed")
    call = {**dict(zip(names, args)), **kwargs}
    paths, all_times, sigma, rng_seed = call["paths"], call["all_times"], call["sigma"], call["rng_seed"]
    ko_indices, ki_indices = call["ko_indices"], call["ki_indices"]
    reverse = bool(product.is_reverse)
    all_times = np.asarray(all_times, dtype=float)
    profile = product.get_ko_observation_profile(env)
    ko_times = np.asarray(profile["observation_times"], dtype=float)
    ko_spots = paths[:, np.asarray(ko_indices) + 1]
    barriers = np.asarray(profile["barriers"], dtype=float)
    is_ko, ko_row = _first(ko_spots <= barriers if reverse else ko_spots >= barriers)
    ko_time = np.where(is_ko, ko_times[np.maximum(ko_row, 0)], np.inf)
    n = paths.shape[0]
    carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    out = dict(n=n, is_ko=is_ko, ko_row=ko_row, ko_time=ko_time, step_ends=all_times, carried=carried)
    if not product.has_ki_barrier:
        out.update(ki_ever=np.zeros(n, bool), ki_first=np.zeros(n, bool), ki_index=np.full(n, -1), continuous=False)
        return out
    level = float(np.asarray(product.barrier_config.ki_barrier).reshape(-1)[0])
    continuous = (product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                  or product.barrier_config.ki_continuous)
    if continuous:
        hit, step = engine._check_ki_barriers_continuous_with_bridge(
            paths=paths, all_times=all_times, ki_barrier=level, sigma=float(sigma), is_reverse=reverse,
            rng_seed=int(rng_seed))
        ki_time = np.where(hit, all_times[np.maximum(step, 0)], np.inf)
        now = (paths[:, 0] >= level) if reverse else (paths[:, 0] <= level)
        ki_time[hit & now] = 0.0
        ki_first = hit & ((ki_time < ko_time) | ((ki_time == ko_time) & (ko_time > 0.0)))
        index = step
    else:
        ki_profile = product.get_ki_observation_profile(env)
        rows = np.asarray(ki_profile["observation_times"], dtype=float)
        spots = paths[:, np.asarray(ki_indices) + 1]
        ki_levels = np.asarray(ki_profile["barriers"], dtype=float)
        hit, index = _first(spots >= ki_levels if reverse else spots <= ki_levels)
        ki_time = np.where(hit, rows[np.maximum(index, 0)], np.inf)
        ki_first = hit & (ki_time < ko_time)
    if carried:
        hit, ki_first, ki_time = np.ones(n, bool), np.ones(n, bool), np.zeros(n)
    out.update(ki_ever=hit, ki_first=ki_first, ki_index=index, ki_time=ki_time, continuous=continuous,
               breached_now=bool(continuous and not carried and np.any(ki_time[ki_first] == 0.0)))
    return out


def ko_reset_paths(engine, product, env) -> Dict[str, np.ndarray]:
    """Per path of a single-batch KO-reset price, from the kernel's own outputs: outcome, determination and
    settlement time, undiscounted payoff, first KI (row / step index) and whether it preceded any knock-out."""
    calls = _spy(engine, "_compute_payoffs_ko_reset")
    try:
        engine.price(product, env)
    finally:
        del engine.__dict__["_compute_payoffs_ko_reset"]
    if len(calls) != 1:
        raise AssertionError(f"a single-batch route prices one kernel call, got {len(calls)}")
    args, kwargs, (payoffs, settle, _stats, extra) = calls[0]
    grid = kwargs["grid"] if "grid" in kwargs else args[3]
    is_pre, is_post = extra["is_pre_ko"], extra["is_post_ko"]
    knocked = extra["ki_triggered"]
    v1 = knocked & ~is_pre & ~is_post
    v0 = ~knocked & ~is_pre & ~is_post
    event = np.where(is_pre, "ko_pre", np.where(is_post, "ko_post", np.where(v1, "ki_settlement", "maturity")))
    pre_maturity = float(product.get_pre_maturity_time(env))
    post_end = extra["post_end_time"]
    if grid["post_profile"]["mode"].name == "ABSOLUTE":
        post_end = np.full(payoffs.size, float(product.get_post_maturity_time(env)))
    det = np.where(is_pre | is_post, extra["ko_time"], np.where(v1, post_end, pre_maturity))
    ki_time = extra["ki_time"]
    if grid["ki_continuous"]:
        steps = np.asarray(grid["all_times"][: grid["ki_horizon_idx"] + 1], dtype=float)
        ki_index = np.where(np.isfinite(ki_time), np.searchsorted(steps, ki_time), -1)
    else:
        rows = np.asarray(grid["ki_times"], dtype=float)
        ki_index = np.where(np.isfinite(ki_time), np.searchsorted(rows, ki_time), -1)
        steps = rows
    return dict(n=payoffs.size, payoffs=payoffs, settle=settle, event=event, det=det, v0=v0, v1=v1,
                ki_first=knocked & ~is_pre, ki_index=ki_index, ki_time=ki_time, continuous=bool(grid["ki_continuous"]),
                step_ends=steps, is_pre=is_pre, is_post=is_post, pre_maturity=pre_maturity, grid=grid)


def numeric_trading_days(horizon: float, bus_days_in_year: float) -> np.ndarray:
    """Day ends k / bus_days_in_year below ``horizon``, then ``horizon`` (a numeric-clock contract, AC13)."""
    days = []
    k = 1
    while k / bus_days_in_year < horizon - 1e-10:
        days.append(k / bus_days_in_year)
        k += 1
    days.append(horizon)
    return np.asarray(days, dtype=float)


def day_overlap(step_ends: np.ndarray, step_mass: np.ndarray, day_ends: np.ndarray) -> np.ndarray:
    """D18 by brute force: each step's mass split over the days it overlaps, proportionally to the overlap."""
    out = np.zeros(day_ends.size)
    starts = np.concatenate([[0.0], step_ends[:-1]])
    day_starts = np.concatenate([[0.0], day_ends[:-1]])
    for a, b, m in zip(starts, step_ends, step_mass):
        if m == 0.0:
            continue
        for d, (lo, hi) in enumerate(zip(day_starts, day_ends)):
            overlap = min(b, hi) - max(a, lo)
            if overlap > 0.0:
                out[d] += m * overlap / (b - a)
    return out
