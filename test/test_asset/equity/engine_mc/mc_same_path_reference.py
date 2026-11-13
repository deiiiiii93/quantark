"""Same-path references for the MC Snowball / KO-reset event statistics (spec T9; the cash-leg bank's checks).

Each reference runs a plain ``price()`` of a single-batch route with a spy on the payoff kernel, takes the exact
paths (and bridge seed) the price used, and decides every outcome here from the definitions:

- Snowball: the first knock-out row, the first knock-in (discrete: the product's KI rows; continuous: the
  engine's own Brownian-bridge crossing step, a level breached at valuation is a knock-in at 0), and whether the
  knock-in came before any knock-out (a knock-out wins a tie with a DISCRETE knock-in at the same instant; a
  continuous touch inside a step precedes the observation closing it, except at valuation); under
  ``disable_ko_after_ki`` a knock-out after the knock-in is void;
- KO-reset: the knock-in, both schedules' knock-outs and the outcome decided from the paths, cross-checked
  against the kernel's flags; the cash is the kernel's per-path arithmetic (the price's).

The continuous knock-in table is rebuilt by brute force: each step's mass split over the days it overlaps,
proportionally to the overlap (spec D18), on a trading-day axis computed here.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.numerical import Tolerance

__all__ = ["day_overlap", "dcn_paths", "ko_reset_paths", "numeric_trading_days", "phoenix_paths", "snowball_paths"]


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
    if product.barrier_config.disable_ko_after_ki:
        # a knock-out after the knock-in is void: the note runs on, so it never knocks out (every later knock-out
        # follows the knock-in too); a knock-out before it stands
        is_ko = is_ko & ~ki_first
        out.update(is_ko=is_ko, ko_row=np.where(is_ko, ko_row, -1), ko_time=np.where(is_ko, ko_time, np.inf))
    out.update(ki_ever=hit, ki_first=ki_first, ki_index=index, ki_time=ki_time, continuous=continuous,
               breached_now=bool(continuous and not carried and np.any(ki_time[ki_first] == 0.0)))
    return out


def ko_reset_paths(engine, product, env) -> Dict[str, np.ndarray]:
    """Per path of a single-batch KO-reset price: every outcome decided here from the paths and the contract's rules.

    The kernel's paths, grid and bridge seed come from a spy on ``_compute_payoffs_ko_reset``. Decided here:

    - knock-in: the grid's KI rows (discrete) or the engine's own bridge crossing step over the first schedule's
      window (continuous; a level breached at valuation is a knock-in at 0); a carried knock-in is in force before
      today (time ``-inf``);
    - first-schedule knock-out: its first row at or through its level, valid when it is not preceded by the
      knock-in (it wins a tie with a DISCRETE knock-in; a continuous touch inside a step precedes the observation
      closing it, except at valuation);
    - second-schedule knock-out (knocked in, no first-schedule knock-out, ``disable_ko_after_ki`` off): ABSOLUTE its
      rows from the knock-in instant on, that instant included; REBASED the rows hung from the knock-in record;
    - the outcome (``ko_pre`` / ``ko_post`` / ``ki_settlement`` / ``maturity``) and its determination time.

    The decided outcome flags are asserted equal to the kernel's (an independent cross-check); the cash
    (``payoffs``, ``settle``) is the kernel's own per-path arithmetic, the price's.
    """
    from quantark.util.enum.option_enums import PostKOScheduleMode

    calls = _spy(engine, "_compute_payoffs_ko_reset")
    try:
        engine.price(product, env)
    finally:
        del engine.__dict__["_compute_payoffs_ko_reset"]
    if len(calls) != 1:
        raise AssertionError(f"a single-batch route prices one kernel call, got {len(calls)}")
    args, kwargs, (payoffs, settle, _stats, extra) = calls[0]
    names = ("product", "pricing_env", "paths", "grid", "r", "T", "sigma", "rng_seed")
    call = {**dict(zip(names, args)), **kwargs}
    paths, grid, sigma, rng_seed = call["paths"], call["grid"], call["sigma"], call["rng_seed"]
    n = paths.shape[0]
    reverse = bool(product.is_reverse)
    all_times = np.asarray(grid["all_times"], dtype=float)
    continuous = bool(grid["ki_continuous"])
    carried = bool(grid["carried_knock_in"])

    def through(spots, levels):
        return spots <= levels if reverse else spots >= levels

    # knock-in
    if carried:
        knocked, ki_index, ki_time = np.ones(n, bool), np.full(n, -1), np.full(n, -np.inf)
        steps = np.zeros(0)
    elif continuous:
        horizon = int(grid["ki_horizon_idx"])
        steps = all_times[: horizon + 1]
        level = float(np.asarray(product.barrier_config.ki_barrier).reshape(-1)[0])
        knocked, ki_index = engine._check_ki_barriers_continuous_with_bridge(
            paths=paths[:, : horizon + 2], all_times=steps, ki_barrier=level, sigma=float(sigma),
            is_reverse=reverse, rng_seed=int(rng_seed), horizon_steps=horizon + 1)
        ki_time = np.where(knocked, steps[np.maximum(ki_index, 0)], np.inf)
        now = (paths[:, 0] >= level) if reverse else (paths[:, 0] <= level)
        ki_time[knocked & now] = 0.0
    else:
        steps = np.asarray(grid["ki_times"], dtype=float)
        ki_spots = paths[:, np.asarray(grid["ki_indices"], dtype=int) + 1]
        ki_levels = np.asarray(grid["ki_barriers"], dtype=float)
        knocked, ki_index = _first(ki_spots >= ki_levels if reverse else ki_spots <= ki_levels)
        ki_time = np.where(knocked, steps[np.maximum(ki_index, 0)], np.inf)

    # first schedule
    pre_times = np.asarray(grid["pre_times"], dtype=float)
    pre_levels = np.asarray(grid["pre_profile"]["barriers"], dtype=float)
    pre_hit, pre_row = _first(through(paths[:, np.asarray(grid["pre_indices"], dtype=int) + 1], pre_levels))
    pre_time = np.where(pre_hit, pre_times[np.maximum(pre_row, 0)], np.inf)
    if continuous:
        ko_first = (pre_time < ki_time) | ((pre_time == 0.0) & (ki_time == 0.0))
    else:
        ko_first = pre_time <= ki_time
    is_pre = pre_hit & ko_first

    # second schedule
    mode = grid["post_profile"]["mode"]
    post_levels = np.asarray(grid["post_profile"]["barriers"], dtype=float)
    post_times = np.asarray(grid["post_times"], dtype=float)
    allowed = knocked & ~is_pre & (not product.barrier_config.disable_ko_after_ki)
    is_post = np.zeros(n, bool)
    post_time = np.full(n, np.inf)
    post_end = np.full(n, np.inf)
    if mode is PostKOScheduleMode.ABSOLUTE:
        post_end[:] = float(product.get_post_maturity_time(env))
        if post_times.size:
            hit = through(paths[:, np.asarray(grid["post_indices"], dtype=int) + 1], post_levels)
            in_force = (post_times[None, :] > ki_time[:, None]) | np.isclose(
                post_times[None, :], ki_time[:, None], rtol=0.0, atol=Tolerance.ZERO)
            is_post, row = _first(hit & in_force)
            is_post &= allowed
            post_time = np.where(is_post, post_times[np.maximum(row, 0)], np.inf)
    else:
        last = int(np.argmax(post_times))
        for p in np.flatnonzero(knocked & (ki_index >= 0)):
            k = int(ki_index[p])
            times_k = np.asarray(grid["post_times_by_ki"][k], dtype=float)
            post_end[p] = times_k[last]
            if not allowed[p]:
                continue
            hit = through(paths[p, np.asarray(grid["post_indices_by_ki"][k], dtype=int) + 1], post_levels)
            if hit.any():
                is_post[p], post_time[p] = True, times_k[int(np.argmax(hit))]

    v1 = knocked & ~is_pre & ~is_post
    v0 = ~knocked & ~is_pre & ~is_post
    for name, mine, kernel in (("is_pre_ko", is_pre, extra["is_pre_ko"]), ("is_post_ko", is_post, extra["is_post_ko"]),
                               ("ki_triggered", knocked, extra["ki_triggered"])):
        if not np.array_equal(mine, kernel):
            raise AssertionError(f"KO-reset reference disagrees with the kernel on {name}: "
                                 f"{int(np.count_nonzero(mine != kernel))} paths")
    event = np.where(is_pre, "ko_pre", np.where(is_post, "ko_post", np.where(v1, "ki_settlement", "maturity")))
    pre_maturity = float(product.get_pre_maturity_time(env))
    det = np.where(is_pre, pre_time, np.where(is_post, post_time, np.where(v1, post_end, pre_maturity)))
    return dict(n=n, payoffs=payoffs, settle=settle, event=event, det=det, v0=v0, v1=v1,
                ki_first=knocked & ~is_pre, ki_index=ki_index, ki_time=ki_time, continuous=continuous,
                step_ends=steps, is_pre=is_pre, is_post=is_post, pre_maturity=pre_maturity, grid=grid)


def phoenix_paths(engine, product, env) -> Dict[str, np.ndarray]:
    """Per path of a single-batch Phoenix price, every outcome decided here from the definitions.

    The kernel's paths, grid and bridge seed are taken from a spy on ``_compute_payoffs``; nothing else of the
    kernel is read. Knock-out: the first row at or through its level; knock-in: the product's KI rows (discrete) or
    the engine's own bridge crossing step (continuous; a level breached at valuation is a knock-in at 0); a
    knock-out wins a tie with a DISCRETE knock-in, a continuous touch inside a step precedes the observation
    closing it (except at valuation). ``disable_ko_after_ki``: a knock-out after the knock-in is void.

    Coupons by a plain per-path loop (spec AC1): memory arrears start at ``initial_coupon_arrears`` (memory only),
    a coupon hit releases the current coupon plus the arrears, a knock-out on a missed coupon releases the
    arrears, a miss on a live row adds the current coupon. INSTANT pays each row at its settlement; EXPIRY pays the
    rolled-up coupons when the note ends (the knock-out's settlement, else the terminal payment).
    """
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
    paths, all_times, sigma, rng_seed = call["paths"], np.asarray(call["all_times"], dtype=float), call["sigma"], \
        call["rng_seed"]
    ko_indices, ki_indices = np.asarray(call["ko_indices"]), np.asarray(call["ki_indices"])
    reverse = bool(product.is_reverse)
    n = paths.shape[0]
    profile = product.get_ko_observation_profile(env)
    ko_times = np.asarray(profile["observation_times"], dtype=float)
    barriers = np.asarray(profile["barriers"], dtype=float)
    redemption = np.asarray(profile["payoffs"], dtype=float)
    records = product.resolve_ko_observations(env)
    settle = np.asarray([r.settlement_time for r in records], dtype=float)
    ko_spots = paths[:, ko_indices]
    is_ko_raw, ko_row = _first(ko_spots <= barriers if reverse else ko_spots >= barriers)
    ko_time = np.where(is_ko_raw, ko_times[np.maximum(ko_row, 0)], np.inf)

    carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False)) and product.has_ki_barrier
    continuous = False
    ki_ever = np.zeros(n, bool)
    ki_index = np.full(n, -1)
    ki_time = np.full(n, np.inf)
    if product.has_ki_barrier:
        continuous = (product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
                      or product.barrier_config.ki_continuous)
        if continuous:
            level = float(np.asarray(product.barrier_config.ki_barrier).reshape(-1)[0])
            ki_ever, ki_index = engine._check_ki_barriers_continuous_with_bridge(
                paths=paths, all_times=all_times, ki_barrier=level, sigma=float(sigma), is_reverse=reverse,
                rng_seed=int(rng_seed))
            ki_time = np.where(ki_ever, all_times[np.maximum(ki_index, 0)], np.inf)
            now = (paths[:, 0] >= level) if reverse else (paths[:, 0] <= level)
            ki_time[ki_ever & now] = 0.0
        else:
            ki_profile = product.get_ki_observation_profile(env)
            rows = np.asarray(ki_profile["observation_times"], dtype=float)
            spots = paths[:, ki_indices]
            levels = np.asarray(ki_profile["barriers"], dtype=float)
            ki_ever, ki_index = _first(spots >= levels if reverse else spots <= levels)
            ki_time = np.where(ki_ever, rows[np.maximum(ki_index, 0)], np.inf)
    if carried:
        ki_ever, ki_time = np.ones(n, bool), np.zeros(n)
    if continuous:
        ki_strictly_first = (ki_time < ko_time) | ((ki_time == ko_time) & (ko_time > 0.0))
    else:
        ki_strictly_first = ki_time < ko_time
    if carried:
        ki_strictly_first = np.ones(n, bool)
    is_ko = is_ko_raw.copy()
    if product.barrier_config.disable_ko_after_ki and product.has_ki_barrier:
        is_ko &= ~(ki_ever & ki_strictly_first)
    ki_first = ki_ever & (~is_ko | ki_strictly_first)
    v0 = ~is_ko & ~ki_ever
    v1 = ~is_ko & ki_ever

    df = env.get_discount_factor
    terminal = engine._payment_timings.terminal
    instant = product.coupon_config.coupon_pay_type == CouponPayType.INSTANT
    memory = bool(product.coupon_config.memory_coupon)
    periods = product.get_coupon_period_year_fractions(env)
    amounts = [product.get_coupon_payoff(i, year_fraction=periods[i]) for i in range(ko_times.size)]
    rows = ko_times.size
    coupon_hit = np.zeros((n, rows), bool)
    coupon_paid = np.zeros((n, rows))
    for p in range(n):
        accrued = float(product.coupon_config.initial_coupon_arrears) if memory else 0.0
        for i in range(rows):
            if is_ko[p] and ko_row[p] < i:
                break
            hit = product.is_coupon_triggered(float(ko_spots[p, i]), i)
            ko_here = bool(is_ko[p] and ko_row[p] == i)
            coupon_hit[p, i] = hit
            if hit:
                coupon_paid[p, i] = amounts[i] + (accrued if memory else 0.0)
                accrued = 0.0
            elif memory and ko_here:
                coupon_paid[p, i] = accrued
                accrued = 0.0
            elif memory:
                accrued += amounts[i]
    end_row = np.where(is_ko, ko_row, rows)                       # rows = the terminal bucket
    end_pay = np.where(is_ko, settle[np.maximum(ko_row, 0)], float(terminal.payment_time))
    end_df = np.array([df(float(t)) for t in end_pay])
    if instant:
        coupon_df = np.tile(np.array([df(float(t)) for t in settle]), (n, 1))
    else:
        coupon_df = np.repeat(end_df[:, None], rows, axis=1)
    path_coupons = coupon_paid.sum(axis=1)
    maturity_cash = np.zeros(n)
    for p in np.flatnonzero(v0):
        maturity_cash[p] = product.get_maturity_payoff_v0(float(paths[p, -1]), pricing_env=env)
    for p in np.flatnonzero(v1):
        maturity_cash[p] = product.get_maturity_payoff_v1(float(paths[p, -1]), pricing_env=env)
    terminal_df = df(float(terminal.payment_time))
    return dict(
        n=n, paths=paths, ko_indices=ko_indices, is_ko=is_ko, ko_row=ko_row, ko_time=ko_time,
        ko_redemption=np.where(is_ko, redemption[np.maximum(ko_row, 0)], 0.0),
        ko_settle_df=np.where(is_ko, np.array([df(float(t)) for t in settle])[np.maximum(ko_row, 0)], 0.0),
        ki_ever=ki_ever, ki_first=ki_first, ki_index=ki_index, ki_time=ki_time, continuous=continuous,
        carried=carried, step_ends=all_times, v0=v0, v1=v1, maturity_cash=maturity_cash, terminal_df=terminal_df,
        coupon_hit=coupon_hit, coupon_paid=coupon_paid, coupon_disc=coupon_paid * coupon_df,
        end_row=end_row, end_df=end_df, path_coupons=path_coupons,
    )


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


def dcn_paths(engine, product, env, *, lifecycle_state=None) -> Dict[str, np.ndarray]:
    """Per path of a DCN price (T11): the paths ``_simulate`` produced for every batch (batch order), and every
    outcome decided here from the contract's rules, one path at a time over its own closes:

    - knock-in: the daily closes from the valuation close (when it is a schedule date) on, ``S <= KI``; a note
      already knocked in is knocked in at column 0; monitoring stops once the note has knocked out;
    - knock-out: the first KO-eligible monthly row with ``S >= KO`` (a knock-in on the same close is tested first);
    - fixed coupon: a coupon-eligible row of an alive note that does not knock out there, not knocked in by that
      close, ``S >= coupon barrier``;
    - maturity: never knocked out; the loss leg ``-(N / S0) x participation x max(K_loss - S_T, 0)`` when
      knocked in.
    """
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    calls = _spy(engine, "_simulate")
    try:
        engine.price(product, env, lifecycle_state=lifecycle_state)
    finally:
        del engine.__dict__["_simulate"]
    order = sorted(range(len(calls)), key=lambda i: -1 if calls[i][1].get("batch_id") is None
                   else int(calls[i][1]["batch_id"]))
    paths = np.concatenate([calls[i][2] for i in order], axis=0)
    ctx = build_dcn_grid_context(product)
    n, n_obs = paths.shape[0], ctx.obs_cols.size
    first_col = 0 if ctx.valuation_monitored else 1
    carried = bool(product.knocked_in_at_valuation)
    ko_row = np.full(n, -1)
    ki_col = np.full(n, -1)
    coupon_paid = np.zeros((n, n_obs), dtype=bool)
    for p in range(n):
        s = paths[p]
        knocked_in = carried
        if carried:
            ki_col[p] = 0
        col = first_col
        for j in range(n_obs):
            c = int(ctx.obs_cols[j])
            while col <= c:
                if not knocked_in and s[col] <= product.ki_barrier:
                    knocked_in, ki_col[p] = True, col
                col += 1
            if ctx.obs_is_ko[j] and s[c] >= product.ko_barrier:
                ko_row[p] = j
                break
            if ctx.obs_is_coupon[j] and not knocked_in and s[c] >= product.coupon_barrier:
                coupon_paid[p, j] = True
    survived = ko_row < 0
    knocked_in = ki_col >= 0
    s_t = paths[:, int(ctx.obs_cols[-1])]
    loss = np.where(survived & knocked_in,
                    -(product.notional / product.initial_price) * product.participation
                    * np.maximum(product.k_loss - s_t, 0.0), 0.0)
    return dict(n=n, paths=paths, ctx=ctx, first_col=first_col, carried=carried, ko_row=ko_row, ki_col=ki_col,
                coupon_paid=coupon_paid, maturity=survived & ~knocked_in, ki_settlement=survived & knocked_in,
                loss=loss)
