"""Shared single-barrier payoff / monitoring core for the vol-model MC and PDE engines.

Centralizes knock-out / knock-in cashflows (with ``rebate`` and ``pay_at_hit``) and the two
Monte-Carlo monitoring estimators — hard breach at discrete observation samples, and a
Brownian-bridge *survival weight* for continuous monitoring (which captures between-step
crossings that grid-node monitoring misses). Kept asset- and process-neutral: it operates on
simulated spots, so the LV / Heston / SLV kernels all reuse the same, tested, semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log


@dataclass(frozen=True)
class BarrierSpec:
    """Immutable description of a single-barrier option (payoff/monitoring only).

    ``participation_rate`` and ``observation_schedule`` are handled at the engine wrapper,
    not here, so this core stays payoff-agnostic.
    """

    is_up: bool
    is_out: bool
    is_call: bool
    barrier: float
    strike: float
    rebate: float
    pay_at_hit: bool


def disc_closure(step_dt: np.ndarray, r_fwd: np.ndarray):
    """Return (disc, node_times): disc(t)->DF(0->t) under piecewise-constant per-step rates;
    node_times holds the cumulative time at each grid node (length n_steps + 1)."""
    node_times = np.concatenate([[0.0], np.cumsum(step_dt)])
    cum_r = np.concatenate([[0.0], np.cumsum(np.asarray(r_fwd, float) * np.asarray(step_dt, float))])

    def disc(t):
        integ = np.interp(np.asarray(t, dtype=float), node_times, cum_r)
        return np.exp(-integ)

    return disc, node_times


def validate_barrier(spec: BarrierSpec, s0: float) -> None:
    if not np.isfinite(spec.barrier) or spec.barrier <= 0:
        raise ValidationError("barrier must be positive and finite")
    if spec.barrier == s0:
        raise ValidationError("barrier equal to spot is degenerate")
    if spec.rebate < 0:
        raise ValidationError("rebate must be non-negative")


def _vanilla(term: np.ndarray, spec: BarrierSpec) -> np.ndarray:
    if spec.is_call:
        return np.maximum(term - spec.strike, 0.0)
    return np.maximum(spec.strike - term, 0.0)


def mc_barrier_cashflows(terminal_s, survival_w, hit_cumT, spec: BarrierSpec, disc, maturity):
    """Per-path present value.

    Args:
        terminal_s: terminal spot per path.
        survival_w: per-path probability the path did NOT breach the barrier (in [0, 1]).
            Hard 0/1 for discrete monitoring; a Brownian-bridge product for continuous.
        hit_cumT: per-path first-hit time (year fraction); used only when ``pay_at_hit``.
        spec: BarrierSpec.
        disc: vectorized discount closure, disc(t) -> DF(0 -> t) for scalar or array t.
        maturity: option maturity T.

    Returns:
        per-path discounted cashflow (np.ndarray).
    """
    payoff = _vanilla(np.asarray(terminal_s, dtype=float), spec)
    w = np.clip(np.asarray(survival_w, dtype=float), 0.0, 1.0)
    df_t = float(disc(maturity))
    reb_df = np.asarray(disc(hit_cumT), dtype=float) if spec.pay_at_hit else df_t
    if spec.is_out:
        # survive -> option pays at T; breach -> rebate (at hit if pay_at_hit else at T)
        return w * payoff * df_t + (1.0 - w) * spec.rebate * reb_df
    # knock-in: breach -> option pays at T; survive -> rebate at expiry
    return (1.0 - w) * payoff * df_t + w * spec.rebate * df_t


def discrete_survival(samples: np.ndarray, spec: BarrierSpec):
    """Hard breach at discrete observation samples.

    Args:
        samples: spot at observation dates, shape (n_paths, n_obs).
        spec: BarrierSpec (uses is_up, barrier).

    Returns:
        (w, first_hit_col): w in {0.0, 1.0} per path (1 = survived); first_hit_col = index of
        the first breaching observation (n_obs if never), for pay_at_hit timing.
    """
    samples = np.asarray(samples, dtype=float)
    hit = samples >= spec.barrier if spec.is_up else samples <= spec.barrier
    breached = np.any(hit, axis=1)
    n_obs = samples.shape[1]
    first = np.where(breached, np.argmax(hit, axis=1), n_obs)
    return np.where(breached, 0.0, 1.0), first


def bridge_survival(nodes: np.ndarray, step_vol: np.ndarray, step_dt: np.ndarray, spec: BarrierSpec):
    """Continuous-monitoring survival weight via a per-step Brownian bridge.

    For a step with endpoints S_i, S_{i+1} and (frozen) local vol sigma_i over dt_i, the
    probability of NOT crossing barrier B in log-space is
        1 - exp(-2 * ln(B/S_i) * ln(B/S_{i+1}) / (sigma_i^2 dt_i))   when both endpoints are
    on the same side of B; a path with either endpoint already beyond B has survival 0 for
    that step. The path survival weight is the product over steps.

    Args:
        nodes: simulated spot at each grid node, shape (n_paths, n_steps + 1).
        step_vol: per-path per-step local vol used on that step, shape (n_paths, n_steps).
        step_dt: per-step dt, shape (n_steps,).
        spec: BarrierSpec.

    Returns:
        (w, first_hit_col): w in [0, 1] per path; first_hit_col = first step whose right node
        is beyond the barrier (n_steps if none), for pay_at_hit timing.
    """
    nodes = np.asarray(nodes, dtype=float)
    step_vol = np.asarray(step_vol, dtype=float)
    dt = np.asarray(step_dt, dtype=float)
    n_steps = step_vol.shape[1]
    B = spec.barrier
    s_i = nodes[:, :-1]
    s_j = nodes[:, 1:]
    var = np.maximum(step_vol * step_vol * dt[None, :], 1e-300)
    # log-distance to barrier at each endpoint (sign encodes side)
    di = safe_log(np.asarray(B / np.maximum(s_i, 1e-300)))
    dj = safe_log(np.asarray(B / np.maximum(s_j, 1e-300)))
    # bridge no-cross prob when both endpoints on same side (di, dj same sign)
    p_no_cross = 1.0 - np.exp(np.minimum(-2.0 * di * dj / var, 0.0))
    same_side = (di * dj) > 0.0
    # hard breach at a node (endpoint beyond barrier)
    beyond = (s_j >= B) if spec.is_up else (s_j <= B)
    start_beyond = (s_i >= B) if spec.is_up else (s_i <= B)
    step_surv = np.where(beyond | start_beyond, 0.0, np.where(same_side, p_no_cross, 1.0))
    w = np.prod(step_surv, axis=1)
    any_beyond = beyond | start_beyond
    breached = np.any(any_beyond, axis=1)
    first = np.where(breached, np.argmax(any_beyond, axis=1), n_steps)
    return w, first
