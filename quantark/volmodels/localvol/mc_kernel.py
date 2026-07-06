"""Monte Carlo pricer for European options under a LocalVolSurface (log-Euler).

dS/S = (r(t) - carry(t)) dt + sigma_LV(S, t) dW. Vol is sampled at the start of each
step; rates enter as per-step forwards and the price is discounted by the supplied
terminal discount factor. Asset-neutral: carry = dividend yield (equity) or foreign
rate (FX).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.barrier import (
    BarrierSpec, bridge_survival, discrete_survival, mc_barrier_cashflows, validate_barrier,
)


def _disc_closure(step_dt: np.ndarray, r_fwd: np.ndarray):
    """Return (disc, node_times): disc(t)->DF(0->t) piecewise-const-rate; node_times per grid node."""
    node_times = np.concatenate([[0.0], np.cumsum(step_dt)])
    cum_r = np.concatenate([[0.0], np.cumsum(r_fwd * step_dt)])  # integral of r to each node

    def disc(t):
        t = np.asarray(t, dtype=float)
        integ = np.interp(t, node_times, cum_r)
        return np.exp(-integ)

    return disc, node_times


def price_european_lv_mc(
    s0: float,
    strike: float,
    is_call: bool,
    lv_surface: LocalVolSurface,
    step_dt: np.ndarray,
    r_fwd: np.ndarray,
    carry_fwd: np.ndarray,
    disc_factor: float,
    num_paths: int = 50_000,
    seed: Optional[int] = 42,
    use_antithetic: bool = False,
    sampler=None,
    return_stderr: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Price a European vanilla under local volatility via Monte Carlo.

    Args:
        s0: spot at t=0 (> 0).
        strike, is_call: option spec.
        lv_surface: positive LocalVolSurface (sigma_LV(S, t)).
        step_dt, r_fwd, carry_fwd: equal-length per-step arrays; time at step i is the
            cumulative sum of step_dt[:i]. Drift over step i is r_fwd[i] - carry_fwd[i].
        disc_factor: discount factor to maturity DF(T) in (0, 1].
        num_paths, seed, use_antithetic, return_stderr: MC controls. With antithetic
            sampling the stderr is computed from pair-average payoffs.
        sampler (optional): a quantark.montecarlo generator exposing ``uniform(n, dim)``.
            QMC dimension layout: columns [z(M)] (one normal stream per step), ndtri-
            transformed. Mutually exclusive with ``use_antithetic``; default None keeps
            the pseudo path bit-identical.
    """
    dt = np.asarray(step_dt, dtype=float)
    rf = np.asarray(r_fwd, dtype=float)
    cf = np.asarray(carry_fwd, dtype=float)
    n = dt.size
    if n < 1 or rf.size != n or cf.size != n:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if not (np.all(np.isfinite(dt)) and np.all(dt > 0)):
        raise ValidationError("step_dt must be finite and positive")
    if not (np.all(np.isfinite(rf)) and np.all(np.isfinite(cf))):
        raise ValidationError("r_fwd and carry_fwd must be finite")
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if not np.isfinite(disc_factor) or disc_factor <= 0:
        raise ValidationError("disc_factor must be finite and positive")
    if num_paths <= 0:
        raise ValidationError("num_paths must be positive")

    half = (num_paths + 1) // 2
    n_eff = 2 * half if use_antithetic else num_paths

    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        block = np.clip(np.asarray(sampler.uniform(num_paths, n), dtype=float),
                        1e-12, 1.0 - 1e-12)
        z_all = ndtri(block)                 # (num_paths, M); one normal stream per step
    else:
        rng = np.random.default_rng(seed)
        z_all = None

    s = np.full(n_eff, float(s0), dtype=float)
    drift = rf - cf
    sqrt_dt = np.sqrt(dt)
    t = 0.0
    for i in range(n):
        sigma = np.asarray(lv_surface.local_vol(s, t), dtype=float)
        if z_all is not None:
            z = z_all[:, i]
        elif use_antithetic:
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
        else:
            z = rng.standard_normal(n_eff)
        s = s * np.exp((drift[i] - 0.5 * sigma * sigma) * dt[i] + sigma * sqrt_dt[i] * z)
        t += dt[i]

    payoff = np.maximum(s - strike, 0.0) if is_call else np.maximum(strike - s, 0.0)
    discounted = float(disc_factor) * payoff

    if use_antithetic:
        pair = 0.5 * (discounted[:half] + discounted[half:2 * half])
        price = float(np.mean(pair))
        if return_stderr:
            stderr = float(np.std(pair, ddof=1) / np.sqrt(half)) if half > 1 else 0.0
            return price, stderr
        return price

    price = float(np.mean(discounted))
    if return_stderr:
        stderr = float(np.std(discounted, ddof=1) / np.sqrt(num_paths)) if num_paths > 1 else 0.0
        return price, stderr
    return price


def price_barrier_lv_mc(
    s0: float,
    strike: float,
    is_call: bool,
    lv_surface: LocalVolSurface,
    step_dt: np.ndarray,
    r_fwd: np.ndarray,
    carry_fwd: np.ndarray,
    disc_factor: float,
    barrier: float,
    is_up: bool,
    is_out: bool,
    rebate: float = 0.0,
    pay_at_hit: bool = False,
    continuous: bool = True,
    observe_idx: Optional[np.ndarray] = None,
    participation: float = 1.0,
    num_paths: int = 50_000,
    seed: Optional[int] = 42,
    use_antithetic: bool = False,
    return_stderr: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Price a single-barrier option under local volatility via Monte Carlo.

    Same log-Euler simulation as ``price_european_lv_mc`` but records the path nodes and
    per-step vol so the shared barrier core can apply continuous (Brownian-bridge) or
    discrete monitoring. ``observe_idx`` (node indices) is required when ``continuous`` is
    False. See ``quantark.volmodels.barrier`` for the payoff/monitoring semantics.
    """
    dt = np.asarray(step_dt, dtype=float)
    rf = np.asarray(r_fwd, dtype=float)
    cf = np.asarray(carry_fwd, dtype=float)
    n = dt.size
    if n < 1 or rf.size != n or cf.size != n:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if not (np.all(np.isfinite(dt)) and np.all(dt > 0)):
        raise ValidationError("step_dt must be finite and positive")
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if num_paths <= 0:
        raise ValidationError("num_paths must be positive")
    spec = BarrierSpec(is_up=bool(is_up), is_out=bool(is_out), is_call=bool(is_call),
                       barrier=float(barrier), strike=float(strike), rebate=float(rebate),
                       pay_at_hit=bool(pay_at_hit))
    validate_barrier(spec, s0)
    if not continuous and observe_idx is None:
        raise ValidationError("discrete monitoring requires observe_idx")
    if continuous and pay_at_hit:
        raise ValidationError(
            "pay_at_hit=True is not supported with continuous (Brownian-bridge) MC monitoring: "
            "a bridge-only crossing has no sampled hit time. Use discrete monitoring for at-hit "
            "rebates, or the continuous PDE engine (which prices the at-hit rebate exactly)."
        )

    half = (num_paths + 1) // 2
    n_eff = 2 * half if use_antithetic else num_paths
    rng = np.random.default_rng(seed)

    nodes = np.empty((n_eff, n + 1), dtype=float)
    vols = np.empty((n_eff, n), dtype=float)
    s = np.full(n_eff, float(s0), dtype=float)
    nodes[:, 0] = s
    drift = rf - cf
    sqrt_dt = np.sqrt(dt)
    t = 0.0
    for i in range(n):
        sigma = np.asarray(lv_surface.local_vol(s, t), dtype=float)
        vols[:, i] = sigma
        if use_antithetic:
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
        else:
            z = rng.standard_normal(n_eff)
        s = s * np.exp((drift[i] - 0.5 * sigma * sigma) * dt[i] + sigma * sqrt_dt[i] * z)
        nodes[:, i + 1] = s
        t += dt[i]

    disc, node_times = _disc_closure(dt, rf)
    T = float(node_times[-1])
    if continuous:
        w, first = bridge_survival(nodes, vols, dt, spec)
        hit_cumT = node_times[np.minimum(first, n)]
    else:
        idx = np.asarray(observe_idx, dtype=int)
        w, first = discrete_survival(nodes[:, idx], spec)
        hit_cumT = node_times[idx[np.minimum(first, idx.size - 1)]]

    pv = mc_barrier_cashflows(nodes[:, -1], w, hit_cumT, spec, disc, T, participation=participation)

    if use_antithetic:
        pair = 0.5 * (pv[:half] + pv[half:2 * half])
        price = float(np.mean(pair))
        if return_stderr:
            return price, (float(np.std(pair, ddof=1) / np.sqrt(half)) if half > 1 else 0.0)
        return price
    price = float(np.mean(pv))
    if return_stderr:
        return price, (float(np.std(pv, ddof=1) / np.sqrt(num_paths)) if num_paths > 1 else 0.0)
    return price
