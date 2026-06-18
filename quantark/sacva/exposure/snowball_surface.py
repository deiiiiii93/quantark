"""Snowball backward-grid -> per-(t, state) value surface (spec §3.2, stateful).

Builds the CVA exposure inputs for a SnowballOption by running the QUAD engine
ONCE with ``record_backward_grids`` on and reading the two-regime continuation
surfaces (v_out = not-yet-knocked-in, v_in = knocked-in) it computes at every
observation time. No re-pricing, no env rolling: the backward sweep already
produces V(spot, t, state) on a single inception-anchored grid that covers the
simulated spot cloud (num_std_devs wide). The per-path KI history and KO
termination are then resolved by ``BarrierStateMachine``.

v1 scope (raise, never approximate, on anything else):
- product is a plain ``SnowballOption`` priced by a plain ``SnowballQuadEngine``
  (Phoenix/KO-reset extend it with richer state — deferred);
- KO redemption settles on its observation date (delayed settlement needs the
  ``pending_receivable_exposure`` machinery — deferred);
- a single, constant KO barrier and (if present) a single constant KI barrier.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.sacva.exposure.value_surface import GridValueSurface
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import KnockInMonitoringMode
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

_TOL = 1e-9
# Settlement classification (immediate vs delayed vs post-maturity) MUST use the same
# tolerance the QUAD engine uses when it decides whether to insert a settlement
# recording node (SnowballQuadEngine._insert_settlement_times), otherwise a sub-
# tolerance delay could be a node on one side and not the other.
_SETTLE_TOL = Tolerance.PRECISION


class SnowballTerminatedAtValuation(Exception):
    """The snowball is already terminated at valuation (immediate KO triggered by the
    valuation spot, or zero maturity), so it has no forward value surface. The exposure
    engine treats it as zero future exposure rather than a hard error — a +1% market
    bump can validly tip a near-barrier snowball over a valuation-date KO."""


@dataclass
class SnowballExposureSurface:
    """Everything the MC engine needs to expose a stateful snowball trade."""

    times: np.ndarray                 # exposure grid (sorted backward-grid keys, t0=0)
    surface: GridValueSurface         # {t: {"alive": (g, v_out), "knocked_in": (g, v_in)}}
    ko_barrier: float
    ko_direction: str                 # "up" (standard) / "down" (reverse)
    ko_monitoring_idx: List[int]
    ko_payoffs: List[float]           # KO redemption per observation (aligned to ko_monitoring_idx)
    ko_settle_idx: List[int]          # settlement grid index per obs (== obs idx if immediate)
    ki_barrier: Optional[float]
    ki_direction: str                 # "down" (standard) / "up" (reverse)
    ki_monitoring_idx: List[int]
    ki_continuous: bool
    vol: float                        # bridge variance for continuous KI
    initial_knocked_in: bool = False  # lifecycle: knocked in before valuation
    # KO-reset: post-KI KO leg (the barrier resets on knock-in). None/empty for a plain
    # snowball, where ko_barrier applies to every alive path regardless of KI state.
    post_ko_barrier: Optional[float] = None
    post_ko_monitoring_idx: List[int] = field(default_factory=list)
    post_ko_payoffs: List[float] = field(default_factory=list)
    post_ko_settle_idx: List[int] = field(default_factory=list)


def _grid_index(times: np.ndarray, t_obs: float) -> int:
    hits = np.where(np.abs(times - t_obs) <= _TOL)[0]
    if hits.size != 1:
        raise ValidationError(
            f"observation time {t_obs} is not a unique node of the exposure grid")
    return int(hits[0])


def _validate_ko_leg(records, trade_id, maturity, leg="") -> float:
    """Validate one KO observation leg and return its (constant) barrier level.

    Checks settlement bounds (settlement in [obs, maturity]; post-maturity is deferred)
    and a single constant barrier across the leg. ``leg`` labels the schedule (""/"post-")
    in error messages. Validated BEFORE the expensive recording price call.
    """
    if not records:
        raise ValidationError(f"{trade_id}: empty {leg}KO observation schedule")
    for rec in records:
        obs_t = float(rec.observation_time)
        if not np.isfinite(obs_t):
            raise ValidationError(f"{trade_id}: non-finite {leg}KO observation time")
        if rec.barrier is None or not (np.isfinite(float(rec.barrier))
                                       and float(rec.barrier) > 0):
            raise ValidationError(
                f"{trade_id}: {leg}KO barrier must be finite and positive")
        settle = obs_t if rec.settlement_time is None else float(rec.settlement_time)
        if not np.isfinite(settle):
            raise ValidationError(f"{trade_id}: non-finite {leg}KO settlement time")
        if settle < obs_t - _SETTLE_TOL:
            raise ValidationError(
                f"{trade_id}: {leg}KO settlement precedes observation "
                f"(obs={obs_t}, settle={settle})")
        if settle > maturity + _SETTLE_TOL:
            raise ValidationError(
                f"{trade_id}: post-maturity {leg}KO settlement (obs={obs_t}, "
                f"settle={settle}, maturity={maturity}) is deferred — it needs the "
                "terminal-payoff settlement tail (no continuation surface past maturity)")
        if rec.payoff is None or not np.isfinite(float(rec.payoff)):
            raise ValidationError(f"{trade_id}: non-finite {leg}KO payoff")
    levels = {round(float(rec.barrier), 12) for rec in records}
    if len(levels) != 1:
        raise ValidationError(
            f"{trade_id}: v1 requires a constant {leg}KO barrier across observations")
    return float(records[0].barrier)


def _index_ko_leg(records, times: np.ndarray, maturity: float):
    """Map a validated KO leg onto the recorded exposure grid.

    Returns (monitoring_idx, payoffs, settle_idx) aligned to ``records``; settle_idx
    equals the observation index for immediate (or sub-tolerance) settlement.
    """
    monitoring_idx = [_grid_index(times, float(rec.observation_time)) for rec in records]
    payoffs = [float(rec.payoff) for rec in records]
    settle_idx = []
    for obs_i, rec in zip(monitoring_idx, records):
        if rec.settlement_time is None:
            settle_idx.append(obs_i)
            continue
        settle = min(float(rec.settlement_time), maturity)
        if settle <= float(rec.observation_time) + _SETTLE_TOL:
            settle_idx.append(obs_i)
        else:
            settle_idx.append(_grid_index(times, settle))
    return monitoring_idx, payoffs, settle_idx


def build_snowball_surface(trade) -> SnowballExposureSurface:
    product, engine, env = trade.product, trade.engine, trade.env
    if not isinstance(product, SnowballOption):
        raise ValidationError(
            "stateful exposure v1 supports only SnowballOption "
            f"(got {type(product).__name__})")
    # Two-regime (alive / knocked-in) exposure supports the plain SnowballQuadEngine and
    # the KO-reset engine (its KO barrier merely resets on KI — still two regimes). Phoenix
    # carries coupon-memory state the two-regime surface cannot represent -> deferred.
    ko_reset = type(engine) is KOResetSnowballQuadEngine
    if not (type(engine) is SnowballQuadEngine or ko_reset):
        raise ValidationError(
            "stateful exposure supports SnowballQuadEngine and KOResetSnowballQuadEngine "
            f"(got {type(engine).__name__}); Phoenix is deferred")

    # ---- v1 fences, validated BEFORE the expensive priced recording ----------
    # KO disabled after KI: the state machine cannot suppress KO for knocked-in
    # paths yet, while the recorded v_in surface assumes it can — would mis-zero.
    if product.barrier_config.disable_ko_after_ki:
        raise ValidationError(
            f"{trade.trade_id}: disable_ko_after_ki exposure is deferred in v1 (the "
            "state machine would wrongly knock out an already-knocked-in path)")

    # KO schedule(s): a single constant barrier per leg (v1). Settlement may be delayed:
    # each redemption is carried as a pending receivable over [obs, settle) by the
    # exposure engine, valued off the engine's settlement-node surfaces. Only settlement
    # BEYOND maturity is deferred (no diffusion past the terminal node).
    maturity = float(product.get_maturity(env))
    ko_records = product.resolve_ko_observations(env)          # pre-KI (v_out) leg
    ko_barrier = _validate_ko_leg(ko_records, trade.trade_id, maturity)
    ko_direction = "down" if product.is_reverse else "up"

    # KO-reset: the post-KI (v_in) leg switches to a different KO barrier/schedule.
    post_ko_records = []
    post_ko_barrier: Optional[float] = None
    if ko_reset:
        if product.post_barrier_config.disable_ko_after_ki:
            raise ValidationError(
                f"{trade.trade_id}: post-KI disable_ko_after_ki is deferred")
        # Mirror the engine's own post-leg maturity filter (it drops post observations
        # past maturity from the priced recursion), so the surface's observation set
        # matches the recorded grid nodes exactly — not a silent fallback.
        post_ko_records = [
            r for r in engine._resolve_ko_records(product, env, product.post_barrier_config)
            if float(r.observation_time) <= maturity + _SETTLE_TOL]
        post_ko_barrier = _validate_ko_leg(
            post_ko_records, trade.trade_id, maturity, leg="post-")

    # KI schedule (optional)
    ki_continuous = bool(product.has_ki_barrier) and (
        product.barrier_config.ki_continuous
        or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS)
    ki_barrier: Optional[float] = None
    ki_records = []
    if product.has_ki_barrier:
        if ki_continuous:
            kib = product.barrier_config.ki_barrier
            if isinstance(kib, list):
                raise ValidationError("continuous KI must have a scalar barrier")
            ki_barrier = float(kib)
        else:
            # BGK mode rewrites discrete KI to a shifted CONTINUOUS barrier inside the
            # engine, so the recorded surface would no longer match a discrete state
            # machine (and the dense KI dates are not recorded grid nodes). Defer it.
            if engine._ki_monitoring_mode() is KnockInMonitoringMode.BGK_APPROXIMATION:
                raise ValidationError(
                    f"{trade.trade_id}: BGK_APPROXIMATION KI monitoring is deferred in v1 "
                    "(surface is continuous-shifted but the state machine is discrete); "
                    "use KnockInMonitoringMode.EXACT_DISCRETE")
            ki_records = product.resolve_ki_observations(env)
            if not ki_records:
                raise ValidationError(
                    f"{trade.trade_id}: KI barrier set but empty KI schedule")
            ki_levels = {round(float(r.barrier), 12) for r in ki_records}
            if len(ki_levels) != 1:
                raise ValidationError(
                    f"{trade.trade_id}: v1 requires a constant KI barrier")
            ki_barrier = float(ki_records[0].barrier)
    ki_direction = "up" if product.is_reverse else "down"
    # seasoned lifecycle: knocked in before valuation (engine prices from v_in)
    initial_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))

    # ---- one recorded price call -> per-observation (spot_grid, v_in, v_out) --
    prev = getattr(engine, "record_backward_grids", False)
    engine.record_backward_grids = True
    try:
        engine.price(product, env)
    finally:
        engine.record_backward_grids = prev
    grids = dict(engine._backward_grids)
    if not grids:
        # immediate KO (valuation spot breaches a t0 KO barrier) or zero maturity:
        # no forward surface exists. Signal termination; the exposure engine maps it
        # to zero future exposure (and still prices any co-netted vanillas).
        raise SnowballTerminatedAtValuation(trade.trade_id)

    # exposure grid = sorted recorded times; per-(t, state) surface
    times = np.array(sorted(grids), dtype=float)
    surface_grids = {}
    for t, (spot_grid, v_in, v_out) in grids.items():
        surface_grids[float(t)] = {
            "alive": (spot_grid, v_out),         # not yet knocked in
            "knocked_in": (spot_grid, v_in),     # knock-in has occurred
        }
    surface = GridValueSurface(times, surface_grids, currency=trade.trade_currency)

    # per-observation KO redemption + settlement node (for the pending receivable),
    # mapped onto the recorded grid. settlement node is present for every delayed
    # settlement <= maturity (the engine recorded a surface there).
    ko_monitoring_idx, ko_payoffs, ko_settle_idx = _index_ko_leg(
        ko_records, times, maturity)
    post_ko_monitoring_idx: List[int] = []
    post_ko_payoffs: List[float] = []
    post_ko_settle_idx: List[int] = []
    if ko_reset:
        post_ko_monitoring_idx, post_ko_payoffs, post_ko_settle_idx = _index_ko_leg(
            post_ko_records, times, maturity)
    if ki_continuous:
        # continuous monitoring over the whole life: every interval is sampled
        # (the state machine bridges between nodes); schedule must be contiguous.
        ki_monitoring_idx = list(range(len(times)))
    elif ki_records:
        ki_monitoring_idx = [_grid_index(times, float(r.observation_time))
                             for r in ki_records]
    else:
        ki_monitoring_idx = []

    vol = float(env.get_vol(product.strike, maturity))

    return SnowballExposureSurface(
        times=times, surface=surface, ko_barrier=ko_barrier,
        ko_direction=ko_direction, ko_monitoring_idx=ko_monitoring_idx,
        ko_payoffs=ko_payoffs, ko_settle_idx=ko_settle_idx,
        ki_barrier=ki_barrier, ki_direction=ki_direction,
        ki_monitoring_idx=ki_monitoring_idx, ki_continuous=ki_continuous, vol=vol,
        initial_knocked_in=initial_knocked_in,
        post_ko_barrier=post_ko_barrier,
        post_ko_monitoring_idx=post_ko_monitoring_idx,
        post_ko_payoffs=post_ko_payoffs, post_ko_settle_idx=post_ko_settle_idx)
