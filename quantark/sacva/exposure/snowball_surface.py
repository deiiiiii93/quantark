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

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.sacva.exposure.value_surface import GridValueSurface
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import KnockInMonitoringMode
from quantark.util.exceptions import ValidationError

_TOL = 1e-9


@dataclass
class SnowballExposureSurface:
    """Everything the MC engine needs to expose a stateful snowball trade."""

    times: np.ndarray                 # exposure grid (sorted backward-grid keys, t0=0)
    surface: GridValueSurface         # {t: {"alive": (g, v_out), "knocked_in": (g, v_in)}}
    ko_barrier: float
    ko_direction: str                 # "up" (standard) / "down" (reverse)
    ko_monitoring_idx: List[int]
    ki_barrier: Optional[float]
    ki_direction: str                 # "down" (standard) / "up" (reverse)
    ki_monitoring_idx: List[int]
    ki_continuous: bool
    vol: float                        # bridge variance for continuous KI
    initial_knocked_in: bool = False  # lifecycle: knocked in before valuation


def _grid_index(times: np.ndarray, t_obs: float) -> int:
    hits = np.where(np.abs(times - t_obs) <= _TOL)[0]
    if hits.size != 1:
        raise ValidationError(
            f"observation time {t_obs} is not a unique node of the exposure grid")
    return int(hits[0])


def build_snowball_surface(trade) -> SnowballExposureSurface:
    product, engine, env = trade.product, trade.engine, trade.env
    if not isinstance(product, SnowballOption):
        raise ValidationError(
            "stateful exposure v1 supports only SnowballOption "
            f"(got {type(product).__name__})")
    # Phoenix / KO-reset SUBCLASS SnowballQuadEngine but carry coupon-memory / reset
    # state the two-regime surface cannot represent -> require the exact base engine.
    if type(engine) is not SnowballQuadEngine:
        raise ValidationError(
            "stateful exposure v1 supports only the plain SnowballQuadEngine "
            f"(got {type(engine).__name__}); Phoenix/KO-reset are deferred")

    # ---- v1 fences, validated BEFORE the expensive priced recording ----------
    # KO disabled after KI: the state machine cannot suppress KO for knocked-in
    # paths yet, while the recorded v_in surface assumes it can — would mis-zero.
    if product.barrier_config.disable_ko_after_ki:
        raise ValidationError(
            f"{trade.trade_id}: disable_ko_after_ki exposure is deferred in v1 (the "
            "state machine would wrongly knock out an already-knocked-in path)")

    # KO schedule: immediate settlement + constant barrier (v1). settlement_time is
    # None for immediate settlement in the engine's own _ko_discount convention.
    ko_records = product.resolve_ko_observations(env)
    if not ko_records:
        raise ValidationError(f"{trade.trade_id}: empty KO observation schedule")
    for rec in ko_records:
        settle = rec.observation_time if rec.settlement_time is None else rec.settlement_time
        if abs(float(settle) - float(rec.observation_time)) > _TOL:
            raise ValidationError(
                f"{trade.trade_id}: delayed KO settlement (obs={rec.observation_time}, "
                f"settle={settle}) is deferred in v1 — needs the pending-receivable "
                "machinery")
    ko_levels = {round(float(rec.barrier), 12) for rec in ko_records}
    if len(ko_levels) != 1:
        raise ValidationError(
            f"{trade.trade_id}: v1 requires a constant KO barrier across observations")
    ko_barrier = float(ko_records[0].barrier)
    ko_direction = "down" if product.is_reverse else "up"

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
        raise ValidationError(
            f"{trade.trade_id}: no backward grids recorded — the snowball is already "
            "terminated (immediate KO or zero maturity) at valuation")

    # exposure grid = sorted recorded times; per-(t, state) surface
    times = np.array(sorted(grids), dtype=float)
    surface_grids = {}
    for t, (spot_grid, v_in, v_out) in grids.items():
        surface_grids[float(t)] = {
            "alive": (spot_grid, v_out),         # not yet knocked in
            "knocked_in": (spot_grid, v_in),     # knock-in has occurred
        }
    surface = GridValueSurface(times, surface_grids, currency=trade.trade_currency)

    ko_monitoring_idx = [_grid_index(times, float(rec.observation_time))
                         for rec in ko_records]
    if ki_continuous:
        # continuous monitoring over the whole life: every interval is sampled
        # (the state machine bridges between nodes); schedule must be contiguous.
        ki_monitoring_idx = list(range(len(times)))
    elif ki_records:
        ki_monitoring_idx = [_grid_index(times, float(r.observation_time))
                             for r in ki_records]
    else:
        ki_monitoring_idx = []

    maturity = float(product.get_maturity(env))
    vol = float(env.get_vol(product.strike, maturity))

    return SnowballExposureSurface(
        times=times, surface=surface, ko_barrier=ko_barrier,
        ko_direction=ko_direction, ko_monitoring_idx=ko_monitoring_idx,
        ki_barrier=ki_barrier, ki_direction=ki_direction,
        ki_monitoring_idx=ki_monitoring_idx, ki_continuous=ki_continuous, vol=vol,
        initial_knocked_in=initial_knocked_in)
