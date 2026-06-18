"""Phoenix backward-grid -> per-(t, KI, memory) value surface (#1b, stateful).

Phoenix autocallables pay coupons WHILE ALIVE (conditional on a coupon barrier, with
optional memory of missed coupons). The QUAD engine carries a STACK of value surfaces
v_*_list[k] indexed by accumulated-missed-coupon count k. The exposure value at an
observation must be EX-coupon (the coupon paid there is received, not forward exposure),
so the engine records the surfaces at the TOP of each step (before that observation's
coupon/KO jump), indexed by the POST-resolution memory a survivor carries into the next
period. This builder reads those stacks into a per-(t, KI, memory) GridValueSurface; the
``PhoenixStateMachine`` then resolves each path's KI / KO / memory history.

v1 scope (raise, never approximate, on anything else):
- product is a PhoenixOption priced by a plain PhoenixQuadEngine;
- INSTANT coupon pay (EXPIRY accumulate-to-maturity needs a coupon-receivable layer);
- immediate KO settlement (delayed needs a memory-dependent KO receivable);
- continuous KI or no KI (discrete KI is deferred);
- a single constant coupon barrier and constant KO barrier.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.sacva.exposure.value_surface import GridValueSurface
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

_TOL = 1e-9


class PhoenixTerminatedAtValuation(Exception):
    """Zero maturity / no forward surface — zero future exposure (see snowball twin)."""


@dataclass
class PhoenixExposureSurface:
    times: np.ndarray
    surface: GridValueSurface             # {t: {("alive"|"knocked_in", k): (grid, vals)}}
    coupon_barrier: float
    ko_barrier: float
    direction: str                        # "up" (standard) / "down" (reverse)
    obs_idx: List[int]                    # coupon/KO observation grid indices
    num_obs: int
    ki_barrier: Optional[float]
    ki_direction: str
    ki_monitoring_idx: List[int]
    ki_continuous: bool
    vol: float
    use_memory: bool
    initial_knocked_in: bool = False


def _grid_index(times: np.ndarray, t_obs: float) -> int:
    hits = np.where(np.abs(times - t_obs) <= _TOL)[0]
    if hits.size != 1:
        raise ValidationError(
            f"observation time {t_obs} is not a unique node of the exposure grid")
    return int(hits[0])


def build_phoenix_surface(trade) -> PhoenixExposureSurface:
    product, engine, env = trade.product, trade.engine, trade.env
    if not isinstance(product, PhoenixOption):
        raise ValidationError(
            f"Phoenix exposure supports only PhoenixOption (got {type(product).__name__})")
    if type(engine) is not PhoenixQuadEngine:
        raise ValidationError(
            "Phoenix exposure supports only the plain PhoenixQuadEngine "
            f"(got {type(engine).__name__})")

    # ---- v1 fences, validated BEFORE the expensive recording price call ----------
    if product.coupon_config.coupon_pay_type is not CouponPayType.INSTANT:
        raise ValidationError(
            f"{trade.trade_id}: Phoenix exposure v1 supports INSTANT coupon pay only; "
            "EXPIRY (accumulate-to-maturity) is deferred (needs a coupon receivable)")
    if product.barrier_config.disable_ko_after_ki:
        raise ValidationError(f"{trade.trade_id}: disable_ko_after_ki is deferred")

    maturity = float(product.get_maturity(env))
    ko_records = product.resolve_ko_observations(env)
    if not ko_records:
        raise ValidationError(f"{trade.trade_id}: empty KO observation schedule")
    for rec in ko_records:
        obs_t = float(rec.observation_time)
        settle = obs_t if rec.settlement_time is None else float(rec.settlement_time)
        if abs(settle - obs_t) > Tolerance.PRECISION:
            raise ValidationError(
                f"{trade.trade_id}: Phoenix exposure v1 requires immediate KO settlement "
                f"(obs={obs_t}, settle={settle}); delayed settlement is deferred "
                "(needs a memory-dependent KO receivable)")
    # Exact constancy (no silent rounding collapse): compare the RAW configured levels
    # BEFORE any float() coercion (which is not injective), and raise on any difference.
    ko_raw = [r.barrier for r in ko_records]
    if any(b != ko_raw[0] for b in ko_raw):
        raise ValidationError(f"{trade.trade_id}: v1 requires a constant KO barrier")
    ko_barrier = float(ko_raw[0])

    # Coupons are evaluated at the KO observation dates (the PhoenixOption has no separate
    # coupon schedule; the engine indexes coupon_barriers/amounts by KO observation), so a
    # single obs_idx drives both KO and coupon/memory resolution. A coupon-barrier LIST
    # must therefore align 1:1 with the KO observations, else the schedule is ambiguous.
    coupon_barrier_cfg = product.coupon_config.coupon_barrier
    if isinstance(coupon_barrier_cfg, list):
        if len(coupon_barrier_cfg) != len(ko_records):
            raise ValidationError(
                f"{trade.trade_id}: coupon-barrier schedule length {len(coupon_barrier_cfg)}"
                f" != KO observations {len(ko_records)} (coupons are evaluated at the KO "
                "observation dates)")
        if any(b != coupon_barrier_cfg[0] for b in coupon_barrier_cfg):
            raise ValidationError(f"{trade.trade_id}: v1 requires a constant coupon barrier")
        coupon_barrier = float(coupon_barrier_cfg[0])
    else:
        coupon_barrier = float(coupon_barrier_cfg)
    if not (np.isfinite(coupon_barrier) and coupon_barrier > 0):
        raise ValidationError(f"{trade.trade_id}: coupon barrier must be finite positive")
    direction = "down" if product.is_reverse else "up"

    ki_continuous = bool(product.has_ki_barrier) and (
        product.barrier_config.ki_continuous
        or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS)
    ki_barrier: Optional[float] = None
    if product.has_ki_barrier:
        if not ki_continuous:
            raise ValidationError(
                f"{trade.trade_id}: Phoenix exposure v1 supports continuous KI or no KI; "
                "discrete KI is deferred")
        kib = product.barrier_config.ki_barrier
        if isinstance(kib, list):
            raise ValidationError("continuous KI must have a scalar barrier")
        ki_barrier = float(kib)
    ki_direction = "up" if product.is_reverse else "down"
    use_memory = bool(product.has_memory_coupon)
    initial_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))

    # ---- one recorded price call -> per-observation per-memory surface stacks -----
    prev = getattr(engine, "record_backward_grids", False)
    engine.record_backward_grids = True
    try:
        engine.price(product, env)
    finally:
        engine.record_backward_grids = prev
    grids = dict(engine._backward_grids)
    if not grids:
        raise PhoenixTerminatedAtValuation(trade.trade_id)

    times = np.array(sorted(grids), dtype=float)
    surface_grids = {}
    for t, (spot_grid, v_in_list, v_out_list) in grids.items():
        if len(v_in_list) != len(v_out_list):
            raise ValidationError("recorded v_in/v_out memory stacks must align")
        slot = {}
        for m in range(len(v_out_list)):
            slot[("alive", m)] = (spot_grid, v_out_list[m])      # not yet knocked in
            slot[("knocked_in", m)] = (spot_grid, v_in_list[m])  # knock-in occurred
        surface_grids[float(t)] = slot
    surface = GridValueSurface(times, surface_grids, currency=trade.trade_currency)

    obs_idx = [_grid_index(times, float(r.observation_time)) for r in ko_records]
    # coupon and KO share these observation nodes; the indices must be unique and
    # strictly increasing (one resolution per node, in time order).
    if obs_idx != sorted(set(obs_idx)):
        raise ValidationError(
            f"{trade.trade_id}: KO/coupon observation indices must be unique and ordered")
    ki_monitoring_idx = list(range(len(times))) if ki_continuous else []
    vol = float(env.get_vol(product.strike, maturity))

    return PhoenixExposureSurface(
        times=times, surface=surface, coupon_barrier=coupon_barrier,
        ko_barrier=ko_barrier, direction=direction, obs_idx=obs_idx,
        num_obs=len(ko_records), ki_barrier=ki_barrier, ki_direction=ki_direction,
        ki_monitoring_idx=ki_monitoring_idx, ki_continuous=ki_continuous, vol=vol,
        use_memory=use_memory, initial_knocked_in=initial_knocked_in)
