"""Pillar-bucket delta and rho_q on a :class:`~quantark.param.CarryCurve`.

The carry curve ``B(T) = log F(0, T)/S`` is pinned by one node per hedge
instrument (index futures, or a call/put pair's synthetic forward). This
module reads the product's exposure node by node:

- **pillar delta** ``∂PV/∂F_i`` at fixed spot — node ``i`` moved to
  ``B_i' = log(F_i'/S)``, every other node held;
- **spot delta** ``∂PV/∂S`` with every node ``B_i`` held (forwards move with
  spot, the usual delta of a carry-curve model);
- **pinned-forward delta** ``∂PV/∂S`` with every ``F_i`` held (nodes re-solved
  as ``B_i' = log(F_i/S')``): what remains after hedging each pillar with its
  own instrument. It is not zero in general — between ``0`` and ``T_1`` the
  curve runs through the anchor ``B(0) = 0``, and beyond ``T_n`` the long-end
  rule may carry ``S`` through (``FLAT_CARRY_YIELD``: ``B(t) = B_n·t/T_n``);
- **rho_q per pillar** — node ``i``'s yield moved, ``B_i' = B_i − ε·T_i``,
  reported per +1% of ``q``; **parallel rho_q** through the view's
  ``parallel_shifted``.

Two identities check the table. The chain rule
``spot_delta = pinned_forward_delta + Σ (F_i/S)·delta_i`` is exact for every
long-end rule; its finite-difference residual is clean only when the spot and
forward relative steps match (each central difference carries its own
``O(h²)`` error). ``parallel_rho_q = Σ rho_q_i`` holds whenever a node shift is
the exact parallel shift (``FLAT_FORWARD_CARRY`` and ``FLAT_CARRY_YIELD``);
under ``ZERO_FORWARD_CARRY`` the flat tail cannot carry ``q + s`` and the
residual is the tail's own exposure.

Spec: ``.ship/2026-11-05-snowball-2y-carry-curve/spec.md`` §Contracts C2.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.riskmeasures.greeks.bump_envs import resolve_bump_engine
from quantark.param.curve.carry_curve import CarryCurve
from quantark.param.curve.interpolation import LongEndRule, long_end_rule
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, safe_log, validate_positive

#: rho_q is reported per this change of q (+1%).
RHO_Q_REPORT_UNIT = 0.01


@dataclass(frozen=True)
class CarryPillarBucket:
    """One carry-curve node's risk.

    Attributes:
        label: Caller label (default ``T1``, ``T2``, …).
        tenor: Node time ``T_i`` (years).
        forward: ``F_i = S·exp(B_i)``.
        delta: ``∂PV/∂F_i`` at fixed spot (central difference).
        hedge_lots: ``−delta / lot_multipliers[i]``; ``None`` without multipliers.
        rho_q: PV change per +1% of ``q`` at node ``i`` (central node bump).
        extrapolated_tail: True when the product matures beyond ``T_n`` and the
            curve's long-end rule reads this node (nodes ``n−1`` and ``n`` under
            ``FLAT_FORWARD_CARRY``, node ``n`` otherwise).
    """

    label: str
    tenor: float
    forward: float
    delta: float
    hedge_lots: Optional[float]
    rho_q: float
    extrapolated_tail: bool


@dataclass(frozen=True)
class CarryCurveBucketReport:
    """Pillar risk of one product on one carry curve; see the module docstring."""

    base_pv: float
    spot: float
    spot_delta: float
    pinned_forward_delta: float
    buckets: Tuple[CarryPillarBucket, ...]
    parallel_rho_q: float
    delta_identity_residual: float
    rho_sum_residual: float


def calculate_carry_curve_buckets(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    carry_curve: CarryCurve,
    *,
    labels: Optional[Sequence[str]] = None,
    lot_multipliers: Optional[Sequence[float]] = None,
    spot_bump_rel: float = 0.01,
    forward_bump_rel: float = 1e-3,
    q_bump: float = 1e-3,
) -> CarryCurveBucketReport:
    """Pillar-bucket delta / rho_q of ``product`` priced on ``carry_curve``.

    Every price (base included) uses a deep copy of ``pricing_env`` whose
    ``div_yield`` is ``carry_curve.to_dividend_yield(pricing_env.rate_curve)``;
    ``pricing_env.div_yield`` is ignored and the caller's env is never mutated.
    All prices go through ``resolve_bump_engine`` so engines with a bump context
    (PDE) keep one frozen layout across the bumps.

    Args:
        product: The product.
        pricing_env: Spot, vol, rate and clock; its dividend slot is replaced.
        engine: Pricing engine.
        carry_curve: The carry curve whose nodes are bucketed.
        labels: One label per node (default ``T1..Tn``).
        lot_multipliers: CNY (currency) per point of ``F_i`` per lot, one per node;
            ``hedge_lots = −delta/multiplier``.
        spot_bump_rel: Relative central spot step for both spot deltas.
        forward_bump_rel: Relative central step on ``F_i`` for pillar deltas.
        q_bump: Central yield step ``ε`` for rho_q (reported per +1%).

    Raises:
        ValidationError: invalid bumps, labels or multipliers, a non-CarryCurve,
            or a curve anchored on another valuation date than the env.
    """
    if not isinstance(carry_curve, CarryCurve):
        raise ValidationError(f"carry_curve must be a CarryCurve, got {type(carry_curve).__name__}")
    validate_positive(spot_bump_rel, "spot_bump_rel")
    validate_positive(forward_bump_rel, "forward_bump_rel")
    validate_positive(q_bump, "q_bump")
    for name, rel in (("spot_bump_rel", spot_bump_rel), ("forward_bump_rel", forward_bump_rel)):
        if not rel < 1.0:  # the down leg S·(1−h) / F·(1−h) must stay positive
            raise ValidationError(f"{name} must be < 1, got {rel}")
    nodes = carry_curve.nodes
    n = len(nodes)
    if labels is None:
        labels = [f"T{i + 1}" for i in range(n)]
    elif len(labels) != n:
        raise ValidationError(f"labels has {len(labels)} entries for {n} nodes")
    if lot_multipliers is not None:
        if len(lot_multipliers) != n:
            raise ValidationError(f"lot_multipliers has {len(lot_multipliers)} entries for {n} nodes")
        for m in lot_multipliers:
            validate_positive(float(m), "lot_multipliers entry")
    curve_vd = carry_curve.valuation_date
    env_vd = getattr(pricing_env, "valuation_date", None)
    if curve_vd is not None and env_vd is not None and curve_vd != env_vd:
        raise ValidationError(
            f"carry curve anchored at {curve_vd:%Y-%m-%d} but the pricing env at {env_vd:%Y-%m-%d}"
        )

    if pricing_env.spot_quote is None:
        raise ValidationError("pricing_env.spot_quote is required")
    rate_curve = pricing_env.rate_curve
    spot = float(pricing_env.spot_quote.spot)

    def env_for(div_yield, spot_value: float = spot) -> PricingEnvironment:
        env = deepcopy(pricing_env)
        env.div_yield = div_yield
        env.spot_quote.spot = spot_value  # type: ignore[union-attr]  # checked above
        return env

    def curve_with(i: int, new_b: float) -> CarryCurve:
        moved = list(nodes)
        moved[i] = (moved[i][0], new_b)
        return carry_curve.with_nodes(moved)

    base_div = carry_curve.to_dividend_yield(rate_curve)
    base_env = env_for(base_div)
    bump_engine = resolve_bump_engine(product, base_env, engine)

    def pv(div_yield, spot_value: float = spot) -> float:
        return float(bump_engine.price(product, env_for(div_yield, spot_value)))

    base_pv = float(bump_engine.price(product, base_env))

    # spot delta, nodes B_i held
    s_up, s_dn = spot * (1.0 + spot_bump_rel), spot * (1.0 - spot_bump_rel)
    spot_delta = (pv(base_div, s_up) - pv(base_div, s_dn)) / (s_up - s_dn)

    # pinned-forward delta: every F_i held, B_i' = B_i − log(S'/S)
    def pinned(s_new: float):
        shift = float(safe_log(s_new / spot))
        return carry_curve.with_nodes([(t, b - shift) for t, b in nodes]).to_dividend_yield(rate_curve)

    pinned_forward_delta = (pv(pinned(s_up), s_up) - pv(pinned(s_dn), s_dn)) / (s_up - s_dn)

    tail_nodes = _tail_reading_nodes(carry_curve, product.get_maturity(pricing_env))
    up_log, dn_log = float(safe_log(1.0 + forward_bump_rel)), float(safe_log(1.0 - forward_bump_rel))
    buckets: List[CarryPillarBucket] = []
    for i, (t_i, b_i) in enumerate(nodes):
        f_i = spot * float(safe_exp(b_i))
        p_up = pv(curve_with(i, b_i + up_log).to_dividend_yield(rate_curve))
        p_dn = pv(curve_with(i, b_i + dn_log).to_dividend_yield(rate_curve))
        delta = (p_up - p_dn) / (2.0 * forward_bump_rel * f_i)
        q_up = pv(curve_with(i, b_i - q_bump * t_i).to_dividend_yield(rate_curve))
        q_dn = pv(curve_with(i, b_i + q_bump * t_i).to_dividend_yield(rate_curve))
        rho_q = (q_up - q_dn) / (2.0 * q_bump) * RHO_Q_REPORT_UNIT
        lots = None if lot_multipliers is None else -delta / float(lot_multipliers[i])
        buckets.append(
            CarryPillarBucket(
                label=str(labels[i]), tenor=float(t_i), forward=f_i, delta=delta,
                hedge_lots=lots, rho_q=rho_q, extrapolated_tail=i in tail_nodes,
            )
        )

    parallel_rho_q = (
        (pv(base_div.parallel_shifted(q_bump)) - pv(base_div.parallel_shifted(-q_bump)))
        / (2.0 * q_bump) * RHO_Q_REPORT_UNIT
    )
    delta_identity_residual = spot_delta - pinned_forward_delta - sum(
        (b.forward / spot) * b.delta for b in buckets
    )
    rho_sum_residual = parallel_rho_q - sum(b.rho_q for b in buckets)
    return CarryCurveBucketReport(
        base_pv=base_pv,
        spot=spot,
        spot_delta=spot_delta,
        pinned_forward_delta=pinned_forward_delta,
        buckets=tuple(buckets),
        parallel_rho_q=parallel_rho_q,
        delta_identity_residual=delta_identity_residual,
        rho_sum_residual=rho_sum_residual,
    )


def _tail_reading_nodes(carry_curve: CarryCurve, maturity: float) -> frozenset:
    """Indices of the nodes the long-end rule reads, when ``maturity`` is past ``T_n``."""
    n = len(carry_curve.tenors)
    if not maturity > carry_curve.tenors[-1]:
        return frozenset()
    if long_end_rule(carry_curve.extrapolation) is LongEndRule.HOLD_FORWARD and n >= 2:
        return frozenset({n - 2, n - 1})
    return frozenset({n - 1})
