"""Multi-leg futures hedge objectives over already weighted book Greeks.

Three policies (revised design section 4), all sized from the exact bucket
positions ``h_i^nodes = -B_i / m_i``:

``nodes``
    every modelled nodal rhoq is neutral; the residual is the pinned-futures
    spot delta ``D_F``, which is NOT zero in general and is not a tail-only
    quantity;
``spot_far``
    spot delta is neutral; the far node keeps ``D_F S T_n``, of either sign;
``spot_parallel``
    the primary policy: spot delta AND parallel rhoq are neutral, with the
    remaining shape risk concentrated as ``(+K, -K)`` on a chosen pair,
    ``K = D_F S T_a T_b / (T_b - T_a)``.

``spot_parallel`` needs two distinct eligible tenors even on a date whose
``D_F`` happens to be negligible.  Fewer nodes, or an explicit pair that is
not available, is a recorded infeasibility -- never a silent fall back to a
different objective.

This strategy is natively multi-leg.  It returns a
:class:`FuturesHedgeTargets` from :meth:`target_legs`; it does not pretend to
answer ``BaseStrategy.calculate_hedge_size`` with one scalar, because a
"number of contracts" that ignored which contract would be meaningless here.
The replay and simulation validators recognise the type explicitly.

The strategy operates only on already weighted numbers.  It has no
``product_quantity`` argument, prices nothing and applies no second quantity
factor: aggregation happened once, in the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from quantark.backtest.futures_risk import FuturesBookRisk, held_book_risk
from quantark.util.exceptions import ValidationError

from .base_strategy import AssetClass, BaseStrategy, HedgingTarget

#: The objectives of design section 4.  There is no fourth, and no implicit
#: substitution between them.
HEDGE_OBJECTIVES: Tuple[str, ...] = ("spot_parallel", "nodes", "spot_far")

__all__ = [
    "HEDGE_OBJECTIVES",
    "FuturesBucketHedgeStrategy",
    "FuturesHedgeTargets",
    "ideal_targets",
]


def ideal_targets(
    risk: FuturesBookRisk,
    objective: str,
    correction_pair: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Optional[Tuple[str, str]]]:
    """Exact fractional holdings for one objective, before ratio and rounding.

    Returns ``(targets, correction, chosen_pair)`` where ``correction`` is the
    part added on top of the exact buckets, so the engine can report the fold
    separately from the bucket legs.
    """
    buckets = risk.buckets
    if not buckets:
        raise ValidationError("at least one futures node is required")
    if objective not in HEDGE_OBJECTIVES:
        raise ValidationError(f"unknown hedge objective: {objective}")
    if correction_pair is not None and objective != "spot_parallel":
        raise ValidationError("correction_pair requires spot_parallel")
    targets = {b.contract: -b.bucket_currency / b.multiplier for b in buckets}
    correction = {b.contract: 0.0 for b in buckets}
    df = risk.delta_f_derived
    chosen: Optional[Tuple[str, str]] = None
    if objective == "spot_far":
        far = buckets[-1]
        correction[far.contract] = -df * risk.spot / (far.multiplier * far.price)
    elif objective == "spot_parallel":
        if len(buckets) < 2:
            raise ValidationError(
                "spot_parallel requires two distinct futures tenors"
            )
        by_contract = {b.contract: b for b in buckets}
        ids = tuple(correction_pair) if correction_pair else (
            buckets[0].contract,
            buckets[-1].contract,
        )
        if (
            len(ids) != 2
            or len(set(ids)) != 2
            or any(c not in by_contract for c in ids)
        ):
            raise ValidationError(
                f"correction pair must name two eligible contracts: {ids}"
            )
        a, b = (by_contract[c] for c in ids)
        gap = b.tenor_years - a.tenor_years
        if gap <= 0:
            raise ValidationError(
                "correction pair must be ordered by increasing tenor"
            )
        correction[a.contract] = (
            -df * b.tenor_years / gap * risk.spot / (a.multiplier * a.price)
        )
        correction[b.contract] = (
            df * a.tenor_years / gap * risk.spot / (b.multiplier * b.price)
        )
        chosen = (str(ids[0]), str(ids[1]))
    for contract in targets:
        targets[contract] += correction[contract]
    return targets, correction, chosen


@dataclass(frozen=True)
class FuturesHedgeTargets:
    """The plan for one date: ideal, scaled and rounded legs plus intent.

    ``ideal_net_delta`` and ``ideal_net_rhoq`` are the residuals the policy
    MEANT to leave, so the engine can tell an intended exposure apart from an
    execution error.  Every mapping is snapshotted, never aliased.
    """

    objective: str
    ideal: Dict[str, float]
    scaled: Dict[str, float]
    rounded: Dict[str, float]
    correction: Dict[str, float]
    correction_pair: Optional[Tuple[str, str]]
    retired_contracts: Tuple[str, ...]
    ideal_net_delta: float
    ideal_net_rhoq: Dict[str, float]
    hedge_ratio: float
    rounded_requested: bool

    def __post_init__(self) -> None:
        for name in ("ideal", "scaled", "rounded", "correction", "ideal_net_rhoq"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        object.__setattr__(
            self, "retired_contracts", tuple(self.retired_contracts)
        )

    @property
    def planned(self) -> Dict[str, float]:
        """The quantities the engine should try to hold before the band."""
        return dict(self.rounded)

    @property
    def ideal_net_parallel_rhoq(self) -> float:
        return sum(self.ideal_net_rhoq.values())

    @property
    def ideal_gross_nodal_rhoq(self) -> float:
        return sum(abs(v) for v in self.ideal_net_rhoq.values())

    def rounding_error(self) -> Dict[str, float]:
        """``rounded - scaled`` per leg."""
        return {c: self.rounded[c] - self.scaled[c] for c in self.rounded}


class FuturesBucketHedgeStrategy(BaseStrategy):
    """Joint spot-delta and carry hedging with several futures legs."""

    def __init__(
        self,
        objective: str = "spot_parallel",
        correction_pair: Optional[Sequence[str]] = None,
        delta_threshold: float = 0.0,
        round_contracts: bool = True,
        hedge_ratio: float = 1.0,
    ) -> None:
        super().__init__(
            name="FuturesBucketHedge",
            asset_class=AssetClass.EQUITY,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument="futures",
        )
        if objective not in HEDGE_OBJECTIVES:
            raise ValidationError(
                f"objective must be one of {HEDGE_OBJECTIVES}, got {objective!r}"
            )
        if correction_pair is not None:
            pair = tuple(str(c) for c in correction_pair)
            if objective != "spot_parallel":
                raise ValidationError("correction_pair requires spot_parallel")
            if len(pair) != 2 or len(set(pair)) != 2:
                raise ValidationError(
                    "correction_pair must name two distinct contracts"
                )
            correction_pair = pair
        if not math.isfinite(delta_threshold) or delta_threshold < 0:
            raise ValidationError("delta_threshold must be finite and non-negative")
        if not math.isfinite(hedge_ratio) or hedge_ratio < 0:
            raise ValidationError("hedge_ratio must be finite and non-negative")
        self.objective = objective
        self.correction_pair = correction_pair
        self.delta_threshold = float(delta_threshold)
        self.round_contracts = bool(round_contracts)
        self.hedge_ratio = float(hedge_ratio)

    # ------------------------------------------------------------------
    # Native multi-leg API
    # ------------------------------------------------------------------

    def target_legs(
        self,
        *,
        buckets,
        delta_q: float,
        spot: float,
        held_contracts: Mapping[str, float] | Sequence[str] = (),
    ) -> FuturesHedgeTargets:
        """The date's complete leg plan from already weighted book Greeks.

        ``buckets`` carries each eligible contract's tenor, price, multiplier
        and signed ``B_i``; ``delta_q`` is the signed ``D``.  Contracts that
        are held but no longer eligible get an explicit zero target, so a
        retired leg is closed rather than forgotten.
        """
        risk = (
            buckets
            if isinstance(buckets, FuturesBookRisk)
            else FuturesBookRisk(spot=spot, delta_q=delta_q, buckets=tuple(buckets))
        )
        if not math.isfinite(float(delta_q)):
            raise ValidationError("delta_q must be finite")
        ideal, correction, chosen = ideal_targets(
            risk, self.objective, self.correction_pair
        )
        ideal_delta, ideal_rho = held_book_risk(risk, ideal)
        scaled = {c: self.hedge_ratio * h for c, h in ideal.items()}
        rounded = {
            c: (float(round(h)) if self.round_contracts else float(h))
            for c, h in scaled.items()
        }
        eligible = set(risk.contracts)
        retired = tuple(
            sorted(c for c in held_contracts if c not in eligible)
        )
        for contract in retired:
            ideal.setdefault(contract, 0.0)
            scaled.setdefault(contract, 0.0)
            rounded.setdefault(contract, 0.0)
            correction.setdefault(contract, 0.0)
        return FuturesHedgeTargets(
            objective=self.objective,
            ideal=ideal,
            scaled=scaled,
            rounded=rounded,
            correction=correction,
            correction_pair=chosen,
            retired_contracts=retired,
            ideal_net_delta=ideal_delta,
            ideal_net_rhoq=ideal_rho,
            hedge_ratio=self.hedge_ratio,
            rounded_requested=self.round_contracts,
        )

    def should_rebalance(
        self, current_contracts: float, target_contracts: float
    ) -> bool:
        """The existing band semantics, applied per leg by the engine.

        The band is deliberately NOT applied inside the target kernel: the
        engine must retain the planned target even on a date it skips the
        trade, so the skipped-trade error stays measurable.
        """
        return (
            abs(float(target_contracts) - float(current_contracts))
            > self.delta_threshold
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "correction_pair": self.correction_pair,
            "delta_threshold": self.delta_threshold,
            "round_contracts": self.round_contracts,
            "hedge_ratio": self.hedge_ratio,
        }

    # ------------------------------------------------------------------
    # BaseStrategy protocol: explicitly unavailable
    # ------------------------------------------------------------------

    def calculate_hedge_size(
        self, current_time, portfolio_greeks, market_data, **kwargs
    ) -> float:
        raise ValidationError(
            "FuturesBucketHedgeStrategy is a multi-leg strategy: use "
            "target_legs(buckets=..., delta_q=..., spot=..., "
            "held_contracts=...). A single contract count cannot express a "
            "hedge spread across several tenors."
        )

    def should_hedge(
        self, current_time, portfolio_greeks, market_data, **kwargs
    ) -> bool:
        raise ValidationError(
            "FuturesBucketHedgeStrategy decides per leg: use "
            "should_rebalance(current, target) on each contract."
        )
