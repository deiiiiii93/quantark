"""Futures bucket sampling and signed book aggregation.

The whole module is written against one callback::

    price_at(spot, dividend) -> float

which prices ONE product, in ONE fixed state, with ONE already resolved
engine.  Nothing here builds an engine, calibrates a model or advances a
lifecycle: a bump that re-derived any of those would be measuring the
recalibration, not the risk.  The replay adapter creates the callback with
``_env_with``, so every scenario shares the day's vol surface, rate curve,
basis yield, valuation date, barriers and lifecycle snapshot by object
identity, and only spot and the dividend curve differ.

Sampling is a central price difference through
:meth:`CarryCurveContext.bump_future`, so both supported conventions are
rebuilt by their own builder.  A failed or non-finite price is an error --
never a zero sensitivity, which would silently read as "this node is already
hedged".

Aggregation happens exactly once, in the engine:

``D = sum_p Q_p D_p`` and ``B_i = sum_p Q_p B_p,i``

so the per-product samples stay unit-position sensitivities and no quantity
is applied twice.  Two samples may only be added when their contexts carry
the same coordinates -- contract order, prices, multipliers, tenors, spot,
valuation date and convention.  A matching node COUNT is not evidence.

The second half of the module is the INDEPENDENT AUDIT.  Everything in
:mod:`quantark.backtest.futures_risk` is algebra over sampled Greeks:
``delta_f_derived`` cancels against the identity that produced it, and
``held_book_risk`` predicts a residual from the same buckets it was handed.
Neither can detect a wrong bucket.  ``audit_held_book`` measures the same
quantities by pricing the book again in each scenario, so a perturbed bucket
makes the comparison fail while the derived identity still returns zero.

Two rules make the audit an audit (design section 6.3):

1. futures quantities are FIXED through every scenario and their
   deterministic scenario P&L is added to the repriced product,
   ``W(s, f) = V(s, f) + sum_i h_i m_i (f_i - F_i_base)``;
2. the holdings are the ACTUAL post-trade ones.  Substituting the ideal
   target into the sizing equations, or subtracting the derived identity,
   reproduces the prediction by construction and audits nothing.

Numerical validity and objective neutrality are separate results.  A correct
``nodes`` audit passes while reporting a large net spot delta, because that
residual is the policy's intent; an audit that was never taken reports
``not_measured``, never a zero.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from quantark.backtest.futures_risk import (
    CarryRiskSettings,
    FuturesBookRisk,
    FuturesBucket,
    held_book_risk,
    rhoq_bp_per_1pct,
    spot_delta_hands,
)
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.util.exceptions import ValidationError

PriceAt = Callable[[float, Any], float]

#: Audit outcomes.  ``inconclusive`` means the measurement's own uncertainty
#: is too wide to decide, which is NOT a pass even when it contains zero.
AUDIT_STATUSES = ("pass", "fail", "not_measured", "inconclusive")

__all__ = [
    "AUDIT_STATUSES",
    "BucketSample",
    "CarryAuditResult",
    "IdentitySample",
    "ProductCarryRisk",
    "aggregate_book_risk",
    "audit_held_book",
    "buckets_of",
    "direct_frozen_curve_book_delta",
    "direct_nodal_rhoq",
    "direct_parallel_rhoq",
    "direct_pinned_delta",
    "measure_product_carry_risk",
    "not_measured_audit",
    "refinement_ladder",
    "sample_buckets",
    "sample_chain_identity",
    "scenario_book_value",
    "stabilised",
    "status_from_interval",
]


def _checked_price(price_at: PriceAt, spot: float, dividend: Any, what: str) -> float:
    value = float(price_at(spot, dividend))
    if not math.isfinite(value):
        raise ValidationError(f"non-finite price for {what}: {value!r}")
    return value


@dataclass(frozen=True)
class BucketSample:
    """One node's central difference, with the prices that produced it."""

    contract: str
    points: float
    price_up: float
    price_down: float

    @property
    def bucket_currency(self) -> float:
        return (self.price_up - self.price_down) / (2.0 * self.points)


@dataclass(frozen=True)
class ProductCarryRisk:
    """One product's UNIT-position carry risk and how it was measured.

    ``delta_q`` is the day's frozen-carry-curve spot delta, taken from the
    same Greek call the state row already records rather than re-priced here.
    """

    context: CarryCurveContext
    delta_q: float
    buckets: Tuple[FuturesBucket, ...]
    samples: Tuple[BucketSample, ...]
    base_price: float
    effective_points: float
    price_calls: int
    elapsed_seconds: float

    @property
    def contracts(self) -> Tuple[str, ...]:
        return tuple(b.contract for b in self.buckets)

    def provenance(self) -> Dict[str, Any]:
        """Flat record of how each Greek was produced, for the artifacts."""
        return {
            "effective_points": self.effective_points,
            "base_price": self.base_price,
            "price_calls": self.price_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "extrapolation": self.context.extrapolation,
            "spot": self.context.spot,
            "samples": tuple(
                {
                    "contract": s.contract,
                    "points": s.points,
                    "price_up": s.price_up,
                    "price_down": s.price_down,
                }
                for s in self.samples
            ),
        }


def sample_buckets(
    price_at: PriceAt, context: CarryCurveContext, points: float
) -> Tuple[BucketSample, ...]:
    """Central ``dV/dF_i`` for every listed node at fixed spot.

    ``B_i = [V(F_i + points) - V(F_i - points)] / (2 * points)`` with spot and
    every other quote pinned.  Exactly ``2n`` price calls.
    """
    points = float(points)
    if not math.isfinite(points) or points <= 0.0:
        raise ValidationError(f"futures bump must be finite and positive: {points!r}")
    samples = []
    for quote in context.quotes:
        up = context.bump_future(quote.contract, points)
        down = context.bump_future(quote.contract, -points)
        pv_up = _checked_price(
            price_at, context.spot, up.dividend(), f"{quote.contract} +{points}"
        )
        pv_down = _checked_price(
            price_at, context.spot, down.dividend(), f"{quote.contract} -{points}"
        )
        samples.append(
            BucketSample(
                contract=quote.contract,
                points=points,
                price_up=pv_up,
                price_down=pv_down,
            )
        )
    return tuple(samples)


def buckets_of(
    context: CarryCurveContext, samples: Sequence[BucketSample]
) -> Tuple[FuturesBucket, ...]:
    """Attach each sampled difference to its coordinate."""
    if len(samples) != len(context.quotes):
        raise ValidationError("one sample per listed contract is required")
    buckets = []
    for quote, sample in zip(context.quotes, samples):
        if sample.contract != quote.contract:
            raise ValidationError(
                f"sample order must follow the curve: expected "
                f"{quote.contract!r}, got {sample.contract!r}"
            )
        buckets.append(
            FuturesBucket(
                contract=quote.contract,
                expiry_date=quote.expiry_date,
                tenor_years=float(quote.maturity),
                price=float(quote.price),
                multiplier=float(quote.multiplier),
                bucket_currency=sample.bucket_currency,
            )
        )
    return tuple(buckets)


def measure_product_carry_risk(
    price_at: PriceAt,
    context: CarryCurveContext,
    *,
    delta_q: float,
    points: float,
    base_price: float | None = None,
) -> ProductCarryRisk:
    """One product's unit-position buckets, with timing and prices recorded."""
    if not math.isfinite(float(delta_q)):
        raise ValidationError(f"delta_q must be finite, got {delta_q!r}")
    started = time.perf_counter()
    calls = 0
    if base_price is None:
        base_price = _checked_price(price_at, context.spot, context.dividend(), "base")
        calls += 1
    else:
        base_price = float(base_price)
        if not math.isfinite(base_price):
            raise ValidationError(f"base_price must be finite, got {base_price!r}")
    samples = sample_buckets(price_at, context, points)
    calls += 2 * len(samples)
    return ProductCarryRisk(
        context=context,
        delta_q=float(delta_q),
        buckets=buckets_of(context, samples),
        samples=samples,
        base_price=base_price,
        effective_points=float(points),
        price_calls=calls,
        elapsed_seconds=time.perf_counter() - started,
    )


def aggregate_book_risk(
    entries: Iterable[Tuple[float, ProductCarryRisk]]
) -> FuturesBookRisk:
    """``sum_p Q_p`` over unit-position product risks, in the given order.

    The quantity is applied here and only here.  The order of the input is
    preserved so a book's sum is reproducible.
    """
    entries = list(entries)
    if not entries:
        raise ValidationError("a book risk needs at least one product sample")
    reference = entries[0][1].context
    for _, risk in entries[1:]:
        if not reference.same_coordinates_as(risk.context):
            raise ValidationError(
                "carry risks on different coordinates cannot be summed: "
                f"{reference.coordinates()} vs {risk.context.coordinates()}"
            )
    delta = 0.0
    bucket_totals = [0.0] * len(reference.quotes)
    for quantity, risk in entries:
        quantity = float(quantity)
        if not math.isfinite(quantity):
            raise ValidationError(f"position quantity must be finite: {quantity!r}")
        delta += quantity * risk.delta_q
        for index, bucket in enumerate(risk.buckets):
            bucket_totals[index] += quantity * bucket.bucket_currency
    buckets = tuple(
        FuturesBucket(
            contract=quote.contract,
            expiry_date=quote.expiry_date,
            tenor_years=float(quote.maturity),
            price=float(quote.price),
            multiplier=float(quote.multiplier),
            bucket_currency=total,
        )
        for quote, total in zip(reference.quotes, bucket_totals)
    )
    return FuturesBookRisk(spot=reference.spot, delta_q=delta, buckets=buckets)


# ---------------------------------------------------------------------------
# Independent audit: directions measured by ACTUAL repricing
# ---------------------------------------------------------------------------


def _checked(value: float, what: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"non-finite price for {what}: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Directions measured by actual repricing
# ---------------------------------------------------------------------------


def scenario_book_value(
    price_at: PriceAt,
    context: CarryCurveContext,
    scenario: CarryCurveContext,
    holdings: Mapping[str, float],
) -> float:
    """``W(s, f)``: the repriced product plus the FIXED hedge's scenario P&L."""
    product = _checked(price_at(scenario.spot, scenario.dividend()), "scenario book")
    base = {q.contract: q for q in context.quotes}
    hedge = sum(
        holdings.get(q.contract, 0.0)
        * q.multiplier
        * (q.price - base[q.contract].price)
        for q in scenario.quotes
    )
    return product + hedge


def direct_pinned_delta(
    price_at: PriceAt, context: CarryCurveContext, spot_step: float
) -> float:
    """``D_F`` by repricing: spot moves, every listed quote is pinned."""
    plus = context.with_spot(context.spot + spot_step)
    minus = context.with_spot(context.spot - spot_step)
    return (
        _checked(price_at(plus.spot, plus.dividend()), "pinned spot up")
        - _checked(price_at(minus.spot, minus.dividend()), "pinned spot down")
    ) / (2 * spot_step)


def direct_frozen_curve_book_delta(
    price_at: PriceAt,
    context: CarryCurveContext,
    holdings: Mapping[str, float],
    spot_step: float,
) -> float:
    """``D_book`` by repricing at a FROZEN carry curve.

    The product is repriced on the base curve -- not on a curve rebuilt from
    pinned futures -- and each futures mark scales by ``s/S``, which is the
    frozen-carry direction the hedge is sized in.
    """
    div = context.dividend()
    product = (
        _checked(price_at(context.spot + spot_step, div), "frozen spot up")
        - _checked(price_at(context.spot - spot_step, div), "frozen spot down")
    ) / (2 * spot_step)
    hedge = sum(
        holdings.get(q.contract, 0.0) * q.multiplier * q.price / context.spot
        for q in context.quotes
    )
    return product + hedge


def direct_nodal_rhoq(
    price_at: PriceAt,
    context: CarryCurveContext,
    holdings: Mapping[str, float],
    yield_step: float,
    *,
    product_only: bool = False,
) -> Dict[str, float]:
    """``R_book,i`` (or ``R_i``) by repricing each node's yield bump."""
    out: Dict[str, float] = {}
    fixed = {} if product_only else holdings
    for quote in context.quotes:
        up = context.bump_node_yield(quote.contract, yield_step)
        down = context.bump_node_yield(quote.contract, -yield_step)
        out[quote.contract] = (
            scenario_book_value(price_at, context, up, fixed)
            - scenario_book_value(price_at, context, down, fixed)
        ) / (2 * yield_step)
    return out


def direct_parallel_rhoq(
    price_at: PriceAt,
    context: CarryCurveContext,
    holdings: Mapping[str, float],
    yield_step: float,
    *,
    product_only: bool = False,
) -> float:
    """``R_book,parallel`` by repricing one parallel curve shift."""
    fixed = {} if product_only else holdings
    up = context.parallel_yield_shift(yield_step)
    down = context.parallel_yield_shift(-yield_step)
    return (
        scenario_book_value(price_at, context, up, fixed)
        - scenario_book_value(price_at, context, down, fixed)
    ) / (2 * yield_step)


# ---------------------------------------------------------------------------
# The audit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentitySample:
    """A matched spot-difference identity, in signed currency Greek units."""

    spot_bump_rel: float
    delta_q_direct: float
    delta_f_direct: float
    residual_hands: float


def sample_chain_identity(
    price_at: PriceAt,
    context: CarryCurveContext,
    risk: FuturesBookRisk,
    settings: CarryRiskSettings,
) -> Tuple[IdentitySample, ...]:
    """Reprice BOTH spot directions at each step, keeping supplied buckets.

    The hedge's pricing delta is intentionally not substituted into this
    local derivative identity. The supplied buckets ARE retained: a wrong
    bucket must remain visible rather than being replaced by an audit's
    newly sampled, correct value. Exactly four price calls per level.
    """
    settings.require_resolved()
    bucket_sum = sum(b.price / risk.spot * b.bucket_currency for b in risk.buckets)
    samples = []
    for bump in settings.identity_spot_bumps_rel:
        step = bump * context.spot
        delta_q = direct_frozen_curve_book_delta(price_at, context, {}, step)
        delta_f = direct_pinned_delta(price_at, context, step)
        samples.append(IdentitySample(
            bump, delta_q, delta_f,
            (delta_q - delta_f - bucket_sum) / settings.reference_multiplier,
        ))
    return tuple(samples)


@dataclass(frozen=True)
class CarryAuditResult:
    """Direct measurements next to the algebraic predictions for the SAME
    actual holdings, plus the separate question of what the policy intended.

    ``status`` checks pricing-Greek reproduction and local chain consistency;
    it does not certify Greeks against an external reference.
    ``objective_achieved`` checks the policy's target. A ``nodes`` book is
    meant to keep ``D_F`` of spot delta, so consistency does not imply neutrality.
    """

    status: str
    reason: str
    holdings_kind: str
    spot_step: float
    yield_step: float
    reference_multiplier: float
    reference_notional: float
    delta_f_derived: float = float("nan")
    delta_f_direct: float = float("nan")
    identity_residual_hands: float = float("nan")
    mapped_net_delta: float = float("nan")
    direct_net_delta: float = float("nan")
    net_delta_audit_error_hands: float = float("nan")
    mapped_net_parallel_rhoq: float = float("nan")
    direct_net_parallel_rhoq: float = float("nan")
    parallel_rhoq_audit_error_bp: float = float("nan")
    mapped_nodal_rhoq: Dict[str, float] = field(default_factory=dict)
    direct_nodal_rhoq: Dict[str, float] = field(default_factory=dict)
    nodal_rhoq_audit_error_bp: Dict[str, float] = field(default_factory=dict)
    product_parallel_mapped: float = float("nan")
    product_parallel_direct: float = float("nan")
    price_calls: int = 0
    objective_achieved: Optional[bool] = None
    objective_reason: str = ""
    # delta_f_direct retains its original audit/pricing bump. Only
    # identity_samples uses the separate matched ladder. The old difference
    # delta_f_derived - delta_f_direct remains observable in this diagnostic.
    finite_bump_identity_residual_hands: float = float("nan")
    finite_bump_identity_status: str = "not_measured"
    finite_bump_identity_reason: str = ""
    #: Pricing delta minus the FINEST ladder delta. A numerical diagnostic of
    #: the pricing bump against a local derivative, and NOT a hedge error:
    #: its reference is deep sub-cell, so on a linear readout it carries the
    #: interpolation staircase and overstates anything hedge-relevant. For a
    #: hedge error use ``pricing_delta_hedge_gap_hands``, whose reference is
    #: the desk's own rebalance resolution.
    pricing_delta_local_gap_hands: float = float("nan")
    #: Pricing delta minus the delta at ``settings.hedge_resolution_rel``.
    #: NaN unless that setting is declared.
    pricing_delta_hedge_gap_hands: float = float("nan")
    hedge_gap_status: str = "not_measured"
    identity_spot_refinement_error_hands: float = float("nan")
    identity_status: str = "not_measured"
    identity_samples: Tuple[IdentitySample, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in AUDIT_STATUSES:
            raise ValidationError(
                f"audit status must be one of {AUDIT_STATUSES}, got {self.status!r}"
            )

    @property
    def measured(self) -> bool:
        return self.status != "not_measured"

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def worst_error_hands(self) -> float:
        return abs(self.net_delta_audit_error_hands)

    @property
    def worst_rhoq_error_bp(self) -> float:
        errors = [abs(self.parallel_rhoq_audit_error_bp)] + [
            abs(v) for v in self.nodal_rhoq_audit_error_bp.values()
        ]
        finite = [e for e in errors if math.isfinite(e)]
        return max(finite) if finite else float("nan")


def not_measured_audit(
    *,
    holdings_kind: str,
    reason: str,
    settings: CarryRiskSettings,
) -> CarryAuditResult:
    """A scheduled-but-unrun audit.  Every measurement stays NaN."""
    settings.require_resolved()
    return CarryAuditResult(
        status="not_measured",
        reason=reason,
        holdings_kind=holdings_kind,
        spot_step=float("nan"),
        yield_step=float("nan"),
        reference_multiplier=settings.reference_multiplier,
        reference_notional=float(settings.reference_notional),
    )


def audit_held_book(
    price_at: PriceAt,
    context: CarryCurveContext,
    risk: FuturesBookRisk,
    holdings: Mapping[str, float],
    *,
    settings: CarryRiskSettings,
    holdings_kind: str = "actual",
    ideal_net_delta: Optional[float] = None,
    ideal_net_parallel_rhoq: Optional[float] = None,
) -> CarryAuditResult:
    """Reprice the held book and compare with the algebraic prediction.

    The comparison is ``direct`` versus ``mapped`` for THESE holdings.  It
    does not require a zero residual: ``nodes``, ``spot_far``, partial ratios
    and rounded books all have intended non-zero residuals, and the optional
    ``ideal_*`` arguments record whether the policy's own target was met as a
    separate flag.
    """
    settings.require_resolved()
    if holdings_kind not in ("actual", "ideal"):
        raise ValidationError(f"unknown holdings kind: {holdings_kind!r}")
    original_price_at = price_at
    price_calls = 0

    def counted_price_at(spot, dividend):
        nonlocal price_calls
        price_calls += 1
        return original_price_at(spot, dividend)

    price_at = counted_price_at
    spot_step = float(settings.audit_spot_bump_rel) * context.spot
    yield_step = float(settings.audit_yield_bump)
    notional = float(settings.reference_notional)
    m_ref = float(settings.reference_multiplier)

    mapped_delta, mapped_rho = held_book_risk(risk, holdings)
    mapped_parallel = sum(mapped_rho.values())

    # The original (usually 1%) pinned spot move is now a diagnostic of the
    # hedge convention, not the local identity's measurement. It may leave
    # the curve's supported yield range even when the local ladder is valid.
    # Retain an explicit unavailable status without withholding the core audit.
    finite_bump_identity_status = "measured"
    finite_bump_identity_reason = ""
    try:
        delta_f_direct = direct_pinned_delta(price_at, context, spot_step)
    except ValidationError as error:
        delta_f_direct = float("nan")
        finite_bump_identity_status = "inconclusive"
        finite_bump_identity_reason = str(error)
    direct_delta = direct_frozen_curve_book_delta(
        price_at, context, holdings, spot_step
    )
    direct_rho = direct_nodal_rhoq(price_at, context, holdings, yield_step)
    direct_parallel = direct_parallel_rhoq(price_at, context, holdings, yield_step)
    product_parallel_direct = direct_parallel_rhoq(
        price_at, context, holdings, yield_step, product_only=True
    )

    finite_bump_identity_residual = (
        risk.delta_q - delta_f_direct - sum(
            b.price / risk.spot * b.bucket_currency for b in risk.buckets
        )
    ) / m_ref
    # The hedge gap measures the pricing delta against the secant across the
    # desk's OWN rebalance band, which is the slope that governs P&L between
    # rebalances. Opt-in: undeclared, it costs nothing and stays NaN rather
    # than substituting a local derivative that means something else.
    hedge_gap, hedge_gap_status = float("nan"), "not_measured"
    if settings.hedge_resolution_rel is not None:
        hedge_step = float(settings.hedge_resolution_rel) * context.spot
        hedge_delta = direct_frozen_curve_book_delta(price_at, context, {}, hedge_step)
        hedge_gap = (risk.delta_q - hedge_delta) / m_ref
        hedge_gap_status = "measured"
    identity_samples = sample_chain_identity(price_at, context, risk, settings)
    identity_residual = identity_samples[-1].residual_hands
    identity_refinement = abs(identity_residual - identity_samples[-2].residual_hands)
    # This is an observed spot-refinement allowance, not a confidence
    # interval or a certificate of spatial/quote-step accuracy. Requiring
    # the WHOLE allowance inside the budget avoids passing a marginal or
    # unstable estimate merely because its centre lies inside the budget.
    if abs(identity_residual) + identity_refinement <= settings.delta_tolerance_hands:
        identity_status = "pass"
    elif abs(identity_residual) - identity_refinement > settings.delta_tolerance_hands:
        identity_status = "fail"
    else:
        identity_status = "inconclusive"
    delta_error = spot_delta_hands(direct_delta - mapped_delta, m_ref)
    parallel_error = rhoq_bp_per_1pct(direct_parallel - mapped_parallel, notional)
    nodal_errors = {
        contract: rhoq_bp_per_1pct(direct_rho[contract] - mapped_rho[contract], notional)
        for contract in mapped_rho
    }

    failures = []
    if abs(delta_error) > settings.delta_tolerance_hands:
        failures.append(
            f"net delta {delta_error:+.4f} hands vs tolerance "
            f"{settings.delta_tolerance_hands}"
        )
    if identity_status == "fail":
        failures.append(
            f"matched identity residual {identity_residual:+.4f} hands "
            f"(refinement {identity_refinement:.4f}) vs tolerance "
            f"{settings.delta_tolerance_hands}"
        )
    if abs(parallel_error) > settings.rhoq_tolerance_bp:
        failures.append(
            f"parallel rhoq {parallel_error:+.4f} bp vs tolerance "
            f"{settings.rhoq_tolerance_bp}"
        )
    for contract, error in nodal_errors.items():
        if abs(error) > settings.rhoq_tolerance_bp:
            failures.append(f"nodal rhoq {contract} {error:+.4f} bp")

    status = "fail" if failures else "pass"
    reason = "; ".join(failures) if failures else "direct measurements agree"
    if not failures and identity_status == "inconclusive":
        status = "inconclusive"
        reason = (f"matched identity {identity_residual:+.6g} hands with spot refinement "
                  f"{identity_refinement:.6g} does not resolve tolerance "
                  f"{settings.delta_tolerance_hands}")

    objective_achieved: Optional[bool] = None
    objective_reason = ""
    if ideal_net_delta is not None and ideal_net_parallel_rhoq is not None:
        delta_gap = spot_delta_hands(direct_delta - float(ideal_net_delta), m_ref)
        parallel_gap = rhoq_bp_per_1pct(
            direct_parallel - float(ideal_net_parallel_rhoq), notional
        )
        objective_achieved = (
            abs(delta_gap) <= settings.delta_tolerance_hands
            and abs(parallel_gap) <= settings.rhoq_tolerance_bp
        )
        objective_reason = (
            f"delta gap {delta_gap:+.4f} hands, parallel gap {parallel_gap:+.4f} bp"
        )

    return CarryAuditResult(
        status=status,
        reason=reason,
        holdings_kind=holdings_kind,
        spot_step=spot_step,
        yield_step=yield_step,
        reference_multiplier=m_ref,
        reference_notional=notional,
        delta_f_derived=risk.delta_f_derived,
        delta_f_direct=delta_f_direct,
        identity_residual_hands=identity_residual,
        finite_bump_identity_residual_hands=finite_bump_identity_residual,
        finite_bump_identity_status=finite_bump_identity_status,
        finite_bump_identity_reason=finite_bump_identity_reason,
        pricing_delta_local_gap_hands=(risk.delta_q - identity_samples[-1].delta_q_direct) / m_ref,
        pricing_delta_hedge_gap_hands=hedge_gap,
        hedge_gap_status=hedge_gap_status,
        identity_spot_refinement_error_hands=identity_refinement,
        identity_status=identity_status,
        identity_samples=identity_samples,
        mapped_net_delta=mapped_delta,
        direct_net_delta=direct_delta,
        net_delta_audit_error_hands=delta_error,
        mapped_net_parallel_rhoq=mapped_parallel,
        direct_net_parallel_rhoq=direct_parallel,
        parallel_rhoq_audit_error_bp=parallel_error,
        mapped_nodal_rhoq=dict(mapped_rho),
        direct_nodal_rhoq=dict(direct_rho),
        nodal_rhoq_audit_error_bp=nodal_errors,
        product_parallel_mapped=risk.parallel_rhoq,
        product_parallel_direct=product_parallel_direct,
        price_calls=price_calls,
        objective_achieved=objective_achieved,
        objective_reason=objective_reason,
    )


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def refinement_ladder(
    measure: Callable[[float], float],
    *,
    initial_step: float,
    levels: int = 3,
    max_levels: int = 8,
    tolerance: float = 0.0,
) -> Tuple[Tuple[float, float], ...]:
    """Halve the step at least ``levels`` times, extending while it moves.

    The ladder never widens a tolerance to declare success: it keeps every
    sample so an unresolved measurement can be reported as ``inconclusive``
    with its whole history attached.
    """
    if levels < 3:
        raise ValidationError("a refinement ladder needs at least three levels")
    if max_levels < levels:
        raise ValidationError("max_levels must be at least levels")
    step = float(initial_step)
    if not math.isfinite(step) or step <= 0.0:
        raise ValidationError("initial_step must be finite and positive")
    samples = []
    for index in range(max_levels):
        samples.append((step, float(measure(step))))
        if index + 1 >= levels:
            previous, current = samples[-2][1], samples[-1][1]
            if abs(current - previous) <= tolerance:
                break
        step /= 2.0
    return tuple(samples)


def stabilised(
    samples: Sequence[Tuple[float, float]], tolerance: float
) -> bool:
    """True when the last two ladder samples agree within ``tolerance``."""
    if len(samples) < 2:
        return False
    return abs(samples[-1][1] - samples[-2][1]) <= tolerance


def status_from_interval(
    estimate: float, half_width: float, tolerance: float
) -> Tuple[str, str]:
    """Decide an audit status from a measurement and its own uncertainty.

    An estimate whose confidence interval is WIDER than the exposure budget
    cannot decide the question, even when the interval contains zero, so it
    is ``inconclusive`` rather than ``pass``.  This is the rule the Monte
    Carlo validation driver applies to seed-batch intervals; a deterministic
    engine passes ``half_width=0``.
    """
    estimate = float(estimate)
    half_width = float(half_width)
    tolerance = float(tolerance)
    if not math.isfinite(estimate) or not math.isfinite(half_width):
        return "not_measured", "estimate or interval is not finite"
    if half_width < 0.0 or tolerance < 0.0:
        raise ValidationError("half_width and tolerance must be non-negative")
    if half_width > tolerance:
        return (
            "inconclusive",
            f"interval half-width {half_width:.6g} exceeds the budget {tolerance:.6g}",
        )
    if abs(estimate) - half_width > tolerance:
        return "fail", f"|{estimate:.6g}| exceeds the budget {tolerance:.6g}"
    return "pass", "estimate lies inside the budget with a resolved interval"
