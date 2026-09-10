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
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

from quantark.backtest.futures_risk import FuturesBookRisk, FuturesBucket
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.util.exceptions import ValidationError

PriceAt = Callable[[float, Any], float]

__all__ = [
    "BucketSample",
    "ProductCarryRisk",
    "aggregate_book_risk",
    "buckets_of",
    "measure_product_carry_risk",
    "sample_buckets",
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
