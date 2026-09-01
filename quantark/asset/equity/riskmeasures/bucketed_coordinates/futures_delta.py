"""FUTURES_DELTA bucketed points + public futures delta bucket table
(moved verbatim from GreeksCalculator)."""

from copy import deepcopy
from typing import Dict, List

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def calculate_points(
    calc, product, pricing_env, engine, request, mode
) -> List[BucketedGreekPoint]:
    if request.futures_curve is None:
        raise ValidationError("FUTURES_DELTA requires request.futures_curve")
    if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
        return _one_sided_points(calc, product, pricing_env, engine, request)
    if mode == BucketedGreekDifferenceMode.CENTRAL:
        return _central_points(calc, product, pricing_env, engine, request)
    raise ValidationError(
        f"unsupported FUTURES_DELTA difference mode: {mode.value}"
    )


def _one_sided_points(
    calc, product, pricing_env, engine, request
) -> List[BucketedGreekPoint]:
    rows = calculate_futures_delta_buckets(
        calc,
        product,
        pricing_env,
        engine,
        request.futures_curve,
        mode=request.futures_carry_mode,
        price_bump=request.futures_price_bump,
    )
    points: List[BucketedGreekPoint] = []
    for row in rows:
        derivative = float(row["delta_bucket"])
        bump_size = float(row["price_bump"])
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.FUTURES_DELTA,
                name=f"futures_delta.{row['contract']}",
                reported=derivative,
                derivative=derivative,
                pnl=derivative * bump_size,
                bump_size=bump_size,
                convention_scale=1.0,
                base_price=0.0,
                difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                contract=str(row["contract"]),
                maturity=float(row["maturity"]),
                future_price=float(row["future_price"]),
                delta_per_hand=float(row["delta_per_hand"]),
                hedge_hands=float(row["hedge_hands"]),
                extrapolated_tail=bool(row["extrapolated_tail"]),
                metadata={"source": "calculate_futures_delta_buckets"},
            )
        )
    return points


def _central_points(
    calc, product, pricing_env, engine, request
) -> List[BucketedGreekPoint]:
    from quantark.asset.equity.market import hedge_hands as _hedge_hands
    from quantark.util.enum import FuturesCarryRiskMode

    futures_curve = request.futures_curve
    resolved_mode = (
        request.futures_carry_mode
        if request.futures_carry_mode is not None
        else futures_curve.mode
    )
    if resolved_mode is not FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
        raise ValidationError(
            "calculate_futures_delta_buckets requires IMPLIED_FUTURES_CARRY "
            f"mode, got {resolved_mode}"
        )

    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    base_env = deepcopy(pricing_env)
    base_env.div_yield = futures_curve.to_dividend_yield_curve(
        pricing_env.rate_curve
    )
    base_price = bump_engine.price(product, base_env)
    maturity = product.get_maturity(pricing_env)
    last_index = len(futures_curve.quotes) - 1
    points: List[BucketedGreekPoint] = []

    for i, quote in enumerate(futures_curve.quotes):
        up_curve = futures_curve.bump_contract(
            quote.contract, request.futures_price_bump
        )
        down_curve = futures_curve.bump_contract(
            quote.contract, -request.futures_price_bump
        )
        up_env = deepcopy(pricing_env)
        up_env.div_yield = up_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        down_env = deepcopy(pricing_env)
        down_env.div_yield = down_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        up_price = bump_engine.price(product, up_env)
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (
            2.0 * request.futures_price_bump
        )
        per_hand = futures_curve.delta_per_hand(quote.contract)
        extrapolated_tail = (
            i == last_index and maturity > quote.maturity
        ) or (i == 0 and maturity < quote.maturity)
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.FUTURES_DELTA,
                name=f"futures_delta.{quote.contract}",
                reported=derivative,
                derivative=derivative,
                pnl=(up_price - down_price) / 2.0,
                bump_size=float(request.futures_price_bump),
                convention_scale=1.0,
                base_price=float(base_price),
                up_price=float(up_price),
                down_price=float(down_price),
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                contract=quote.contract,
                maturity=float(quote.maturity),
                future_price=float(quote.price),
                delta_per_hand=float(per_hand),
                hedge_hands=float(_hedge_hands(derivative, per_hand)),
                extrapolated_tail=bool(extrapolated_tail),
                metadata={"source": "central_futures_mark_bump"},
            )
        )
    return points


def calculate_futures_delta_buckets(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    futures_curve,
    *,
    mode=None,
    price_bump: float = 1.0,
) -> List[Dict[str, object]]:
    """
    Futures-tenor bucket deltas: delta_bucket_i = dPV / dF_i.

    Bumps one futures mark at a time, rebuilds the implied q(T) curve,
    and reprices (one-sided up bump). The base and bumped legs reuse the
    same engine/params unchanged, so MC engines with a fixed seed price
    with common random numbers. The base PV is always computed internally
    under the implied-carry environment (div_yield rebuilt from
    ``futures_curve``); no ``base_price`` parameter is accepted because a
    caller's PV under ``pricing_env.div_yield`` would silently shift every
    bucket.

    hedge_hands = -delta_bucket / delta_per_hand (fractional, unrounded).
    Rows are flagged ``extrapolated_tail`` when the product maturity lies
    outside the quoted node range (flat-extrapolated carry).
    """
    from quantark.asset.equity.market import hedge_hands as _hedge_hands
    from quantark.util.enum import FuturesCarryRiskMode

    resolved_mode = mode if mode is not None else futures_curve.mode
    if resolved_mode is not FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
        raise ValidationError(
            "calculate_futures_delta_buckets requires IMPLIED_FUTURES_CARRY "
            f"mode, got {resolved_mode}"
        )
    if price_bump <= 0.0:
        raise ValidationError("price_bump must be positive")

    engine = calc._resolve_bump_engine(product, pricing_env, engine)
    base_env = deepcopy(pricing_env)
    base_env.div_yield = futures_curve.to_dividend_yield_curve(
        pricing_env.rate_curve
    )
    base_price = engine.price(product, base_env)

    maturity = product.get_maturity(pricing_env)
    last_index = len(futures_curve.quotes) - 1
    rows: List[Dict[str, object]] = []
    for i, quote in enumerate(futures_curve.quotes):
        bumped_curve = futures_curve.bump_contract(quote.contract, price_bump)
        bumped_env = deepcopy(pricing_env)
        bumped_env.div_yield = bumped_curve.to_dividend_yield_curve(
            pricing_env.rate_curve
        )
        bumped_price = engine.price(product, bumped_env)
        delta_bucket = (bumped_price - base_price) / price_bump
        per_hand = futures_curve.delta_per_hand(quote.contract)
        extrapolated_tail = (
            i == last_index and maturity > quote.maturity
        ) or (i == 0 and maturity < quote.maturity)
        rows.append(
            {
                "contract": quote.contract,
                "maturity": quote.maturity,
                "future_price": quote.price,
                "price_bump": price_bump,
                "delta_bucket": delta_bucket,
                "delta_per_hand": per_hand,
                "hedge_hands": _hedge_hands(delta_bucket, per_hand),
                "extrapolated_tail": extrapolated_tail,
            }
        )
    return rows
