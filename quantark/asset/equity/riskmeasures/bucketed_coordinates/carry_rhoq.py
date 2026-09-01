"""CARRY_RHOQ bucketed points + public futures rhoq bucket table
(moved verbatim from GreeksCalculator)."""

from copy import deepcopy
from typing import Dict, List, Optional

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
    div_bump = request.carry_bump or calc._bump_config.div_bump
    if request.futures_curve is not None:
        if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
            return _futures_one_sided_points(
                calc, product, pricing_env, engine, request, div_bump
            )
        if mode == BucketedGreekDifferenceMode.CENTRAL:
            return _futures_central_points(
                calc, product, pricing_env, engine, request, div_bump
            )
        raise ValidationError(
            f"unsupported CARRY_RHOQ difference mode: {mode.value}"
        )
    return _generic_points(
        calc, product, pricing_env, engine, request, mode, div_bump
    )


def _futures_one_sided_points(
    calc, product, pricing_env, engine, request, div_bump
) -> List[BucketedGreekPoint]:
    rows = calculate_futures_rhoq_buckets(
        calc,
        product,
        pricing_env,
        engine,
        request.futures_curve,
        mode=request.futures_carry_mode,
        div_bump=div_bump,
    )
    points: List[BucketedGreekPoint] = []
    for row in rows:
        reported = float(row["rhoq_bucket"])
        derivative = reported / 0.01
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.CARRY_RHOQ,
                name=f"carry_rhoq.{row['contract']}",
                reported=reported,
                derivative=derivative,
                pnl=derivative * float(row["div_bump"]),
                bump_size=float(row["div_bump"]),
                convention_scale=0.01,
                base_price=0.0,
                difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                contract=str(row["contract"]),
                maturity=float(row["maturity"]),
                future_price=float(row["future_price"]),
                extrapolated_tail=bool(row["extrapolated_tail"]),
                metadata={"source": "calculate_futures_rhoq_buckets"},
            )
        )
    return points


def _futures_central_points(
    calc, product, pricing_env, engine, request, div_bump
) -> List[BucketedGreekPoint]:
    from quantark.asset.equity.market import bump_term_yield_node
    from quantark.asset.equity.report.term_structure import BucketedDividendYield
    from quantark.param.div import ContinuousDividendYield
    from quantark.util.enum import FuturesCarryRiskMode

    futures_curve = request.futures_curve
    resolved_mode = (
        request.futures_carry_mode
        if request.futures_carry_mode is not None
        else futures_curve.mode
    )
    if resolved_mode is FuturesCarryRiskMode.MARKET_PRICE:
        raise ValidationError(
            "calculate_futures_rhoq_buckets does not support MARKET_PRICE "
            "mode (it supplies no carry curve for repricing the option)"
        )

    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    maturity = product.get_maturity(pricing_env)
    last_index = len(futures_curve.quotes) - 1

    def _tail_flag(i, quote):
        return (i == last_index and maturity > quote.maturity) or (
            i == 0 and maturity < quote.maturity
        )

    points: List[BucketedGreekPoint] = []
    if resolved_mode is FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
        base_div = futures_curve.to_dividend_yield_curve(pricing_env.rate_curve)
        base_env = deepcopy(pricing_env)
        base_env.div_yield = base_div
        base_price = bump_engine.price(product, base_env)
        for i, quote in enumerate(futures_curve.quotes):
            up_env = deepcopy(pricing_env)
            up_env.div_yield = bump_term_yield_node(base_div, i, div_bump)
            down_env = deepcopy(pricing_env)
            down_env.div_yield = bump_term_yield_node(base_div, i, -div_bump)
            up_price = bump_engine.price(product, up_env)
            down_price = bump_engine.price(product, down_env)
            derivative = (up_price - down_price) / (2.0 * div_bump)
            points.append(
                carry_rhoq_point(
                    name=f"carry_rhoq.{quote.contract}",
                    derivative=derivative,
                    bump_size=div_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=down_price,
                    difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                    contract=quote.contract,
                    maturity=quote.maturity,
                    future_price=quote.price,
                    extrapolated_tail=_tail_flag(i, quote),
                    source="central_futures_implied_carry_node_bump",
                )
            )
        return points

    base_price = bump_engine.price(product, pricing_env)
    base_div = pricing_env.div_yield or ContinuousDividendYield(0.0)
    edges = [0.0] + [q.maturity for q in futures_curve.quotes]
    if maturity > edges[-1]:
        edges[-1] = maturity
    for i, quote in enumerate(futures_curve.quotes):
        up_env = deepcopy(pricing_env)
        up_env.div_yield = BucketedDividendYield(
            base=base_div,
            bucket_start=edges[i],
            bucket_end=edges[i + 1],
            bump=div_bump,
        )
        down_env = deepcopy(pricing_env)
        down_env.div_yield = BucketedDividendYield(
            base=base_div,
            bucket_start=edges[i],
            bucket_end=edges[i + 1],
            bump=-div_bump,
        )
        up_price = bump_engine.price(product, up_env)
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (2.0 * div_bump)
        points.append(
            carry_rhoq_point(
                name=f"carry_rhoq.{quote.contract}",
                derivative=derivative,
                bump_size=div_bump,
                base_price=base_price,
                up_price=up_price,
                down_price=down_price,
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                contract=quote.contract,
                maturity=quote.maturity,
                future_price=quote.price,
                extrapolated_tail=_tail_flag(i, quote),
                source="central_theoretical_carry_bucket_bump",
            )
        )
    return points


def _generic_points(
    calc, product, pricing_env, engine, request, mode, div_bump
) -> List[BucketedGreekPoint]:
    from quantark.asset.equity.report.term_structure import (
        BucketedDividendYield,
        default_tenor_buckets,
    )
    from quantark.param.div import ContinuousDividendYield

    if mode not in (
        BucketedGreekDifferenceMode.ONE_SIDED_UP,
        BucketedGreekDifferenceMode.CENTRAL,
    ):
        raise ValidationError(
            f"unsupported CARRY_RHOQ difference mode: {mode.value}"
        )
    # spec WP3.3: with no explicit buckets and a term-structure carry
    # curve, buckets align to the curve's calibrated NODES (node bumps
    # rebuild the curve; no window-step discontinuities)
    if request.tenor_buckets is None and hasattr(
        pricing_env.div_yield, "times"
    ):
        return _node_aligned_points(
            calc, product, pricing_env, engine, mode, div_bump
        )
    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    maturity = product.get_maturity(pricing_env)
    buckets = tuple(request.tenor_buckets or default_tenor_buckets(maturity))
    base_div = pricing_env.div_yield or ContinuousDividendYield(0.0)
    base_price = bump_engine.price(product, pricing_env)
    points: List[BucketedGreekPoint] = []
    for bucket in buckets:
        up_env = deepcopy(pricing_env)
        up_env.div_yield = BucketedDividendYield(
            base=base_div,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=div_bump,
        )
        up_price = bump_engine.price(product, up_env)
        if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
            derivative = (up_price - base_price) / div_bump
            points.append(
                carry_rhoq_point(
                    name=f"carry_rhoq.{bucket.label}",
                    derivative=derivative,
                    bump_size=div_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=None,
                    difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                    bucket=bucket_label(bucket),
                    source="generic_tenor_carry_bucket_bump",
                )
            )
            continue

        down_env = deepcopy(pricing_env)
        down_env.div_yield = BucketedDividendYield(
            base=base_div,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=-div_bump,
        )
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (2.0 * div_bump)
        points.append(
            carry_rhoq_point(
                name=f"carry_rhoq.{bucket.label}",
                derivative=derivative,
                bump_size=div_bump,
                base_price=base_price,
                up_price=up_price,
                down_price=down_price,
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                bucket=bucket_label(bucket),
                source="generic_tenor_carry_bucket_bump",
            )
        )
    return points


def _node_aligned_points(
    calc, product, pricing_env, engine, mode, div_bump
) -> List[BucketedGreekPoint]:
    """Central node bumps on a TermStructureDividendYield, one point per
    CALIBRATED node (spec WP3.3). The term curve's own interpolation
    rebuilds carry between nodes."""
    from quantark.asset.equity.market.index_futures_curve import (
        bump_term_yield_node,
    )
    from quantark.param.node_roles import NodeRole, resolve_node_roles

    term_div = pricing_env.div_yield
    info = resolve_node_roles(
        list(term_div.times),
        getattr(term_div, "node_roles", None),
        getattr(term_div, "last_observable_tenor", None),
    )
    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    base_price = bump_engine.price(product, pricing_env)
    points: List[BucketedGreekPoint] = []
    for i, (tenor, role) in enumerate(zip(term_div.times, info.roles)):
        if role is not NodeRole.CALIBRATED:
            continue
        up_env = deepcopy(pricing_env)
        up_env.div_yield = bump_term_yield_node(term_div, i, +div_bump)
        down_env = deepcopy(pricing_env)
        down_env.div_yield = bump_term_yield_node(term_div, i, -div_bump)
        up_price = bump_engine.price(product, up_env)
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (2.0 * div_bump)
        point = carry_rhoq_point(
            name=f"carry_rhoq.node_{tenor:g}y",
            derivative=derivative,
            bump_size=div_bump,
            base_price=base_price,
            up_price=up_price,
            down_price=down_price,
            difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
            maturity=float(tenor),
            extrapolated_tail=bool(tenor > info.last_observable_tenor),
            source="node_aligned_term_yield_bump",
        )
        points.append(point)
    return points


def bucket_label(bucket) -> str:
    return f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)"


def carry_rhoq_point(
    *,
    name,
    derivative,
    bump_size,
    base_price,
    up_price,
    down_price,
    difference_mode,
    source,
    bucket=None,
    contract=None,
    maturity=None,
    future_price=None,
    extrapolated_tail=None,
) -> BucketedGreekPoint:
    reported = derivative * 0.01
    if difference_mode == BucketedGreekDifferenceMode.CENTRAL.value:
        pnl = (up_price - down_price) / 2.0
    else:
        pnl = up_price - base_price
    return BucketedGreekPoint(
        coordinate=BucketedGreekCoordinate.CARRY_RHOQ,
        name=name,
        reported=reported,
        derivative=derivative,
        pnl=pnl,
        bump_size=float(bump_size),
        convention_scale=0.01,
        base_price=float(base_price),
        up_price=None if up_price is None else float(up_price),
        down_price=None if down_price is None else float(down_price),
        difference_mode=difference_mode,
        bucket=bucket,
        contract=contract,
        maturity=None if maturity is None else float(maturity),
        future_price=None if future_price is None else float(future_price),
        extrapolated_tail=extrapolated_tail,
        metadata={"source": source},
    )


def calculate_futures_rhoq_buckets(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    futures_curve,
    *,
    mode=None,
    div_bump: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    Bucketed rhoq diagnostics per futures tenor (carry coordinate).
    The base PV is always computed internally (per-mode base environment);
    no ``base_price`` parameter — see calculate_futures_delta_buckets.

    One-sided **up** dividend bump, matching the scalar
    ``calculate_numerical_dividend_rho`` convention; output scaled to
    per-1% yield change via ``* (0.01 / div_bump)``.

    IMPLIED_FUTURES_CARRY: bumps one implied q(T_i) node at a time on the
    curve rebuilt from ``futures_curve``. THEORETICAL_CARRY: bumps
    ``pricing_env.div_yield`` on the interval (T_{i-1}, T_i] via
    ``BucketedDividendYield`` (the futures curve supplies metadata only).
    MARKET_PRICE is rejected: a zero bucket table can look like a real
    hedge result.
    """
    from quantark.asset.equity.market import bump_term_yield_node
    from quantark.asset.equity.report.term_structure import (
        BucketedDividendYield,
    )
    from quantark.param.div import ContinuousDividendYield
    from quantark.util.enum import FuturesCarryRiskMode

    resolved_mode = mode if mode is not None else futures_curve.mode
    if resolved_mode is FuturesCarryRiskMode.MARKET_PRICE:
        raise ValidationError(
            "calculate_futures_rhoq_buckets does not support MARKET_PRICE "
            "mode (it supplies no carry curve for repricing the option)"
        )
    div_bump = div_bump if div_bump is not None else calc._bump_config.div_bump
    if div_bump <= 0.0:
        raise ValidationError("div_bump must be positive")
    engine = calc._resolve_bump_engine(product, pricing_env, engine)

    maturity = product.get_maturity(pricing_env)
    last_index = len(futures_curve.quotes) - 1

    def _tail_flag(i: int, quote) -> bool:
        return (i == last_index and maturity > quote.maturity) or (
            i == 0 and maturity < quote.maturity
        )

    rows: List[Dict[str, object]] = []
    if resolved_mode is FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY:
        base_div = futures_curve.to_dividend_yield_curve(pricing_env.rate_curve)
        base_env = deepcopy(pricing_env)
        base_env.div_yield = base_div
        base_price = engine.price(product, base_env)
        for i, quote in enumerate(futures_curve.quotes):
            bumped_env = deepcopy(pricing_env)
            bumped_env.div_yield = bump_term_yield_node(base_div, i, div_bump)
            bumped_price = engine.price(product, bumped_env)
            rows.append(
                _rhoq_bucket_row(
                    quote, div_bump, base_price, bumped_price, _tail_flag(i, quote)
                )
            )
    else:  # THEORETICAL_CARRY: pricing_env.div_yield is the carry source
        base_price = engine.price(product, pricing_env)
        base_div = pricing_env.div_yield
        if base_div is None:
            base_div = ContinuousDividendYield(0.0)
        edges = [0.0] + [q.maturity for q in futures_curve.quotes]
        # a product maturing beyond the last futures tenor still carries
        # dividend exposure on (T_last, T*]; attribute that tail to the
        # last contract's bucket (roll-hedge convention, mirrored from the
        # implied mode's flat extrapolation) instead of dropping it
        if maturity > edges[-1]:
            edges[-1] = maturity
        for i, quote in enumerate(futures_curve.quotes):
            bumped_env = deepcopy(pricing_env)
            bumped_env.div_yield = BucketedDividendYield(
                base=base_div,
                bucket_start=edges[i],
                bucket_end=edges[i + 1],
                bump=div_bump,
            )
            bumped_price = engine.price(product, bumped_env)
            rows.append(
                _rhoq_bucket_row(
                    quote, div_bump, base_price, bumped_price, _tail_flag(i, quote)
                )
            )
    return rows


def _rhoq_bucket_row(quote, div_bump, base_price, bumped_price, extrapolated_tail):
    return {
        "contract": quote.contract,
        "maturity": quote.maturity,
        "future_price": quote.price,
        "div_bump": div_bump,
        "rhoq_bucket": (bumped_price - base_price) * (0.01 / div_bump),
        "extrapolated_tail": extrapolated_tail,
    }
