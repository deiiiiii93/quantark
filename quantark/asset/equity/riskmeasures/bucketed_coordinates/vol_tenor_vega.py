"""VOL_TENOR_VEGA bucketed points (moved verbatim from GreeksCalculator)."""

from copy import deepcopy
from typing import List

from quantark.asset.equity.riskmeasures.bucketed_coordinates.carry_rhoq import (
    bucket_label,
)
from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
)
from quantark.util.exceptions import NumericalError, ValidationError


def calculate_points(
    calc, product, pricing_env, engine, request, mode
) -> List[BucketedGreekPoint]:
    from quantark.asset.equity.report.term_structure import (
        BucketedVolSurface,
        default_tenor_buckets,
    )
    from quantark.volmodels.heston import HestonParams

    if isinstance(getattr(engine, "model_params", None), HestonParams):
        engine_name = type(engine).__name__
        if "SLV" in engine_name:
            raise ValidationError(
                "VOL_TENOR_VEGA bumps pricing_env.vol_surface directly; "
                "SLV market vega requires local-vol/leverage recalibration. "
                "Request MARKET_IV_VEGA instead."
            )
        raise ValidationError(
            "VOL_TENOR_VEGA bumps pricing_env.vol_surface directly; "
            "Heston uses model params calibrated from market IV. "
            "Request MARKET_IV_VEGA instead."
        )
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for bucketed vega.")
    if mode not in (
        BucketedGreekDifferenceMode.ONE_SIDED_UP,
        BucketedGreekDifferenceMode.CENTRAL,
    ):
        raise ValidationError(
            f"unsupported VOL_TENOR_VEGA difference mode: {mode.value}"
        )

    # spec WP3.3: with no explicit buckets and a term-structure ATM
    # curve, bump NODE vols and rebuild total variance (no window-step
    # calendar-arb discontinuities on dense grids)
    if request.tenor_buckets is None and hasattr(
        pricing_env.vol_surface, "times"
    ):
        return _node_aligned_points(calc, product, pricing_env, engine, request)

    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    maturity = product.get_maturity(pricing_env)
    buckets = tuple(request.tenor_buckets or default_tenor_buckets(maturity))
    base_price = bump_engine.price(product, pricing_env)
    points: List[BucketedGreekPoint] = []
    for bucket in buckets:
        up_env = deepcopy(pricing_env)
        up_env.vol_surface = BucketedVolSurface(
            base=pricing_env.vol_surface,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=request.vol_bump,
        )
        up_price = bump_engine.price(product, up_env)
        if mode == BucketedGreekDifferenceMode.ONE_SIDED_UP:
            derivative = (up_price - base_price) / request.vol_bump
            points.append(
                _vol_tenor_vega_point(
                    bucket=bucket,
                    derivative=derivative,
                    bump_size=request.vol_bump,
                    base_price=base_price,
                    up_price=up_price,
                    down_price=None,
                    difference_mode=BucketedGreekDifferenceMode.ONE_SIDED_UP.value,
                )
            )
            continue

        down_env = deepcopy(pricing_env)
        down_env.vol_surface = BucketedVolSurface(
            base=pricing_env.vol_surface,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=-request.vol_bump,
        )
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (2.0 * request.vol_bump)
        points.append(
            _vol_tenor_vega_point(
                bucket=bucket,
                derivative=derivative,
                bump_size=request.vol_bump,
                base_price=base_price,
                up_price=up_price,
                down_price=down_price,
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
            )
        )
    return points


def _node_aligned_points(
    calc, product, pricing_env, engine, request
) -> List[BucketedGreekPoint]:
    """Central node-vol bumps on a term ATM curve, one point per
    CALIBRATED node, reported per +1 vol pt (spec WP3.3). If a down bump
    produces negative forward variance (calendar arbitrage on the pricing
    grid), fall back to one-sided-up and record the adjustment."""
    from quantark.param.node_roles import NodeRole, resolve_node_roles

    surface = pricing_env.vol_surface
    info = resolve_node_roles(
        list(surface.times),
        getattr(surface, "node_roles", None),
        getattr(surface, "last_observable_tenor", None),
    )
    vol_bump = float(request.vol_bump)
    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    base_price = bump_engine.price(product, pricing_env)
    points: List[BucketedGreekPoint] = []
    for i, (tenor, role) in enumerate(zip(surface.times, info.roles)):
        if role is not NodeRole.CALIBRATED:
            continue
        def _try_price(node_bump: float):
            env = deepcopy(pricing_env)
            env.vol_surface = bump_term_vol_node(
                surface, i, node_bump
            )
            try:
                return bump_engine.price(product, env), None
            except NumericalError as exc:
                # negative forward variance: this bump direction is not
                # representable on the pricing grid (calendar arbitrage)
                return None, str(exc)

        up_price, up_error = _try_price(+vol_bump)
        down_price, down_error = _try_price(-vol_bump)
        adjusted = up_price is None or down_price is None
        status, error = "ok", None
        if up_price is not None and down_price is not None:
            derivative = (up_price - down_price) / (2.0 * vol_bump)
            pnl = (up_price - down_price) / 2.0
            diff_mode = BucketedGreekDifferenceMode.CENTRAL.value
        elif up_price is not None:
            # one-sided-up is explicitly allowed when recorded (§6.3)
            derivative = (up_price - base_price) / vol_bump
            pnl = up_price - base_price
            diff_mode = BucketedGreekDifferenceMode.ONE_SIDED_UP.value
        elif down_price is not None:
            derivative = (base_price - down_price) / vol_bump
            pnl = base_price - down_price
            diff_mode = "one_sided_down"
        else:
            derivative, pnl = None, None
            diff_mode = BucketedGreekDifferenceMode.CENTRAL.value
            status = "failed"
            error = f"up: {up_error}; down: {down_error}"
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.VOL_TENOR_VEGA,
                name=f"vol_tenor_vega.node_{tenor:g}y",
                reported=(
                    None if derivative is None else derivative * 0.01
                ),
                derivative=derivative,
                pnl=pnl,
                bump_size=vol_bump,
                convention_scale=0.01,
                base_price=float(base_price),
                up_price=None if up_price is None else float(up_price),
                down_price=(
                    None if down_price is None else float(down_price)
                ),
                difference_mode=diff_mode,
                status=status,
                error=error,
                maturity=float(tenor),
                extrapolated_tail=bool(tenor > info.last_observable_tenor),
                metadata={
                    "unit": "per_1volpt",
                    "source": "node_aligned_term_vol_bump",
                    "rebuild_rule": "node vol bump, total variance "
                    "rebuilt by term interpolation",
                    "adjusted_to_one_sided": adjusted,
                    "roles_inferred": info.roles_inferred,
                },
            )
        )
    return points


def bump_term_vol_node(surface, node_index: int, bump: float):
    """Copy of a term ATM vol curve with one node's vol bumped."""
    from quantark.param.vol.vol_surface import TermStructureVolSurface

    vols = [float(v) for v in surface.vols]
    vols[node_index] += float(bump)
    return TermStructureVolSurface(
        times=list(surface.times),
        vols=vols,
        node_roles=getattr(surface, "node_roles", None),
        last_observable_tenor=getattr(
            surface, "last_observable_tenor", None
        ),
    )


def _vol_tenor_vega_point(
    *,
    bucket,
    derivative,
    bump_size,
    base_price,
    up_price,
    down_price,
    difference_mode,
) -> BucketedGreekPoint:
    if difference_mode == BucketedGreekDifferenceMode.CENTRAL.value:
        pnl = (up_price - down_price) / 2.0
    else:
        pnl = up_price - base_price
    return BucketedGreekPoint(
        coordinate=BucketedGreekCoordinate.VOL_TENOR_VEGA,
        name=f"vol_tenor_vega.{bucket.label}",
        reported=derivative * 0.01,
        derivative=derivative,
        pnl=pnl,
        bump_size=float(bump_size),
        convention_scale=0.01,
        base_price=float(base_price),
        up_price=float(up_price),
        down_price=None if down_price is None else float(down_price),
        difference_mode=difference_mode,
        bucket=bucket_label(bucket),
        metadata={"source": "generic_tenor_vol_bucket_bump"},
    )
