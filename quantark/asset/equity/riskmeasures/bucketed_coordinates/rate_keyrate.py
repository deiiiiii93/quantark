"""RATE_KEYRATE bucketed points (moved verbatim from GreeksCalculator)."""

from copy import deepcopy
from typing import List

from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
    RateKeyrateConvention,
)
from quantark.util.exceptions import ValidationError

_REBUILD_RULE = {
    RateKeyrateConvention.CARRY_INVARIANT: (
        "zero-rate pillar bump; carry-invariant q re-derivation (F unchanged) -> pure discounting"
    ),
    RateKeyrateConvention.DIVIDEND_HELD: (
        "zero-rate pillar bump; dividend yield held (F moves with r)"
    ),
}


def calculate_points(
    calc, product, pricing_env, engine, request, mode
) -> List[BucketedGreekPoint]:
    """Per-CALIBRATED-pillar zero-rate bumps + a parallel reconciliation
    point (spec WP3.3). Reported per +1bp; central differences.

    ``request.rate_keyrate_convention`` selects what the bump holds fixed:
    CARRY_INVARIANT (default, unchanged) re-derives the dividend yield so the
    forward is held; DIVIDEND_HELD (patch spec 2026-09-03 §6) replaces only the
    rate curve so the forward moves with the rate."""
    from quantark.param.node_roles import NodeRole, resolve_node_roles
    from quantark.param.rrf import ParallelShiftRateCurve
    from quantark.param.rrf.key_rate import key_rate_bumped_zero_curve

    if mode != BucketedGreekDifferenceMode.CENTRAL:
        raise ValidationError(
            f"RATE_KEYRATE supports central mode only, got {mode.value}"
        )
    curve = pricing_env.rate_curve
    tenors = list(getattr(curve, "tenors", []) or [])
    if not tenors:
        raise ValidationError(
            "RATE_KEYRATE requires an interpolated rate curve with pillars"
        )
    info = resolve_node_roles(
        tenors,
        getattr(curve, "node_roles", None),
        getattr(curve, "last_observable_tenor", None),
    )
    bump = request.rate_bump if request.rate_bump is not None else 1e-4
    convention = request.rate_keyrate_convention
    rebuild_rule = _REBUILD_RULE[convention]
    bump_engine = calc._resolve_bump_engine(product, pricing_env, engine)
    base_price = bump_engine.price(product, pricing_env)

    def _rate_bumped_env(bumped_curve):
        env = deepcopy(pricing_env)
        env.rate_curve = bumped_curve
        if convention is RateKeyrateConvention.DIVIDEND_HELD:
            # the dividend yield is held: only the rate curve moves, so the
            # forward moves with it (the PnL-explain factor model's rate step)
            return env
        # desk convention (spec WP3.3): carry B(T) is the invariant, so
        # a discount bump re-derives q pointwise and F(0,T) is unchanged
        # -> the bump is pure discounting. This applies ALSO when
        # div_yield is None (PricingEnvironment treats None as zero
        # yield): wrap an explicit zero-yield base, otherwise the
        # forward would move and the F-unchanged metadata would be false.
        from quantark.param.div import ContinuousDividendYield
        from quantark.param.div.dividend_yield import (
            CarryInvariantDividendYield,
        )

        base_div = (
            pricing_env.div_yield
            if pricing_env.div_yield is not None
            else ContinuousDividendYield(0.0)
        )
        env.div_yield = CarryInvariantDividendYield(
            base=base_div,
            base_rate_curve=curve,
            bumped_rate_curve=bumped_curve,
        )
        return env

    points: List[BucketedGreekPoint] = []
    for tenor, role in zip(tenors, info.roles):
        if role is not NodeRole.CALIBRATED:
            continue
        up_env = _rate_bumped_env(
            key_rate_bumped_zero_curve(curve, tenor, +bump)
        )
        down_env = _rate_bumped_env(
            key_rate_bumped_zero_curve(curve, tenor, -bump)
        )
        up_price = bump_engine.price(product, up_env)
        down_price = bump_engine.price(product, down_env)
        derivative = (up_price - down_price) / (2.0 * bump)
        points.append(
            BucketedGreekPoint(
                coordinate=BucketedGreekCoordinate.RATE_KEYRATE,
                name=f"rate_keyrate.{tenor:g}y",
                reported=derivative * 1e-4,
                derivative=derivative,
                pnl=(up_price - down_price) / 2.0,
                bump_size=float(bump),
                convention_scale=1e-4,
                base_price=float(base_price),
                up_price=float(up_price),
                down_price=float(down_price),
                difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
                maturity=float(tenor),
                extrapolated_tail=bool(
                    tenor > info.last_observable_tenor
                ),
                metadata={
                    "unit": "per_1bp",
                    "roles_inferred": info.roles_inferred,
                    "rebuild_rule": rebuild_rule,
                    "convention": convention.value,
                },
            )
        )
    par_up = _rate_bumped_env(ParallelShiftRateCurve(curve, +bump))
    par_down = _rate_bumped_env(ParallelShiftRateCurve(curve, -bump))
    par_derivative = (
        bump_engine.price(product, par_up)
        - bump_engine.price(product, par_down)
    ) / (2.0 * bump)
    parallel_per_1bp = par_derivative * 1e-4
    sum_of_buckets = float(
        sum(pt.reported for pt in points if pt.reported is not None)
    )
    points.append(
        BucketedGreekPoint(
            coordinate=BucketedGreekCoordinate.RATE_KEYRATE,
            name="rate_keyrate.parallel",
            reported=parallel_per_1bp,
            derivative=par_derivative,
            pnl=parallel_per_1bp,
            bump_size=float(bump),
            convention_scale=1e-4,
            base_price=float(base_price),
            difference_mode=BucketedGreekDifferenceMode.CENTRAL.value,
            metadata={
                "unit": "per_1bp",
                "sum_of_buckets": sum_of_buckets,
                "reconciles": bool(
                    abs(sum_of_buckets - parallel_per_1bp)
                    <= 0.05 * max(abs(parallel_per_1bp), 1e-12)
                ),
                "roles_inferred": info.roles_inferred,
                "rebuild_rule": "ParallelShiftRateCurve; " + rebuild_rule,
                "convention": convention.value,
            },
        )
    )
    return points
