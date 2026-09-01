"""MARKET_IV_VEGA / MODEL_ARTIFACT bucketed points (moved verbatim from
GreeksCalculator)."""

from typing import List

from quantark.asset.equity.riskmeasures.bucketed_greeks import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
)
from quantark.util.exceptions import ValidationError


def calculate_points(
    calc, product, pricing_env, engine, request, coordinate, mode
) -> List[BucketedGreekPoint]:
    from quantark.asset.equity.riskmeasures.vol_model_risk import (
        VolModelRiskCalculator,
    )

    if mode != BucketedGreekDifferenceMode.CENTRAL:
        raise ValidationError(
            f"{coordinate.value} does not support {mode.value} in v1; "
            "request central mode or extend VolModelRiskCalculator"
        )
    risk_calc = VolModelRiskCalculator(
        heston_calibration_spec=request.heston_calibration_spec,
        slv_calibration_spec=request.slv_calibration_spec,
    )
    if coordinate == BucketedGreekCoordinate.MARKET_IV_VEGA:
        vol_result = risk_calc.calculate_market_vega(
            product, pricing_env, engine, request.market_vega_request
        )
        return _vol_risk_result_to_bucketed_points(
            coordinate=BucketedGreekCoordinate.MARKET_IV_VEGA,
            result=vol_result,
            convention_scale=0.01,
        )
    if request.model_risk_request is None:
        raise ValidationError("MODEL_ARTIFACT requires request.model_risk_request")
    model_result = risk_calc.calculate_model_risk(
        product, pricing_env, engine, request.model_risk_request
    )
    return _vol_risk_result_to_bucketed_points(
        coordinate=BucketedGreekCoordinate.MODEL_ARTIFACT,
        result=model_result,
        convention_scale=1.0,
    )


def _vol_risk_result_to_bucketed_points(
    *, coordinate, result, convention_scale
) -> List[BucketedGreekPoint]:
    model = result.metadata.get("model")
    points: List[BucketedGreekPoint] = []
    for point in result.points:
        reported = None
        if point.derivative is not None:
            reported = point.derivative * convention_scale
        if point.status == "failed":
            points.append(
                BucketedGreekPoint.failed(
                    coordinate=coordinate,
                    name=point.name,
                    bump_size=point.bump_size,
                    base_price=result.base_price,
                    error=point.error or "vol-model scenario failed",
                    convention_scale=convention_scale,
                    difference_mode=point.difference_mode,
                    up_price=point.up_price,
                    down_price=point.down_price,
                    model=model,
                    metadata=result.metadata,
                )
            )
            continue
        points.append(
            BucketedGreekPoint(
                coordinate=coordinate,
                name=point.name,
                reported=reported,
                derivative=point.derivative,
                pnl=point.pnl,
                bump_size=point.bump_size,
                convention_scale=convention_scale,
                base_price=result.base_price,
                up_price=point.up_price,
                down_price=point.down_price,
                difference_mode=point.difference_mode,
                status=point.status,
                error=point.error,
                model=model,
                metadata=result.metadata,
            )
        )
    return points
