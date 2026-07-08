from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.engine.pde import LocalVolPDESolver
from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import (
    BucketedGreekCoordinate,
    BucketedGreekDifferenceMode,
    BucketedGreekPoint,
    BucketedGreeksRequest,
    BucketedGreeksResult,
    GreeksCalculator,
    VolModelRiskCalculator,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    GridVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.risk import MarketVegaRequest, ModelRiskRequest, SurfaceBump


def test_bucketed_request_rejects_empty_coordinates():
    with pytest.raises(ValidationError, match="coordinates must be non-empty"):
        BucketedGreeksRequest(coordinates=())


def test_bucketed_request_rejects_bad_difference_mode():
    with pytest.raises(ValidationError, match="difference_mode"):
        BucketedGreeksRequest(difference_mode="central")


def test_bucketed_request_rejects_bad_override_coordinate():
    with pytest.raises(ValidationError, match="difference_mode_overrides keys"):
        BucketedGreeksRequest(
            difference_mode_overrides={
                "carry_rhoq": BucketedGreekDifferenceMode.CENTRAL
            }
        )


def test_bucketed_request_rejects_bad_override_mode():
    with pytest.raises(ValidationError, match="difference_mode_overrides values"):
        BucketedGreeksRequest(
            difference_mode_overrides={
                BucketedGreekCoordinate.CARRY_RHOQ: "central"
            }
        )


def test_bucketed_request_rejects_nonpositive_bumps():
    with pytest.raises(ValidationError, match="futures_price_bump"):
        BucketedGreeksRequest(futures_price_bump=0.0)
    with pytest.raises(ValidationError, match="vol_bump"):
        BucketedGreeksRequest(vol_bump=-0.01)
    with pytest.raises(ValidationError, match="carry_bump"):
        BucketedGreeksRequest(carry_bump=0.0)


def test_bucketed_result_filters_points():
    ok = BucketedGreekPoint(
        coordinate=BucketedGreekCoordinate.CARRY_RHOQ,
        name="carry_rhoq.IC01",
        reported=-1.0,
        derivative=-100.0,
        pnl=-0.01,
        bump_size=0.0001,
        convention_scale=0.01,
        base_price=10.0,
        status="ok",
    )
    failed = BucketedGreekPoint.failed(
        coordinate=BucketedGreekCoordinate.MARKET_IV_VEGA,
        name="market_iv.parallel",
        bump_size=0.01,
        base_price=10.0,
        error="failed scenario",
    )
    result = BucketedGreeksResult(points=(ok, failed))

    assert result.successful_points == (ok,)
    assert result.failed_points == (failed,)
    assert result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ) == (ok,)


def _basic_env(spot=100.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 8),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.20),
        div_yield=ContinuousDividendYield(0.01),
    )


def _basic_call():
    return EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.5)


def _basic_curve(spot=100.0):
    return IndexFuturesCurve(
        underlying="IC",
        spot=spot,
        quotes=[
            IndexFuturesQuote("IC00", maturity=0.25, price=100.2, multiplier=200.0),
            IndexFuturesQuote("IC01", maturity=0.50, price=100.7, multiplier=200.0),
        ],
    )


def test_facade_requires_futures_curve_for_futures_delta():
    request = BucketedGreeksRequest(
        coordinates=(BucketedGreekCoordinate.FUTURES_DELTA,)
    )
    with pytest.raises(ValidationError, match="FUTURES_DELTA requires"):
        GreeksCalculator().calculate_bucketed_greeks(
            _basic_call(), _basic_env(), BlackScholesEngine(), request
        )


def test_facade_rejects_override_for_unrequested_coordinate():
    request = BucketedGreeksRequest(
        coordinates=(BucketedGreekCoordinate.VOL_TENOR_VEGA,),
        difference_mode_overrides={
            BucketedGreekCoordinate.CARRY_RHOQ: BucketedGreekDifferenceMode.CENTRAL
        },
    )
    with pytest.raises(ValidationError, match="override coordinate"):
        GreeksCalculator().calculate_bucketed_greeks(
            _basic_call(), _basic_env(), BlackScholesEngine(), request
        )


def test_facade_resolves_default_futures_coordinates_when_curve_supplied():
    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(),
        _basic_env(),
        BlackScholesEngine(),
        BucketedGreeksRequest(futures_curve=_basic_curve()),
    )
    assert {
        point.coordinate for point in result.points
    } == {
        BucketedGreekCoordinate.FUTURES_DELTA,
        BucketedGreekCoordinate.CARRY_RHOQ,
    }


def test_facade_futures_delta_one_sided_matches_existing_rows():
    calc = GreeksCalculator()
    product = _basic_call()
    env = _basic_env()
    engine = BlackScholesEngine()
    curve = _basic_curve()
    expected = calc.calculate_futures_delta_buckets(
        product, env, engine, curve, price_bump=1.0
    )

    result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.FUTURES_DELTA,),
            futures_curve=curve,
            futures_price_bump=1.0,
        ),
    )

    points = result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)
    assert len(points) == len(expected)
    for point, row in zip(points, expected):
        assert point.contract == row["contract"]
        assert point.derivative == pytest.approx(row["delta_bucket"])
        assert point.reported == pytest.approx(row["delta_bucket"])
        assert point.pnl == pytest.approx(row["delta_bucket"] * row["price_bump"])
        assert point.delta_per_hand == row["delta_per_hand"]
        assert point.hedge_hands == pytest.approx(row["hedge_hands"])
        assert point.extrapolated_tail is row["extrapolated_tail"]
        assert point.difference_mode == "one_sided_up"


def test_facade_futures_delta_central_override_uses_central_prices():
    calc = GreeksCalculator()
    product = _basic_call()
    env = _basic_env()
    engine = BlackScholesEngine()
    curve = _basic_curve()

    result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.FUTURES_DELTA,),
            futures_curve=curve,
            futures_price_bump=0.5,
            difference_mode_overrides={
                BucketedGreekCoordinate.FUTURES_DELTA: BucketedGreekDifferenceMode.CENTRAL
            },
        ),
    )

    points = result.by_coordinate(BucketedGreekCoordinate.FUTURES_DELTA)
    assert points
    assert all(point.difference_mode == "central" for point in points)
    assert all(point.up_price is not None for point in points)
    assert all(point.down_price is not None for point in points)
    assert all(
        point.pnl == pytest.approx((point.up_price - point.down_price) / 2.0)
        for point in points
    )


def test_facade_futures_rhoq_one_sided_matches_existing_rows():
    calc = GreeksCalculator()
    product = _basic_call()
    env = _basic_env()
    engine = BlackScholesEngine()
    curve = _basic_curve()
    expected = calc.calculate_futures_rhoq_buckets(
        product, env, engine, curve, div_bump=0.0001
    )

    result = calc.calculate_bucketed_greeks(
        product,
        env,
        engine,
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,),
            futures_curve=curve,
            carry_bump=0.0001,
        ),
    )

    points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    assert len(points) == len(expected)
    for point, row in zip(points, expected):
        assert point.contract == row["contract"]
        assert point.reported == pytest.approx(row["rhoq_bucket"])
        assert point.derivative == pytest.approx(row["rhoq_bucket"] / 0.01)
        assert point.pnl == pytest.approx(point.derivative * row["div_bump"])
        assert point.difference_mode == "one_sided_up"


def test_facade_futures_rhoq_central_override_uses_central_prices():
    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(),
        _basic_env(),
        BlackScholesEngine(),
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,),
            futures_curve=_basic_curve(),
            carry_bump=0.0001,
            difference_mode_overrides={
                BucketedGreekCoordinate.CARRY_RHOQ: BucketedGreekDifferenceMode.CENTRAL
            },
        ),
    )
    points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    assert points
    assert all(point.difference_mode == "central" for point in points)
    assert all(point.up_price is not None for point in points)
    assert all(point.down_price is not None for point in points)
    assert all(
        point.reported == pytest.approx(point.derivative * 0.01)
        for point in points
    )


def test_facade_generic_carry_rhoq_without_futures_curve():
    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(),
        _basic_env(),
        BlackScholesEngine(),
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,),
            carry_bump=0.0001,
        ),
    )
    points = result.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    assert points
    assert all(point.bucket for point in points)
    assert all(
        point.reported == pytest.approx(point.derivative * 0.01)
        for point in points
    )


def test_facade_generic_vol_tenor_vega_defaults_to_central():
    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(),
        _basic_env(),
        BlackScholesEngine(),
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.VOL_TENOR_VEGA,),
            vol_bump=0.01,
        ),
    )
    points = result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    assert points
    assert all(point.difference_mode == "central" for point in points)
    assert all(
        point.reported == pytest.approx(point.derivative * 0.01)
        for point in points
    )
    assert all(
        point.up_price is not None and point.down_price is not None
        for point in points
    )


def test_facade_generic_vol_tenor_vega_one_sided_override():
    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(),
        _basic_env(),
        BlackScholesEngine(),
        BucketedGreeksRequest(
            coordinates=(BucketedGreekCoordinate.VOL_TENOR_VEGA,),
            vol_bump=0.01,
            difference_mode_overrides={
                BucketedGreekCoordinate.VOL_TENOR_VEGA: BucketedGreekDifferenceMode.ONE_SIDED_UP
            },
        ),
    )
    points = result.by_coordinate(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    assert points
    assert all(point.difference_mode == "one_sided_up" for point in points)
    assert all(point.down_price is None for point in points)


def _grid_surface(vol=0.2):
    return GridVolSurface(
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.25, 0.75, 1.5],
        iv_grid=np.full((3, 3), vol),
    )


def _grid_env():
    return _basic_env()


def test_facade_market_iv_vega_delegates_to_vol_model_calculator():
    env = _grid_env()
    env.vol_surface = _grid_surface()
    engine = LocalVolPDESolver(params=PDEParams(grid_size=80, time_steps=40))
    request = BucketedGreeksRequest(
        coordinates=(BucketedGreekCoordinate.MARKET_IV_VEGA,),
        market_vega_request=MarketVegaRequest(
            surface_bumps=(SurfaceBump.parallel(0.01),)
        ),
    )

    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(), env, engine, request
    )
    direct = VolModelRiskCalculator().calculate_market_vega(
        _basic_call(), env, engine, request.market_vega_request
    )

    point = result.points[0]
    direct_point = direct.points[0]
    assert point.coordinate == BucketedGreekCoordinate.MARKET_IV_VEGA
    assert point.name == direct_point.name
    assert point.derivative == pytest.approx(direct_point.derivative)
    assert point.pnl == pytest.approx(direct_point.pnl)
    assert point.reported == pytest.approx(direct_point.derivative * 0.01)
    assert point.model == direct.metadata["model"]


def test_facade_market_iv_vega_rejects_one_sided_override():
    env = _grid_env()
    env.vol_surface = _grid_surface()
    engine = LocalVolPDESolver(params=PDEParams(grid_size=80, time_steps=40))
    with pytest.raises(
        ValidationError, match="market_iv_vega does not support one_sided_up"
    ):
        GreeksCalculator().calculate_bucketed_greeks(
            _basic_call(),
            env,
            engine,
            BucketedGreeksRequest(
                coordinates=(BucketedGreekCoordinate.MARKET_IV_VEGA,),
                difference_mode_overrides={
                    BucketedGreekCoordinate.MARKET_IV_VEGA: BucketedGreekDifferenceMode.ONE_SIDED_UP
                },
            ),
        )


def test_facade_model_artifact_delegates_to_vol_model_calculator():
    env = _grid_env()
    env.vol_surface = _grid_surface()
    engine = LocalVolPDESolver(params=PDEParams(grid_size=80, time_steps=40))
    request = BucketedGreeksRequest(
        coordinates=(BucketedGreekCoordinate.MODEL_ARTIFACT,),
        model_risk_request=ModelRiskRequest(
            parameter_names=(),
            surface_bumps=(SurfaceBump.parallel(0.01),),
        ),
    )

    result = GreeksCalculator().calculate_bucketed_greeks(
        _basic_call(), env, engine, request
    )
    direct = VolModelRiskCalculator().calculate_model_risk(
        _basic_call(), env, engine, request.model_risk_request
    )

    point = result.points[0]
    direct_point = direct.points[0]
    assert point.coordinate == BucketedGreekCoordinate.MODEL_ARTIFACT
    assert point.name == direct_point.name
    assert point.derivative == pytest.approx(direct_point.derivative)
    assert point.pnl == pytest.approx(direct_point.pnl)


def test_autocallable_report_bucketed_helper_uses_facade_compatible_units():
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.asset.equity.report.autocallable_risk_report import (
        _compute_bucketed_greeks,
    )
    from quantark.util.enum import ObservationType

    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=0.5,
        is_reverse=False,
    )
    env = _basic_env()
    df = _compute_bucketed_greeks(
        product, env, SnowballQuadEngine(), vol_bump=0.01, div_bump=0.0001
    )

    assert list(df.columns) == [
        "bucket",
        "bucket_vega",
        "bucket_rho_q",
        "bucket_rho_b",
    ]
    assert not df.empty
    assert np.allclose(df["bucket_rho_b"], -df["bucket_rho_q"])
