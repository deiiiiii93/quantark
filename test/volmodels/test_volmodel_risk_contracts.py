import numpy as np
import pytest

from quantark.param import GridVolSurface
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.risk import (
    ModelRiskRequest,
    SlvCalibrationSpec,
    SlvLeverageRiskMode,
    SurfaceBucketKind,
    SurfaceBump,
    VolRiskPoint,
    VolRiskResult,
    all_surface_bumps,
    bump_grid_vol_surface,
    bump_heston_parameter,
    bump_leverage_surface,
    bump_local_vol_surface,
    heston_parameter_bump_size,
)
from quantark.volmodels.slv import BinMethod, LeverageSurface


def _grid_surface():
    return GridVolSurface(
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.5, 1.0],
        iv_grid=np.array([[0.20, 0.21, 0.22], [0.23, 0.24, 0.25]]),
    )


def test_market_request_defaults_to_parallel_sticky_strike_bump():
    from quantark.volmodels.risk import MarketVegaRequest

    request = MarketVegaRequest()
    assert request.surface_bumps == (SurfaceBump.parallel(),)
    assert request.surface_bumps[0].bump_size == pytest.approx(0.01)


def test_all_surface_bumps_are_parallel_rows_columns_then_nodes():
    bumps = all_surface_bumps(_grid_surface())
    assert len(bumps) == 1 + 2 + 3 + 2 * 3
    assert bumps[0].kind == SurfaceBucketKind.PARALLEL
    assert [b.kind for b in bumps[1:3]] == [SurfaceBucketKind.MATURITY_ROW] * 2
    assert [b.kind for b in bumps[3:6]] == [SurfaceBucketKind.STRIKE_COLUMN] * 3
    assert all(b.kind == SurfaceBucketKind.NODE for b in bumps[6:])
    assert bumps[1].label == "maturity_row[0]"
    assert bumps[3].label == "strike_column[0]"


def test_grid_surface_node_bump_is_immutable_and_sticky_strike():
    surface = _grid_surface()
    bump = SurfaceBump.node(maturity_index=1, strike_index=2, bump_size=0.01)
    bumped = bump_grid_vol_surface(surface, bump, direction=1)
    assert surface.iv_grid[1, 2] == pytest.approx(0.25)
    assert bumped.iv_grid[1, 2] == pytest.approx(0.26)
    assert bumped.strikes == surface.strikes
    assert bumped.maturities == surface.maturities


def test_local_vol_bump_rejects_nonpositive_down_surface():
    surface = LocalVolSurface(
        strike_grid=np.array([90.0, 110.0]),
        time_grid=np.array([0.5]),
        lv_grid=np.array([[0.005, 0.02]]),
    )
    with pytest.raises(ValidationError):
        bump_local_vol_surface(surface, SurfaceBump.parallel(0.01), direction=-1)


def test_heston_parameter_bumps_use_named_scaling_and_do_not_clip():
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    up, h = bump_heston_parameter(params, "v0", direction=1)
    assert h == pytest.approx(4e-4)
    assert up.v0 == pytest.approx(0.0404)

    rho_up, rho_h = bump_heston_parameter(params, "rho", direction=1)
    assert rho_h == pytest.approx(1e-3)
    assert rho_up.rho == pytest.approx(-0.699)

    with pytest.raises(ValidationError):
        bump_heston_parameter(
            HestonParams(v0=0.0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7),
            "v0",
            direction=-1,
        )
    with pytest.raises(ValidationError, match="relative_bump"):
        heston_parameter_bump_size(params, "v0", relative_bump=-0.01)


def test_structured_result_records_prices_pnl_status_and_error():
    point = VolRiskPoint.success(
        name="heston.v0",
        bump_size=0.001,
        base_price=10.0,
        up_price=10.2,
        down_price=9.9,
    )
    assert point.derivative == pytest.approx(150.0)
    assert point.pnl == pytest.approx(0.15)
    assert point.status == "ok"
    assert point.error is None

    result = VolRiskResult(base_price=10.0, points=(point,))
    assert result.successful_points == (point,)
    assert result.failed_points == ()


def test_failed_result_records_partial_scenario_price():
    point = VolRiskPoint.failed(
        name="heston.v0",
        bump_size=0.001,
        base_price=10.0,
        error="down: invalid boundary",
        up_price=10.2,
    )
    assert point.status == "failed"
    assert point.up_price == pytest.approx(10.2)
    assert point.down_price is None
    assert point.derivative is None


def test_leverage_surface_bump_is_multiplicative_one_percent():
    surface = LeverageSurface(
        strike_grid=np.array([90.0, 110.0]),
        time_grid=np.array([0.5]),
        leverage_grid=np.array([[0.8, 1.2]]),
    )
    bumped = bump_leverage_surface(surface, SurfaceBump.parallel(), direction=1)
    assert np.allclose(bumped.leverage_grid, [[0.808, 1.212]])
    assert np.allclose(surface.leverage_grid, [[0.8, 1.2]])


def test_slv_model_risk_requires_explicit_leverage_mode():
    request = ModelRiskRequest()
    assert request.slv_leverage_mode is None
    assert SlvLeverageRiskMode.FROZEN.value == "frozen"


def test_model_risk_request_rejects_unknown_parameter_name():
    with pytest.raises(ValidationError, match="unknown model parameter"):
        ModelRiskRequest(parameter_names=("vol_of_vol",))


def test_risk_requests_freeze_sequences_and_validate_types():
    request = ModelRiskRequest(parameter_names=["v0"], surface_bumps=[SurfaceBump.parallel()])
    assert request.parameter_names == ("v0",)
    assert request.surface_bumps == (SurfaceBump.parallel(),)
    with pytest.raises(ValidationError, match="surface_bumps"):
        ModelRiskRequest(surface_bumps=["parallel"])


def test_slv_calibration_spec_accepts_existing_bin_enum_and_requires_seed():
    assert SlvCalibrationSpec(bin_method=BinMethod.EQUAL_WEIGHTED).bin_method == "equal_weighted"
    with pytest.raises(ValidationError, match="seed"):
        SlvCalibrationSpec(seed=None)
