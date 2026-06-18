from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.equity.engine.mc import HestonSLVMCEngine, LocalVolMCEngine
from quantark.asset.equity.engine.pde import HestonPDESolver, HestonSLVPDESolver, LocalVolPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import VolModelRiskCalculator
from quantark.asset.fx.engine.analytical import FxHestonAnalyticalEngine
from quantark.asset.fx.engine.mc import FxLocalVolMCEngine
from quantark.asset.fx.engine.pde import FxLocalVolPDESolver
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.riskmeasures import FxVolModelRiskCalculator
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import FxPricingEnvironment, PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams, heston_implied_vol
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.risk import (
    HestonCalibrationSpec,
    MarketVegaRequest,
    ModelRiskRequest,
    SlvCalibrationSpec,
    SlvLeverageRiskMode,
    SurfaceBump,
)
from quantark.volmodels.slv import calibrate_leverage_surface
from quantark.util.enum.engine_enums import LeverageCalibrationMethod


P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)


def _grid(vol=0.2):
    strikes = list(100.0 * np.exp(np.linspace(-0.35, 0.35, 5)))
    maturities = [0.25, 0.75, 1.5]
    return GridVolSurface(strikes, maturities, np.full((3, 5), vol))


def _eq_env(surface=None):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.02),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=surface or _grid(),
        div_yield=ContinuousDividendYield(0.01),
    )


def _call():
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)


def test_equity_heston_model_risk_returns_parameter_vector_without_scalar_vega():
    result = VolModelRiskCalculator().calculate_model_risk(
        _call(),
        _eq_env(),
        HestonAnalyticalEngine(P),
        ModelRiskRequest(parameter_names=("v0", "rho"), surface_bumps=()),
    )
    assert [point.name for point in result.points] == ["heston.v0", "heston.rho"]
    assert all(point.status == "ok" and np.isfinite(point.derivative) for point in result.points)
    assert "vega" not in HestonAnalyticalEngine(P).calculate_greeks(_call(), _eq_env())


def test_heston_boundary_failure_is_reported_and_one_sided_is_opt_in():
    boundary = HestonParams(v0=0.0, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    calc = VolModelRiskCalculator()
    failed = calc.calculate_model_risk(
        _call(), _eq_env(), HestonAnalyticalEngine(boundary),
        ModelRiskRequest(parameter_names=("v0",), surface_bumps=()),
    )
    assert failed.points[0].status == "failed"

    one_sided = calc.calculate_model_risk(
        _call(), _eq_env(), HestonAnalyticalEngine(boundary),
        ModelRiskRequest(parameter_names=("v0",), surface_bumps=(), allow_one_sided=True),
    )
    assert one_sided.points[0].status == "ok"
    assert one_sided.points[0].difference_mode == "one_sided_up"


def test_dupire_market_vega_rebuilds_surface_and_reports_invalid_bucket():
    calc = VolModelRiskCalculator()
    engine = LocalVolPDESolver(params=PDEParams(grid_size=100, time_steps=50))
    ok = calc.calculate_market_vega(_call(), _eq_env(), engine)
    assert ok.points[0].name == "market_iv.parallel"
    assert ok.points[0].status == "ok"

    failed = calc.calculate_market_vega(
        _call(),
        _eq_env(_grid(vol=0.005)),
        engine,
        MarketVegaRequest(
            surface_bumps=(SurfaceBump.parallel(0.01), SurfaceBump.parallel(0.001))
        ),
    )
    assert failed.points[0].status == "failed"
    assert failed.points[1].status == "ok"
    assert failed.base_price > 0.0


def test_dupire_model_risk_bumps_resolved_local_vol_directly():
    result = VolModelRiskCalculator().calculate_model_risk(
        _call(),
        _eq_env(),
        LocalVolPDESolver(params=PDEParams(grid_size=100, time_steps=50)),
        ModelRiskRequest(parameter_names=(), surface_bumps=(SurfaceBump.parallel(),)),
    )
    assert result.points[0].name == "local_vol.parallel"
    assert result.points[0].status == "ok"


def test_local_vol_mc_market_vega_is_deterministic_with_common_seeds():
    engine = LocalVolMCEngine(
        MCParams(num_paths=20_000, time_steps=30, seed=19, use_antithetic=True)
    )
    calc = VolModelRiskCalculator()
    first = calc.calculate_market_vega(_call(), _eq_env(), engine)
    second = calc.calculate_market_vega(_call(), _eq_env(), engine)
    assert first == second


def _synthetic_heston_surface():
    strikes = [85.0, 100.0, 115.0]
    maturities = [0.5, 1.0, 1.5]
    iv = np.array([
        [heston_implied_vol(100.0, k, t, P, 0.02, 0.01) for k in strikes]
        for t in maturities
    ])
    return GridVolSurface(strikes, maturities, iv)


def test_heston_market_vega_calibrates_base_and_bumped_surface():
    calc = VolModelRiskCalculator(
        heston_calibration_spec=HestonCalibrationSpec(
            initial_params=P, regularize_feller=0.0, max_nfev=300,
        )
    )
    result = calc.calculate_market_vega(
        _call(), _eq_env(_synthetic_heston_surface()), HestonAnalyticalEngine(P),
    )
    assert result.points[0].name == "market_iv.parallel"
    assert result.points[0].status == "ok"
    assert np.isfinite(result.points[0].derivative)
    assert "base_heston_params" in result.metadata


def _slv_pde():
    env = _eq_env()
    lv = build_dupire_local_vol(env.vol_surface, env.spot, env.rate_curve, env.get_div_yield)
    steps = 30
    leverage = calibrate_leverage_surface(
        env.spot,
        P,
        lv,
        np.full(steps, 1.0 / steps),
        np.full(steps, 0.02),
        np.full(steps, 0.01),
        method=LeverageCalibrationMethod.MC_BINNING,
        num_paths=12_000,
        num_bins=15,
        seed=5,
    )
    return env, HestonSLVPDESolver(P, leverage, n_x=60, n_v=30, n_t=30)


def test_slv_parameter_risk_requires_and_supports_both_leverage_modes():
    env, engine = _slv_pde()
    calc = VolModelRiskCalculator(
        slv_calibration_spec=SlvCalibrationSpec(
            method=LeverageCalibrationMethod.MC_BINNING,
            num_paths=12_000, time_steps=30, num_bins=15, seed=5,
        )
    )
    with pytest.raises(ValidationError, match="explicit slv_leverage_mode"):
        calc.calculate_model_risk(
            _call(), env, engine,
            ModelRiskRequest(parameter_names=("rho",), surface_bumps=()),
        )

    frozen = calc.calculate_model_risk(
        _call(), env, engine,
        ModelRiskRequest(
            parameter_names=("rho",), surface_bumps=(),
            slv_leverage_mode=SlvLeverageRiskMode.FROZEN,
        ),
    )
    recalibrated = calc.calculate_model_risk(
        _call(), env, engine,
        ModelRiskRequest(
            parameter_names=("rho",), surface_bumps=(),
            slv_leverage_mode=SlvLeverageRiskMode.RECALIBRATE,
        ),
    )
    assert frozen.points[0].status == "ok"
    assert recalibrated.points[0].status == "ok"
    assert frozen.metadata["slv_leverage_mode"] == "frozen"
    assert recalibrated.metadata["slv_leverage_mode"] == "recalibrate"


def test_slv_exposes_direct_leverage_bucket_and_market_iv_recalibration():
    env, engine = _slv_pde()
    calc = VolModelRiskCalculator(
        slv_calibration_spec=SlvCalibrationSpec(
            method=LeverageCalibrationMethod.MC_BINNING,
            num_paths=6_000, time_steps=20, num_bins=12, seed=7,
        )
    )
    direct = calc.calculate_model_risk(
        _call(),
        env,
        engine,
        ModelRiskRequest(
            parameter_names=(),
            surface_bumps=(SurfaceBump.parallel(),),
            slv_leverage_mode=SlvLeverageRiskMode.FROZEN,
        ),
    )
    assert direct.points[0].name == "leverage.parallel"
    assert direct.points[0].status == "ok"

    market = calc.calculate_market_vega(_call(), env, engine)
    assert market.points[0].name == "market_iv.parallel"
    assert market.points[0].status == "ok"
    assert market.metadata["market_vega_convention"] == "recalibrated_leverage"


def test_frozen_slv_pde_artifact_risk_does_not_require_market_iv_surface():
    env, engine = _slv_pde()
    env.vol_surface = None
    result = VolModelRiskCalculator().calculate_model_risk(
        _call(),
        env,
        engine,
        ModelRiskRequest(
            parameter_names=("rho",),
            surface_bumps=(SurfaceBump.parallel(),),
            slv_leverage_mode=SlvLeverageRiskMode.FROZEN,
        ),
    )
    assert all(point.status == "ok" for point in result.points)


def test_frozen_slv_leverage_risk_is_consistent_across_mc_and_pde():
    env, pde = _slv_pde()
    local_vol = build_dupire_local_vol(
        env.vol_surface, env.spot, env.rate_curve, env.get_div_yield
    )
    mc = HestonSLVMCEngine(
        P,
        eta=1.0,
        params=MCParams(num_paths=100_000, time_steps=30, seed=5),
        local_vol_surface=local_vol,
        leverage_surface=pde.leverage_surface,
    )
    request = ModelRiskRequest(
        parameter_names=(),
        surface_bumps=(SurfaceBump.parallel(),),
        slv_leverage_mode=SlvLeverageRiskMode.FROZEN,
    )
    calc = VolModelRiskCalculator()
    mc_risk = calc.calculate_model_risk(_call(), env, mc, request).points[0].derivative
    pde_risk = calc.calculate_model_risk(_call(), env, pde, request).points[0].derivative
    assert np.sign(mc_risk) == np.sign(pde_risk)
    assert mc_risk == pytest.approx(pde_risk, rel=0.25)


def test_heston_model_parameter_risk_is_consistent_across_analytical_and_pde():
    request = ModelRiskRequest(parameter_names=("v0",), surface_bumps=())
    calc = VolModelRiskCalculator()
    analytical = calc.calculate_model_risk(
        _call(), _eq_env(), HestonAnalyticalEngine(P), request
    ).points[0].derivative
    pde = calc.calculate_model_risk(
        _call(), _eq_env(), HestonPDESolver(P, n_x=180, n_v=80, n_t=80), request
    ).points[0].derivative
    assert pde == pytest.approx(analytical, rel=0.08)


def test_fx_heston_model_risk_preserves_contract_sizing():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=1.2),
        domestic_curve=FlatRateCurve(0.02),
        foreign_curve=FlatRateCurve(0.01),
        vol_surface=GridVolSurface(
            [1.0, 1.2, 1.4],
            [0.5, 1.0, 1.5],
            np.full((3, 3), 0.12),
        ),
    )
    one = FxVanillaOption(
        strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1_000_000,
    )
    two = FxVanillaOption(
        strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=2_000_000,
    )
    request = ModelRiskRequest(parameter_names=("v0",), surface_bumps=())
    calc = FxVolModelRiskCalculator()
    first = calc.calculate_model_risk(one, env, FxHestonAnalyticalEngine(P), request)
    second = calc.calculate_model_risk(two, env, FxHestonAnalyticalEngine(P), request)
    assert second.points[0].derivative == pytest.approx(2.0 * first.points[0].derivative)


def test_fx_dupire_market_vega_preserves_contract_sizing():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=1.2),
        domestic_curve=FlatRateCurve(0.02),
        foreign_curve=FlatRateCurve(0.01),
        vol_surface=GridVolSurface(
            [1.0, 1.2, 1.4],
            [0.5, 1.0, 1.5],
            np.full((3, 3), 0.12),
        ),
    )
    one = FxVanillaOption(
        strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1_000_000,
    )
    two = FxVanillaOption(
        strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=2_000_000,
    )
    engine = FxLocalVolPDESolver(grid_size=120, time_steps=60)
    calc = FxVolModelRiskCalculator()
    first = calc.calculate_market_vega(one, env, engine)
    second = calc.calculate_market_vega(two, env, engine)
    assert first.points[0].status == "ok"
    assert second.points[0].derivative == pytest.approx(2.0 * first.points[0].derivative)


def test_mc_structured_risk_requires_a_fixed_seed():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=1.2),
        domestic_curve=FlatRateCurve(0.02),
        foreign_curve=FlatRateCurve(0.01),
        vol_surface=GridVolSurface(
            [1.0, 1.2, 1.4],
            [0.5, 1.0, 1.5],
            np.full((3, 3), 0.12),
        ),
    )
    product = FxVanillaOption(
        strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1_000_000,
    )
    with pytest.raises(ValidationError, match="fixed MC seed"):
        FxVolModelRiskCalculator().calculate_market_vega(
            product, env, FxLocalVolMCEngine(num_paths=1_000, time_steps=10, seed=None),
        )
