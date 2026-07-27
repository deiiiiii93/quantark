from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.capabilities import (
    VolDynamicsType,
    validate_engine_capability,
)
from quantark.asset.equity.engine.pde import (
    HestonBarrierPDESolver,
    HestonPhoenixPDESolver,
    HestonSnowballPDESolver,
    HestonSLVBarrierPDESolver,
    HestonSLVPhoenixPDESolver,
    HestonSLVSnowballPDESolver,
    LocalVolBarrierPDESolver,
    LocalVolPhoenixPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    GridVolSurface,
    SpotQuote,
)
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface


def _grid_vol_surface(spot=100.0, vol=0.20):
    strikes = list(float(spot) * np.exp(np.linspace(-0.6, 0.6, 9)))
    maturities = [0.25, 0.5, 1.0, 2.0]
    return GridVolSurface(
        strikes,
        maturities,
        np.full((len(maturities), len(strikes)), float(vol)),
    )


def _term_env(spot=100.0):
    return PricingEnvironment(
        rate_curve=LinearRateCurve(
            [(0.25, 0.020), (0.5, 0.035), (1.0, 0.025), (2.0, 0.030)]
        ),
        valuation_date=datetime(2026, 7, 7),
        spot_quote=SpotQuote(float(spot)),
        vol_surface=_grid_vol_surface(spot),
        div_yield=TermStructureDividendYield(
            times=[0.25, 0.5, 1.0, 2.0],
            yields=[-0.015, 0.020, -0.005, 0.010],
        ),
    )


def _collapsed_flat_env(env_term, maturity, ref_strike=100.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(float(maturity))),
        valuation_date=env_term.valuation_date,
        spot_quote=SpotQuote(float(env_term.spot)),
        vol_surface=_grid_vol_surface(env_term.spot, env_term.get_vol(ref_strike, maturity)),
        div_yield=ContinuousDividendYield(env_term.get_div_yield(float(maturity))),
    )


def _up_out_call():
    return BarrierOption(
        strike=100.0,
        maturity=1.0,
        option_type=OptionType.CALL,
        barrier=125.0,
        barrier_type=BarrierType.UP_OUT,
        observation_type=ObservationType.CONTINUOUS,
        rebate=0.0,
    )


def _snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _phoenix():
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0,
            coupon_rate=0.02,
            memory_coupon=False,
        ),
    )


def _heston_params():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _unit_leverage(s0=100.0, maturity=1.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, maturity, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def test_local_vol_barrier_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _up_out_call()
    solver = LocalVolBarrierPDESolver(PDEParams())

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_local_vol_snowball_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _snowball()
    solver = LocalVolSnowballPDESolver(
        PDEParams()
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_local_vol_phoenix_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _phoenix()
    solver = LocalVolPhoenixPDESolver(
        PDEParams()
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_heston_barrier_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _up_out_call()
    solver = HestonBarrierPDESolver(_heston_params(), n_x=70, n_v=24, n_t=32)

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_slv_barrier_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _up_out_call()
    solver = HestonSLVBarrierPDESolver(
        _heston_params(),
        _unit_leverage(),
        n_x=70,
        n_v=24,
        n_t=32,
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_heston_snowball_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _snowball()
    solver = HestonSnowballPDESolver(
        _heston_params(),
        n_x=60,
        n_v=20,
        n_t=24,
        grid_style="uniform",
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_heston_phoenix_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _phoenix()
    solver = HestonPhoenixPDESolver(
        _heston_params(),
        n_x=60,
        n_v=20,
        n_t=24,
        grid_style="uniform",
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_slv_snowball_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _snowball()
    solver = HestonSLVSnowballPDESolver(
        _heston_params(),
        _unit_leverage(),
        n_x=60,
        n_v=20,
        n_t=24,
        grid_style="uniform",
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_slv_phoenix_pde_sees_non_flat_rate_and_carry():
    env_term = _term_env()
    product = _phoenix()
    solver = HestonSLVPhoenixPDESolver(
        _heston_params(),
        _unit_leverage(),
        n_x=60,
        n_v=20,
        n_t=24,
        grid_style="uniform",
    )

    px_term = solver.price(product, env_term)
    px_collapsed = solver.price(product, _collapsed_flat_env(env_term, 1.0))

    assert np.isfinite(px_term)
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


@pytest.mark.parametrize(
    "dynamics",
    [VolDynamicsType.LOCAL_VOL, VolDynamicsType.HESTON, VolDynamicsType.SLV],
)
def test_vol_model_quad_requests_are_rejected(dynamics):
    with pytest.raises(ValidationError, match="QUAD is not supported"):
        validate_engine_capability(dynamics, EngineType.QUADRATURE)
