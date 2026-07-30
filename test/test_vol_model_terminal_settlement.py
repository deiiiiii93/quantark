"""Terminal settlement contract across supported equity volatility models."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.equity.engine.capabilities import (
    SettlementSupport,
    VolDynamicsType,
    get_engine_capability,
    validate_engine_capability,
)
from quantark.asset.equity.engine.mc import (
    HestonMCEngine,
    HestonSLVMCEngine,
    LocalVolMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonPDESolver,
    HestonSLVPDESolver,
    LocalVolPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.execution.errors import CapabilityError
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, HestonMCScheme
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv import LeverageSurface


MATURITY = 0.25
LAG = 2.0 / 365.0
HESTON = HestonParams(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.6,
)


@pytest.fixture(scope="module")
def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 7, 30),
    )


@pytest.fixture(scope="module")
def local_vol():
    strikes = np.array([50.0, 100.0, 150.0])
    times = np.array([0.0, MATURITY])
    return LocalVolSurface(
        strike_grid=strikes,
        time_grid=times,
        lv_grid=np.full((times.size, strikes.size), 0.20),
    )


@pytest.fixture(scope="module")
def leverage():
    strikes = np.array([50.0, 100.0, 150.0])
    times = np.array([0.0, MATURITY])
    return LeverageSurface(
        time_grid=times,
        strike_grid=strikes,
        leverage_grid=np.ones((times.size, strikes.size)),
    )


def _option(*, convention=None):
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=MATURITY,
        settlement_convention=convention,
    )


def _engine(engine_name, local_vol, leverage):
    if engine_name == "heston_analytical":
        return HestonAnalyticalEngine(HESTON)
    if engine_name == "heston_mc":
        return HestonMCEngine(
            HESTON,
            scheme=HestonMCScheme.QUADEXP,
            params=MCParams(
                num_paths=4096,
                time_steps=8,
                seed=17,
                use_antithetic=True,
            ),
        )
    if engine_name == "local_vol_mc":
        return LocalVolMCEngine(
            params=MCParams(
                num_paths=4096,
                time_steps=8,
                seed=19,
                use_antithetic=True,
            ),
            local_vol_surface=local_vol,
        )
    if engine_name == "slv_mc":
        return HestonSLVMCEngine(
            HESTON,
            params=MCParams(
                num_paths=4096,
                time_steps=8,
                seed=23,
                use_antithetic=True,
            ),
            local_vol_surface=local_vol,
            leverage_surface=leverage,
        )
    if engine_name == "heston_pde":
        return HestonPDESolver(HESTON, n_x=48, n_v=18, n_t=16)
    if engine_name == "local_vol_pde":
        return LocalVolPDESolver(
            params=PDEParams(grid_size=101, time_steps=32),
            local_vol_surface=local_vol,
        )
    if engine_name == "slv_pde":
        return HestonSLVPDESolver(
            HESTON,
            leverage,
            n_x=48,
            n_v=18,
            n_t=16,
        )
    raise AssertionError(f"unknown engine fixture {engine_name!r}")


SUPPORTED_ENGINES = [
    "heston_analytical",
    "heston_mc",
    "local_vol_mc",
    "slv_mc",
    "heston_pde",
    "local_vol_pde",
    "slv_pde",
]


@pytest.mark.parametrize("engine_name", SUPPORTED_ENGINES)
def test_terminal_delay_scales_value_without_changing_dynamics(
    engine_name, env, local_vol, leverage
):
    engine = _engine(engine_name, local_vol, leverage)
    immediate = _option()
    delayed = _option(
        convention=SettlementConvention(
            lag=LAG,
            lag_unit=SettlementLagUnit.YEAR_FRACTION,
        )
    )

    immediate_pv = engine.price(immediate, env)
    delayed_pv = engine.price(delayed, env)
    expected_ratio = (
        env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )

    assert delayed_pv == pytest.approx(
        immediate_pv * expected_ratio,
        rel=2.0e-12,
        abs=2.0e-12,
    )


@pytest.mark.parametrize("engine_name", SUPPORTED_ENGINES)
def test_zero_lag_is_existing_price_identity(
    engine_name, env, local_vol, leverage
):
    engine = _engine(engine_name, local_vol, leverage)

    assert engine.price(
        _option(convention=SettlementConvention()),
        env,
    ) == pytest.approx(
        engine.price(_option(), env),
        rel=1.0e-14,
        abs=1.0e-14,
    )


def test_heston_mc_uses_payment_df_but_keeps_path_grid_at_determination(
    monkeypatch, env
):
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return 7.0

    import quantark.asset.equity.engine.mc.heston_mc_engine as module

    monkeypatch.setattr(module, "price_european_heston_mc", _spy)
    delayed = _option(
        convention=SettlementConvention(
            lag=LAG,
            lag_unit=SettlementLagUnit.YEAR_FRACTION,
        )
    )

    HestonMCEngine(
        HESTON,
        params=MCParams(num_paths=16, time_steps=8, seed=1),
    ).price(delayed, env)

    assert np.sum(captured["step_dt"]) == pytest.approx(MATURITY)
    assert len(captured["step_dt"]) == 8
    assert captured["disc_factor"] == pytest.approx(
        env.get_discount_factor(MATURITY + LAG)
    )


@pytest.mark.parametrize(
    ("dynamics", "engine_type"),
    [
        (VolDynamicsType.LOCAL_VOL, EngineType.MONTE_CARLO),
        (VolDynamicsType.LOCAL_VOL, EngineType.PDE),
        (VolDynamicsType.HESTON, EngineType.MONTE_CARLO),
        (VolDynamicsType.HESTON, EngineType.PDE),
        (VolDynamicsType.SLV, EngineType.MONTE_CARLO),
        (VolDynamicsType.SLV, EngineType.PDE),
    ],
)
def test_supported_vol_model_cells_declare_terminal_settlement(
    dynamics, engine_type
):
    assert (
        get_engine_capability(dynamics, engine_type).settlement_support
        is SettlementSupport.TERMINAL_ONLY
    )


@pytest.mark.parametrize(
    "dynamics",
    [
        VolDynamicsType.LOCAL_VOL,
        VolDynamicsType.HESTON,
        VolDynamicsType.SLV,
    ],
)
def test_unsupported_vol_model_quad_requests_raise_capability_error(
    dynamics,
):
    with pytest.raises(CapabilityError, match="QUAD is not supported"):
        validate_engine_capability(dynamics, EngineType.QUADRATURE)
