import numpy as np
import pytest
from datetime import datetime

from quantark.asset.equity.engine.mc import (
    HestonSLVSnowballMCEngine,
    HestonSLVQESnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType
from quantark.util.enum.engine_enums import HestonMCScheme, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface


def _env(vol=0.20, s0=100.0, r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    surface = GridVolSurface(
        strikes,
        maturities,
        np.full((len(maturities), len(strikes)), vol),
    )
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=surface,
        div_yield=ContinuousDividendYield(q),
    )


def _snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
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


def _principal_excluded_snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=2.0,
        contract_multiplier=1.0,
        is_reverse=False,
        payoff_config=PayoffConfig(include_principal=False),
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[i / 12.0 for i in range(1, 25)],
            ki_barrier=80.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _heston():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _unit_leverage(s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def test_local_vol_snowball_pde_matches_flat_bsm_pde():
    product = _snowball()
    env = _env()
    params = PDEParams(grid_size=90, time_steps=48, auto_grid=False)

    bsm = SnowballPDESolver(params).price(product, env)
    lv = LocalVolSnowballPDESolver(params).price(product, env)

    assert lv == pytest.approx(bsm, rel=0.015)


def test_snowball_vol_model_mc_engines_run():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=2_000, time_steps=40, seed=11)

    engines = [
        LocalVolSnowballMCEngine(params),
        HestonSnowballMCEngine(hp, params),
        HestonSLVSnowballMCEngine(hp, params=params, leverage_surface=_unit_leverage()),
        HestonSLVQESnowballMCEngine(
            hp,
            params=params,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        price = engine.price(product, env)
        assert np.isfinite(price)
        assert 0.0 < price < product.initial_price * product.contract_multiplier * 1.2


@pytest.mark.parametrize(
    "scheme",
    [HestonMCScheme.EULERLOG, HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M],
)
def test_snowball_heston_mc_qe_schemes_run(scheme):
    product = _snowball()
    env = _env()
    params = MCParams(num_paths=1_024, time_steps=24, seed=19)

    engine = HestonSnowballMCEngine(_heston(), params, scheme=scheme)
    price = engine.price(product, env)

    assert np.isfinite(price)
    assert 0.0 < price < product.initial_price * product.contract_multiplier * 1.2


def test_standalone_qe_snowball_mc_matches_heston_qe():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=31)

    standalone = QESnowballMCEngine(hp, params).price(product, env)
    explicit = HestonSnowballMCEngine(
        hp,
        params,
        scheme=HestonMCScheme.QUADEXP,
    ).price(product, env)

    assert standalone == pytest.approx(explicit, rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    ("martingale_correction", "scheme"),
    [
        (False, HestonMCScheme.QUADEXP),
        (True, HestonMCScheme.QUADEXP_M),
    ],
)
def test_standalone_slv_qe_snowball_mc_reduces_to_heston_qe(
    martingale_correction,
    scheme,
):
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=37)

    slv_qe = HestonSLVQESnowballMCEngine(
        hp,
        params=params,
        leverage_surface=_unit_leverage(),
        martingale_correction=martingale_correction,
    ).price(product, env)
    heston_qe = HestonSnowballMCEngine(hp, params, scheme=scheme).price(product, env)

    assert slv_qe == pytest.approx(heston_qe, rel=1e-13, abs=1e-8)


def test_snowball_vol_model_mc_qmc_engines_run():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=23)

    engines = [
        LocalVolSnowballMCEngine(params, method=MonteCarloMethod.QUASI),
        HestonSnowballMCEngine(
            hp,
            params,
            method=MonteCarloMethod.QUASI,
            scheme=HestonMCScheme.QUADEXP,
        ),
        HestonSLVSnowballMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.QUASI,
            leverage_surface=_unit_leverage(),
        ),
        HestonSLVQESnowballMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.QUASI,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        price = engine.price(product, env)
        assert np.isfinite(price)
        assert 0.0 < price < product.initial_price * product.contract_multiplier * 1.2


def test_snowball_vol_model_mc_rqmc_engines_run():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(
        num_paths=256,
        time_steps=16,
        seed=29,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
    )

    engines = [
        LocalVolSnowballMCEngine(params, method=MonteCarloMethod.RANDOMIZED_QUASI),
        HestonSnowballMCEngine(
            hp,
            params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            scheme=HestonMCScheme.QUADEXP,
        ),
        HestonSLVSnowballMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            leverage_surface=_unit_leverage(),
        ),
        HestonSLVQESnowballMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        price = engine.price(product, env)
        result = engine.get_last_result()
        assert np.isfinite(price)
        assert result is not None
        assert result.batches_used == 2
        assert result.num_paths == 2 * params.num_paths
        assert 0.0 < price < product.initial_price * product.contract_multiplier * 1.2


def test_snowball_heston_and_slv_pde_engines_run():
    product = _snowball()
    env = _env()
    hp = _heston()

    heston = HestonSnowballPDESolver(hp, n_x=60, n_v=24, n_t=24).price(product, env)
    slv = HestonSLVSnowballPDESolver(
        hp, _unit_leverage(), n_x=60, n_v=24, n_t=24
    ).price(product, env)

    assert np.isfinite(heston)
    assert np.isfinite(slv)
    assert 0.0 < heston < product.initial_price * product.contract_multiplier * 1.2
    assert slv == pytest.approx(heston, rel=0.02)


def test_snowball_heston_pde_auto_grid_focuses_ki():
    product = _principal_excluded_snowball()
    env = _env()
    solver = HestonSnowballPDESolver(_heston(), n_x=80, n_v=24, n_t=24)

    assert solver._grid_concentration_spot(product, env) == pytest.approx(80.0)

    core = solver._make_core(product, env, product.get_maturity(env))
    assert float(np.min(np.abs(core.S_grid - 80.0))) == pytest.approx(0.0, abs=1e-10)


def test_snowball_heston_pde_can_pin_critical_spots_for_diagnostics():
    product = _principal_excluded_snowball()
    env = _env()
    solver = HestonSnowballPDESolver(
        _heston(), n_x=80, n_v=24, n_t=24, pin_critical_spots=True
    )

    core = solver._make_core(product, env, product.get_maturity(env))
    for level in [80.0, 100.0, 103.0]:
        assert float(np.min(np.abs(core.S_grid - level))) == pytest.approx(0.0, abs=1e-10)


def test_snowball_heston_pde_grid_focus_override_keeps_legacy_ko_focus():
    product = _principal_excluded_snowball()
    env = _env()
    solver = HestonSnowballPDESolver(
        _heston(), n_x=80, n_v=24, n_t=24, grid_focus="ko"
    )

    assert solver._grid_concentration_spot(product, env) == pytest.approx(103.0)


def test_snowball_heston_pde_rejects_unknown_grid_focus():
    with pytest.raises(ValidationError):
        HestonSnowballPDESolver(_heston(), grid_focus="coupon")


def test_snowball_vol_model_engines_reject_non_snowball_products():
    vanilla = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = _env()
    hp = _heston()

    with pytest.raises(PricingError):
        LocalVolSnowballMCEngine(MCParams(num_paths=100)).price(vanilla, env)
    with pytest.raises(PricingError):
        LocalVolSnowballPDESolver(PDEParams(grid_size=50, time_steps=20)).price(vanilla, env)
    with pytest.raises(PricingError):
        HestonSnowballPDESolver(hp, n_x=30, n_v=12, n_t=10).price(vanilla, env)
