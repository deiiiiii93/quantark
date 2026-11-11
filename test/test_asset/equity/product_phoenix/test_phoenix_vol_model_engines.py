from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import StatisticsRefused

from quantark.asset.equity.engine.mc import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
    QEPhoenixMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonPhoenixPDESolver,
    HestonSLVPhoenixPDESolver,
    LocalVolPhoenixPDESolver,
    PhoenixPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType, OptionType
from quantark.util.enum.engine_enums import HestonMCScheme, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.cashleg.event_distribution import EventType


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


def _phoenix(memory_coupon: bool = False):
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
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0,
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=memory_coupon,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def _heston():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _unit_leverage(s0=100.0, maturity=1.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, maturity, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def test_local_vol_phoenix_pde_matches_flat_bsm_pde():
    product = _phoenix(memory_coupon=False)
    env = _env()
    params = PDEParams()

    bsm = PhoenixPDESolver(params).price(product, env)
    lv = LocalVolPhoenixPDESolver(params).price(product, env)

    assert lv == pytest.approx(bsm, rel=0.02)


def test_phoenix_vol_model_mc_engines_run():
    product = _phoenix(memory_coupon=True)
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=2_000, time_steps=40, seed=11)

    engines = [
        LocalVolPhoenixMCEngine(params),
        HestonPhoenixMCEngine(hp, params),
        HestonSLVPhoenixMCEngine(hp, params=params, leverage_surface=_unit_leverage()),
        HestonSLVQEPhoenixMCEngine(
            hp,
            params=params,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        price = engine.price(product, env)
        assert np.isfinite(price)
        assert 0.0 < price < product.initial_price * 1.2


@pytest.mark.parametrize(
    "scheme",
    [HestonMCScheme.EULERLOG, HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M],
)
def test_phoenix_heston_mc_qe_schemes_run(scheme):
    product = _phoenix(memory_coupon=False)
    env = _env()
    params = MCParams(num_paths=1_024, time_steps=24, seed=19)

    engine = HestonPhoenixMCEngine(_heston(), params, scheme=scheme)
    price = engine.price(product, env)

    assert np.isfinite(price)
    assert 0.0 < price < product.initial_price * 1.2


def test_standalone_qe_phoenix_mc_matches_heston_qe():
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=31)

    standalone = QEPhoenixMCEngine(hp, params).price(product, env)
    explicit = HestonPhoenixMCEngine(
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
def test_standalone_slv_qe_phoenix_mc_reduces_to_heston_qe(
    martingale_correction,
    scheme,
):
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=37)

    slv_qe = HestonSLVQEPhoenixMCEngine(
        hp,
        params=params,
        leverage_surface=_unit_leverage(),
        martingale_correction=martingale_correction,
    ).price(product, env)
    heston_qe = HestonPhoenixMCEngine(hp, params, scheme=scheme).price(product, env)

    assert slv_qe == pytest.approx(heston_qe, rel=1e-13, abs=1e-8)


def test_phoenix_vol_model_mc_qmc_engines_run():
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=1_024, time_steps=24, seed=23)

    engines = [
        LocalVolPhoenixMCEngine(params, method=MonteCarloMethod.QUASI),
        HestonPhoenixMCEngine(
            hp,
            params,
            method=MonteCarloMethod.QUASI,
            scheme=HestonMCScheme.QUADEXP,
        ),
        HestonSLVPhoenixMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.QUASI,
            leverage_surface=_unit_leverage(),
        ),
        HestonSLVQEPhoenixMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.QUASI,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        price = engine.price(product, env)
        assert np.isfinite(price)
        assert 0.0 < price < product.initial_price * 1.2


def test_phoenix_vol_model_mc_rqmc_engines_run():
    product = _phoenix(memory_coupon=False)
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
        LocalVolPhoenixMCEngine(params, method=MonteCarloMethod.RANDOMIZED_QUASI),
        HestonPhoenixMCEngine(
            hp,
            params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            scheme=HestonMCScheme.QUADEXP,
        ),
        HestonSLVPhoenixMCEngine(
            hp,
            params=params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            leverage_surface=_unit_leverage(),
        ),
        HestonSLVQEPhoenixMCEngine(
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
        assert 0.0 < price < product.initial_price * 1.2


def _assert_phoenix_event_statistics(stats, product: PhoenixOption):
    """The MC engines report the unified ``EventStatistics`` (T10): a coupon table on the observation grid, every
    outcome direct, both residuals at round-off."""
    from quantark.asset.equity.engine.event_stats import EventStatistics

    assert isinstance(stats, EventStatistics) and stats.product_family == "phoenix"
    n = len(product.barrier_config.ko_observation_dates)
    assert stats.observations.size == stats.coupons.size == n
    assert np.all(stats.observations.probability >= 0.0) and np.all(stats.coupons.probability >= 0.0)
    assert stats.knock_in.probability is not None
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    assert abs(stats.probability_residual) <= 1e-12
    assert abs(stats.reconciliation_error) <= 1e-12 * max(1.0, abs(stats.pv))


def _assert_phoenix_pde_event_statistics(stats, product: PhoenixOption):
    """The PDE 1D march (local vol, and the 2D solvers' flat-vol surrogate) reports ``EventStatistics`` since
    unified event statistics T12: coupon table on the observation grid, KI settlement marched, P(KI) pending the
    forward pass (T13); the ledger reconciles to the reported pv up to ``reconciliation_error``."""
    from quantark.asset.equity.engine.event_stats import EventStatistics

    assert isinstance(stats, EventStatistics) and stats.product_family == "phoenix"
    n = len(product.barrier_config.ko_observation_dates)
    assert stats.observations.size == stats.coupons.size == n
    assert np.isfinite(stats.pv)
    assert np.all(np.isfinite(stats.observations.probability)) and np.all(np.isfinite(stats.coupons.probability))
    assert np.all(stats.observations.probability >= -1e-10) and np.all(stats.coupons.probability >= -1e-10)
    assert stats.knock_in.probability is None and "T13" in stats.unavailable["knock_in.probability"]
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    parts = float(np.sum(stats.ledger.discounted_cash))
    assert float(stats.pv) - parts == pytest.approx(float(stats.reconciliation_error), abs=1e-7)


def test_phoenix_vol_model_mc_engines_calculate_event_stats():
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=768, time_steps=24, seed=41)

    engines = [
        LocalVolPhoenixMCEngine(params),
        HestonPhoenixMCEngine(hp, params),
        QEPhoenixMCEngine(hp, params),
        HestonSLVPhoenixMCEngine(hp, params=params, leverage_surface=_unit_leverage()),
        HestonSLVQEPhoenixMCEngine(
            hp,
            params=params,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        _assert_phoenix_event_statistics(engine.calculate_event_stats(product, env), product)


def test_phoenix_vol_model_pde_engines_calculate_event_stats():
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()
    params = PDEParams()

    engines = [
        LocalVolPhoenixPDESolver(params),
        HestonPhoenixPDESolver(hp, n_x=48, n_v=18, n_t=16, grid_style="uniform", statistics="flat_vol_surrogate"),
        HestonSLVPhoenixPDESolver(
            hp,
            _unit_leverage(),
            n_x=48,
            n_v=18,
            n_t=16,
            grid_style="uniform",
            statistics="flat_vol_surrogate",
        ),
    ]
    # the 2D solvers report statistics only on request (spec 3B decision 8)
    with pytest.raises(StatisticsRefused):
        HestonPhoenixPDESolver(hp, n_x=48, n_v=18, n_t=16, grid_style="uniform", statistics="refuse").calculate_event_stats(product, env)

    for engine in engines:
        price = float(engine.price(product, env))
        stats = engine.calculate_event_stats(product, env)
        _assert_phoenix_pde_event_statistics(stats, product)
        assert float(stats.pv) == pytest.approx(price, rel=0.0, abs=1e-8)

        result = engine.price_with_events(product, env)
        assert float(result.npv) == pytest.approx(price, rel=0.0, abs=1e-8)
        assert result.event_distribution.stream_times[EventType.KO].shape == stats.observations.times.shape


def test_phoenix_heston_and_slv_pde_engines_run():
    product = _phoenix(memory_coupon=False)
    env = _env()
    hp = _heston()

    heston = HestonPhoenixPDESolver(
        hp, n_x=60, n_v=24, n_t=24, grid_style="uniform"
    ).price(product, env)
    slv = HestonSLVPhoenixPDESolver(
        hp, _unit_leverage(), n_x=60, n_v=24, n_t=24, grid_style="uniform"
    ).price(product, env)

    assert np.isfinite(heston)
    assert np.isfinite(slv)
    assert 0.0 < heston < product.initial_price * 1.2
    assert slv == pytest.approx(heston, rel=0.02)


def test_phoenix_heston_pde_auto_grid_focuses_coupon_without_ki():
    product = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=120.0,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.5, 1.0],
            ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=92.0,
            coupon_rate=0.02,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=False),
    )
    solver = HestonPhoenixPDESolver(_heston(), n_x=80, n_v=24, n_t=24)

    assert solver._grid_concentration_spot(product, _env()) == pytest.approx(92.0)


def test_phoenix_heston_pde_prices_memory_coupons():
    engine = HestonPhoenixPDESolver(_heston(), n_x=48, n_v=18, n_t=24)
    product = _phoenix(memory_coupon=True)
    price = engine.price(product, _env())
    stats = engine.calculate_event_stats(product, _env())
    assert np.isfinite(price)
    assert stats.pv == pytest.approx(price)
    assert not stats.provenance.surrogate


def test_phoenix_vol_model_engines_reject_non_phoenix_products():
    vanilla = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = _env()
    hp = _heston()

    with pytest.raises(PricingError):
        LocalVolPhoenixMCEngine(MCParams(num_paths=100)).price(vanilla, env)
    with pytest.raises(PricingError):
        LocalVolPhoenixPDESolver(PDEParams()).price(
            vanilla, env
        )
    with pytest.raises(PricingError):
        HestonPhoenixPDESolver(hp, n_x=30, n_v=12, n_t=10).price(vanilla, env)
