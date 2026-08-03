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
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
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


def _discrete_snowball():
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
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ),
    )


def _heston():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _sigma_collapse_heston():
    return HestonParams(
        v0=0.14027,
        kappa=3.0,
        theta=0.00306,
        sigma=0.00311,
        rho=-0.5,
    )


def _unit_leverage(s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def _constant_leverage(value, s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.full((4, strikes.size), float(value)),
    )


def _smile_leverage(s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    smile = np.linspace(1.15, 0.90, strikes.size)
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.vstack(
            [smile, 0.75 * smile + 0.25, 0.5 * smile + 0.5, np.ones_like(smile)]
        ),
    )


def test_local_vol_snowball_pde_matches_flat_bsm_pde():
    product = _snowball()
    env = _env()
    params = PDEParams()

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


def test_heston_qe_rqmc_integrates_affine_spot_factor_inside_each_sobol_point():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=128,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )
    engine = QESnowballMCEngine(
        _heston(),
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_affine_spot_factor=True,
    )
    spec = engine.build_rqmc_session_spec(product, env)

    assert spec is not None
    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    assert aux is not None
    assert paths.shape[0] == params.num_paths
    assert paths.shape[1] == spec.time_steps + 1
    loadings = aux["log_spot_factor_loadings"]
    assert aux["affine_spot_factor"] == "standard_normal"
    assert loadings.shape == paths.shape
    assert np.all(loadings[:, 0] == 0.0)
    assert np.all(loadings[:, 1:] > 0.0)
    assert np.all(np.diff(loadings, axis=1) >= 0.0)
    payoffs = spec.pricer_fn(paths, aux)
    assert payoffs.shape == (params.num_paths,)
    assert np.all(np.isfinite(payoffs))
    assert "#affine-spot-factor" in spec.scheme
    assert np.isfinite(engine.price(product, env))


def test_slv_qe_rqmc_stratifies_spot_factor_and_collapses_to_outer_paths():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=64,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )
    engine = HestonSLVQESnowballMCEngine(
        _heston(),
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_unit_leverage(),
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_heston_conditional_control=True,
        rqmc_spot_strata=4,
    )
    spec = engine.build_rqmc_session_spec(product, env)

    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    assert paths.shape == (4 * params.num_paths, spec.time_steps + 1)
    assert aux["control_paths"].shape == paths.shape
    assert aux["control_base_paths"].shape == (
        params.num_paths,
        spec.time_steps + 1,
    )
    assert aux["conditional_group_size"] == 4
    assert spec.path_valuation_multiplier == 4
    payoffs = spec.pricer_fn(paths, aux)
    control_payoffs = spec.control_pricer_fn(paths, aux)
    assert payoffs.shape == (params.num_paths,)
    assert np.all(np.isfinite(payoffs))
    # Unit leverage makes the sampled SLV and Heston control paths identical;
    # the controlled estimator therefore equals exact Heston conditioning.
    assert np.max(np.abs(paths - aux["control_paths"])) < 2e-10
    np.testing.assert_allclose(payoffs, control_payoffs, rtol=0.0, atol=1e-9)
    assert np.isfinite(engine.price(product, env))


def test_slv_qe_rqmc_antithetic_strata_are_unbiased_grouped_pairs():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=32,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )
    engine = HestonSLVQESnowballMCEngine(
        _heston(),
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_unit_leverage(),
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_heston_conditional_control=True,
        rqmc_spot_strata=2,
        rqmc_spot_antithetic=True,
    )
    spec = engine.build_rqmc_session_spec(product, env)

    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    assert paths.shape == (4 * params.num_paths, spec.time_steps + 1)
    assert aux["conditional_group_size"] == 4
    assert "#spot-strata-2#spot-antithetic" in spec.scheme
    payoffs = spec.pricer_fn(paths, aux)
    control_payoffs = spec.control_pricer_fn(paths, aux)
    assert payoffs.shape == (params.num_paths,)
    np.testing.assert_allclose(payoffs, control_payoffs, rtol=0.0, atol=1e-9)


def test_slv_qe_rqmc_stratifies_second_spot_bridge_factor():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=16,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )
    engine = HestonSLVQESnowballMCEngine(
        _heston(),
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_unit_leverage(),
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_heston_conditional_control=True,
        rqmc_spot_strata=2,
        rqmc_spot_bridge_strata=2,
    )
    spec = engine.build_rqmc_session_spec(product, env)

    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    assert paths.shape == (4 * params.num_paths, spec.time_steps + 1)
    assert aux["control_base_paths"].shape == (
        2 * params.num_paths,
        spec.time_steps + 1,
    )
    assert aux["conditional_group_size"] == 2
    assert aux["conditional_outer_group_size"] == 2
    assert spec.path_valuation_multiplier == 4
    assert "#spot-bridge-strata-2" in spec.scheme
    payoffs = spec.pricer_fn(paths, aux)
    control_payoffs = spec.control_pricer_fn(paths, aux)
    assert payoffs.shape == (params.num_paths,)
    assert control_payoffs.shape == (params.num_paths,)
    np.testing.assert_allclose(payoffs, control_payoffs, rtol=0.0, atol=1e-9)


def test_slv_frozen_leverage_proxy_is_exact_for_a_constant_surface():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=32,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )
    engine = HestonSLVQESnowballMCEngine(
        _heston(),
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_constant_leverage(1.1),
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_heston_conditional_control=True,
        rqmc_frozen_leverage_conditional_control=True,
        rqmc_spot_strata=4,
    )
    spec = engine.build_rqmc_session_spec(product, env)

    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    assert np.max(np.abs(paths - aux["control_paths"])) < 2e-10
    assert "#frozen-leverage-proxy" in spec.scheme
    payoffs = spec.pricer_fn(paths, aux)
    assert payoffs.shape == (params.num_paths,)
    assert np.all(np.isfinite(payoffs))


def test_slv_single_stratum_control_preserves_the_target_paths():
    product = _discrete_snowball()
    env = _env()
    params = MCParams(
        num_paths=32,
        seed=2903,
        rqmc_min_batches=2,
        rqmc_max_batches=2,
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )

    class _BridgeOrderedProvider:
        dimension = 24
        label = "test-bridge-ordered"
        randomization_key = ("test-bridge-ordered", 2903)

        def draws(self, *, n_paths, dt_array, batch_id):
            rng = np.random.default_rng(2903 + int(batch_id or 0))
            shape = (int(n_paths), len(dt_array))
            return (
                rng.standard_normal(shape),
                rng.standard_normal(shape),
                rng.random(shape),
            )

    provider = _BridgeOrderedProvider()
    common = dict(
        params=params,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_smile_leverage(),
        martingale_correction=True,
        substeps_per_interval=2,
        rqmc_qe_draw_provider=provider,
    )
    baseline = HestonSLVQESnowballMCEngine(_heston(), **common)
    controlled = HestonSLVQESnowballMCEngine(
        _heston(),
        rqmc_heston_conditional_control=True,
        rqmc_frozen_leverage_conditional_control=True,
        rqmc_spot_strata=1,
        **common,
    )

    baseline_spec = baseline.build_rqmc_session_spec(product, env)
    controlled_spec = controlled.build_rqmc_session_spec(product, env)
    baseline_paths, _ = baseline_spec.path_generator.generate_paths(
        batch_id=0, return_aux=True
    )
    target_paths, _ = controlled_spec.path_generator.generate_paths(
        batch_id=0, return_aux=True
    )

    assert np.max(np.abs(baseline_paths - target_paths)) < 2e-10


def _assert_snowball_event_stats(stats: AutocallableEventStats, product: SnowballOption):
    assert isinstance(stats, AutocallableEventStats)
    assert stats.ko_times.shape == stats.ko_probability.shape
    assert stats.ko_times.shape == stats.survival_probability.shape
    assert stats.ko_times.shape == stats.expected_discounted_ko_cashflow.shape
    assert stats.ko_times.size == len(product.barrier_config.ko_observation_dates)
    assert np.isfinite(stats.pv)
    assert np.all(np.isfinite(stats.ko_probability))
    assert np.all(stats.ko_probability >= -1e-10)
    assert np.all(stats.survival_probability <= 1.0 + 1e-10)
    assert stats.ki_ever_probability is not None
    assert stats.ki_survive_knocked_in_probability is not None

    parts = (
        float(np.sum(stats.expected_discounted_ko_cashflow))
        + float(stats.expected_discounted_maturity_cashflow)
    )
    assert float(stats.pv) - parts == pytest.approx(
        float(stats.reconciliation_error), abs=1e-7
    )


def test_snowball_vol_model_mc_engines_calculate_event_stats():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = MCParams(num_paths=768, time_steps=24, seed=41)

    engines = [
        LocalVolSnowballMCEngine(params),
        HestonSnowballMCEngine(hp, params),
        QESnowballMCEngine(hp, params),
        HestonSLVSnowballMCEngine(hp, params=params, leverage_surface=_unit_leverage()),
        HestonSLVQESnowballMCEngine(
            hp,
            params=params,
            leverage_surface=_unit_leverage(),
        ),
    ]

    for engine in engines:
        _assert_snowball_event_stats(engine.calculate_event_stats(product, env), product)


def test_snowball_vol_model_pde_engines_calculate_event_stats():
    product = _snowball()
    env = _env()
    hp = _heston()
    params = PDEParams()

    engines = [
        LocalVolSnowballPDESolver(params),
        HestonSnowballPDESolver(hp, n_x=48, n_v=18, n_t=16),
        HestonSLVSnowballPDESolver(hp, _unit_leverage(), n_x=48, n_v=18, n_t=16),
    ]

    for engine in engines:
        price = float(engine.price(product, env))
        stats = engine.calculate_event_stats(product, env)
        _assert_snowball_event_stats(stats, product)
        assert float(stats.pv) == pytest.approx(price, rel=0.0, abs=1e-8)

        result = engine.price_with_events(product, env)
        assert float(result.npv) == pytest.approx(price, rel=0.0, abs=1e-8)
        assert result.event_distribution.event_times.shape == stats.ko_times.shape


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


def test_structured_model_pde_bump_contexts_clone_full_state():
    """``create_bump_context`` must clone LV/Heston/SLV solvers with their
    full constructor state (model params, leverage surface, ADI dimensions).

    Regression: the base hook used ``type(self)(params=...)``, which raised
    ``TypeError`` for every solver whose constructor requires more than
    ``params`` — killing numerical greeks for structured-model engines.
    """
    product = _snowball()
    env = _env()
    hp = _heston()

    engines = [
        LocalVolSnowballPDESolver(),
        HestonSnowballPDESolver(hp, n_x=48, n_v=18, n_t=16),
        HestonSLVSnowballPDESolver(hp, _unit_leverage(), n_x=48, n_v=18, n_t=16),
    ]
    for engine in engines:
        base = float(engine.price(product, env))
        ctx = engine.create_bump_context(product, env)
        assert ctx is not engine
        if hasattr(engine, "model_params"):
            assert ctx.model_params == engine.model_params
            assert (ctx.n_x, ctx.n_v, ctx.n_t) == (
                engine.n_x,
                engine.n_v,
                engine.n_t,
            )
        # the GreeksCalculator path: bumped repricing through the clone
        assert np.isfinite(float(ctx.price(product, _env(s0=101.0))))
        # base-market repricing through the clone reproduces the base price
        assert float(ctx.price(product, env)) == pytest.approx(base, rel=1e-12)


def test_heston_2d_bump_context_freezes_s_axis():
    """The 2D bump context must freeze the ADI S-axis at the SOLVE
    configuration (n_x, num_std=8) and reuse it by identity under market
    bumps — otherwise 2D greeks mix market sensitivity with grid movement."""
    product = _snowball()
    env = _env()
    engine = HestonSnowballPDESolver(_heston(), n_x=48, n_v=18, n_t=16)

    ctx = engine.create_bump_context(product, env)
    frozen = ctx._frozen_x_layout
    assert frozen is not None
    assert frozen.spatial.x.size == 48  # bound at the ADI configuration
    tau = product.get_maturity(env)
    for bumped in (
        _env(s0=101.0),
        _env(vol=0.22),
        _env(r=0.04),
        _env(q=0.02),
    ):
        x = ctx._layer_x_nodes(product, bumped, tau)
        assert x is frozen.spatial.x  # SAME array — no per-bump rebuild
    # end-to-end: bumped repricing through the frozen clone works
    assert np.isfinite(float(ctx.price(product, _env(s0=101.0))))


def test_heston_2d_calculate_greeks_resolves_the_frozen_context(monkeypatch):
    product = _snowball()
    env = _env()
    engine = HestonSnowballPDESolver(_heston(), n_x=48, n_v=18, n_t=16)
    create_context = engine.create_bump_context
    captured = {}

    def capture_context(product_arg, env_arg):
        context = create_context(product_arg, env_arg)
        captured["context"] = context
        return context

    monkeypatch.setattr(engine, "create_bump_context", capture_context)

    greeks = engine.calculate_greeks(product, env)

    assert captured["context"]._frozen_x_layout is not None
    assert np.isfinite(greeks["price"])
    assert np.isfinite(greeks["delta"])
    assert np.isfinite(greeks["gamma"])


def test_heston_2d_greeks_reuse_one_solved_surface(monkeypatch):
    """Spot-only finite bumps do not change a frozen Heston PDE surface.

    The production Greek path should therefore march V0/V1 once and read the
    three stencil prices from that surface, rather than repeat the full ADI
    solve for down/base/up.
    """
    product = _snowball()
    env = _env()
    engine = HestonSnowballPDESolver(_heston(), n_x=48, n_v=18, n_t=16)
    original = HestonSnowballPDESolver._solve_live_surface
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HestonSnowballPDESolver, "_solve_live_surface", counted)
    greeks = engine.calculate_greeks(product, env)

    assert len(calls) == 1
    assert all(np.isfinite(greeks[key]) for key in ("price", "delta", "gamma"))


def test_heston_2d_one_surface_greeks_match_frozen_three_reprices():
    product = _snowball()
    env = _env()
    engine = HestonSnowballPDESolver(_heston(), n_x=48, n_v=18, n_t=16)

    expected = BaseEngine.calculate_greeks(engine, product, env)
    actual = engine.calculate_greeks(product, env)

    assert actual == pytest.approx(expected, rel=0.0, abs=2e-10)


def test_slv_2d_one_surface_greeks_match_frozen_three_reprices():
    product = _snowball()
    env = _env()
    engine = HestonSLVSnowballPDESolver(
        _heston(), _unit_leverage(), n_x=48, n_v=18, n_t=16
    )

    expected = BaseEngine.calculate_greeks(engine, product, env)
    actual = engine.calculate_greeks(product, env)

    assert actual == pytest.approx(expected, rel=0.0, abs=2e-10)


def test_heston_2d_greeks_fall_back_when_stencil_changes_valuation_state(
    monkeypatch,
):
    # Continuous KI is live just above 75 but already knocked in at the 1%
    # down point. A single surface cannot represent both states.
    product = _snowball()
    env = _env(s0=75.5)
    engine = HestonSnowballPDESolver(_heston(), n_x=120, n_v=16, n_t=14)
    original = HestonSnowballPDESolver._solve_live_surface
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(bool(kwargs["knocked_in"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HestonSnowballPDESolver, "_solve_live_surface", counted)
    greeks = engine.calculate_greeks(product, env)

    assert len(calls) == 3
    assert set(calls) == {False, True}
    assert all(np.isfinite(greeks[key]) for key in ("price", "delta", "gamma"))


def test_dense_discrete_ki_crossing_selects_aligned_greek_time_grid():
    """A barrier-straddling 1% stencil gets eight steps per schedule tick.

    The synthetic schedule uses a 252-tick year.  The policy must infer that
    clock and choose 8 * 252 = 2016 steps, while an ordinary spot remains on
    the configured production grid.
    """
    ki_times = [i / 252.0 for i in range(1, 253)]
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=ki_times,
            ki_continuous=False,
        ),
    )
    engine = HestonSnowballPDESolver(_heston(), n_x=48, n_v=18, n_t=400)

    near = engine.greek_time_grid_policy(product, _env(s0=75.5))
    ordinary = engine.greek_time_grid_policy(product, _env(s0=100.0))

    assert near["refined"] is True
    assert near["clock_basis"] == 252
    assert near["resolved_n_t"] == 2016
    assert near["steps_per_tick"] == 8
    assert ordinary["refined"] is False
    assert ordinary["resolved_n_t"] == 400


@pytest.mark.parametrize(
    "make_engine",
    [
        pytest.param(
            lambda **kwargs: HestonSnowballPDESolver(_heston(), **kwargs),
            id="heston",
        ),
        pytest.param(
            lambda **kwargs: HestonSLVSnowballPDESolver(
                _heston(), _unit_leverage(), **kwargs
            ),
            id="heston_slv",
        ),
    ],
)
def test_snowball_heston_default_variance_grid_is_power_graded(make_engine):
    engine = make_engine(n_x=48, n_v=18, n_t=16)

    assert engine.v_grid_power == pytest.approx(2.5)
    assert engine.variance_grid_mode == "power"
    core = engine._make_core(_snowball(), _env(), 1.0)
    expected = core.V_max * np.linspace(0.0, 1.0, core.N_V) ** 2.5

    assert core._v_grid_power == pytest.approx(2.5)
    assert core.V_grid == pytest.approx(expected)
    assert np.diff(core.V_grid)[0] < np.diff(core.V_grid)[-1] / 10.0


@pytest.mark.parametrize(
    "make_engine",
    [
        pytest.param(
            lambda **kwargs: HestonSnowballPDESolver(
                _sigma_collapse_heston(), **kwargs
            ),
            id="heston",
        ),
        pytest.param(
            lambda **kwargs: HestonSLVSnowballPDESolver(
                _sigma_collapse_heston(), _unit_leverage(), **kwargs
            ),
            id="heston_slv",
        ),
    ],
)
def test_snowball_sigma_collapse_uses_monotone_path_focused_variance_grid(
    make_engine,
):
    engine = make_engine(n_x=80, n_v=30, n_t=16)
    core = engine._make_core(_snowball(), _env(), 1.0)
    diagnostics = core.variance_operator_diagnostics()

    assert engine.variance_grid_mode == "path_focused"
    assert engine.v_grid_power == 0.0
    assert core.variance_grid_mode == "path_focused"
    assert diagnostics["centered_non_monotone_nodes"] > 0
    assert diagnostics["fallback_nodes"] > 0
    assert diagnostics["monotone"] is True
    assert diagnostics["theta_is_node"] is True
    assert diagnostics["v0_is_node"] is True


@pytest.mark.parametrize(
    "make_engine",
    [
        pytest.param(
            lambda **kwargs: HestonSnowballPDESolver(_heston(), **kwargs),
            id="heston",
        ),
        pytest.param(
            lambda **kwargs: HestonSLVSnowballPDESolver(
                _heston(), _unit_leverage(), **kwargs
            ),
            id="heston_slv",
        ),
    ],
)
def test_snowball_heston_variance_grid_control_and_legacy_opt_out(make_engine):
    legacy = make_engine(
        n_x=48, n_v=18, n_t=16, grid_style="concentrated", v_grid_power=0.0
    )
    custom = make_engine(
        n_x=48, n_v=18, n_t=16, grid_style="concentrated", v_grid_power=3.0
    )
    uniform = make_engine(n_x=48, n_v=18, n_t=16, grid_style="uniform")

    assert legacy._make_core(_snowball(), _env(), 1.0)._v_grid_power == 0.0
    assert custom._make_core(_snowball(), _env(), 1.0)._v_grid_power == 3.0
    assert uniform.v_grid_power == 0.0
    assert uniform._make_core(_snowball(), _env(), 1.0)._v_grid_power == 0.0


@pytest.mark.parametrize("bad_power", [-1.0, 0.5, float("nan"), float("inf")])
def test_snowball_heston_variance_grid_power_is_validated(bad_power):
    with pytest.raises(ValidationError, match="v_grid_power"):
        HestonSnowballPDESolver(_heston(), v_grid_power=bad_power)

    with pytest.raises(ValidationError, match="v_grid_power"):
        HestonSLVSnowballPDESolver(
            _heston(), _unit_leverage(), v_grid_power=bad_power
        )


def test_snowball_heston_uniform_grid_rejects_explicit_power_grading():
    with pytest.raises(ValidationError, match="v_grid_power"):
        HestonSnowballPDESolver(
            _heston(), grid_style="uniform", v_grid_power=2.5
        )


@pytest.mark.parametrize(
    "make_engine",
    [
        pytest.param(
            lambda: HestonSnowballPDESolver(_heston(), v_grid_power=3.0),
            id="heston",
        ),
        pytest.param(
            lambda: HestonSLVSnowballPDESolver(
                _heston(), _unit_leverage(), v_grid_power=3.0
            ),
            id="heston_slv",
        ),
    ],
)
def test_snowball_heston_session_clone_preserves_variance_grid_power(make_engine):
    from quantark.asset.equity.engine.pde.pde_execution_adapters import (
        Heston2DAutocallableSessionAdapter,
    )

    clone = Heston2DAutocallableSessionAdapter()._clone_engine(make_engine())

    assert clone.v_grid_power == pytest.approx(3.0)
    core = clone._make_core(_snowball(), _env(), 1.0)
    assert core._v_grid_power == pytest.approx(3.0)


def test_snowball_heston_session_clone_preserves_variance_operator_policy():
    from quantark.asset.equity.engine.pde.pde_execution_adapters import (
        Heston2DAutocallableSessionAdapter,
    )

    engine = HestonSnowballPDESolver(
        _sigma_collapse_heston(),
        variance_grid_mode="path_focused",
        v_drift_scheme="centered",
    )
    clone = Heston2DAutocallableSessionAdapter()._clone_engine(engine)

    assert clone.variance_grid_mode == "path_focused"
    assert clone.v_drift_scheme == "centered"


def test_snowball_heston_pde_auto_grid_focuses_ki():
    product = _principal_excluded_snowball()
    env = _env()
    solver = HestonSnowballPDESolver(_heston(), n_x=80, n_v=24, n_t=24)

    assert solver._grid_concentration_spot(product, env) == pytest.approx(80.0)

    core = solver._make_core(product, env, product.get_maturity(env))
    # Grid-redesign §4.4: barriers are CONCENTRATION targets, not pinned
    # nodes (cell-average projection is placement-independent); the
    # invariant is eps_crit-tight local spacing at the barrier.
    assert float(np.min(np.abs(core.S_grid - 80.0))) / 80.0 < 0.003


def test_snowball_heston_pde_can_pin_critical_spots_for_diagnostics():
    product = _principal_excluded_snowball()
    env = _env()
    solver = HestonSnowballPDESolver(
        _heston(), n_x=80, n_v=24, n_t=24, pin_critical_spots=True
    )

    core = solver._make_core(product, env, product.get_maturity(env))
    # Grid-redesign §4.4: every critical level gets eps_crit-tight LOCAL
    # spacing from the shared spatial builder (exact pinning retired with
    # the placement-independent cell-average projection).
    for level in [80.0, 100.0, 103.0]:
        assert float(np.min(np.abs(core.S_grid - level))) / level < 0.003


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
        LocalVolSnowballPDESolver(PDEParams()).price(vanilla, env)
    with pytest.raises(PricingError):
        HestonSnowballPDESolver(hp, n_x=30, n_v=12, n_t=10).price(vanilla, env)
