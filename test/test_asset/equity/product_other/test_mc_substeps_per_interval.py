"""Sub-observation stepping for schedule-based Snowball/Phoenix vol MC engines.

The contractual time grid of these engines is the observation schedule (plus
daily nodes only under continuous KI monitoring), so a sparse schedule takes
ONE SDE step per interval. The 2D attribution study (2026-07-24) measured the
resulting stride bias at ~1% of premium for a quarterly-observed Heston
phoenix. ``substeps_per_interval=n`` (mirroring ``HestonDCNMCEngine``) refines
every contractual interval into n equal SDE steps while recording only the
contractual nodes consumed by the payoff kernels.

Contract pinned here:
- ``substeps_per_interval=1`` (the default) is bitwise-identical to the
  pre-feature engines.
- ``substeps_per_interval=n`` is bitwise-identical to driving the SAME engine
  with the n-refined dt array and subsampling every n-th node column.
- The RQMC session spec's Sobol ``dimension`` tracks the fine grid while
  ``time_steps`` stays contractual (path nodes are contractual).
"""

import numpy as np
import pytest
from datetime import datetime

from quantark.asset.equity.engine.mc import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

QUARTERS = [0.25, 0.5, 0.75, 1.0]


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


def _discrete_snowball():
    """Quarterly KO + quarterly discrete KI: a genuinely sparse contractual grid."""
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
            ko_observation_dates=QUARTERS,
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=QUARTERS,
            ki_continuous=False,
        ),
    )


def _phoenix():
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=110.0,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.5, 1.0],
            ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[80.0, 80.0],
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
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


def _generated_nodes(engine, env, dt_array, num_paths=64):
    """White-box: run the engine's path generator outside price()."""
    engine._term_ctx = (env, 100.0)
    gen = engine._create_path_generator(
        100.0, 0.03, 0.01, 0.20, float(np.sum(dt_array)),
        np.asarray(dt_array, dtype=float), num_paths=num_paths,
    )
    paths, _ = gen.generate_paths()
    return np.asarray(paths)


def _snowball_engine_builders():
    hp = _heston()
    return [
        (
            "lv-snowball",
            lambda params, **kw: LocalVolSnowballMCEngine(params=params, **kw),
        ),
        (
            "heston-snowball",
            lambda params, **kw: HestonSnowballMCEngine(
                model_params=hp, params=params, **kw
            ),
        ),
        (
            "slv-snowball",
            lambda params, **kw: HestonSLVSnowballMCEngine(
                model_params=hp,
                params=params,
                leverage_surface=_unit_leverage(),
                **kw,
            ),
        ),
    ]


def _phoenix_engine_builders():
    hp = _heston()
    return [
        (
            "lv-phoenix",
            lambda params, **kw: LocalVolPhoenixMCEngine(params=params, **kw),
        ),
        (
            "heston-phoenix",
            lambda params, **kw: HestonPhoenixMCEngine(
                model_params=hp, params=params, **kw
            ),
        ),
    ]


class TestSubstepsValidation:
    @pytest.mark.parametrize("bad", [0, -2, 1.5, True])
    def test_snowball_engine_rejects_invalid_substeps(self, bad):
        with pytest.raises(ValidationError):
            HestonSnowballMCEngine(
                model_params=_heston(),
                params=MCParams(num_paths=100, seed=3),
                substeps_per_interval=bad,
            )

    @pytest.mark.parametrize("bad", [0, -2, 1.5, True])
    def test_phoenix_engine_rejects_invalid_substeps(self, bad):
        with pytest.raises(ValidationError):
            HestonPhoenixMCEngine(
                model_params=_heston(),
                params=MCParams(num_paths=100, seed=3),
                substeps_per_interval=bad,
            )


class TestFineGridEquivalence:
    """engine(substeps=n) on the contractual grid must equal the SAME engine
    driven with the n-refined dt array, subsampled back to contractual nodes."""

    SUB = 3

    @pytest.mark.parametrize(
        "name,build", _snowball_engine_builders() + _phoenix_engine_builders()
    )
    def test_pseudo_mc_equivalence(self, name, build):
        env = _env()
        params = MCParams(num_paths=48, seed=17)
        dt = np.diff(np.concatenate([[0.0], QUARTERS]))

        sub_engine = build(params, substeps_per_interval=self.SUB)
        nodes_sub = _generated_nodes(sub_engine, env, dt)

        ref_engine = build(params)
        fine_dt = np.repeat(dt / self.SUB, self.SUB)
        nodes_fine = _generated_nodes(ref_engine, env, fine_dt)

        assert nodes_sub.shape == (nodes_fine.shape[0], dt.size + 1)
        np.testing.assert_array_equal(nodes_sub, nodes_fine[:, :: self.SUB])

    def test_qmc_equivalence(self):
        env = _env()
        params = MCParams(num_paths=32, seed=17)
        dt = np.diff(np.concatenate([[0.0], QUARTERS]))
        hp = _heston()

        sub_engine = HestonSnowballMCEngine(
            model_params=hp,
            params=params,
            method=MonteCarloMethod.QUASI,
            substeps_per_interval=self.SUB,
        )
        nodes_sub = _generated_nodes(sub_engine, env, dt)

        ref_engine = HestonSnowballMCEngine(
            model_params=hp, params=params, method=MonteCarloMethod.QUASI
        )
        fine_dt = np.repeat(dt / self.SUB, self.SUB)
        nodes_fine = _generated_nodes(ref_engine, env, fine_dt)

        np.testing.assert_array_equal(nodes_sub, nodes_fine[:, :: self.SUB])


class TestDefaultPathUnchanged:
    def test_substeps_one_price_is_bitwise_default(self):
        env = _env()
        product = _discrete_snowball()
        params = MCParams(num_paths=2_000, seed=11)
        hp = _heston()

        p_default = HestonSnowballMCEngine(model_params=hp, params=params).price(
            product, env
        )
        p_one = HestonSnowballMCEngine(
            model_params=hp, params=params, substeps_per_interval=1
        ).price(product, env)
        assert p_one == p_default


class TestEndToEnd:
    def test_snowball_price_runs_with_substeps(self):
        env = _env()
        product = _discrete_snowball()
        params = MCParams(num_paths=2_000, seed=11)
        hp = _heston()

        p1 = HestonSnowballMCEngine(model_params=hp, params=params).price(
            product, env
        )
        p4 = HestonSnowballMCEngine(
            model_params=hp, params=params, substeps_per_interval=4
        ).price(product, env)
        assert np.isfinite(p4)
        assert p4 != p1  # finer SDE grid => different discretization

    def test_phoenix_price_runs_with_substeps(self):
        env = _env()
        product = _phoenix()
        params = MCParams(num_paths=2_000, seed=11)

        p2 = HestonPhoenixMCEngine(
            model_params=_heston(), params=params, substeps_per_interval=2
        ).price(product, env)
        assert np.isfinite(p2)


class TestRQMCSpecDimension:
    def test_dimension_tracks_fine_grid_time_steps_stay_contractual(self):
        env = _env()
        product = _discrete_snowball()
        params = MCParams(num_paths=64, seed=5, rqmc_min_batches=2, rqmc_max_batches=4)
        hp = _heston()

        base = HestonSnowballMCEngine(
            model_params=hp, params=params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        spec_base = base.build_rqmc_session_spec(product, env)

        sub = HestonSnowballMCEngine(
            model_params=hp, params=params,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            substeps_per_interval=3,
        )
        spec_sub = sub.build_rqmc_session_spec(product, env)

        assert spec_sub.time_steps == spec_base.time_steps
        assert spec_sub.dimension == 3 * spec_base.dimension
