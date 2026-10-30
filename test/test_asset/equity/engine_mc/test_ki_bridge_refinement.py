"""Opt-in continuous-KI bridge refinement (``ki_bridge_nodes_per_day``) and SDE step cap.

Spec: .ship/2026-10-30-mc-ki-bridge-refinement/spec.md (AC1-AC4). The defaults must leave every engine
bit-for-bit unchanged; the accuracy gates live in test_ki_bridge_refinement_accuracy.py.
"""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from golden_compare import assert_close
from quantark.asset.equity.engine.mc import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
    QEPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.richardson import richardson_pair_price
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import HestonMCScheme, MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import merge_time_nodes
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv.leverage import LeverageSurface

ROOT = Path(__file__).resolve().parents[4]
RESEARCH = ROOT / "docs" / "research" / "mc_ki_bridge_refinement"
KO_DATES = [0.25, 0.5, 0.75, 1.0]
HESTON = HestonParams(v0=0.06, kappa=2.0, theta=0.06, sigma=0.5, rho=-0.7)


def _env():
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
                              div_yield=ContinuousDividendYield(0.01))


def _snowball(continuous=True):
    ki = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if continuous
          else dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=KO_DATES))
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=KO_DATES, ki_barrier=80.0, **ki))


def _ko_reset():
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=KO_DATES,
                                     ki_barrier=80.0, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=98.0, ko_rate=0.03, ko_observation_dates=KO_DATES))


def _phoenix():
    return create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=103.0,
                                   ko_rate=0.0, ki_barrier=80.0, coupon_barrier=85.0, coupon_rate=0.01,
                                   num_observations=4)


def _expected_nodes(env, n, horizon, extra):
    count = int(env.bus_days_in_year * n * horizon) + 1
    ki = list(np.linspace(0.0, horizon, count + 1)[1:])
    return merge_time_nodes([t for t in ki + list(extra) if t > 0.0] + [1.0], pin=1.0)


# --- AC2: the continuous-KI node set -------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 4, 16])
def test_snowball_continuous_ki_nodes_scale_with_nodes_per_day(n):
    env, product = _env(), _snowball()
    all_times = SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=n)._build_time_grid(product, env, 1.0)[0]
    np.testing.assert_array_equal(all_times, _expected_nodes(env, n, 1.0, KO_DATES))


def test_snowball_one_node_per_day_is_the_default_grid():
    env, product = _env(), _snowball()
    default = SnowballMCEngine(MCParams())._build_time_grid(product, env, 1.0)
    explicit = SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=1)._build_time_grid(product, env, 1.0)
    for a, b in zip(default, explicit):
        np.testing.assert_array_equal(a, b)


def test_discrete_ki_grid_ignores_nodes_per_day():
    env, product = _env(), _snowball(continuous=False)
    default = SnowballMCEngine(MCParams())._build_time_grid(product, env, 1.0)
    refined = SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=16)._build_time_grid(product, env, 1.0)
    for a, b in zip(default, refined):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 4, 16])
def test_phoenix_continuous_ki_nodes_scale_with_nodes_per_day(n):
    env, product = _env(), _phoenix()
    ko_times = product.get_ko_observation_profile(env)["observation_times"]
    all_times = PhoenixMCEngine(MCParams(), ki_bridge_nodes_per_day=n)._build_time_grid(product, env, 1.0)[0]
    np.testing.assert_array_equal(all_times, _expected_nodes(env, n, 1.0, ko_times))


@pytest.mark.parametrize("n", [1, 4, 16])
def test_ko_reset_knock_in_horizon_nodes_scale_with_nodes_per_day(n):
    env, product = _env(), _ko_reset()
    grid = SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=n)._build_time_grid_ko_reset(product, env, 1.0)
    horizon = float(product.get_pre_maturity_time(env))
    count = int(env.bus_days_in_year * n * horizon) + 1
    ki = np.linspace(0.0, horizon, count + 1)[1:]
    assert all(np.min(np.abs(grid["all_times"] - t)) < 1e-12 for t in ki)
    default = SnowballMCEngine(MCParams())._build_time_grid_ko_reset(product, env, 1.0)
    if n == 1:
        np.testing.assert_array_equal(grid["all_times"], default["all_times"])
    else:
        assert len(grid["all_times"]) > len(default["all_times"])


# --- AC3: validation ------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "4", None])
@pytest.mark.parametrize("engine_cls", [SnowballMCEngine, PhoenixMCEngine])
def test_invalid_nodes_per_day_raises(engine_cls, bad):
    with pytest.raises(ValidationError):
        engine_cls(MCParams(), ki_bridge_nodes_per_day=bad)


def test_numpy_integer_nodes_per_day_is_accepted():
    assert SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=np.int64(4)).ki_bridge_nodes_per_day == 4


# --- the keyword reaches every engine that shares the bridge -------------------------------------------

def _vol_engines(**kw):
    from quantark.volmodels.slv.leverage import LeverageSurface

    strikes = np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                          leverage_grid=np.ones((4, strikes.size)))
    p = MCParams()
    return [
        LocalVolSnowballMCEngine(p, **kw),
        HestonSnowballMCEngine(HESTON, p, **kw),
        QESnowballMCEngine(HESTON, p, **kw),
        HestonSLVSnowballMCEngine(HESTON, params=p, leverage_surface=lev, **kw),
        HestonSLVQESnowballMCEngine(HESTON, params=p, leverage_surface=lev, **kw),
        LocalVolPhoenixMCEngine(p, **kw),
        HestonPhoenixMCEngine(HESTON, p, **kw),
        QEPhoenixMCEngine(HESTON, p, **kw),
        HestonSLVPhoenixMCEngine(HESTON, params=p, leverage_surface=lev, **kw),
        HestonSLVQEPhoenixMCEngine(HESTON, params=p, leverage_surface=lev, **kw),
    ]


def test_every_vol_engine_forwards_nodes_per_day():
    for engine in _vol_engines(ki_bridge_nodes_per_day=8):
        assert engine.ki_bridge_nodes_per_day == 8, type(engine).__name__
    for engine in _vol_engines():
        assert engine.ki_bridge_nodes_per_day == 1, type(engine).__name__


def test_scheme_label_names_the_refinement_only_when_set():
    for default, refined in zip(_vol_engines(), _vol_engines(ki_bridge_nodes_per_day=8)):
        assert "kib" not in default._rqmc_scheme_label()
        assert refined._rqmc_scheme_label() == default._rqmc_scheme_label() + "+kib:8"
    assert SnowballMCEngine(MCParams(), ki_bridge_nodes_per_day=8)._rqmc_scheme_label().endswith("+kib:8")
    assert PhoenixMCEngine(MCParams(), ki_bridge_nodes_per_day=8)._rqmc_scheme_label().endswith("+kib:8")


# --- AC1: defaults equal the prices captured on the base commit ----------------------------------------

def _bitwise_module():
    spec = importlib.util.spec_from_file_location("mc_ki_bridge_bitwise_check", RESEARCH / "bitwise_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defaults_match_base_prices():
    """Frozen on ARM64 at the base commit (same-machine byte compare in the run record); cross-arch tolerance."""
    module = _bitwise_module()
    golden = json.loads((RESEARCH / "bitwise_base.json").read_text())["defaults"]
    env, prods = module.environment(), module.products()
    live, explicit = {}, {}
    for name, (make, i) in module.engines(env, explicit=False).items():
        live[name] = float(make().price(prods[i], env))
    for name, (make, i) in module.engines(env, explicit=True).items():
        explicit[name] = float(make().price(prods[i], env))
    assert_close(live, {k: float(v) for k, v in golden.items()})
    assert explicit == live


# --- AC3 (cap): validation and the unsupported combinations -------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, 0.0, float("nan"), float("inf"), True, "1"])
def test_invalid_step_cap_raises(bad):
    with pytest.raises(ValidationError):
        LocalVolSnowballMCEngine(MCParams(), max_sde_step_days=bad)
    with pytest.raises(ValidationError):
        LocalVolPhoenixMCEngine(MCParams(), max_sde_step_days=bad)


def test_step_cap_int_and_float_are_one_setting():
    a = LocalVolSnowballMCEngine(MCParams(), max_sde_step_days=1)
    b = LocalVolSnowballMCEngine(MCParams(), max_sde_step_days=1.0)
    assert a.max_sde_step_days == b.max_sde_step_days == 1.0
    assert isinstance(a.max_sde_step_days, float)
    assert a._rqmc_scheme_label() == b._rqmc_scheme_label()
    assert a._rqmc_scheme_label().endswith("+cap:1.0")
    assert "cap" not in LocalVolSnowballMCEngine(MCParams())._rqmc_scheme_label()


@pytest.mark.parametrize("method", [MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI])
def test_step_cap_refuses_quasi_methods(method):
    params = MCParams(use_antithetic=False)
    with pytest.raises(ValidationError, match="max_sde_step_days"):
        LocalVolSnowballMCEngine(params, method=method, max_sde_step_days=1)
    with pytest.raises(ValidationError, match="max_sde_step_days"):
        LocalVolPhoenixMCEngine(params, method=method, max_sde_step_days=1)


@pytest.mark.parametrize("flags", [
    dict(rqmc_affine_spot_factor=True),
    dict(rqmc_heston_conditional_control=True),
    dict(rqmc_heston_conditional_control=True, rqmc_spot_bridge_strata=2),
    dict(rqmc_heston_conditional_control=True, rqmc_frozen_leverage_conditional_control=True,
         rqmc_conditional_control_only=True),
    dict(rqmc_heston_conditional_control=True, rqmc_spot_strata=2),
    dict(rqmc_heston_conditional_control=True, rqmc_spot_antithetic=True),
])
def test_step_cap_refuses_rqmc_conditioning_flags(flags):
    """CP1: on PSEUDO, the SLV-QE bridge-stratified route returned fine-grid paths marked contractual
    under a cap -- a silent wrong price. Every conditioning flag refuses a cap at construction."""
    strikes = np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                          leverage_grid=np.ones((4, strikes.size)))
    for make in (lambda **kw: HestonSnowballMCEngine(HESTON, MCParams(use_antithetic=False), **kw),
                 lambda **kw: HestonSLVQESnowballMCEngine(HESTON, params=MCParams(use_antithetic=False),
                                                          leverage_surface=lev, **kw)):
        try:
            make(**flags)
        except ValidationError:
            continue                      # the flag set alone is invalid on this engine
        with pytest.raises(ValidationError, match="max_sde_step_days"):
            make(**flags, max_sde_step_days=1)


def test_step_cap_refuses_one_step_survival():
    with pytest.raises(ValidationError, match="max_sde_step_days"):
        LocalVolSnowballMCEngine(MCParams(), estimator="one_step_survival", max_sde_step_days=1)


def test_step_cap_refused_at_price_time_when_the_method_changed():
    engine = LocalVolSnowballMCEngine(MCParams(use_antithetic=False), max_sde_step_days=1)
    engine.method = MonteCarloMethod.QUASI
    with pytest.raises(ValidationError, match="max_sde_step_days"):
        engine.price(_short_products()["snowball"][1], _skew_env())


def test_richardson_refuses_a_capped_leg():
    def factory(n):
        return LocalVolSnowballMCEngine(MCParams(seed=11 if n == 1 else 12), substeps_per_interval=n,
                                        max_sde_step_days=1)
    with pytest.raises(ValidationError, match="max_sde_step_days"):
        richardson_pair_price(factory, _short_products()["snowball"][1], _skew_env(), substeps=1)


# --- AC4: the cap splits only what it must, and keeps the contractual grid ----------------------------

def _skew_env():
    strikes = list(100.0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    skew = 0.22 - 0.10 * np.linspace(-0.5, 0.5, 9)
    surface = GridVolSurface(strikes, [0.1, 0.25, 0.5, 1.0], np.tile(skew, (4, 1)))
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=100.0), vol_surface=surface,
                              div_yield=ContinuousDividendYield(0.01))


def _mixin_engines(env, **kw):
    lv = build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
                                div_yield=env.get_div_yield)
    strikes = np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    smile = np.linspace(1.15, 0.90, strikes.size)
    lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 4), strike_grid=strikes,
                          leverage_grid=np.vstack([smile, 0.75 * smile + 0.25, 0.5 * smile + 0.5,
                                                   np.ones_like(smile)]))
    p = MCParams(num_paths=2048, seed=5, use_antithetic=True)
    slv = dict(leverage_surface=lev, local_vol_surface=lv)
    return {
        "lv_snowball": (LocalVolSnowballMCEngine(p, local_vol_surface=lv, **kw), "snowball"),
        "heston_qe": (HestonSnowballMCEngine(HESTON, p, **kw), "snowball"),
        "heston_eulerlog": (HestonSnowballMCEngine(HESTON, p, scheme=HestonMCScheme.EULERLOG, **kw), "snowball"),
        "heston_qem": (HestonSnowballMCEngine(HESTON, p, scheme=HestonMCScheme.QUADEXP_M, **kw), "snowball"),
        "qe_standalone": (QESnowballMCEngine(HESTON, p, **kw), "snowball"),
        "slv": (HestonSLVSnowballMCEngine(HESTON, params=p, **slv, **kw), "snowball"),
        "slv_qe": (HestonSLVQESnowballMCEngine(HESTON, params=p, **slv, **kw), "snowball"),
        "lv_phoenix": (LocalVolPhoenixMCEngine(p, local_vol_surface=lv, **kw), "phoenix"),
        "heston_phoenix": (HestonPhoenixMCEngine(HESTON, p, **kw), "phoenix"),
        "qe_phoenix": (QEPhoenixMCEngine(HESTON, p, **kw), "phoenix"),
        "slv_phoenix": (HestonSLVPhoenixMCEngine(HESTON, params=p, **slv, **kw), "phoenix"),
        "slv_qe_phoenix": (HestonSLVQEPhoenixMCEngine(HESTON, params=p, **slv, **kw), "phoenix"),
    }


def _short_products():
    """One-month contracts: a continuous-KI pair (daily grid, so a 1-day cap splits nothing) and a
    maturity-only pair (one 30-day interval, so a cap splits it)."""
    T = 30 / 365

    def snowball(**ki):
        return SnowballOption(
            initial_price=100.0, strike=100.0, maturity=T, contract_multiplier=1.0, is_reverse=False,
            barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                         ko_observation_type=ObservationType.DISCRETE,
                                         ko_observation_dates=[T], ki_barrier=90.0, **ki))

    def phoenix(**ki):
        return create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=T, ko_barrier=103.0,
                                       ko_rate=0.0, ki_barrier=90.0, coupon_barrier=95.0, coupon_rate=0.01,
                                       num_observations=1, **ki)

    continuous = dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    at_maturity = dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=[T])
    # create_standard_phoenix sets ki_continuous=True unless told otherwise
    return {"snowball": (snowball(**continuous), snowball(**at_maturity)),
            "phoenix": (phoenix(), phoenix(**at_maturity, ki_continuous=False))}


def test_a_cap_that_splits_nothing_is_bitwise_the_uncapped_engine():
    env, products = _skew_env(), _short_products()
    plain, capped = _mixin_engines(env), _mixin_engines(env, max_sde_step_days=1)
    for name, (engine, kind) in plain.items():
        product = products[kind][0]
        assert capped[name][0].price(product, env) == engine.price(product, env), name


def test_a_cap_that_splits_one_interval_equals_uniform_substeps():
    """A 30-day single interval with a cap of an eighth of it (plus a hair) is split into exactly 8
    sub-steps -- the SDE steps, readout and bridge fold of substeps_per_interval=8 -- on every engine."""
    env, products = _skew_env(), _short_products()
    cap_days = 30 / 365 * env.bus_days_in_year / 8 * (1 + 1e-6)
    capped = _mixin_engines(env, max_sde_step_days=cap_days)
    uniform = _mixin_engines(env, substeps_per_interval=8)
    for name, (engine, kind) in capped.items():
        product = products[kind][1]
        assert engine.price(product, env) == uniform[name][0].price(product, env), name


def test_non_uniform_counts_read_contractual_nodes_and_fold_variance():
    """Mixin level: per-interval counts drive the readout columns and the reduceat variance fold."""
    engine = LocalVolSnowballMCEngine(MCParams(), max_sde_step_days=1)
    engine._substep_counts = np.array([1, 3, 2])
    fine = np.arange(4 * 7, dtype=float).reshape(4, 7)          # 4 paths, 6 fine steps + t=0
    generator = engine._make_path_generator(lambda batch_id=None, seed=None: fine, 4, None)
    np.testing.assert_array_equal(generator.generate_paths()[0], fine[:, [0, 1, 4, 6]])
    variance = np.arange(4 * 6, dtype=float).reshape(4, 6)
    engine._record_step_log_variance(variance)
    np.testing.assert_array_equal(
        engine._step_log_variance,
        np.stack([variance[:, 0], variance[:, 1:4].sum(axis=1), variance[:, 4:6].sum(axis=1)], axis=1))


def test_refined_dt_array_counts_follow_the_cap():
    env = _skew_env()
    engine = LocalVolSnowballMCEngine(MCParams(), substeps_per_interval=2, max_sde_step_days=1)
    engine._term_ctx = (env, 100.0)
    bday = 1.0 / env.bus_days_in_year
    dt = np.array([0.5 * bday, 5.0 * bday, 1.0 * bday])
    fine = engine._refined_dt_array(dt)
    np.testing.assert_array_equal(engine._substep_counts, [2, 5, 2])
    assert len(fine) == 9
    np.testing.assert_allclose([fine[:2].sum(), fine[2:7].sum(), fine[7:].sum()], dt, rtol=1e-15)
