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
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import merge_time_nodes
from quantark.volmodels.heston import HestonParams

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
