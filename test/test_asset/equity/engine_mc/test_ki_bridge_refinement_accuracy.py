"""Accuracy gates for the opt-in KI-bridge refinement and SDE step cap (spec AC5-AC7).

Seeded, deterministic Monte Carlo against deterministic references, in bp of notional (initial price):
  - local vol (AH and Dupire on the iv_surface_20240208 fixture): the LV PDE at 3200x64;
  - Heston / SLV: the frozen bridge-free MC reference in docs/research/mc_ki_bridge_refinement/
    references.json (dense discrete KI extrapolated to continuous monitoring; reference.py rebuilds it).
Every gate also checks that today's default misses, so the gate measures the refinement, not luck.
"""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import pytest

from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import LocalVolPhoenixMCEngine
from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

pytestmark = pytest.mark.slow

RESEARCH = Path(__file__).resolve().parents[4] / "docs" / "research" / "mc_ki_bridge_refinement"
BAR_BP = 2.0
PATHS = 1 << 17
PHOENIX_SEEDS = tuple(range(8))


@lru_cache(maxsize=1)
def _ref():
    spec = importlib.util.spec_from_file_location("mc_ki_bridge_reference", RESEARCH / "reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _frozen():
    return json.loads((RESEARCH / "references.json").read_text())


@lru_cache(maxsize=1)
def _fixture():
    return _ref().fixture()


def _bp(price, reference, product):
    return 1e4 * (price - reference) / product.initial_price


def _lv_mc_bp(lv, product, env, reference, seeds, **kw):
    price, se = _ref().mean_price(lambda p, **k: LocalVolSnowballMCEngine(p, local_vol_surface=lv, **k),
                                  product, env, seeds, PATHS, **kw)
    return _bp(price, reference, product), 1e4 * se / product.initial_price


# --- AC5: near-KI two-week contracts on local vol at 16 nodes per day ---------------------------------

@pytest.mark.parametrize("surface", ["AH", "Dupire"])
def test_lv_snowball_near_ki_reprices_the_pde_at_16_nodes_per_day(surface):
    env, spot, surfaces = _fixture()
    lv, product = surfaces[surface], _ref().a1(spot)
    reference = LocalVolSnowballPDESolver(_ref().pde_params(), local_vol_surface=lv).price(product, env)
    refined, _ = _lv_mc_bp(lv, product, env, reference, _ref().SEEDS_A, ki_bridge_nodes_per_day=16)
    assert abs(refined) <= BAR_BP, refined
    if surface == "AH":
        default, _ = _lv_mc_bp(lv, product, env, reference, _ref().SEEDS_A)
        assert abs(default) > BAR_BP, default


def _phoenix(spot):
    initial = spot / 0.86
    return create_standard_phoenix(initial_price=initial, strike=initial, maturity=_ref().T_A1,
                                   ko_barrier=1.03 * initial, ko_rate=0.15, ki_barrier=0.80 * initial,
                                   coupon_barrier=0.85 * initial, coupon_rate=0.01, num_observations=2)


@pytest.mark.parametrize("surface", ["AH", "Dupire"])
def test_lv_phoenix_near_ki_at_16_nodes_per_day(surface):
    """AH: within the bar. Dupire carries a ~+2 bp offset flat in the node density (not bridge bias,
    spec Goal): there the refined error must be no worse than the default's plus two standard errors."""
    env, spot, surfaces = _fixture()
    lv, product = surfaces[surface], _phoenix(spot)
    reference = LocalVolPhoenixPDESolver(_ref().pde_params(), local_vol_surface=lv).price(product, env)

    def price(**kw):
        value, se = _ref().mean_price(lambda p, **k: LocalVolPhoenixMCEngine(p, local_vol_surface=lv, **k),
                                      product, env, PHOENIX_SEEDS, PATHS, **kw)
        return _bp(value, reference, product), 1e4 * se / product.initial_price

    refined, se = price(ki_bridge_nodes_per_day=16)
    default, _ = price()
    if surface == "AH":
        assert abs(refined) <= BAR_BP, refined
        assert abs(default) > BAR_BP, default
    else:
        assert abs(refined) <= abs(default) + 2.0 * se, (refined, default)


# --- AC6: Heston and SLV against the frozen bridge-free reference -------------------------------------

@pytest.mark.parametrize("model", ["heston", "slv"])
def test_stochastic_vol_near_ki_reprices_the_bridge_free_reference(model):
    ref = _ref()
    make, env, product = ref.heston_models()[model], ref.heston_env(), ref.a1(100.0)
    reference = _frozen()[f"{model}_a1_bridge_free"]["price"]
    refined, _ = ref.mean_price(make, product, env, PHOENIX_SEEDS, PATHS, ki_bridge_nodes_per_day=16)
    default, _ = ref.mean_price(make, product, env, PHOENIX_SEEDS, PATHS)
    refined_bp, default_bp = _bp(refined, reference, product), _bp(default, reference, product)
    assert abs(refined_bp) <= BAR_BP, refined_bp
    assert abs(default_bp) > abs(refined_bp), (default_bp, refined_bp)


# --- AC7: the step cap on a one-interval contract -----------------------------------------------------

@pytest.mark.parametrize("surface", ["AH", "Dupire"])
def test_step_cap_removes_the_single_step_bias(surface):
    env, spot, surfaces = _fixture()
    lv, product = surfaces[surface], _ref().case_b(spot)
    reference = LocalVolSnowballPDESolver(_ref().pde_params(), local_vol_surface=lv).price(product, env)
    default, _ = _lv_mc_bp(lv, product, env, reference, _ref().SEEDS_A)
    one_day, _ = _lv_mc_bp(lv, product, env, reference, _ref().SEEDS_A, max_sde_step_days=1)
    tenth, _ = _lv_mc_bp(lv, product, env, reference, _ref().SEEDS_A, max_sde_step_days=0.1)
    assert abs(default) > 20.0, default
    assert abs(one_day) <= 5.0, one_day
    assert abs(tenth) <= BAR_BP, tenth
