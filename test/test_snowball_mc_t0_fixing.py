"""SnowballMCEngine: an observation exactly at valuation is decided on the known spot (path column 0)."""
import pytest

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.util.enum.engine_enums import MonteCarloMethod
from test_snowball_mc_engine import create_basic_barrier_config, create_pricing_env, create_standard_snowball


def _price(config, spot=100.0, maturity=1.0):
    engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42), method=MonteCarloMethod.PSEUDO)
    product = create_standard_snowball(barrier_config=config, maturity=maturity)
    env = create_pricing_env(spot=spot)
    return engine.price(product, env), product, env


def test_legacy_grids_are_bitwise_unchanged():
    # pinned before the t=0 fix: no legacy schedule has an observation at t=0
    assert _price(create_basic_barrier_config())[0].hex() == "0x1.df1b3ddedad2fp+19"
    discrete = dict(ki_continuous=False, ki_observation_dates=[0.25, 0.5, 0.75, 1.0])
    assert _price(create_basic_barrier_config(**discrete))[0].hex() == "0x1.e334b9c910f9ap+19"
    assert _price(create_basic_barrier_config(disable_ko_after_ki=True, **discrete))[0].hex() == "0x1.e331289aec461p+19"


def test_todays_ko_fixing_above_the_barrier_pays_the_ko_cash_without_simulation_steps():
    config = create_basic_barrier_config(ko_observation_dates=[0.0, 0.25, 0.5])
    price, product, env = _price(config, spot=105.0, maturity=0.5)
    first = product.resolve_ko_observations(env)[0]
    assert first.observation_time == 0.0
    assert price == pytest.approx(first.payoff * env.get_discount_factor(first.settlement_time) if first.settlement_time else first.payoff,
                                  rel=1e-12)


def test_an_untriggered_t0_fixing_changes_nothing():
    with_t0 = _price(create_basic_barrier_config(ko_observation_dates=[0.0, 0.25, 0.5]), maturity=0.5)[0]
    without = _price(create_basic_barrier_config(ko_observation_dates=[0.25, 0.5]), maturity=0.5)[0]
    assert with_t0 == without


def test_todays_discrete_ki_fixing_below_the_barrier_knocks_in_at_t0():
    discrete = dict(ki_continuous=False)
    knocked = _price(create_basic_barrier_config(ko_observation_dates=[0.25, 0.5], ki_observation_dates=[0.0, 0.25, 0.5], **discrete),
                     spot=70.0, maturity=0.5)[0]
    later = _price(create_basic_barrier_config(ko_observation_dates=[0.25, 0.5], ki_observation_dates=[0.25, 0.5], **discrete),
                   spot=70.0, maturity=0.5)[0]
    assert knocked < later          # knocked in for certain vs only if still below at 0.25
