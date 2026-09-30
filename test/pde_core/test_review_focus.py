"""Inputs the spec implies and no spec test exercises (plan "Review Focus")."""
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from term_structure_benchmarks import make_term_env

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def test_cache_disabled_agrees_with_cache_enabled_to_golden_tolerance():
    # Review Focus 5.
    from golden_compare import assert_close
    p = make_snowball(SB); env = make_term_env("kinked")
    on = SnowballPDESolver(PDEParams()).price(p, env)
    off = SnowballPDESolver(PDEParams(cache_enabled=False)).price(p, env)
    assert_close(off, on)   # not ==: the rounded dt key shares ULP-different intervals (§5.7)
