"""A prepared session prices exactly what the direct call prices (final-review fix pass).

Knock-in barriers are composites (spec 2 decision 21): the session's stored artefacts
belong to the knock-in product's own request, which no leg marches on. The knock-out
leg must bind its own hard-edged layout, not reuse them. And a session honours
``cache_enabled=False`` as the direct call does: no shared factorization pack.
"""
import pytest

from quantark.asset.equity.engine.pde import BarrierPDESolver, DoubleBarrierPDESolver, SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption
from quantark.execution import PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.enum.option_enums import DoubleBarrierType

from pde_bitwise.capture_book import env_flat
from term_structure_benchmarks import make_term_env


def _up_in_call():
    return BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=120.0,
                         barrier_type=BarrierType.UP_IN, observation_type=ObservationType.CONTINUOUS)


def _double_knock_in():
    return DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                               maturity=1.0)


def _session_value(engine, product, env):
    with PricingSession() as session:
        return session.execute(engine, PricingRequest(product=product, pricing_env=env)).value


@pytest.mark.parametrize("cls,make", [(BarrierPDESolver, _up_in_call),
                                      (DoubleBarrierPDESolver, _double_knock_in)])
def test_a_knock_in_through_a_session_prices_as_direct(cls, make):
    product, env = make(), env_flat()
    direct = cls(PDEParams()).price(product, env)
    assert _session_value(cls(PDEParams()), product, env) == direct


def test_a_cache_disabled_session_warms_no_factorization(monkeypatch):
    # spec 2 §5.7: the factorization cache is bypassed when cache_enabled is false,
    # so a session must not warm (and then seed the march with) a shared pack
    from quantark.asset.equity.engine.pde.core import march as march_mod

    warmed = []
    real = march_mod.warm_factorizations
    monkeypatch.setattr(march_mod, "warm_factorizations",
                        lambda *a, **k: warmed.append(1) or real(*a, **k))
    SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
              rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)
    product, env = make_snowball(SB), make_term_env("kinked")
    params = PDEParams(cache_enabled=False)
    direct = SnowballPDESolver(params).price(product, env)
    assert _session_value(SnowballPDESolver(params), product, env) == direct
    assert warmed == []
