"""The base family prices per contract (spec 3C decision 24, §7.2 'contract multiplier' row): price, delta, gamma
and session pv at multiplier 10 are 10 x the unit values; the analytic engines agree."""
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.pde import BarrierPDESolver, DoubleBarrierPDESolver, EuropeanPDESolver
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, EuropeanVanillaOption)
from quantark.util.enum import BarrierType, DoubleBarrierType, ObservationType, OptionType

REL = 1e-12


def european(m):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0, contract_multiplier=m)


def barrier(m, bt=BarrierType.UP_OUT, rebate=5.0):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=bt, maturity=1.0,
                         rebate=rebate, observation_type=ObservationType.CONTINUOUS, contract_multiplier=m)


def double(m, bt=DoubleBarrierType.KNOCK_OUT, rebate=0.0):
    return DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
                               barrier_type=bt, maturity=1.0, rebate=rebate,
                               observation_type=ObservationType.CONTINUOUS, contract_multiplier=m)


CASES = [
    ("european", EuropeanPDESolver, european, 100.0),
    ("up_out_rebate", BarrierPDESolver, barrier, 100.0),
    ("up_in", BarrierPDESolver, lambda m: barrier(m, BarrierType.UP_IN, 0.0), 100.0),
    ("up_in_knocked_in", BarrierPDESolver, lambda m: barrier(m, BarrierType.UP_IN, 0.0), 135.0),
    ("double_out", DoubleBarrierPDESolver, double, 100.0),
    ("double_in", DoubleBarrierPDESolver, lambda m: double(m, DoubleBarrierType.KNOCK_IN), 100.0),
    ("up_out_hit", BarrierPDESolver, barrier, 135.0),                      # already hit: the rebate scales
]


def _ratio_ok(a, b):
    return abs(b - 10.0 * a) <= REL * max(1.0, abs(10.0 * a))


@pytest.mark.parametrize("name,cls,make,spot", CASES, ids=[c[0] for c in CASES])
def test_price_greeks_and_session_scale(name, cls, make, spot):
    env = env_const(spot)
    one, ten = make(1.0), make(10.0)
    assert _ratio_ok(cls().price(one, env), cls().price(ten, env))
    g1, g10 = cls().calculate_greeks(one, env), cls().calculate_greeks(ten, env)
    for k in ("price", "delta", "gamma"):
        assert _ratio_ok(g1[k], g10[k]), k
    assert _ratio_ok(cls()._session_outputs(one, env).npv, cls()._session_outputs(ten, env).npv)


def test_expired_product_scales_alike():
    """A tenor product aged to its maturity, as ``time_shift`` leaves it (``maturity = 0.0`` after construction:
    a tenor of 0.0 cannot be constructed, §4.12, and a dated product raises on its exercise date)."""
    env = env_const(125.0)

    def expired(m):
        product = european(m)
        product.maturity = 0.0
        return product

    assert _ratio_ok(EuropeanPDESolver().price(expired(1.0), env), EuropeanPDESolver().price(expired(10.0), env))


def test_analytic_engines_scale():
    env = env_const(100.0)
    assert _ratio_ok(BarrierAnalyticalEngine().price(barrier(1.0), env), BarrierAnalyticalEngine().price(barrier(10.0), env))
    assert _ratio_ok(DoubleBarrierOptionAnalyticalEngine().price(double(1.0), env),
                     DoubleBarrierOptionAnalyticalEngine().price(double(10.0), env))


def test_multiplier_one_is_bit_identical_to_the_unit_solve():
    """x * 1.0 is exact: a product built without the keyword prices to the same bits."""
    env = env_const(100.0)
    plain = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                          maturity=1.0, rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    assert BarrierPDESolver().price(plain, env).hex() == BarrierPDESolver().price(barrier(1.0), env).hex()


# --- Task 12: a knock-in with a rebate per contract (spec 3C decision 8) ------------------------------------------

@pytest.mark.parametrize("cls,make,spot", [
    (BarrierPDESolver, lambda m: barrier(m, BarrierType.UP_IN, 5.0), 100.0),
    (DoubleBarrierPDESolver, lambda m: double(m, DoubleBarrierType.KNOCK_IN, 5.0), 100.0),
    (BarrierPDESolver, lambda m: barrier(m, BarrierType.UP_IN, 5.0), 135.0),        # knocked in: the rebate is forfeited
], ids=["up_in_rebate", "double_in_rebate", "up_in_rebate_knocked_in"])
def test_knock_in_with_rebate_scales(cls, make, spot):
    env = env_const(spot)
    one, ten = make(1.0), make(10.0)
    assert _ratio_ok(cls().price(one, env), cls().price(ten, env))
    g1, g10 = cls().calculate_greeks(one, env), cls().calculate_greeks(ten, env)
    for k in ("price", "delta", "gamma"):
        assert _ratio_ok(g1[k], g10[k]), k
