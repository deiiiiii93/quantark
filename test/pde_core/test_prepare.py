"""prepare → march → read for the base family; the knock-in composite (spec 2 §5.1-§5.3, decision 21)."""
from quantark.asset.equity.engine.pde import BarrierPDESolver, EuropeanPDESolver
from quantark.asset.equity.engine.pde.core.prepare import (CompositeProblem, Operation, OutputRequest, prepare,
                                                          run)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
from quantark.util.enum import BarrierType, ObservationType, OptionType

from pde_bitwise.capture_book import env_flat, expired_barrier


def replace_barrier_type(p, barrier_type):
    return BarrierOption(strike=p.strike, maturity=p.maturity, option_type=p.option_type, barrier=p.barrier,
                         barrier_type=barrier_type, observation_type=p.observation_type)


def _up_in_call():
    return BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=120.0,
                         barrier_type=BarrierType.UP_IN, observation_type=ObservationType.CONTINUOUS)


def test_knock_in_prepares_two_problems_on_their_own_domains():
    comp = prepare(BarrierPDESolver(PDEParams()), _up_in_call(), env_flat(), OutputRequest(Operation.PRICE))
    assert isinstance(comp, CompositeProblem) and len(comp.parts) == 2
    vanilla, ko = comp.parts
    assert vanilla.layout.spatial.bounds[1] > 200.0          # automatic envelope (measured 222.554…)
    assert ko.layout.spatial.bounds[1] == 120.0              # the barrier's hard edge


def test_knock_in_composition_is_vanilla_minus_knock_out_for_price_and_greeks():
    ki = _up_in_call()
    env = env_flat()
    ko = replace_barrier_type(ki, BarrierType.UP_OUT)
    van = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g_ki = BarrierPDESolver(PDEParams()).calculate_greeks(ki, env)
    g_ko = BarrierPDESolver(PDEParams()).calculate_greeks(ko, env)
    g_v = EuropeanPDESolver(PDEParams()).calculate_greeks(van, env)
    for k in ("price", "delta", "gamma"):
        assert g_ki[k] == ki.participation_rate * g_v[k] - g_ko[k]


def test_expired_product_has_a_terminal_plan_and_no_layout():
    # The plan's dated BarrierOption valued after its exercise date raises in
    # get_maturity on every operation (Task 1 ruling); a tenor product aged to
    # maturity 0 reaches the short-circuit.
    p = expired_barrier()
    env = env_flat()
    prep = prepare(BarrierPDESolver(PDEParams()), p, env, OutputRequest(Operation.PRICE))
    assert prep.terminal is not None and prep.layout is None
    assert run(prep) == BarrierPDESolver(PDEParams()).price(p, env)
