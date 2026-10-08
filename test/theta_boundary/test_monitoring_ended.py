"""A contract whose monitoring has ended (spec 3.3): every engine prices the remaining contract."""
import numpy as np
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                  DoubleSharkfinOption, EuropeanVanillaOption, OneTouchOption,
                                                  SingleSharkfinOption)
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationType, OptionType,
                                TouchType)

D = 1 / 252
T = 0.5
TB = T - 2 * D
DISC = ObservationType.DISCRETE


def ended(p):
    p.time_shift(2 * D, None, env_const(), keep_landing=True)     # the only record was at D: monitoring has ended
    return p


def vanilla(strike, call, multiplier=1.0):
    return float(BlackScholesEngine().price(
        EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL if call else OptionType.PUT, maturity=TB,
                              contract_multiplier=multiplier), env_const()))


def df():
    return float(env_const().get_discount_factor(TB))


CASES = {
    "uo_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_OUT, maturity=T, rebate=5.0,
                                            participation_rate=0.8, contract_multiplier=10.0,
                                            observation_type=DISC, observation_dates=[D])),
                lambda: 0.8 * vanilla(100.0, True, 10.0)),
    "ui_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_IN, maturity=T, rebate=5.0,
                                            observation_type=DISC, observation_dates=[D])),
                lambda: 5.0 * df()),
    "dko_put": (lambda: ended(DoubleBarrierOption(100.0, OptionType.PUT, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT,
                                                  maturity=T, rebate=5.0, observation_type=DISC,
                                                  observation_dates=[D])),
                lambda: vanilla(100.0, False)),
    "dki_call": (lambda: ended(DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                                                   maturity=T, rebate=5.0, observation_type=DISC,
                                                   observation_dates=[D])),
                 lambda: 5.0 * df()),
    "one_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                               rebate=5.0, touch_type=TouchType.ONE_TOUCH, observation_type=DISC,
                                               observation_dates=[D])),
                  lambda: 0.0),
    "no_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                              rebate=5.0, touch_type=TouchType.NO_TOUCH, observation_type=DISC,
                                              observation_dates=[D])),
                 lambda: 5.0 * df()),
    "double_no_touch": (lambda: ended(DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=T,
                                                           rebate=5.0, touch_type=TouchType.DOUBLE_NO_TOUCH,
                                                           observation_type=DISC, observation_dates=[D])),
                        lambda: 5.0 * df()),
    "double_one_touch": (lambda: ended(DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=T,
                                                            rebate=5.0, touch_type=TouchType.DOUBLE_ONE_TOUCH,
                                                            observation_type=DISC, observation_dates=[D])),
                         lambda: 0.0),
    "sharkfin_call": (lambda: ended(SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                                         maturity=T, participation_rate=0.8, knock_out_rebate=5.0,
                                                         no_hit_rebate=3.0, observation_type=DISC,
                                                         observation_dates=[D])),
                      lambda: 0.8 * (vanilla(100.0, True) - vanilla(130.0, True)) + 3.0 * df()),
    "double_sharkfin_put": (lambda: ended(DoubleSharkfinOption(strike=100.0, option_type=OptionType.PUT,
                                                               lower_barrier=80.0, upper_barrier=130.0, maturity=T,
                                                               participation_rate=0.8, knock_out_rebate=5.0,
                                                               no_hit_rebate=3.0, observation_type=DISC,
                                                               observation_dates=[D])),
                            lambda: 0.8 * (vanilla(100.0, False) - vanilla(80.0, False)) + 3.0 * df()),
}


def engines_for(name):
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
    from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
        SingleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
    from quantark.asset.equity.engine.mc.double_sharkfin_option_mc_engine import DoubleSharkfinOptionMCEngine
    from quantark.asset.equity.engine.mc.single_sharkfin_option_mc_engine import SingleSharkfinOptionMCEngine
    from quantark.asset.equity.param import MCParams
    mc = MCParams(num_paths=2048, time_steps=4, seed=7)
    return {
        "uo_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "ui_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "dko_put": [DoubleBarrierOptionAnalyticalEngine()],
        "dki_call": [DoubleBarrierOptionAnalyticalEngine()],
        "one_touch": [OneTouchAnalyticalEngine()],
        "no_touch": [OneTouchAnalyticalEngine()],
        "double_no_touch": [],
        "double_one_touch": [],
        "sharkfin_call": [SingleSharkfinOptionAnalyticalEngine(), SingleSharkfinOptionMCEngine(params=mc)],
        "double_sharkfin_put": [DoubleSharkfinOptionAnalyticalEngine(), DoubleSharkfinOptionMCEngine(params=mc)],
    }[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_ended_monitoring_prices_the_remaining_contract(name):
    make, truth = CASES[name]
    for engine in engines_for(name):
        assert float(engine.price(make(), env_const())) == pytest.approx(truth(), abs=1e-10), type(engine).__name__


def test_remaining_legs_of_a_knocked_in_barrier_are_the_vanilla():
    from quantark.asset.equity.engine.monitoring_ended import remaining_replication
    make, _ = CASES["ui_call"]
    assert remaining_replication(make(), knocked_in=True).price(env_const()) == pytest.approx(vanilla(100.0, True),
                                                                                          abs=1e-10)


def test_ended_accumulator_prices_its_realized_accrual():
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    from quantark.asset.equity.engine.mc.accumulator_mc_engine import AccumulatorMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import AccumulatorOption

    def make():
        p = AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=T,
                              option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                              observation_dates=[D], past_observations=[(-D, 98.0)])
        return ended(p)
    a = float(AccumulatorAnalyticalEngine().price(make(), env_const()))
    m = float(AccumulatorMCEngine(params=MCParams(num_paths=1024, time_steps=4, seed=3)).price(make(), env_const()))
    assert a == pytest.approx(m, abs=1e-8)


# --- Task 4: PDE, QUAD and vol engines ---------------------------------------------------------------------

def pde_quad_engines(name):
    from pde_routes.cases import surface_const
    from quantark.asset.equity.engine.pde import (BarrierPDESolver, DoubleBarrierPDESolver, DoubleOneTouchPDESolver,
                                                  OneTouchPDESolver)
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
    from quantark.asset.equity.engine.quad.discrete_quad_engine import DiscreteQuadEngine
    from quantark.asset.equity.engine.mc.barrier_vol_mc_engines import LocalVolBarrierMCEngine
    single = [BarrierPDESolver(), LocalVolBarrierPDESolver(local_vol_surface=surface_const()), DiscreteQuadEngine(),
              LocalVolBarrierMCEngine(local_vol_surface=surface_const())]
    return {
        "uo_call": single, "ui_call": single,
        "dko_put": [DoubleBarrierPDESolver(), DiscreteQuadEngine()],
        "dki_call": [DoubleBarrierPDESolver(), DiscreteQuadEngine()],
        "one_touch": [OneTouchPDESolver(), DiscreteQuadEngine()],
        "no_touch": [OneTouchPDESolver(), DiscreteQuadEngine()],
        "double_no_touch": [DoubleOneTouchPDESolver(), DiscreteQuadEngine()],
        "double_one_touch": [DoubleOneTouchPDESolver(), DiscreteQuadEngine()],
        "sharkfin_call": [], "double_sharkfin_put": [],
    }[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_ended_monitoring_on_pde_and_quad(name):
    make, truth = CASES[name]
    for engine in pde_quad_engines(name):
        assert float(engine.price(make(), env_const())) == pytest.approx(truth(), abs=1e-10), type(engine).__name__


def test_ended_monitoring_greeks_are_the_legs():
    from quantark.asset.equity.engine.pde import BarrierPDESolver
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    make, _ = CASES["uo_call"]
    g = GreeksCalculator().calculate(product=make(), pricing_env=env_const(), engine=BarrierPDESolver(),
                                     greeks=["price", "delta", "gamma"])
    v = GreeksCalculator().calculate(product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                                   maturity=TB, contract_multiplier=10.0),
                                     pricing_env=env_const(), engine=BlackScholesEngine(),
                                     greeks=["price", "delta", "gamma"])
    for k in ("price", "delta", "gamma"):
        assert g[k] == pytest.approx(0.8 * v[k], rel=1e-9), k


def test_heston_barrier_engines_refuse_an_ended_schedule():
    from quantark.asset.equity.engine.mc.barrier_vol_mc_engines import HestonBarrierMCEngine
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import HestonBarrierPDESolver
    from quantark.execution.errors import CapabilityError
    from quantark.volmodels.heston import HestonParams
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.6)
    make, _ = CASES["uo_call"]
    for engine in (HestonBarrierPDESolver(hp), HestonBarrierMCEngine(hp)):
        with pytest.raises(CapabilityError, match="monitoring has ended"):
            engine.price(make(), env_const())


@pytest.mark.parametrize("spot", [98.0, 92.0, 106.0])
def test_accumulator_observation_at_the_valuation_instant_is_its_known_settlement(spot):
    """A record at t = 0 (theta's landing record, R1) is observed now: the price of the product with it equals the
    product without it plus that observation's settlement on the spot (a breach: none, the rebate leg pays)."""
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    from quantark.asset.equity.engine.mc.accumulator_mc_engine import AccumulatorMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import AccumulatorOption

    def make(dates):
        return AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=dates[-1],
                                 option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                                 observation_dates=list(dates))
    later = [k / 252 for k in range(1, 20)]
    env = env_const(spot)
    now = make([0.0] + later)
    for engine in (AccumulatorAnalyticalEngine(), AccumulatorMCEngine(params=MCParams(num_paths=4096, time_steps=19,
                                                                                       seed=11))):
        with_now = float(engine.price(now, env))
        if spot >= 105.0:
            assert with_now == pytest.approx(0.0, abs=1e-9)      # TERMINATION knocked out now, no rebate
            continue
        settlement = now.get_observation_payoff(spot)
        without = float(engine.price(make(later), env))
        assert with_now == pytest.approx(without + settlement, rel=1e-9), type(engine).__name__


def _knocked_in_engines():
    from pde_routes.cases import surface_const
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
    from quantark.asset.equity.engine.mc.barrier_vol_mc_engines import LocalVolBarrierMCEngine
    from quantark.asset.equity.engine.pde import BarrierPDESolver
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
    from quantark.asset.equity.param import MCParams
    return [BarrierAnalyticalEngine(), BarrierPDESolver(), LocalVolBarrierPDESolver(local_vol_surface=surface_const()),
            BarrierOptionMCEngine(params=MCParams(num_paths=4096, time_steps=50, seed=3)),
            LocalVolBarrierMCEngine(local_vol_surface=surface_const(), params=MCParams(num_paths=4096, time_steps=50,
                                                                                      seed=3))]


@pytest.mark.parametrize("observation", ["continuous", "record_now"])
def test_a_carried_knock_in_is_priced_as_the_engine_prices_its_own(observation):
    """Theta carries a knock-in decided at t0 (spec R4); the engine must price it at t1 as it priced it at t0 - its
    own triggered knock-in - or theta differences two methods (the PDE's grid vanilla at t0 against the analytic
    replication at t1 moved the byte book's knocked-in theta by 3%). Same spot beyond the barrier, flag or not."""
    def make():
        kw = (dict(observation_type=ObservationType.CONTINUOUS) if observation == "continuous"
              else dict(observation_type=DISC, observation_dates=[0.0, 0.5, 1.0]))
        return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                             barrier_type=BarrierType.UP_IN, maturity=1.0, rebate=2.0, **kw)
    env = env_const(131.0)
    engines = _knocked_in_engines()
    if observation == "record_now":
        from quantark.asset.equity.engine.quad.discrete_quad_engine import DiscreteQuadEngine
        engines.append(DiscreteQuadEngine())
    for engine in engines:
        decided = float(engine.price(make(), env))
        carried = make()
        carried._otc_lifecycle_knocked_in = True
        assert float(engine.price(carried, env)) == pytest.approx(decided, rel=1e-12, abs=1e-12), type(engine).__name__


def _heston_slv_barrier_engines():
    from quantark.asset.equity.engine.mc.barrier_vol_mc_engines import HestonBarrierMCEngine, HestonSLVBarrierMCEngine
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import (HestonBarrierPDESolver,
                                                                          HestonSLVBarrierPDESolver)
    from quantark.asset.equity.param import MCParams
    from quantark.volmodels.heston import HestonParams
    from quantark.volmodels.slv.leverage import LeverageSurface
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    lev = LeverageSurface(time_grid=np.linspace(0, 1, 4), strike_grid=100.0 * np.exp(np.linspace(-1, 1, 5)),
                          leverage_grid=np.ones((4, 5)))
    mc = MCParams(num_paths=4096, time_steps=50, seed=3)
    return [HestonBarrierPDESolver(hp, n_x=64, n_v=24, n_t=40), HestonSLVBarrierPDESolver(hp, lev, n_x=64, n_v=24, n_t=40),
            HestonBarrierMCEngine(hp, mc), HestonSLVBarrierMCEngine(hp, lev, mc)]


@pytest.mark.parametrize("observation", ["continuous", "discrete"])
def test_heston_and_slv_barrier_engines_take_a_carried_knock_in_as_happened(observation):
    """A carried knock-in is an event that happened (user ruling 2026-10-08): the engine prices the knocked-in
    contract by its own path - the one a knock-in breached at the valuation spot takes. A knock-in's value does not
    depend on the barrier level, so the same contract with its up barrier at 99 (breached at spot 100) is that path.
    They refused it before."""
    def make(barrier, carried):
        kw = (dict(observation_type=ObservationType.CONTINUOUS) if observation == "continuous"
              else dict(observation_type=DISC, observation_dates=[0.0, 0.5, 1.0]))
        p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=barrier, barrier_type=BarrierType.UP_IN,
                          maturity=1.0, participation_rate=0.8, **kw)
        if carried:
            p._otc_lifecycle_knocked_in = True
        return p
    env = env_const(100.0)
    for engine in _heston_slv_barrier_engines():
        carried = float(engine.price(make(130.0, True), env))
        assert carried == pytest.approx(float(engine.price(make(99.0, False), env)), rel=1e-12), type(engine).__name__
        assert carried > float(engine.price(make(130.0, False), env)) + 1.0, type(engine).__name__

@pytest.mark.parametrize("make", [
    lambda b, dates: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=b, barrier_type=BarrierType.UP_IN,
                                   maturity=1.0, rebate=2.0, observation_type=DISC, observation_dates=dates),
    lambda b, dates: DoubleBarrierOption(100.0, OptionType.CALL, b, 60.0, DoubleBarrierType.KNOCK_IN, maturity=1.0,
                                         rebate=2.0, observation_type=DISC, observation_dates=dates),
], ids=["single", "double"])
def test_quad_honours_a_carried_knock_in(make):
    """QUAD ignored the flag (a carried knock-in priced as alive, 6.9378 for 9.6011). Its own triggered value at
    the same spot is the knock-in its barrier breaches now; a knock-in's value does not depend on the barrier level."""
    from quantark.asset.equity.engine.quad.discrete_quad_engine import DiscreteQuadEngine
    q, env = DiscreteQuadEngine(), env_const(100.0)
    carried = make(130.0, [0.25, 0.5, 1.0])
    carried._otc_lifecycle_knocked_in = True
    decided = make(99.0, [0.0, 0.25, 0.5, 1.0])
    assert float(q.price(carried, env)) == pytest.approx(float(q.price(decided, env)), rel=1e-12)
