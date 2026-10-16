import numpy as np
import pytest
from datetime import datetime

from quantark.param import GridVolSurface, FlatRateCurve, SpotQuote
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.pde import (
    LocalVolBarrierPDESolver, HestonBarrierPDESolver, HestonSLVBarrierPDESolver,
)
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.util.enum import OptionType, BarrierType, ObservationType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.asset.equity.engine.pde.grid.config import GridConfig


def _env(vol=0.20, s0=100., r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    surf = GridVolSurface(strikes, list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), vol))
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
                              spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=ContinuousDividendYield(q))


def _uo_call():
    return BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS)


def _unit_lev(s0=100.):
    ks = np.array(list(s0 * np.exp(np.linspace(-1.2, 1.2, 15))))
    return LeverageSurface(time_grid=np.linspace(0, 1, 6), strike_grid=ks, leverage_grid=np.ones((6, ks.size)))


def test_lv_barrier_pde_matches_analytical():
    px = LocalVolBarrierPDESolver(PDEParams(grid=GridConfig(points=500, time_steps=150))).price(_uo_call(), _env())
    ana = BarrierAnalyticalEngine().price(_uo_call(), _env())
    assert abs(px - ana) < 0.3


def test_participation_scales():
    prod2 = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                          barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS,
                          participation_rate=3.0)
    eng = LocalVolBarrierPDESolver(PDEParams(grid=GridConfig(points=300, time_steps=100)))
    # linear in participation up to the last ULP or two (x86 and ARM round the scaled payoff differently)
    assert eng.price(prod2, _env()) == pytest.approx(abs(3.0 * eng.price(_uo_call(), _env())), rel=1e-12)


def test_heston_slv_barrier_pde_run_and_reject():
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    ph = HestonBarrierPDESolver(hp, n_x=160, n_v=48, n_t=80).price(_uo_call(), _env())
    ps = HestonSLVBarrierPDESolver(hp, _unit_lev(), n_x=160, n_v=48, n_t=80).price(_uo_call(), _env())
    assert ph > 0 and ps > 0
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    import pytest
    with pytest.raises(PricingError):
        HestonBarrierPDESolver(hp).price(EuropeanVanillaOption(strike=100., option_type=OptionType.CALL, maturity=1.0), _env())


def _disc_call(dates):
    return BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                         observation_dates=dates)


def _monthly():
    return _disc_call([k / 12.0 for k in range(1, 13)])


# spec 3C §5.3: the local-vol barrier is a core product - the kernel route's refusals are honoured. Measured on the
# monthly up-and-out call (Task 18): each non-default field moves the price; event_damping_steps=1 and the cell
# average are the core's defaults and leave it unchanged.
@pytest.mark.parametrize("params,moves", [
    (PDEParams(stepper=CrankNicolson(terminal_damping_steps=2)), True),     # +1.3e-6
    (PDEParams(stepper=CrankNicolson(event_damping_steps=1)), False),       # the core default
    (PDEParams(stepper=CrankNicolson(event_theta=0.7)), True),              # -4.0e-6
    (PDEParams(event_projection="nodal"), True),                            # -3.4e-2
    (PDEParams(event_projection="cell_average"), False),                    # the core default
    (PDEParams(grid=GridConfig(eps_crit=0.001)), True),                     # +1.1e-5
    (PDEParams(grid=GridConfig(bounds=(None, 500.0))), True),               # +1.9e-5
])
def test_lv_barrier_honours_the_fields_the_kernel_refused(params, moves):
    base = LocalVolBarrierPDESolver().price(_monthly(), _env())
    assert (LocalVolBarrierPDESolver(params).price(_monthly(), _env()) != base) is moves


def test_lv_barrier_explicit_bound_inside_a_continuous_barrier_is_the_cores_refusal():
    # a continuous barrier is a hard domain edge; an explicit far bound conflicts with it (the core's own rule)
    with pytest.raises(ValidationError, match="hard_upper"):
        LocalVolBarrierPDESolver(PDEParams(grid=GridConfig(bounds=(None, 500.0)))).price(_uo_call(), _env())


def _layout(params, product):
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, bind_layout
    solver, env = LocalVolBarrierPDESolver(params), _env()
    return solver._with_surface(env, lambda: bind_layout(solver, product, env, OutputRequest(Operation.PRICE)))


def test_lv_barrier_prices_on_the_core_march(monkeypatch):
    import quantark.asset.equity.engine.pde.core.prepare as core_prepare
    calls = []
    real = core_prepare.march

    def spy(problem):
        calls.append(problem)
        return real(problem)

    monkeypatch.setattr(core_prepare, "march", spy)
    LocalVolBarrierPDESolver().price(_uo_call(), _env())
    assert len(calls) == 1


def test_lv_barrier_points_budget_and_theta_reach_the_core():
    layout = _layout(PDEParams(grid=GridConfig(points=300, time_steps=120)), _uo_call())
    assert len(layout.spatial.x) == 300 and layout.time.requested_steps == 120
    base = LocalVolBarrierPDESolver().price(_uo_call(), _env())
    assert LocalVolBarrierPDESolver(PDEParams(stepper=CrankNicolson(theta=0.6))).price(_uo_call(), _env()) != base


def test_lv_discrete_observation_dates_are_exact_nodes():
    dates = [0.1234, 0.5, 0.7771]
    t = _layout(PDEParams(), _disc_call(dates)).time.t
    for d in dates:
        assert np.min(np.abs(t - d)) <= 1e-12


def test_lv_discrete_endpoint_dates_price_against_the_recursion():
    # Review Focus 4 (valuation-date and maturity observations): on the core both are the readout's and the terminal's
    # decision (spec 3C decisions 25, 30); the price is the exact recursion's on the constant surface
    from pde_routes.barrier_reference import barrier_reference
    exact = barrier_reference(kind="UO", call=True, strike=100.0, level=130.0, rebate=0.0, at_hit=False,
                              obs=[0.5, 1.0], spots=[100.0], vol=0.20, r=0.03, q=0.01)[100.0][0]
    assert abs(LocalVolBarrierPDESolver().price(_disc_call([0.0, 0.5, 1.0]), _env()) - exact) <= 1e-3


def test_lv_discrete_barrier_matches_analytic_oracle():
    from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
    p = _disc_call([k / 12.0 for k in range(1, 13)])
    px = LocalVolBarrierPDESolver(
        PDEParams(grid=GridConfig(points=600, time_steps=240))
    ).price(p, _env())
    ana = BarrierAnalyticalEngine().price(p, _env())
    # Measured 2026-09-29 (exact event nodes): |PDE - BGK| = 0.1303. The gap is
    # the BGK approximation itself at monthly monitoring, not PDE error: a
    # 1600 x 2400 PDE (every date on a node) sits 0.1310 from BGK and 7.0e-4
    # from this price. Bound = measured gap + margin.
    assert abs(px - ana) < 0.15


# The vol-of-vol -> 0 limit: the variance is deterministic, so the 2D solvers price a 1D local-vol barrier on the core.
DECAYING = HestonParams(v0=0.09, kappa=1.5, theta=0.04, sigma=1e-9, rho=-0.5)
SKEW = LeverageSurface(time_grid=np.array([0.0, 1.0]), strike_grid=np.array([50.0, 80.0, 100.0, 120.0, 160.0]),
                       leverage_grid=np.array([[1.3, 1.1, 1.0, 0.9, 0.8], [1.2, 1.05, 1.0, 0.95, 0.85]]))


def _limit_products():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    lagged = SettlementConvention(lag=0.05, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    return {
        "uo_cont": _uo_call(),
        "uo_monthly": _disc_call([k / 12.0 for k in range(1, 13)]),
        "di_put_rebate": BarrierOption(strike=100., maturity=1.0, option_type=OptionType.PUT, barrier=80.,
                                       barrier_type=BarrierType.DOWN_IN, observation_type=ObservationType.CONTINUOUS,
                                       rebate=1.5),
        "uo_participation_multiplier_lagged": BarrierOption(
            strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130., barrier_type=BarrierType.UP_OUT,
            observation_type=ObservationType.CONTINUOUS, rebate=2.0, participation_rate=0.6, contract_multiplier=10.0,
            settlement_convention=lagged),
    }


@pytest.mark.parametrize("name", list(_limit_products()))
@pytest.mark.parametrize("leverage", [None, SKEW], ids=["heston", "slv"])
def test_the_deterministic_limit_marches_on_the_core(name, leverage):
    from quantark.volmodels.heston.deterministic_limit import DeterministicVarianceLocalVol
    product, env = _limit_products()[name], _env()
    engine = (HestonBarrierPDESolver(DECAYING) if leverage is None
              else HestonSLVBarrierPDESolver(DECAYING, leverage))
    core = LocalVolBarrierPDESolver(local_vol_surface=DeterministicVarianceLocalVol(DECAYING, leverage))
    assert engine.price(product, env) == core.price(product, env)


@pytest.mark.parametrize("product", [
    _uo_call(),
    BarrierOption(strike=100., maturity=1.0, option_type=OptionType.PUT, barrier=80.,
                  barrier_type=BarrierType.DOWN_IN, observation_type=ObservationType.CONTINUOUS),
], ids=["uo_call_130", "di_put_80"])
@pytest.mark.parametrize("make", [
    lambda p: HestonBarrierPDESolver(p),
    lambda p: HestonSLVBarrierPDESolver(p, _unit_lev()),
], ids=["heston", "slv"])
def test_the_flat_deterministic_limit_is_the_black_scholes_barrier(product, make):
    flat = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=1e-9, rho=0.0)
    ana = BarrierAnalyticalEngine().price(product, _env())
    # measured 2026-10-06 on the standard preset: 1.84e-4 (uo) and 3.17e-4 (di); the 1D LV kernel this replaced
    # missed by 3.06e-4 and 4.58e-3
    assert abs(make(flat).price(product, _env()) - ana) < 5e-4


@pytest.mark.parametrize("product", [
    _uo_call(),
    BarrierOption(strike=100., maturity=1.0, option_type=OptionType.PUT, barrier=80.,
                  barrier_type=BarrierType.DOWN_IN, observation_type=ObservationType.CONTINUOUS),
], ids=["uo_call_130", "di_put_80"])
@pytest.mark.parametrize("make", [
    lambda p: HestonBarrierPDESolver(p),
    lambda p: HestonSLVBarrierPDESolver(p, _unit_lev()),
], ids=["heston", "slv"])
def test_the_deterministic_limit_domain_follows_the_model_not_the_environment(product, make):
    """An environment quoting 0.05 must not narrow the grid of a model running at 0.30 (final review I4)."""
    flat = HestonParams(v0=0.09, kappa=1.0, theta=0.09, sigma=1e-9, rho=0.0)
    ana = BarrierAnalyticalEngine().price(product, _env(vol=0.30))
    assert abs(make(flat).price(product, _env(vol=0.05)) - ana) < 5e-4


def test_the_deterministic_limit_keeps_the_2d_solvers_refusals():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    from quantark.execution.errors import CapabilityError
    lagged_hit = BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                               barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS,
                               rebate=2.0, pay_at_hit=True,
                               settlement_convention=SettlementConvention(
                                   lag=0.05, lag_unit=SettlementLagUnit.YEAR_FRACTION))
    with pytest.raises(CapabilityError):
        HestonBarrierPDESolver(DECAYING).price(lagged_hit, _env())
    with pytest.raises(CapabilityError):
        HestonSLVBarrierPDESolver(DECAYING, SKEW).price(lagged_hit, _env())
