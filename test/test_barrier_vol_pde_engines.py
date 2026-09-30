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
    assert eng.price(prod2, _env()) == abs(3.0 * eng.price(_uo_call(), _env()))


def test_heston_slv_barrier_pde_run_and_reject():
    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    ph = HestonBarrierPDESolver(hp, n_x=160, n_v=48, n_t=80).price(_uo_call(), _env())
    ps = HestonSLVBarrierPDESolver(hp, _unit_lev(), n_x=160, n_v=48, n_t=80).price(_uo_call(), _env())
    assert ph > 0 and ps > 0
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    import pytest
    with pytest.raises(PricingError):
        HestonBarrierPDESolver(hp).price(EuropeanVanillaOption(strike=100., option_type=OptionType.CALL, maturity=1.0), _env())


@pytest.mark.parametrize("params", [
    PDEParams(stepper=CrankNicolson(terminal_damping_steps=2)),
    PDEParams(stepper=CrankNicolson(event_damping_steps=1)),
    PDEParams(stepper=CrankNicolson(event_theta=0.7)),
    PDEParams(event_projection="nodal"),
    PDEParams(event_projection="cell_average"),
    PDEParams(grid=GridConfig(eps_crit=0.001)),
    PDEParams(grid=GridConfig(bounds=(None, 500.0))),
])
def test_lv_barrier_rejects_inapplicable_and_rejected_fields(params):
    with pytest.raises(ValidationError, match="local-vol barrier"):
        LocalVolBarrierPDESolver(params).price(_uo_call(), _env())


def test_lv_barrier_points_budget_and_theta_reach_the_kernel(monkeypatch):
    import quantark.asset.equity.engine.pde.barrier_vol_pde_solvers as mod
    seen = {}
    real = mod.price_barrier_lv_pde

    def spy(*a, **kw):
        seen.update(kw, step_dt=a[5])
        return real(*a, **kw)

    monkeypatch.setattr(mod, "price_barrier_lv_pde", spy)
    LocalVolBarrierPDESolver(PDEParams(
        stepper=CrankNicolson(theta=0.6), cache_enabled=False,
        grid=GridConfig(points=300, time_steps=120)
    )).price(_uo_call(), _env())
    assert seen["n_s"] == 300 and len(seen["step_dt"]) == 120 and seen["theta"] == 0.6


def _disc_call(dates):
    return BarrierOption(strike=100., maturity=1.0, option_type=OptionType.CALL, barrier=130.,
                         barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                         observation_dates=dates)


def test_lv_discrete_observation_dates_are_exact_nodes(monkeypatch):
    import quantark.asset.equity.engine.pde.barrier_vol_pde_solvers as mod
    seen = {}
    real = mod.price_barrier_lv_pde

    def spy(*a, **kw):
        seen["t"] = np.concatenate([[0.0], np.cumsum(a[5])])
        seen["obs"] = kw["observe_steps"]
        return real(*a, **kw)

    monkeypatch.setattr(mod, "price_barrier_lv_pde", spy)
    dates = [0.1234, 0.5, 0.7771]
    LocalVolBarrierPDESolver(PDEParams()).price(_disc_call(dates), _env())
    for d, k in zip(dates, seen["obs"]):
        assert seen["t"][k] == pytest.approx(d, abs=1e-12)


def test_lv_discrete_endpoint_dates_keep_todays_indices(monkeypatch):
    # Review Focus 4: valuation-date and maturity observations are not interior events
    import quantark.asset.equity.engine.pde.barrier_vol_pde_solvers as mod
    seen = {}
    real = mod.price_barrier_lv_pde

    def spy(*a, **kw):
        seen["n"] = len(a[5])
        seen["obs"] = kw["observe_steps"]
        return real(*a, **kw)

    monkeypatch.setattr(mod, "price_barrier_lv_pde", spy)
    LocalVolBarrierPDESolver(PDEParams()).price(_disc_call([0.0, 0.5, 1.0]), _env())
    assert seen["obs"][0] == 0 and seen["obs"][-1] == seen["n"]


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
