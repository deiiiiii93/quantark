"""Tier-1 tests for the BasePDESolver grid-layer seam (plan Task 8)."""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde.base_pde_solver import BasePDESolver
from quantark.asset.equity.engine.pde.grid import GridRequest, MarketSnapshot
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError, ValidationError


def create_pricing_env(spot=100.0, vol=0.20, rate=0.05, div_yield=0.02):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


class _SeamProbe(BasePDESolver):
    """Minimal concrete solver exposing the seam without a PDE run."""

    def set_terminal_condition(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError

    def set_boundary_conditions(self, *a, **kw):  # pragma: no cover
        raise NotImplementedError


class _MigratedProbe(_SeamProbe):
    def _uses_grid_layer(self):
        return True

    def grid_request(self, product, market, tau):
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, product.strike),
            critical_prices=(market.spot, product.strike),
            hard_lower=None,
            hard_upper=None,
            event_times=(0.5,),
            day_count=252,
        )


OPTION = EuropeanVanillaOption(
    strike=100.0, option_type=OptionType.CALL, maturity=1.0
)


def test_defaults_unmigrated():
    s = _SeamProbe()
    assert s._uses_grid_layer() is False
    with pytest.raises(NotImplementedError):
        s.grid_request(OPTION, MarketSnapshot(100.0, 0.2, 0.05, 0.02, bus_days_in_year=252), 1.0)


def test_representative_vol_is_strike_selected():
    s = _SeamProbe()
    env = create_pricing_env(vol=0.31)
    assert s.representative_vol(OPTION, env) == pytest.approx(0.31)


def test_market_snapshot_fields():
    s = _SeamProbe()
    env = create_pricing_env(spot=105.0, vol=0.25, rate=0.04, div_yield=0.01)
    m = s.market_snapshot(OPTION, env)
    assert m.spot == 105.0 and m.sigma_ref == pytest.approx(0.25)
    assert m.r_ref == pytest.approx(0.04) and m.q_ref == pytest.approx(0.01)


def test_binder_cache_follows_cache_enabled():
    env = create_pricing_env()
    on = _MigratedProbe(params=PDEParams())
    m = on.market_snapshot(OPTION, env)
    r = on.grid_request(OPTION, m, 1.0)
    assert on.grid_binder.bind(r, m) is on.grid_binder.bind(r, m)

    off = _MigratedProbe(params=PDEParams(cache_enabled=False))
    assert off.grid_binder.bind(r, m) is not off.grid_binder.bind(r, m)


def test_external_layout_check_accepts_and_rejects():
    env = create_pricing_env()
    s = _MigratedProbe()
    m = s.market_snapshot(OPTION, env)
    layout = s.grid_binder.bind(s.grid_request(OPTION, m, 1.0), m)

    s._external_layout_check(OPTION, env, layout)  # exact match passes
    # spot drift within domain still passes (concentration drift allowed)
    s._external_layout_check(OPTION, create_pricing_env(spot=101.0), layout)
    # far spot violates coverage
    with pytest.raises(NumericalError):
        s._external_layout_check(OPTION, create_pricing_env(spot=500.0), layout)

    class _Shifted(_MigratedProbe):
        def grid_request(self, product, market, tau):
            base = super().grid_request(product, market, tau)
            return GridRequest(
                tau=base.tau,
                bound_anchors=base.bound_anchors,
                critical_prices=base.critical_prices,
                hard_lower=None,
                hard_upper=None,
                event_times=(0.25,),  # different schedule,
                day_count=252,
            )

    with pytest.raises(ValidationError):
        _Shifted()._external_layout_check(OPTION, env, layout)


def test_damping_counts_drive_layout_damping():
    """The damping schedule is the stepper's two counts (spec 2 §4.2): the
    route defaults ((1, 1) on the core since flip f), 0 disables, explicit counts flow through."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson, resolve_stepper
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS

    def st(**kw):
        return resolve_stepper(_MigratedProbe(params=PDEParams(stepper=CrankNicolson(**kw))).params,
                               DAMPING_DEFAULTS["core"])

    # the core row's counts (spec 3A §4.2, flip f; 0.7.0 had (1, 2))
    assert (st().terminal_damping_steps, st().event_damping_steps) == (1, 1)
    off = st(terminal_damping_steps=0, event_damping_steps=0)
    assert (off.terminal_damping_steps, off.event_damping_steps) == (0, 0)
    multi = st(terminal_damping_steps=3, event_damping_steps=1)
    assert (multi.terminal_damping_steps, multi.event_damping_steps) == (3, 1)


def test_damping_counts_reach_the_time_layout():
    """End-to-end: disabled damping schedules no damped step on the bound
    layout; multi-step terminal damping schedules that many steps."""
    env = create_pricing_env()

    from quantark.asset.equity.engine.pde.core.components import CrankNicolson, resolve_stepper
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS

    off = _MigratedProbe(
        params=PDEParams(stepper=CrankNicolson(terminal_damping_steps=0, event_damping_steps=0))
    )
    m = off.market_snapshot(OPTION, env)
    layout = off.grid_binder.bind(off.grid_request(OPTION, m, 1.0), m)
    theta_off = resolve_stepper(off.params, DAMPING_DEFAULTS["core"]).schedule(layout)
    assert (theta_off == 0.5).all()

    multi = _MigratedProbe(params=PDEParams(stepper=CrankNicolson(terminal_damping_steps=3)))
    layout3 = multi.grid_binder.bind(multi.grid_request(OPTION, m, 1.0), m)
    theta = resolve_stepper(multi.params, DAMPING_DEFAULTS["core"]).schedule(layout3)
    assert (theta[-3:] == 1.0).all()  # damped steps are backward Euler
