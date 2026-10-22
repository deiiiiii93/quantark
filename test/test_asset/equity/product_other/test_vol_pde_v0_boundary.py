"""``v0_boundary`` plumbing for the 2D Heston / Heston-SLV autocallable PDE solvers.

The 0.4.0 re-baseline spec (§7A.6) measured a −0.540%-of-notional PDE-vs-MC gap
at Feller ratio ``2*kappa*theta/sigma**2 == 1.0`` under the ADI core's default
``neumann`` treatment of the ``v = 0`` edge, against +0.156% under
``degenerate_pde``.  Because ``enforce_feller=True`` lands constrained fits *on*
that boundary rather than comfortably inside it, the degenerate treatment is the
required default for these solvers — not an opt-in refinement.  The core's own
default stays ``neumann`` (see ``test_adi_degenerate_boundary.py``); the override
lives here, alongside the products that need it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
    HestonSLVPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.product.option.phoenix_config import (
    CouponBarrierConfig,
    CouponPayType,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

# 2*kappa*theta = 0.08 vs sigma**2 = 0.25 -> Feller ratio 0.32, the regime in
# which the v=0 edge treatment actually bites.
FELLER_VIOLATING = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


def _env(vol=0.20, s0=100.0, r=0.03, q=0.01):
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=GridVolSurface(
            strikes, maturities, np.full((len(maturities), len(strikes)), vol)
        ),
        div_yield=ContinuousDividendYield(q),
    )


def _snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _phoenix():
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0,
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def _unit_leverage(s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def _heston_snowball(**kwargs):
    return HestonSnowballPDESolver(model_params=FELLER_VIOLATING, **kwargs)


def _slv_snowball(**kwargs):
    return HestonSLVSnowballPDESolver(
        model_params=FELLER_VIOLATING, leverage_surface=_unit_leverage(), **kwargs
    )


def _heston_phoenix(**kwargs):
    return HestonPhoenixPDESolver(model_params=FELLER_VIOLATING, **kwargs)


def _slv_phoenix(**kwargs):
    return HestonSLVPhoenixPDESolver(
        model_params=FELLER_VIOLATING, leverage_surface=_unit_leverage(), **kwargs
    )


SOLVERS = [
    pytest.param(_heston_snowball, _snowball, id="heston_snowball"),
    pytest.param(_slv_snowball, _snowball, id="slv_snowball"),
    pytest.param(_heston_phoenix, _phoenix, id="heston_phoenix"),
    pytest.param(_slv_phoenix, _phoenix, id="slv_phoenix"),
]


@pytest.mark.parametrize("make_solver, make_product", SOLVERS)
def test_v0_boundary_defaults_to_degenerate_pde(make_solver, make_product):
    assert make_solver().v0_boundary == "degenerate_pde"


@pytest.mark.parametrize("make_solver, make_product", SOLVERS)
def test_default_v0_boundary_reaches_the_adi_core(make_solver, make_product):
    core = make_solver()._make_core(make_product(), _env(), 1.0)

    assert core.v0_boundary == "degenerate_pde"
    assert core._degenerate_v0 is True


@pytest.mark.parametrize("make_solver, make_product", SOLVERS)
def test_neumann_remains_selectable_and_reaches_the_core(make_solver, make_product):
    core = make_solver(v0_boundary="neumann")._make_core(make_product(), _env(), 1.0)

    assert core.v0_boundary == "neumann"
    assert core._degenerate_v0 is False


@pytest.mark.parametrize("make_solver, make_product", SOLVERS)
def test_unknown_v0_boundary_is_rejected_at_construction(make_solver, make_product):
    with pytest.raises(ValidationError, match="v0_boundary"):
        make_solver(v0_boundary="dirichlet")


@pytest.mark.parametrize("make_solver, make_product", SOLVERS)
def test_session_clone_preserves_v0_boundary(make_solver, make_product):
    """The execution session rebuilds these engines from an explicit kwargs
    list, so a constructor argument missing from that list is silently replaced
    by its default -- and the session then prices a different model than the
    engine it was handed, with no error.  ``v0_boundary`` changes the price by
    0.66% of notional in the Feller-violating regime, so this is not cosmetic.
    """
    from quantark.asset.equity.engine.pde.pde_execution_adapters import (
        Heston2DAutocallableSessionAdapter,
    )

    engine = make_solver(v0_boundary="neumann")
    clone = Heston2DAutocallableSessionAdapter()._clone_engine(engine)

    assert clone.v0_boundary == "neumann"


def test_v0_boundary_changes_the_price_when_feller_is_violated():
    """The flag must reach the numerics, not just the constructor.

    Coarse grid on purpose: the point is that the two treatments disagree in
    the Feller-violating regime, which is what makes the default a decision
    rather than a formality.
    """
    product, env = _snowball(), _env()
    # Isolate the boundary-row treatment from the Snowball solver's graded
    # variance-grid default; that grading independently reduces the coarse-grid
    # near-zero error this test is designed to expose.
    grid = dict(n_x=80, n_v=40, n_t=60, v_grid_power=0.0)

    degenerate = _heston_snowball(**grid).price(product, env)
    neumann = _heston_snowball(v0_boundary="neumann", **grid).price(product, env)

    notional = product.initial_price * product.contract_multiplier
    assert abs(degenerate - neumann) / notional > 1e-4
