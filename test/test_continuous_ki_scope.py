"""What the continuous-KI machinery is allowed to touch.

The FIRST_PASSAGE correction (PDE) and the Brownian-bridge crossing estimate
(MC) exist to price a barrier that is monitored CONTINUOUSLY.  A discretely
monitored knock-in is observed exactly on its own dates, and neither device
should come anywhere near it: a byte compare of 54 engine/product/monitoring
combinations against the tree before this work found every discrete,
European and no-KI price identical to the last bit.

Pinning that as prices would be a golden, and a fragile one.  What actually
holds is structural -- the machinery is never even built -- so that is what
these tests assert, together with the positive controls that prove the
assertion has teeth.

The one deliberate exception is ``BGK_APPROXIMATION``, whose entire purpose
is to replace a dense discrete schedule with continuous monitoring at a
shifted barrier.  It is a continuous treatment by choice, and is pinned here
as such so the exception stays visible.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import (
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams

S0 = 100.0
MONTHLY = [i / 12.0 for i in range(1, 13)]


def _env():
    strikes = list(S0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=S0),
        vol_surface=GridVolSurface(
            strikes, maturities, np.full((len(maturities), len(strikes)), 0.20)
        ),
        div_yield=ContinuousDividendYield(0.01),
    )


def _snowball(**ki):
    return SnowballOption(
        initial_price=S0, strike=S0, maturity=1.0, contract_multiplier=10_000.0,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            **ki,
        ),
    )


DISCRETE_KI = dict(
    ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
    ki_observation_dates=MONTHLY, ki_continuous=False,
)
EUROPEAN_KI = dict(
    ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
    ki_observation_dates=[1.0], ki_continuous=False,
)
NO_KI: dict = {}
CONTINUOUS_KI = dict(
    ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS,
    ki_continuous=True,
)

_HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _pde_solvers():
    return [
        SnowballPDESolver(PDEParams()),
        LocalVolSnowballPDESolver(PDEParams()),
        HestonSnowballPDESolver(_HESTON, n_x=96, n_v=24, n_t=48),
    ]


def _mc_engines():
    params = MCParams(num_paths=512, time_steps=26, seed=5)
    return [
        SnowballMCEngine(params=params),
        LocalVolSnowballMCEngine(params=params),
        HestonSnowballMCEngine(model_params=_HESTON, params=params),
    ]


@pytest.mark.parametrize(
    "monitoring", [DISCRETE_KI, EUROPEAN_KI, NO_KI], ids=["discrete", "european", "no_ki"]
)
def test_the_pde_first_passage_state_is_never_built_off_continuous_monitoring(monitoring):
    product, env = _snowball(**monitoring), _env()
    for solver in _pde_solvers():
        solver.price(product, env)
        assert solver._ki_fp is None, type(solver).__name__


def test_the_pde_first_passage_state_is_built_for_continuous_monitoring():
    """The control: without this the assertions above would pass vacuously."""
    product, env = _snowball(**CONTINUOUS_KI), _env()
    for solver in _pde_solvers():
        solver.price(product, env)
        assert solver._ki_fp is not None, type(solver).__name__


@pytest.mark.parametrize(
    "monitoring", [DISCRETE_KI, EUROPEAN_KI, NO_KI], ids=["discrete", "european", "no_ki"]
)
def test_the_mc_bridge_is_never_armed_off_continuous_monitoring(monitoring):
    product, env = _snowball(**monitoring), _env()
    for engine in _mc_engines():
        engine.price(product, env)
        assert engine._ki_bridge_wanted is False, type(engine).__name__
        assert getattr(engine, "_step_log_variance", None) is None


def test_the_mc_bridge_is_armed_for_continuous_monitoring():
    product, env = _snowball(**CONTINUOUS_KI), _env()
    for engine in _mc_engines():
        engine.price(product, env)
        assert engine._ki_bridge_wanted is True, type(engine).__name__


def test_bgk_treats_a_discrete_schedule_as_continuous_on_purpose():
    """BGK_APPROXIMATION replaces a dense discrete schedule with continuous
    Brownian-bridge monitoring at a shifted barrier, so it DOES arm the
    correction -- the one discrete-schedule price this work moves."""
    product, env = _snowball(**DISCRETE_KI), _env()
    solver = LocalVolSnowballPDESolver(
        PDEParams(ki_monitoring_mode="bgk_approximation")
    )
    solver.price(product, env)
    assert solver._ki_fp is not None
