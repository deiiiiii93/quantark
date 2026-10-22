"""ADI backward-time exactness: tau never overshoots T, final node lands on T.

Regression cover for the negative-tau kernel bug: ``adi_core.solve`` used to
accumulate ``tau += dt`` with ``dt = T / n_t``, which is not a binary fraction
for most ``n_t``, so the final landing node missed T by an ULP.  The 2D
snowball/phoenix wrappers read calendar time back out as ``T - tau`` and hand
it to the environment's rate curve, which fails closed on negative times, so
an overshoot raised ``ValidationError`` for ordinary grids (``T=1.0, n_t=20``).
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    _Heston2DSnowballPDEBase,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.numerical import is_close
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.4, rho=-0.6)

# (T, n_t) pairs whose naive ``sum(T / n_t)`` overshoots T in float64 — the
# configurations that raised before the fix.  n_t=20 is an ordinary grid.
OVERSHOOTING_GRIDS = [(0.5, 20), (1.0, 20), (2.0, 20), (3.0, 23)]


def _naive_accumulated_tau(T: float, n_t: int) -> float:
    """The pre-fix accumulation, as a witness that these grids really drift."""
    dt = T / max(n_t, 1)
    tau = 0.5 * dt + 0.5 * dt
    for _ in range(n_t - 1):
        tau += dt
    return tau


def _env(rate=0.025, spot=100.0, vol=0.22, q=0.01):
    """Plain FlatRateCurve — fails closed on negative times, by design."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="TEST"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=q),
        valuation_date=datetime(2026, 1, 5),
    )


def _snowball(T: float):
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=T,
        ko_barrier=103.0,
        ki_barrier=75.0,
        ko_rate=0.15,
        num_observations=max(int(round(T * 12)), 1),
        contract_multiplier=10_000.0,
    )


def _phoenix(T: float):
    return create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=T,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=85.0,
        coupon_rate=0.01,
        num_observations=max(int(round(T * 12)), 1),
        memory_coupon=False,
        contract_multiplier=10_000.0,
    )


def _unit_leverage_surface(s0: float, T: float) -> LeverageSurface:
    strikes = s0 * np.exp(np.linspace(-1.0, 1.0, 5))
    return LeverageSurface(
        time_grid=np.linspace(0.0, T, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, 5)),
    )


#: Spatial points for the wrapper-level pricing checks below.
#:
#: These tests are about TIME — that tau never overshoots T — and the spatial
#: axis is scaffolding.  It still has to clear ``GridConfig.eps_crit``, which
#: bounds the achieved node spacing near the barriers.  The log-space domain
#: widens like sigma*sqrt(T), so holding that spacing fixed needs points
#: growing like sqrt(T); a constant count starves the long tenors and raises
#: "achieved spacing ... exceeds 2x target eps_crit" — a SPATIAL complaint that
#: says nothing about the temporal defect under test.
#:
#: Measured minima for these fixtures are 60/60/70/80 at T=0.5/1/2/3, so the
#: rule below clears every one with margin while keeping the short tenors cheap.
def _n_x_for(T: float) -> int:
    return int(np.ceil(60.0 * np.sqrt(max(T, 1.0))))


# ---------------------------------------------------------------- kernel level

@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_chosen_grids_really_drift_under_naive_accumulation(T, n_t):
    """Guard the fixture: these grids must exercise the overshoot."""
    assert _naive_accumulated_tau(T, n_t) > T


@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
@pytest.mark.parametrize("rannacher", [True, False])
@pytest.mark.parametrize("scheme", [ADIScheme.DOUGLAS, ADIScheme.CRAIG_SNEYD])
def test_step_hook_tau_never_exceeds_maturity(T, n_t, rannacher, scheme):
    """No consumer may ever observe tau > T, and the last node must BE T."""
    core = HestonSLVADICore(100.0, 100.0, T, 0.025, 0.01, P, 40, 20, n_t)
    seen: list[float] = []

    def hook(U, tau):
        seen.append(float(tau))
        return U

    core.solve(True, scheme, 0.5, rannacher, step_hook=hook)

    assert seen, "step_hook was never called"
    assert max(seen) <= T, f"tau overshot T by {max(seen) - T:.3e}"
    assert seen[-1] == T, f"final node tau={seen[-1]!r} is not exactly T={T!r}"
    assert seen == sorted(seen), "tau must advance monotonically"
    # T - tau is what the wrappers hand to the rate curve.
    assert min(T - t for t in seen) >= 0.0


@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_final_node_calendar_time_is_exactly_zero(T, n_t):
    """The valuation-date readout node must sit at calendar time 0.0 exactly."""
    core = HestonSLVADICore(100.0, 100.0, T, 0.025, 0.01, P, 40, 20, n_t)
    taus: list[float] = []
    core.solve(True, ADIScheme.CRAIG_SNEYD, 0.5, True,
               step_hook=lambda U, tau: (taus.append(float(tau)), U)[1])
    assert T - taus[-1] == 0.0


def test_single_step_grid_lands_on_maturity():
    """n_t=1: the Rannacher pair alone must land the final node on T."""
    core = HestonSLVADICore(100.0, 100.0, 0.3, 0.025, 0.01, P, 40, 20, 1)
    taus: list[float] = []
    core.solve(True, ADIScheme.DOUGLAS, 0.5, True,
               step_hook=lambda U, tau: (taus.append(float(tau)), U)[1])
    assert taus[-1] == 0.3


def test_step_count_is_unchanged_by_the_clamp():
    """The clamp must not swallow or add steps: n_t landings after tau=0."""
    n_t = 20
    for rannacher in (True, False):
        core = HestonSLVADICore(100.0, 100.0, 1.0, 0.025, 0.01, P, 40, 20, n_t)
        taus: list[float] = []
        core.solve(True, ADIScheme.DOUGLAS, 0.5, rannacher,
                   step_hook=lambda U, tau: (taus.append(float(tau)), U)[1])
        # tau=0 terminal call, then one landing per step (Rannacher splits the
        # first step into two half-steps, so it reports one extra node).
        expected = n_t + 1 + (1 if rannacher else 0)
        assert len(taus) == expected


# ------------------------------------------------------------ wrapper level

@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_heston_snowball_prices_on_overshooting_grid(T, n_t):
    """Used to raise ValidationError('Time to maturity must be non-negative')."""
    price = HestonSnowballPDESolver(
        model_params=P, params=PDEParams(), n_x=_n_x_for(T), n_v=24, n_t=n_t
    ).price(_snowball(T), _env())
    assert np.isfinite(price)


@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_heston_phoenix_prices_on_overshooting_grid(T, n_t):
    """The phoenix wrapper shares adi_core, so it shared the bug."""
    price = HestonPhoenixPDESolver(
        model_params=P, params=PDEParams(), n_x=_n_x_for(T), n_v=24, n_t=n_t
    ).price(_phoenix(T), _env())
    assert np.isfinite(price)


def test_heston_slv_snowball_prices_on_overshooting_grid():
    T, n_t = 1.0, 20
    price = HestonSLVSnowballPDESolver(
        model_params=P,
        leverage_surface=_unit_leverage_surface(100.0, T),
        params=PDEParams(),
        n_x=_n_x_for(T),
        n_v=24,
        n_t=n_t,
    ).price(_snowball(T), _env())
    assert np.isfinite(price)


@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_valuation_date_event_key_survives_the_snap(T, n_t):
    """The final node must still resolve to the valuation-date event key.

    Valuation-date (t=0) observations are applied by the wrapper's step hook
    keyed on ``round(tau / dt)``, while the event map is keyed on
    ``round((T - obs_time) / dt)``.  Snapping the final tau to exactly T must
    keep those two integers equal to n_t, or a t=0 observation would silently
    stop firing.  ``_hook_tau_key`` additionally rejects any tau more than
    1e-8 of a step off an integer key, so the snapped value has to stay on it.
    """
    dt = T / n_t
    assert _Heston2DSnowballPDEBase._hook_tau_key(T, dt) == n_t
    assert _Heston2DSnowballPDEBase._integer_tau_key(T - 0.0, dt) == n_t


@pytest.mark.parametrize("T,n_t", OVERSHOOTING_GRIDS)
def test_final_node_is_close_to_maturity_for_readout_gating(T, n_t):
    """``is_close(tau, T)`` gates the t=0 readout; the snap must keep it true."""
    assert is_close(T, T)  # sanity: the gate is reflexive
    core = HestonSLVADICore(100.0, 100.0, T, 0.025, 0.01, P, 40, 20, n_t)
    taus: list[float] = []
    core.solve(True, ADIScheme.CRAIG_SNEYD, 0.5, True,
               step_hook=lambda U, tau: (taus.append(float(tau)), U)[1])
    assert is_close(taus[-1], T)
