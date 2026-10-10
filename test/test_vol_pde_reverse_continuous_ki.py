"""The FIRST_PASSAGE continuous-KI crossing correction on a reverse (up) barrier in the 2D Heston ADI solver (W7b).

``_ContinuousKIFirstPassage._geometry`` reads lambda = d[k*] / a[k*] (the V1 - V0 slope at the barrier) and the
reflection-principle profile G = exp(-2 mu_e a / sig2) [m Phi(m/s) + s phi(m/s)]. Two defects, both reverse-only and both
on the near-zero-variance columns of the 2D surface (sig2 floored at 1e-10 on the v = 0 column):

* the half-cell floor that keeps k* at least half a cell from the barrier measured the cell as a[live[1]] - a[live[0]],
  the two live nodes of lowest index: on an up barrier those are the far edge's nodes, the "cell" was negative and the
  floor never acted. A node a few thousandths of a cell below the barrier then made lambda a step height over a
  near-zero distance, and V0 grew ~x8 per backward step (KO 97 / KI 125 diverged at n_x = 312, 317, 384);
* on an up barrier mu_e < 0, so the reflection factor is exp(+...): a node within ~1e-5 (log) of the barrier overflowed
  it to inf while the bracket underflowed to 0, and inf x 0 = NaN reached V0.

Evidence: docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/ (c03-c08, d01, REPORT.md "W7b").
"""

import math
from datetime import datetime

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.pde import snowball_pde_solver as sps
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver, _ContinuousKIFirstPassage
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

S0, R, VOL, T = 100.0, 0.03, 0.2, 2.0
HP = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)          # W3's Heston
BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)
# ADI_2D_PRODUCTION_ENGINE_CONTROLS (example/mo_volmodels/11_pde_convergence_gate.py) on the medium PV grid
PRODUCTION = dict(
    grid_style="concentrated", v0_boundary="degenerate_pde", variance_grid_mode="auto", v_drift_scheme="auto",
    barrier_greek_steps_per_tick=HestonSnowballPDESolver.PRODUCTION_BARRIER_GREEK_STEPS_PER_TICK,
    greek_min_n_x=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_N_X,
    greek_min_n_v=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_N_V,
    greek_min_steps_per_year=HestonSnowballPDESolver.PRODUCTION_GREEK_MIN_STEPS_PER_YEAR,
    barrier_greek_min_n_x=HestonSnowballPDESolver.PRODUCTION_BARRIER_GREEK_MIN_N_X)
MEDIUM = (200, 60, int(math.ceil(400 * T)))

# Library Monte Carlo references (d01_w7b_mc_references.py: HestonSnowballMCEngine, QE, continuous-KI bridge, 4
# substeps, daily grid, 200,000 antithetic paths x 8 batches, seed 20261010): (value, standard error).
MC_REFERENCE = {125.0: (95.41574689509987, 4.575e-02),
                139.80: (98.2960595799618, 4.356e-02),
                114.45: (94.68248138957318, 4.575e-02)}
# Room for the PDE's own discretization error on top of the MC's 4 standard errors: KI 125 moves 0.012 between
# (600, 48, 192) and (600, 60, 800) (c05) and 0.005 across n_x alignments (c03); 0.02 covers both.
PDE_BUDGET = 0.02


def _env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=S0, asset_name="X"), vol_surface=FlatVolSurface(volatility=VOL),
                              rate_curve=FlatRateCurve(rate=R), valuation_date=datetime(2024, 1, 2))


def _snowball(ki, ko=97.0, reverse=True, continuous=True):
    obs = [(i + 1) / 24 * T for i in range(24)]
    barrier = BarrierConfig(
        ko_barrier=ko, ko_rate=0.15,
        ko_observation_schedule=ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ko) for t in obs]),
        ki_barrier=ki,
        ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
        ki_continuous=continuous,
        ki_observation_schedule=None if continuous else ObservationSchedule(
            records=[ObservationRecord(observation_time=t, barrier=ki) for t in obs]))
    return SnowballOption(initial_price=S0, strike=S0, barrier_config=barrier,
                          payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=True),
                          accrual_config=AccrualConfig(is_annualized_rebate=False),
                          maturity=T, tenor=T, is_reverse=reverse)


_SQRT_2PI = math.sqrt(2.0 * math.pi)


def legacy_geometry(self, dt, mu, sig2, s_vec, barrier):
    """``_ContinuousKIFirstPassage._geometry`` before W7b, verbatim: the floor's cell is a[live[1]] - a[live[0]] and G
    is the plain product. The same-process twin the inertness tests compare against."""
    key = (round(dt, 12), round(mu, 12), round(sig2, 12), round(float(barrier), 12), len(s_vec))
    cached = self._cache.get(key)
    if cached is not None:
        return cached
    result = (None, None, None)
    if dt > 0.0 and sig2 > 0.0 and barrier > 0.0:
        log_s = np.log(np.asarray(s_vec, dtype=float))
        log_b = np.log(float(barrier))
        if self._is_reverse:
            a = log_b - log_s
            mu_e = -mu
        else:
            a = log_s - log_b
            mu_e = mu
        s_step = np.sqrt(sig2 * dt)
        band = np.flatnonzero((a > 0.0) & (a < self._BAND_STDS * s_step))
        if band.size:
            ab = a[band]
            m = mu_e * dt - ab
            t = m / s_step
            bracket = m * ndtr(t) + s_step * np.exp(-0.5 * t * t) / _SQRT_2PI
            G = np.zeros(len(s_vec), dtype=float)
            G[band] = np.exp(-2.0 * mu_e * ab / sig2) * bracket
            live = np.flatnonzero(a > 0.0)
            if live.size > 1:
                resolved = live[a[live] >= 0.5 * (a[live[1]] - a[live[0]])]
                if resolved.size:
                    live = resolved
            k_star = int(live[np.argmin(np.abs(a[live] - s_step))])
            result = (G, k_star, float(a[k_star]))
    self._cache[key] = result
    return result


def _price_2d(n, ki, hp=HP, product=None, **solver_kw):
    solver = HestonSnowballPDESolver(hp, n_x=n[0], n_v=n[1], n_t=n[2], **solver_kw)
    return solver.price(_snowball(ki, **(product or {})), _env())


def _close_to_mc(value, ki):
    mc, se = MC_REFERENCE[ki]
    assert abs(value - mc) <= 4.0 * se + PDE_BUDGET, (value, mc, se)


class TestGeometry:
    """The closed form's two inputs on an up barrier, directly."""

    def _fp(self, sig2):
        return _ContinuousKIFirstPassage(dt=np.array([T / 192]), mu=np.array([R - 0.5 * sig2]),
                                         sig2=np.array([sig2]), is_reverse=True)

    def test_slope_is_read_at_least_half_a_cell_below_an_up_barrier(self):
        """A low-variance column (v = 3e-5: sigma sqrt(dt) ~ 1.8e-5, well under the 0.003 cell) and a node 1e-5
        (0.003 of a cell) below the barrier: the slope must be read on the next node down."""
        b, h, sig2 = 125.0, 0.003, 3e-5
        s = b * np.exp(-1e-5 - h * np.arange(60)[::-1])   # ascending; the live side is below the up barrier
        G, k_star, dist = self._fp(sig2)._geometry(T / 192, R - 0.5 * sig2, sig2, s, b)
        assert G is not None
        assert dist >= 0.5 * h
        assert k_star == len(s) - 2

    def test_reflection_profile_is_finite_on_a_near_zero_variance_column(self):
        b = 125.0
        s = b * np.exp(-np.array([0.5, 0.2, 0.1, 0.05, 5e-6]))   # the last node: exponent 2 |mu| a / sig2 ~ 3000
        G, _, _ = self._fp(1e-10)._geometry(T / 192, R - 0.5e-10, 1e-10, s, b)
        assert G is not None and np.all(np.isfinite(G)) and np.all(G >= 0.0)


class TestReverseContinuousKIIsFinite:
    """The cases that diverged or returned NaN / raised before W7b."""

    @pytest.mark.parametrize("n_x,neighbour", [(312, 313), (317, 318), (384, 382)])
    def test_diverging_grids(self, n_x, neighbour):
        """KO 97 / KI 125 at (n_x, 48, 192): the nearest live node 0.0039 / 0.0182 / 0.0046 of a cell below the
        barrier (it diverged to -1.3e173 / -1.1e30 / -1.7e158). Against the nearest grid whose node sits beyond half a
        cell (W7b leaves it bit for bit): within the n_x-alignment spread c03 measured (95.4333 ... 95.4381)."""
        v = _price_2d((n_x, 48, 192), 125.0)
        assert math.isfinite(v)
        assert v == pytest.approx(_price_2d((neighbour, 48, 192), 125.0), abs=5e-3)
        _close_to_mc(v, 125.0)

    def test_production_greeks_ki_139_80(self):
        """The production Greek grid (300, 135, 3200) puts a node 0.0010 of a cell below KI 139.80: the readout was
        non-finite (PricingError)."""
        solver = HestonSnowballPDESolver(HP, n_x=MEDIUM[0], n_v=MEDIUM[1], n_t=MEDIUM[2], **PRODUCTION)
        g = solver.calculate_greeks(_snowball(139.80), _env())
        assert all(math.isfinite(float(v)) for v in g.values())
        _close_to_mc(float(g["price"]), 139.80)

    def test_production_pv_ki_114_45(self):
        """The production PV grid (200, 60, 800) puts a node 0.0009 of a cell below KI 114.45: the reflection factor
        overflowed on the v = 0 column and the price was NaN."""
        v = _price_2d(MEDIUM, 114.45, **PRODUCTION)
        assert math.isfinite(v)
        _close_to_mc(v, 114.45)


INERT = {
    "2D standard continuous": lambda: _price_2d((384, 48, 192), 75.0, product=dict(ko=103.0, reverse=False)),
    "2D reverse discrete": lambda: _price_2d((384, 48, 192), 125.0, product=dict(continuous=False)),
    "2D reverse BSM limit": lambda: _price_2d((384, 48, 192), 125.0, hp=BSM_LIMIT),
    "2D reverse, node beyond half a cell": lambda: _price_2d((313, 48, 192), 125.0),
    "1D standard continuous": lambda: SnowballPDESolver().price(_snowball(75.0, ko=103.0, reverse=False), _env()),
    "1D reverse continuous": lambda: SnowballPDESolver().price(_snowball(125.0), _env()),
    "1D reverse continuous KI 115": lambda: SnowballPDESolver().price(_snowball(115.0), _env()),
}


@pytest.mark.parametrize("name", sorted(INERT))
def test_the_fix_is_bit_identical_where_the_old_floor_measured_the_right_cell(name, monkeypatch):
    """Same process, two arms: the library's ``_geometry`` and its pre-W7b form. The standard orientation's floor
    already measured the barrier cell, every 1D layout is barrier-aligned, discrete KI takes no correction, the BSM
    limit has no near-zero-variance column, and a node beyond half a cell is untouched by either floor."""
    now = INERT[name]()
    monkeypatch.setattr(sps._ContinuousKIFirstPassage, "_geometry", legacy_geometry)
    before = INERT[name]()
    assert float(now).hex() == float(before).hex(), (name, now, before)


# --- continuous KI on the production Greek policy path (user ruling 2026-10-10) ----------------------------------
# ADI_2D_PRODUCTION_ENGINE_CONTROLS on the medium PV grid; calculate_greeks resolves its own grid through
# greek_time_grid_policy (n_x >= PRODUCTION_GREEK_MIN_N_X = 300, n_v >= 135, 1600 steps a year: (300, 135, 3200) here;
# PRODUCTION_BARRIER_GREEK_MIN_N_X = 600 stays a dense-discrete-KI floor). Reverse KO 97 with KI 114.45 / 125 / 139.80;
# the standard orientation, KO 103, takes the same levels mirrored through the spot (100^2 / KI: 87.37 / 80 / 71.53) --
# a standard knock-in above the spot is knocked in at valuation. SLV runs a unit leverage surface: the Heston dynamics
# through the SLV solver (its barrier leverage and operator), so the Heston MC references apply to it too.

KI_LEVELS = (114.45, 125.0, 139.80)


def _unit_leverage():
    ks = 100.0 * np.exp(np.linspace(-1.5, 1.5, 21))
    return LeverageSurface(time_grid=np.linspace(0.0, T, 9), strike_grid=ks, leverage_grid=np.ones((9, ks.size)))


def _production_solver(model, n_x=MEDIUM[0]):
    if model == "heston":
        return HestonSnowballPDESolver(HP, n_x=n_x, n_v=MEDIUM[1], n_t=MEDIUM[2], **PRODUCTION)
    return HestonSLVSnowballPDESolver(HP, _unit_leverage(), n_x=n_x, n_v=MEDIUM[1], n_t=MEDIUM[2], **PRODUCTION)


def _production_product(orientation, ki):
    if orientation == "reverse":
        return _snowball(ki)
    return _snowball(round(S0 * S0 / ki, 2), ko=103.0, reverse=False)


@pytest.mark.slow
@pytest.mark.parametrize("ki", KI_LEVELS)
@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("orientation", ["reverse", "standard"])
def test_production_continuous_ki_pv_and_greeks(orientation, model, ki):
    solver = _production_solver(model)
    product = _production_product(orientation, ki)
    policy = solver.greek_time_grid_policy(product, _env())
    assert (policy["resolved_n_x"], policy["resolved_n_v"], policy["resolved_n_t"]) == (300, 135, 3200)
    pv = solver.price(product, _env())
    greeks = solver.calculate_greeks(product, _env())
    assert math.isfinite(pv) and all(math.isfinite(float(v)) for v in greeks.values()), (pv, greeks)
    assert float(greeks["price"]) == pytest.approx(pv, abs=PDE_BUDGET)      # the Greek grid refines the PV grid
    if orientation == "reverse":
        _close_to_mc(pv, ki)
        _close_to_mc(float(greeks["price"]), ki)


# n_x around the production defaults: the PV grid's 200 and the Greek grid's 300 (n_x above 300 is the Greek grid
# itself). Every n_x must be finite and agree with the default within the alignment spread.
PV_SWEEP = tuple(range(192, 209, 2))
GREEK_SWEEP = (300, 302, 304, 306, 308, 310)


@pytest.mark.slow
@pytest.mark.parametrize("ki", KI_LEVELS)
@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("orientation", ["reverse", "standard"])
def test_production_pv_n_x_sweep(orientation, model, ki):
    product = _production_product(orientation, ki)
    values = {n_x: _production_solver(model, n_x).price(product, _env()) for n_x in PV_SWEEP}
    assert all(math.isfinite(v) for v in values.values()), values
    assert max(values.values()) - min(values.values()) <= PDE_BUDGET, values
    if orientation == "reverse":
        for v in values.values():
            _close_to_mc(v, ki)


@pytest.mark.slow
@pytest.mark.parametrize("ki", KI_LEVELS)
@pytest.mark.parametrize("model", ["heston", "slv"])
def test_production_greek_n_x_sweep_reverse(model, ki):
    product = _production_product("reverse", ki)
    prices = {}
    for n_x in GREEK_SWEEP:
        g = _production_solver(model, n_x).calculate_greeks(product, _env())
        assert all(math.isfinite(float(v)) for v in g.values()), (n_x, g)
        prices[n_x] = float(g["price"])
        _close_to_mc(prices[n_x], ki)
    assert max(prices.values()) - min(prices.values()) <= PDE_BUDGET, prices
