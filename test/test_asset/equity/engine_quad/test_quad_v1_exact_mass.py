"""QUAD V1 exact-mass diffusion kernel (0.12.1).

V1 samples the Gaussian transition density at lattice offsets and sums the samples with trapezoid weights. By Poisson
summation that lattice sum is the exact integral times Theta(r, phi) = 1 + 2 sum_n e^{-2 pi^2 n^2 r^2} cos(2 pi n phi)
(r = s/h, phi the kernel centre's offset from the lattice, in cells). A step's mass error multiplies the value every
step, so it compounds; dividing each step by Theta makes every step's mass exact.
"""
import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from golden_compare import PINNED_MOVE_REL_TOL

from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.engine.quad.local_step import UnderResolvedStepError
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath, lattice_mass_factor
from quantark.asset.equity.engine.quad.v1_adjoint import forward_pass
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def _brute_theta(r: float, phi: float) -> float:
    """The lattice sum itself, h sum_k g(k h - mu) / integral, by direct summation (|k| <= 2000 cells)."""
    k = np.arange(-2000, 2001, dtype=float)
    return float(np.sum(np.exp(-0.5 * ((k - phi) / r) ** 2)) / (r * math.sqrt(2.0 * math.pi)))


def _tau_for(r: float, h: float) -> float:
    return 0.5 * (r * h) ** 2                   # s = sqrt(2 tau) = r h


@pytest.mark.parametrize("r", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.5, 0.77])
def test_factor_is_the_lattice_sum(r, phi):
    h = 0.0031
    got = lattice_mass_factor(_tau_for(r, h), h, phi)
    assert got == pytest.approx(_brute_theta(r, phi), rel=1e-13, abs=0.0)


@pytest.mark.parametrize("r", [1.40, 1.6, 2.5, 13.0])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.25, 0.5, 0.77])
def test_factor_is_exactly_one_where_the_alias_is_below_rounding(r, phi):
    h = 0.0031
    assert lattice_mass_factor(_tau_for(r, h), h, phi) == 1.0


@pytest.mark.parametrize("r", [0.72, 1.0, 1.3])
@pytest.mark.parametrize("phi", [0.0, 0.1, 0.5, 0.9])
def test_factor_differs_from_one_where_the_alias_survives(r, phi):
    assert abs(math.cos(2.0 * math.pi * phi)) >= 0.5
    h = 0.0031
    assert lattice_mass_factor(_tau_for(r, h), h, phi) != 1.0


def test_factor_depends_only_on_the_fractional_offset():
    h, tau = 0.0031, _tau_for(0.8, 0.0031)
    assert lattice_mass_factor(tau, h, 0.3) == lattice_mass_factor(tau, h, 7.3) == lattice_mass_factor(tau, h, -0.7)


# --- the engine operator (AC2) ----------------------------------------------------------------------------------------

@pytest.fixture
def switch():
    """Set the class switch for one test (spawned workers and rungs follow the class) and restore it."""
    saved = SnowballQuadEngine._exact_lattice_mass

    def _set(on: bool):
        SnowballQuadEngine._exact_lattice_mass = on

    yield _set
    SnowballQuadEngine._exact_lattice_mass = saved


def _operator(r: float, filter_alpha: float):
    math_utils = QuadratureMath(1001, 100.0, 1.0, 0.2, integration_rule="trapezoid", fft_padding_factor=2,
                                fft_filter_alpha=filter_alpha)
    h = float(math_utils.h)
    tau, alpha, beta = _tau_for(r, h), 3.0, 9.25
    omega = np.exp(-(math_utils.z_grid ** 2) / (4.0 * tau) - alpha * math_utils.z_grid)
    base = math.exp(-beta * tau) / math.sqrt(math.pi * tau) / 2.0
    n = math_utils.grid.size

    def diffuse(engine, values):
        prefactor = engine._step_prefactor(base, tau, alpha, h)
        return engine._diffuse_fft(values, math_utils, omega, prefactor, 0, n - 1, (n - 1) % 2, alpha, beta, tau)

    return diffuse, n, math.exp((alpha * alpha - beta) * tau), tau, alpha, h


@pytest.mark.parametrize("r", [0.75, 0.9, 1.2])
def test_a_step_carries_the_exact_transition_mass(r, switch):
    """AC2a: without the spectral filter, rolling a constant back one step returns the exact one-step discount at
    every interior node; the sampled kernel without the factor is off by its alias mass."""
    diffuse, n, exact, _, _, _ = _operator(r, filter_alpha=0.0)
    interior = slice(int(50 * r) + 5, n - int(50 * r) - 5)
    switch(True)
    on = diffuse(SnowballQuadEngine(QuadParams()), np.ones(n))[interior]
    np.testing.assert_allclose(on, exact, rtol=1e-13, atol=0.0)
    switch(False)
    off = diffuse(SnowballQuadEngine(QuadParams()), np.ones(n))[interior]
    _, _, _, tau, alpha, h = _operator(r, filter_alpha=0.0)
    theta = lattice_mass_factor(tau, h, -2.0 * alpha * tau / h)
    assert theta != 1.0
    np.testing.assert_allclose(off / exact, theta, rtol=1e-13, atol=0.0)        # the uncorrected error IS the alias


@pytest.mark.parametrize("r", [0.75, 0.9, 1.2])
def test_under_the_default_filter_the_factor_is_exactly_one_over_theta(r, switch):
    """AC2b: the spectral filter's zero-frequency gain is 1, so on the default operator the correction is exactly the
    scalar 1/Theta on the lattice part (interior, where the analytic tail is zero)."""
    diffuse, n, _, tau, alpha, h = _operator(r, filter_alpha=12.0)
    interior = slice(200, n - 200)
    values = np.cos(np.linspace(0.0, 3.0, n)) + 2.0
    switch(True)
    on = diffuse(SnowballQuadEngine(QuadParams()), values)[interior]
    switch(False)
    off = diffuse(SnowballQuadEngine(QuadParams()), values)[interior]
    theta = lattice_mass_factor(tau, h, -2.0 * alpha * tau / h)
    np.testing.assert_allclose(on * theta, off, rtol=1e-14, atol=0.0)


# --- priced contracts (AC3, AC4, AC6, AC7) ----------------------------------------------------------------------------

VAL = date(2026, 9, 11)
QUARTER_HOUR = 1.0 / (365.0 * 96.0)


def _env(vol):
    return PricingEnvironment(spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=datetime(2026, 9, 11))


def _weekdays(years):
    out, d = [], VAL + timedelta(days=1)
    while (d - VAL).days <= round(365 * years):
        if d.weekday() < 5:
            out.append((d - VAL).days / 365.0)
        d += timedelta(days=1)
    return out


def _monthly(years):
    """Monthly observations on trading days: the first weekday on or after each anniversary (the last weekday when
    the final anniversary falls on a weekend)."""
    days = _weekdays(years)
    return sorted({next((t for t in days if t >= k / 12.0 - 1e-12), days[-1])
                   for k in range(1, int(round(12 * years)) + 1)})


def _snowball(ko, ki=None, continuous=False):
    barrier = dict(ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko), ki_barrier=75.0)
    if continuous:
        barrier.update(ki_observation_type=ObservationType.CONTINUOUS)
    else:
        barrier.update(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki))
    return SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko[-1],
                          barrier_config=BarrierConfig(**barrier))


def _daily_ko(years, continuous=False):
    days = _weekdays(years)
    return _snowball(days, None if continuous else [days[-1]], continuous)


#: (product, vol, params, 32001-node reference, 0.12.0 price, bound on |price - reference| / 100). Daily knock-out with
#: knock-in at maturity at default params: the lattice is sized from the knock-in dates only, so the daily steps span
#: 0.77-0.87 cells and their alias mass compounded (0.12.0 errors 2.1e-3, 5.2e-4, 6.2e-5; 3.7e-3 for 64 quarter-hour
#: steps on an explicit 8001-node grid). The 2y vol 10% residual is the 1001-node lattice's ordinary O(h^2) error
#: (-6.4e-5 at 1.25x the nodes, i.e. ~-1.0e-4 at 1001), not mass. References are independent of the fix (every step of
#: a 32001-node lattice is resolved, Theta == 1.0).
CASES = {
    "daily_ko_2y_vol10": (lambda: _daily_ko(2.0), 0.10, None, 101.41963976289662, 101.63337576245549, 1.5e-4),
    "daily_ko_3y_vol20": (lambda: _daily_ko(3.0), 0.20, None, 99.20475081375459, 99.25719086812408, 3e-5),
    "daily_ko_1y_vol05": (lambda: _daily_ko(1.0), 0.05, None, 101.15702928847864, 101.16321079384667, 2e-5),
    "quarter_hour_ki_8001": (
        lambda: _snowball(_monthly(1.0), sorted(set([0.5 + k * QUARTER_HOUR for k in range(64)] + [1.0]))), 0.05,
        QuadParams(grid_points=8001), 101.12422759080776, 101.49409113510632, 1e-6),
}


@pytest.mark.parametrize("name", list(CASES))
def test_compounding_alias_is_gone(name, switch):
    """AC3: the price is within the pinned bound of the resolved reference; with the switch off it is 0.12.0's."""
    build, vol, params, reference, old, bound = CASES[name]
    switch(True)
    price = SnowballQuadEngine(params or QuadParams()).price(build(), _env(vol))
    assert abs(price - reference) / 100.0 <= bound
    switch(False)
    # `old` is a frozen 0.12.0 literal (ARM64): cross-arch drift ~1e-14, so not `==` (golden_compare).
    assert SnowballQuadEngine(params or QuadParams()).price(build(), _env(vol)) == pytest.approx(
        old, rel=PINNED_MOVE_REL_TOL, abs=0.0)


def test_resolved_contracts_are_bit_identical(switch):
    """AC4 probe: a standard snowball (daily knock-in sizes the lattice to >= 2.5 cells per step) never meets a
    factor other than exactly 1.0; nor does the Simpson rule."""
    cases = ((_snowball(_monthly(1.0), _weekdays(1.0)), QuadParams()),
             (_snowball(_monthly(1.0), _weekdays(1.0)), QuadParams(integration_rule="simpson")),
             (_daily_ko(1.0), QuadParams(integration_rule="simpson")))
    for product, params in cases:
        switch(True)
        on = SnowballQuadEngine(params).price(product, _env(0.2))
        switch(False)
        assert SnowballQuadEngine(params).price(product, _env(0.2)) == on


AC6_CASES = {**{name: (case[0], case[1], case[2]) for name, case in CASES.items()},
             "daily_ko_2y_vol10_continuous_ki": (lambda: _daily_ko(2.0, True), 0.10, None)}


@pytest.mark.parametrize("name", list(AC6_CASES))
def test_statistics_and_the_forward_pass_close_on_corrected_steps(name, switch):
    """AC6: on every AC3 contract (its lattice steps corrected) and a continuous-KI one (the bridge transpose), the
    outcome probabilities partition 1 (0.12.0 lost the alias mass: -2.2e-3 at 2y vol 10%), cash reconciles with the
    price, and the transposed march reproduces every backward column."""
    build, vol, params = AC6_CASES[name]
    product = build()
    switch(False)
    old = SnowballQuadEngine(params or QuadParams()).calculate_event_stats(product, _env(vol))
    assert abs(old.probability_residual) > 1e-5
    switch(True)
    engine = SnowballQuadEngine(params or QuadParams())
    stats = engine.calculate_event_stats(product, _env(vol))
    assert abs(stats.probability_residual) <= 1e-11                     # ~500 steps of rounding
    assert abs(stats.reconciliation_error) <= 1e-12 * max(1.0, abs(stats.pv))
    log = engine._last_v1_march
    result = forward_pass(log)
    for key, backward in log.columns.items():
        got = result.sources.get(key, 0.0) if key[0] == "ko" else result.terminal[key[1]]
        assert got == pytest.approx(backward, abs=1e-10), key


def test_routing_and_refusals_are_unchanged(switch):
    """AC7: a contract 0.12.0 refused (daily same-barrier steps narrower than a cell) is refused with the same
    message."""
    messages = []
    for on in (False, True):
        switch(on)
        with pytest.raises(UnderResolvedStepError) as info:
            SnowballQuadEngine(QuadParams()).price(_daily_ko(2.0), _env(0.05))
        messages.append(str(info.value))
    assert messages[0] == messages[1]


def test_the_switch_reaches_auto_converge_rungs(switch, monkeypatch):
    """A switch flipped on an instance reaches every engine its auto_converge ladder builds, on the price path and on
    the statistics path (the accepted rung's candidate): every step prefactor is computed under the instance's value.
    Observed directly, because the accepted rung may resolve every step (Theta == 1.0) and hide a missed copy."""
    seen = []
    original = SnowballQuadEngine._step_prefactor

    def spy(self, prefactor, tau_step, alpha, h):
        seen.append(self._exact_lattice_mass)
        return original(self, prefactor, tau_step, alpha, h)

    monkeypatch.setattr(SnowballQuadEngine, "_step_prefactor", spy)
    switch(True)
    product = _daily_ko(1.0)
    engine = SnowballQuadEngine(QuadParams(auto_converge=True))
    engine._exact_lattice_mass = False
    engine.price(product, _env(0.05))
    assert seen and not any(seen)
    seen.clear()
    engine.calculate_event_stats(product, _env(0.05))
    assert seen and not any(seen)
