"""QUAD V1 prices the first step from the valuation instant exactly (sub-resolution run S3b, spec "F").

The first step runs from a point (the spot) to the first event. Within a cell or two of a barrier observed there the
lattice cannot see where the post-event surface jumps inside a cell: a step narrower than a cell was refused, and a
step a few cells wide was priced with errors up to 1.1e-3 (one-day lead) and 1.1e-2 (just above the guard threshold,
vol 60%). F applies the first event exactly at the spot on the pre-event surfaces (Gauss-Legendre on pieces cut at the
barriers, the lattice nodes and standardized points), on every V1 route.

References: ``data/subresolution_v1_fixture.json`` (S1: ``Fref``, the semi-analytic first step on an 8001-node
continuation, half-width its refinement difference to 4001 nodes) and ``data/first_step_refs.json`` (S3b: the same
reference for the cases below -- one-week leads, Phoenix, KO-reset, continuous knock-in, reverse, knock-out disabled
after a knock-in, the switch thresholds -- computed by ``CASES`` with F forced on 8001 and 4001 nodes).
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngine,
    PhoenixQuadEngine,
    SnowballQuadEngine,
)
from quantark.asset.equity.engine.quad import first_step as first_step_module
from quantark.asset.equity.engine.quad.first_step import GaussianFirstStep
from quantark.asset.equity.engine.quad.v1_adjoint import forward_pass
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

DATA = Path(__file__).parent / "data"
FIXTURE = json.loads((DATA / "subresolution_v1_fixture.json").read_text())
REFS_PATH = DATA / "first_step_refs.json"
REFS = json.loads(REFS_PATH.read_text()) if REFS_PATH.exists() else {}
SECOND = 1.0 / (365.0 * 86400.0)
MINUTE = 60.0 * SECOND
DAY = 1.0 / 365.0
QUARTERS = (0.25, 0.5, 0.75, 1.0)


def env(vol=0.2, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def default_h(vol, maturity=1.0):
    return 2.0 * (10.0 * vol * math.sqrt(maturity) + (1.0 + 0.5 * vol * vol) * maturity) / 1000.0


def snowball(ko_times, ki_times, *, ki="discrete", reverse=False, disable=False, ko=103.0, ki_barrier=75.0):
    ki_kw = (dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
             if ki == "continuous" else
             dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                  ki_observation_dates=list(ki_times)))
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko_times[-1], is_reverse=reverse,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.12, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_times), disable_ko_after_ki=disable, **ki_kw),
    )


def phoenix(ko_times, *, memory=False, pay=CouponPayType.INSTANT, ki="discrete"):
    ki_kw = (dict(ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
             if ki == "continuous" else
             dict(ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ko_times)))
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=ko_times[-1], contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_times), **ki_kw),
        coupon_config=CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=0.02, coupon_pay_type=pay,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=memory),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


def ko_reset(pre_times, post_times, *, ki_times=None):
    ki_kw = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if ki_times is None else
             dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_times)))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=post_times[-1], contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(pre_times), ki_barrier=80.0, **ki_kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_dates=list(post_times)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )


def first_ko(lead, **kw):
    return snowball((lead,) + QUARTERS, QUARTERS, **kw)


def first_ki(lead, **kw):
    return snowball(QUARTERS, (lead,) + QUARTERS, **kw)


def _near(level, cells, vol, below=True):
    """A spot ``cells`` default cells from ``level`` (below it, or above)."""
    d = cells * default_h(vol)
    return level * math.exp(-d if below else d)


#: name -> (engine class, product factory, vol, spot); every case's first step is priced by F at default params
CASES = {
    # AC3b: one-week leads near the barrier (S1 distances), knock-out and knock-in first
    **{f"week_{kind}_{vol}_{dist}": (SnowballQuadEngine,
                                     (lambda kind=kind: (first_ko if kind == "ko" else first_ki)(7 * DAY)),
                                     vol, (103.0 * math.exp(-d) if kind == "ko" else 75.0 * math.exp(d)))
       for kind in ("ko", "ki") for vol in (0.05, 0.2, 0.6)
       for dist, d in (("on", 0.0), ("0.5cell", 0.5 * default_h(vol)), ("2cells", 2.0 * default_h(vol)),
                       ("1pct", math.log(1.01)), ("5pct", math.log(1.05)))},
    # AC5: Phoenix (coupon barrier 85, knock-out 103), the first observation at the lead
    "phoenix_5min_ko": (PhoenixQuadEngine, lambda: phoenix((5 * MINUTE,) + QUARTERS), 0.2, _near(103.0, 0.5, 0.2)),
    "phoenix_5min_coupon": (PhoenixQuadEngine, lambda: phoenix((5 * MINUTE,) + QUARTERS), 0.2,
                            _near(85.0, 0.5, 0.2, below=False)),
    "phoenix_memory_5min_coupon": (PhoenixQuadEngine, lambda: phoenix((5 * MINUTE,) + QUARTERS, memory=True), 0.2,
                                   _near(85.0, 0.5, 0.2, below=True)),
    "phoenix_expiry_1d_ko": (PhoenixQuadEngine, lambda: phoenix((DAY,) + QUARTERS, pay=CouponPayType.EXPIRY), 0.2,
                             _near(103.0, 0.5, 0.2)),
    "phoenix_continuous_5min": (PhoenixQuadEngine, lambda: phoenix((5 * MINUTE,) + QUARTERS, ki="continuous"), 0.2,
                                _near(75.0, 0.5, 0.2, below=False)),
    # AC5: KO-reset, the first schedule's first observation (or a knock-in date) at the lead
    "ko_reset_5min_ko": (KOResetSnowballQuadEngine, lambda: ko_reset((5 * MINUTE, 0.25, 0.5), (0.75, 1.0),
                                                                    ki_times=(0.25, 0.5)), 0.2, _near(103.0, 0.5, 0.2)),
    "ko_reset_5min_ki": (KOResetSnowballQuadEngine, lambda: ko_reset((0.25, 0.5), (0.75, 1.0),
                                                                    ki_times=(5 * MINUTE, 0.25, 0.5)), 0.2,
                         _near(80.0, 0.5, 0.2, below=False)),
    # (5% above the continuous level: at half a cell the default lattice's own continuous-knock-in continuation is
    # 1.4e-4 from the 16001-node one, common to every first-step treatment)
    "ko_reset_continuous_1h": (KOResetSnowballQuadEngine, lambda: ko_reset((MINUTE * 60, 0.25, 0.5), (0.75, 1.0)),
                               0.2, 80.0 * 1.05),
    # AC5: continuous knock-in on the first step (bridge from the spot), reverse, knock-out disabled after knock-in
    "continuous_5min_near_ki": (SnowballQuadEngine, lambda: first_ko(5 * MINUTE, ki="continuous"), 0.2,
                                _near(75.0, 0.5, 0.2, below=False)),
    "continuous_1d_near_ki": (SnowballQuadEngine, lambda: first_ko(DAY, ki="continuous"), 0.2,
                              _near(75.0, 2.0, 0.2, below=False)),
    "reverse_5min_ko": (SnowballQuadEngine, lambda: first_ko(5 * MINUTE, reverse=True, ko=97.0, ki_barrier=125.0),
                        0.2, _near(97.0, 0.5, 0.2, below=False)),
    "disable_ko_after_ki_5min": (SnowballQuadEngine,
                                 lambda: snowball((5 * MINUTE,) + QUARTERS, (5 * MINUTE,) + QUARTERS, disable=True),
                                 0.2, _near(103.0, 0.5, 0.2)),
    # the first schedule ends at the first event: the not-knocked-in state holds its maturity seed there (evaluated on
    # the nodes, its redemption's kink at the strike a breakpoint)
    "ko_reset_seeded_1d": (KOResetSnowballQuadEngine, lambda: ko_reset((DAY,), (0.25, 0.5, 0.75, 1.0),
                                                                      ki_times=(DAY,)), 0.2, 100.5),
    # AC5c: just below / above the kernel guard (s/h 0.708: today refused below, priced above; F on both sides) and
    # just below / above S_F (F below, today's operator above)
    **{f"guard_{side}": (SnowballQuadEngine, (lambda r=r: first_ko((r * default_h(0.2) / 0.2) ** 2)), 0.2,
                         _near(103.0, 0.5, 0.2))
       for side, r in (("below", 0.70), ("above", 0.72))},
    **{f"sf_{side}": (SnowballQuadEngine, (lambda r=r: first_ko((r * default_h(0.6) / 0.6) ** 2)), 0.6,
                      _near(103.0, 0.5, 0.6))
       for side, r in (("below", 12.9), ("above", 13.1))},
}


#: AC5c with greeks: just below / above each switch threshold, spot away from the barriers so the bump delta is gated.
#: name -> (engine class, product factory of the lead, lead, vol, spot, theta step). Kernel guard (s/h 0.708, F on both
#: sides); S_F (F below, the lattice above; vol 5% on a half-year note, so 13 cells fit before the next date); the
#: rule-(i) sizing cap (a first knock-in date whose spacing needs more than max_adaptive_grid_points: priced locally
#: on 1001 nodes below, the grid grown above; F on both sides).
_SIZING_LEAD = (2.0 * (10.0 * 0.2 + 1.02) * 2.5 / (0.2 * 5000.0)) ** 2
THRESHOLDS = {
    **{f"thr_guard_{side}": (SnowballQuadEngine, first_ko, (r * default_h(0.2) / 0.2) ** 2, 0.2,
                             103.0 * math.exp(-0.05), 1.0 / (365.0 * 24.0))
       for side, r in (("below", 0.70), ("above", 0.72))},
    **{f"thr_sf_{side}": (SnowballQuadEngine,
                          lambda lead: snowball((lead, 0.25, 0.375, 0.5), (0.25, 0.375, 0.5)),
                          (r * default_h(0.05, 0.5) / 0.05) ** 2, 0.05, 95.0, DAY)
       for side, r in (("below", 12.9), ("above", 13.1))},
    **{f"thr_sizing_{side}": (SnowballQuadEngine, first_ki, _SIZING_LEAD * f, 0.2, 75.0 * math.exp(0.05),
                              1.0 / (365.0 * 24.0))
       for side, f in (("below", 0.96), ("above", 1.04))},
}
#: each threshold case is priced at its point and bumped: spot +-1% (delta), the lead shortened by the theta step
#: (theta: the first step's length moves), vol + 1% (vega: s/h moves with the grid's vol^2 half-width term)
BUMPS = ("base", "spot_up", "spot_dn", "theta", "vega")


def threshold_point(name, bump, *, grid_points=None, force=False):
    cls, make, lead, vol, spot, dt = THRESHOLDS[name]
    params = QuadParams() if grid_points is None else QuadParams(grid_points=grid_points)
    engine = cls(params)
    if force:
        engine._FIRST_STEP_S_F = float("inf")
    product = make(lead - dt if bump == "theta" else lead)
    s = spot * {"spot_up": 1.01, "spot_dn": 0.99}.get(bump, 1.0)
    return engine, product, env(vol + (0.01 if bump == "vega" else 0.0), s)


def solve(name, *, grid_points=None, force=False, spot=None, engine_cls=None):
    cls, make, vol, s = CASES[name]
    params = QuadParams() if grid_points is None else QuadParams(grid_points=grid_points)
    engine = (engine_cls or cls)(params)
    if force:
        engine._FIRST_STEP_S_F = float("inf")
    return engine, make(), env(vol, s if spot is None else spot)


def within(value, ref, hw, rel, floor):
    return abs(value - ref) <= rel * abs(ref) + hw + floor


class TestQuadrature:
    """S1's unit check of the F quadrature: mass, mean and a closed-form barrier split to round-off for every step
    width, with and without lattice nodes inside the Gaussian's support."""

    @pytest.mark.parametrize("n_nodes, h", [(1001, 0.006), (2001, 0.0015)])
    @pytest.mark.parametrize("s", [7.07e-7, 2.8e-6, 3.6e-5, 2.1e-4, 6.2e-4, 2.1e-3, 1e-2])
    def test_mass_mean_and_barrier_split_are_exact(self, n_nodes, h, s):
        grid = (np.arange(n_nodes) - n_nodes // 2) * h + 0.3 * h
        m = 0.37 * s
        b = m + 0.4 * s
        fs = GaussianFirstStep(grid, h, mean=m, std=s, disc=1.0, centres=[0.0], breakpoints=[b])
        ones = np.ones(n_nodes)
        assert fs.expect(fs.lift(ones))[0] == pytest.approx(1.0, abs=1e-13)
        assert fs.expect(fs.lift(grid))[0] == pytest.approx(m, abs=1e-13 * max(1.0, s / h))
        # E[(X - b) 1{X < b}] in closed form: the barrier split
        survive = np.where(fs.y < b, fs.lift(grid - b), 0.0)
        d = (b - m) / s
        exact = (m - b) * ndtr(d) - s * math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
        assert fs.expect(survive)[0] == pytest.approx(exact, rel=1e-12, abs=1e-18)

    def test_several_centres_are_independent(self):
        grid = np.linspace(-1.0, 1.0, 1001)
        h = float(grid[1] - grid[0])
        centres = [0.0, grid[480], grid[530]]
        fs = GaussianFirstStep(grid, h, mean=1e-4, std=3e-3, disc=0.99, centres=centres, breakpoints=[0.01])
        together = fs.expect(fs.lift(np.cos(grid)))
        for k, c in enumerate(centres):
            alone = GaussianFirstStep(grid, h, mean=1e-4, std=3e-3, disc=0.99, centres=[c], breakpoints=[0.01])
            assert together[k] == pytest.approx(alone.expect(alone.lift(np.cos(grid)))[0], rel=1e-14)
        # a smooth function: E[cos(c + X)] = cos(c + m) exp(-s^2/2), up to the cubic interpolant (h^4)
        assert together[1] == pytest.approx(0.99 * math.cos(centres[1] + 1e-4) * math.exp(-0.5 * 9e-6), rel=1e-9)

    @pytest.mark.parametrize("std", [3e-4, 2e-3, 1.5e-2, 6e-2, 2e-1])
    def test_both_layouts_agree_on_a_barrier_split(self, std):
        """Below a cell each centre has its own nodes; from a cell up the centres share the lattice-cell nodes. Both
        give E[1{c + X >= b}] = Phi((c + m - b) / s) to round-off."""
        grid = np.linspace(-1.0, 1.0, 1001)
        h = float(grid[1] - grid[0])
        b = grid[500] + 0.3 * h
        centres = np.array([0.0, grid[499], grid[501], grid[505]])
        fs = GaussianFirstStep(grid, h, mean=2e-5, std=std, disc=1.0, centres=centres, breakpoints=[b])
        got = fs.expect(np.where(fs.y >= b, 1.0, 0.0))
        assert np.allclose(got, ndtr((centres + 2e-5 - b) / std), rtol=0.0, atol=1e-13)

    def test_lift_transpose_is_the_adjoint(self):
        rng = np.random.default_rng(11)
        grid = np.linspace(-0.5, 0.5, 301)
        h = float(grid[1] - grid[0])
        fs = GaussianFirstStep(grid, h, mean=0.0, std=2e-3, disc=1.0, centres=[0.0, grid[10]], breakpoints=[0.001])
        v = rng.normal(size=(2, grid.size))
        d = rng.normal(size=(2, fs.y.size))
        assert np.sum(fs.lift(v) * d) == pytest.approx(np.sum(v * fs.lift_transpose(d)), rel=1e-12)

    def test_touch_is_the_bridge_from_the_spot(self):
        """Down barrier at -0.01: P(min of the bridge from 0 to y <= b) = exp(-2 (y - b)(0 - b) / s^2) above it."""
        grid = np.linspace(-1.0, 1.0, 1001)
        h = float(grid[1] - grid[0])
        fs = GaussianFirstStep(grid, h, mean=0.0, std=0.02, disc=1.0, centres=[0.0], breakpoints=[-0.01],
                               touch=(-0.01, False))
        above = fs.y > -0.01
        touch = fs.P[0]
        assert np.allclose(touch[above], np.exp(-2.0 * (fs.y[above] + 0.01) * 0.01 / 4e-4), rtol=1e-14)
        assert np.all(touch[~above] == 1.0)
        # reflection principle: P(min <= b) = 2 P(X <= b) for a driftless path
        zero, one = np.zeros(fs.y.size), np.ones(fs.y.size)
        assert fs.expect_bridged(zero, one)[0] == pytest.approx(2.0 * ndtr(-0.5), abs=1e-13)

    @pytest.mark.parametrize("std", [3e-4, 2e-2])
    def test_leased_matrices_are_private_to_a_live_quadrature(self, std):
        """``W`` / ``P`` live in storage reused across quadratures (S3c, ``_MatrixPool``): a live quadrature keeps its
        own, and one built on recycled storage equals one built on fresh storage, bit for bit."""
        grid = np.linspace(-1.0, 1.0, 1001)
        h = float(grid[1] - grid[0])
        rng = np.random.default_rng(5)

        def build(shift=0.0):
            return GaussianFirstStep(grid, h, mean=1e-4 + shift, std=std, disc=0.98, centres=[0.0, *grid[480:540]],
                                     breakpoints=[0.01], touch=(-0.02, False))

        first = build()
        w, p = first.W.copy(), first.P.copy()
        v_out, v_in = rng.normal(size=first.y.size), rng.normal(size=first.y.size)
        bridged = first.expect_bridged(v_out, v_in)
        for k in range(6):                     # others are built and collected meanwhile
            other = build(1e-3 * (k + 1))
            other.expect_bridged(rng.normal(size=other.y.size), rng.normal(size=other.y.size))
            del other
        assert np.array_equal(first.W, w) and np.array_equal(first.P, p)
        assert np.array_equal(first.expect_bridged(v_out, v_in), bridged)
        del first
        again = build()                        # on storage the collected quadratures gave back
        assert np.array_equal(again.W, w) and np.array_equal(again.P, p)
        assert np.array_equal(again.expect_bridged(v_out, v_in), bridged)
        pool = first_step_module._POOL
        original = pool._MAX_BYTES
        pool._MAX_BYTES = 0                    # every matrix above the cap: fresh arrays, as before the pool
        try:
            fresh = build()
        finally:
            pool._MAX_BYTES = original
        assert fresh.W.base is None and np.array_equal(fresh.W, w) and np.array_equal(fresh.P, p)
        assert np.array_equal(fresh.expect_bridged(v_out, v_in), bridged)


# --- the engine ----------------------------------------------------------------------------------------------------


def first_step_rungs():
    return [r for r in FIXTURE["rungs"] if r["case"] in ("first_ko", "first_ki")]


def _ladder_product(rung):
    return (first_ko if rung["case"] == "first_ko" else first_ki)(rung["x_years"])


class TestFirstStepLadder:
    """AC3 / AC3b (V1): the S1 first-step rungs -- leads 2e-10 y .. 60 min (refused or past the sizing cap before)
    and one day (priced before, up to 1.1e-3 off) -- at every S1 distance and vol, against S1's ``Fref``."""

    @pytest.mark.parametrize("rung", first_step_rungs(),
                             ids=lambda r: f"{r['case']}-{r['vol']}-{r['rung']}-{r['dist']}")
    def test_price_matches_the_reference(self, rung):
        engine = SnowballQuadEngine(QuadParams())
        value = engine.price(_ladder_product(rung), env(rung["vol"], rung["spot"]))
        assert engine._last_first_step is not None
        assert within(value, rung["ref_price"], rung["hw_price"], 1e-4, 1e-6), (value, rung["ref_price"])

    @pytest.mark.parametrize("rung", [r for r in first_step_rungs() if r["delta_gated"]],
                             ids=lambda r: f"{r['case']}-{r['vol']}-{r['rung']}-{r['dist']}")
    def test_greeks_delta_matches_the_reference(self, rung):
        """The engine's own bump delta (central 1%, ``calculate_greeks``) away from the barrier."""
        delta = SnowballQuadEngine(QuadParams()).calculate_greeks(
            _ladder_product(rung), env(rung["vol"], rung["spot"]))["delta"]
        assert within(delta, rung["ref_delta"], rung["hw_delta"], 1e-3, 1e-4), (delta, rung["ref_delta"])

    @pytest.mark.parametrize("name", [n for n in CASES if n.startswith("week_")])
    def test_one_week_leads_match_the_reference(self, name):
        """AC3b: a one-week lead near the barrier (priced before by the lattice) is within the AC3 tolerance."""
        ref = REFS["cases"][name]
        engine, product, market = solve(name)
        value = engine.price(product, market)
        assert engine._last_first_step is not None
        assert within(value, ref["ref_price"], ref["hw_price"], 1e-4, 1e-6), (value, ref["ref_price"])
        _, vol, spot = CASES[name][2], CASES[name][2], CASES[name][3]
        barrier = 103.0 if "_ko_" in name else 75.0
        if abs(math.log(spot / barrier)) >= 3.0 * max(vol * math.sqrt(7 * DAY), 0.01):
            delta = engine.calculate_greeks(product, market)["delta"]
            assert within(delta, ref["ref_delta"], ref["hw_delta"], 1e-3, 1e-4), (delta, ref["ref_delta"])


ROUTE_CASES = [n for n in CASES if not n.startswith(("week_", "guard_", "sf_"))]


class TestRoutes:
    """AC5 (V1): every engine and route reads the first step the same way."""

    @pytest.mark.parametrize("name", ROUTE_CASES)
    def test_price_matches_the_reference(self, name):
        ref = REFS["cases"][name]
        engine, product, market = solve(name)
        value = engine.price(product, market)
        assert engine._last_first_step is not None
        assert within(value, ref["ref_price"], ref["hw_price"], 1e-4, 1e-6), (value, ref["ref_price"])

    @pytest.mark.parametrize("name", ROUTE_CASES)
    def test_statistics_reconcile_with_the_price(self, name):
        """``pv`` is ``price()`` bit for bit, the columns march through the same exact first step (their cash adds up
        to the price), and the forward adjoint pass reproduces every backward column (AC14)."""
        engine, product, market = solve(name)
        price = engine.price(product, market)
        stats = engine.calculate_event_stats(product, market)
        assert stats.pv == price
        expiry = isinstance(product, PhoenixOption) and product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
        assert abs(stats.reconciliation_error) <= (1e-8 if expiry else 1e-12) * max(1.0, abs(price))
        log = engine._last_v1_march
        assert log.first_step is not None
        result = forward_pass(log)
        for key, mass in result.sources.items():
            assert mass == pytest.approx(log.columns[key], abs=1e-12)
        for name_, mass in result.terminal.items():
            assert mass == pytest.approx(log.columns[("terminal", name_)], abs=1e-11)

    @pytest.mark.parametrize("name", [n for n in ROUTE_CASES if isinstance(REFS["cases"].get(n, {}).get("ref_pki"),
                                                                            float)])
    def test_knock_in_probability_matches_the_reference(self, name):
        """P(knock in before any knock-out) from the adjoint pass, whose first interval starts at the spot."""
        ref = REFS["cases"][name]
        engine, product, market = solve(name)
        stats = engine.calculate_event_stats(product, market)
        assert stats.knock_in.probability == pytest.approx(ref["ref_pki"], abs=1e-4 + ref["hw_pki"])

    @pytest.mark.parametrize("name", [n for n in ROUTE_CASES if CASES[n][0] is SnowballQuadEngine
                                      and not n.startswith("continuous")])
    def test_the_transition_readout_is_the_exact_first_step(self, name):
        """F replaces the readout: both modes give the same price and statistics."""
        engine, product, market = solve(name)
        transition = SnowballQuadEngine(QuadParams(readout="transition"))
        assert transition.price(product, market) == engine.price(product, market)
        a = transition.calculate_event_stats(product, market)
        b = engine.calculate_event_stats(product, market)
        assert np.array_equal(a.observations.probability, b.observations.probability)

    @pytest.mark.parametrize("name", ["phoenix_5min_ko", "phoenix_5min_coupon", "reverse_5min_ko",
                                      "disable_ko_after_ki_5min", "continuous_5min_near_ki", "week_ki_0.2_on"])
    def test_forward_density_probabilities_match_the_stacked_columns(self, name):
        """``event_stats_mode='forward_density'`` seeds its first observation from the same quadrature."""
        engine, product, market = solve(name)
        stacked = engine.calculate_event_stats(product, market)
        forward = CASES[name][0](QuadParams(event_stats_mode="forward_density")).calculate_event_stats(product, market)
        np.testing.assert_allclose(forward.observations.probability, stacked.observations.probability, atol=1e-4)
        assert forward.maturity.probability == pytest.approx(stacked.maturity.probability, abs=1e-4)
        assert forward.ki_settlement.probability == pytest.approx(stacked.ki_settlement.probability, abs=1e-4)

    @pytest.mark.parametrize("name", ["phoenix_5min_ko", "ko_reset_5min_ko", "reverse_5min_ko", "week_ko_0.2_on",
                                      "week_ki_0.05_5pct", "week_ko_0.05_5pct"])
    def test_the_t0_surfaces_agree_with_the_price(self, name):
        """The spot-greeks grid and the recorded t=0 surfaces carry F near the first event's barriers: the nodes around
        the spot hold the price the engine gives at their spots."""
        engine, product, market = solve(name)
        engine.record_backward_grids = True
        engine.price(product, market)
        spots, values = engine._last_spot_greeks_grid
        recorded = engine._backward_grids[0.0]
        j = int(np.argmin(np.abs(np.log(spots / market.spot))))
        for k in (j - 1, j, j + 1):
            at_node = solve(name, spot=float(spots[k]))
            node_price = at_node[0].price(at_node[1], at_node[2])
            assert values[k] == pytest.approx(node_price, rel=1e-12)
        assert np.array_equal(np.asarray(recorded[0]), spots)

    @pytest.mark.parametrize("name", ["thr_guard_below", "thr_sizing_below"])
    def test_the_t0_surface_around_a_spot_away_from_the_barriers_is_the_exact_first_step(self, name):
        """The band also covers the spot's neighbourhood: here the spot is beyond the Gaussian's reach of every
        first-event barrier, and the nodes around it still hold the price the engine gives at their spots."""
        engine, product, market = threshold_point(name, "base")
        engine.price(product, market)
        spots, values = engine._last_spot_greeks_grid
        j = int(np.argmin(np.abs(np.log(spots / market.spot))))
        for k in (j - 1, j, j + 1):
            node_engine, node_product, node_market = threshold_point(name, "base")
            node_market.spot_quote.spot = float(spots[k])
            assert values[k] == pytest.approx(node_engine.price(node_product, node_market), rel=1e-12)


class TestThresholds:
    """AC5c (V1): continuity across the kernel guard (F on both sides) and across S_F (F below, the lattice above),
    each side within the AC3 tolerance of the reference."""

    @pytest.mark.parametrize("name, exact", [("guard_below", True), ("guard_above", True), ("sf_below", True),
                                             ("sf_above", False)])
    def test_both_sides_match_the_reference(self, name, exact):
        ref = REFS["cases"][name]
        engine, product, market = solve(name)
        value = engine.price(product, market)
        assert (engine._last_first_step is not None) == exact
        assert within(value, ref["ref_price"], ref["hw_price"], 1e-4, 1e-6), (value, ref["ref_price"])

    def test_a_first_step_at_or_above_s_f_keeps_today_operator(self):
        """Bitwise: a first step spanning S_F cells or more is the lattice's, as before S3b."""
        engine, product, market = solve("sf_above")
        today = SnowballQuadEngine(QuadParams())
        today._FIRST_STEP_S_F = 0.0
        assert engine.price(product, market) == today.price(product, market)
        assert np.array_equal(engine._last_spot_greeks_grid[1], today._last_spot_greeks_grid[1])

    # CP1: price, bump delta and theta just below / above each switch, with the bumps that move a contract across it.
    # Measured (default engine, 2026-11-23): theta's one-hour shortening of the lead crosses the guard
    # (thr_guard_above -> a local first step) and the sizing cap (thr_sizing_above: 4905 -> 1001 nodes); vega's +1%
    # crosses the guard (below -> above) and S_F (thr_sf_below -> the lattice).

    @pytest.mark.parametrize("name, exact", [("thr_guard_below", True), ("thr_guard_above", True),
                                             ("thr_sf_below", True), ("thr_sf_above", False),
                                             ("thr_sizing_below", True), ("thr_sizing_above", True)])
    @pytest.mark.parametrize("bump", BUMPS)
    def test_price_on_both_sides_matches_the_reference(self, name, exact, bump):
        ref = REFS["thresholds"][name][bump]
        engine, product, market = threshold_point(name, bump)
        value = engine.price(product, market)
        if bump in ("base", "spot_up", "spot_dn"):
            assert (engine._last_first_step is not None) == exact
        hw = abs(ref["ref_price"] - ref["coarse_price"])
        assert within(value, ref["ref_price"], hw, 1e-4, 1e-6), (value, ref["ref_price"])

    @pytest.mark.parametrize("name", list(THRESHOLDS))
    def test_bump_delta_on_both_sides_matches_the_reference(self, name):
        """Central 1% spot bumps (``calculate_greeks``), the spot several step widths from every barrier."""
        refs = REFS["thresholds"][name]
        spot = THRESHOLDS[name][4]

        def diff(key):
            return (refs["spot_up"][key] - refs["spot_dn"][key]) / (0.02 * spot)

        engine, product, market = threshold_point(name, "base")
        delta = engine.calculate_greeks(product, market)["delta"]
        ref, hw = diff("ref_price"), abs(diff("ref_price") - diff("coarse_price"))
        assert within(delta, ref, hw, 1e-3, 1e-4), (delta, ref)

    @pytest.mark.parametrize("name", list(THRESHOLDS))
    def test_theta_on_both_sides_matches_the_reference(self, name):
        """Theta as the first step shortens by the case's step (one hour, or a day). Where that crosses a switch each
        end is the AC price tolerance from its own reference, so theta is bounded by their sum over the step; the
        largest crossing (sizing cap, 4905 -> 1001 nodes) moves the price 9.1e-4 in the hour (1.1e-5 relative)."""
        refs = REFS["thresholds"][name]
        dt = THRESHOLDS[name][5]
        values = {}
        for bump in ("base", "theta"):
            engine, product, market = threshold_point(name, bump)
            values[bump] = engine.price(product, market)
        theta = (values["theta"] - values["base"]) / dt
        ref = (refs["theta"]["ref_price"] - refs["base"]["ref_price"]) / dt
        hw = abs(ref - (refs["theta"]["coarse_price"] - refs["base"]["coarse_price"]) / dt)
        tol = sum(1e-4 * abs(refs[b]["ref_price"]) + 1e-6 for b in ("base", "theta")) / dt
        assert abs(theta - ref) <= 1e-3 * abs(ref) + hw + tol, (theta, ref)

    def test_auto_converge_prices_a_one_day_first_step(self):
        rung = next(r for r in first_step_rungs()
                    if r["case"] == "first_ko" and r["vol"] == 0.2 and r["rung"] == "1d" and r["dist"] == "0.5cell")
        value = SnowballQuadEngine(QuadParams(auto_converge=True)).price(
            _ladder_product(rung), env(rung["vol"], rung["spot"]))
        assert within(value, rung["ref_price"], rung["hw_price"], 1e-4, 1e-6), (value, rung["ref_price"])


class TestTerminalKinks:
    """A state holding its maturity seed at the first event is evaluated on the quadrature nodes: every kink of its
    redemption is a breakpoint, so no Gauss-Legendre piece straddles one."""

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("annualized_ki", [False, True])
    def test_every_kink_of_both_redemptions_is_a_breakpoint(self, reverse, annualized_ki):
        from quantark.asset.equity.product.option.snowball_config import AirbagConfig
        from quantark.util.enum import ProtectionType

        p = SnowballOption(
            initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=1.0, is_reverse=reverse,
            barrier_config=BarrierConfig(ko_barrier=97.0 if reverse else 103.0, ko_rate=0.12,
                                         ko_observation_dates=list(QUARTERS), ki_barrier=125.0 if reverse else 75.0,
                                         ki_observation_type=ObservationType.DISCRETE,
                                         ki_observation_dates=list(QUARTERS)),
            payoff_config=PayoffConfig(call_rebate_enabled=True, call_strike=104.0, call_participation_rate=0.5,
                                       participation_rate=0.8, protection_type=ProtectionType.PARTIAL,
                                       protection_rate=0.12),
            airbag_config=AirbagConfig(airbag_barrier=110.0 if reverse else 90.0, airbag_participation_rate=0.6,
                                       airbag_strike=104.0 if reverse else 96.0),
            accrual_config=AccrualConfig(is_annualized_ki=annualized_ki),
        )
        market = env(0.2, 100.0)
        levels = np.array(SnowballQuadEngine._terminal_levels(p, market))
        spots = np.linspace(40.0, 160.0, 240001)
        for payoff in (p.get_maturity_payoff_v0, p.get_maturity_payoff_v1):
            values = np.array([payoff(float(s), market) for s in spots])
            second = np.abs(np.diff(values, 2))
            kinks = spots[1:-1][second > 1e-9 * max(1.0, float(np.max(np.abs(values))))]
            for k in kinks:
                assert np.min(np.abs(levels - k)) <= 1e-3, (payoff.__name__, k, levels)


class TestKinkAwareInterpolation:
    @pytest.mark.parametrize("offset", [0.0, 0.37])
    def test_a_kink_is_never_interpolated_across(self, offset):
        """A continuous knock-in level joins the not-knocked-in surface to the knocked-in one with a slope jump: the
        stencil keeps to the point's own side, so a piecewise-linear surface is reproduced exactly on both sides."""
        grid = np.linspace(-1.0, 1.0, 401)
        h = float(grid[1] - grid[0])
        kink = grid[200] + offset * h
        fs = GaussianFirstStep(grid, h, mean=0.0, std=0.3 * h, disc=1.0, centres=[kink + 0.2 * h, kink - 0.6 * h],
                               breakpoints=[kink], kinks=[kink])
        surface = np.abs(grid - kink) * 3.0 + np.where(grid > kink, 0.5 * (grid - kink), 0.0)
        exact = np.abs(fs.y - kink) * 3.0 + np.where(fs.y > kink, 0.5 * (fs.y - kink), 0.0)
        assert np.allclose(fs.lift(surface), exact, rtol=0.0, atol=1e-13)
