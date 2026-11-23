"""QUAD V2 prices an interval its mesh cannot resolve within ``max_nodes`` locally (sub-resolution run S4, "EaA").

V2 sizes its mesh from the shortest interval after the first; a knock-in minutes after a knock-out used to need more
than ``max_nodes`` nodes and was refused (``ResourceLimitError``). Now, ONLY where today's ``make_mesh`` refuses, the
smallest interval variances are excluded from the sizing in turn until the mesh fits; every excluded interval is
integrated by an exact Gaussian cell kernel (each source cell's polynomial integrated against the density between its
exact limits), and the mesh's cell edges sit on the (at most two) jump breaks of the events that end such intervals,
so the projection of their discontinuities stays exact. Everything V2 priced before is bit-identical (bitwise
priority). The first step from the valuation instant is V2's exact readout, untouched.

Refused with a reason (still ``ResourceLimitError``): more than two jump breaks to align; a locally priced interval
(or a chain of them) between two observations of the same barrier (the mass crossing between them lies in a layer
narrower than a cell); a deterministic (zero-variance) interval alongside.

References: ``data/subresolution_v2_fixture.json`` (S2 demos): the increment reference -- default V2 at the coincident
booking and at resolved gaps, fitted in the gap -- with its half-width; routes against V2 with ``max_nodes`` raised.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.v2 import basis
from quantark.asset.equity.engine.quad.v2.context import interval_moments
from quantark.asset.equity.engine.quad.v2.engine import (
    KOResetSnowballQuadEngineV2,
    PhoenixQuadEngineV2,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.engine.quad.v2.basis import ResourceLimitError
from quantark.asset.equity.engine.quad.v2.operator import Operator
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

FIXTURE = json.loads((Path(__file__).parent / "data" / "subresolution_v2_fixture.json").read_text())
SECOND = 1.0 / (365.0 * 86400.0)
MINUTE = 60.0 * SECOND
DAY = 1.0 / 365.0
QUARTERS = (0.25, 0.5, 0.75, 1.0)
PRICE_REL, PRICE_FLOOR = 1e-4, 1e-6
DELTA_REL, DELTA_FLOOR = 1e-3, 1e-4
PROBABILITY_TOL = 1e-4
BIG = dict(max_nodes=10 ** 7, max_work_bytes=8 * 256 * 1024 ** 2, max_cache_bytes=256 * 1024 ** 2)


def env(vol=0.2, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol), rate_curve=FlatRateCurve(0.03),
        div_yield=ContinuousDividendYield(0.01), valuation_date=datetime(2026, 9, 11),
    )


def snowball(ko_times, ki_times=None, *, ki_barrier=75.0, continuous=False, maturity=None, ko_schedule=None):
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        maturity=maturity if maturity is not None else (ko_times[-1] if ko_times else 1.0),
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12,
            ko_observation_dates=None if ko_schedule is not None else list(ko_times),
            ko_observation_schedule=ko_schedule, ki_barrier=ki_barrier,
            ki_observation_type=ObservationType.CONTINUOUS if continuous else ObservationType.DISCRETE,
            ki_continuous=continuous, ki_observation_dates=None if continuous else list(ki_times),
        ),
    )


def two_event(gap, ki_barrier=75.0):
    return snowball(QUARTERS, (0.25, 0.5 + gap, 0.75, 1.0), ki_barrier=ki_barrier)


def first_step(kind, lead):
    return snowball((lead,) + QUARTERS, QUARTERS) if kind == "ko" else snowball(QUARTERS, (lead,) + QUARTERS)


def schedule(pairs):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=b) for t, b in pairs])


def phoenix(ko_dates, ki_dates):
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(ko_dates), ki_barrier=70.0,
                                     ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(ki_dates)),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0] * len(ko_dates), coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def ko_reset(gap):
    post = tuple(t + gap for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)) + (2.0,)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=2.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule([(t, 103.0) for t in QUARTERS]),
                                     ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule([(t, 80.0) for t in QUARTERS])),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule([(t, 95.0) for t in post])),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True,
                                     coupon_pay_type=CouponPayType.INSTANT),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)


def cki_stepdown(gap):
    """Continuous KI 75; the knock-out level steps from 103 to 101 for one observation ``gap`` after the 0.5 one."""
    return snowball((), continuous=True, maturity=1.0, ko_schedule=schedule(
        [(0.25, 103.0), (0.5, 103.0), (0.5 + gap, 101.0), (0.75, 103.0), (1.0, 103.0)]))


def cki_stepdown_limit():
    """``cki_stepdown`` as the gap vanishes: a knock-out at 103 then at 101 an instant later pays the same coupon as
    one observation at 101."""
    return snowball((), continuous=True, maturity=1.0, ko_schedule=schedule(
        [(0.25, 103.0), (0.5, 101.0), (0.75, 103.0), (1.0, 103.0)]))


def phoenix_three_breaks(gap):
    """KO 105 + coupon 80 at 0.5; KO 106 + coupon 81 + KI 70 at 0.5 + gap: three jump breaks, none shared."""
    ko = schedule([(0.25, 105.0), (0.5, 105.0), (0.5 + gap, 106.0), (0.75, 105.0), (1.0, 105.0)])
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=ko, ki_barrier=70.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.25, 0.5 + gap, 0.75, 1.0]),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0, 80.0, 81.0, 80.0, 80.0], coupon_rate=0.02,
                                          coupon_pay_type=CouponPayType.INSTANT,
                                          day_count_convention=DayCountConvention.ACT_365, memory_coupon=True),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


#: name -> (engine class, product(gap), the gap -> 0 limit booking); every route's end events have at most two jump
#: breaks and no shared barrier
ROUTES = {
    "phoenix_one_break": (PhoenixQuadEngineV2, lambda g: phoenix(QUARTERS, (0.25, 0.5 + g, 0.75, 1.0)), None),
    "ko_reset_short_steps": (KOResetSnowballQuadEngineV2, ko_reset, None),
    "continuous_ki_stepdown": (SnowballQuadEngineV2, cki_stepdown, cki_stepdown_limit),
    # the knocked-in and alive maturity payoffs differ at the barrier; the reflection kernel vanishes there, so the
    # crossing layer the mesh cannot carry costs O(s^2) (measured with S2's patch: 1.2e-6 rel at vol 60%, 5 min)
    "continuous_ki_last_knock_out_before_maturity": (
        SnowballQuadEngineV2, lambda g: snowball((0.25, 0.5, 0.75, 1.0 - g), continuous=True, maturity=1.0), None),
}


def within(value, ref, hw, rel, floor):
    return abs(value - ref) <= rel * abs(ref) + abs(hw) + floor


def bump_delta(engine_class, product, vol, spot, params=None):
    up = engine_class(params or QuadV2Params()).price(product, env(vol, spot * 1.01))
    down = engine_class(params or QuadV2Params()).price(product, env(vol, spot * 0.99))
    return (up - down) / (2.0 * spot * 0.01)


def local_intervals(engine):
    return engine._prepared.diagnostics.get("local_intervals", 0)


def refused_two_event_rungs():
    return [r for r in FIXTURE["rungs"] if r["case"].startswith("two_event") and r["v2_today"] != "priced"]


def rung_id(r):
    return f"{r['case']}-vol{r['vol']}-{r['rung']}" + (f"-{r['dist']}" if r.get("dist") else "")


class TestTwoEventLadder:
    """AC4 (AC2 on V2): every two-event rung V2 refused before prices within tolerance at default params."""

    @pytest.mark.parametrize("rung", refused_two_event_rungs(), ids=rung_id)
    def test_price_and_bump_delta(self, rung):
        barrier = 97.0 if rung["case"] == "two_event_ki97" else 75.0
        product = two_event(rung["x_years"], ki_barrier=barrier)
        engine = SnowballQuadEngineV2()
        price = engine.price(product, env(rung["vol"]))
        assert local_intervals(engine) >= 1
        assert engine._prepared.diagnostics["nodes"] <= QuadV2Params().max_nodes
        assert within(price, rung["ref_price"], rung["hw_price"], PRICE_REL, PRICE_FLOOR), (price, rung["ref_price"])
        delta = bump_delta(SnowballQuadEngineV2, product, rung["vol"], 100.0)
        assert within(delta, rung["ref_delta"], rung["hw_delta"], DELTA_REL, DELTA_FLOOR), (delta, rung["ref_delta"])

    @pytest.mark.parametrize("vol", [0.2, 0.6])
    def test_increment_against_the_coincident_booking_is_the_reference_increment(self, vol):
        """The 2e-10 y rung differs from the coincident booking by the reference's own (tiny) increment, to 1e-9."""
        rung = next(r for r in FIXTURE["rungs"]
                    if r["case"] == "two_event" and r["vol"] == vol and r["rung"] == "2e-10y")
        coincident = SnowballQuadEngineV2().price(two_event(0.0), env(vol))
        local = SnowballQuadEngineV2().price(two_event(2e-10), env(vol))
        increment = rung["ref_price"] - coincident
        assert abs((local - coincident) - increment) <= 1e-9 * coincident + rung["hw_price"]


class TestFirstStepUntouched:
    """AC4 (AC3 on V2): the first step from the valuation instant is V2's exact readout (S2: already exact)."""

    @pytest.mark.parametrize("rung", [r for r in FIXTURE["rungs"] if r["case"] in ("first_ko", "first_ki")],
                             ids=rung_id)
    def test_first_step_rung(self, rung):
        product = first_step("ko" if rung["case"] == "first_ko" else "ki", rung["x_years"])
        engine = SnowballQuadEngineV2()
        price = engine.price(product, env(rung["vol"], rung["spot"]))
        assert local_intervals(engine) == 0
        assert within(price, rung["ref_price"], rung["hw_price"], PRICE_REL, PRICE_FLOOR)
        if rung["delta_gated"]:
            delta = bump_delta(SnowballQuadEngineV2, product, rung["vol"], rung["spot"])
            assert within(delta, rung["ref_delta"], rung["hw_delta"], DELTA_REL, DELTA_FLOOR)


class TestExactKernel:
    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")  # the reference's own round-off notice
    @pytest.mark.parametrize("s_over_h", [1e-4, 0.05, 0.6, 1.9])
    @pytest.mark.parametrize("lo, hi", [(0.0, 1.0), (0.3, 0.8)])
    def test_local_kernel_integrates_cell_polynomials_exactly(self, s_over_h, lo, hi):
        """Each source cell's polynomial against the Gaussian between the exact limits, against an independent
        adaptive quadrature of the same integral (robust at every s/h below the cells' resolution)."""
        from scipy.integrate import quad as adaptive

        mesh = basis.Mesh(0.0, 0.05, 40, 8)
        params = QuadV2Params()
        op = Operator(mesh, params, local_variance=1.0)
        sd = s_over_h * mesh.h
        mean = 0.3 * sd
        kernel, _ = op.kernel(mean, sd * sd, lo, hi)
        band = (len(kernel) - 1) // 2
        theta = mesh.theta
        for d in (-1, 0, 1):
            for q in (0, 3, 7):
                for r in (0, 4, 7):
                    def integrand(t):
                        z = ((d + t - theta[r]) * mesh.h - mean) / sd
                        return float(basis.lagrange(theta, np.array([t]))[q, 0]) * math.exp(-0.5 * z * z) / (
                            math.sqrt(2 * math.pi) * sd) * mesh.h
                    centre = theta[r] - d + mean / mesh.h
                    points = [p for p in (centre - 8 * sd / mesh.h, centre, centre + 8 * sd / mesh.h) if lo < p < hi]
                    expected = adaptive(integrand, lo, hi, points=points or None, limit=400, epsabs=1e-15,
                                        epsrel=1e-13)[0]
                    # kernel entries are O(1); the adaptive reference itself carries ~4e-13 at s/h = 1e-4
                    assert kernel[band + d, q, r] == pytest.approx(expected, abs=1e-12)

    def test_resolved_intervals_keep_todays_kernel_bitwise(self):
        mesh = basis.Mesh(0.0, 0.05, 40, 8)
        params = QuadV2Params()
        today, _ = Operator(mesh, params).kernel(0.001, 0.01)
        split, _ = Operator(mesh, params, local_variance=0.005).kernel(0.001, 0.01)
        assert today.tobytes() == split.tobytes()


def _reference(engine_class, product, vol, *, stats=False):
    params3, params2 = QuadV2Params(cells_per_sd=3.0, **BIG), QuadV2Params(cells_per_sd=2.0, **BIG)
    if stats:
        return engine_class(params3).calculate_event_stats(product, env(vol))
    p3 = engine_class(params3).price(product, env(vol))
    p2 = engine_class(params2).price(product, env(vol))
    return p3, abs(p3 - p2)


def _probabilities(stats):
    out = {}
    for table in ("observations", "maturity", "ki_settlement", "knock_in", "coupons"):
        t = getattr(stats, table, None)
        p = getattr(t, "probability", None) if t is not None else None
        if p is not None:
            out[table] = np.atleast_1d(np.asarray(p, dtype=float))
    return out


class TestRoutes:
    """AC5: Phoenix (one jump break), KO-reset (seven local steps), continuous KI (barrier + one break)."""

    @pytest.mark.parametrize("name", sorted(ROUTES))
    def test_price_and_delta_at_five_minutes(self, name):
        cls, make, _ = ROUTES[name]
        product = make(5 * MINUTE)
        engine = cls()
        price = engine.price(product, env(0.2))
        assert local_intervals(engine) >= 1
        ref, hw = _reference(cls, product, 0.2)
        assert within(price, ref, hw, PRICE_REL, PRICE_FLOOR), (price, ref)
        ref_delta = bump_delta(cls, product, 0.2, 100.0, QuadV2Params(cells_per_sd=3.0, **BIG))
        assert within(bump_delta(cls, product, 0.2, 100.0), ref_delta, 0.0, DELTA_REL, DELTA_FLOOR)

    @pytest.mark.parametrize("name", sorted(ROUTES))
    def test_two_tenths_of_a_nanoyear_is_the_coincident_booking(self, name):
        """The 2e-10 y rung against its gap -> 0 limit booking (a different mesh: the continuous routes carry V2's own
        mesh difference, ~1e-7 rel)."""
        cls, make, limit = ROUTES[name]
        booked = cls().price(make(0.0) if limit is None else limit(), env(0.2))
        assert within(cls().price(make(2e-10), env(0.2)), booked, 0.0, PRICE_REL, PRICE_FLOOR)

    @pytest.mark.parametrize("name", sorted(ROUTES) + ["two_event"])
    def test_statistics_pv_is_price_and_probabilities_match_the_resolved_mesh(self, name):
        cls, make, _ = ROUTES[name] if name != "two_event" else (SnowballQuadEngineV2, two_event, None)
        product = make(5 * MINUTE)
        stats = cls().calculate_event_stats(product, env(0.2))
        price = cls().price(product, env(0.2))
        assert np.float64(stats.pv).tobytes() == np.float64(price).tobytes()
        ref = _probabilities(_reference(cls, product, 0.2, stats=True))
        got = _probabilities(stats)
        assert got.keys() == ref.keys()
        for table in ref:
            assert np.max(np.abs(got[table] - ref[table])) <= PROBABILITY_TOL, table


class TestRefusals:
    """What the local treatment cannot price exactly stays refused, with the reason and the knob."""

    def _refused(self, cls, product, *words):
        with pytest.raises(ResourceLimitError) as refused:
            cls().price(product, env(0.2))
        message = str(refused.value)
        assert "resource limit" in message and "max_nodes" in message
        for word in words:
            assert word in message, message
        return message

    def test_more_than_two_jump_breaks(self):
        self._refused(PhoenixQuadEngineV2, phoenix_three_breaks(5 * MINUTE), "3 distinct value-jump breaks")

    def test_s2_three_break_phoenix_shares_the_knock_in_barrier(self):
        """S2's phx_3brk (KI at 0.5, then KO + coupon + KI 5 min later) is refused by the same-barrier rule first."""
        g = 5 * MINUTE
        product = phoenix((0.25, 0.5 + g, 0.75, 1.0), (0.25, 0.5, 0.5 + g, 0.75, 1.0))
        self._refused(PhoenixQuadEngineV2, product, "same barrier")

    @pytest.mark.parametrize("gap", [2e-10, 5 * MINUTE])
    def test_two_observations_of_the_same_knock_out_barrier(self, gap):
        self._refused(SnowballQuadEngineV2, snowball((0.25, 0.5, 0.5 + gap, 0.75, 1.0), QUARTERS), "same barrier")

    def test_two_observations_of_the_same_knock_in_barrier(self):
        product = snowball(QUARTERS, (0.25, 0.5, 0.5 + 5 * MINUTE, 0.75, 1.0))
        self._refused(SnowballQuadEngineV2, product, "same barrier")

    @pytest.mark.parametrize("vol, second", [(0.6, 103.001), (0.05, 103.0001)])
    def test_two_knock_out_levels_within_the_crossing_layer(self, vol, second):
        """CP2: levels a hair apart cut the same layer (width ~ s of the local intervals between them): measured, the
        second observation's knock-out probability is 1.6e-4 (vol 60%, 103.001) / 2.1e-4 (vol 5%, 103.0001) off."""
        g = 5 * MINUTE
        product = snowball((), ko_schedule=schedule([(0.25, 103.0), (0.5, 103.0), (0.5 + g, second), (0.75, 103.0),
                                                     (1.0, 103.0)]), ki_times=QUARTERS, maturity=1.0)
        with pytest.raises(ResourceLimitError, match="same barrier"):
            SnowballQuadEngineV2().price(product, env(vol))

    def test_same_barrier_across_a_chain_of_local_intervals(self):
        """KI 75 at 0.5, KO 103 at +5 min, KI 75 at +10 min: the second KI's layer crosses the KO event unresolved."""
        g = 5 * MINUTE
        product = snowball((0.25, 0.5 + g, 0.75, 1.0), (0.25, 0.5, 0.5 + 2 * g, 0.75, 1.0))
        self._refused(SnowballQuadEngineV2, product, "same barrier")

    def test_continuous_ki_same_knock_out_barrier(self):
        g = 5 * MINUTE
        product = snowball((0.25, 0.5, 0.5 + g, 0.75, 1.0), continuous=True)
        self._refused(SnowballQuadEngineV2, product, "same barrier")

    def test_a_deterministic_interval_alongside_local_ones(self):
        """A zero-variance interval shifts the grid functions deterministically, moving the local intervals' jump
        breaks off the cell edges: refused (the rule, on a two-event contract whose last interval is made
        deterministic)."""
        from quantark.asset.equity.engine.quad.v2.context import _local_breaks

        contract = SnowballQuadEngineV2()._compile(two_event(5 * MINUTE), env(0.2), False, "before", None)
        _, variances, _ = interval_moments(env(0.2), 100.0, [e.time for e in contract.events])
        threshold = float(np.unique(variances[1:][variances[1:] > 0])[1])
        assert _local_breaks(contract, variances, threshold, {})          # prices with every interval diffusing
        variances = variances.copy()
        variances[-1] = 0.0
        with pytest.raises(ResourceLimitError, match="zero variance"):
            _local_breaks(contract, variances, threshold, {})

    def test_refusal_happens_before_the_large_allocation(self):
        product = snowball((0.25, 0.5, 0.5 + 1e-9, 0.75, 1.0), QUARTERS)
        with pytest.raises(ResourceLimitError) as refused:
            SnowballQuadEngineV2(QuadV2Params(max_nodes=20_000)).price(product, env(0.2))
        assert repr(0.5) in str(refused.value) and repr(0.5 + 1e-9) in str(refused.value)
        assert "event_time_tolerance" in str(refused.value)


def _g_star(product_of_gap, vol, lo=5 * MINUTE, hi=53 * MINUTE):
    """The smallest gap today's mesh sizing fits ``max_nodes`` (bisection on ``make_mesh`` alone)."""
    params = QuadV2Params()

    def fits(gap):
        product = product_of_gap(gap)
        engine = SnowballQuadEngineV2()
        contract = engine._compile(product, env(vol), False, "before", None)
        means, variances, _ = interval_moments(env(vol), product.strike, [e.time for e in contract.events])
        width = engine.spot_width()
        try:
            basis.make_mesh(np.log([100.0 * (1 - width), 100.0 * (1 + width)]), means, variances, params,
                            align=contract.continuous_barrier)
            return True
        except ResourceLimitError:
            return False

    while hi / lo - 1 > 1e-9:
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


class TestContinuityAtTheThreshold:
    """AC5c: price, bump delta and 1-day theta just below (local) and just above (today's mesh) ``max_nodes``'s
    threshold gap g*; the locally priced side is gated against the increment reference, the jump is reported."""

    @pytest.mark.parametrize("vol", [0.2, 0.05, 0.6])
    def test_both_sides_of_g_star(self, vol):
        g = _g_star(two_event, vol)

        def theta_twin(gap):
            q = tuple(t - DAY for t in QUARTERS)
            return snowball(q, (q[0], 0.5 - DAY + gap, q[2], q[3]), maturity=1.0 - DAY)

        sides = {}
        for label, gap in (("below", g * (1 - 1e-6)), ("above", g * (1 + 1e-6))):
            engine = SnowballQuadEngineV2()
            price = engine.price(two_event(gap), env(vol))
            local = local_intervals(engine)
            delta = bump_delta(SnowballQuadEngineV2, two_event(gap), vol, 100.0)
            theta = SnowballQuadEngineV2().price(theta_twin(gap), env(vol)) - price
            sides[label] = (price, delta, theta, local)
        assert sides["below"][3] >= 1 and sides["above"][3] == 0
        # the increment reference at g*: default V2 at the coincident booking and at resolved gaps, fitted in the gap
        gaps = np.array([0.0, DAY, 2 * DAY, 4 * DAY, 8 * DAY])
        prices = np.array([SnowballQuadEngineV2().price(two_event(x), env(vol)) for x in gaps])
        deltas = np.array([bump_delta(SnowballQuadEngineV2, two_event(x), vol, 100.0) for x in gaps])

        def fit(y):
            c = np.linalg.lstsq(np.stack([gaps[1:], gaps[1:] ** 2], 1), y[1:] - y[0], rcond=None)[0]
            return y[0] + c[0] * g + c[1] * g * g

        ref_price, ref_delta = fit(prices), fit(deltas)
        hw_price = abs(fit(prices) - (prices[0] + (prices[1] - prices[0]) / DAY * g))
        below = sides["below"]
        assert within(below[0], ref_price, hw_price, PRICE_REL, PRICE_FLOOR)
        assert within(below[1], ref_delta, 0.0, DELTA_REL, DELTA_FLOOR)
        above = sides["above"]
        # reported (spec AC5c): the jumps across the threshold are within the gates of each side
        assert abs(above[0] - below[0]) <= PRICE_REL * abs(ref_price)
        assert abs(above[1] - below[1]) <= DELTA_REL * abs(ref_delta)
        assert abs(above[2] - below[2]) <= PRICE_REL * abs(ref_price)


class TestChains:
    """AC5d: S2's chains -- 252 daily 5-min local steps (c1) and monthly Phoenix with a daily KI 5 min before each day
    (c3, two jump breaks) price; KI twice a day on one barrier (c2) and KO + coupon + KI at the monthly instants (c4,
    S2: three jump breaks) refuse -- both by the same-barrier rule (c4's monthly KI follows the daily KI 5 min
    earlier)."""

    D = np.arange(1, 253) / 252.0
    G = 5 * MINUTE

    def c1(self):
        return snowball(list(self.D), list(self.D - self.G), maturity=1.0)

    def c2(self):
        return snowball(list(self.D[20::21]), sorted(list(self.D - self.G) + list(self.D)), maturity=1.0)

    def c3(self):
        return phoenix(list(self.D[20::21]), list(self.D - self.G))

    def c4(self):
        return phoenix(list(self.D[20::21]), sorted(list(self.D - self.G) + list(self.D[20::21])))

    @pytest.mark.parametrize("name, cls", [("c1", SnowballQuadEngineV2), ("c3", PhoenixQuadEngineV2)])
    def test_chain_prices(self, name, cls):
        ref = FIXTURE["chains"][name]
        product = getattr(self, name)()
        engine = cls()
        price = engine.price(product, env(0.2))
        assert local_intervals(engine) >= 1
        assert within(price, ref["ref_price"], ref["hw_price"], PRICE_REL, PRICE_FLOOR)
        delta = (cls().price(product, env(0.2, 101.0)) - cls().price(product, env(0.2, 99.0))) / 2.0
        assert within(delta, ref["ref_delta"], ref["hw_delta"], DELTA_REL, DELTA_FLOOR)

    @pytest.mark.parametrize("name, cls, reason", [("c2", SnowballQuadEngineV2, "same barrier"),
                                                   ("c4", PhoenixQuadEngineV2, "same barrier")])
    def test_chain_refuses(self, name, cls, reason):
        with pytest.raises(ResourceLimitError, match=reason):
            cls().price(getattr(self, name)(), env(0.2))


class TestSpotBatchKeepsItsFallback:
    """S2 concern 1: a session whose wide prepare exceeds ``max_nodes`` falls back to scalar contexts, bitwise
    ``price()``; the exclusion rule never engages on the wide prepare."""

    def test_wide_prepare_still_falls_back(self):
        product = two_event(DAY)
        engine = SnowballQuadEngineV2()
        narrow = engine.prepare(product, env(0.2)).diagnostics["nodes"]
        wide = engine.prepare(product, env(0.2), [60.0, 160.0]).diagnostics["nodes"]
        params = QuadV2Params(max_nodes=(narrow + wide) // 2)
        ladder = [70.0, 85.0, 100.0, 115.0, 140.0]
        want = [SnowballQuadEngineV2(params).price(product, env(0.2, s)) for s in ladder]
        session = SnowballQuadEngineV2(params)
        with session.spot_batch((60.0, 160.0)):
            got = [session.price(product, env(0.2, s)) for s in ladder]
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()

    def test_session_prices_a_local_contract_as_the_scalar_engine(self):
        product = two_event(5 * MINUTE)
        ladder = [90.0, 100.0, 110.0]
        want = [SnowballQuadEngineV2().price(product, env(0.2, s)) for s in ladder]
        session = SnowballQuadEngineV2()
        with session.spot_batch((60.0, 160.0)):
            got = [session.price(product, env(0.2, s)) for s in ladder]
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
