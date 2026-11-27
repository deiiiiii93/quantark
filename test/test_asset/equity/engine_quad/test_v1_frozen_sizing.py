"""QUAD V1 bump greeks price every leg on the base lattice (0.16.0 D2, "freeze sizing at base").

Every V1 solve sizes its lattice from the inputs: the node count N (``_resolve_grid_points``, from maturity, vol and
the knock-in spacing, with the 5001-node cap), the domain and so the cell h (``QuadratureMath``'s half-width, from
maturity and vol) and the opt-in cell stretch. A bumped leg that re-sizes mixes the lattice's own O(h^2) change into
the greek: a 1 h theta across the sizing cap read -7.9/y against a reference of 0, and a plain 1y monthly-KO /
daily-KI 1-day theta was 2.7% off. ``create_bump_context`` now returns a clone holding the base sizing record; every
leg of the same contract prices on it, while the time-dependent routing (exact first step, local steps, the kernel
guard and its refusals) is recomputed on the frozen h for each leg.
"""
import copy
import math
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngine,
    PhoenixQuadEngine,
    SnowballQuadEngine,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.engine.quad.local_step import UnderResolvedStepError
from quantark.asset.equity.param import BumpConfig, EngineParams, QuadParams
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option import PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.riskmeasures.greeks.theta_shift import bumped_value, shift_for_theta
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.vol.vol_surface import BlackImpliedVolSurface, TermStructureVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

VAL = datetime(2026, 9, 16)                    # a Wednesday: a 1-day theta drops Thursday's knock-in date
HOUR = 1.0 / (365.0 * 24.0)
QUARTERS = (0.25, 0.5, 0.75, 1.0)
GREEKS = ["price", "delta", "gamma", "vega", "theta", "rho", "dividend_rho"]


class SpotSmile(BlackImpliedVolSurface):
    """A spot-dependent smile, sigma = a + b log(K / S): a spot bump moves the vol, so the base domain is frozen
    against a different vol on the delta legs."""
    is_smile = True

    def __init__(self, a=0.22, b=-0.15):
        self.a, self.b = float(a), float(b)

    def get_vol(self, strike, time_to_maturity, spot):
        return max(0.05, self.a + self.b * math.log(float(strike) / float(spot)))

    def parallel_shifted(self, shift):
        return SpotSmile(self.a + shift, self.b)


def env(vol=0.2, spot=100.0, surface=None, val=VAL):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=surface or FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=val)


def term_env():
    return env(surface=TermStructureVolSurface(times=[0.25, 0.5, 1.0, 2.0], vols=[0.25, 0.22, 0.2, 0.19]))


def weekdays(start, years):
    out, d = [], start + timedelta(days=1)
    while d <= start + timedelta(days=round(365 * years)):
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def months(start, n, step=1):
    out = []
    for k in range(step, n + 1, step):
        d = datetime(start.year + (start.month - 1 + k) // 12, (start.month - 1 + k) % 12 + 1, min(start.day, 28))
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(d)
    return out


def sched(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def dated_snowball(years=1.0, ko_step=1, ki="daily", ki_level=75.0):
    ko_dates = months(VAL, int(round(12 * years)), ko_step)
    if ki == "daily":
        ki_kw = dict(ki_barrier=ki_level, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_schedule=sched([d for d in weekdays(VAL, 2.5) if d <= ko_dates[-1]], ki_level))
    elif ki == "discrete":
        ki_kw = dict(ki_barrier=ki_level, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_schedule=sched([ko_dates[-1]], ki_level))
    else:
        ki_kw = dict(ki_barrier=ki_level, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=VAL, exercise_date=ko_dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=sched(ko_dates, 103.0), **ki_kw),
        payoff_config=PayoffConfig(rebate_rate=0.12, include_principal=True))


def dated_phoenix():
    ko_dates = months(VAL, 12)
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=VAL, exercise_date=ko_dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=sched(ko_dates, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=sched(ko_dates, 75.0)),
        coupon_config=CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=0.10, memory_coupon=True),
        payoff_config=PayoffConfig(include_principal=True))


def dated_ko_reset():
    pre, post = months(VAL, 6), months(VAL, 12)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=VAL, exercise_date=post[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=sched(pre, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=sched(post, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=sched(post, 95.0)),
        payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=True))


def float_snowball(ko_times, ki_times, ki_level=75.0):
    return SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko_times[-1],
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12,
                                                       ko_observation_dates=list(ko_times), ki_barrier=ki_level,
                                                       ki_observation_type=ObservationType.DISCRETE,
                                                       ki_observation_dates=list(ki_times)))


def default_h(vol, maturity=1.0):
    return 2.0 * (10.0 * vol * math.sqrt(maturity) + (1.0 + 0.5 * vol * vol) * maturity) / 1000.0


def calc(**bump):
    return GreeksCalculator(EngineParams(bump_config=BumpConfig(**bump)) if bump else None)


def hexes(values):
    return {k: float(v).hex() for k, v in values.items()}


#: name -> (engine, params, product, env); float.hex of GreeksCalculator's price / delta / gamma / rho / dividend rho
#: at the base commit (31f46955): spot-independent vol, so the frozen lattice is the one every such leg sized anyway.
#: Re-pinned by 0.16.0 T5 (V1 convolutions on fast FFT lengths, owner-signed round-off; max rel move 2.8e-16..4.3e-15
#: price, 1.1e-10 rho -- a bump difference); superseded values (999795df):
#:   snowball_daily_ki: price 0x1.901a9e6b477a0p+6 delta 0x1.aeed3df87ea00p-3 gamma -0x1.6ae800999c000p-5
#:     rho -0x1.96f7c8f51b000p-3 dividend_rho -0x1.e4a58bf787000p-3
#:   snowball_term_vol: price 0x1.8ec5eb8c9ebcep+6 delta 0x1.e4284f10b4900p-3 gamma -0x1.4c1d5eb2da800p-5
#:     rho -0x1.7f35468a49800p-3 dividend_rho -0x1.fa5bbba553800p-3
#:   snowball_continuous_2y: delta 0x1.76ce644b6e574p-2 gamma -0x1.2478b1cc3d174p-6
#:   phoenix: price 0x1.90aa68aaee231p+6 delta 0x1.32e3e27e50800p-3 gamma -0x1.2672b6ec89000p-5
#:     dividend_rho -0x1.85a0ef8c40800p-3
#:   ko_reset: price 0x1.9a7a5194111b2p+6 gamma -0x1.f9d9cd80ea000p-6 rho -0x1.170deed1d9400p-2
#:     dividend_rho -0x1.32ba384a58000p-5
#:   spot smile (test_a_spot_dependent_smile_moves_delta_by_the_frozen_domain): price 0x1.8bee41fe00008p+6
#:     rho -0x1.383027dcda800p-3
SPOT_INDEPENDENT = {
    "snowball_daily_ki": (SnowballQuadEngine, QuadParams(), dated_snowball, env, {
        "price": "0x1.901a9e6b477bep+6", "delta": "0x1.aeed3df87e600p-3", "gamma": "-0x1.6ae800999f000p-5",
        "rho": "-0x1.96f7c8f485000p-3", "dividend_rho": "-0x1.e4a58bf7eb000p-3"}),
    "snowball_term_vol": (SnowballQuadEngine, QuadParams(), dated_snowball, term_env, {
        "price": "0x1.8ec5eb8c9ebbdp+6", "delta": "0x1.e4284f10b4d00p-3", "gamma": "-0x1.4c1d5eb2e3800p-5",
        "rho": "-0x1.7f35468b05000p-3", "dividend_rho": "-0x1.fa5bbba5dd000p-3"}),
    "snowball_continuous_2y": (SnowballQuadEngine, QuadParams(), lambda: dated_snowball(2.0, 3, "continuous"),
                               lambda: env(0.35, 110.0), {
        "price": "0x1.8263bd9b359d6p+6", "delta": "0x1.76ce644b6e4ffp-2", "gamma": "-0x1.2478b1cc3c43ap-6",
        "rho": "-0x1.5ba99a1c86800p-3", "dividend_rho": "-0x1.83850efc23400p-2"}),
    "snowball_cell_stretch": (SnowballQuadEngine, QuadParams(align_cell_stretch=0.02),
                              lambda: float_snowball([t for t in (1 / 12 * k for k in range(1, 13))], [1.0]),
                              lambda: env(0.2, 95.0), None),
    "phoenix": (PhoenixQuadEngine, QuadParams(), dated_phoenix, env, {
        "price": "0x1.90aa68aaee233p+6", "delta": "0x1.32e3e27e50a00p-3", "gamma": "-0x1.2672b6ec8a000p-5",
        "rho": "-0x1.e28b7099bf800p-3", "dividend_rho": "-0x1.85a0ef8c59800p-3"}),
    # re-pinned after 0.16.0 run 5 D1 (KO-reset V1 knock-outs on the cell-average projection, a sanctioned move);
    # at 31f46955: price 0x1.9a7a04adb6823p+6, delta -0x1.a1fb6ee4c4000p-4, gamma -0x1.f9be62b912000p-6,
    # rho -0x1.1706a55e54c00p-2, dividend_rho -0x1.32ad67fde6000p-5
    "ko_reset": (KOResetSnowballQuadEngine, QuadParams(), dated_ko_reset, env, {
        "price": "0x1.9a7a5194111b4p+6", "delta": "-0x1.a205f2aa99400p-4", "gamma": "-0x1.f9d9cd80ec000p-6",
        "rho": "-0x1.170deed1e5c00p-2", "dividend_rho": "-0x1.32ba384abc000p-5"}),
}


class TestBaseIsUnchanged:
    """AC5: the base price is bitwise; on spot-independent vol the spot and carry legs size exactly as the base
    did, so delta, gamma, rho and dividend rho are bitwise too."""

    @pytest.mark.parametrize("name", list(SPOT_INDEPENDENT))
    def test_the_clone_prices_the_base_bitwise(self, name):
        cls, params, make, market, _ = SPOT_INDEPENDENT[name]
        product, pricing_env = make(), market()
        engine = cls(params)
        clone = engine.create_bump_context(product, pricing_env)
        assert clone is not engine
        assert clone.price(product, pricing_env).hex() == cls(params).price(product, pricing_env).hex()
        base_stats = cls(params).calculate_event_stats(product, pricing_env)
        stats = clone.calculate_event_stats(product, pricing_env)
        for field in ("pv", "expected_lifetime", "total_discounted_cash"):
            assert getattr(stats, field) == getattr(base_stats, field), field

    @pytest.mark.parametrize("name", [n for n, case in SPOT_INDEPENDENT.items() if case[4] is not None])
    def test_spot_and_carry_greeks_are_bitwise(self, name):
        cls, params, make, market, pins = SPOT_INDEPENDENT[name]
        greeks = calc().calculate(make(), market(), cls(params), greeks=list(pins))
        assert hexes(greeks) == pins

    def test_the_cell_stretch_is_frozen_with_the_cell(self):
        cls, params, make, market, _ = SPOT_INDEPENDENT["snowball_cell_stretch"]
        product, pricing_env = make(), market()
        engine = cls(params)
        clone = engine.create_bump_context(product, pricing_env)
        record = clone._frozen_sizing
        engine.price(product, pricing_env)
        assert engine._last_sizing == record
        unsnapped = 2.0 * (10.0 * 0.2 + 1.02) / (record.grid_points - 1)
        assert record.cell_width > unsnapped * (1.0 + 1e-9)       # the base cell was widened to the barriers
        vega_env = env(0.21, 95.0)
        clone.price(product, vega_env)
        assert clone._last_sizing == record
        own = cls(params)
        own.price(product, vega_env)
        assert own._last_sizing.cell_width != record.cell_width   # the leg's own sizing chose another cell

    def test_a_spot_dependent_smile_moves_delta_by_the_frozen_domain(self):
        """A spot bump moves the vol, so each spot leg used to re-size its domain; on the frozen domain delta and
        gamma move (listed), price and rho stay bitwise."""
        product, pricing_env = dated_snowball(), env(surface=SpotSmile())
        greeks = calc().calculate(product, pricing_env, SnowballQuadEngine(QuadParams()),
                                  greeks=["price", "delta", "gamma", "rho"])
        assert greeks["price"].hex() == "0x1.8bee41fe00012p+6"
        assert greeks["rho"].hex() == "-0x1.383027dcce000p-3"
        # 31f46955: delta 0x1.f5b470606e800p-3 (0.24497306627546322), gamma -0x1.43733f89a2000p-6
        assert greeks["delta"] == pytest.approx(0.24497306627546322, rel=1e-3)
        assert greeks["delta"].hex() != "0x1.f5b470606e800p-3"


class TestCloneContract:
    """AC5: the clone is idempotent, a different contract sizes itself, routing follows the frozen h."""

    def test_the_clone_is_its_own_bump_context(self):
        product, pricing_env = dated_snowball(), env()
        clone = SnowballQuadEngine(QuadParams()).create_bump_context(product, pricing_env)
        assert clone.create_bump_context(product, pricing_env) is clone
        assert clone.create_bump_context(dated_snowball(2.0), env(0.3)) is clone

    @pytest.mark.parametrize("auto", [False, True])
    def test_a_different_contract_sizes_itself(self, auto):
        params = QuadParams(auto_converge=auto)
        product, pricing_env = dated_snowball(1.0, 3, "discrete"), env()
        clone = SnowballQuadEngine(params).create_bump_context(product, pricing_env)
        for other in (dated_snowball(1.0, 3, "discrete", ki_level=70.0), dated_snowball(2.0, 3, "discrete")):
            for market in (pricing_env, env(0.25)):
                assert clone.price(other, market).hex() == SnowballQuadEngine(params).price(other, market).hex()

    def test_a_time_aged_copy_of_the_contract_prices_on_the_frozen_lattice(self):
        product, pricing_env = dated_snowball(), env()
        clone = SnowballQuadEngine(QuadParams()).create_bump_context(product, pricing_env)
        clone.price(product, pricing_env)
        record = clone._last_sizing
        shift = shift_for_theta(product, pricing_env, 1.0 / 365.0, VAL + timedelta(days=1))
        assert shift.product is not product
        clone.price(shift.product, shift.env)
        assert clone._last_sizing == record
        engine = SnowballQuadEngine(QuadParams())
        engine.price(shift.product, shift.env)
        assert engine._last_sizing != record              # the leg's own sizing differs (N 1441 -> 1439 class)

    def test_auto_converge_runs_the_ladder_once_at_the_base(self, monkeypatch):
        product, pricing_env = dated_snowball(1.0, 3, "discrete"), env()
        engine = SnowballQuadEngine(QuadParams(auto_converge=True))
        base = engine.price(product, pricing_env)
        clone = engine.create_bump_context(product, pricing_env)
        calls = []
        original = SnowballQuadEngine._price_with_convergence
        monkeypatch.setattr(SnowballQuadEngine, "_price_with_convergence",
                            lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
        assert clone.price(product, pricing_env).hex() == base.hex()
        clone.price(product, env(0.21))
        clone.calculate_event_stats(product, env(0.21))
        assert calls == []
        assert clone._frozen_sizing.grid_points == engine._last_convergence_info["grid_points"]

    def test_routing_is_recomputed_on_the_frozen_cell(self):
        """A knock-out 0.735 default cells of diffusion after a knock-in date is resolved at vol 20%; at vol 19% on
        the frozen cell its kernel spans 0.698 cells, below the kernel guard (~0.708): the leg prices it with the
        local stencil, as any solve on that cell does today (its own, smaller, cell would span 0.723 and resolve it)."""
        delta = (0.735 * default_h(0.2) / 0.2) ** 2
        product = float_snowball((0.25, 0.5 + delta, 0.75, 1.0), QUARTERS)
        engine = SnowballQuadEngine(QuadParams())
        clone = engine.create_bump_context(product, env(0.2))
        clone.price(product, env(0.2))
        assert clone._last_local_steps == ()
        leg = clone.price(product, env(0.19))
        h = clone._frozen_sizing.cell_width
        assert clone._last_sizing == clone._frozen_sizing
        assert clone._last_local_steps == (2,)
        assert clone._local_step(0.5 * 0.19 ** 2 * delta, h)
        unfrozen = SnowballQuadEngine(QuadParams())
        unfrozen.price(product, env(0.19))
        assert unfrozen._last_local_steps == ()
        assert math.isfinite(leg)

    def test_a_leg_the_frozen_cell_refuses_prices_on_its_own_sizing(self):
        """The same knock-out barrier observed on both sides of that step: the stencil cannot carry it, so on the
        frozen cell the vol-19% leg is refused. A greek that priced must keep pricing: that leg alone falls back to
        its own sizing (which resolves the step), bitwise today's leg, and is counted."""
        delta = (0.735 * default_h(0.2) / 0.2) ** 2
        product = float_snowball((0.25, 0.5, 0.5 + delta, 0.75, 1.0), QUARTERS)
        params = QuadParams()
        clone = SnowballQuadEngine(params).create_bump_context(product, env(0.2))
        base = clone.price(product, env(0.2))
        assert base.hex() == SnowballQuadEngine(params).price(product, env(0.2)).hex()
        assert clone.frozen_sizing_fallbacks == 0
        leg = clone.price(product, env(0.19))
        assert leg.hex() == SnowballQuadEngine(params).price(product, env(0.19)).hex()
        assert clone.frozen_sizing_fallbacks == 1
        assert clone.price(product, env(0.21)) != SnowballQuadEngine(params).price(product, env(0.21))   # frozen
        assert clone.frozen_sizing_fallbacks == 1
        clone.calculate_event_stats(product, env(0.19))           # statistics fall back the same way
        assert clone.frozen_sizing_fallbacks >= 2

    def test_under_auto_converge_a_refused_leg_runs_its_own_ladder(self, monkeypatch):
        """The fallback is the leg's own sizing, which under ``auto_converge`` is its ladder (forced refusal: the
        frozen rung resolves every step of a natural case)."""
        product, pricing_env, leg_env = dated_snowball(1.0, 3, "discrete"), env(), env(0.19)
        clone = SnowballQuadEngine(QuadParams(auto_converge=True)).create_bump_context(product, pricing_env)
        original = SnowballQuadEngine._require_resolved_kernels

        def refuse_on_the_frozen_cell(self, times, tau, h, **kw):
            vol_19 = tau[-1] < 0.5 * 0.195 ** 2 * (times[-1] - times[-2])
            if self._frozen_sizing is not None and not self.__dict__.get("_frozen_suspended") and vol_19:
                raise UnderResolvedStepError("forced")
            return original(self, times, tau, h, **kw)

        monkeypatch.setattr(SnowballQuadEngine, "_require_resolved_kernels", refuse_on_the_frozen_cell)
        leg = clone.price(product, leg_env)
        assert clone.frozen_sizing_fallbacks == 1
        assert leg.hex() == SnowballQuadEngine(QuadParams(auto_converge=True)).price(product, leg_env).hex()

    def test_volga_that_priced_still_prices(self):
        delta = (0.735 * default_h(0.2) / 0.2) ** 2
        product = float_snowball((0.25, 0.5, 0.5 + delta, 0.75, 1.0), QUARTERS)
        volga = calc().calculate(product, env(0.2), SnowballQuadEngine(QuadParams()), greeks=["volga"])["volga"]
        assert math.isfinite(volga)

    def test_a_base_the_guard_refuses_still_raises(self):
        """Base pricing is untouched: a contract today's sizing refuses is refused on the clone, with today's error."""
        delta = (0.69 * default_h(0.2) / 0.2) ** 2
        product = float_snowball((0.25, 0.5, 0.5 + delta, 0.75, 1.0), QUARTERS)
        with pytest.raises(UnderResolvedStepError, match="same barrier") as today:
            SnowballQuadEngine(QuadParams()).price(product, env(0.2))
        clone = SnowballQuadEngine(QuadParams()).create_bump_context(product, env(0.2))
        with pytest.raises(UnderResolvedStepError, match="same barrier") as frozen:
            clone.price(product, env(0.2))
        assert str(frozen.value) == str(today.value)


SIZING_LEAD = (2.0 * (10.0 * 0.2 + 1.02) * 2.5 / (0.2 * 5000.0)) ** 2


def first_ki(lead):
    return float_snowball(QUARTERS, (lead,) + QUARTERS)


def first_ko(lead):
    return float_snowball((lead,) + QUARTERS, QUARTERS)


def hour_theta(engine, product, pricing_env):
    """``V(1 h later) - V(now)`` through the riskmeasures theta machinery (``shift_for_theta`` + ``bumped_value``
    on the bump context): the date clock cannot carry an hour, so the scenario is built at the hour directly."""
    bump = engine.create_bump_context(product, pricing_env)
    shift = shift_for_theta(product, pricing_env, HOUR, pricing_env.valuation_date + timedelta(hours=1))
    return bumped_value(bump, shift) - bump.price(product, pricing_env)


class TestBumpGreeksOnTheBaseLattice:
    """AC4."""

    def test_an_hour_theta_across_the_sizing_cap_is_flat(self):
        """(i) The first knock-in 2.4e-4 y ahead needs 4905 nodes; an hour later it is past the 5001-node cap and the
        leg used to drop to 1001 nodes (price jump -9.0e-4, theta -7.9/y against 0)."""
        product, pricing_env = first_ki(SIZING_LEAD * 1.04), env(0.2, 75.0 * math.exp(0.05))
        frozen = hour_theta(SnowballQuadEngine(QuadParams()), product, pricing_env)
        reference = hour_theta(SnowballQuadEngine(QuadParams(grid_points=16001)), product, pricing_env)
        assert abs(frozen - reference) <= 1e-6, (frozen, reference)

    def test_vega_across_the_exact_first_step_threshold(self):
        """(ii) The first knock-out 12.9 cells of diffusion ahead (vol 5%): on the vol-up leg its kernel spans 15.5
        frozen cells, past S_F = 13, so the recomputed routing prices that leg's first step on the lattice. Vega is
        GreeksCalculator's (the P&L of one vol point). Measured: frozen 0.1456450 (error 1.84e-4), unfrozen 0.1456943
        (2.33e-4), V1-16001 0.1454611: the residual is the S_F switch itself (routing is never frozen), not sizing."""
        lead = (12.9 * default_h(0.05, 0.5) / 0.05) ** 2
        product = float_snowball((lead, 0.25, 0.375, 0.5), (0.25, 0.375, 0.5))
        pricing_env, up = env(0.05, 95.0), env(spot=95.0, surface=FlatVolSurface(0.05 + 0.01))
        frozen = calc().calculate(product, pricing_env, SnowballQuadEngine(QuadParams()), greeks=["vega"])["vega"]
        base_engine = SnowballQuadEngine(QuadParams())
        unfrozen = base_engine.price(product, up) - base_engine.price(product, pricing_env)
        ref_engine = SnowballQuadEngine(QuadParams(grid_points=16001))
        reference = ref_engine.price(product, up) - ref_engine.price(product, pricing_env)
        assert abs(frozen - reference) < abs(unfrozen - reference)
        assert abs(frozen - reference) <= 2e-3, (frozen, reference)

    def test_a_dated_one_day_theta_and_its_vega_match_v2(self):
        """(iii) Mid-week, daily knock-in: the 1-day theta drops a knock-in date (the step count changes)."""
        product, pricing_env = dated_snowball(), env()
        greeks = calc().calculate(product, pricing_env, SnowballQuadEngine(QuadParams()), greeks=["vega", "theta"])
        v2 = calc().calculate(product, pricing_env, SnowballQuadEngineV2(QuadV2Params()), greeks=["vega", "theta"])
        for name in ("vega", "theta"):
            assert abs(greeks[name] - v2[name]) <= 2e-4 * abs(v2[name]), (name, greeks[name], v2[name])

    def test_trade_risk_and_trade_value_theta_come_from_one_frozen_lattice(self):
        from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
        from quantark.cashleg.base import LegDirection
        from quantark.cashleg.event_distribution import EventType
        from quantark.cashleg.leg_valuator import value_leg
        from quantark.portfolio.equity.position import EquityPosition

        product, pricing_env = dated_snowball(), env()
        leg = AutocallableCashLeg(
            direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE, notional=1_000.0, rate=0.05,
            terminal_accrual_factor=1.0, terminal_settlement_time=product.get_maturity(pricing_env),
            accrual_basis=AccrualBasis.KO_MATURITY, terminal_events=frozenset({EventType.MATURITY_NO_KO}))
        engine = SnowballQuadEngine(QuadParams())
        position = EquityPosition(product=product, quantity=2.0, entry_price=100.0, underlying="TEST",
                                  engine=engine, entry_timestamp=VAL, cash_legs=[leg])
        risk = position.get_trade_risk(pricing_env, calc(), ["theta", "vega"])

        clone = engine.create_bump_context(product, pricing_env)
        streams = position._required_streams()
        unit = position._get_unit_notional

        def trade(prod, market, legs):
            result = clone.price_with_events(prod, market, streams=streams)
            return float(result.npv), sum(value_leg(x, result.event_distribution, market, unit(market)) for x in legs)

        base_npv, base_legs = trade(product, pricing_env, [leg])
        shift = shift_for_theta(product, pricing_env, 1.0 / 365.0, VAL + timedelta(days=1))
        from quantark.asset.equity.riskmeasures.greeks.theta_shift import shift_cash_legs
        npv_t, legs_t = trade(shift.product, shift.env, shift_cash_legs([leg], shift.time_bump))
        assert risk.product["theta"] == 2.0 * (npv_t + shift.fixed_pv - base_npv)
        assert risk.total["theta"] == 2.0 * (npv_t + shift.fixed_pv - base_npv) + (legs_t - base_legs)
        npv_v, legs_v = trade(product, env(surface=FlatVolSurface(0.2 + 0.01)), [leg])
        assert risk.product["vega"] == 2.0 * (npv_v - base_npv)
        # the leg npv is the frozen lattice's price, which differs from the leg's own sizing
        assert npv_t == clone.price(shift.product, shift.env)
        assert npv_t != SnowballQuadEngine(QuadParams()).price(shift.product, shift.env)
