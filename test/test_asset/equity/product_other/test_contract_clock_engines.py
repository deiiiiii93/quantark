"""Contract clock end to end (T9; spec .ship/2026-11-23-contract-clock AC3, AC4, AC6).

Two spellings of one instant are one contract: a booking whose KO and KI (or whose two KO-reset schedules) sit one
ulp apart prices, reports event statistics, raises tracker events and books cash exactly as the booking that holds the
run's LAST float in both places -- on QUAD V1, QUAD V2, PDE and MC. Two genuinely distinct instants one second apart
stay two instants everywhere: PDE and MC price them alike, QUAD V1 and V2 refuse what their lattices cannot resolve.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde import KOResetSnowballPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngine,
    KOResetSnowballQuadEngineV2,
    SnowballQuadEngine,
    SnowballQuadEngineV2,
)
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker, KOResetLifecycleTracker
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, PostKOScheduleMode
from quantark.util.exceptions import NumericalError

UP = lambda t: float(np.nextafter(t, np.inf))      # noqa: E731
DOWN = lambda t: float(np.nextafter(t, -np.inf))   # noqa: E731
SECOND = 1.0 / (365.0 * 86400.0)
START = pd.Timestamp("2026-01-05")


def env(vol=0.2, r=0.03, q=0.01, spot=100.0):
    surface = FlatVolSurface(max(vol, 1e-8))
    surface.volatility = vol
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=surface, rate_curve=FlatRateCurve(r),
                              div_yield=ContinuousDividendYield(q), valuation_date=START.to_pydatetime())


def snowball(ko_times, ki_times, ki_barrier=75.0, **barrier):
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko_times[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(ko_times),
                                     ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(ki_times), **barrier),
    )


def ko_reset(pre, post, ki):
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=max(pre[-1], post[-1]),
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=list(pre),
                                     ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(ki)),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.12, ko_observation_dates=list(post)),
    )


SNOWBALL_ENGINES = {
    "quad_v1": lambda: SnowballQuadEngine(QuadParams()),
    "quad_v2": lambda: SnowballQuadEngineV2(),
    "pde": lambda: SnowballPDESolver(PDEParams()),
    "mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=20_000)),
}
KO_RESET_ENGINES = {
    "quad_v1": lambda: KOResetSnowballQuadEngine(QuadParams()),
    "quad_v2": lambda: KOResetSnowballQuadEngineV2(),
    "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
    "mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=20_000)),
}

KO = (0.25, 0.5, 0.75, 1.0)
# one tie direction each: the KI one ulp AFTER the KO (LAST = the KI float) and one ulp BEFORE (LAST = the KO float)
SNOWBALL_TWINS = {
    "ki_after": (snowball(KO, (0.25, UP(0.5), 0.75, 1.0)),
                 snowball((0.25, UP(0.5), 0.75, 1.0), (0.25, UP(0.5), 0.75, 1.0))),
    "ki_before": (snowball(KO, (0.25, DOWN(0.5), 0.75, 1.0)), snowball(KO, KO)),
}
PRE = [(i + 1) / 12 * 1.0 for i in range(12)]      # double rounding: 11/12 sits one ulp off 11/18 * 1.5
POST = [(i + 1) / 18 * 1.5 for i in range(18)]
KI = [(i + 1) / 12 * 1.0 for i in range(12)]


def _ko_reset_twins():
    noisy = ko_reset(PRE, POST, KI)
    pre = [r.observation_time for r in noisy.barrier_config.ko_observation_schedule.records]
    post = [r.observation_time for r in noisy.post_barrier_config.ko_observation_schedule.records]
    ki = [r.observation_time for r in noisy.barrier_config.ki_observation_schedule.records]
    return noisy, ko_reset(pre, post, ki)


def same(a, b) -> bool:
    """Bitwise equality through dataclasses, mappings, sequences and arrays (NaN equal to NaN)."""
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return np.float64(a).tobytes() == np.float64(b).tobytes()
    return a == b


def _ledger(tracker_cls, product, path):
    tracker = tracker_cls(product=product, quantity=1.0, start_date=START)
    events = []
    for day, spot in path:
        live = tracker.product_for_lifecycle()
        events += [(e.event_type, e.observation_index, e.cashflow)
                   for e in tracker.observe(START + pd.Timedelta(days=day), live, env(spot=spot), spot)]
    ledger = [(c.event_type, c.amount if hasattr(c, "amount") else c) for c in tracker.lifecycle.ledger.cashflows]
    return events, ledger, tracker.lifecycle.realized_cashflows


class TestTwinBookings:
    """AC3: a 1-ulp booking and the booking at the run's LAST float are one contract on every engine."""

    @pytest.mark.parametrize("direction", sorted(SNOWBALL_TWINS))
    def test_snowball_contract_is_canonical(self, direction):
        noisy, equal = SNOWBALL_TWINS[direction]
        for name in ("ko_observation_schedule", "ki_observation_schedule"):
            a = [r.observation_time for r in getattr(noisy.barrier_config, name).records]
            b = [r.observation_time for r in getattr(equal.barrier_config, name).records]
            assert same(a, b)

    @pytest.mark.parametrize("engine", sorted(SNOWBALL_ENGINES))
    @pytest.mark.parametrize("direction", sorted(SNOWBALL_TWINS))
    def test_snowball_price_and_event_statistics_bitwise(self, direction, engine):
        noisy, equal = SNOWBALL_TWINS[direction]
        make = SNOWBALL_ENGINES[engine]
        assert same(make().price(noisy, env()), make().price(equal, env()))
        assert same(make().calculate_event_stats(noisy, env()), make().calculate_event_stats(equal, env()))

    @pytest.mark.parametrize("engine", sorted(KO_RESET_ENGINES))
    def test_ko_reset_price_and_event_statistics_bitwise(self, engine):
        noisy, equal = _ko_reset_twins()
        make = KO_RESET_ENGINES[engine]
        assert same(make().price(noisy, env()), make().price(equal, env()))
        assert same(make().calculate_event_stats(noisy, env()), make().calculate_event_stats(equal, env()))

    @pytest.mark.parametrize("direction", sorted(SNOWBALL_TWINS))
    def test_snowball_tracker_events_and_cash_ledger(self, direction):
        noisy, equal = SNOWBALL_TWINS[direction]
        # a knock-in on the 0.5 instant, then a knock-out on the next KO date
        path = [(30, 99.0), (91, 99.0), (182, 70.0), (183, 72.0), (274, 110.0)]
        assert _ledger(AutocallableLifecycleTracker, noisy, path) == _ledger(AutocallableLifecycleTracker, equal, path)

    def test_ko_reset_tracker_events_and_cash_ledger(self):
        noisy, equal = _ko_reset_twins()
        path = [(30, 99.0), (61, 78.0), (92, 90.0), (335, 96.0), (365, 96.0)]
        assert _ledger(KOResetLifecycleTracker, noisy, path) == _ledger(KOResetLifecycleTracker, equal, path)


class TestKnockOutWinsTheTie:
    """AC4: a KO and a discrete KI one ulp apart, both breached by the same spot: the KO wins on every engine, as it
    does for the coincident booking. A 1% volatility and zero carry hold the spot at ~110 (9 sigma above the KO level
    103, 12 below the KI level 120); ``disable_ko_after_ki`` makes the order observable: a KI taken first would
    disable the KO and run the note to maturity instead of paying the KO coupon."""

    @pytest.mark.parametrize("engine", sorted(SNOWBALL_ENGINES))
    @pytest.mark.parametrize("ki_time", [UP(0.5), DOWN(0.5)])
    def test_ko_wins(self, engine, ki_time):
        flat = env(vol=0.01, r=0.0, q=0.0, spot=110.0)
        split = snowball((0.5, 1.0), (ki_time, 1.0), ki_barrier=120.0, disable_ko_after_ki=True)
        tie = split.barrier_config.ko_observation_schedule.records[0].observation_time
        coincident = snowball((tie, 1.0), (tie, 1.0), ki_barrier=120.0, disable_ko_after_ki=True)
        make = SNOWBALL_ENGINES[engine]
        assert same(make().price(split, flat), make().price(coincident, flat))
        ko_payoff = float(split.get_ko_observation_profile(flat)["payoffs"][0])
        assert make().price(split, flat) == pytest.approx(ko_payoff, abs=1e-3)


class TestDistinctInstantsStayDistinct:
    """AC6, owner risk #1: events one second apart are two instants on every engine."""

    def _split(self):
        return snowball(KO, (0.25, 0.5 + SECOND, 0.75, 1.0))

    def test_two_instants_in_the_contract(self):
        p = self._split()
        ki = [r.observation_time for r in p.barrier_config.ki_observation_schedule.records]
        assert ki[1] == 0.5 + SECOND and p.barrier_config.ko_observation_schedule.records[1].observation_time == 0.5

    def test_pde_and_mc_price_them_alike(self):
        p = self._split()
        pde = SnowballPDESolver(PDEParams()).price(p, env())
        mc_engine = SnowballMCEngine(params=MCParams(seed=11, num_paths=200_000))
        mc = mc_engine.price(p, env())
        assert abs(pde - mc) < 3.0 * mc_engine.get_last_result().std_error

    @pytest.mark.parametrize("engine", ["quad_v1", "quad_v2"])
    def test_lattice_engines_refuse_what_they_cannot_resolve(self, engine):
        with pytest.raises(NumericalError):
            SNOWBALL_ENGINES[engine]().price(self._split(), env())
