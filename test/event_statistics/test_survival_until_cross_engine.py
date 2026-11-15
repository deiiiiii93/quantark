"""Spec D32 / AC35 cross-engine: a ``SurvivalAnchor.PAYMENT`` accrual leg has one value on MC, QUAD V2 and PDE 1D.

Under PAYMENT a leg reads only termination probabilities and their payment times, so its value is affine in the
probabilities: ``v = c - sum_e w_e m_e`` with ``w_e = sum_i a_i DF(pay_i) g_e(ref_i)`` (``g_e`` = 1 once row e is paid
before the period's reference). The gate is the smoke's gate (``test_cross_engine_smoke``) carried through that
linear map:

    |v_grid - v_MC| <= K x SE(v_MC) + sum_e |w_e| x allowance_e + ABS_FLOOR

- SE(v_MC) from the reference's independent runs (the leg valued on each run's own distribution);
- allowance_e: the grid engine's banked discretisation allowance of the probability the row is read from
  (``allowances.json``: observations / maturity / knocked-in settlement probability);
- the EXPIRY Snowball pays every knock-out at maturity (T+k for each row); paying a coupon later moves no probability,
  so its grid probabilities are asserted equal to the banked INSTANT case's before its allowances are used.

Each engine's leg also equals the definition summed on its own probabilities (round-off), and every engine carries the
same payment times.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from event_statistics.cases import CASES, SEEDS
from quantark.cashleg import SurvivalAnchor
from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.base import LegDirection
from quantark.cashleg.base_amount import BaseAmount, BaseAmountMode
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.enum import CouponPayType
from test_asset.equity.engine_mc import mc_price_path_routes as routes

warnings.filterwarnings("ignore", category=DeprecationWarning)

BANK = json.loads((Path(__file__).resolve().parent / "allowances.json").read_text())
K = 4.0
ABS_FLOOR = 1e-10
ROUND_OFF = 1e-12
ZERO_TOL = 1e-10
GRID = ("pde_1d", "quad_v2")

#: case -> (banked allowance case, product factory, leg horizon)
SETUPS = {
    "snowball_instant": ("flat_snowball_cki", routes.snowball, 1.0),
    "snowball_expiry": ("flat_snowball_cki", lambda: routes.snowball(pay=CouponPayType.EXPIRY), 1.0),
    "ko_reset_absolute": ("flat_ko_reset_absolute_dki", routes.ko_reset, 2.0),
}
_SOURCES = {EventType.KO: "observations.probability", EventType.MATURITY_NO_KO: "maturity.probability",
            EventType.MATURITY_WITH_KI: "ki_settlement.probability"}


def _legs(horizon):
    ends = np.arange(1, int(round(horizon / 0.25)) + 1) * 0.25
    ends = np.r_[ends, horizon + 0.02]                       # one period running past the note's end
    starts = np.r_[0.0, ends[:-1]]

    def leg(basis):
        return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                          base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
                          schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                          payment_convention=PaymentConvention.AT_PERIOD_END, ko_behavior=KOBehavior.TRUNCATE_AT_KO,
                          survival_basis=basis, survival_until=SurvivalAnchor.PAYMENT)
    return {basis.value: leg(basis) for basis in SurvivalBasis}


def _rows(dist):
    """(event, payment times, masses) of the three termination streams."""
    return [(e, np.atleast_1d(np.asarray(dist.payment_times[e], dtype=float)),
             np.atleast_1d(np.asarray(dist.probabilities[e], dtype=float))) for e in _SOURCES]


def _weights(leg, dist, env):
    """``w_e`` per termination row, from the definition: a row paid before the period's reference (or at a start)
    removes the period's amount."""
    entering = leg.survival_basis is SurvivalBasis.ENTER_PERIOD
    refs = leg.schedule.period_starts if entering else leg.schedule.period_ends
    amounts = leg.rate * leg.base.value * (leg.schedule.period_ends - leg.schedule.period_starts) * np.array(
        [env.get_discount_factor(float(t)) for t in leg.schedule.payment_times])
    out = {}
    for event, pays, _ in _rows(dist):
        gone = (pays[None, :] < refs[:, None] - ZERO_TOL)
        if entering:
            gone |= np.abs(pays[None, :] - refs[:, None]) <= ZERO_TOL
        out[event] = (amounts[:, None] * gone).sum(axis=0)
    return float(np.sum(amounts)), out


def _definition(leg, dist, env):
    total, w = _weights(leg, dist, env)
    return total - sum(float(np.dot(w[e], m)) for e, _, m in _rows(dist))


@lru_cache(maxsize=None)
def _mc(name):
    bank_case, product, horizon = SETUPS[name]
    case = CASES[bank_case]
    env = case.env()
    dists = [EventDistribution.from_statistics(case.mc(seed).calculate_event_stats(product(), env)) for seed in SEEDS]
    return dists, env


@lru_cache(maxsize=None)
def _grid(name, engine):
    bank_case, product, _ = SETUPS[name]
    case = CASES[bank_case]
    stats = case.grid[engine]["L0"]().calculate_event_stats(product(), case.env())
    return stats, EventDistribution.from_statistics(stats)


@pytest.mark.parametrize("name", list(SETUPS))
@pytest.mark.parametrize("engine", ("mc", *GRID))
def test_each_engine_values_the_definition_on_its_own_probabilities(name, engine):
    horizon = SETUPS[name][2]
    if engine == "mc":
        dist, env = _mc(name)[0][0], _mc(name)[1]
    else:
        dist, env = _grid(name, engine)[1], _mc(name)[1]
    for leg in _legs(horizon).values():
        assert leg.value(dist, env, 0.0) == pytest.approx(_definition(leg, dist, env), rel=ROUND_OFF)


@pytest.mark.parametrize("name", list(SETUPS))
def test_every_engine_carries_the_same_payment_times(name):
    reference = _rows(_mc(name)[0][0])
    for engine in GRID:
        for (e, pays, _), (_, ref_pays, _) in zip(_rows(_grid(name, engine)[1]), reference):
            np.testing.assert_allclose(pays, ref_pays, rtol=0.0, atol=ROUND_OFF, err_msg=f"{engine} {e.value}")
    if name == "snowball_expiry":
        ko_pays = reference[0][1]
        assert np.all(ko_pays == ko_pays[-1]) and ko_pays[0] > 0.25 + ROUND_OFF      # every knock-out paid T+k


def test_paying_later_moves_no_probability():
    """The EXPIRY Snowball's grid probabilities are the banked INSTANT case's (so are its allowances)."""
    for engine in GRID:
        expiry, _ = _grid("snowball_expiry", engine)
        instant, _ = _grid("snowball_instant", engine)
        for path in _SOURCES.values():
            table, field = path.split(".")
            np.testing.assert_allclose(getattr(getattr(expiry, table), field), getattr(getattr(instant, table), field),
                                       rtol=0.0, atol=1e-12, err_msg=f"{engine} {path}")


@pytest.mark.parametrize("name", list(SETUPS))
def test_payment_legs_agree_across_engines_within_the_matrix_gate(name):
    bank_case, _, horizon = SETUPS[name]
    dists, env = _mc(name)
    failures = []
    for leg_name, leg in _legs(horizon).items():
        runs = np.array([leg.value(d, env, 0.0) for d in dists])
        mc, se = float(np.mean(runs)), float(np.std(runs, ddof=1) / np.sqrt(runs.size))
        if name == "snowball_expiry":
            # every termination pays at 1.0: alive on every path before it, gone on every path after -- the leg is
            # the same number on every run and every engine (a knock-out dropped before its payment would show)
            assert se == 0.0
        else:
            assert se > 0.0
        for engine in GRID:
            _, dist = _grid(name, engine)
            value = leg.value(dist, env, 0.0)
            allowance = BANK[bank_case][engine]["allowance"]
            _, w = _weights(leg, dist, env)
            extra = sum(float(np.dot(np.abs(w[e]), np.atleast_1d(np.asarray(allowance[_SOURCES[e]], dtype=float))))
                        for e in _SOURCES)
            gap = abs(value - mc)
            if gap > K * se + extra + ABS_FLOOR:
                failures.append(f"{engine} {leg_name}: grid {value!r} mc {mc!r} se {se:.3g} allowance {extra:.3g} "
                                f"z {(gap - extra) / se:.2f}")
    assert not failures, "\n".join(failures)
