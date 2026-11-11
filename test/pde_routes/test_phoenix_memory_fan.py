"""The arrears-linear memory fan reproduces the old 2(M+1)-surface fan (spec 2026-10-15, AC1-AC5).

The old fan's numbers are frozen in ``phoenix_memory_fan_reference.json`` (captured at base 1e32c27c by
``phoenix_memory_fan_capture.py``). Prices, every event-statistics number and session outputs compare with
``golden_compare.close``; the BUMP Greeks carry the prices' noise through their stencil (``anchor_tolerance``
with a conservative L1 weight per Greek), so they are never held to the bare price tolerance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL, close
from golden.pde_bitwise.capture_book import _dated, _dated_env, _first_obs, _phoenix_expiry
from pde_routes.phoenix_memory_fan_capture import CASES, phoenix, stats_fields
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.core.groups import VALUE, Backend
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.modelvalidation.anchors import anchor_tolerance
from quantark.util.enum import CouponPayType

REFERENCE = json.loads(Path(__file__).with_name("phoenix_memory_fan_reference.json").read_text())

#: Generous noise weights of the BUMP Greeks (price noise times this bounds the Greek's): at least the L1
#: stencil weight in each Greek's reported units at the matrix's spots -- delta 1/(0.01 S) and gamma
#: 4/(0.01 S)^2 at S >= 100 (the dated rows at 78 sit within rel 1e-9 x 6.6 of a frozen value either way,
#: a million times real cross-arch noise); vega, theta and the rhos are deliberately loose
GREEK_WEIGHTS = {"delta": 1.0, "gamma": 4.0, "vega": 100.0, "theta": 365.0, "rho": 1e4, "dividend_rho": 1e4}

#: rows too slow for the default loop: the M=36 note and every Greeks row
SLOW = {"memory_m36"}


def _cases(names):
    return [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in names]


def _build(name):
    return CASES[name]["make"]()


@pytest.mark.parametrize("name", _cases(sorted(CASES)))
def test_price_matches_the_frozen_fan(name):
    product, env, solver_cls, params = _build(name)
    live = float(solver_cls(params).price(product, env))
    assert close(live, REFERENCE[name]["price"]), (live, REFERENCE[name]["price"])


#: frozen legacy fields with no counterpart under the unified definitions (T12): the knock-in-ever probability
#: (a knock-in after a knock-out is no knock-in) and the legacy per-date KI arrays (the forward pass, T13)
DROPPED = {"ki_ever_probability", "ki_times", "ki_event_probability", "ki_survival_probability"}


def legacy_view(stats) -> dict:
    """The frozen (legacy) field names read off ``EventStatistics`` where the definition is the same (unified event
    statistics T12; the D21 mapping for the maturity total). The legacy ledger's single maturity row is the
    maturity row plus the knock-in settlement row of the new ledger (same determination and payment)."""
    obs, coupons, ledger = stats.observations, stats.coupons, stats.ledger
    det, pay = list(ledger.determination_times[:-1]), list(ledger.payment_times[:-1])
    disc = list(ledger.discounted_cash[:-2]) + [float(ledger.discounted_cash[-2] + ledger.discounted_cash[-1])]
    undisc = list(ledger.cash_to_pay[:-2]) + [float(ledger.cash_to_pay[-2] + ledger.cash_to_pay[-1])]
    assert ledger.determination_times[-1] == ledger.determination_times[-2]
    assert ledger.payment_times[-1] == ledger.payment_times[-2]
    dates = lambda d: None if d is None else repr(tuple(d[:-1]))   # noqa: E731
    return stats_fields_view({
        "pv": stats.pv, "ko_times": obs.times, "ko_probability": obs.probability,
        "survival_probability": np.maximum(0.0, obs.survival), "expected_discounted_ko_cashflow": obs.discounted_cash,
        "coupon_probability": coupons.probability, "expected_discounted_coupon_cashflow": coupons.discounted_cash,
        "coupon_payment_is_path_dependent": bool(np.all(coupons.payment_is_path_dependent)),
        "ki_probability": stats.ki_settlement.probability,
        "ki_survive_knocked_in_probability": stats.ki_settlement.probability,
        "expected_discounted_maturity_cashflow": stats.maturity.discounted_cash + stats.ki_settlement.discounted_cash,
        "reconciliation_error": stats.reconciliation_error,
        "determination_times": det, "payment_times": pay, "expected_discounted_cashflows": disc,
        "expected_undiscounted_cashflows": undisc, "determination_dates": dates(ledger.determination_dates),
        "payment_dates": dates(ledger.payment_dates), "provenance": stats.provenance,
    })


def stats_fields_view(fields: dict) -> dict:
    from pde_routes.phoenix_memory_fan_capture import _number
    return {k: (v if isinstance(v, str) or v is None else _number(v)) for k, v in fields.items()}


@pytest.mark.parametrize("name", _cases(sorted(CASES)))
def test_event_stats_match_the_frozen_fan(name):
    """The frozen fan's statistics, field for field where the unified definitions keep them (T12): nothing moved but
    the dropped knock-in-ever fields (the reconciliation error is round-off on both sides)."""
    product, env, solver_cls, params = _build(name)
    live = legacy_view(solver_cls(params).calculate_event_stats(product, env))
    want = {k: v for k, v in REFERENCE[name]["stats"].items() if k not in DROPPED}
    assert live.keys() == want.keys()
    bad = {}
    for k in want:
        if k == "reconciliation_error":
            if abs(live[k]) > 1e-10 or abs(want[k]) > 1e-10:
                bad[k] = (live[k], want[k])
        elif k == "provenance":
            if live[k] != want[k]:
                bad[k] = (live[k], want[k])
        elif not close(live[k], want[k]):
            bad[k] = (live[k], want[k])
    assert not bad, bad


#: the dated fixture's coupon is 0.12 * 100 / 12 = 1.0, so on its own a coupon missed at valuation owes exactly
#: one unit and reads the same from the unit carrier S(1) as from S(0) + earned·(S(1) - S(0)) (spec C3b);
#: these rows owe 1.5 (an 18% coupon) and 3.0 (1.0 plus 2.0 carried from before the pricing date)
@pytest.mark.parametrize("pay_type", ["instant", "expiry"])
@pytest.mark.parametrize("coupon_rate,arrears", [(0.18, 0.0), (0.12, 2.0)], ids=["coupon_1.5", "arrears_2"])
@pytest.mark.parametrize("spot", [78.0, 95.0], ids=["miss", "pay"])
def test_a_coupon_decided_at_valuation_reconciles_whatever_it_owes(spot, coupon_rate, arrears, pay_type):
    import dataclasses

    p, _ = _dated("phoenix", fixed_fraction=1.0 / 12.0)
    p.coupon_config = dataclasses.replace(p.coupon_config, coupon_rate=coupon_rate, initial_coupon_arrears=arrears)
    if pay_type == "expiry":
        p = _phoenix_expiry(p)
    env = _dated_env(p, _first_obs(p), spot)
    solver = PhoenixPDESolver(PDEParams(accuracy="standard"))
    stats = solver.calculate_event_stats(p, env)
    ledger = float(np.sum(stats.ledger.discounted_cash))
    # the value reads S(0) + earned·u at a miss (C3); the statistics must compose the same state (C3b)
    assert abs(float(stats.pv) - ledger) <= 1e-10, (stats.pv, ledger)
    assert close(float(stats.pv), float(PhoenixPDESolver(PDEParams(accuracy="standard")).price(p, env)))


@pytest.mark.parametrize("name", _cases(sorted(n for n in CASES if CASES[n]["session"])))
def test_session_outputs_match_the_frozen_fan(name):
    product, env, solver_cls, params = _build(name)
    session = solver_cls(params)._session_outputs(product, env, want_grid=True)
    want = REFERENCE[name]["session"]
    assert close(float(session.npv), want["npv"])
    solution = [float(x) for x in np.asarray(session.solution.solution_vec).ravel()]
    assert close(solution, want["solution"])
    readout = session.solution.readout_vec
    assert (readout is None) == (want["readout"] is None)
    if readout is not None:
        assert close([float(x) for x in np.asarray(readout).ravel()], want["readout"])


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(n for n in CASES if CASES[n]["greeks"]))
def test_greeks_match_the_frozen_fan(name):
    product, env, solver_cls, params = _build(name)
    case = CASES[name]
    live = GreeksCalculator().calculate(product, env, solver_cls(params), greeks=list(case["greeks"]))
    want = REFERENCE[name]["greeks"]
    pv = REFERENCE[name]["price"]
    for key, expected in want.items():
        got = float(live[key])
        tol = anchor_tolerance(expected, pv=pv, weight=GREEK_WEIGHTS[key], rel_tol=GOLDEN_REL_TOL,
                               abs_tol=GOLDEN_ABS_TOL)
        assert math.isfinite(got) and abs(got - expected) <= tol, (key, got, expected, tol)


# --- the layout (AC4) and the observation limit (AC5) -------------------------------------------------

def _value_groups(product):
    p = prepare(PhoenixPDESolver(PDEParams()), product, CASES["memory"]["make"]()[1],
                OutputRequest(Operation.PRICE))
    return p.groups


@pytest.mark.parametrize("num_obs", [12, 36])
@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_memory_phoenix_marches_two_states_whatever_the_observation_count(num_obs, pay_type):
    groups = _value_groups(phoenix(num_obs=num_obs, pay_type=pay_type))
    surfaces = ("v0[0]", "v0[1]", "v1[0]", "v1[1]")
    if pay_type == CouponPayType.EXPIRY:
        surfaces += ("w0", "w1")
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in groups] == [
        ("value", surfaces, Backend.BANDED, VALUE)]
    assert not groups[0].one_dimensional_rhs


def test_sixty_observation_memory_phoenix_prices_finite():
    price = PhoenixPDESolver(PDEParams(accuracy="standard")).price(phoenix(num_obs=60), CASES["memory"]["make"]()[1])
    assert math.isfinite(price)
