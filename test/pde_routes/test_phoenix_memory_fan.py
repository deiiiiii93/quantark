"""The arrears-linear memory fan reproduces the old 2(M+1)-surface fan (spec 2026-10-15, AC1-AC5).

The old fan's numbers are frozen in ``phoenix_memory_fan_reference.json`` (captured at base 1e32c27c by
``phoenix_memory_fan_capture.py``). Prices and session outputs compare with ``golden_compare.close``; the
BUMP Greeks carry the prices' noise through their stencil (``anchor_tolerance`` with a conservative L1
weight per Greek), so they are never held to the bare price tolerance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL, close
from pde_routes.phoenix_memory_fan_capture import CASES, phoenix
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.core.groups import VALUE, Backend
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.modelvalidation.anchors import anchor_tolerance
from quantark.util.enum import CouponPayType

REFERENCE = json.loads(Path(__file__).with_name("phoenix_memory_fan_reference.json").read_text())

#: L1 stencil weights of the BUMP Greeks: price noise times this bounds the Greek's noise
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
