"""EXPIRY monitoring on the double barrier and the touch products: static replication on the analytic engines.

A barrier observed only at maturity makes the value a function of S_T alone: the terminal value of spec 3C
decision 30 (the payoff with its terminal payment discount, then the maturity decision), every cash paid at the
terminal payment time. Before maturity it is therefore exactly a static combination of European vanillas and
cash-or-nothing digitals on the product's own expiry and settlement convention, priced on ``BlackScholesEngine``
and ``DigitalOptionAnalyticalEngine``. Each leg reads its own strike's vol, so under a smile the value is the smile's
static replication (the march would use one strike-selected vol). Delta and gamma are five-point central
differences of that exact value. Where S_T sits on a barrier (a null set) does not change the value.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, NamedTuple, Optional, Sequence

from quantark.util.enum import OptionType

from ..prepare import CompositeProblem, Operation, OutputRequest, bind_layout

__all__ = ["Leg", "double_barrier_legs", "one_touch_legs", "double_one_touch_legs", "ExpiryReplication",
           "ReplicationSolution", "replication_problem"]

_C, _P = OptionType.CALL, OptionType.PUT
#: the five-point stencil's step, relative to the spot it differences at
_STEP_REL = 1e-3
DESCRIPTION = "EXPIRY: static replication by European vanillas and cash digitals (analytic engines)"


class Leg(NamedTuple):
    """``weight`` units of a vanilla (``kind`` "vanilla") or of a digital paying 1 (``kind`` "digital")."""
    weight: float
    kind: str
    strike: float
    option_type: OptionType


def _corridor_legs(product) -> list:
    """The vanilla payoff on L < S_T < U: a call is C(a) - C(U) - (U - K) D(U) + (a - K) D(a), a = max(K, L);
    a put is P(b) - P(L) - (K - L) DP(L) + (K - b) DP(b), b = min(K, U) (D a call digital, DP a put digital)."""
    K, L, U = product.strike, product.lower_barrier, product.upper_barrier
    if product.is_call():
        a = max(K, L)
        if a >= U:
            return []
        return [Leg(1.0, "vanilla", a, _C), Leg(-1.0, "vanilla", U, _C), Leg(-(U - K), "digital", U, _C),
                Leg(a - K, "digital", a, _C)]
    b = min(K, U)
    if b <= L:
        return []
    return [Leg(1.0, "vanilla", b, _P), Leg(-1.0, "vanilla", L, _P), Leg(-(K - L), "digital", L, _P),
            Leg(K - b, "digital", b, _P)]


def double_barrier_legs(product) -> list:
    """Knock-out: the vanilla inside the corridor, R outside it. Knock-in: the vanilla outside, R inside."""
    L, U, R = product.lower_barrier, product.upper_barrier, product.rebate
    corridor = _corridor_legs(product)
    if product.is_knock_out:
        return corridor + [Leg(R, "digital", L, _P), Leg(R, "digital", U, _C)]
    return ([Leg(1.0, "vanilla", product.strike, product.option_type)]
            + [leg._replace(weight=-leg.weight) for leg in corridor]
            + [Leg(R, "digital", L, _C), Leg(-R, "digital", U, _C)])


def one_touch_legs(product) -> list:
    """A one-touch pays R on the barrier's far side (S_T >= B up, <= B down), a no-touch on the near side."""
    beyond = _C if product.is_up_barrier else _P
    short = _P if product.is_up_barrier else _C
    return [Leg(product.rebate, "digital", product.barrier, beyond if product.is_one_touch else short)]


def double_one_touch_legs(product) -> list:
    """A double one-touch pays R outside the corridor, a double no-touch inside it."""
    L, U, R = product.lower_barrier, product.upper_barrier, product.rebate
    if product.is_double_one_touch:
        return [Leg(R, "digital", L, _P), Leg(R, "digital", U, _C)]
    return [Leg(R, "digital", L, _C), Leg(-R, "digital", U, _C)]


class ExpiryReplication:
    """The legs as products on the parent's expiry, settlement and ``multiplier``, each with its analytic engine."""

    def __init__(self, legs: Sequence[Leg], product, multiplier: float):
        from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
        from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine
        from quantark.asset.equity.product.option import CashOrNothingDigitalOption, EuropeanVanillaOption

        engines = {"vanilla": BlackScholesEngine(), "digital": DigitalOptionAnalyticalEngine()}
        self.parts = []
        for leg in legs:
            if leg.weight == 0.0:                       # a + 0 x leg (the strike inside the corridor's edge)
                continue
            common = dict(strike=leg.strike, option_type=leg.option_type, maturity=product.maturity,
                          exercise_date=product.exercise_date, settlement_date=product.settlement_date,
                          settlement_convention=product.settlement_convention, contract_multiplier=multiplier)
            leg_product = (EuropeanVanillaOption(**common) if leg.kind == "vanilla"
                           else CashOrNothingDigitalOption(payout=1.0, **common))
            self.parts.append((leg.weight, engines[leg.kind], leg_product))

    def price(self, env) -> float:
        return float(sum(weight * engine.price(leg, env) for weight, engine, leg in self.parts))

    def point(self, env, spot: Optional[float] = None) -> tuple:
        """(price, delta, gamma) at ``spot`` (the environment's own when None)."""
        base = float(env.spot) if spot is None else float(spot)
        h = _STEP_REL * base
        values = {}
        for j in (-2, -1, 0, 1, 2):
            if j == 0 and spot is None:
                values[j] = self.price(env)
                continue
            moved = deepcopy(env)
            moved.spot_quote.spot = base + j * h
            values[j] = self.price(moved)
        delta = (-values[2] + 8.0 * values[1] - 8.0 * values[-1] + values[-2]) / (12.0 * h)
        gamma = (-values[2] + 16.0 * values[1] - 30.0 * values[0] + 16.0 * values[-1] - values[-2]) / (12.0 * h * h)
        return values[0], float(delta), float(gamma)


class ReplicationSolution(NamedTuple):
    """The session's GRID carrier: each level read from the replication itself; no levels, the nodes the
    product's own grid request binds."""

    replication: ExpiryReplication
    env: object
    nodes: Callable[[], Sequence[float]]

    def rows(self, spot_levels=None):
        levels = [float(s) for s in (self.nodes() if spot_levels is None else spot_levels)]
        out = []
        for spot in levels:
            price, delta, gamma = self.replication.point(self.env, spot)
            out.append({"spot": spot, "price": price, "delta": delta, "gamma": gamma,
                        "calculation_mode": "static_replication"})
        return out


def replication_problem(adapter, product, env, request) -> CompositeProblem:
    """The composite ``prepare`` returns for EXPIRY before maturity: no part is marched, ``compose`` reads the
    replication in the operation's own form."""
    replication = ExpiryReplication(adapter._replication_legs(product), product, adapter._replication_multiplier(product))
    op = request.operation
    if op is Operation.GREEKS:
        def compose(_):
            price, delta, gamma = replication.point(env)
            return {"price": price, "delta": delta, "gamma": gamma}
    elif op is Operation.SESSION_OUTPUTS:
        def compose(_):
            solution = None
            if request.grid:
                solver = adapter.solver
                solution = ReplicationSolution(replication, env, lambda: [float(s) for s in bind_layout(
                    solver, product, env, OutputRequest(Operation.SESSION_OUTPUTS, grid=True)).spatial.s])
            return adapter._session_outputs(replication.price(env), solution)
    else:
        def compose(_):
            return replication.price(env)
    return CompositeProblem(parts=(), compose=compose, request=request, description=DESCRIPTION)
