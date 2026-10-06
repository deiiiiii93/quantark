"""EXPIRY monitoring on the double barrier and the touch products: static replication on the analytic engines.

A barrier observed only at maturity makes the value a function of S_T alone: the terminal value of spec 3C
decision 30 (the payoff with its terminal payment discount, then the maturity decision), every cash paid at the
terminal payment time. Before maturity it is therefore exactly a static combination of European vanillas and
cash-or-nothing digitals on the product's own expiry and settlement convention, priced on ``BlackScholesEngine``
and ``DigitalOptionAnalyticalEngine``. Each leg reads its own strike's vol, so under a smile the value is the smile's
static replication (the march would use one strike-selected vol). No grid is requested or bound for it. Every greek
is ``GreeksCalculator``'s own of the legs (``BaseEngine.static_replication``): under the caller's method and bump
configuration, and on the facade (``calculate_greeks``, session GRID rows) the calculator on the solver's params.
Where S_T sits on a barrier (a null set) does not change the value.
"""
from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple, Sequence

from quantark.util.enum import OptionType

from ..prepare import CompositeProblem, Operation

__all__ = ["Leg", "double_barrier_legs", "one_touch_legs", "double_one_touch_legs", "ExpiryReplication",
           "ReplicationSolution", "replication_problem", "FACADE_GREEKS"]

_C, _P = OptionType.CALL, OptionType.PUT
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


#: the greeks a PDE facade reads (spec 3C decision 21: price, delta, gamma)
FACADE_GREEKS = ("price", "delta", "gamma")


def _calculator_greeks(solver, product, env) -> dict:
    """price, delta, gamma as ``GreeksCalculator(params=solver.params)`` gives them: the weighted sum of its own
    greeks of the legs (closed forms for the vanillas, its bumps for the digitals)."""
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator

    out = GreeksCalculator(params=solver.params).calculate(product, env, solver, greeks=list(FACADE_GREEKS))
    return {key: float(out[key]) for key in FACADE_GREEKS}


class ReplicationSolution(NamedTuple):
    """The session's GRID carrier: each requested level read through ``GreeksCalculator`` at that spot. A
    replication binds no grid, so it has no nodes to default to: a GRID request names its levels."""

    solver: object
    product: object
    env: object

    def rows(self, spot_levels=None):
        if spot_levels is None:
            from quantark.execution.errors import CapabilityError

            raise CapabilityError(
                f"{type(self.solver).__name__} prices this product by static replication and builds no grid: a "
                "GRID request must name its levels (grid_spot_levels)")
        out = []
        for spot in (float(s) for s in spot_levels):
            moved = deepcopy(self.env)
            moved.spot_quote.spot = spot
            g = _calculator_greeks(self.solver, self.product, moved)
            out.append({"spot": spot, "price": g["price"], "delta": g["delta"], "gamma": g["gamma"],
                        "calculation_mode": "static_replication"})
        return out


def replication_problem(adapter, replication, product, env, request) -> CompositeProblem:
    """The composite ``prepare`` returns for a replicated product: no part is marched and no grid is built;
    ``compose`` reads the legs in the operation's own form."""
    solver, op = adapter.solver, request.operation
    if op is Operation.GREEKS:
        def compose(_):
            return _calculator_greeks(solver, product, env)
    elif op is Operation.SESSION_OUTPUTS:
        def compose(_):
            solution = ReplicationSolution(solver, product, env) if request.grid else None
            return adapter._session_outputs(replication.price(env), solution)
    else:
        def compose(_):
            return replication.price(env)
    return CompositeProblem(parts=(), compose=compose, request=request, description=DESCRIPTION)
