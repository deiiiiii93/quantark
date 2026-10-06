"""EXPIRY monitoring on the double barrier and the touch products: a barrier observed only at maturity makes the
value a function of S_T alone, priced before maturity as a static combination of European vanillas and cash
digitals on the analytic engines. The reference here is independent of that algebra: quadrature of each product's
terminal value against the lognormal density, delta and gamma by the likelihood-ratio weights under the integral."""
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pde_routes.cases import CONST, env_book, env_const
from pde_routes.sessions import GRID, PV, session
from quantark.asset.equity.engine.pde import DoubleBarrierPDESolver, DoubleOneTouchPDESolver, OneTouchPDESolver
from quantark.asset.equity.product.option import DoubleBarrierOption
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import BarrierDirection, DoubleBarrierType, ObservationType, OptionType, TouchType

X = ObservationType.EXPIRY
T, LAG, M = 1.0, 0.25, 10.0
K, L, U, R = 100.0, 80.0, 130.0, 2.5
B_UP, B_DN, LO, HI, CASH = 120.0, 85.0, 85.0, 120.0, 10.0


def conv():
    """Terminal payment a quarter-year after maturity (test_maturity.py's convention)."""
    return SettlementConvention(lag=LAG, lag_unit=SettlementLagUnit.YEAR_FRACTION)


def _db(kind, opt, lagged):
    kw = dict(maturity=T, observation_type=X, rebate=R, contract_multiplier=M)
    if lagged:
        kw["settlement_convention"] = conv()
    return DoubleBarrierOption(K, opt, U, L, kind, **kw)


def _ot(direction, touch, lagged):
    kw = dict(maturity=T, observation_type=X, rebate=CASH, payment_at_hit=False, touch_type=touch)
    if lagged:
        kw["settlement_convention"] = conv()
    return OneTouchOption(barrier=B_UP if direction is BarrierDirection.UP else B_DN, barrier_direction=direction, **kw)


def _dot(touch, lagged):
    kw = dict(maturity=T, observation_type=X, rebate=CASH, payment_at_hit=False, touch_type=touch)
    if lagged:
        kw["settlement_convention"] = conv()
    return DoubleOneTouchOption(upper_barrier=HI, lower_barrier=LO, **kw)


def _vanilla(opt):
    return (lambda s: max(s - K, 0.0)) if opt is OptionType.CALL else (lambda s: max(K - s, 0.0))


# name -> (solver class, product builder(lagged), terminal value f(S_T) per the contract, breakpoints)
CASES = {
    "dko_call": (DoubleBarrierPDESolver, lambda g: _db(DoubleBarrierType.KNOCK_OUT, OptionType.CALL, g),
                 lambda s: M * (_vanilla(OptionType.CALL)(s) if L < s < U else R), (L, K, U)),
    "dko_put": (DoubleBarrierPDESolver, lambda g: _db(DoubleBarrierType.KNOCK_OUT, OptionType.PUT, g),
                lambda s: M * (_vanilla(OptionType.PUT)(s) if L < s < U else R), (L, K, U)),
    "dki_call": (DoubleBarrierPDESolver, lambda g: _db(DoubleBarrierType.KNOCK_IN, OptionType.CALL, g),
                 lambda s: M * (R if L < s < U else _vanilla(OptionType.CALL)(s)), (L, K, U)),
    "dki_put": (DoubleBarrierPDESolver, lambda g: _db(DoubleBarrierType.KNOCK_IN, OptionType.PUT, g),
                lambda s: M * (R if L < s < U else _vanilla(OptionType.PUT)(s)), (L, K, U)),
    "one_touch_up": (OneTouchPDESolver, lambda g: _ot(BarrierDirection.UP, TouchType.ONE_TOUCH, g),
                     lambda s: CASH * (s >= B_UP), (B_UP,)),
    "no_touch_up": (OneTouchPDESolver, lambda g: _ot(BarrierDirection.UP, TouchType.NO_TOUCH, g),
                    lambda s: CASH * (s < B_UP), (B_UP,)),
    "one_touch_down": (OneTouchPDESolver, lambda g: _ot(BarrierDirection.DOWN, TouchType.ONE_TOUCH, g),
                       lambda s: CASH * (s <= B_DN), (B_DN,)),
    "no_touch_down": (OneTouchPDESolver, lambda g: _ot(BarrierDirection.DOWN, TouchType.NO_TOUCH, g),
                      lambda s: CASH * (s > B_DN), (B_DN,)),
    "double_one_touch": (DoubleOneTouchPDESolver, lambda g: _dot(TouchType.DOUBLE_ONE_TOUCH, g),
                         lambda s: CASH * (s <= LO or s >= HI), (LO, HI)),
    "double_no_touch": (DoubleOneTouchPDESolver, lambda g: _dot(TouchType.DOUBLE_NO_TOUCH, g),
                        lambda s: CASH * (LO < s < HI), (LO, HI)),
}


def reference(name, spot, lagged):
    """(price, delta, gamma): E[DF(0, T_pay) f(S_T)] by quadrature in log S_T, the spot derivatives by the
    likelihood-ratio weights of the lognormal density (no differencing)."""
    _, _, f, breaks = CASES[name]
    r, q, vol = CONST["rate"], CONST["div"], CONST["vol"]
    s2 = vol * vol * T
    mean = log(spot) + (r - q - 0.5 * vol * vol) * T
    df = exp(-r * (T + LAG if lagged else T))
    pts = sorted(log(b) for b in breaks)
    lo, hi = mean - 12 * sqrt(s2), mean + 12 * sqrt(s2)

    def integral(weight):
        def g(x):
            return f(exp(x)) * norm.pdf(x, mean, sqrt(s2)) * weight(x)
        return quad(g, lo, hi, points=pts, limit=400, epsabs=1e-13, epsrel=1e-12)[0]

    score = lambda x: (x - mean) / (s2 * spot)                                         # noqa: E731
    d_score = lambda x: -1.0 / (s2 * spot * spot) - (x - mean) / (s2 * spot * spot)    # noqa: E731
    return (df * integral(lambda x: 1.0), df * integral(score),
            df * integral(lambda x: score(x) ** 2 + d_score(x)))


SPOTS = (100.0, 78.0, 125.0, 140.0)
#: the facade's 1 % spot bump on the exact value, error / max(1, |exact|): measured over the 10 products x 2 payment
#: times x spots 78/100/125/140 (80 cells) at delta max 9.4e-4 (dko_call, 140), median 7.2e-5; gamma max 6.9e-5
BUMP_BOUND = 2e-3


@pytest.mark.parametrize("lagged", [False, True], ids=["T", "T+lag"])
@pytest.mark.parametrize("name", list(CASES))
def test_price_is_the_expectation_of_the_terminal_value(name, lagged):
    cls, make, _, _ = CASES[name]
    for spot in SPOTS:
        want = reference(name, spot, lagged)[0]
        assert abs(cls().price(make(lagged), env_const(spot)) - want) <= 1e-9 * max(1.0, abs(want)), (name, spot)


@pytest.mark.parametrize("name", list(CASES))
def test_greeks_are_the_derivatives_of_that_expectation(name):
    cls, make, _, _ = CASES[name]
    for spot in SPOTS:
        price, delta, gamma = reference(name, spot, True)
        got = cls().calculate_greeks(make(True), env_const(spot))
        assert abs(got["price"] - price) <= 1e-9 * max(1.0, abs(price)), (name, spot)
        assert abs(got["delta"] - delta) <= 1e-6 * max(1.0, abs(delta)), (name, spot)
        assert abs(got["gamma"] - gamma) <= 1e-5 * max(1.0, abs(gamma)), (name, spot)


@pytest.mark.parametrize("market", ["flat", "smile"])
@pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
def test_knock_out_plus_knock_in_is_the_vanilla_plus_the_rebate(opt, market):
    """Each S_T pays the vanilla on one side of the corridor and R on the other, so the pair is a vanilla plus R
    at the terminal payment time, per contract; under a smile (the digital legs replicated off the smile's call
    spread) as well."""
    env = env_const(100.0) if market == "flat" else env_book(100.0)
    ko = DoubleBarrierPDESolver().price(_db(DoubleBarrierType.KNOCK_OUT, opt, True), env)
    ki = DoubleBarrierPDESolver().price(_db(DoubleBarrierType.KNOCK_IN, opt, True), env)
    from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    vanilla = BlackScholesEngine().price(EuropeanVanillaOption(strike=K, option_type=opt, maturity=T,
                                                              settlement_convention=conv(), contract_multiplier=M), env)
    cash = M * R * exp(-CONST["rate"] * (T + LAG))
    assert abs(ko + ki - (vanilla + cash)) <= 1e-10 * (vanilla + cash)


@pytest.mark.parametrize("name", ["dko_call", "dki_put", "one_touch_up", "double_no_touch"])
def test_session_grid_and_spot_curve_read_the_replication_at_each_level(name):
    cls, make, _, _ = CASES[name]
    product, env, engine = make(True), env_const(100.0), cls()
    levels = [78.0, 100.0, 125.0]
    out = session(engine, product, env, {PV, GRID}, levels=levels)
    assert out.pv == engine.price(product, env)
    curve = engine.calculate_spot_greeks_curve(product, env, levels)
    assert list(out.grid) == curve
    for row, spot in zip(curve, levels):
        price, delta, gamma = reference(name, spot, True)
        assert abs(row["price"] - price) <= 1e-9 * max(1.0, abs(price))
        assert abs(row["delta"] - delta) <= 1e-6 * max(1.0, abs(delta))
        assert abs(row["gamma"] - gamma) <= 1e-5 * max(1.0, abs(gamma))
    nodes = session(engine, product, env, {PV, GRID}).grid               # no levels: the grid's own nodes
    assert len(nodes) > 50 and all(np.isfinite(r["price"]) for r in nodes)


@pytest.mark.parametrize("name", ["dko_put", "dki_call", "no_touch_down", "double_one_touch"])
def test_bump_greeks_and_explain_take_the_replication(name):
    cls, make, _, _ = CASES[name]
    product, env, engine = make(False), env_const(100.0), cls()
    from quantark.asset.equity.engine.base_engine import BaseEngine
    bumped = BaseEngine.calculate_greeks(engine, product, env)              # through create_bump_context
    assert bumped["price"] == engine.price(product, env)
    _, delta, gamma = reference(name, 100.0, False)
    # the facade's spot bump differences the exact value: O(h^2) off the derivatives  BOUND: measure
    assert abs(bumped["delta"] - delta) <= BUMP_BOUND * max(1.0, abs(delta))
    assert abs(bumped["gamma"] - gamma) <= BUMP_BOUND * max(1.0, abs(gamma))
    assert "static replication" in engine.explain(product, env).route["composition"]
