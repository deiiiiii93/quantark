"""What the holder receives on default in the hazard-rate convertible engines (jump-diffusion PDE, HW trinomial).

On default the stock drops to (1 - eta) S and the holder takes the better of the recovery R F and conversion
into the surviving shares, max(kappa (1 - eta) S, R F), when conversion is open (Ayache, Forsyth & Vetzal
2003, eq. 4.9). With q = 0, no coupons and no call, early conversion is never optimal (the holder's payoff
dominates kappa shares, and the stock is a martingale), so a convertible open for conversion throughout has
the exact value

    V0 = e^{-lambda T} [F e^{-rT} + kappa C(S, F / kappa, T)]
         + int_0^T lambda e^{-lambda t} [R F e^{-rt} + kappa (1 - eta) C(S, R F / (kappa (1 - eta)), t)] dt

with C the Black-Scholes call on the surviving stock (dividend yield -lambda * eta).
"""
import math
from datetime import datetime

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from quantark.asset.bond.engine.pde.convertible import ConvertibleBondJumpDiffusionEngine, ConvertibleBondPDEParams
from quantark.asset.bond.engine.tree.convertible import ConvertibleBondTreeParams, ConvertibleBondTrinomialEngine
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTrinomialVolScheme
from quantark.asset.bond.product.convertible import ConvertibleBond
from quantark.param.quote import SpotQuote
from quantark.param.rrf import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import PricingEnvironment

VALUATION = datetime(2024, 6, 1)
MATURITY = datetime(2029, 1, 1)
T = (MATURITY - VALUATION).days / 365.0
SPOT, VOL, RATE, RATIO, FACE, RECOVERY = 12.0, 0.30, 0.05, 10.0, 100.0, 0.4

ENV = PricingEnvironment(
    valuation_date=VALUATION,
    spot_quote=SpotQuote(spot=SPOT),
    vol_surface=FlatVolSurface(volatility=VOL),
    rate_curve=FlatRateCurve(rate=RATE),
)

# (hazard rate, stock drop on default)
CREDIT = {"default-eta0.4": (0.01, 0.4), "partial-eta0": (0.05, 0.0), "total-eta1": (0.05, 1.0)}


def _american_zero(hazard, eta):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=MATURITY,
        face_value=FACE,
        coupon_rate=0.0,
        conversion_ratio=RATIO,
        hazard_rate=hazard,
        recovery_rate=RECOVERY,
        stock_jump_on_default=eta,
    )


def _d1_d2(strike, t, hazard, eta):
    """Black-Scholes d1, d2 for the surviving stock (dividend yield -hazard * eta)."""
    d1 = (math.log(SPOT / strike) + (RATE + hazard * eta + 0.5 * VOL**2) * t) / (VOL * math.sqrt(t))
    return d1, d1 - VOL * math.sqrt(t)


def _call(strike, t, hazard, eta):
    d1, d2 = _d1_d2(strike, t, hazard, eta)
    return SPOT * math.exp(hazard * eta * t) * norm.cdf(d1) - strike * math.exp(-RATE * t) * norm.cdf(d2)


def _exact_american_zero(hazard, eta):
    survival = math.exp(-hazard * T) * (FACE * math.exp(-RATE * T) + RATIO * _call(FACE / RATIO, T, hazard, eta))
    shares = RATIO * (1.0 - eta)  # shares the holder converts into on default

    def on_default(t):
        conversion = shares * _call(RECOVERY * FACE / shares, t, hazard, eta) if shares > 0.0 else 0.0
        return RECOVERY * FACE * math.exp(-RATE * t) + conversion

    default, _ = quad(lambda t: hazard * math.exp(-hazard * t) * on_default(t), 0.0, T, epsabs=1e-10, limit=200)
    return survival + default


def _exact_conversion_probability(hazard, eta):
    """P(convert) = P(survive, convert at T) + P(default, convert into the surviving shares)."""
    at_maturity = math.exp(-hazard * T) * norm.cdf(_d1_d2(FACE / RATIO, T, hazard, eta)[1])
    if eta >= 1.0:
        return at_maturity
    strike = RECOVERY * FACE / (RATIO * (1.0 - eta))
    at_default, _ = quad(
        lambda t: hazard * math.exp(-hazard * t) * norm.cdf(_d1_d2(strike, t, hazard, eta)[1]), 0.0, T, limit=200
    )
    return at_maturity + at_default


def _jd_result(bond):
    params = ConvertibleBondPDEParams(
        num_space_steps=800, num_time_steps=2000, max_stock_multiple=max(5.0, math.exp(6.0 * VOL * math.sqrt(T)))
    )
    return ConvertibleBondJumpDiffusionEngine(ENV, params).price_with_details(bond)


def _trinomial(scheme):
    def price(bond):
        params = ConvertibleBondTreeParams(num_steps=1000, trinomial_vol_scheme=scheme)
        return ConvertibleBondTrinomialEngine(ENV, params).price_with_details(bond).dirty_price
    return price


ENGINES = {
    "jump-diffusion": lambda bond: _jd_result(bond).dirty_price,
    **{f"trinomial-{s.value}": _trinomial(s) for s in ConvertibleBondTrinomialVolScheme},
}


@pytest.mark.parametrize("credit", list(CREDIT))
@pytest.mark.parametrize("engine", list(ENGINES))
def test_holder_converts_into_the_surviving_shares_on_default(engine, credit):
    # The trees converge at O(dt): partial-eta0 misses by -0.029, -0.013, -0.006, -0.002 at 250..2000 steps
    hazard, eta = CREDIT[credit]
    tolerance = 0.005 if engine == "jump-diffusion" else 0.01
    assert ENGINES[engine](_american_zero(hazard, eta)) == pytest.approx(_exact_american_zero(hazard, eta), abs=tolerance)


@pytest.mark.parametrize("credit", list(CREDIT))
def test_jump_diffusion_conversion_probability_counts_conversion_on_default(credit):
    hazard, eta = CREDIT[credit]
    p = _jd_result(_american_zero(hazard, eta)).conversion_probability
    assert p == pytest.approx(_exact_conversion_probability(hazard, eta), abs=0.01)
