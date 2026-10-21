"""Stock drift under default in the hazard-rate convertible engines (jump-diffusion PDE, HW trinomial).

The stock drops by a fraction eta on default at intensity lambda, so it stays a martingale only if it
drifts at r - q + lambda * eta while it survives. A European zero-coupon convertible then has the exact
value

    V0 = e^{-lambda T} [F e^{-rT} + kappa * BS_call(S, F/kappa, r, q - lambda * eta, sigma, T)]
         + R F lambda / (r + lambda) * (1 - e^{-(r + lambda) T})

(survival: redemption or conversion at T; default: recovery R F when it happens).
"""
import dataclasses
import math
from datetime import datetime

import pytest
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
SPOT, VOL, RATE, RATIO, FACE, RECOVERY = 12.0, 0.30, 0.05, 10.0, 100.0, 0.4

ENV = PricingEnvironment(
    valuation_date=VALUATION,
    spot_quote=SpotQuote(spot=SPOT),
    vol_surface=FlatVolSurface(volatility=VOL),
    rate_curve=FlatRateCurve(rate=RATE),
)

# (hazard rate, stock drop on default): Hull's stock-to-zero at a 2% spread, and the product defaults
CREDIT = {"hull-eta1": (0.02 / (1 - RECOVERY), 1.0), "default-eta0.4": (0.01, 0.4)}


def _european_zero(hazard, eta, **terms):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=MATURITY,
        face_value=FACE,
        coupon_rate=0.0,
        conversion_ratio=RATIO,
        conversion_start_date=MATURITY,
        hazard_rate=hazard,
        recovery_rate=RECOVERY,
        stock_jump_on_default=eta,
        **terms,
    )


def _exact_european_zero(hazard, eta):
    T = (MATURITY - VALUATION).days / 365.0
    K, q_eff = FACE / RATIO, -hazard * eta
    d1 = (math.log(SPOT / K) + (RATE - q_eff + 0.5 * VOL**2) * T) / (VOL * math.sqrt(T))
    d2 = d1 - VOL * math.sqrt(T)
    call = SPOT * math.exp(-q_eff * T) * norm.cdf(d1) - K * math.exp(-RATE * T) * norm.cdf(d2)
    survival = math.exp(-hazard * T) * (FACE * math.exp(-RATE * T) + RATIO * call)
    recovery = RECOVERY * FACE * hazard / (RATE + hazard) * (1.0 - math.exp(-(RATE + hazard) * T))
    return survival + recovery


def _jd(bond):
    T = bond.time_to_maturity(VALUATION)
    params = ConvertibleBondPDEParams(
        num_space_steps=800, num_time_steps=2000, max_stock_multiple=max(5.0, math.exp(6.0 * VOL * math.sqrt(T)))
    )
    return ConvertibleBondJumpDiffusionEngine(ENV, params).price_with_details(bond).dirty_price


def _trinomial(scheme):
    def price(bond):
        params = ConvertibleBondTreeParams(num_steps=1000, trinomial_vol_scheme=scheme)
        return ConvertibleBondTrinomialEngine(ENV, params).price_with_details(bond).dirty_price
    return price


ENGINES = {"jump-diffusion": _jd, **{f"trinomial-{s.value}": _trinomial(s) for s in ConvertibleBondTrinomialVolScheme}}


@pytest.mark.parametrize("credit", list(CREDIT))
@pytest.mark.parametrize("engine", list(ENGINES))
def test_european_zero_coupon_convertible_matches_the_exact_value(engine, credit):
    hazard, eta = CREDIT[credit]
    assert ENGINES[engine](_european_zero(hazard, eta)) == pytest.approx(_exact_european_zero(hazard, eta), abs=0.01)


@pytest.mark.parametrize("engine", list(ENGINES))
def test_coupon_bond_matches_hulls_tree(engine):
    # Demo bond (2.5% semi-annual, American conversion) at Hull's stock-to-zero credit (lambda = 2% / (1 - R)).
    # Reference: Hull OFOD 27.4 binomial tree at 2000 steps (a replica that reproduces the book's
    # Example 27.1, 107.44, and agrees with the drift-corrected jump-diffusion PDE to 0.011): 141.3302.
    hazard, eta = CREDIT["hull-eta1"]
    bond = dataclasses.replace(
        _european_zero(hazard, eta), coupon_rate=0.025, conversion_start_date=datetime(2024, 1, 1), credit_spread=0.02
    )
    assert ENGINES[engine](bond) == pytest.approx(141.3302, abs=0.05)
