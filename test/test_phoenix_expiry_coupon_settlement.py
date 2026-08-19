"""`coupon_pay_type=EXPIRY` on a phoenix: coupons roll up and pay at TERMINATION.

Termination means the knock-out date when the note knocks out, and maturity
when it does not. The engines disagreed about this and each was wrong in its
own way: the Monte Carlo engine forfeited every coupon accrued before a
knock-out and then deferred the knock-out payoff itself to maturity, while both
deterministic engines credited each coupon at its observation node discounted
to maturity, so an accrued coupon survived the knock-out but was paid years
after the note had ended.

The tests below need no benchmark. The product is built so that every path
knocks out at the sixth of twelve observations -- the KO barrier is
unreachable before it and certain from it -- and the coupon barrier is certain
throughout, so the payoff is deterministic and the whole of it is a SINGLE
cashflow at the knock-out date. Two consequences pin the semantics exactly:

* at zero rates, EXPIRY and INSTANT must agree, because the only difference
  between them is when cash moves; anything forfeited breaks this; and
* at a positive rate the EXPIRY price must be the zero-rate price discounted
  by exactly the knock-out settlement, because that is the only date any cash
  moves. Discounting from maturity instead breaks this.

Rates are set with r == q so the drift, and therefore every path, is identical
across the two environments.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType
from quantark.util.enum.engine_enums import MonteCarloMethod

OBSERVATIONS = 12
KO_INDEX = 6                      # the observation every path knocks out on
KO_TIME = KO_INDEX / OBSERVATIONS  # 0.5y
RATE = 0.05


def environment(rate: float) -> PricingEnvironment:
    """r == q keeps the drift (and so every path) the same across rates."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=rate),
        valuation_date=datetime(2024, 1, 1),
    )


def product(pay_type: CouponPayType):
    """Unreachable KO until observation 6, certain from it; coupon always paid."""
    return create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=[500.0] * (KO_INDEX - 1) + [1.0] * (OBSERVATIONS - KO_INDEX + 1),
        ko_rate=0.08,
        ki_barrier=1.0,
        coupon_barrier=1.0,
        coupon_rate=0.02,
        num_observations=OBSERVATIONS,
        memory_coupon=False,
        coupon_pay_type=pay_type,
    )


def mc_price(pay_type, rate):
    engine = PhoenixMCEngine(
        params=MCParams(seed=20260819, num_paths=8192, use_qmc=True,
                        rqmc_min_batches=1, rqmc_max_batches=1,
                        rqmc_paths_mode="per_batch"),
        method=MonteCarloMethod.RANDOMIZED_QUASI)
    return engine.price(product(pay_type), environment(rate))


def pde_price(pay_type, rate):
    return PhoenixPDESolver(params=PDEParams(accuracy="standard")).calculate_greeks(
        product(pay_type), environment(rate))["price"]


def quad_price(pay_type, rate):
    return PhoenixQuadEngine(params=QuadParams(grid_points=1001)).price(
        product(pay_type), environment(rate))


ENGINES = {"mc": mc_price, "pde": pde_price, "quad": quad_price}

#: The deterministic engines value a rolled-up coupon as a fixed discount from
#: the CONTRACTUAL maturity, so an accrued coupon survives a knock-out (they
#: never forfeited) but is paid after the note has already ended. Pricing it
#: correctly needs the coupon discounted by a "1 paid at termination" claim --
#: an auxiliary surface carried through the backward induction with its own
#: boundary conditions -- rather than by a scalar. Strict xfail: this flips to a
#: failure the moment someone implements it, which is the point.
_PAYS_AT_MATURITY_NOT_TERMINATION = pytest.mark.xfail(
    strict=True,
    reason="PDE/QUAD discount the roll-up from maturity, not from the knock-out "
    "date; needs a termination-value surface. Phoenix EXPIRY is uncertified "
    "for exactly this reason.",
)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_nothing_is_forfeited_at_zero_rates(engine):
    """Zero rates make payment timing free, so the two conventions coincide.

    All three engines satisfy this: the deterministic pair always did, and the
    Monte Carlo engine does now that a knock-out no longer voids the roll-up.
    """
    price = ENGINES[engine]
    assert price(CouponPayType.EXPIRY, 0.0) == pytest.approx(
        price(CouponPayType.INSTANT, 0.0), rel=1e-6
    )


@pytest.mark.parametrize(
    "engine",
    [
        "mc",
        pytest.param("pde", marks=_PAYS_AT_MATURITY_NOT_TERMINATION),
        pytest.param("quad", marks=_PAYS_AT_MATURITY_NOT_TERMINATION),
    ],
)
def test_the_roll_up_is_paid_on_the_knock_out_date(engine):
    """The whole payoff is one cashflow at knock-out, so one discount factor
    separates the zero-rate price from the discounted one."""
    price = ENGINES[engine]
    expected = price(CouponPayType.EXPIRY, 0.0) * np.exp(-RATE * KO_TIME)
    assert price(CouponPayType.EXPIRY, RATE) == pytest.approx(expected, rel=1e-4)
