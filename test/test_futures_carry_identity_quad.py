"""Historical Gate D regression with frozen inputs and the real QUAD engine.

The JSON freezes the original calendar and market snapshot, so this test
does not depend on local market caches or re-solve the coupon. It validates
internal chain closure, not the external accuracy of the QUAD delta.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.backtest.futures_risk import CarryRiskSettings, FuturesBookRisk
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    audit_held_book, buckets_of, direct_frozen_curve_book_delta, sample_buckets,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


@pytest.mark.parametrize("readout", ["legacy_linear", "transition"])
def test_worst_historical_date_closes_local_identity_while_preserving_secant_gap(readout):
    case = json.loads((Path(__file__).parent / "data/carry_identity_20240119.json").read_text())
    s0 = case["initial_price"]
    original = create_standard_snowball(
        initial_price=s0, strike=s0, maturity=case["maturity"],
        contract_multiplier=case["notional"] / s0,
        ko_barrier=1.03 * s0, ki_barrier=.75 * s0, ko_rate=case["coupon"],
        rebate_rate=case["coupon"], include_principal=False,
        num_observations=len(case["ko_days"]),
        ko_observation_dates=[d / 365. for d in case["ko_days"]],
        ki_observation_dates=[d / 365. for d in case["ki_days"]],
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
    )
    original.initial_date = datetime.fromisoformat(case["inception"])
    context = CarryCurveContext(
        tuple(IndexFuturesQuote(**q) for q in case["quotes"]), case["spot"],
        FlatRateCurve(rate=case["rate"]), "flat_q", "CSI1000", case["valuation"],
    )

    def env(spot, dividend):
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=spot),
            vol_surface=FlatVolSurface(volatility=case["vol"]),
            rate_curve=context.rate_curve, div_yield=dividend,
            valuation_date=datetime.fromisoformat(case["valuation"]),
        )

    tracker = AutocallableLifecycleTracker(
        product=original, quantity=-1., start_date=pd.Timestamp(case["inception"]))
    product = tracker.product_for_pricing(pd.Timestamp(case["valuation"]),
                                          env(context.spot, context.dividend()))
    engine = SnowballQuadEngine(QuadParams(grid_points=401, readout=readout))

    def price_at(spot, dividend):
        return -engine.price(product, env(spot, dividend))

    pricing_delta = direct_frozen_curve_book_delta(price_at, context, {}, .01 * context.spot)
    risk = FuturesBookRisk(context.spot, pricing_delta,
                           buckets_of(context, sample_buckets(price_at, context, 1.)))
    result = audit_held_book(
        price_at, context, risk, {"IM2402": 91.},
        settings=CarryRiskSettings(reference_notional=case["notional"], audit_spot_bump_rel=.01),
    )
    assert result.status == "pass"
    assert result.identity_status == "pass"
    assert abs(result.identity_residual_hands) < .001
    assert result.finite_bump_identity_residual_hands > .5
    assert abs(result.pricing_delta_local_gap_hands) > .1
    assert abs(result.net_delta_audit_error_hands) < 1e-9
    if readout == "legacy_linear":
        assert result.finite_bump_identity_residual_hands == pytest.approx(.5482832642, abs=1e-7)
        assert pricing_delta / 200 == pytest.approx(-91.4013448101, abs=1e-7)
