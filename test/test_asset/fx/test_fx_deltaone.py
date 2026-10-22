"""
Tests for FX delta-one products (spot, forward, swap) and FxDeltaOneEngine.
"""

import math
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward, FxSpot, FxSwap
from quantark.param import FlatRateCurve, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar import BusinessDayConvention, Calendar
from quantark.util.exceptions import PricingError, ValidationError

VALUATION = datetime(2026, 6, 12)
SPOT = 1.20
R_DOM = 0.05  # quote currency (USD)
R_FOR = 0.03  # base currency (EUR)
NOTIONAL = 1_000_000.0
PAIR = CurrencyPair("EUR", "USD")


def make_env(**overrides):
    kwargs = dict(
        valuation_date=VALUATION,
        spot_quote=SpotQuote(spot=SPOT),
        domestic_curve=FlatRateCurve(rate=R_DOM),
        foreign_curve=FlatRateCurve(rate=R_FOR),
    )
    kwargs.update(overrides)
    return FxPricingEnvironment(**kwargs)


class TestFxForward:
    MATURITY_DATE = datetime(2027, 6, 12)

    def make_forward(self, contract_rate):
        return FxForward(
            currency_pair=PAIR,
            notional_base=NOTIONAL,
            contract_rate=contract_rate,
            maturity_date=self.MATURITY_DATE,
        )

    def test_zero_npv_at_market_forward(self):
        env = make_env()
        t = self.make_forward(1.0).get_maturity(env)
        fair_forward = SPOT * math.exp((R_DOM - R_FOR) * t)
        price = FxDeltaOneEngine().price(self.make_forward(fair_forward), env)
        assert price == pytest.approx(0.0, abs=1e-8)

    def test_known_value(self):
        env = make_env()
        forward = self.make_forward(1.21)
        t = forward.get_maturity(env)
        fwd_mkt = SPOT * math.exp((R_DOM - R_FOR) * t)
        expected = NOTIONAL * (fwd_mkt - 1.21) * math.exp(-R_DOM * t)
        assert FxDeltaOneEngine().price(forward, env) == pytest.approx(
            expected, rel=1e-12
        )

    def test_details_base_currency_npv(self):
        env = make_env()
        forward = self.make_forward(1.21)
        t = forward.get_maturity(env)
        fwd_mkt = SPOT * math.exp((R_DOM - R_FOR) * t)
        details = FxDeltaOneEngine().price_details(forward, env)
        expected_base = details["npv_quote_currency"] / SPOT
        assert details["npv_base_currency"] == pytest.approx(
            expected_base, rel=1e-10
        )
        assert details["market_forward_rate"] == pytest.approx(fwd_mkt, rel=1e-12)

    def test_forward_points(self):
        env = make_env()
        details = FxDeltaOneEngine().price_details(self.make_forward(1.21), env)
        assert details["forward_points"] == pytest.approx(
            details["market_forward_rate"] - SPOT, rel=1e-10
        )

    def test_is_linear(self):
        assert self.make_forward(1.21).is_linear is True

    def test_notional_must_be_positive(self):
        with pytest.raises(ValidationError):
            FxForward(
                currency_pair=PAIR,
                notional_base=-1.0,
                contract_rate=1.21,
                maturity_date=self.MATURITY_DATE,
            )

    def test_date_adjustment_modified_following(self):
        # 2027-06-12 is a Saturday; MODIFIED_FOLLOWING moves to Monday 14th
        forward = FxForward(
            currency_pair=PAIR,
            notional_base=NOTIONAL,
            contract_rate=1.21,
            maturity_date=datetime(2027, 6, 12),
            calendar=Calendar(),
            business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        )
        assert forward.get_adjusted_maturity_date() == datetime(2027, 6, 14)


class TestFxSpot:
    def test_pv_discounted_to_settlement(self):
        env = make_env()
        spot_trade = FxSpot(
            currency_pair=PAIR,
            notional_base=NOTIONAL,
            contract_rate=1.19,
            value_date=VALUATION,
            settlement_days=2,
        )
        t = spot_trade.get_maturity(env)
        fwd_mkt = SPOT * math.exp((R_DOM - R_FOR) * t)
        expected = NOTIONAL * (fwd_mkt - 1.19) * math.exp(-R_DOM * t)
        assert FxDeltaOneEngine().price(spot_trade, env) == pytest.approx(
            expected, rel=1e-10
        )

    def test_settlement_date_skips_weekend(self):
        # Friday 2026-06-12 + 2 business days = Tuesday 2026-06-16
        spot_trade = FxSpot(
            currency_pair=PAIR,
            notional_base=NOTIONAL,
            contract_rate=1.19,
            value_date=VALUATION,
            settlement_days=2,
        )
        assert spot_trade.get_settlement_date() == datetime(2026, 6, 16)


class TestFxSwap:
    NEAR = datetime(2026, 6, 16)
    FAR = datetime(2026, 9, 16)

    def make_swap(self, near_rate=1.1990, far_rate=1.2050):
        return FxSwap(
            currency_pair=PAIR,
            notional_base=NOTIONAL,
            near_rate=near_rate,
            far_rate=far_rate,
            near_date=self.NEAR,
            far_date=self.FAR,
        )

    def test_swap_points(self):
        swap = self.make_swap()
        assert swap.swap_points == pytest.approx(0.0060)

    def test_npv_components(self):
        env = make_env()
        swap = self.make_swap()
        t_near = swap.get_near_time(env)
        t_far = swap.get_maturity(env)

        df_d_near = math.exp(-R_DOM * t_near)
        df_d_far = math.exp(-R_DOM * t_far)
        df_f_near = math.exp(-R_FOR * t_near)
        df_f_far = math.exp(-R_FOR * t_far)

        # Near leg: sell base / receive quote; far leg: buy base / pay quote
        expected_quote = NOTIONAL * swap.near_rate * df_d_near - (
            NOTIONAL * swap.far_rate * df_d_far
        )
        expected_base = NOTIONAL * df_f_far - NOTIONAL * df_f_near

        details = FxDeltaOneEngine().price_details(swap, env)
        assert details["npv_quote_currency"] == pytest.approx(
            expected_quote, rel=1e-10
        )
        assert details["npv_base_currency"] == pytest.approx(
            expected_base, rel=1e-10
        )

        total = FxDeltaOneEngine().price(swap, env)
        assert total == pytest.approx(expected_quote + expected_base * SPOT, rel=1e-10)

    def test_expired_near_leg_drops_out(self):
        env = make_env(valuation_date=datetime(2026, 7, 1))
        swap = self.make_swap()
        t_far = swap.get_maturity(env)

        expected_quote = -NOTIONAL * swap.far_rate * math.exp(-R_DOM * t_far)
        expected_base = NOTIONAL * math.exp(-R_FOR * t_far)

        details = FxDeltaOneEngine().price_details(swap, env)
        assert details["near_leg_expired"] is True
        assert details["npv_quote_currency"] == pytest.approx(
            expected_quote, rel=1e-10
        )
        assert details["npv_base_currency"] == pytest.approx(
            expected_base, rel=1e-10
        )

    def test_near_leg_is_included_on_settlement_date(self):
        details = FxDeltaOneEngine().price_details(
            self.make_swap(), make_env(valuation_date=self.NEAR)
        )
        assert details["near_leg_expired"] is False
        assert details["domestic_discount_factor_near"] == pytest.approx(1.0)
        assert details["foreign_discount_factor_near"] == pytest.approx(1.0)

    def test_near_after_far_rejected(self):
        with pytest.raises(ValidationError):
            FxSwap(
                currency_pair=PAIR,
                notional_base=NOTIONAL,
                near_rate=1.1990,
                far_rate=1.2050,
                near_date=self.FAR,
                far_date=self.NEAR,
            )


class TestEngineGuards:
    def test_rejects_non_deltaone_product(self):
        from quantark.asset.fx.product.option import FxVanillaOption
        from quantark.util.enum import OptionType

        option = FxVanillaOption(
            strike=1.25,
            option_type=OptionType.CALL,
            maturity=1.0,
            notional_foreign=NOTIONAL,
        )
        with pytest.raises(PricingError):
            FxDeltaOneEngine().price(option, make_env())
