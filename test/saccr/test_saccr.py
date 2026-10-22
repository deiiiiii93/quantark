"""SA-CCR validation against the Basel Annex 4a worked examples.

Reference: Basel Committee SA-CCR document, Annex 4a.
"""

import pytest

from quantark.saccr import (
    SACCRCalculator,
    SACCRTrade,
    SACCRNettingSet,
    AssetClass,
    Position,
    OptionType,
    CreditRating,
    IndexGrade,
    CommodityType,
)


class TestExample1:
    """Example 1: Interest Rate derivatives (unmargined). EAD ~ 569."""

    @pytest.fixture
    def netting_set(self):
        trades = [
            SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                       start_date=0, end_date=10, maturity=10, position=Position.LONG),
            SACCRTrade("IR2", AssetClass.INTEREST_RATE, 10_000, -20, currency="USD",
                       start_date=0, end_date=4, maturity=4, position=Position.SHORT),
            SACCRTrade("IR3", AssetClass.INTEREST_RATE, 5_000, 50, currency="EUR",
                       start_date=1, end_date=11, maturity=5.5, position=Position.LONG,
                       is_option=True, option_type=OptionType.PUT,
                       underlying_price=0.06, strike_price=0.05, exercise_date=1.0),
        ]
        return SACCRNettingSet("Example1", trades, is_margined=False)

    def test_replacement_cost(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.rc == pytest.approx(60, rel=0.01)

    def test_multiplier(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.multiplier == pytest.approx(1.0, rel=0.01)

    def test_ir_addon(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.addon_aggregate == pytest.approx(347, rel=0.05)

    def test_ead(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.ead == pytest.approx(569, rel=0.05)


class TestExample2:
    """Example 2: Credit derivatives (unmargined). EAD ~ 381."""

    @pytest.fixture
    def netting_set(self):
        trades = [
            SACCRTrade("CR1", AssetClass.CREDIT, 10_000, 20, reference_entity="Firm A",
                       credit_rating=CreditRating.AA, is_index=False,
                       start_date=0, end_date=3, maturity=3, position=Position.LONG),
            SACCRTrade("CR2", AssetClass.CREDIT, 10_000, -40, reference_entity="Firm B",
                       credit_rating=CreditRating.BBB, is_index=False,
                       start_date=0, end_date=6, maturity=6, position=Position.SHORT),
            SACCRTrade("CR3", AssetClass.CREDIT, 10_000, 0, reference_entity="CDX.IG",
                       index_grade=IndexGrade.IG, is_index=True,
                       start_date=0, end_date=5, maturity=5, position=Position.LONG),
        ]
        return SACCRNettingSet("Example2", trades, is_margined=False)

    def test_replacement_cost(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.rc == pytest.approx(0, abs=1)

    def test_multiplier(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.multiplier == pytest.approx(0.965, rel=0.02)

    def test_credit_addon(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.addon_aggregate == pytest.approx(282, rel=0.05)

    def test_ead(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.ead == pytest.approx(381, rel=0.05)


class TestExample3:
    """Example 3: Commodity derivatives (unmargined). EAD ~ 5406."""

    @pytest.fixture
    def netting_set(self):
        trades = [
            SACCRTrade("COM1", AssetClass.COMMODITY, 10_000, -50,
                       commodity_type=CommodityType.CRUDE_OIL,
                       start_date=0, end_date=0.75, maturity=0.75, position=Position.LONG),
            SACCRTrade("COM2", AssetClass.COMMODITY, 20_000, -30,
                       commodity_type=CommodityType.CRUDE_OIL,
                       start_date=0, end_date=2, maturity=2, position=Position.SHORT),
            SACCRTrade("COM3", AssetClass.COMMODITY, 10_000, 100,
                       commodity_type=CommodityType.SILVER,
                       start_date=0, end_date=5, maturity=5, position=Position.LONG),
        ]
        return SACCRNettingSet("Example3", trades, is_margined=False)

    def test_replacement_cost(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.rc == pytest.approx(20, rel=0.01)

    def test_multiplier(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.multiplier == pytest.approx(1.0, rel=0.01)

    def test_commodity_addon(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.addon_aggregate == pytest.approx(3841, rel=0.05)

    def test_ead(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.ead == pytest.approx(5406, rel=0.05)


class TestExample4:
    """Example 4: combined IR + Credit (unmargined). EAD ~ 936."""

    @pytest.fixture
    def netting_set(self):
        ir = [
            SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                       start_date=0, end_date=10, maturity=10, position=Position.LONG),
            SACCRTrade("IR2", AssetClass.INTEREST_RATE, 10_000, -20, currency="USD",
                       start_date=0, end_date=4, maturity=4, position=Position.SHORT),
            SACCRTrade("IR3", AssetClass.INTEREST_RATE, 5_000, 50, currency="EUR",
                       start_date=1, end_date=11, maturity=5.5, position=Position.LONG,
                       is_option=True, option_type=OptionType.PUT,
                       underlying_price=0.06, strike_price=0.05, exercise_date=1.0),
        ]
        credit = [
            SACCRTrade("CR1", AssetClass.CREDIT, 10_000, 20, reference_entity="Firm A",
                       credit_rating=CreditRating.AA, start_date=0, end_date=3, maturity=3,
                       position=Position.LONG),
            SACCRTrade("CR2", AssetClass.CREDIT, 10_000, -40, reference_entity="Firm B",
                       credit_rating=CreditRating.BBB, start_date=0, end_date=6, maturity=6,
                       position=Position.SHORT),
            SACCRTrade("CR3", AssetClass.CREDIT, 10_000, 0, reference_entity="CDX.IG",
                       index_grade=IndexGrade.IG, is_index=True,
                       start_date=0, end_date=5, maturity=5, position=Position.LONG),
        ]
        return SACCRNettingSet("Example4", ir + credit, is_margined=False)

    def test_replacement_cost(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.rc == pytest.approx(40, rel=0.01)

    def test_aggregate_addon(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.addon_aggregate == pytest.approx(629, rel=0.05)

    def test_ead(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.ead == pytest.approx(936, rel=0.05)


class TestExample5:
    """Example 5: IR + Commodity (margined, weekly). EAD ~ 1879."""

    @pytest.fixture
    def netting_set(self):
        ir = [
            SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                       start_date=0, end_date=10, maturity=10, position=Position.LONG),
            SACCRTrade("IR2", AssetClass.INTEREST_RATE, 10_000, -20, currency="USD",
                       start_date=0, end_date=4, maturity=4, position=Position.SHORT),
            SACCRTrade("IR3", AssetClass.INTEREST_RATE, 5_000, 50, currency="EUR",
                       start_date=1, end_date=11, maturity=5.5, position=Position.LONG,
                       is_option=True, option_type=OptionType.PUT,
                       underlying_price=0.06, strike_price=0.05, exercise_date=1.0),
        ]
        commodity = [
            SACCRTrade("COM1", AssetClass.COMMODITY, 10_000, -50,
                       commodity_type=CommodityType.CRUDE_OIL,
                       start_date=0, end_date=0.75, maturity=0.75, position=Position.LONG),
            SACCRTrade("COM2", AssetClass.COMMODITY, 20_000, -30,
                       commodity_type=CommodityType.CRUDE_OIL,
                       start_date=0, end_date=2, maturity=2, position=Position.SHORT),
            SACCRTrade("COM3", AssetClass.COMMODITY, 10_000, 100,
                       commodity_type=CommodityType.SILVER,
                       start_date=0, end_date=5, maturity=5, position=Position.LONG),
        ]
        return SACCRNettingSet(
            "Example5", ir + commodity, is_margined=True,
            threshold=0, minimum_transfer_amount=5,
            independent_collateral_received=150, independent_collateral_posted=0,
            net_collateral=200, mpor_days=14,
        )

    def test_market_value(self, netting_set):
        assert netting_set.market_value == pytest.approx(80, rel=0.01)

    def test_replacement_cost(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.rc == pytest.approx(0, abs=1)

    def test_aggregate_addon(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.addon_aggregate == pytest.approx(1401, rel=0.10)

    def test_multiplier(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.multiplier == pytest.approx(0.958, rel=0.05)

    def test_ead(self, netting_set):
        result = SACCRCalculator().calculate(netting_set)
        assert result.ead == pytest.approx(1879, rel=0.10)
