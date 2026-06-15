"""Component and edge-case tests for the SA-CCR module.

These complement the Basel Annex 4a end-to-end tests in ``test_saccr.py`` with
unit-level checks (exact where hand-computable) and validation/edge behavior.
"""

import math

import pytest
from scipy.stats import norm

from quantark.util.numerical import is_close
from quantark.util.exceptions import ValidationError
from quantark.util.enum.option_enums import OptionType


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
def test_enums_and_commodity_map():
    from quantark.saccr.models.enums import (
        AssetClass, CommodityHedgingSet, CommodityType, COMMODITY_TYPE_TO_HEDGING_SET)
    assert {a.name for a in AssetClass} == {
        "INTEREST_RATE", "FX", "CREDIT", "EQUITY", "COMMODITY"}
    assert COMMODITY_TYPE_TO_HEDGING_SET[CommodityType.CRUDE_OIL] is CommodityHedgingSet.ENERGY
    assert COMMODITY_TYPE_TO_HEDGING_SET[CommodityType.GOLD] is CommodityHedgingSet.METALS
    assert OptionType.CALL is not OptionType.PUT


# --------------------------------------------------------------------------- #
# Supervisory parameters                                                       #
# --------------------------------------------------------------------------- #
def test_supervisory_factors_values():
    from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
    from quantark.saccr.models.enums import AssetClass, CreditRating, CommodityType
    assert SP.ALPHA == 1.4
    assert SP.FLOOR == 0.05
    assert SP.get_supervisory_factor(AssetClass.INTEREST_RATE) == 0.005
    assert SP.get_supervisory_factor(AssetClass.FX) == 0.04
    assert SP.get_supervisory_factor(AssetClass.CREDIT, credit_rating=CreditRating.AA) == 0.0038
    assert SP.get_supervisory_factor(
        AssetClass.COMMODITY, commodity_type=CommodityType.ELECTRICITY) == 0.40


def test_supervisory_tables_immutable():
    from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
    from quantark.saccr.models.enums import CreditRating
    with pytest.raises(TypeError):
        SP.SF_CREDIT_SINGLE[CreditRating.AA] = 0.0  # MappingProxyType is read-only


def test_supervisory_unknown_lookups_raise():
    from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
    from quantark.saccr.models.enums import AssetClass
    with pytest.raises(ValidationError):
        SP.get_supervisory_factor(AssetClass.CREDIT, credit_rating=None)
    with pytest.raises(ValidationError):
        SP.get_supervisory_factor(AssetClass.COMMODITY, commodity_type=None)


# --------------------------------------------------------------------------- #
# Supervisory maths                                                            #
# --------------------------------------------------------------------------- #
def test_supervisory_duration_10y():
    from quantark.saccr.engines.maths import supervisory_duration
    assert is_close(supervisory_duration(0, 10), (1 - math.exp(-0.5)) / 0.05)


def test_maturity_factors():
    from quantark.saccr.engines.maths import (
        maturity_factor_unmargined, maturity_factor_margined)
    assert is_close(maturity_factor_unmargined(0.75), math.sqrt(0.75))
    assert is_close(maturity_factor_unmargined(5.0), 1.0)            # capped at 1y
    assert is_close(maturity_factor_margined(14 / 250), 1.5 * math.sqrt(14 / 250))


def test_supervisory_delta_linear_and_option():
    from quantark.saccr.engines.maths import supervisory_delta
    from quantark.saccr.models.enums import Position
    assert supervisory_delta(Position.LONG) == 1.0
    assert supervisory_delta(Position.SHORT) == -1.0
    d1 = (math.log(0.06 / 0.05) + 0.5 * 0.5 ** 2 * 1) / (0.5 * 1)
    got = supervisory_delta(Position.LONG, is_option=True, option_type=OptionType.PUT,
                            underlying_price=0.06, strike_price=0.05, exercise_date=1.0,
                            supervisory_volatility=0.5)
    assert is_close(got, -float(norm.cdf(-d1)))


def test_supervisory_delta_cdo_tranche():
    from quantark.saccr.engines.maths import supervisory_delta
    from quantark.saccr.models.enums import Position
    a, d = 0.03, 0.07
    expected = 15.0 / ((1 + 14 * a) * (1 + 14 * d))
    assert is_close(
        supervisory_delta(Position.LONG, is_cdo_tranche=True,
                          attachment_point=a, detachment_point=d), -expected)


def test_supervisory_delta_rejects_bad_domain():
    from quantark.saccr.engines.maths import supervisory_delta
    from quantark.saccr.models.enums import Position
    with pytest.raises(ValidationError):
        supervisory_delta(Position.LONG, is_option=True, option_type=OptionType.CALL,
                          underlying_price=-1, strike_price=0.05, exercise_date=1.0,
                          supervisory_volatility=0.5)


def test_supervisory_duration_rejects_inverted_dates_even_with_floor():
    from quantark.saccr.engines.maths import supervisory_duration
    # raw end < start; flooring e to 10/250 must NOT hide it
    with pytest.raises(ValidationError):
        supervisory_duration(0.02, 0.01)
    with pytest.raises(ValidationError):
        supervisory_duration(-1, 5)


def test_legacy_cdf_was_materially_inaccurate():
    """Documents WHY the hand-rolled CDF was replaced: it was inaccurate (~3.7e-2),
    so SciPy is a correction (the two must NOT agree to high precision)."""
    def _legacy_cdf(x):  # verbatim copy of the removed source approximation
        a1, a2, a3, a4, a5, p = (0.254829592, -0.284496736, 1.421413741,
                                 -1.453152027, 1.061405429, 0.3275911)
        s = 1 if x >= 0 else -1
        x = abs(x)
        t = 1 / (1 + p * x)
        y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
        return 0.5 * (1 + s * y)
    d1 = (math.log(0.06 / 0.05) + 0.5 * 0.5 ** 2 * 1) / (0.5 * 1)
    assert abs(float(norm.cdf(-d1)) - _legacy_cdf(-d1)) > 1e-2
    assert is_close(float(norm.cdf(-d1)), 0.269395, abs_tol=1e-5)


# --------------------------------------------------------------------------- #
# Replacement cost & PFE                                                       #
# --------------------------------------------------------------------------- #
def test_replacement_cost():
    from quantark.saccr.engines.replacement_cost import (
        calculate_rc_unmargined, calculate_rc_margined)
    assert is_close(calculate_rc_unmargined(60, 0), 60)
    assert is_close(calculate_rc_unmargined(-20, 0), 0)
    assert is_close(calculate_rc_margined(80, 79.5, 0, 1, 0), 1)


def test_pfe_multiplier():
    from quantark.saccr.engines.pfe import calculate_multiplier, calculate_pfe
    assert is_close(calculate_multiplier(10, 0, 100), 1.0)        # under-collateralized
    m = calculate_multiplier(-20, 0, 282)
    assert is_close(m, 0.05 + 0.95 * math.exp(-20 / (2 * 0.95 * 282)))
    assert is_close(calculate_pfe(282, 0.965), 282 * 0.965)


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def test_trade_maturity_floor_and_validation():
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass
    t = SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, 0.0, currency="USD",
                   start_date=0, end_date=0.01, maturity=0.001)
    assert t.maturity == pytest.approx(10 / 250)
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, 0.0)  # missing currency
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.FX, 1e6, 0.0, currency_pair="EURUSD")  # not canonical
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.INTEREST_RATE, -1, 0.0, currency="USD")  # bad notional
    with pytest.raises(ValidationError):
        SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, 0.0, currency="USD",
                   start_date=5, end_date=2)  # end < start


def test_netting_set_aggregates_and_collateral():
    from quantark.saccr.models.netting_set import SACCRNettingSet
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass

    def _t(mv):
        return SACCRTrade("T", AssetClass.INTEREST_RATE, 1e6, mv, currency="USD",
                          start_date=0, end_date=5, maturity=5)

    ns = SACCRNettingSet("NS", [_t(30), _t(-20)], is_margined=True,
                         independent_collateral_received=150,
                         independent_collateral_posted=0, net_collateral=200, mpor_days=14)
    assert ns.market_value == pytest.approx(10)
    assert ns.collateral == pytest.approx(200)
    assert ns.nica == pytest.approx(150)
    assert ns.mpor_years == pytest.approx(14 / 250)

    # collateral from components when net_collateral not given (segregated excluded)
    ns2 = SACCRNettingSet("NS2", [_t(0)], variation_margin=10,
                          independent_collateral_received=5, independent_collateral_posted=2,
                          independent_collateral_posted_segregated=99)
    assert ns2.collateral == pytest.approx(13)   # 10 + 5 - 2 (segregated excluded)
    assert ns2.nica == pytest.approx(3)

    with pytest.raises(ValidationError):
        SACCRNettingSet("empty", [])


# --------------------------------------------------------------------------- #
# Per-asset-class add-ons (anchored to Basel sub-figures)                      #
# --------------------------------------------------------------------------- #
def _unmargined(trades):
    from quantark.saccr.models.netting_set import SACCRNettingSet
    return SACCRNettingSet("NS", trades, is_margined=False)


def test_ir_addon_matches_basel_example1():
    from quantark.saccr.engines.addons.interest_rate import InterestRateAddOn
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass, Position
    trades = [
        SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                   start_date=0, end_date=10, maturity=10, position=Position.LONG),
        SACCRTrade("IR2", AssetClass.INTEREST_RATE, 10_000, -20, currency="USD",
                   start_date=0, end_date=4, maturity=4, position=Position.SHORT),
        SACCRTrade("IR3", AssetClass.INTEREST_RATE, 5_000, 50, currency="EUR",
                   start_date=1, end_date=11, maturity=5.5, position=Position.LONG,
                   is_option=True, option_type=OptionType.PUT, underlying_price=0.06,
                   strike_price=0.05, exercise_date=1.0),
    ]
    total, bd = InterestRateAddOn().calculate_with_breakdown(trades, _unmargined(trades))
    assert total == pytest.approx(347, rel=0.05)
    assert {"IR:USD", "IR:EUR"} <= set(bd) and is_close(sum(bd.values()), total)


def test_credit_addon_matches_basel_example2():
    from quantark.saccr.engines.addons.credit import CreditAddOn
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass, Position, CreditRating, IndexGrade
    trades = [
        SACCRTrade("CR1", AssetClass.CREDIT, 10_000, 20, reference_entity="A",
                   credit_rating=CreditRating.AA, start_date=0, end_date=3, maturity=3,
                   position=Position.LONG),
        SACCRTrade("CR2", AssetClass.CREDIT, 10_000, -40, reference_entity="B",
                   credit_rating=CreditRating.BBB, start_date=0, end_date=6, maturity=6,
                   position=Position.SHORT),
        SACCRTrade("CR3", AssetClass.CREDIT, 10_000, 0, reference_entity="CDX.IG",
                   is_index=True, index_grade=IndexGrade.IG, start_date=0, end_date=5,
                   maturity=5, position=Position.LONG),
    ]
    total, _ = CreditAddOn().calculate_with_breakdown(trades, _unmargined(trades))
    assert total == pytest.approx(282, rel=0.05)


def test_commodity_addon_matches_basel_example3():
    from quantark.saccr.engines.addons.commodity import CommodityAddOn
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass, Position, CommodityType
    trades = [
        SACCRTrade("C1", AssetClass.COMMODITY, 10_000, -50, commodity_type=CommodityType.CRUDE_OIL,
                   start_date=0, end_date=0.75, maturity=0.75, position=Position.LONG),
        SACCRTrade("C2", AssetClass.COMMODITY, 20_000, -30, commodity_type=CommodityType.CRUDE_OIL,
                   start_date=0, end_date=2, maturity=2, position=Position.SHORT),
        SACCRTrade("C3", AssetClass.COMMODITY, 10_000, 100, commodity_type=CommodityType.SILVER,
                   start_date=0, end_date=5, maturity=5, position=Position.LONG),
    ]
    total, bd = CommodityAddOn().calculate_with_breakdown(trades, _unmargined(trades))
    assert total == pytest.approx(3841, rel=0.05)
    assert bd["COMMODITY:ENERGY"] == pytest.approx(2041, rel=0.05)
    assert bd["COMMODITY:METALS"] == pytest.approx(1800, rel=0.05)


def test_equity_single_name_addon_sign_and_sf():
    from quantark.saccr.engines.addons.equity import EquityAddOn
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass, Position
    t = SACCRTrade("EQ", AssetClass.EQUITY, 10_000, 0, reference_entity="AAPL",
                   start_date=0, end_date=1, maturity=1, position=Position.LONG)
    total, bd = EquityAddOn().calculate_with_breakdown([t], _unmargined([t]))
    # single-name SF=0.32, MF=1 (1y), delta=+1 -> addon ~ 0.32 * 10_000
    assert total == pytest.approx(0.32 * 10_000, rel=0.05) and "EQUITY" in bd


def test_fx_addon_value_and_pair_normalization_offset():
    from quantark.saccr.engines.addons.fx import FXAddOn
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.enums import AssetClass, Position
    one = SACCRTrade("FX1", AssetClass.FX, 10_000, 0, currency_pair="EUR/USD",
                     start_date=0, end_date=1, maturity=1, position=Position.LONG)
    total1, bd1 = FXAddOn().calculate_with_breakdown([one], _unmargined([one]))
    assert total1 == pytest.approx(0.04 * 10_000, rel=0.05)
    assert "FX:EUR/USD" in bd1
    # EUR/USD and USD/EUR are the SAME hedging set (normalized) and offset:
    two = [one, SACCRTrade("FX2", AssetClass.FX, 10_000, 0, currency_pair="USD/EUR",
                           start_date=0, end_date=1, maturity=1, position=Position.SHORT)]
    total2, bd2 = FXAddOn().calculate_with_breakdown(two, _unmargined(two))
    assert set(bd2) == {"FX:EUR/USD"}            # one normalized hedging set
    assert total2 == pytest.approx(0.0, abs=1e-6)  # long EUR/USD offsets short USD/EUR


def test_commodity_unknown_type_rejected_via_supervisory_lookup():
    from quantark.saccr.parameters.supervisory import SupervisoryParameters as SP
    from quantark.saccr.models.enums import AssetClass
    with pytest.raises(ValidationError):
        SP.get_supervisory_factor(AssetClass.COMMODITY, commodity_type=None)


# --------------------------------------------------------------------------- #
# Calculator orchestration & public API                                       #
# --------------------------------------------------------------------------- #
def test_calculator_example1_smoke():
    from quantark.saccr.calculator import SACCRCalculator
    from quantark.saccr.models.trade import SACCRTrade
    from quantark.saccr.models.netting_set import SACCRNettingSet
    from quantark.saccr.models.enums import AssetClass, Position
    ns = SACCRNettingSet("E1", [
        SACCRTrade("IR1", AssetClass.INTEREST_RATE, 10_000, 30, currency="USD",
                   start_date=0, end_date=10, maturity=10, position=Position.LONG),
    ], is_margined=False)
    r = SACCRCalculator().calculate(ns)
    assert r.alpha == 1.4 and r.is_margined is False and r.ead_capped is False
    assert "IR:USD" in r.addon_by_hedging_set
    assert r.ead_uncapped == r.ead


def test_public_api_surface():
    import quantark.saccr as m
    for name in ["SACCRCalculator", "SACCRResult", "SACCRTrade", "SACCRNettingSet",
                 "AssetClass", "Position", "CreditRating", "IndexGrade",
                 "CommodityHedgingSet", "CommodityType", "TransactionType", "OptionType"]:
        assert hasattr(m, name), name
    assert set(m.__all__) >= {"SACCRCalculator", "SACCRTrade", "SACCRNettingSet"}
