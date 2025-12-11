"""
Test date-based accrual for observation records with the new data structure.
"""

from datetime import datetime
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ResolvedObservationRecord,
    ObservationSchedule,
)
from util.calendar import DayCountConvention
from util.enum import TenorEnd


def test_observation_record_new_fields():
    """Test that ObservationRecord accepts new accrual fields."""
    rec = ObservationRecord(
        observation_time=0.25,
        barrier=100.0,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=datetime(2024, 1, 1),
        settlement_date=datetime(2024, 3, 31),
        maturity_date=datetime(2024, 12, 31),
        tenor_end=TenorEnd.SETTLEMENT,
        day_count_convention=DayCountConvention.ACT_365,
    )

    assert rec.initial_date == datetime(2024, 1, 1)
    assert rec.settlement_date == datetime(2024, 3, 31)
    assert rec.maturity_date == datetime(2024, 12, 31)
    assert rec.tenor_end == TenorEnd.SETTLEMENT
    assert rec.day_count_convention == DayCountConvention.ACT_365
    assert rec.is_rate_annualized == True
    print("✓ ObservationRecord new fields test passed")


def test_resolved_observation_record_new_fields():
    """Test that ResolvedObservationRecord contains only final values."""
    resolved = ResolvedObservationRecord(
        observation_time=0.25,
        barrier=100.0,
        payoff=5.0,
        settlement_time=0.30,
    )

    # Check that essential fields are present
    assert resolved.observation_time == 0.25
    assert resolved.barrier == 100.0
    assert resolved.payoff == 5.0
    assert resolved.settlement_time == 0.30

    # Verify that intermediate calculation fields are NOT in resolved record
    assert not hasattr(resolved, 'return_rate')
    assert not hasattr(resolved, 'day_count_fraction')
    assert not hasattr(resolved, 'is_rate_annualized')

    print("✓ ResolvedObservationRecord new fields test passed")


def test_date_validation_for_annualized_rates():
    """Test that validation catches missing dates for annualized rates."""
    # Should fail: annualized rate without sufficient date info
    rec = ObservationRecord(
        observation_time=0.25,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=datetime(2024, 1, 1),
        # Missing: tenor_end and corresponding dates
    )

    try:
        rec.validate()
        assert False, "Should have raised ValidationError"
    except Exception as e:
        assert "Annualized return_rate requires" in str(e)
        print("✓ Date validation test passed")


def test_payoff_calculation_flow():
    """Test the payoff calculation flow as specified."""

    # Test 1: Explicit payoff
    rec1 = ObservationRecord(
        observation_time=0.25,
        payoff=0.05
    )
    assert rec1.get_payoff(0.0) == 0.05

    # Test 2: Non-annualized return_rate
    rec2 = ObservationRecord(
        observation_time=0.25,
        return_rate=0.03,
        is_rate_annualized=False
    )
    assert rec2.get_payoff(0.0) == 0.03

    # Test 3: Pre-calculated day_count_fraction
    rec3 = ObservationRecord(
        observation_time=0.25,
        return_rate=0.12,
        is_rate_annualized=True,
        day_count_fraction=0.25
    )
    assert abs(rec3.get_payoff(0.0) - 0.03) < 1e-8

    # Test 4: Calculate day_count_fraction with tenor_end
    rec4 = ObservationRecord(
        observation_date=datetime(2024, 6, 30),
        observation_time=0.5,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=datetime(2024, 1, 1),
        settlement_date=datetime(2024, 3, 31),
        tenor_end=TenorEnd.SETTLEMENT,
        day_count_convention=DayCountConvention.ACT_365
    )
    payoff4 = rec4.get_payoff(0.0)
    expected4 = 90 / 365 * 0.12  # 90 days from Jan 1 to Mar 31
    assert abs(payoff4 - expected4) < 1e-6

    print("✓ Payoff calculation flow test passed")


def test_backward_compatibility():
    """Test that existing code without new fields continues to work."""
    rec = ObservationRecord(
        observation_time=0.25,
        barrier=100.0,
    )

    # Should not raise
    rec.validate()

    # New fields should have defaults
    assert rec.initial_date is None
    assert rec.settlement_date is None
    assert rec.maturity_date is None
    assert rec.tenor_end is None
    assert rec.day_count_convention is None
    assert rec.day_count_fraction is None
    assert rec.is_rate_annualized == False  # Default
    print("✓ Backward compatibility test passed")


def test_tenor_end_options():
    """Test different tenor_end options."""
    base_date = datetime(2024, 1, 1)
    obs_date = datetime(2024, 12, 31)

    # SETTLEMENT
    rec_settle = ObservationRecord(
        observation_date=obs_date,
        observation_time=1.0,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=base_date,
        settlement_date=datetime(2024, 3, 31),
        tenor_end=TenorEnd.SETTLEMENT,
        day_count_convention=DayCountConvention.ACT_365
    )
    payoff_settle = rec_settle.get_payoff(0.0)
    expected_settle = 90 / 365 * 0.12
    assert abs(payoff_settle - expected_settle) < 1e-6

    # MATURITY
    rec_mat = ObservationRecord(
        observation_date=obs_date,
        observation_time=1.0,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=base_date,
        maturity_date=datetime(2024, 6, 30),
        tenor_end=TenorEnd.MATURITY,
        day_count_convention=DayCountConvention.ACT_365
    )
    payoff_mat = rec_mat.get_payoff(0.0)
    expected_mat = 181 / 365 * 0.12  # Jan 1 to Jun 30 (181 days)
    assert abs(payoff_mat - expected_mat) < 1e-6

    # Default to observation_date
    rec_obs = ObservationRecord(
        observation_date=obs_date,
        observation_time=1.0,
        return_rate=0.12,
        is_rate_annualized=True,
        initial_date=base_date,
        day_count_convention=DayCountConvention.ACT_365
    )
    payoff_obs = rec_obs.get_payoff(0.0)
    expected_obs = 365 / 365 * 0.12  # Jan 1 to Dec 31 is 365 days
    assert abs(payoff_obs - expected_obs) < 1e-6

    print("✓ Tenor end options test passed")


def test_settlement_time_resolution():
    """Test that settlement_time is properly resolved for discounting."""
    from priceenv import PricingEnvironment
    from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )

    # Test 1: With explicit settlement_date
    rec1 = ObservationRecord(
        observation_date=datetime(2024, 6, 30),
        barrier=95.0,
        payoff=5.0,
        settlement_date=datetime(2024, 7, 15),  # Settles 15 days after observation
    )

    schedule = ObservationSchedule(records=[rec1])
    resolved = schedule.resolve(pricing_env)

    assert len(resolved) == 1
    assert resolved[0].observation_time > 0  # Around 0.496 years
    assert resolved[0].settlement_time > resolved[0].observation_time  # Settlement after observation
    assert resolved[0].payoff == 5.0

    # Test 2: Without settlement_date (defaults to observation_time)
    rec2 = ObservationRecord(
        observation_date=datetime(2024, 6, 30),
        barrier=95.0,
        payoff=5.0,
        # No settlement_date specified
    )

    schedule2 = ObservationSchedule(records=[rec2])
    resolved2 = schedule2.resolve(pricing_env)

    assert len(resolved2) == 1
    assert resolved2[0].settlement_time == resolved2[0].observation_time  # Same time

    print("✓ Settlement time resolution test passed")


if __name__ == "__main__":
    test_observation_record_new_fields()
    test_resolved_observation_record_new_fields()
    test_date_validation_for_annualized_rates()
    test_payoff_calculation_flow()
    test_backward_compatibility()
    test_tenor_end_options()
    test_settlement_time_resolution()
    print("\n✅ All tests passed!")
