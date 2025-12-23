
import pytest
from datetime import datetime
from asset.equity.product.option.asian_option import AsianObservationRecord
from priceenv import PricingEnvironment
from param.rrf import FlatRateCurve
from util.exceptions import ValidationError

def test_asian_observation_record_time_resolution():
    # Test resolution using observation_time
    record = AsianObservationRecord(observation_time=0.5)
    assert record.resolve_time(None) == 0.5
    assert record.is_observed() is False

def test_asian_observation_record_date_resolution():
    # Test resolution using observation_date
    val_date = datetime(2025, 1, 1)
    obs_date = datetime(2025, 7, 1)
    
    # We need a real PricingEnvironment to resolve dates
    rate_curve = FlatRateCurve(rate=0.05)
    pe = PricingEnvironment(rate_curve=rate_curve, valuation_date=val_date)
    
    record = AsianObservationRecord(observation_date=obs_date)
    # 2025-07-01 is roughly 0.5 years from 2025-01-01
    t = record.resolve_time(pe)
    assert 0.49 < t < 0.51

def test_asian_observation_record_validation():
    # Test validation
    record = AsianObservationRecord(observation_time=0.5, observed_price=100.0)
    record.validate()
    assert record.is_observed() is True
    
    # Invalid observed price
    invalid_record = AsianObservationRecord(observation_time=0.5, observed_price=-10.0)
    with pytest.raises(ValidationError, match="observed_price must be positive"):
        invalid_record.validate()
    
    # Missing both time and date
    empty_record = AsianObservationRecord()
    with pytest.raises(ValidationError, match="must provide observation_time or observation_date"):
        empty_record.validate()
