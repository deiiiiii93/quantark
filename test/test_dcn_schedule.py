"""DCN schedule fingerprints — exact values from the problem statement."""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.util.exceptions import ValidationError

from dcn_fixtures import DCN_A, DCN_B, SSE, schedule_kwargs


def _build(contract):
    return build_dcn_schedule(**schedule_kwargs(contract))


def test_dcn_a_fingerprints():
    s = _build(DCN_A)
    assert len(s.monthly) == 22
    assert s.monthly[0].observation_date == datetime(2023, 4, 3)
    assert s.monthly[0].is_coupon_obs and s.monthly[0].is_ko_obs
    assert s.monthly[-1].observation_date == datetime(2025, 1, 3)
    assert s.monthly[-1].coupon_payment_date == datetime(2025, 1, 7)
    assert s.monthly[-1].ko_payment_date == datetime(2025, 1, 7)


def test_dcn_b_fingerprints():
    s = _build(DCN_B)
    assert len(s.monthly) == 34
    assert s.monthly[0].observation_date == datetime(2023, 4, 3)
    # coupon-only lock window: first KO obs is k=6 -> 2023-07-03
    coupon_only = [m.observation_date for m in s.monthly
                   if m.is_coupon_obs and not m.is_ko_obs]
    assert coupon_only == [
        datetime(2023, 4, 3), datetime(2023, 5, 4), datetime(2023, 6, 5)
    ]
    first_ko = next(m for m in s.monthly if m.is_ko_obs)
    assert first_ko.observation_date == datetime(2023, 7, 3)
    assert s.monthly[-1].observation_date == datetime(2026, 1, 5)
    assert s.monthly[-1].coupon_payment_date == datetime(2026, 1, 5)  # offset 0


def test_no_merges_in_samples_and_month_indices_complete():
    for c in (DCN_A, DCN_B):
        s = _build(c)
        ks = [k for m in s.monthly for k in m.month_indices]
        assert ks == list(range(c["lock_months"], c["tenor_months"] + 1))
        assert all(len(m.month_indices) == 1 for m in s.monthly)


def test_maturity_mismatch_raises():
    kw = schedule_kwargs(DCN_A)
    kw["maturity_date"] = datetime(2025, 1, 6)  # generated last obs is 2025-01-03
    with pytest.raises(ValidationError):
        build_dcn_schedule(**kw)


def test_daily_ki_dates_bounds_and_membership():
    s = _build(DCN_A)
    assert s.daily_ki_dates[0] == datetime(2023, 1, 3)   # valuation (trading day)
    assert s.daily_ki_dates[-1] == datetime(2025, 1, 3)  # maturity obs
    dates = set(s.daily_ki_dates)
    assert all(m.observation_date in dates for m in s.monthly)
    assert all(SSE.is_business_day(d) for d in s.daily_ki_dates)


def test_eom_clamp_regression():
    # Jan-31 anchor: relativedelta keeps day-of-month with end-of-month clamp
    from quantark.asset.equity.product.option.dcn_schedule import _benchmark_dates
    bms = _benchmark_dates(datetime(2023, 1, 31), lock_months=1, tenor_months=3)
    assert [b.date().isoformat() for _, b in bms] == [
        "2023-02-28", "2023-03-31", "2023-04-30"
    ]


def test_coverage_guard_raises_beyond_holiday_file():
    kw = schedule_kwargs(DCN_A)
    kw["initial_date"] = datetime(2029, 1, 3)
    kw["valuation_date"] = datetime(2029, 1, 3)
    kw["maturity_date"] = datetime(2031, 1, 3)  # beyond 2030 holiday coverage
    kw["settlement_date"] = datetime(2031, 1, 7)
    with pytest.raises(ValidationError):
        build_dcn_schedule(**kw)


def test_spec_rebuild_round_trip():
    import dataclasses
    s = _build(DCN_A)
    rebuilt = s.spec.build(s.calendar)
    assert rebuilt.monthly == s.monthly
    rolled = dataclasses.replace(
        s.spec, valuation_date=datetime(2023, 6, 1)
    ).build(s.calendar)
    assert rolled.monthly[0].observation_date == datetime(2023, 6, 5)
    assert rolled.daily_ki_dates[0] == datetime(2023, 6, 1)


def test_to_dataframe_columns():
    df = _build(DCN_A).to_dataframe()
    assert list(df.columns) == [
        "month_indices", "benchmark_date", "observation_date",
        "is_coupon_obs", "is_ko_obs", "coupon_payment_date", "ko_payment_date",
    ]
    assert len(df) == 22
