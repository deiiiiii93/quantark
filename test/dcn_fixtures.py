"""Shared DCN sample-contract fixtures (values from the DCN problem statement)."""
from datetime import datetime

from quantark.util.calendar import CalendarType, create_calendar

SSE = create_calendar(CalendarType.CHINA_SSE)

FLAT = dict(r=0.0356, q=0.1406, sigma=0.184)

DCN_A = dict(
    initial_date=datetime(2023, 1, 3),
    valuation_date=datetime(2023, 1, 3),
    maturity_date=datetime(2025, 1, 3),
    tenor_months=24,
    lock_months=3,
    ko_lock_months=3,
    coupon_settlement_offset=2,
    ko_settlement_offset=2,
    settlement_date=datetime(2025, 1, 7),
    notional=1_000_000.0,
    initial_price=6000.0,
    coupon_barrier_ratio=0.80,
    ko_barrier_ratio=1.00,
    ki_barrier_ratio=0.75,
    ki_put_strike_ratio=1.10,
    coupon_rate=0.12,
    ko_coupon_rate=0.12,
    participation=1.0,
    coupon_counted_days=30,
    coupon_days_denom=360,
)

DCN_B = dict(
    DCN_A,
    maturity_date=datetime(2026, 1, 5),
    tenor_months=36,
    lock_months=3,
    ko_lock_months=6,
    coupon_settlement_offset=0,
    ko_settlement_offset=0,
    settlement_date=datetime(2026, 1, 7),
    ki_put_strike_ratio=1.15,
)

SCHEDULE_KEYS = (
    "initial_date", "maturity_date", "tenor_months", "lock_months",
    "ko_lock_months", "coupon_settlement_offset", "ko_settlement_offset",
    "valuation_date", "settlement_date",
)


def schedule_kwargs(contract: dict) -> dict:
    kw = {k: contract[k] for k in SCHEDULE_KEYS}
    kw["calendar"] = SSE
    return kw
