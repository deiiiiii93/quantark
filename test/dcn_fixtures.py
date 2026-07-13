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


PRODUCT_KEYS = (
    "notional", "initial_price", "coupon_barrier_ratio", "ko_barrier_ratio",
    "ki_barrier_ratio", "ki_put_strike_ratio", "coupon_rate", "ko_coupon_rate",
    "participation", "coupon_counted_days", "coupon_days_denom",
    "settlement_date",
)


def make_dcn(contract: dict, **overrides):
    from quantark.asset.equity.product.option.dcn_option import (
        DCNDirection, DCNOption,
    )
    from quantark.asset.equity.product.option.dcn_schedule import (
        build_dcn_schedule,
    )

    c = dict(contract, **{k: v for k, v in overrides.items() if k in contract})
    schedule = build_dcn_schedule(**schedule_kwargs(c))
    kwargs = {k: c[k] for k in PRODUCT_KEYS}
    kwargs.update(
        direction=overrides.get("direction", DCNDirection.BUYER),
        schedule=schedule,
        knocked_in_at_valuation=overrides.get("knocked_in_at_valuation", False),
    )
    for k, v in overrides.items():
        if k in kwargs:
            kwargs[k] = v
    return DCNOption(**kwargs)


def flat_env(r, q, sigma, spot=6000.0):
    from datetime import datetime as _dt

    from quantark.param import (
        ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
    )
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import DayCountConvention

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=sigma),
        rate_curve=FlatRateCurve(rate=r),
        div_yield=ContinuousDividendYield(div_yield=q),
        valuation_date=_dt(2023, 1, 3),
        day_count_convention=DayCountConvention.ACT_365,
    )
