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


def term_env(r, q, sigma, spot=6000.0, tenors=(0.5, 1.0, 2.0)):
    """Flat values on TERM-STRUCTURE objects: exercises the term-aware code
    paths (node-aligned buckets etc.) while PVs match flat_env."""
    from datetime import datetime as _dt

    from quantark.param import SpotQuote
    from quantark.param.div.dividend_yield import TermStructureDividendYield
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.vol.vol_surface import TermStructureVolSurface
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import DayCountConvention

    tenors = [float(t) for t in tenors]
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=TermStructureVolSurface(
            times=tenors, vols=[sigma] * len(tenors)
        ),
        rate_curve=LinearRateCurve([(t, r) for t in tenors]),
        div_yield=TermStructureDividendYield(
            times=tenors, yields=[q] * len(tenors)
        ),
        valuation_date=_dt(2023, 1, 3),
        day_count_convention=DayCountConvention.ACT_365,
    )


def synthetic_quote_book(sigma=0.20, spot=6000.0, r=0.0356, f_target=5800.0):
    """(quotes, valuation_date, spot, rate_curve, carry_curve): one expiry,
    clean OTM quotes priced exactly at Black-76(sigma), plus one bad quote
    per cleaning rule (crossed, zero-bid, wide-spread, illiquid,
    above-upper-bound, no-price, ITM side)."""
    from datetime import datetime as _dt, timedelta as _td

    from quantark.param import FlatRateCurve
    from quantark.param.div.forward_carry_curve import ForwardCarryCurve
    from quantark.param.vol.marketquotes import OptionQuote, black_price

    import numpy as np

    valuation_date = _dt(2023, 1, 3)
    expiry = valuation_date + _td(days=91)
    t = 91 / 365.0
    rate_curve = FlatRateCurve(rate=r)
    b0 = float(np.log(f_target / spot))
    carry_curve = ForwardCarryCurve([(t, b0), (2 * t, 2 * b0)])
    df = float(np.exp(-r * t))
    liquid = dict(volume=100.0, open_interest=500.0)

    def _quote(strike, cp, mid=None, spread=0.02, **kw):
        is_call = cp.lower() in ("c", "call")
        if mid is None:
            mid = black_price(f_target, strike, t, sigma, df, is_call)
        params = dict(bid=mid - spread / 2, ask=mid + spread / 2, **liquid)
        params.update(kw)
        return OptionQuote(expiry=expiry, strike=strike, call_put=cp, **params)

    quotes = [
        _quote(6000.0, "C"),                       # clean OTM call
        _quote(6400.0, "call"),                    # clean OTM call (alias)
        _quote(5600.0, "P"),                       # clean OTM put
        _quote(5200.0, "put"),                     # clean OTM put (parity pair)
        _quote(5200.0, "C"),                       # ITM call -> itm_side
        _quote(6200.0, "C", bid=5.0, ask=4.0),     # crossed
        _quote(6300.0, "C", bid=0.0, ask=4.0),     # zero bid
        _quote(6800.0, "C", mid=20.0, spread=15.0),   # wide spread (75%)
        _quote(5000.0, "P", volume=0.0, open_interest=1.0),  # illiquid
        _quote(5600.0, "P", mid=df * 5600.0 * 1.05, spread=1.0),  # > upper bound
        OptionQuote(expiry=expiry, strike=6100.0, call_put="C",
                    last=30.0, **liquid),          # no bid/ask -> no_price
    ]
    return quotes, valuation_date, spot, rate_curve, carry_curve


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


def synthetic_svi_slices():
    """Two calendar-consistent SVISliceFit objects (T=0.25, 0.5) from known
    params, scaled so w(y, 0.5) > w(y, 0.25) on y in [-1.5, 1.5]."""
    import numpy as np

    from quantark.param.vol.svi import SVIParams, fit_svi_slice

    y = np.linspace(-0.8, 0.8, 15)
    p1 = SVIParams(a=0.008, b=0.10, rho=-0.3, m=0.02, sigma=0.2)
    p2 = SVIParams(a=0.020, b=0.16, rho=-0.3, m=0.02, sigma=0.25)
    f1 = fit_svi_slice(y, p1.total_variance(y), None, expiry_t=0.25)
    f2 = fit_svi_slice(y, p2.total_variance(y), None, expiry_t=0.5)
    return [f1, f2]
