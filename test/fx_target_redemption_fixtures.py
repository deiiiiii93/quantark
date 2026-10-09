"""Shared builders for the FX TARN / TARF lifecycle tests (spec 2026-10-09-fx-tarn-lifecycle-design.md, section 2.2
fixture: twelve monthly fixings at k/12, TARN notional 100 / coupon 12% / strike 1.10, TARF strike 1.15 / notional 1,
EURUSD 1.20, r_d 5%, r_f 3%, vol 10%)."""
from datetime import datetime, timedelta

from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import PendingPayment, TargetRedemptionState
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.calendar.business_calendar import CalendarType, create_calendar

PAIR = CurrencyPair("EUR", "USD")
VAL = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]
INCEPTION = TargetRedemptionState(0.0)


def env(spot=1.20, val=VAL, vol=0.10, dc=DayCountConvention.CALENDAR_DAYS, calendar=None, spot_days=0):
    return FxPricingEnvironment(
        valuation_date=val, spot_quote=SpotQuote(spot=spot), domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03), vol_surface=FlatVolSurface(volatility=vol),
        day_count_convention=dc, calendar=calendar, spot_days=spot_days,
    )


def mc(n=20_000, seed=7, method=None):
    kw = dict(num_paths=n, seed=seed)
    if method is not None:
        kw["method"] = method
    return FxMCParams(**kw)


def note(fix=FIX, state=INCEPTION, **kw):
    """Numeric TARN; the first listed period starts at 0.0 unless the caller states accrual_factors or accrual_start."""
    base = dict(fixing_times=fix, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR, state=state)
    if "accrual_factors" not in kw:
        base["accrual_start"] = 0.0
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def engine_for(product, n=20_000, seed=7, method=None):
    params = mc(n=n, seed=seed, method=method)
    if isinstance(product, FxTargetRedemptionNote):
        return FxTargetRedemptionNoteMCEngine(params)
    return FxTarnForwardMCEngine(params)


def tarf(fix=FIX, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_times=fix, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


# --- dated form (spec 4.2 / 4.3) -----------------------------------------------------------------------------------

FIX_DATES = [datetime(2026 + (6 + m) // 12, (6 + m) % 12 + 1, 15) for m in range(12)]  # 2026-07-15 .. 2027-06-15
LAG2 = [d + timedelta(days=2) for d in FIX_DATES]
TARGET_CAL = create_calendar(CalendarType.TARGET, year_range=(2026, 2028))


def following(d):
    """Test-side business-day adjustment (the product takes dates as stated)."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def dated_note(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(fixing_dates=fixing_dates, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR,
                state=state)
    if "accrual_factors" not in kw:
        base["initial_date"] = VAL
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def dated_tarf(fixing_dates=FIX_DATES, state=INCEPTION, **kw):
    base = dict(strike=1.15, fixing_dates=fixing_dates, notional=1.0, currency_pair=PAIR, state=state)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


def _numeric_twin(dated, fx_env):
    """Spec 4.3: the numeric contract with the dated contract's unbanked fixings, resolved factors and state."""
    view = dated.resolve(fx_env, warn_missing_fixings=False)
    state = TargetRedemptionState(view.accumulated,
                                  pending=tuple(PendingPayment(a, pay_time=t) for t, a in view.pending))
    common = dict(fixing_times=list(view.fixing_times), pay_times=list(view.pay_times), target=dated.target,
                  currency_pair=dated.currency_pair, state=state)
    if isinstance(dated, FxTargetRedemptionNote):
        return FxTargetRedemptionNote(coupon_rate=dated.coupon_rate, notional=dated.notional, strike=dated.strike,
                                      is_above=dated.is_above, lower=dated.lower, upper=dated.upper,
                                      include_principal=dated.include_principal,
                                      accrual_factors=list(view.accrual_factors), **common)
    return FxTargetRedemptionForward(strike=dated.strike, notional=dated.notional, gearing=dated.gearing,
                                     is_buy_base=dated.is_buy_base, **common)
