"""Small autocallables on the quoted clock (k/12) and on dates, for the accrual-time tests."""
from copy import deepcopy
from datetime import datetime, timedelta

from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PostKOScheduleMode

T0 = datetime(2026, 1, 5)


def times(*ts, level=103.0):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in ts])


def dates(*days, level=103.0, start=T0):
    return ObservationSchedule(records=[ObservationRecord(observation_date=start + timedelta(days=d), barrier=level)
                                        for d in days])


def env(valuation=T0, spot=100.0, **kw):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.0), valuation_date=valuation, **kw)


def barriers(ko, ki=None, rate=0.12):
    return BarrierConfig(ko_barrier=103.0, ko_rate=rate, ko_observation_schedule=ko,
                         ki_barrier=None if ki is None else 75.0, ki_observation_schedule=ki)


def snowball(ko=None, ki=None, accrual=None, rate=0.12, maturity=0.25, **kw):
    ko = ko if ko is not None else times(1 / 12, 2 / 12, 3 / 12)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barriers(ko, ki, rate),
                          payoff_config=kw.pop("payoff_config", PayoffConfig(rebate_rate=0.05)),
                          accrual_config=accrual or AccrualConfig(), maturity=maturity, **kw)


def phoenix(ko=None, ki=None, accrual=None, coupon=None, maturity=0.25, **kw):
    ko = ko if ko is not None else times(1 / 12, 2 / 12, 3 / 12)
    return PhoenixOption(initial_price=100.0, strike=100.0, barrier_config=barriers(ko, ki),
                         coupon_config=coupon or CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12),
                         accrual_config=accrual or AccrualConfig(), maturity=maturity, **kw)


def ko_reset(mode=PostKOScheduleMode.ABSOLUTE, pre=None, ki=None, post=None, accrual=None, maturity=None, **kw):
    pre = pre if pre is not None else times(0.25, 0.5)
    ki = ki if ki is not None else times(0.25, 0.5, level=75.0)
    post = post if post is not None else times(0.25, 0.5, 0.75, level=100.0)
    if maturity is None:
        maturity = 1.25 if mode is PostKOScheduleMode.REBASED else 0.75
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barriers(pre, ki),
        post_barrier_config=BarrierConfig(ko_barrier=100.0, ko_rate=0.05, ko_observation_schedule=post),
        payoff_config=kw.pop("payoff_config", PayoffConfig(rebate_rate=0.05)), accrual_config=accrual or AccrualConfig(),
        maturity=maturity, post_ko_mode=mode, **kw)


def periods(schedule):
    return [r.accrual for r in schedule.records]


def rebuilt(product, **changes):
    """A copy with changed terms, the way this codebase copies a product: deepcopy, set, validate()."""
    copy = deepcopy(product)
    for name, value in changes.items():
        setattr(copy, name, value)
    copy.validate()
    return copy
