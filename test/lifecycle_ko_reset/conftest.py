"""Shared builders for the KO-reset lifecycle tests (spec 2026-10-08)."""
import pandas as pd

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType

START = pd.Timestamp("2026-01-05")


def env(spot=100.0, date=START):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(0.25),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=pd.Timestamp(date).to_pydatetime())


def schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def snowball(ko_times, ki_times, maturity, *, annualized=False, ko=103.0, ki=75.0):
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule(ko_times, ko), ki_barrier=ki,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule(ki_times, ki)),
        accrual_config=AccrualConfig(is_annualized=annualized),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))
