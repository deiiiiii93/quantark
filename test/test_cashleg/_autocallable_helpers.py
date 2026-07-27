"""Shared builders for AutocallableCashLeg tests (env / products / engines / legs)."""

from datetime import datetime

import numpy as np

from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.cashleg.base import LegDirection
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg,
    AutocallableLegType,
    PvFormula,
)


def make_env(spot=100.0, vol=0.20, rate=0.03, div_yield=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def make_snowball(ko_dates=(0.5, 1.0), ko_barrier=103.0, ko_rate=0.15,
                  ki_barrier=75.0, maturity=1.0):
    barrier = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier,
        contract_multiplier=1.0, maturity=maturity, is_reverse=False,
    )


def make_phoenix(ko_dates=(0.5, 1.0), ko_barrier=105.0,
                 coupon_barrier=(80.0, 80.0), memory=False,
                 coupon_pay=CouponPayType.INSTANT, maturity=1.0,
                 ki_barrier=None, disable_ko_after_ki=False):
    barrier = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=ki_barrier,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_barrier else ObservationType.DISCRETE
        ),
        ki_continuous=ki_barrier is not None,
        disable_ko_after_ki=disable_ko_after_ki,
    )
    coupon = CouponBarrierConfig(
        coupon_barrier=list(coupon_barrier), coupon_rate=0.02,
        coupon_pay_type=coupon_pay,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=memory,
    )
    return PhoenixOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier,
        coupon_config=coupon,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        contract_multiplier=1.0, maturity=maturity,
    )


_ENGINES = {
    ("snowball", "mc"): lambda: SnowballMCEngine(params=MCParams(num_paths=60_000, seed=7)),
    ("snowball", "pde"): lambda: SnowballPDESolver(params=PDEParams()),
    ("snowball", "quad"): lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001)),
    ("phoenix", "mc"): lambda: PhoenixMCEngine(params=MCParams(num_paths=60_000, seed=7)),
    ("phoenix", "pde"): lambda: PhoenixPDESolver(params=PDEParams()),
    ("phoenix", "quad"): lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001)),
}


def make_engine(kind, asset="snowball"):
    return _ENGINES[(asset, kind)]()


def future_event_times(product, engine, env):
    """Parent's filtered future observation grid the leg must align to."""
    result = engine.price_with_events(product, env, emit_distribution=True)
    return np.asarray(result.event_distribution.event_times, dtype=float)


def make_margin_leg(obs, notional=1_000_000.0, rate=0.04,
                    direction=LegDirection.BUYER_RECEIVES):
    obs = [float(t) for t in obs]
    n = len(obs)
    return AutocallableCashLeg(
        direction=direction, leg_type=AutocallableLegType.MARGIN,
        notional=notional, rate=rate,
        observation_schedule=tuple(obs),
        accrual_factors=tuple(np.linspace(0.25, 1.0, n)),
        settlement_schedule=tuple(obs),
        terminal_accrual_factor=1.0, terminal_settlement_time=obs[-1],
        pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF,
    )
