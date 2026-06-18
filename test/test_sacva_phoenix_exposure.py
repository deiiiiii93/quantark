"""SA-CVA stateful exposure for Phoenix memory-coupon autocallables (#1b).

Phoenix pays coupons WHILE ALIVE (conditional on a coupon barrier, with optional
memory of missed coupons), so the state is 3-D: (spot, KI, accumulated-missed-coupon
count). The QUAD engine records an EX-coupon per-memory surface stack; the
PhoenixStateMachine tracks each path's KI / KO / memory and reprice_phoenix selects the
matching (KI, memory) slice.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.sacva.exposure.phoenix_surface import build_phoenix_surface
from quantark.sacva.exposure.repricer import reprice_phoenix
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.exposure.statemachine import PhoenixStateMachine
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVATrade
from quantark.util.enum import CouponPayType, ObservationType


def _env(spot=100.0, vol=0.20, rate=0.05, div=0.02):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="UND"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1))


def _phoenix(memory=True, coupon_barrier=90.0, ko_barrier=105.0, ki=None,
             coupon_rate=0.03, pay_type=CouponPayType.INSTANT,
             ko_dates=(0.25, 0.5, 0.75, 1.0)):
    barrier = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=ki,
        ki_observation_type=(ObservationType.CONTINUOUS if ki is not None
                             else ObservationType.DISCRETE),
        ki_continuous=ki is not None)
    coupon = CouponBarrierConfig(
        coupon_barrier=coupon_barrier, coupon_rate=coupon_rate,
        coupon_pay_type=pay_type, memory_coupon=memory,
        fixed_coupon_year_fraction=0.25)
    return PhoenixOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier, coupon_config=coupon,
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def _trade(product, quantity=1.0, grid_points=301):
    return CVATrade(
        trade_id="phx", product=product,
        engine=PhoenixQuadEngine(params=QuadParams(grid_points=grid_points)),
        env=_env(), quantity=quantity, trade_currency="USD")


def _counterparty(trades):
    return Counterparty(
        name="CP", netting_sets=[NettingSet("ns", list(trades))],
        credit_curve=PillarHazardCurve([0.5, 1.0, 3.0, 5.0, 10.0], [0.02] * 5,
                                       recovery_rate=0.4),
        bucket=2, credit_quality=CreditQuality.IG)


def test_phoenix_surface_builds_memory_stack():
    spec = build_phoenix_surface(_trade(_phoenix(memory=True)))
    assert spec.use_memory is True
    # at a later observation the surface must carry several memory states
    last_obs = spec.obs_idx[-1]
    t = float(spec.times[last_obs])
    keys = spec.surface.grids[t]
    mems = sorted({k[1] for k in keys if k[0] == "alive"})
    assert mems == list(range(len(mems))) and len(mems) >= 2


def test_phoenix_t0_equals_price_memory():
    trade = _trade(_phoenix(memory=True), quantity=6.0)
    price0 = PhoenixQuadEngine(params=QuadParams(grid_points=301)).price(
        trade.product, trade.env)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=8)).compute(
            _counterparty([trade]))
    assert profile.times[0] == 0.0
    assert profile.epe_discounted[0] == pytest.approx(6.0 * price0, rel=3e-3)
    assert np.all(np.isfinite(profile.epe_discounted))
    assert profile.regulatory_eligible and len(profile.times) > 1


def test_phoenix_t0_equals_price_no_memory():
    trade = _trade(_phoenix(memory=False), quantity=2.0)
    price0 = PhoenixQuadEngine(params=QuadParams(grid_points=301)).price(
        trade.product, trade.env)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=8)).compute(
            _counterparty([trade]))
    assert profile.epe_discounted[0] == pytest.approx(2.0 * price0, rel=3e-3)


def _independent_cashflows(spots, spec, product, env, memory):
    """Re-derive the Phoenix payoff pathwise, INDEPENDENTLY of the exposure code, from
    the simulated spots: cf[p, j] = (undiscounted) cashflow received at observation node
    j. Validates the memory/KO/coupon model against the engine price. No-KI only (so the
    maturity payoff is the spot-independent V0 base)."""
    n_paths, n_t = spots.shape
    obs_idx = spec.obs_idx
    yf = product.get_coupon_period_year_fractions(
        [float(spec.times[j]) for j in obs_idx])
    coupon_amt = np.array([product.get_coupon_payoff(i, year_fraction=yf[i])
                           for i in range(len(obs_idx))])
    base_ko = np.array([product.get_ko_payoff(0.01, i, 0.0)  # spot below coupon -> no cur
                        for i in range(len(obs_idx))])
    v0_base = float(product.get_maturity_payoff_v0(100.0, 0.0))
    cb, kb = spec.coupon_barrier, spec.ko_barrier
    cf = np.zeros((n_paths, n_t))
    acc = np.zeros(n_paths)
    dead = np.zeros(n_paths, dtype=bool)
    for idx, j in enumerate(obs_idx):
        is_last = idx == len(obs_idx) - 1
        s = spots[:, j]
        alive = ~dead
        cond = s >= cb
        ko = alive & (s >= kb)
        pay = alive & ~ko & cond
        accm = acc if memory else np.zeros_like(acc)
        ko_cur = np.where(cond, coupon_amt[idx], 0.0)
        cf[:, j] += np.where(ko, base_ko[idx] + accm + ko_cur, 0.0)
        cf[:, j] += np.where(pay, coupon_amt[idx] + accm, 0.0)
        if is_last:
            cf[:, j] += np.where(alive & ~ko, v0_base, 0.0)
        if memory:
            miss = alive & ~ko & ~cond
            acc = np.where(pay, 0.0, np.where(miss, acc + coupon_amt[idx], acc))
        dead = dead | ko
        if is_last:
            dead = dead | alive
    return cf


def test_phoenix_realized_cashflow_mean_equals_price():
    # the mean discounted realized payoff (independent model) must equal the engine price
    # -> validates the memory/coupon/KO model and the simulated measure.
    product = _phoenix(memory=True, coupon_barrier=100.0, ko_barrier=130.0, ki=None)
    trade = _trade(product)
    env = trade.env
    price0 = PhoenixQuadEngine(params=QuadParams(grid_points=301)).price(product, env)
    spec = build_phoenix_surface(trade)
    times = spec.times
    from quantark.sacva.exposure.paths import StatePathGenerator
    spots = StatePathGenerator(
        keys=["UND"], spots=[100.0], vols=[spec.vol],
        rates=[float(env.get_rate(float(times[-1])))],
        divs=[float(env.get_div_yield(float(times[-1])))],
        corr=[[1.0]], grid_times=times, num_paths=60000, seed=21).generate()["UND"]
    df = np.array([float(env.get_discount_factor(float(t))) for t in times])
    cf = _independent_cashflows(spots, spec, product, env, memory=True)
    realized_pv = (cf * df[None, :]).sum(axis=1).mean()
    assert realized_pv == pytest.approx(price0, rel=5e-3)


def test_phoenix_value_process_martingale_memory():
    # df_j * V_ex(t_j) [production surface + memory state machine] + banked cashflows up
    # to t_j (independent model) == price0 at an interior observation -> ties the per-
    # memory surface selection to the realized coupons (no value leaked across memory).
    product = _phoenix(memory=True, coupon_barrier=100.0, ko_barrier=130.0, ki=None)
    trade = _trade(product)
    env = trade.env
    price0 = PhoenixQuadEngine(params=QuadParams(grid_points=301)).price(product, env)
    spec = build_phoenix_surface(trade)
    times = spec.times
    n_paths = 60000
    from quantark.sacva.exposure.paths import StatePathGenerator
    spots = StatePathGenerator(
        keys=["UND"], spots=[100.0], vols=[spec.vol],
        rates=[float(env.get_rate(float(times[-1])))],
        divs=[float(env.get_div_yield(float(times[-1])))],
        corr=[[1.0]], grid_times=times, num_paths=n_paths, seed=21).generate()["UND"]
    state = PhoenixStateMachine(
        coupon_barrier=spec.coupon_barrier, ko_barrier=spec.ko_barrier,
        direction=spec.direction, obs_idx=spec.obs_idx, num_obs=spec.num_obs,
        ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
        ki_monitoring_idx=spec.ki_monitoring_idx, times=times, seed=22,
        continuous=spec.ki_continuous, vol=spec.vol, use_memory=spec.use_memory).run(spots)
    v_ex = reprice_phoenix(spec.surface, spots, state, times, 1.0,
                           exposure_idx=list(range(len(times))))
    df = np.array([float(env.get_discount_factor(float(t))) for t in times])
    cf = _independent_cashflows(spots, spec, product, env, memory=True)
    # interior observation (not maturity): bank holds only coupons/redemptions strictly
    # before it, so V_ex (ex the obs coupon) + bank reconstructs the full price.
    j = spec.obs_idx[1]
    banked = (cf[:, :j + 1] * df[None, :j + 1]).sum(axis=1)
    total = df[j] * v_ex[:, j] + banked
    assert total.mean() == pytest.approx(price0, rel=6e-3)


def test_phoenix_expiry_coupon_deferred():
    trade = _trade(_phoenix(pay_type=CouponPayType.EXPIRY))
    with pytest.raises(Exception, match="INSTANT"):
        build_phoenix_surface(trade)


def test_phoenix_discrete_ki_deferred():
    # discrete KI -> deferred in v1
    barrier = BarrierConfig(
        ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[0.25, 0.5, 0.75, 1.0], ki_continuous=False)
    coupon = CouponBarrierConfig(coupon_barrier=90.0, coupon_rate=0.03,
                                 fixed_coupon_year_fraction=0.25)
    product = PhoenixOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier, coupon_config=coupon,
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)
    with pytest.raises(Exception, match="discrete KI"):
        build_phoenix_surface(_trade(product))
