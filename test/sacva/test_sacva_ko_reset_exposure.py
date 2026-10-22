"""SA-CVA stateful exposure for KO-reset snowballs (#1a).

A KO-reset snowball's KO barrier RESETS on knock-in: not-yet-KI paths follow the
pre-KI KO leg (baked into v_out), knocked-in paths follow the post-KI KO leg (baked
into v_in). The two-regime surface still suffices; the state machine applies the
pre barrier to not-KI paths and the post barrier to KI'd paths, and the receivable
picks the matching leg's payoff/settlement.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.sacva.exposure.simulator import (
    MonteCarloExposureConfig,
    MonteCarloExposureEngine,
)
from quantark.sacva.exposure.snowball_surface import build_snowball_surface
from quantark.sacva.exposure.statemachine import BarrierStateMachine
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.credit_curve import PillarHazardCurve
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVATrade
from quantark.util.enum import ObservationType, PostKOScheduleMode


def _env(spot=100.0, vol=0.20, rate=0.05, div=0.02):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="UND"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1))


def _counterparty(trades):
    return Counterparty(
        name="CP", netting_sets=[NettingSet("ns", list(trades))],
        credit_curve=PillarHazardCurve([0.5, 1.0, 3.0, 5.0, 10.0], [0.02] * 5,
                                       recovery_rate=0.4),
        bucket=2, credit_quality=CreditQuality.IG)


def _ko_reset(pre_ko=105.0, post_ko=98.0, ki=85.0,
              ko_dates=(0.25, 0.5, 0.75, 1.0), post_dates=(0.25, 0.5, 0.75, 1.0)):
    pre = BarrierConfig(
        ko_barrier=pre_ko, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=ki, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True)
    post = BarrierConfig(
        ko_barrier=post_ko, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(post_dates))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=pre, post_barrier_config=post,
        contract_multiplier=1.0, maturity=1.0, is_reverse=False,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE)


def _trade(product, quantity=1.0, grid_points=301):
    return CVATrade(
        trade_id=" kor", product=product,
        engine=KOResetSnowballQuadEngine(params=QuadParams(grid_points=grid_points)),
        env=_env(), quantity=quantity, trade_currency="USD")


def test_ko_reset_builds_post_leg():
    spec = build_snowball_surface(_trade(_ko_reset()))
    assert spec.post_ko_barrier == pytest.approx(98.0)
    assert spec.ko_barrier == pytest.approx(105.0)
    assert len(spec.post_ko_monitoring_idx) == len(spec.post_ko_payoffs)
    assert len(spec.post_ko_payoffs) > 0


def test_ko_reset_t0_equals_price():
    trade = _trade(_ko_reset(), quantity=5.0)
    price0 = KOResetSnowballQuadEngine(params=QuadParams(grid_points=301)).price(
        trade.product, trade.env)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=9)).compute(
            _counterparty([trade]))
    assert profile.times[0] == 0.0
    assert profile.epe_discounted[0] == pytest.approx(5.0 * price0, rel=3e-3)
    assert np.all(np.isfinite(profile.epe_discounted))
    assert profile.regulatory_eligible and len(profile.times) > 1


def test_ko_reset_post_barrier_certain_oracle():
    # ki_barrier 101 > spot 100 -> every path is knocked in at valuation (continuous KI
    # samples node 0), so all follow the POST leg. pre-KO 500 is unreachable, post-KO 1.0
    # is certain at the first post observation -> closed-form exposure using the POST
    # payoff, which only matches if the regime-conditional KO uses the post barrier+payoff.
    product = _ko_reset(pre_ko=500.0, post_ko=1.0, ki=101.0,
                        ko_dates=(0.5, 1.0), post_dates=(0.5,))
    trade = _trade(product, quantity=4.0)
    env = trade.env
    post_recs = trade.engine._resolve_ko_records(
        product, env, product.post_barrier_config)
    R = float(post_recs[0].payoff)                  # post-leg redemption at obs 0.5
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=4)).compute(
            _counterparty([trade]))
    times = profile.times
    df_obs = float(env.get_discount_factor(0.5))
    target = 4.0 * R * df_obs
    obs_idx = int(np.argmin(np.abs(times - 0.5)))
    assert profile.epe_discounted[0] == pytest.approx(target, rel=4e-3)
    for j in range(obs_idx, len(times)):
        assert profile.epe_discounted[j] == pytest.approx(0.0, abs=target * 4e-3)


def test_ko_reset_value_process_martingale():
    # discounted value process incl. realized redemptions (pre- or post-leg payoff by the
    # path's KO regime) is a Q-martingale -> the recorded reset surfaces + regime KO are
    # mutually consistent.
    product = _ko_reset()
    trade = _trade(product, quantity=1.0)
    env = trade.env
    price0 = KOResetSnowballQuadEngine(params=QuadParams(grid_points=301)).price(
        product, env)
    spec = build_snowball_surface(trade)
    times = spec.times
    n_paths = 40000

    from quantark.sacva.exposure.paths import StatePathGenerator
    spots = StatePathGenerator(
        keys=["UND"], spots=[float(env.spot)], vols=[spec.vol],
        rates=[float(env.get_rate(float(times[-1])))],
        divs=[float(env.get_div_yield(float(times[-1])))],
        corr=[[1.0]], grid_times=times, num_paths=n_paths, seed=2).generate()["UND"]
    state = BarrierStateMachine(
        ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
        ko_barrier=spec.ko_barrier, ko_direction=spec.ko_direction,
        ki_monitoring_idx=spec.ki_monitoring_idx,
        ko_monitoring_idx=spec.ko_monitoring_idx,
        post_ko_barrier=spec.post_ko_barrier,
        post_ko_monitoring_idx=spec.post_ko_monitoring_idx,
        times=times, seed=2, continuous=spec.ki_continuous, vol=spec.vol).run(spots)

    df = np.array([float(env.get_discount_factor(float(t))) for t in times])
    ko_idx = state["ko_idx"]
    ko_post = state["ko_post"]
    # redemption discounted to t0, choosing the leg by the path's KO regime
    pre_red = np.zeros(len(times))
    for obs, pay, settle in zip(
            spec.ko_monitoring_idx, spec.ko_payoffs, spec.ko_settle_idx):
        pre_red[obs] = pay * df[settle]
    post_red = np.zeros(len(times))
    for obs, pay, settle in zip(
            spec.post_ko_monitoring_idx, spec.post_ko_payoffs, spec.post_ko_settle_idx):
        post_red[obs] = pay * df[settle]
    realized = np.where(ko_idx >= 0,
                        np.where(ko_post, post_red[ko_idx], pre_red[ko_idx]), 0.0)

    for j in (len(times) - 1, len(times) // 2):
        alive = state["alive"][:, j]
        ki = state["knocked_in"][:, j]
        v = np.zeros(n_paths)
        for label, sel in (("alive", alive & ~ki), ("knocked_in", alive & ki)):
            if sel.any():
                v[sel] = spec.surface.value_at(spots[sel, j], float(times[j]), label)
        dead_by_j = (ko_idx >= 0) & (ko_idx <= j)
        total = df[j] * v + np.where(dead_by_j, realized, 0.0)
        assert total.mean() == pytest.approx(price0, rel=0.03), f"leak at node {j}"


def test_ko_reset_regime_split_is_used():
    # a reset where the pre barrier is unreachable but the post barrier bites confirms the
    # state machine actually routes KI'd paths through the post leg (some ko_post True).
    product = _ko_reset(pre_ko=500.0, post_ko=100.0, ki=90.0,
                        ko_dates=(0.5, 1.0), post_dates=(0.5, 1.0))
    trade = _trade(product)
    spec = build_snowball_surface(trade)
    times = spec.times
    from quantark.sacva.exposure.paths import StatePathGenerator
    env = trade.env
    spots = StatePathGenerator(
        keys=["UND"], spots=[100.0], vols=[spec.vol],
        rates=[float(env.get_rate(float(times[-1])))],
        divs=[float(env.get_div_yield(float(times[-1])))],
        corr=[[1.0]], grid_times=times, num_paths=20000, seed=1).generate()["UND"]
    state = BarrierStateMachine(
        ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
        ko_barrier=spec.ko_barrier, ko_direction=spec.ko_direction,
        ki_monitoring_idx=spec.ki_monitoring_idx,
        ko_monitoring_idx=spec.ko_monitoring_idx,
        post_ko_barrier=spec.post_ko_barrier,
        post_ko_monitoring_idx=spec.post_ko_monitoring_idx,
        times=times, seed=1, continuous=spec.ki_continuous, vol=spec.vol).run(spots)
    # pre barrier 500 is never hit, so every knock-out must be a post-leg (reset) KO
    ko = state["ko_idx"] >= 0
    assert ko.any()
    assert np.all(state["ko_post"][ko])
