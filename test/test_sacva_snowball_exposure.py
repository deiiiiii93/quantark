"""SA-CVA stateful (snowball) exposure: backward-grid recording + MC wiring.

The snowball QUAD engine runs a two-regime backward recursion (v_in: knocked-in,
v_out: not-yet) on a single inception-anchored spot grid. With ``record_backward_grids``
on it exposes those per-observation continuation surfaces, which the CVA exposure
layer reads into a per-(t, state) GridValueSurface — no re-pricing.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def _env(spot=100.0, vol=0.20, rate=0.05, div=0.02):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="UND"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball(ko_dates=(0.25, 0.5, 0.75, 1.0), ki_continuous=True):
    cfg = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=75.0,
        ki_observation_type=(ObservationType.CONTINUOUS if ki_continuous
                             else ObservationType.DISCRETE),
        ki_continuous=ki_continuous)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg,
                          contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def test_recording_off_by_default_and_no_grids():
    eng = SnowballQuadEngine(params=QuadParams(grid_points=301))
    assert eng.record_backward_grids is False
    eng.price(_snowball(), _env())
    assert eng._backward_grids == {}


def test_recorded_grids_cover_observation_times_and_t0():
    eng = SnowballQuadEngine(params=QuadParams(grid_points=301))
    eng.record_backward_grids = True
    eng.price(_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0)), _env())
    keys = sorted(eng._backward_grids)
    # every KO observation time + maturity + the t0 valuation slice
    assert 0.0 in keys
    for t in (0.25, 0.5, 0.75, 1.0):
        assert any(abs(k - t) <= 1e-9 for k in keys), f"missing surface at {t}"
    # each slice is (spot_grid, v_in, v_out) of equal length
    for spot_grid, v_in, v_out in eng._backward_grids.values():
        assert spot_grid.shape == v_in.shape == v_out.shape
        assert np.all(np.isfinite(spot_grid))
        assert np.all(np.isfinite(v_in)) and np.all(np.isfinite(v_out))


def test_t0_surface_reprices_to_engine_price():
    # the recorded t0 v_out, interpolated at the inception spot, must equal price()
    eng = SnowballQuadEngine(params=QuadParams(grid_points=301))
    eng.record_backward_grids = True
    snowball, env = _snowball(), _env()
    price = eng.price(snowball, env)
    spot_grid, v_in, v_out = eng._backward_grids[0.0]
    v0 = float(np.interp(env.spot, spot_grid, v_out))  # not knocked-in at valuation
    # linear interp on the recorded spot grid vs the engine's native log-grid
    # interpolate differ only at interpolation order (301 grid points -> ~1e-4 rel)
    assert v0 == pytest.approx(price, rel=1e-3)


# --- MC exposure wiring -----------------------------------------------------

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
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


def _snowball_trade(quantity=1.0, grid_points=301):
    return CVATrade(
        trade_id="snow", product=_snowball(),
        engine=SnowballQuadEngine(params=QuadParams(grid_points=grid_points)),
        env=_env(), quantity=quantity, trade_currency="USD")


def _counterparty(trades):
    return Counterparty(
        name="CP", netting_sets=[NettingSet("ns", list(trades))],
        credit_curve=PillarHazardCurve([0.5, 1.0, 3.0, 5.0, 10.0], [0.02] * 5,
                                       recovery_rate=0.4),
        bucket=2, credit_quality=CreditQuality.IG)


def test_stateful_exposure_t0_equals_price():
    trade = _snowball_trade(quantity=7.0)
    price0 = SnowballQuadEngine(params=QuadParams(grid_points=301)).price(
        trade.product, trade.env)
    eng = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=7))
    profile = eng.compute(_counterparty([trade]))
    # at t0 the spot is deterministic, so discounted EPE(0) = quantity * price0
    assert profile.times[0] == 0.0
    assert profile.epe_discounted[0] == pytest.approx(7.0 * price0, rel=2e-3)
    assert np.all(np.isfinite(profile.epe_discounted))
    assert profile.regulatory_eligible and len(profile.times) > 1


def test_value_process_martingale_with_ko_redemptions():
    # The discounted value process INCLUDING realised KO redemptions is a Q-martingale:
    #   E[ df_j * V_alive(t_j) + (redemption*df at KO for paths KO'd by t_j) ] == price0
    # for every node. This proves the continuation surface and the discrete KO
    # termination are mutually consistent (no value leaked at the KO boundary).
    trade = _snowball_trade(quantity=1.0)
    env = trade.env
    price0 = SnowballQuadEngine(params=QuadParams(grid_points=301)).price(
        trade.product, env)
    spec = build_snowball_surface(trade)
    times = spec.times
    n_paths = 40000

    from quantark.sacva.exposure.paths import StatePathGenerator
    spots = StatePathGenerator(
        keys=["UND"], spots=[float(env.spot)], vols=[spec.vol],
        rates=[float(env.get_rate(float(times[-1])))],
        divs=[float(env.get_div_yield(float(times[-1])))],
        corr=[[1.0]], grid_times=times, num_paths=n_paths, seed=3).generate()["UND"]
    state = BarrierStateMachine(
        ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
        ko_barrier=spec.ko_barrier, ko_direction=spec.ko_direction,
        ki_monitoring_idx=spec.ki_monitoring_idx,
        ko_monitoring_idx=spec.ko_monitoring_idx, times=times, seed=3,
        continuous=spec.ki_continuous, vol=spec.vol).run(spots)

    df = np.array([float(env.get_discount_factor(float(t))) for t in times])
    ko_idx = state["ko_idx"]
    payoff_arr = np.zeros(len(times))
    for rec in trade.product.resolve_ko_observations(env):
        j = int(np.argmin(np.abs(times - float(rec.observation_time))))
        payoff_arr[j] = float(rec.payoff)
    redemption = np.where(ko_idx >= 0, payoff_arr[ko_idx] * df[ko_idx], 0.0)

    for j in (len(times) - 1, len(times) // 2):
        alive = state["alive"][:, j]
        ki = state["knocked_in"][:, j]
        v = np.zeros(n_paths)
        for label, sel in (("alive", alive & ~ki), ("knocked_in", alive & ki)):
            if sel.any():
                v[sel] = spec.surface.value_at(spots[sel, j], float(times[j]), label)
        dead_by_j = (ko_idx >= 0) & (ko_idx <= j)
        total = df[j] * v + np.where(dead_by_j, redemption, 0.0)
        assert total.mean() == pytest.approx(price0, rel=0.03), f"leak at node {j}"


def test_v1_scope_guards():
    # multiple trades on one counterparty -> raise (no shared grid yet)
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=2000))
    with pytest.raises(Exception):
        eng.compute(_counterparty([_snowball_trade(), _snowball_trade()]))
    # Phoenix engine (richer state) -> builder raises
    phoenix_trade = CVATrade(
        trade_id="ph", product=_snowball(),
        engine=PhoenixQuadEngine(params=QuadParams(grid_points=301)),
        env=_env(), quantity=1.0, trade_currency="USD")
    with pytest.raises(Exception):
        build_snowball_surface(phoenix_trade)


def test_snowball_end_to_end_capital():
    # full façade: snowball -> MC exposure -> regulatory CVA -> credit-spread delta -> capital
    from quantark.sacva.exposure.simulator import (
        MonteCarloExposureConfig as _Cfg,
        MonteCarloExposureEngine as _Eng,
    )
    from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
    from quantark.sacva.sacva_engine import SACVAEngine

    trade = _snowball_trade(quantity=100.0)
    portfolio = CVATradePortfolio(counterparties=[_counterparty([trade])], hedges=[])
    result = SACVAEngine(exposure_engine=_Eng(_Cfg(num_paths=12000, seed=9))).compute(
        portfolio)
    assert result.total_capital > 0.0
    assert result.delta_capital > 0.0
    assert result.counterparty_cva["CP"] > 0.0
    assert result.exposure_profiles["CP"].epe_discounted[0] > 0.0


def test_snowball_nets_with_vanilla_on_shared_grid():
    # a snowball nets with a vanilla on the same underlying, both on the snowball
    # observation grid; at t0 the netted exposure is the positive part of the sum.
    from quantark.asset.equity.product.option.european_vanilla_option import (
        EuropeanVanillaOption,
    )
    from quantark.asset.equity.engine.analytical.black_scholes_engine import (
        BlackScholesEngine,
    )
    from quantark.util.enum import OptionType

    snow = _snowball_trade(quantity=1.0)
    van_opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT,
                                    maturity=1.0)
    van = CVATrade("van", van_opt, BlackScholesEngine(), _env(), quantity=2.0,
                   trade_currency="USD")
    prof = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=16000, seed=4)).compute(
        _counterparty([snow, van]))

    p_snow = SnowballQuadEngine(params=QuadParams(grid_points=301)).price(
        snow.product, snow.env)
    p_van = BlackScholesEngine().price(van.product, van.env)
    expected0 = max(1.0 * p_snow + 2.0 * p_van, 0.0)
    assert prof.epe_discounted[0] == pytest.approx(expected0, rel=3e-3)


def test_two_snowballs_raise():
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(num_paths=2000))
    with pytest.raises(Exception):
        eng.compute(_counterparty([_snowball_trade(), _snowball_trade()]))


# --- review fixes: seasoned KI, deferred features, shared-underlying drift -----

def _discrete_ki_snowball(disable_ko_after_ki=False):
    cfg = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[0.5, 1.0], ki_continuous=False,
        disable_ko_after_ki=disable_ko_after_ki)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg,
                          contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def test_seasoned_knocked_in_carried_to_state():
    trade = _snowball_trade()
    trade.product._otc_lifecycle_knocked_in = True   # knocked in before valuation
    spec = build_snowball_surface(trade)
    assert spec.initial_knocked_in is True
    # the state machine seeds the KI history so t0 is knocked-in (selects v_in)
    sm = BarrierStateMachine(
        ki_barrier=spec.ki_barrier, ki_direction=spec.ki_direction,
        ko_barrier=spec.ko_barrier, ko_direction=spec.ko_direction,
        ki_monitoring_idx=spec.ki_monitoring_idx,
        ko_monitoring_idx=spec.ko_monitoring_idx, times=spec.times, seed=1,
        continuous=spec.ki_continuous, vol=spec.vol, initial_knocked_in=True)
    st = sm.run(np.full((8, len(spec.times)), 100.0))
    assert st["knocked_in"][:, 0].all()


def test_disable_ko_after_ki_deferred():
    trade = CVATrade("s", _discrete_ki_snowball(disable_ko_after_ki=True),
                     SnowballQuadEngine(params=QuadParams(grid_points=301)), _env(),
                     trade_currency="USD")
    with pytest.raises(Exception):
        build_snowball_surface(trade)


def test_bgk_ki_mode_deferred():
    from quantark.util.enum.engine_enums import KnockInMonitoringMode
    eng = SnowballQuadEngine(params=QuadParams(
        grid_points=301, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION))
    trade = CVATrade("s", _discrete_ki_snowball(), eng, _env(), trade_currency="USD")
    with pytest.raises(Exception):
        build_snowball_surface(trade)


def test_snowball_drift_sourced_from_its_own_env_not_co_netted_vanilla():
    # a co-netted vanilla on the SAME underlying with a very different dividend and
    # negligible quantity must NOT change the snowball's drift (the snowball sources
    # rate/div from its own env). With the fix the profile matches snowball-alone.
    from quantark.asset.equity.product.option.european_vanilla_option import (
        EuropeanVanillaOption,
    )
    from quantark.asset.equity.engine.analytical.black_scholes_engine import (
        BlackScholesEngine,
    )
    from quantark.util.enum import OptionType

    cfg = MonteCarloExposureConfig(num_paths=12000, seed=4)
    alone = MonteCarloExposureEngine(cfg).compute(
        _counterparty([_snowball_trade(quantity=1.0)]))

    snow = _snowball_trade(quantity=1.0)
    bad_env = _env(div=0.10)                      # same underlying "UND", div 10% vs 2%
    van = CVATrade("van", EuropeanVanillaOption(strike=100.0,
                   option_type=OptionType.PUT, maturity=1.0),
                   BlackScholesEngine(), bad_env, quantity=1e-9, trade_currency="USD")
    # vanilla FIRST in the set: with the old bug the snowball would inherit its div
    netted = MonteCarloExposureEngine(cfg).compute(_counterparty([van, snow]))

    assert np.allclose(netted.epe_discounted, alone.epe_discounted, rtol=1e-3, atol=1e-6)
