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


def _vanilla(env, div_env=None, rate=None, quantity=1.0, asset_div=None):
    from quantark.asset.equity.product.option.european_vanilla_option import (
        EuropeanVanillaOption,
    )
    from quantark.asset.equity.engine.analytical.black_scholes_engine import (
        BlackScholesEngine,
    )
    from quantark.util.enum import OptionType
    return CVATrade("van", EuropeanVanillaOption(strike=100.0,
                    option_type=OptionType.PUT, maturity=1.0),
                    BlackScholesEngine(), env, quantity=quantity, trade_currency="USD")


def test_inconsistent_same_underlying_market_data_raises():
    # a co-netted vanilla on the SAME underlying with a different dividend is ambiguous
    # (one GBM factor per underlying) -> raise rather than produce an order-dependent CVA
    snow = _snowball_trade(quantity=1.0)
    van = _vanilla(_env(div=0.10))               # same underlying "UND", div 10% vs 2%
    with pytest.raises(Exception):
        MonteCarloExposureEngine(
            MonteCarloExposureConfig(num_paths=4000, seed=4)).compute(
            _counterparty([van, snow]))


def test_inconsistent_discount_curve_raises():
    # two different reporting discount curves in one counterparty -> raise
    snow = _snowball_trade(quantity=1.0)
    van = _vanilla(_env(rate=0.07))              # rate 7% vs snowball's 5%
    with pytest.raises(Exception):
        MonteCarloExposureEngine(
            MonteCarloExposureConfig(num_paths=4000, seed=4)).compute(
            _counterparty([van, snow]))


def test_corr_matrix_extracts_principal_submatrix():
    # one configured (superset) correlation serves both the full run and a
    # terminated-snowball fallback that prices only a subset of underlyings.
    from quantark.sacva.exposure.correlation import CorrelationModel
    cm = CorrelationModel(keys=["S", "A", "B"],
                          matrix=[[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    eng = MonteCarloExposureEngine(MonteCarloExposureConfig(correlation=cm))
    assert np.allclose(eng._corr_matrix(["S", "A", "B"]),
                       [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    assert np.allclose(eng._corr_matrix(["A", "B"]), [[1.0, 0.4], [0.4, 1.0]])
    assert np.allclose(eng._corr_matrix(["B", "A"]), [[1.0, 0.4], [0.4, 1.0]])  # reorder
    with pytest.raises(Exception):
        eng._corr_matrix(["A", "Z"])                # missing underlying


def test_co_netted_vol_mismatch_on_snowball_underlying_raises():
    # a co-netted trade on the snowball underlying must ride the snowball path vol
    snow = _snowball_trade(quantity=1.0)         # UND, flat vol 0.20 = spec.vol
    van = _vanilla(_env(vol=0.35))               # UND, flat vol 0.35
    with pytest.raises(Exception):
        MonteCarloExposureEngine(
            MonteCarloExposureConfig(num_paths=4000, seed=4)).compute(
            _counterparty([snow, van]))


def _t0_ko_snowball(ko_barrier):
    cfg = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.0, 0.5, 1.0], ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=cfg,
                          contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def test_t0_ko_below_barrier_builds_normally():
    # a valuation-date KO observation with spot BELOW the barrier is a live, valid
    # snowball -> it must build a surface and produce exposure (not be rejected).
    trade = CVATrade("s", _t0_ko_snowball(ko_barrier=103.0),
                     SnowballQuadEngine(params=QuadParams(grid_points=301)), _env(),
                     trade_currency="USD")
    spec = build_snowball_surface(trade)        # spot 100 < 103: does not raise
    assert 0.0 in [round(float(t), 9) for t in spec.times]
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=8000, seed=7)).compute(_counterparty([trade]))
    assert np.all(np.isfinite(profile.epe_discounted))


def test_immediate_ko_terminates_to_zero_exposure():
    # valuation spot 100 breaches a t0 KO barrier of 95 -> immediate KO -> the snowball
    # is terminated at valuation: build signals termination, exposure is zero.
    from quantark.sacva.exposure.snowball_surface import SnowballTerminatedAtValuation
    trade = CVATrade("s", _t0_ko_snowball(ko_barrier=95.0),
                     SnowballQuadEngine(params=QuadParams(grid_points=301)), _env(),
                     trade_currency="USD")
    with pytest.raises(SnowballTerminatedAtValuation):
        build_snowball_surface(trade)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=4000, seed=7)).compute(_counterparty([trade]))
    assert np.allclose(profile.epe_discounted, 0.0)


# --- #2 delayed KO settlement -----------------------------------------------

from datetime import timedelta

from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.enum import CouponPayType


def _certain_ko_snowball(pay_type, ko_obs=0.5, maturity=1.0):
    # KO barrier far below spot -> certain knock-out at the FIRST observation; no KI.
    # The exposure profile is then a closed form (deterministic redemption), an exact
    # oracle for the settlement timing. INSTANT settles at the obs; EXPIRY at maturity.
    cfg = BarrierConfig(
        ko_barrier=1.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[ko_obs])
    return SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=cfg,
        accrual_config=AccrualConfig(coupon_pay_type=pay_type),
        contract_multiplier=1.0, maturity=maturity, is_reverse=False)


def _certain_ko_trade(pay_type, quantity=1.0):
    return CVATrade(
        trade_id="cko", product=_certain_ko_snowball(pay_type),
        engine=SnowballQuadEngine(params=QuadParams(grid_points=301)),
        env=_env(), quantity=quantity, trade_currency="USD")


def test_immediate_ko_certain_exposure_oracle():
    # certain KO at obs=0.5, settled immediately: discounted EE = q*R*DF(0,0.5) at t0,
    # zero once the redemption is paid (no forward exposure).
    trade = _certain_ko_trade(CouponPayType.INSTANT, quantity=3.0)
    env = trade.env
    R = float(trade.product.resolve_ko_observations(env)[0].payoff)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=11)).compute(
            _counterparty([trade]))
    times = profile.times
    df_obs = float(env.get_discount_factor(0.5))
    target = 3.0 * R * df_obs
    obs_idx = int(np.argmin(np.abs(times - 0.5)))
    assert profile.epe_discounted[0] == pytest.approx(target, rel=3e-3)
    # at and after the KO observation the cash is settled -> zero exposure
    for j in range(obs_idx, len(times)):
        assert profile.epe_discounted[j] == pytest.approx(0.0, abs=target * 3e-3)


def test_delayed_ko_settlement_expiry_exposure_oracle():
    # certain KO at obs=0.5, redemption settled at maturity (EXPIRY): the counterparty
    # owes R@T from t0 to T, so discounted EE is the CONSTANT q*R*DF(0,T) at every node
    # before maturity (continuation value pre-KO, pending receivable post-KO), 0 at T.
    trade = _certain_ko_trade(CouponPayType.EXPIRY, quantity=3.0)
    env = trade.env
    R = float(trade.product.resolve_ko_observations(env)[0].payoff)
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=11)).compute(
            _counterparty([trade]))
    times = profile.times
    target = 3.0 * R * float(env.get_discount_factor(1.0))
    for j, t in enumerate(times):
        if t < 1.0 - 1e-9:
            assert profile.epe_discounted[j] == pytest.approx(target, rel=3e-3), \
                f"node {t}"
        else:
            assert profile.epe_discounted[j] == pytest.approx(0.0, abs=target * 3e-3)


def _date_settled_ko_snowball(obs_days=182, settle_days=189, maturity=1.0):
    """Certain-KO snowball whose single KO observation settles ``settle_days`` after
    valuation (strictly between the observation and maturity) -> the QUAD engine must
    record a continuation surface at the settlement node."""
    val = datetime(2024, 1, 1)
    rec = ObservationRecord(
        observation_date=val + timedelta(days=obs_days),
        settlement_date=val + timedelta(days=settle_days),
        barrier=1.0, return_rate=0.15, is_rate_annualized=False)
    schedule = ObservationSchedule(records=[rec])
    cfg = BarrierConfig(
        ko_barrier=1.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=schedule)
    return SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=cfg,
        contract_multiplier=1.0, maturity=maturity, is_reverse=False)


def test_delayed_ko_settlement_records_settlement_node():
    # An intermediate settlement (obs ~0.498y, settle ~0.518y) is not an observation
    # node, so the engine must insert it as a recording node; the exposure profile is
    # then the closed-form q*R*DF(0,settle) up to settlement and 0 after.
    product = _date_settled_ko_snowball(obs_days=182, settle_days=189)
    trade = CVATrade(
        trade_id="dko", product=product,
        engine=SnowballQuadEngine(params=QuadParams(grid_points=301)),
        env=_env(), quantity=2.0, trade_currency="USD")
    env = trade.env
    rec = product.resolve_ko_observations(env)[0]
    obs_t, settle_t, R = (float(rec.observation_time),
                          float(rec.settlement_time), float(rec.payoff))
    assert settle_t > obs_t  # genuinely delayed
    spec = build_snowball_surface(trade)
    # the settlement time is a recorded grid node (engine diffused through it)
    assert any(abs(float(t) - settle_t) <= 1e-9 for t in spec.times), \
        "settlement node not recorded by the engine"
    profile = MonteCarloExposureEngine(
        MonteCarloExposureConfig(num_paths=20000, seed=5)).compute(
            _counterparty([trade]))
    times = profile.times
    target = 2.0 * R * float(env.get_discount_factor(settle_t))
    for j, t in enumerate(times):
        if t < settle_t - 1e-9:
            assert profile.epe_discounted[j] == pytest.approx(target, rel=4e-3), \
                f"node {t}"
        else:
            assert profile.epe_discounted[j] == pytest.approx(0.0, abs=target * 4e-3)


def test_value_process_martingale_delayed_settlement():
    # EXPIRY: every KO redemption settles at maturity. The discounted value process
    # including the realized redemption (discounted to t0 via its SETTLEMENT node) must
    # stay a Q-martingale on a realistic probabilistic-KO+KI snowball -> the recorded
    # delayed-settlement surfaces are consistent (no value leaked at the KO boundary).
    cfg = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True)
    product = SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=cfg,
        accrual_config=AccrualConfig(coupon_pay_type=CouponPayType.EXPIRY),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)
    trade = CVATrade("snowd", product,
                     SnowballQuadEngine(params=QuadParams(grid_points=301)),
                     _env(), 1.0, "USD")
    env = trade.env
    price0 = SnowballQuadEngine(params=QuadParams(grid_points=301)).price(product, env)
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
    # redemption per KO observation, discounted to t0 via its settlement node
    redemption_t0 = np.zeros(len(times))
    for obs, pay, settle in zip(
            spec.ko_monitoring_idx, spec.ko_payoffs, spec.ko_settle_idx):
        redemption_t0[obs] = pay * df[settle]
    realized = np.where(ko_idx >= 0, redemption_t0[ko_idx], 0.0)

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


def test_post_maturity_ko_settlement_deferred_raises():
    # a KO settling AFTER maturity has no continuation surface past the terminal node
    # -> the builder rejects it (deferred), rather than approximating.
    product = _date_settled_ko_snowball(obs_days=360, settle_days=400, maturity=1.0)
    trade = CVATrade(
        trade_id="pmk", product=product,
        engine=SnowballQuadEngine(params=QuadParams(grid_points=301)),
        env=_env(), quantity=1.0, trade_currency="USD")
    with pytest.raises(Exception, match="post-maturity"):
        build_snowball_surface(trade)
