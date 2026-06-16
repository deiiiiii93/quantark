import pytest

from quantark.asset.fx.engine.analytical.vannavolga.barrier_bs import (
    reiner_rubinstein_barrier,
)
from quantark.param.vol.vannavolga import GKInput, price_gk

S, RD, RF, VOL, TAU = 1.20, 0.02, 0.01, 0.10, 0.75


def _vanilla(strike, is_call):
    return price_gk(is_call, GKInput(S, strike, RD, RF, VOL, TAU))


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("is_up,barrier", [(True, 1.35), (False, 1.05)])
def test_in_out_parity(is_call, is_up, barrier):
    ko = reiner_rubinstein_barrier(
        S, 1.20, barrier, VOL, TAU, RD, RF,
        is_up=is_up, is_call=is_call, knock_in=False, rebate=0.0,
    )
    ki = reiner_rubinstein_barrier(
        S, 1.20, barrier, VOL, TAU, RD, RF,
        is_up=is_up, is_call=is_call, knock_in=True, rebate=0.0,
    )
    assert ko + ki == pytest.approx(_vanilla(1.20, is_call), rel=1e-9, abs=1e-9)


def test_ko_bounded_by_vanilla_and_nonneg():
    ko = reiner_rubinstein_barrier(
        S, 1.20, 1.35, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert 0.0 <= ko <= _vanilla(1.20, True) + 1e-12


def test_no_rebate_pricing_survives_negative_rates():
    # b ~ sigma^2/2 with rd < 0 makes mu ~ 0 and 2r/sigma^2 < 0, which would
    # crash an eager lam = sqrt(mu^2 + 2r/sigma^2). No-rebate pricing must work.
    rd, rf, vol = -0.001, -0.006, 0.10  # b = rd - rf = 0.005 = 0.5*vol^2
    val = reiner_rubinstein_barrier(
        S, 1.20, 1.35, vol, TAU, rd, rf,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert val == val and val >= 0.0  # finite, non-negative


def test_breached_states_positive_tau():
    # Up barrier already touched (spot >= H) with time remaining.
    ko = reiner_rubinstein_barrier(
        1.40, 1.20, 1.35, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert ko == pytest.approx(0.0)  # knocked out, no rebate -> worthless
    ki = reiner_rubinstein_barrier(
        1.40, 1.20, 1.35, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=True, rebate=0.0,
    )
    # Already knocked in -> plain vanilla.
    assert ki == pytest.approx(
        price_gk(True, GKInput(1.40, 1.20, RD, RF, VOL, TAU))
    )


def test_far_up_barrier_call_approaches_vanilla():
    # Up-and-out call with the barrier very far above spot -> ~ vanilla.
    ko = reiner_rubinstein_barrier(
        S, 1.20, 5.0, VOL, TAU, RD, RF,
        is_up=True, is_call=True, knock_in=False, rebate=0.0,
    )
    assert ko == pytest.approx(_vanilla(1.20, True), rel=1e-4)


def test_rr_matches_quantlib_down_out_call():
    pytest.importorskip("QuantLib")
    import QuantLib as q
    today = q.Date(15, 6, 2026)
    q.Settings.instance().evaluationDate = today
    dc = q.Actual365Fixed()
    spot_h = q.QuoteHandle(q.SimpleQuote(S))
    r_ts = q.YieldTermStructureHandle(q.FlatForward(today, RD, dc))
    q_ts = q.YieldTermStructureHandle(q.FlatForward(today, RF, dc))
    vol_ts = q.BlackVolTermStructureHandle(
        q.BlackConstantVol(today, q.NullCalendar(), VOL, dc)
    )
    process = q.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    exercise = q.EuropeanExercise(today + q.Period(int(round(TAU * 365)), q.Days))
    payoff = q.PlainVanillaPayoff(q.Option.Call, 1.20)
    opt = q.BarrierOption(q.Barrier.DownOut, 1.05, 0.0, payoff, exercise)
    opt.setPricingEngine(q.AnalyticBarrierEngine(process))
    ql_price = opt.NPV()
    ours = reiner_rubinstein_barrier(
        S, 1.20, 1.05, VOL, TAU, RD, RF,
        is_up=False, is_call=True, knock_in=False, rebate=0.0,
    )
    assert ours == pytest.approx(ql_price, rel=2e-3)
