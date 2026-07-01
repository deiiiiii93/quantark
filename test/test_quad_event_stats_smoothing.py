"""QUAD event-stats barrier-smoothing accuracy tests (spec 2026-07-01)."""
from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment


def _env(vol=0.22, rate=0.03, div=0.0, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _ref_phoenix():
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
    )


def _mc_engine():
    return PhoenixMCEngine(
        params=MCParams(num_paths=300_000, time_steps=252, use_qmc=True, seed=7)
    )


def test_first_ko_probability_matches_analytic_digital():
    env = _env()
    ph = _ref_phoenix()
    # First KO observation is at t=1/12; P(S_t >= 103) = N(d2).
    t, S, K, r, q, sig = 1.0 / 12.0, 100.0, 103.0, 0.03, 0.0, 0.22
    d2 = (np.log(S / K) + (r - q - 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
    analytic = float(norm.cdf(d2))

    stats = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    ko0 = float(np.asarray(stats.ko_probability)[0])
    assert abs(ko0 - analytic) / analytic < 0.01, (ko0, analytic)  # measured +0.20%


def test_coupon_probability_matches_mc():
    env = _env()
    ph = _ref_phoenix()
    q = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    m = _mc_engine().calculate_event_stats(ph, env)
    qc = np.asarray(q.coupon_probability, dtype=float)
    mc = np.asarray(m.coupon_probability, dtype=float)
    # Measured: aggregate +0.59%, max per-obs +2.68%. Bounds carry MC-seed margin.
    assert abs(qc.sum() - mc.sum()) / mc.sum() < 0.015, (qc.sum(), mc.sum())
    rel = np.abs(qc - mc) / np.maximum(mc, 1e-6)
    assert np.max(rel) < 0.035, rel


def test_ko_probability_matches_mc():
    """Full per-observation KO probability vs MC — protects the KO stream that
    downstream SA-CVA / cash-leg consumers rely on, not just KO[0]."""
    env = _env()
    ph = _ref_phoenix()
    q = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    m = _mc_engine().calculate_event_stats(ph, env)
    qk = np.asarray(q.ko_probability, dtype=float)
    mk = np.asarray(m.ko_probability, dtype=float)
    # Measured: max per-obs abs diff 0.0025 (KO[0]); bound carries MC-seed margin.
    assert np.max(np.abs(qk - mk)) < 0.008, (qk, mk)


def test_survival_matches_mc():
    env = _env()
    ph = _ref_phoenix()
    q = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    m = _mc_engine().calculate_event_stats(ph, env)
    qsv = np.asarray(q.survival_probability, dtype=float)
    msv = np.asarray(m.survival_probability, dtype=float)
    assert np.max(np.abs(qsv - msv)) < 0.01, (qsv, msv)  # measured 0.0025


def test_maturity_cashflow_matches_mc():
    """Validate the residual maturity field INDEPENDENTLY against MC, so an
    over-smoothing error cannot hide in the residual. The total identity
    (Sum(ko)+Sum(coupon)+maturity == PV) is only a wiring invariant (maturity is
    the residual plug), so it is NOT used as the correctness signal here."""
    env = _env()
    ph = _ref_phoenix()
    q = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    m = _mc_engine().calculate_event_stats(ph, env)
    # Wiring invariant: streams sum to PV by construction.
    total = (
        float(np.sum(q.expected_discounted_ko_cashflow))
        + float(np.sum(q.expected_discounted_coupon_cashflow))
        + float(q.expected_discounted_maturity_cashflow)
    )
    assert abs(total - float(q.pv)) < 1e-6 * max(1.0, abs(float(q.pv)))
    # Independent correctness check on the residual stream itself.
    assert abs(
        q.expected_discounted_maturity_cashflow
        - m.expected_discounted_maturity_cashflow
    ) < 0.08 * max(1.0, abs(m.expected_discounted_maturity_cashflow))


def _overlap_phoenix(ko_barrier):
    # Coupon barrier (100) below KO barrier: the region S>=ko_barrier is in BOTH
    # the coupon-pay region and the KO region, so a path can KO and pay a coupon
    # on the same observation.
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=ko_barrier, ki_barrier=75.0, coupon_barrier=100.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
    )


@pytest.mark.parametrize("cells", [1, 0])
def test_simultaneous_ko_coupon_retained(cells):
    """The coupon on a simultaneous-KO observation is retained (spec §2): the
    coupon indicator is the FULL pay_w after KO scaling, not ko_w*pay_w.

    Attribution fingerprint (holds at ANY smoothing width): coupon_probability[0]
    is INVARIANT to the KO barrier, because at the first observation every path
    above the coupon barrier is paid whether or not it also KOs. A wrongly-scaled
    update (ko_w*pay_w / (1-ko_w)*pay_w) would drop ~P(S0>=ko_barrier) (~0.22 for
    ko=105) — an order of magnitude larger than hard-mask grid noise (~0.03), so a
    loose bound cleanly separates the two. Checked at default AND cells=0.
    """
    env = _env()

    def c0(ko_barrier):
        s = PhoenixQuadEngine(
            params=QuadParams(grid_points=401, event_smoothing_cells=cells)
        ).calculate_event_stats(_overlap_phoenix(ko_barrier), env)
        return float(np.asarray(s.coupon_probability)[0])

    assert abs(c0(105.0) - c0(130.0)) < 0.05, (c0(105.0), c0(130.0))


def test_simultaneous_ko_coupon_matches_digital_smoothed():
    """With smoothing (the accurate path), coupon_probability[0] for a KO-overlap
    config equals the analytic digital P(S_{t0} >= coupon_barrier) — confirming
    correct attribution AND barrier-at-spot accuracy in one shot. (The hard-mask
    path cells=0 does NOT match here — that ATM-barrier discretization error is
    exactly what this fix removes — so this accuracy claim is smoothed-only.)"""
    env = _env()
    t, S, Kc, r, q, sig = 1.0 / 12.0, 100.0, 100.0, 0.03, 0.0, 0.22
    d2 = (np.log(S / Kc) + (r - q - 0.5 * sig * sig) * t) / (sig * np.sqrt(t))
    digital = float(norm.cdf(d2))  # ~0.503
    s = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(
        _overlap_phoenix(105.0), env
    )
    c0 = float(np.asarray(s.coupon_probability)[0])
    assert abs(c0 - digital) < 0.006, (c0, digital)


def test_cells_zero_reproduces_hard_mask_baseline():
    """event_smoothing_cells=0 must reproduce the pre-change hard-mask output
    bit-for-bit — proving the smoothing is purely additive."""
    env = _env()
    ph = _ref_phoenix()
    stats = PhoenixQuadEngine(
        params=QuadParams(grid_points=401, event_smoothing_cells=0)
    ).calculate_event_stats(ph, env)
    ko_baseline = np.array([
        0.354468887686, 0.140439831531, 0.077573430563, 0.050661052323,
        0.036378003003, 0.027774839276, 0.022137332992, 0.018206780906,
        0.015330151732, 0.013141766061, 0.011424788127, 0.010044303939,
    ])
    coupon_baseline = np.array([
        0.993817910015, 0.607795810328, 0.435407594056, 0.33466259599,
        0.268523202241, 0.222035236917, 0.187722221082, 0.16142555321,
        0.140670083009, 0.123903915985, 0.110109488865, 0.098592290015,
    ])
    np.testing.assert_allclose(
        np.asarray(stats.ko_probability), ko_baseline, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(stats.coupon_probability), coupon_baseline, rtol=0, atol=1e-9
    )


from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.util.enum import ObservationType


def _discrete_ki_phoenix(ko_barrier=101.0, ki_barrier=100.0, disable_ko_after_ki=True):
    # KO and KI barriers adjacent and near spot, discrete monthly KI on the same
    # dates as KO: exercises the smoothed KI transition and its interaction with
    # the smoothed KO partition (the disable_ko_after_ki=True case where v_in is
    # not KO-scaled but v_out is).
    dates = [i / 12 for i in range(1, 13)]
    obs = ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=ki_barrier) for t in dates]
    )
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=ko_barrier, ki_barrier=ki_barrier, coupon_barrier=85.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=obs, disable_ko_after_ki=disable_ko_after_ki,
    )


@pytest.mark.parametrize("disable_ko_after_ki", [True, False])
def test_discrete_ki_ko_stream_matches_mc(disable_ko_after_ki):
    """Smoothed discrete-KI transition must not regress the KO / maturity streams
    vs MC — especially disable_ko_after_ki=True, where v_in is NOT KO-scaled while
    v_out IS, so a hard KI copy would mix soft-KO with an un-scaled surface. The
    smoothed blend v_out=(1-ki_w_eff)*v_out+ki_w_eff*v_in keeps the default path at
    least as close to MC as the hard-mask (cells=0) path."""
    env = _env()
    ph = _discrete_ki_phoenix(disable_ko_after_ki=disable_ko_after_ki)
    q = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    m = _mc_engine().calculate_event_stats(ph, env)
    ko_q = float(np.sum(q.ko_probability))
    ko_m = float(np.sum(m.ko_probability))
    assert abs(ko_q - ko_m) < 0.012, (ko_q, ko_m)
    assert abs(
        q.expected_discounted_maturity_cashflow
        - m.expected_discounted_maturity_cashflow
    ) < 0.06 * max(1.0, abs(m.expected_discounted_maturity_cashflow))


def test_discrete_ki_cells_zero_reduces_to_hard_mask():
    """The smoothed KI transition reduces EXACTLY to the old hard `ki_mask & ~ko_mask`
    at event_smoothing_cells=0 (frozen discrete-KI baseline)."""
    env = _env()
    ph = _discrete_ki_phoenix(ko_barrier=103.0, ki_barrier=75.0, disable_ko_after_ki=False)
    s = PhoenixQuadEngine(
        params=QuadParams(grid_points=401, event_smoothing_cells=0)
    ).calculate_event_stats(ph, env)
    ko_baseline = np.array([
        0.354468887677, 0.140439806223, 0.077572281329, 0.050650821005,
        0.036339860516, 0.027689214398, 0.021996606161, 0.018017655092,
        0.015108290509, 0.01290530371, 0.011189958953, 0.009823491326,
    ])
    coupon_baseline = np.array([
        0.993817761713, 0.607749348541, 0.435040182387, 0.33364855706,
        0.266761879525, 0.219645023918, 0.184928740459, 0.158469343047,
        0.137756038844, 0.121182902483, 0.107678604857, 0.096503578694,
    ])
    np.testing.assert_allclose(np.asarray(s.ko_probability), ko_baseline, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(s.coupon_probability), coupon_baseline, rtol=0, atol=1e-9)


def test_event_stat_streams_are_plausible():
    env = _env()
    ph = _ref_phoenix()
    s = PhoenixQuadEngine(params=QuadParams(grid_points=401)).calculate_event_stats(ph, env)
    ko = np.asarray(s.ko_probability, dtype=float)
    cp = np.asarray(s.coupon_probability, dtype=float)
    sv = np.asarray(s.survival_probability, dtype=float)
    assert np.all(ko >= -1e-9) and np.all(ko <= 1.0 + 1e-9)
    assert np.all(cp >= -1e-9) and np.all(cp <= 1.0 + 1e-9)
    assert float(np.sum(ko)) <= 1.0 + 1e-9
    # survival monotone non-increasing and in [0,1]
    assert np.all(np.diff(sv) <= 1e-9)
    assert np.all(sv >= -1e-9) and np.all(sv <= 1.0 + 1e-9)
