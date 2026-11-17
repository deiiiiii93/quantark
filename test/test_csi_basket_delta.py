"""
Tests for the shared pricer of ``example/case_studies/csi_basket_delta``.

The study scores per-name delta methods for a European option on a
customized strategy index (CSI) against an exact MC pathwise reference, with a
moment-matched (MM) lognormal as the desk-style fast model.  These tests pin
the identities the study's conclusions rest on:

* the MM pricer collapses to Black-Scholes for a one-name basket (also with a
  decrement and a mid-period stub);
* the MM per-name chain rule equals a fixed-strike finite difference;
* bad inputs (weights, correlation, schedule state) raise ``ValidationError``;
* the MC pathwise deltas satisfy the Euler identity path by path;
* the exchangeable control, the concentration sign, the moments and the
  rebalance-frequency ordering behave as the design states.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

REPO = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO / "example" / "case_studies" / "csi_basket_delta"


def _load_common():
    path = STUDY_DIR / "_common.py"
    spec = importlib.util.spec_from_file_location("csi_basket_delta_common", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cm = _load_common()


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _bs_call(s, k, t, sigma, r, q):
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / v
    d2 = d1 - v
    price = s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    delta = math.exp(-q * t) * norm.cdf(d1)
    return price, delta


def _single_name(rebalance="none", decrement=0.0):
    return cm.CSIBasket(
        names=("A",),
        target_weights=np.array([1.0]),
        vols=np.array([0.3]),
        div_yields=np.array([0.02]),
        corr=np.array([[1.0]]),
        rate=0.025,
        decrement=decrement,
        rebalance=rebalance,
    )


def _single_state(elapsed=0.0, move=1.0):
    return cm.CSIState(
        index_at_rebalance=1000.0,
        spots_at_rebalance=np.array([50.0]),
        moves=np.array([move]),
        elapsed=elapsed,
    )


# ----------------------------------------------------------------------------
# T1: MM pricer, validation, schedule
# ----------------------------------------------------------------------------


class TestMomentMatchedSingleName:
    """AC1 (MM part): one-name basket == Black-Scholes."""

    def test_mm_equals_bs_rule_a(self):
        basket = _single_name()
        state = _single_state()
        option = cm.CSIOption(moneyness=1.05, maturity=1.0)
        res = cm.mm_price_and_deltas(basket, state, option)

        q = cm.state_quantities(basket, state)
        n = q.shares[0]
        s = q.spots[0]
        k = option.strike(q.index_level)
        # V = n * BS(S, K/n)  (option on n shares of A)
        price, delta = _bs_call(s, k / n, 1.0, 0.3, 0.025, 0.02)
        assert is_close(res.price, n * price, rel_tol=1e-10)
        assert is_close(res.deltas[0], n * delta, rel_tol=1e-10)
        assert is_close(res.delta_index * q.index_level, res.deltas[0] * s, rel_tol=1e-10)

    def test_mm_equals_bs_with_decrement_and_stub(self):
        """Monthly, elapsed = 0.4 P, d = 5%: BS with q + d."""
        basket = _single_name(rebalance="monthly", decrement=0.05)
        period = cm.period_length("monthly")
        state = _single_state(elapsed=0.4 * period, move=1.07)
        option = cm.CSIOption(moneyness=0.95, maturity=1.0, is_call=True)
        res = cm.mm_price_and_deltas(basket, state, option)

        q = cm.state_quantities(basket, state)
        n, s = q.shares[0], q.spots[0]
        k = option.strike(q.index_level)
        price, delta = _bs_call(s, k / n, 1.0, 0.3, 0.025, 0.02 + 0.05)
        assert is_close(res.price, n * price, rel_tol=1e-10)
        assert is_close(res.deltas[0], n * delta, rel_tol=1e-10)

    def test_mm_put_parity_single_name(self):
        basket = _single_name(rebalance="quarterly")
        state = _single_state(elapsed=0.1)
        call = cm.mm_price_and_deltas(basket, state, cm.CSIOption(1.0, 0.5, True))
        put = cm.mm_price_and_deltas(basket, state, cm.CSIOption(1.0, 0.5, False))
        q = cm.state_quantities(basket, state)
        k = 1.0 * q.index_level
        df = math.exp(-0.025 * 0.5)
        assert is_close(call.price - put.price, df * (call.forward - k), rel_tol=1e-10)
        # parity on deltas: dC - dP = df * dF/dS
        dfwd = df * call.forward / q.index_level * q.shares[0]
        assert is_close(call.deltas[0] - put.deltas[0], dfwd, rel_tol=1e-10)

    def test_mm_moment_pin_single_name(self):
        """m1, m2 equal the lognormal closed form."""
        basket = _single_name(rebalance="weekly", decrement=0.03)
        period = cm.period_length("weekly")
        state = _single_state(elapsed=0.5 * period)
        option = cm.CSIOption(1.0, 0.75)
        res = cm.mm_price_and_deltas(basket, state, option)
        i0 = cm.state_quantities(basket, state).index_level
        mu = 0.025 - 0.02 - 0.03
        assert is_close(res.m1, i0 * math.exp(mu * 0.75), rel_tol=1e-12)
        assert is_close(res.m2, i0**2 * math.exp((2 * mu + 0.09) * 0.75), rel_tol=1e-12)
        assert is_close(res.sigma_eff, 0.3, rel_tol=1e-10)


class TestMomentMatchedChainRule:
    """AC8: analytic per-name deltas == fixed-strike central FD on MM."""

    @pytest.mark.parametrize("is_call", [True, False])
    def test_chain_rule_matches_fd(self, is_call):
        basket = cm.base_basket(rebalance="monthly")
        period = cm.period_length("monthly")
        moves = np.ones(5)
        moves[basket.names.index("TECH")] = 1.3
        state = cm.CSIState(
            index_at_rebalance=1000.0,
            spots_at_rebalance=np.array([20.0, 35.0, 80.0, 4.0, 6.0]),
            moves=moves,
            elapsed=0.5 * period,
        )
        option = cm.CSIOption(1.0, 1.0, is_call)
        res = cm.mm_price_and_deltas(basket, state, option)
        fd = cm.mm_fd_deltas(basket, state, option, bump=1e-5)
        scale = np.max(np.abs(res.deltas))
        assert np.max(np.abs(res.deltas - fd)) <= 1e-6 * scale

    def test_delta_index_is_weighted_sum(self):
        basket = cm.base_basket(rebalance="quarterly")
        state = cm.default_state(basket)
        res = cm.mm_price_and_deltas(basket, state, cm.CSIOption(1.1, 0.5))
        q = cm.state_quantities(basket, state)
        assert is_close(
            float(np.dot(q.spots, res.deltas)), res.delta_index * q.index_level, rel_tol=1e-10
        )

    def test_strike_held_fixed_under_bump(self):
        """D14: moving a name's price does not move the strike."""
        basket = cm.base_basket()
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        k0 = option.strike(cm.state_quantities(basket, state).index_level)
        bumped = cm.bump_state(state, 0, 0.01)
        res_b = cm.mm_price_and_deltas(basket, bumped, option, strike=k0)
        res_0 = cm.mm_price_and_deltas(basket, state, option)
        assert res_b.strike == k0 == res_0.strike
        assert res_b.price > res_0.price  # call, positive delta


class TestValidation:
    """AC9."""

    def _kw(self, **over):
        kw = dict(
            names=("A", "B"),
            target_weights=np.array([0.5, 0.5]),
            vols=np.array([0.2, 0.3]),
            div_yields=np.array([0.0, 0.0]),
            corr=np.array([[1.0, 0.3], [0.3, 1.0]]),
            rate=0.02,
        )
        kw.update(over)
        return kw

    def test_weights_sum(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(target_weights=np.array([0.5, 0.6])))

    def test_corr_symmetric(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(corr=np.array([[1.0, 0.3], [0.2, 1.0]])))

    def test_corr_unit_diagonal(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(corr=np.array([[0.9, 0.3], [0.3, 1.0]])))

    def test_corr_bounds(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(corr=np.array([[1.0, 1.2], [1.2, 1.0]])))

    def test_corr_not_psd(self):
        """Uniform rho = -0.3 on 5 names is below the PSD floor -1/(N-1)."""
        corr = np.full((5, 5), -0.3)
        np.fill_diagonal(corr, 1.0)
        with pytest.raises(ValidationError):
            cm.CSIBasket(
                names=tuple("ABCDE"),
                target_weights=np.full(5, 0.2),
                vols=np.full(5, 0.25),
                div_yields=np.zeros(5),
                corr=corr,
                rate=0.02,
            )

    def test_singular_psd_accepted(self):
        corr = np.ones((5, 5))
        basket = cm.CSIBasket(
            names=tuple("ABCDE"),
            target_weights=np.full(5, 0.2),
            vols=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            div_yields=np.zeros(5),
            corr=corr,
            rate=0.02,
        )
        f = cm.corr_factor(basket.corr)
        assert np.allclose(f @ f.T, corr, atol=1e-12)

    def test_vol_positive(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(vols=np.array([0.2, 0.0])))

    def test_unknown_rebalance(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(rebalance="hourly"))

    def test_length_mismatch(self):
        with pytest.raises(ValidationError):
            cm.CSIBasket(**self._kw(vols=np.array([0.2, 0.3, 0.4])))

    def test_elapsed_outside_period(self):
        basket = cm.CSIBasket(**self._kw(rebalance="monthly"))
        state = cm.CSIState(
            index_at_rebalance=100.0,
            spots_at_rebalance=np.array([10.0, 10.0]),
            elapsed=cm.period_length("monthly"),
        )
        with pytest.raises(ValidationError):
            cm.state_quantities(basket, state)

    def test_maturity_positive(self):
        with pytest.raises(ValidationError):
            cm.CSIOption(moneyness=1.0, maturity=0.0)

    def test_arrays_read_only(self):
        basket = cm.CSIBasket(**self._kw())
        with pytest.raises(ValueError):
            basket.vols[0] = 0.5


class TestSchedule:
    """D4 / D19."""

    def test_none_is_single_period(self):
        basket = _single_name()
        times = cm.schedule_times(basket, _single_state(elapsed=0.3), 1.0)
        assert np.array_equal(times, np.array([1.0]))

    def test_monthly_from_rebalance_date(self):
        basket = _single_name(rebalance="monthly")
        times = cm.schedule_times(basket, _single_state(), 1.0)
        assert len(times) == 12
        assert is_close(times[0], 1.0 / 12, rel_tol=1e-12)
        assert times[-1] == 1.0

    def test_no_sliver_period_at_maturity(self):
        basket = _single_name(rebalance="daily")
        times = cm.schedule_times(basket, _single_state(), 1.0)
        assert len(times) == 244
        assert np.min(np.diff(times)) > 0.5 / 244

    def test_stub_then_periods(self):
        basket = _single_name(rebalance="quarterly")
        times = cm.schedule_times(basket, _single_state(elapsed=0.1), 1.0)
        assert is_close(times[0], 0.15, rel_tol=1e-12)
        assert np.allclose(np.diff(times[:-1]), 0.25)
        assert times[-1] == 1.0

    def test_maturity_inside_stub(self):
        basket = _single_name(rebalance="quarterly")
        times = cm.schedule_times(basket, _single_state(elapsed=0.1), 0.1)
        assert np.array_equal(times, np.array([0.1]))


# ----------------------------------------------------------------------------
# T2: MC reference, delta methods, metrics
# ----------------------------------------------------------------------------

PATHS = 2**12
BATCHES = 16
SEED = 20261117


def _mc(basket, state, option, paths=PATHS, batches=BATCHES):
    return cm.mc_price_and_deltas(
        basket, state, option, n_paths=paths, n_batches=batches, seed=SEED
    )


def _within(diff, se, k=3.0, extra=0.0):
    return np.all(np.abs(np.atleast_1d(diff)) <= k * np.atleast_1d(se) + extra)


class TestMCSingleName:
    """AC1 (MC part)."""

    @pytest.mark.parametrize(
        "rebalance,decrement,elapsed_frac,move",
        [("none", 0.0, 0.0, 1.0), ("monthly", 0.05, 0.4, 1.07)],
    )
    def test_mc_matches_bs(self, rebalance, decrement, elapsed_frac, move):
        basket = _single_name(rebalance=rebalance, decrement=decrement)
        period = cm.period_length(rebalance) or 0.0
        state = _single_state(elapsed=elapsed_frac * period, move=move)
        option = cm.CSIOption(moneyness=1.0, maturity=1.0)
        mc = _mc(basket, state, option)
        q = cm.state_quantities(basket, state)
        n, s = q.shares[0], q.spots[0]
        price, delta = _bs_call(s, option.strike(q.index_level) / n, 1.0, 0.3, 0.025, 0.02 + decrement)
        assert _within(mc.price - n * price, mc.price_se)
        assert _within(mc.deltas_pathwise[0] - n * delta, mc.deltas_pathwise_se[0])


class TestEuler:
    """AC2."""

    def test_pathwise_euler_identity(self):
        basket = cm.base_basket(rebalance="monthly", decrement=0.05)
        moves = np.ones(5)
        moves[2] = 1.3
        state = cm.default_state(basket, elapsed=0.4 * cm.period_length("monthly"), moves=moves)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option, paths=2**10, batches=4)
        q = cm.state_quantities(basket, state)
        assert mc.euler_max_rel_err <= 1e-12
        assert is_close(
            float(np.dot(q.spots, mc.deltas_pathwise)),
            mc.delta_index * q.index_level,
            rel_tol=1e-10,
        )


class TestControlAndConcentration:
    """AC3, AC4."""

    def test_exchangeable_control(self):
        basket = cm.control_basket(rebalance="monthly")
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option)
        mm = cm.mm_price_and_deltas(basket, state, option)
        methods = cm.delta_methods(basket, state, option, mc, mm)
        bias = cm.fd_bias_allowance(basket, state, option, mc.fd_bump)
        for key in ("M1", "M2"):
            assert _within(methods[key].diff_mean, methods[key].diff_se), key
        assert _within(methods["M3"].diff_mean, methods["M3"].diff_se, extra=bias)
        assert np.allclose(methods["M5"].deltas, methods["M4"].deltas, rtol=1e-10, atol=0.0)

    def test_concentration_sign(self):
        basket = cm.concentration_basket()
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option)
        q = cm.state_quantities(basket, state)
        gap = mc.batches["deltas_pathwise"] / q.shares - mc.batches["delta_index"][:, None]
        mean = gap.mean(axis=0)
        se = gap.std(axis=0, ddof=1) / math.sqrt(gap.shape[0])
        etf, gold = basket.names.index("ETF"), basket.names.index("GOLD")
        assert mean[etf] > 3.0 * se[etf]
        assert mean[gold] < -3.0 * se[gold]


class TestMoments:
    """AC5 and AC14 (moment part)."""

    @pytest.mark.parametrize(
        "basket_fn,elapsed_frac",
        [
            (lambda: cm.base_basket("none"), 0.0),
            (lambda: cm.base_basket("monthly"), 0.0),
            (lambda: cm.base_basket("daily"), 0.0),
            (lambda: cm.base_basket("monthly", decrement=0.05), 0.4),
            (lambda: cm.base_basket("none", corr=cm.uniform_corr(5, 1.0)), 0.0),
            (lambda: cm.two_name_basket(-1.0), 0.0),
            (lambda: cm.two_name_basket(-1.0, rebalance="monthly"), 0.4),
        ],
    )
    def test_sample_moments_match_closed_form(self, basket_fn, elapsed_frac):
        basket = basket_fn()
        period = cm.period_length(basket.rebalance) or 0.0
        state = cm.default_state(basket, elapsed=elapsed_frac * period)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option, paths=2**11, batches=16)
        mm = cm.mm_price_and_deltas(basket, state, option)
        assert _within(mc.sample_m1 - mm.m1, mc.sample_m1_se)
        assert _within(mc.sample_m2 - mm.m2, mc.sample_m2_se)
        assert mc.euler_max_rel_err <= 1e-12
        assert np.isfinite(mc.price) and np.isfinite(mm.price)


class TestRebalanceOrderingAndFD:
    """AC6, AC7."""

    def _max_m1_gap_cash(self, rebalance):
        basket = cm.base_basket(rebalance=rebalance)
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option)
        mm = cm.mm_price_and_deltas(basket, state, option)
        methods = cm.delta_methods(basket, state, option, mc, mm)
        q = cm.state_quantities(basket, state)
        err, _ = cm.delta_cash_error_per_pct(methods["M1"], q.spots, option.units(q.index_level))
        return float(np.max(np.abs(err)))

    def test_daily_gap_below_fixed_share_gap(self):
        assert self._max_m1_gap_cash("daily") < self._max_m1_gap_cash("none")

    def test_fd_matches_pathwise(self):
        basket = cm.base_basket()
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        mc = _mc(basket, state, option)
        mm = cm.mm_price_and_deltas(basket, state, option)
        methods = cm.delta_methods(basket, state, option, mc, mm)
        bias = cm.fd_bias_allowance(basket, state, option, mc.fd_bump)
        assert _within(methods["M3"].diff_mean, methods["M3"].diff_se, extra=bias)


class TestEvaluateScenario:
    """T3: the JSON-ready scenario record the sweep scripts write."""

    def test_record_is_json_and_consistent(self):
        import json

        basket = cm.concentration_basket()
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        rec = cm.evaluate_scenario(basket, state, option, "conc", n_paths=2**12, n_batches=16)
        json.dumps(rec)
        m1 = rec["methods"]["M1"]
        etf = rec["names"].index("ETF")
        assert m1["resolved"][etf] and m1["gap_cash_per_pct"][etf] < 0.0
        for key in ("M1", "M2", "M3", "M4", "M5"):
            gap = np.array(rec["methods"][key]["gap_cash_per_pct"])
            se = np.array(rec["methods"][key]["gap_se"])
            allow = np.array(rec["fd_bias_cash_per_pct"]) if key == "M3" else 0.0
            assert np.array_equal(
                np.array(rec["methods"][key]["resolved"]), np.abs(gap) > 3.0 * se + allow
            )
        tot = rec["total_delta_cash_per_pct"]["mc"]
        assert is_close(sum(rec["ref_cash_per_pct"]), tot, rel_tol=1e-10)

    def test_corr_band_zero_when_unchanged(self):
        basket = cm.base_basket()
        state = cm.default_state(basket)
        option = cm.CSIOption(1.0, 1.0)
        band = cm.ref_corr_band(
            basket, state, option, basket.corr, basket.corr, n_paths=2**10, n_batches=4
        )
        assert band == [0.0] * 5


def test_common_imports_in_fresh_interpreter():
    """Scripts load _common.py first; it must not hit the montecarlo import cycle."""
    import subprocess

    code = (
        "import importlib.util, sys;"
        f"s = importlib.util.spec_from_file_location('c', r'{STUDY_DIR / '_common.py'}');"
        "m = importlib.util.module_from_spec(s); sys.modules['c'] = m; s.loader.exec_module(m);"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


class TestResidualSigma:
    """D16 estimator."""

    def test_constant_errors_give_exact_norm(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        e = np.tile(np.array([100.0, -50.0]), (8, 1))
        res = cm.residual_daily_pnl_sigma(e, cov)
        assert is_close(res.sigma, math.sqrt(e[0] @ cov @ e[0]), rel_tol=1e-12)
        assert res.q_se == 0.0 and res.resolved

    def test_pure_noise_is_not_resolved(self):
        rng = np.random.default_rng(7)
        cov = np.eye(3) * 1e-4
        e = rng.normal(size=(40, 3))
        assert not cm.residual_daily_pnl_sigma(e, cov).resolved
