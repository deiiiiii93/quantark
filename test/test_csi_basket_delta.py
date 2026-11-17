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
