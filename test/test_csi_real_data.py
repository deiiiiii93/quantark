"""
Offline tests for ``example/case_studies/csi_real_data``.

The study builds a CSI from real daily prices (akshare, cached outside the
repo) under five rebalance rules and compares a single-underlying Black model
(index vol from the index's own history, total delta split by current weights)
with a per-name Monte Carlo reference.  Nothing here touches the network: every
test runs on small synthetic price frames and calendars.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.util.exceptions import MarketDataError, ValidationError

REPO = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO / "example" / "case_studies" / "csi_real_data"


def _load(name: str):
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"csi_real_data_{name.lstrip('_')}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data = _load("_data")
index = _load("_index")
models = _load("_models")

RULES = ("none", "quarterly", "monthly", "weekly", "daily")


def _holiday_calendar():
    """2023-12 .. 2024-06 business days minus a few holidays (incl. Mon 2024-01-01, 2024-04-01, May 1-3)."""
    days = pd.bdate_range("2023-12-01", "2024-06-28")
    holidays = pd.DatetimeIndex(["2024-01-01", "2024-04-01", "2024-04-04", "2024-05-01", "2024-05-02", "2024-05-03"])
    return days.difference(holidays)


def _price_frame(sessions, seed=3):
    rng = np.random.default_rng(seed)
    vols = np.array([0.25, 0.25, 0.45, 0.18, 0.10]) / np.sqrt(244)
    steps = rng.standard_normal((len(sessions), 5)) * vols
    levels = np.array([1500.0, 35.0, 48.0, 4.1, 6.2]) * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(levels, index=sessions, columns=[n.code for n in data.BASKET])


def _sessions(start="2024-01-02", n=30):
    return pd.bdate_range(start, periods=n)


# ----------------------------------------------------------------------------
# T2: data layer
# ----------------------------------------------------------------------------


class TestDataLayer:
    """AC1, AC2."""

    def test_basket_weights_sum_to_one(self):
        assert len(data.BASKET) == 5
        assert abs(sum(n.weight for n in data.BASKET) - 1.0) < 1e-12

    def test_missing_cache_names_fetch_script(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CSI_REAL_DATA_CACHE", str(tmp_path / "nothing_here"))
        with pytest.raises(MarketDataError, match="01_fetch.py"):
            data.load_prices()

    def test_cache_inside_repo_is_refused(self, monkeypatch):
        monkeypatch.setenv("CSI_REAL_DATA_CACHE", str(STUDY_DIR / "cache"))
        with pytest.raises(ValidationError):
            data.cache_dir()

    def test_suspended_session_is_forward_filled_and_counted(self):
        sessions = _sessions()
        a = pd.Series(np.linspace(10.0, 12.9, 30), index=sessions)
        b = pd.Series(np.linspace(5.0, 7.9, 30), index=sessions).drop(sessions[[7, 8]])
        prices, fills = data.align_prices({"A": a, "B": b}, sessions)
        assert fills == {"A": 0, "B": 2}
        assert prices.loc[sessions[7], "B"] == b.loc[sessions[6]]
        assert prices.loc[sessions[8], "B"] == b.loc[sessions[6]]
        assert list(prices.index) == list(sessions)

    def test_close_on_non_session_day_raises(self):
        sessions = _sessions()
        a = pd.Series(1.0, index=sessions)
        b = pd.Series(1.0, index=sessions.append(pd.DatetimeIndex(["2024-01-06"])))  # a Saturday
        with pytest.raises(MarketDataError):
            data.align_prices({"A": a, "B": b}, sessions)

    def test_late_starter_truncates_the_range_and_empty_or_duplicate_raises(self):
        sessions = _sessions()
        a = pd.Series(1.0, index=sessions)
        b = pd.Series(1.0, index=sessions[5:])
        prices, _ = data.align_prices({"A": a, "B": b}, sessions)
        assert prices.index[0] == sessions[5]                   # range starts when every name has traded
        with pytest.raises(MarketDataError):
            data.align_prices({"A": a, "B": pd.Series(dtype=float)}, sessions)
        dup = pd.Series(1.0, index=sessions[:3].append(sessions[2:3]))
        with pytest.raises(MarketDataError):
            data.align_prices({"A": a, "B": dup}, sessions)

    def test_intraday_bar_is_not_a_close(self):
        sessions = pd.DatetimeIndex(["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19"])
        assert data.last_closed_session(sessions, pd.Timestamp("2026-11-18 11:30")) == sessions[1]
        assert data.last_closed_session(sessions, pd.Timestamp("2026-11-18 15:45")) == sessions[2]
        assert data.last_closed_session(sessions, pd.Timestamp("2026-11-21 09:00")) == sessions[3]

    def test_round_trip_through_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CSI_REAL_DATA_CACHE", str(tmp_path))
        sessions = _sessions()
        raw = {n.code: pd.Series(np.linspace(1.0, 2.0, 30) * (k + 1), index=sessions)
               for k, n in enumerate(data.BASKET)}
        data.save_cache(raw, sessions, {"akshare": "test", "sources": {}})
        prices = data.load_prices()
        assert list(prices.columns) == [n.code for n in data.BASKET]
        assert len(prices) == 30
        assert list(data.load_calendar()) == list(sessions)
        assert data.load_meta()["akshare"] == "test"


# ----------------------------------------------------------------------------
# T3: index construction
# ----------------------------------------------------------------------------


class TestRebalanceDates:
    """AC3."""

    CAL = _holiday_calendar()
    START, END = pd.Timestamp("2023-12-01"), pd.Timestamp("2024-06-28")

    def _dates(self, rule):
        return list(index.rebalance_dates(self.CAL, rule, self.START, self.END).strftime("%Y-%m-%d"))

    def test_quarterly_first_session_of_quarter(self):
        assert self._dates("quarterly") == ["2024-01-02", "2024-04-02"]

    def test_monthly_first_session_of_month_skips_holidays(self):
        assert self._dates("monthly") == ["2024-01-02", "2024-02-01", "2024-03-01", "2024-04-02",
                                          "2024-05-06", "2024-06-03"]

    def test_weekly_first_session_of_iso_week(self):
        weekly = self._dates("weekly")
        assert "2024-01-02" in weekly and "2024-01-01" not in weekly       # week starting on a holiday
        assert "2024-05-06" in weekly and "2024-04-29" in weekly
        assert all(pd.Timestamp(d).isocalendar()[1] != pd.Timestamp(e).isocalendar()[1]
                   for d, e in zip(weekly, weekly[1:]))

    def test_daily_and_none(self):
        assert len(self._dates("daily")) == len(self.CAL) - 1               # base date excluded
        assert self._dates("none") == []

    def test_base_date_is_not_a_rebalance(self):
        assert "2023-12-01" not in self._dates("monthly")


class TestBuildIndex:
    """AC4."""

    CAL = _holiday_calendar()
    PRICES = _price_frame(CAL)
    W = data.target_weights()

    def _build(self, rule):
        return index.build_index(self.PRICES, self.W, rule, self.CAL[0], 1000.0)

    @pytest.mark.parametrize("rule", RULES)
    def test_level_identity_with_shares_before_the_close(self, rule):
        path = self._build(rule)
        assert path.level.iloc[0] == 1000.0
        held = path.shares.shift(1).iloc[1:]                                # shares held into each close
        recomputed = (held * self.PRICES.iloc[1:]).sum(axis=1)
        np.testing.assert_allclose(recomputed.values, path.level.iloc[1:].values, rtol=1e-12)

    @pytest.mark.parametrize("rule", ["quarterly", "monthly", "weekly", "daily"])
    def test_weights_reset_to_target_on_rebalance_dates(self, rule):
        path = self._build(rule)
        assert len(path.rebalance_dates) > 0
        for d in path.rebalance_dates:
            np.testing.assert_allclose(path.weights.loc[d].values, self.W.values, rtol=1e-12)

    def test_daily_rule_return_is_weighted_name_return(self):
        path = self._build("daily")
        ret = path.level.pct_change().iloc[1:]
        expected = (self.PRICES.pct_change().iloc[1:] * self.W.values).sum(axis=1)
        np.testing.assert_allclose(ret.values, expected.values, rtol=1e-11, atol=1e-14)

    def test_none_rule_keeps_shares_and_drifts_weights(self):
        path = self._build("none")
        assert (path.shares.nunique() == 1).all()
        assert np.abs(path.weights.iloc[-1].values - self.W.values).max() > 1e-3

    def test_rebalance_does_not_move_the_level(self):
        path = self._build("monthly")
        d = path.rebalance_dates[1]
        before = float(path.shares.shift(1).loc[d] @ self.PRICES.loc[d])
        after = float(path.shares.loc[d] @ self.PRICES.loc[d])
        assert abs(after / before - 1.0) < 1e-12


# ----------------------------------------------------------------------------
# T4: models
# ----------------------------------------------------------------------------


def _bs_call(s, k, t, sigma, r):
    from scipy.stats import norm

    v = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / v
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d1 - v)


class TestSchedule:
    """AC5 and the expiry / weekday-fallback helpers."""

    CAL = _holiday_calendar()

    def _n(self, a, b):
        return int(self.CAL.get_loc(pd.Timestamp(b)) - self.CAL.get_loc(pd.Timestamp(a)))

    def test_monthly_schedule_counts_sessions_and_ends_at_expiry(self):
        asof, expiry = "2024-01-15", "2024-05-10"
        times = models.remaining_schedule(self.CAL, "monthly", asof, expiry)
        expected = [self._n(asof, d) for d in ("2024-02-01", "2024-03-01", "2024-04-02", "2024-05-06", expiry)]
        np.testing.assert_allclose(times * index.TRADING_DAYS, expected, rtol=0, atol=1e-9)

    def test_asof_rebalance_excluded_and_dates_at_or_after_expiry_dropped(self):
        times = models.remaining_schedule(self.CAL, "monthly", "2024-02-01", "2024-05-06")
        expected = [self._n("2024-02-01", d) for d in ("2024-03-01", "2024-04-02", "2024-05-06")]
        np.testing.assert_allclose(times * index.TRADING_DAYS, expected, rtol=0, atol=1e-9)

    def test_none_rule_is_just_the_expiry(self):
        times = models.remaining_schedule(self.CAL, "none", "2024-01-15", "2024-05-10")
        assert len(times) == 1

    def test_expiry_date_is_n_sessions_ahead_with_weekday_fallback(self):
        expiry, fallback = models.expiry_date(self.CAL, "2024-06-24", 10)  # 4 listed sessions + 6 weekdays
        assert fallback and expiry == pd.Timestamp("2024-07-08")            # 2024-06-28 is the last listed session
        expiry, fallback = models.expiry_date(self.CAL, "2024-01-15", 5)
        assert not fallback and self._n("2024-01-15", expiry) == 5

    def test_non_session_dates_raise_validation_error(self):
        with pytest.raises(ValidationError):
            models.remaining_schedule(self.CAL, "monthly", "2024-04-01", "2024-05-10")
        with pytest.raises(ValidationError):
            models.expiry_date(self.CAL, "2024-01-01", 5)

    def test_extended_calendar_appends_weekdays(self):
        ext = models.extended_calendar(self.CAL, pd.Timestamp("2024-07-05"))
        assert ext[-1] == pd.Timestamp("2024-07-05") and ext[: len(self.CAL)].equals(self.CAL)


class TestCalibration:
    """AC7 (convention D13)."""

    CAL = _holiday_calendar()
    PRICES = _price_frame(CAL, seed=11)

    def test_matches_numpy_with_the_d13_convention(self):
        level = index.build_index(self.PRICES, data.target_weights(), "monthly", self.CAL[0]).level
        asof = self.CAL[60]
        cal = models.calibrate(self.PRICES, level, asof, window=40)
        r = np.diff(np.log(self.PRICES.loc[:asof].iloc[-41:].to_numpy()), axis=0)
        cov = np.cov(r, rowvar=False, ddof=1) * 244
        np.testing.assert_allclose(cal.vols, np.sqrt(np.diag(cov)), rtol=1e-12)
        np.testing.assert_allclose(cal.corr, cov / np.outer(cal.vols, cal.vols), rtol=1e-12, atol=1e-14)
        ri = np.diff(np.log(level.loc[:asof].iloc[-41:].to_numpy()))
        assert abs(cal.index_vol - np.std(ri, ddof=1) * np.sqrt(244)) < 1e-14
        assert cal.n_obs == 40 and not cal.psd_repaired

    def test_non_session_asof_raises(self):
        level = index.build_index(self.PRICES, data.target_weights(), "none", self.CAL[0]).level
        with pytest.raises(ValidationError):
            models.calibrate(self.PRICES, level, pd.Timestamp("2024-04-01"), window=40)   # a holiday

    def test_short_history_raises(self):
        level = index.build_index(self.PRICES, data.target_weights(), "none", self.CAL[0]).level
        with pytest.raises(ValidationError):
            models.calibrate(self.PRICES, level, self.CAL[30], window=40)


class TestBlack:
    """AC8."""

    def test_price_delta_gamma(self):
        s, k, t, v, r = 1000.0, 950.0, 0.8, 0.22, 0.018
        pv, delta, gamma = models.black_call(s, k, t, v, r)
        assert abs(pv - _bs_call(s, k, t, v, r)) < 1e-9
        h = 1e-4 * s                                  # O(h²) truncation ≈ 1e-8 on Δ
        up, dn = _bs_call(s + h, k, t, v, r), _bs_call(s - h, k, t, v, r)
        assert abs(delta - (up - dn) / (2 * h)) < 1e-7
        assert abs(gamma - (up - 2 * pv + dn) / h**2) < 1e-6


def _engine_inputs(rule="monthly"):
    spots = np.array([1500.0, 35.0, 48.0, 4.1, 6.2])
    w = data.target_weights().to_numpy()
    drift = np.array([1.05, 0.97, 1.12, 1.0, 0.98])
    shares = w * 1000.0 / (spots / drift)                       # weights drifted since the last reset
    vols = np.array([0.25, 0.25, 0.45, 0.18, 0.10])
    corr = np.full((5, 5), 0.5)
    np.fill_diagonal(corr, 1.0)
    corr[4, :4] = corr[:4, 4] = 0.05
    cal = models.Calibration(vols=vols, corr=corr, index_vol=0.2, n_obs=244, psd_repaired=False)
    return spots, shares, w, cal


class TestReference:
    """AC9, AC10 and the D10 state encoding."""

    def test_state_encoding_keeps_shares_and_spots(self):
        spots, shares, w, _ = _engine_inputs()
        basket = models.engine_basket(_engine_inputs()[3], w, "monthly", 0.018)
        state = models.engine_state(spots, shares, w)
        q = models.cm.state_quantities(basket, state)
        np.testing.assert_allclose(q.shares, shares, rtol=1e-12)
        np.testing.assert_allclose(q.spots, spots, rtol=1e-12)

    def test_constant_mix_limit_matches_black(self):
        spots, _, w, cal = _engine_inputs()
        shares = w * 1000.0 / spots                              # at target weights, level 1000
        sched = np.arange(1, 245) / 244.0                        # daily rule, 1-session stub, T = 1
        ref = models.reference_call(spots, shares, w, cal, "daily", 1000.0, sched, 0.018,
                                    n_paths=2**12, n_batches=8, seed=7)
        sigma = cal.vols * cal.corr * cal.vols[:, None]
        s_i = float(np.sqrt(w @ sigma @ w))
        pv, delta, _ = models.black_call(1000.0, 1000.0, 1.0, s_i, 0.018)
        assert abs(ref.pv - pv) <= 3 * ref.pv_se + 0.005 * pv
        assert abs(ref.delta_index - delta) <= 3 * ref.delta_index_se + 0.005

    def test_decomposition_is_exact_and_split_sums_to_zero(self):
        spots, shares, w, cal = _engine_inputs()
        level = float(shares @ spots)
        sched = np.array([0.05, 0.05 + 1 / 12, 0.05 + 2 / 12, 0.5])
        ref = models.reference_call(spots, shares, w, cal, "monthly", level, sched, 0.018,
                                    n_paths=2**10, n_batches=4, seed=3)
        weights_now = shares * spots / level
        units = 1e8 / level
        test_dindex = ref.delta_index * 1.03                    # any test-model index delta
        total, split = models.gap_decomposition(spots * ref.deltas_pathwise, test_dindex, ref.delta_index,
                                                weights_now, level, units)
        gap = (weights_now * level * test_dindex - spots * ref.deltas_pathwise) * units * 0.01
        np.testing.assert_allclose(total + split, gap, rtol=1e-9, atol=1e-9 * np.abs(gap).max())
        assert abs(split.sum()) < 1e-9 * np.abs(split).max()

    def test_first_order_split_signs_and_flag(self):
        spots, shares, w, cal = _engine_inputs()
        level = float(shares @ spots)
        weights_now = shares * spots / level
        pred, valid = models.first_order_split(cal, weights_now, level, level, 0.1, 1.0, 0.018,
                                               1e8 / level, "monthly")
        assert valid and abs(pred.sum()) < 1e-9 * np.abs(pred).max()
        assert pred[2] < 0 < pred[4]                             # high-beta TECH under, GOLD over
        _, valid_none = models.first_order_split(cal, weights_now, level, level, 1.0, 1.0, 0.018,
                                                 1e8 / level, "none")
        assert not valid_none
