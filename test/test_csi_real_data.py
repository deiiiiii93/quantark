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

    def test_noise_covariance_matches_the_reported_ses(self):
        spots, shares, w, cal = _engine_inputs()
        level = float(shares @ spots)
        sched = np.array([0.05, 0.05 + 1 / 12, 0.5])
        ref = models.reference_call(spots, shares, w, cal, "monthly", level, sched, 0.018,
                                    n_paths=2**9, n_batches=8, seed=4)
        assert ref.noise_cov.shape == (6, 6)
        np.testing.assert_allclose(np.diag(ref.noise_cov)[:5], ref.deltas_bump_se**2, rtol=1e-10)
        assert abs(ref.noise_cov[5, 5] - ref.delta_index_se**2) < 1e-12 * max(1.0, ref.delta_index_se**2)

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


# ----------------------------------------------------------------------------
# T5: Stage A — the gap through time
# ----------------------------------------------------------------------------


stage_a = _load("03_gap_through_time")


class TestStageA:
    """AC11 on a small synthetic history (window 40, few paths)."""

    SESSIONS = pd.bdate_range("2023-01-02", "2024-03-29")
    PRICES = _price_frame(SESSIONS, seed=21)

    def test_eval_dates_are_the_tenth_session_of_each_month(self):
        dates = stage_a.eval_dates(self.SESSIONS, pd.Timestamp("2024-01-01"))
        assert [d.strftime("%Y-%m-%d") for d in dates] == ["2024-01-12", "2024-02-14", "2024-03-14"]

    @pytest.fixture(scope="class")
    def records(self):
        path = index.build_index(self.PRICES, data.target_weights(), "monthly", self.SESSIONS[0])
        asof = pd.Timestamp("2024-02-14")
        return stage_a.evaluate_date(self.PRICES, path.level, path.shares, path.weights, self.SESSIONS,
                                     "monthly", asof, (0.9, 1.0, 1.1), 0.018, 2**8, 4, 5, window=40)

    def test_one_record_per_moneyness_with_the_contract_fields(self, records):
        assert [r["moneyness"] for r in records] == [0.9, 1.0, 1.1]
        need = {"date", "rule", "moneyness", "strike", "index_level", "tau", "tau_sessions", "expiry",
                "calendar_fallback", "index_vol_hist", "index_vol_model", "pv_test_cny", "pv_ref_cny",
                "pv_ref_se_cny", "dindex_test", "dindex_ref", "dindex_ref_se", "weights",
                "test_cash", "ref_bump", "ref_bump_se", "ref_pathwise", "ref_pathwise_se", "gap_bump",
                "gap_total_part", "gap_split_part", "bump_minus_pathwise", "first_order",
                "first_order_valid", "gap_sigma_cny_day"}
        assert need <= set(records[0])

    def test_record_invariants(self, records):
        for r in records:
            total = np.array(r["gap_total_part"]) + np.array(r["gap_split_part"])
            pathwise_gap = np.array(r["test_cash"]) - np.array(r["ref_pathwise"])
            np.testing.assert_allclose(total, pathwise_gap, rtol=1e-9, atol=1e-6)
            assert abs(sum(r["gap_split_part"])) < 1e-6 * max(1.0, max(map(abs, r["gap_split_part"])))
            np.testing.assert_allclose(np.array(r["test_cash"]).sum() / 1e8 * 100 / r["index_level"]
                                       * r["index_level"], r["dindex_test"], rtol=1e-12)
            assert r["calendar_fallback"] is True                         # the fixture calendar ends in March
            assert 0.0 < r["tau"] <= 1.0 and r["first_order_valid"] is True


# ----------------------------------------------------------------------------
# T6: Stage B — hedge P&L on the real path
# ----------------------------------------------------------------------------


stage_b = _load("04_hedge_backtest")


class TestHedgeAccounting:
    """AC12 accounting (spec D3, D4, D9)."""

    RATE = 0.018
    G = float(np.exp(0.018 / 244))

    def test_financed_path_gives_zero_pnl(self):
        n = 6
        spots = np.outer(self.G ** np.arange(n), [10.0, 20.0])
        marks = 5.0 * self.G ** np.arange(n)
        holdings = np.random.default_rng(0).normal(size=(n - 1, 2))
        pnl = stage_b.daily_pnl(holdings, spots, marks, self.RATE)
        np.testing.assert_allclose(pnl, 0.0, atol=1e-12)

    def test_paired_difference_is_the_pure_hedge_difference(self):
        rng = np.random.default_rng(1)
        spots = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(9, 3)), axis=0))
        marks = 3.0 + rng.normal(0, 0.1, size=9)
        ha, hb = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        diff = stage_b.daily_pnl(ha, spots, marks, self.RATE) - stage_b.daily_pnl(hb, spots, marks, self.RATE)
        expected = ((ha - hb) * (spots[1:] - self.G * spots[:-1])).sum(axis=1)
        np.testing.assert_allclose(diff, expected, rtol=1e-12, atol=1e-12)

    def test_terminal_pnl_matches_a_cash_account_simulation(self):
        rng = np.random.default_rng(2)
        n = 12
        spots = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(n, 2)), axis=0))
        marks = np.abs(rng.normal(2.0, 0.3, size=n))
        payoff = 1.7
        marks[-1] = payoff
        holdings = rng.normal(size=(n - 1, 2))
        premium = 2.2
        cash, h_prev = premium, np.zeros(2)
        for t in range(n - 1):                                     # rebalance at each close, finance at r
            cash -= (holdings[t] - h_prev) @ spots[t]
            h_prev = holdings[t]
            cash *= self.G
        cash += h_prev @ spots[-1] - payoff
        pnl = stage_b.daily_pnl(holdings, spots, marks, self.RATE)
        terminal = stage_b.terminal_pnl(premium, marks[0], pnl, self.RATE)
        assert abs(terminal - cash) < 1e-10


class TestStageBStatistics:
    """AC12 statistics."""

    def test_block_bootstrap_keeps_sessions_together_and_covers_the_truth(self):
        rng = np.random.default_rng(3)
        sessions = np.repeat(np.arange(400), 3)                    # three live starts per session
        common = rng.normal(0, 2.0, size=400)
        values = common[sessions] + rng.normal(0, 0.1, size=sessions.size)
        lo, hi = stage_b.block_bootstrap_std(values, sessions, block=20, n_boot=400, seed=4)
        assert lo < np.std(values, ddof=1) < hi
        assert lo > 1.6 and hi < 2.4

    def test_pair_noise_variance_matches_simulation(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=(6, 6))
        cov = a @ a.T * 1e-6                                       # [bump_1..5, Δ_I] estimator covariance
        x = rng.normal(0, 0.5, size=5)                              # S(t+1) − g·S(t)
        n = np.abs(rng.normal(1.0, 0.3, size=5))
        units = 1e5
        nv = stage_b.pair_noise_var(cov, x, n, units)
        eps = rng.multivariate_normal(np.zeros(6), cov, size=200_000)
        noise = {"reference-test": units * eps[:, :5] @ x,
                 "attribution-test": units * eps[:, 5] * (n @ x),
                 "reference-attribution": units * (eps[:, :5] @ x - eps[:, 5] * (n @ x))}
        for key, sample in noise.items():
            assert abs(nv[key] / np.var(sample) - 1.0) < 0.02, key

    def test_noise_corrected_std_recovers_the_signal(self):
        rng = np.random.default_rng(9)
        sessions = np.repeat(np.arange(500), 4)
        signal = rng.normal(0, 1.0, size=500)[sessions] + rng.normal(0, 0.3, size=sessions.size)
        noise_sd = rng.uniform(1.0, 2.0, size=sessions.size)
        d = signal + rng.normal(0, 1.0, size=sessions.size) * noise_sd
        point, lo, hi = stage_b.block_bootstrap_corrected_std(d, noise_sd**2, sessions, block=20, n_boot=300, seed=1)
        truth = np.std(signal, ddof=1)
        assert lo < truth < hi and abs(point / truth - 1.0) < 0.1

    def test_noise_share_estimator(self):
        rng = np.random.default_rng(5)
        signal = rng.normal(0, 1.0, size=20000)
        d_n = signal + rng.normal(0, 0.2, size=signal.size)             # MC noise sd 0.2 at N paths
        d_2n = signal + rng.normal(0, 0.2 / np.sqrt(2), size=signal.size)
        share = stage_b.mc_noise_share(d_n, d_2n)
        assert abs(share - 0.2 / np.sqrt(1.04)) < 0.01


class TestFixedContract:
    """D9: strike and units are fixed at the start while the index moves."""

    SESSIONS = pd.bdate_range("2023-01-02", "2024-06-28")
    PRICES = _price_frame(SESSIONS, seed=31)

    def test_strike_and_units_stay_at_their_start_values(self, monkeypatch):
        seen = []

        def fake_reference(spots, shares, w_t, cal, rule, strike, schedule, rate, n_paths, n_batches, seed):
            seen.append((strike, schedule[-1]))
            z = np.zeros(len(spots))
            return models.RefResult(1.0, 0.0, 0.5, 0.0, z + 0.1, z, z + 0.1, z)

        monkeypatch.setattr(stage_b._models, "reference_call", fake_reference)
        path = index.build_index(self.PRICES, data.target_weights(), "monthly", self.SESSIONS[0])
        start = pd.Timestamp("2024-01-12")
        res = stage_b.run_start(self.PRICES, path.level, path.shares, self.SESSIONS, "monthly", start,
                                0.018, n_paths=2**6, n_batches=2, seed=1, window=40, sessions=20)
        k0 = float(path.level.loc[start])
        assert res["strike"] == k0 and res["units"] == 1e8 / k0
        assert len(seen) == 20 and all(s == k0 for s, _ in seen)
        assert [round(t * 244) for _, t in seen] == list(range(20, 0, -1))   # remaining maturity shrinks
        assert path.level.loc[start:].iloc[:21].std() > 0                    # the index did move
        assert set(res["arms"]) == {"test", "reference", "attribution"}
        assert all(len(res["arms"][a]["daily_pnl"]) == 20 for a in res["arms"])


# ----------------------------------------------------------------------------
# T7: the HTML report (reads the committed results JSON only)
# ----------------------------------------------------------------------------


RESULTS_DIR = STUDY_DIR / "results"
RAW_PRICE_KEYS = {"close", "closes", "price", "prices", "spot", "spots"}


def _results(stage):
    import json

    return json.loads((RESULTS_DIR / f"{stage}.json").read_text())


def _has_new_stage_b_fields(hedge):
    pair = hedge["summary"]["none"]["pairs"]["reference-test"]
    return {"corrected_std_cny", "corrected_ci90", "resolved", "noise_std_cny"} <= set(pair)


@pytest.fixture(scope="module")
def report():
    return _load("05_report")


@pytest.fixture(scope="module")
def report_page(report):
    return report.build_html()


class TestReport:
    """AC13 (page built from JSON only, chrome, MathML, sections) and AC15 (no raw price series)."""

    def test_build_reads_only_the_results_json(self, report, monkeypatch):
        import builtins

        opened = []
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy)
        monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: spy(self, *a, **k).read())
        page = report.build_html()
        assert page.startswith("<!doctype html>")
        assert opened and all(p.endswith(".json") for p in opened)
        assert {Path(p).name for p in opened} == {"02_indices.json", "03_gap.json", "04_hedge.json"}

    def test_frozen_nav_block(self, report_page):
        assert "position:sticky" in report_page
        assert '<nav class="menu" id="menu">' in report_page
        assert "IntersectionObserver" in report_page
        assert "scroll-padding-top" in report_page

    def test_theme_switch_with_guarded_storage(self, report_page):
        import re

        assert 'id="theme"' in report_page and "<button" in report_page
        uses = report_page.count("localStorage")
        guarded = re.findall(r"try\{[^}]*localStorage[^}]*\}catch", report_page)
        assert uses >= 2 and len(guarded) == uses
        assert "prefers-color-scheme" in report_page
        assert 'data-theme="light"' in report_page

    def test_mathml_and_no_sub_tags(self, report_page):
        assert "<math" in report_page and "<mfrac>" in report_page
        assert "<sub>" not in report_page

    def test_no_negative_zero(self, report_page):
        import re

        assert "-0.000" not in report_page and "−0.000" not in report_page
        assert not re.search("−0(?:\\.0+)?(?![\\d.,])", report_page)

    def test_every_nav_section_present(self, report_page):
        import re

        hrefs = re.findall(r'<a href="#(s\d+)">', report_page)
        assert len(hrefs) >= 7
        for sid in hrefs:
            assert f'<section id="{sid}"' in report_page
        assert set(re.findall(r'<section id="(s\d+)"', report_page)) == set(hrefs)

    def test_links_the_synthetic_study(self, report_page):
        assert 'href="../../csi_basket_delta/results/csi_basket_delta_report.html"' in report_page

    def test_known_stage_a_figure_appears(self, report_page):
        rec = next(r for r in _results("03_gap")["records"] if r["rule"] == "none" and r["moneyness"] == 1.0)
        v = round(100.0 * (rec["pv_test_cny"] / rec["pv_ref_cny"] - 1.0), 2) + 0.0
        assert f"{v:+.2f}%".replace("-", "−") in report_page
        assert "includes MC noise" in report_page

    def test_known_index_figure_appears(self, report_page):
        spread = _results("02_indices")["terminal_spread"]
        assert f"{100.0 * spread:.2f}%" in report_page

    def test_stage_b_noise_corrected_pairs(self, report_page):
        hedge = _results("04_hedge")
        if not _has_new_stage_b_fields(hedge):
            pytest.skip("04_hedge.json predates the noise-corrected pair statistics")
        for rule, s in hedge["summary"].items():
            p = s["pairs"]["reference-test"]
            assert f"{round(p['corrected_std_cny']):,}" in report_page, rule
        if any(not p["resolved"] for s in hedge["summary"].values() for p in s["pairs"].values()):
            assert "not resolved" in report_page

    def test_stage_b_renders_new_fields_from_a_synthesised_file(self, report, tmp_path):
        import json
        import shutil

        for stage in ("02_indices", "03_gap"):
            shutil.copy(RESULTS_DIR / f"{stage}.json", tmp_path / f"{stage}.json")
        hedge = _results("04_hedge")
        for s in hedge["summary"].values():
            for p in s["pairs"].values():
                p.update({"noise_std_cny": 111.0, "corrected_std_cny": 98765.4,
                          "corrected_ci90": [0.0, 123456.7], "resolved": False})
        hedge["summary"]["daily"]["pairs"]["reference-attribution"].update(
            {"corrected_std_cny": 0.0, "corrected_ci90": [0.0, 0.0]})         # all noise: σ and CI collapse to 0
        hedge["noise_probe"].update({"noise_std_measured_cny": 222.0, "noise_std_predicted_cny": 333.0,
                                     "predicted_over_measured": 1.5, "validates": False})
        (tmp_path / "04_hedge.json").write_text(json.dumps(hedge))
        page = report.build_html(tmp_path)
        assert "98,765" in page and "123,457" in page
        assert "not resolved" in page
        assert "does not validate" in page

    def test_no_raw_price_series_in_results(self):
        """AC15: no key named like a raw price series holds a per-name daily list."""

        def series_like(v):
            if isinstance(v, list):
                return len(v) > 1
            if isinstance(v, dict):
                return any(isinstance(x, (list, dict)) for x in v.values())
            return False

        def walk(node, where):
            if isinstance(node, dict):
                for k, v in node.items():
                    assert not (k.lower() in RAW_PRICE_KEYS and series_like(v)), f"{where}/{k}"
                    walk(v, f"{where}/{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{where}[{i}]")

        for stage in ("02_indices", "03_gap", "04_hedge"):
            walk(_results(stage), stage)
        for rule in _results("02_indices")["rules"].values():
            assert "shares" not in rule                                   # D14: shares withheld
