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
