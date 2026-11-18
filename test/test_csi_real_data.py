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

    def test_name_without_close_at_the_first_session_raises(self):
        sessions = _sessions()
        a = pd.Series(1.0, index=sessions)
        b = pd.Series(1.0, index=sessions[5:])
        prices, _ = data.align_prices({"A": a, "B": b}, sessions)
        assert prices.index[0] == sessions[5]                   # range starts when every name has traded
        with pytest.raises(MarketDataError):
            data.align_prices({"A": a, "B": pd.Series(dtype=float)}, sessions)

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
