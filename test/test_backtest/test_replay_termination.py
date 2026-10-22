"""Pending-settlement KO termination (spec §8, plan Task 14)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.asset.equity.lifecycle.autocallable import (  # noqa: E402
    AutocallableLifecycleTracker,
)
from quantark.backtest.replay import (  # noqa: E402
    HedgeSpec,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
)
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402


def _run_single(config):
    return AutocallableBacktestEngine(config).run()


def _shift_ko_settlement(monkeypatch, shift_years: float):
    """Push every KO settlement_time out by ``shift_years`` via the tracker."""
    original = AutocallableLifecycleTracker._scheduled_records

    def shifted(self, product, env, kind):
        records = original(self, product, env, kind)
        if kind != "ko":
            return records
        for rec in records:
            rec["settlement_time"] = rec["time"] + shift_years
            rec["settlement_date"] = self._date_resolver(
                (
                    rec["date"] + pd.Timedelta(days=int(round(shift_years * 365)))
                ).normalize()
            )
        return records

    monkeypatch.setattr(AutocallableLifecycleTracker, "_scheduled_records", shifted)


def test_t0_termination_stops_on_ko_date():
    config = fixtures.make_scalar_bsm_config()
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    states = results.states_df
    # Path KOs on 2024-01-04 (spot 104 >= 103); T+0 settlement ends the run there.
    assert states.index[-1] == pd.Timestamp("2024-01-04")
    summary = results.get_summary()
    assert summary["termination_reason"] == "ko"
    assert summary["days_replayed"] == 3
    assert summary["days_in_contract"] == 5
    assert summary["days_replayed"] < summary["days_in_contract"]
    # KO cash posted on the observation date.
    assert states.loc[pd.Timestamp("2024-01-04"), "cashflows"] != 0.0


def test_flag_off_replays_to_data_end_matching_goldens():
    config = fixtures.make_scalar_bsm_config()  # fixture pins flag off
    results = _run_single(config)
    assert len(results.states_df) == 5
    assert results.get_summary()["termination_reason"] == "ko"
    assert results.get_summary()["days_replayed"] == 5


def test_t5_settlement_carries_receivable_then_posts(monkeypatch):
    _shift_ko_settlement(monkeypatch, 2.0 / 365.0)  # settle two days after obs
    config = fixtures.make_scalar_bsm_config()
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    states = results.states_df

    ko_day = pd.Timestamp("2024-01-04")
    settle_day = pd.Timestamp("2024-01-06")
    # No KO cash on the observation date; the run continues to settlement.
    assert states.loc[ko_day, "cashflows"] == 0.0
    # Between observation and settlement the receivable is carried at PV:
    # portfolio value stays near the terminal cash, not collapsed to hedge-only.
    assert states.loc[ko_day, "portfolio_value"] != states.loc[ko_day, "cash"]
    assert states.index[-1] == settle_day
    assert states.loc[settle_day, "cashflows"] != 0.0
    summary = results.get_summary()
    assert summary["termination_reason"] == "ko"
    assert summary["days_replayed"] == 5


def test_expiry_paid_ko_runs_dead_to_maturity(monkeypatch):
    # Settlement at the product's final maturity (6/365): beyond the 5-day
    # data window, so the run must go to data end with the receivable open —
    # and be LABELED data_end, never a completed "ko" (review finding).
    _shift_ko_settlement(monkeypatch, 4.0 / 365.0)
    config = fixtures.make_scalar_bsm_config()
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    states = results.states_df
    assert len(states) == 5  # data end reached, still unsettled
    ko_day = pd.Timestamp("2024-01-04")
    assert states.loc[ko_day:, "cashflows"].eq(0.0).all()
    summary = results.get_summary()
    assert summary["days_replayed"] == 5
    assert summary["termination_reason"] == "data_end"
    assert summary["all_settled"] is False
    assert summary["outstanding_receivable"] != 0.0


def test_pnl_identity_holds_across_pending_settlement(monkeypatch):
    """portfolio_value - total_pnl must be constant (the initial book value)
    on observation, pending, and settlement days — the receivable belongs in
    marked P&L, not only in portfolio value (review finding: phantom P&L)."""
    _shift_ko_settlement(monkeypatch, 2.0 / 365.0)
    config = fixtures.make_scalar_bsm_config()
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    states = results.states_df
    anchor = states["portfolio_value"] - states["total_pnl"]
    assert (anchor - anchor.iloc[0]).abs().max() < 1e-9
    # No phantom jump at settlement: day-over-day P&L move across the
    # settlement boundary is small next to the receivable magnitude.
    settle_day = pd.Timestamp("2024-01-06")
    prev_day = pd.Timestamp("2024-01-05")
    jump = abs(
        states.loc[settle_day, "total_pnl"] - states.loc[prev_day, "total_pnl"]
    )
    assert jump < 100.0  # discounting drift only, not the ~10k terminal cash


def _extended_market(spots):
    """8-day market window (the golden product matures on day 6)."""
    dates = pd.date_range("2024-01-02", periods=len(spots), freq="D")
    spot_data = pd.DataFrame({"date": dates, "spot": spots})
    vol_data = pd.DataFrame({"date": dates, "volatility": [0.22] * len(spots)})
    rate_data = pd.DataFrame({"date": dates, "rate": [fixtures.RATE] * len(spots)})
    futures_rows = []
    for d, spot in zip(dates, spots):
        futures_rows.append(
            {
                "date": d,
                "contract": "IF2402",
                "futures_price": spot * 1.01,
                "expiry_date": pd.Timestamp("2024-02-16"),
                "multiplier": 300.0,
            }
        )
    return fixtures.AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data,
        vol_data=vol_data,
        rate_data=rate_data,
        futures_data=pd.DataFrame(futures_rows),
    )


def test_clean_maturity_terminates_with_reason_maturity():
    # Flat path: no KO (barrier 103), no KI (barrier 97); the golden product
    # matures at 6/365 -> settles 2024-01-08, inside the 8-day window.
    config = fixtures.make_scalar_bsm_config()
    config.market_data = _extended_market([100.0] * 8)
    config.calculate_surfaces = False
    config.calculate_event_probabilities = False
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    summary = results.get_summary()
    assert summary["termination_reason"] == "maturity"
    assert results.states_df.index[-1] == pd.Timestamp("2024-01-08")
    assert summary["days_replayed"] < summary["days_in_contract"]


def test_ki_maturity_terminates_with_reason_ki_maturity():
    # Dip to 96 on the KI observation (2024-01-03 / 2024-01-05 window), never
    # touching KO at 103: knocked-in run to maturity.
    config = fixtures.make_scalar_bsm_config()
    config.market_data = _extended_market(
        [100.0, 96.0, 98.0, 96.0, 99.0, 100.0, 100.0, 100.0]
    )
    config.calculate_surfaces = False
    config.calculate_event_probabilities = False
    config.terminate_on_lifecycle_end = True
    results = _run_single(config)
    summary = results.get_summary()
    assert summary["termination_reason"] == "ki_maturity"
    lifecycle_events = results.get_lifecycle_events()
    assert (lifecycle_events["event_type"] == "KI").any()


def test_book_terminates_only_when_all_products_settled():
    # Product A KOs on 2024-01-04 (spot 104 >= 103, T+0); product B has an
    # unreachable KO and no KI touch, so it matures on 2024-01-08. The book
    # must run until BOTH are settled.
    product_b = fixtures.create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=6.0 / 365.0,
        contract_multiplier=100.0,
        ko_barrier=1000.0,
        ki_barrier=1.0,
        ko_rate=0.02,
        num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=fixtures.ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0],
        include_principal=True,
    )
    config = ReplayBacktestConfig(
        products=[
            ReplayProduct(product=fixtures._snowball_product(), quantity=-1.0,
                          position_id=1, has_lifecycle=True),
            ReplayProduct(product=product_b, quantity=-1.0, position_id=2,
                          has_lifecycle=True),
        ],
        market_data=_extended_market(
            [100.0, 100.0, 104.0, 105.0, 104.0, 104.0, 104.0, 104.0]
        ),
        hedge=HedgeSpec(kind="futures"),
        engine_config=fixtures.make_scalar_bsm_config().engine_config,
        transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        terminate_on_lifecycle_end=True,
    )
    results = ReplayBacktestEngine(config).run()
    states = results.states_df()
    assert states["date"].iloc[-1] == pd.Timestamp("2024-01-08")
    assert results.get_summary()["termination_reason"] == "ko"
