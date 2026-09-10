"""Carry risk frames: fixed columns, both result APIs, explicit NaN.

An empty risk frame must still carry its columns.  A reader looking for
``net_rhoq_bp`` and finding no column cannot tell "this run had no exposure"
from "this run never measured it", and the second is the one that must not be
mistaken for a pass.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from quantark.backtest.replay.results import (
    AutocallableBacktestResults,
    BookBacktestResults,
)
from quantark.backtest.replay.schema import (
    HEDGE_ATTRIBUTION_COLUMNS,
    HEDGE_LEG_COLUMNS,
    HEDGE_STRESS_COLUMNS,
    STATE_COLUMNS,
    TRADE_COLUMNS,
)

EMPTY = dict(
    states=[],
    greeks=[],
    rebalances=[],
    trades=[],
    actions=[],
    daily_event_summary=[],
    event_probabilities=[],
    surfaces=[],
)


def single_results(**kwargs) -> AutocallableBacktestResults:
    return AutocallableBacktestResults(config=None, **EMPTY, **kwargs)


def book_results(**kwargs) -> BookBacktestResults:
    return BookBacktestResults(config=None, products_meta=[], **EMPTY, **kwargs)


def leg_row(date, contract, **overrides):
    row = {name: float("nan") for name in HEDGE_LEG_COLUMNS}
    row.update(
        date=date,
        contract=contract,
        audit_status="not_measured",
        retired=False,
        is_curve_node=True,
        is_correction_leg=False,
        is_last_curve_node=False,
        has_product_tail=False,
    )
    row.update(overrides)
    return row


def attribution_row(date, **overrides):
    row = {name: float("nan") for name in HEDGE_ATTRIBUTION_COLUMNS}
    row.update(
        date=date,
        carry_family="futures_curve",
        objective="spot_parallel",
        audit_status="not_measured",
        attribution_status="ok",
    )
    row.update(overrides)
    return row


def stress_row(date, scenario_id, **overrides):
    row = {name: float("nan") for name in HEDGE_STRESS_COLUMNS}
    row.update(
        date=date,
        scenario_id=scenario_id,
        scenario_family="independent_tail",
        holdings_kind="actual",
        objective="spot_parallel",
        shock_definition="tail +0.01",
        bump_metadata="{}",
        source_basis_assumption="listed futures quotes",
    )
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Empty frames still have columns
# ---------------------------------------------------------------------------


def test_the_single_result_exposes_three_empty_but_shaped_frames():
    results = single_results()
    for frame, columns in (
        (results.hedge_legs_df, HEDGE_LEG_COLUMNS),
        (results.hedge_attribution_df, HEDGE_ATTRIBUTION_COLUMNS),
        (results.hedge_stresses_df, HEDGE_STRESS_COLUMNS),
    ):
        assert frame.empty
        assert frame.index.name == "date"
        assert list(frame.columns) == [c for c in columns if c != "date"]


def test_the_book_result_exposes_the_same_frames_as_methods_with_a_date_column():
    results = book_results()
    for frame, columns in (
        (results.hedge_legs_df(), HEDGE_LEG_COLUMNS),
        (results.hedge_attribution_df(), HEDGE_ATTRIBUTION_COLUMNS),
        (results.hedge_stresses_df(), HEDGE_STRESS_COLUMNS),
    ):
        assert frame.empty
        assert list(frame.columns) == list(columns)
        assert frame.index.name is None


def test_the_column_order_is_the_specs_order():
    assert HEDGE_LEG_COLUMNS[:3] == ("date", "contract", "maturity")
    assert HEDGE_LEG_COLUMNS[-1] == "audit_status"
    assert HEDGE_ATTRIBUTION_COLUMNS[0] == "date"
    assert HEDGE_ATTRIBUTION_COLUMNS[-1] == "spot_gamma_frozen_q_diagnostic"
    assert HEDGE_STRESS_COLUMNS[-1] == "source_basis_assumption"
    # Every column appears once.
    for columns in (
        HEDGE_LEG_COLUMNS,
        HEDGE_ATTRIBUTION_COLUMNS,
        HEDGE_STRESS_COLUMNS,
    ):
        assert len(set(columns)) == len(columns)


def test_the_existing_schema_tuples_are_untouched():
    assert STATE_COLUMNS[0] == "date"
    assert "portfolio_value" in STATE_COLUMNS
    assert TRADE_COLUMNS[0] == "date"
    for new in HEDGE_LEG_COLUMNS[1:]:
        assert new not in STATE_COLUMNS


# ---------------------------------------------------------------------------
# Populated frames
# ---------------------------------------------------------------------------


def test_populated_leg_rows_keep_duplicate_dates_for_different_contracts():
    date = datetime(2025, 3, 3)
    rows = [
        leg_row(date, "IF2503", held_after=-3.0),
        leg_row(date, "IF2506", held_after=1.5),
        leg_row(datetime(2025, 3, 4), "IF2503", held_after=-2.0),
    ]
    frame = single_results(hedge_legs=rows).hedge_legs_df
    assert len(frame) == 3
    assert list(frame.loc[date, "contract"]) == ["IF2503", "IF2506"]
    assert frame.index.is_monotonic_increasing

    book_frame = book_results(hedge_legs=rows).hedge_legs_df()
    assert len(book_frame) == 3
    assert set(book_frame["contract"]) == {"IF2503", "IF2506"}
    assert pd.api.types.is_datetime64_any_dtype(book_frame["date"])


def test_unmeasured_direct_fields_stay_nan_and_keep_their_status():
    rows = [leg_row(datetime(2025, 3, 3), "IF2503", net_rhoq_bp=2.5)]
    frame = single_results(hedge_legs=rows).hedge_legs_df
    assert frame["net_rhoq_bp"].iloc[0] == 2.5
    assert math.isnan(frame["direct_net_rhoq_bp"].iloc[0])
    assert math.isnan(frame["rhoq_audit_error_bp"].iloc[0])
    assert frame["audit_status"].iloc[0] == "not_measured"


def test_explicit_statuses_and_reasons_survive_the_round_trip():
    rows = [
        attribution_row(
            datetime(2025, 3, 3),
            audit_status="fail",
            attribution_status="coordinate_set_changed",
            financing_pnl=0.0,
        )
    ]
    frame = single_results(hedge_attribution=rows).hedge_attribution_df
    assert frame["audit_status"].iloc[0] == "fail"
    assert frame["attribution_status"].iloc[0] == "coordinate_set_changed"
    assert frame["financing_pnl"].iloc[0] == 0.0


def test_stress_rows_carry_their_holdings_kind_and_scenario_definition():
    rows = [
        stress_row(datetime(2025, 3, 3), "tail+0.01", product_pnl=-1234.0, hedge_pnl=0.0),
        stress_row(
            datetime(2025, 3, 3),
            "tail+0.01",
            holdings_kind="ideal",
            product_pnl=-1234.0,
            hedge_pnl=0.0,
        ),
    ]
    frame = book_results(hedge_stresses=rows).hedge_stresses_df()
    assert list(frame["holdings_kind"]) == ["actual", "ideal"]
    assert set(frame["hedge_pnl"]) == {0.0}
    assert frame["scenario_family"].iloc[0] == "independent_tail"


def test_rows_are_copied_not_aliased():
    rows = [leg_row(datetime(2025, 3, 3), "IF2503", held_after=-3.0)]
    results = single_results(hedge_legs=rows)
    rows[0]["held_after"] = 99.0
    rows.clear()
    frame = results.hedge_legs_df
    assert len(frame) == 1
    assert frame["held_after"].iloc[0] == -3.0


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_old_callers_construct_both_results_without_the_new_arguments():
    single = AutocallableBacktestResults(config=None, **EMPTY)
    book = BookBacktestResults(config=None, products_meta=[], **EMPTY)
    assert single.hedge_legs_df.empty
    assert book.hedge_stresses_df().empty
    # The legacy frames are still derived from their rows, not fixed.
    assert list(single.states_df.columns) == []
    assert single.get_total_pnl() == 0.0


def test_the_legacy_frame_helper_is_unchanged_for_the_old_frames():
    rows = [{"date": datetime(2025, 3, 3), "total_pnl": 1.0}]
    single = AutocallableBacktestResults(
        config=None, **{**EMPTY, "states": rows}
    )
    frame = single.states_df
    assert list(frame.columns) == ["total_pnl"]
    assert frame.index.name == "date"


def test_the_single_wrapper_forwards_all_three_lists():
    import inspect

    from quantark.backtest.replay import single

    source = inspect.getsource(single.AutocallableBacktestEngine.run)
    for name in ("hedge_legs", "hedge_attribution", "hedge_stresses"):
        assert f"{name}=" in source, name


def test_the_engine_owns_the_three_row_lists():
    import inspect

    from quantark.backtest.replay import engine

    source = inspect.getsource(engine.ReplayBacktestEngine)
    for name in ("hedge_legs", "hedge_attribution", "hedge_stresses"):
        assert f"self._{name}: list" in source, name
        assert f"{name}=self._{name}" in source, name
