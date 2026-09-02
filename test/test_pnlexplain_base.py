"""Row schema, additivity contract, frame schema (spec §5.4)."""
import math
from datetime import datetime

import pandas as pd
import pytest

from quantark.pnlexplain.base import (
    FRAME_COLUMNS, MOVE_KEYS, ExplainMethod, ExplainRow, Factor, RowKind,
    ValueBreakdown, PnLExplainResult, component_sum, make_total_row, rows_to_frame,
)
from quantark.util.exceptions import NumericalError, ValidationError


def _row(factor, term, method, kind, pnl, level="instrument", **kw):
    return ExplainRow(factor=factor, term=term, method=method, kind=kind, level=level,
                      pnl=pnl, **kw)


def _result(rows, total):
    vb0 = ValueBreakdown(10.0, 0.0, 0.0)
    vb1 = ValueBreakdown(10.0 + total, 0.0, 0.0)
    return PnLExplainResult(
        date_t0=datetime(2026, 6, 26), date_t1=datetime(2026, 6, 29),
        pv_t0=vb0, pv_alive_t1=vb1, pv_t1=vb1, total_pnl=total, moves=None,
        rows=tuple(rows), unexplained=None, metadata={},
    )


def test_value_breakdown_total_is_sum():
    vb = ValueBreakdown(contingent_mtm=1.0, pending_receivable_pv=2.0, paid_cash=3.0)
    assert vb.total == 6.0


def test_row_rejects_unknown_level_and_move_key_and_nonfinite():
    with pytest.raises(ValidationError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0, level="book")
    with pytest.raises(ValidationError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0,
             moves={"bogus": 1.0})
    with pytest.raises(NumericalError):
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, math.nan)


def test_component_sum_excludes_informational_and_summary_and_other_method():
    rows = [
        _row(Factor.TIME, "time", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1.0),
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 2.0),
        _row(Factor.LIFECYCLE_EVENT, "lifecycle_event", ExplainMethod.SHARED, RowKind.COMPONENT, 0.5),
        _row(Factor.TIME, "theta", ExplainMethod.TAYLOR, RowKind.COMPONENT, 0.9),
        _row(Factor.TIME, "r_theta", ExplainMethod.TAYLOR, RowKind.INFORMATIONAL, 0.4),
        _row(Factor.UNEXPLAINED, "unexplained", ExplainMethod.TAYLOR, RowKind.COMPONENT, 2.1),
        make_total_row("instrument", 3.5),
    ]
    assert component_sum(rows, ExplainMethod.WATERFALL) == pytest.approx(3.5)
    assert component_sum(rows, ExplainMethod.TAYLOR) == pytest.approx(3.5)
    # Summing every row is NOT the total (the additivity contract is per method).
    assert sum(r.pnl for r in rows) != pytest.approx(3.5)
    res = _result(rows, 3.5)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=1e-12)
    assert res.by_factor(ExplainMethod.WATERFALL) == {
        "time": 1.0, "spot": 2.0, "lifecycle_event": 0.5}
    assert [r.term for r in res.rows_for(ExplainMethod.TAYLOR, kind=RowKind.INFORMATIONAL)] == ["r_theta"]


def test_value_breakdown_and_result_reject_non_finite():
    with pytest.raises(NumericalError):
        ValueBreakdown(math.nan, 0.0, 0.0)
    with pytest.raises(NumericalError):
        _result([], math.inf)


def test_frame_schema_and_empty_frame():
    rows = [
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, 2.0,
             moves={"spot_return": 0.01}, step=2),
    ]
    df = rows_to_frame(rows, date=datetime(2026, 6, 29), position_id="p1", underlying="X")
    assert list(df.columns) == ["date", *FRAME_COLUMNS]
    assert df.loc[0, "factor"] == "spot" and df.loc[0, "method"] == "waterfall"
    assert df.loc[0, "spot_return"] == pytest.approx(0.01)
    assert math.isnan(df.loc[0, "vol_pts"])
    assert str(df["date"].dtype) == "datetime64[ns]"
    empty = rows_to_frame([])
    assert list(empty.columns) == FRAME_COLUMNS and len(empty) == 0


def test_relabel_and_to_dict_keep_order():
    r = _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.5,
             greek=3.0, cash_greek=0.03, moves={"vol_pts": 0.5})
    p = r.relabel("position")
    assert p.level == "position" and p.pnl == 1.5 and r.level == "instrument"
    d = p.to_dict()
    assert list(d.keys()) == FRAME_COLUMNS
    assert d["term"] == "vega" and d["kind"] == "component"
