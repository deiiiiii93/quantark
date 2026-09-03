"""Row schema, additivity contract, frame schema (spec §5.4)."""
import math
from datetime import datetime

import pytest

from quantark.pnlexplain.base import (
    FRAME_COLUMNS, ExplainMethod, ExplainRow, Factor, RowKind,
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
    vb = ValueBreakdown(10.0, 0.0, 0.0)                 # finite endpoints: the RESULT must refuse
    with pytest.raises(NumericalError):
        PnLExplainResult(date_t0=datetime(2026, 6, 26), date_t1=datetime(2026, 6, 29), pv_t0=vb,
                         pv_alive_t1=vb, pv_t1=vb, total_pnl=math.inf, moves=None, rows=(),
                         unexplained=None, metadata={})


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


def test_row_and_result_normalise_numeric_fields():
    from decimal import Decimal
    r = _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, Decimal("1.5"),
             greek="0.25", cash_greek=Decimal("2.5"), moves={"vol_pts": "1.0"})
    assert (r.pnl, r.greek, r.cash_greek, r.moves["vol_pts"]) == (1.5, 0.25, 2.5, 1.0)
    assert all(type(v) is float for v in (r.pnl, r.greek, r.cash_greek, r.moves["vol_pts"]))
    with pytest.raises(ValidationError, match="greek"):
        _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, greek="big")
    with pytest.raises(NumericalError, match="cash_greek"):
        _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, cash_greek=math.inf)
    with pytest.raises(NumericalError, match="move vol_pts"):
        _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, moves={"vol_pts": math.nan})
    with pytest.raises(ValidationError, match="move vol_pts"):
        _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, moves={"vol_pts": "x"})
    with pytest.raises(ValidationError, match="PnL"):
        _row(Factor.VOL, "vega", ExplainMethod.TAYLOR, RowKind.COMPONENT, "1.0.0")
    vb = ValueBreakdown(10.0, 0.0, 0.0)
    kw = dict(date_t0=datetime(2026, 6, 26), date_t1=datetime(2026, 6, 29), pv_t0=vb, pv_alive_t1=vb,
              pv_t1=vb, moves=None, rows=(), metadata={})
    res = PnLExplainResult(total_pnl=Decimal("3.5"), unexplained=Decimal("0.5"), **kw)
    assert type(res.total_pnl) is float and res.total_pnl == 3.5 and res.unexplained == 0.5
    assert res.reconcile(ExplainMethod.TAYLOR) == 3.5                    # float arithmetic, no Decimal leak
    with pytest.raises(NumericalError, match="unexplained"):
        PnLExplainResult(total_pnl=3.5, unexplained=math.inf, **kw)
    with pytest.raises(ValidationError, match="total_pnl"):
        PnLExplainResult(total_pnl="lots", unexplained=None, **kw)


def test_value_breakdown_and_by_factor_use_checked_exact_sums():
    with pytest.raises(ValidationError, match="contingent_mtm"):
        ValueBreakdown("x", 0.0, 0.0)
    with pytest.raises(NumericalError):                       # int too large for a float
        ValueBreakdown(10 ** 400, 0.0, 0.0)
    with pytest.raises(NumericalError, match="total"):        # finite parts, overflowing total
        ValueBreakdown(1e308, 1e308, 0.0)
    assert ValueBreakdown(1e16, 1.0, -1e16).total == 1.0      # exactly rounded, not left-associative
    rows = [
        _row(Factor.SPOT, "delta", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1e16),
        _row(Factor.SPOT, "gamma", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0),
        _row(Factor.SPOT, "speed", ExplainMethod.TAYLOR, RowKind.COMPONENT, -1e16),
    ]
    res = _result(rows, 1.0)
    assert res.by_factor(ExplainMethod.TAYLOR) == {"spot": 1.0}
    assert res.reconcile(ExplainMethod.TAYLOR) == 0.0
    with pytest.raises(ValidationError, match="moves"):
        _row(Factor.SPOT, "delta", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, moves=None)
    with pytest.raises(ValidationError, match="moves"):
        _row(Factor.SPOT, "delta", ExplainMethod.TAYLOR, RowKind.COMPONENT, 1.0, moves=0)


def test_reconcile_scale_tracks_endpoints_and_components():
    from quantark.pnlexplain.equity.explain import reconcile_scale
    rows = [
        _row(Factor.TIME, "time", ExplainMethod.WATERFALL, RowKind.COMPONENT, 1e9),
        _row(Factor.SPOT, "spot", ExplainMethod.WATERFALL, RowKind.COMPONENT, -1e9 + 3.5),
        _row(Factor.TIME, "r_theta", ExplainMethod.TAYLOR, RowKind.INFORMATIONAL, 1e15),   # not summed
        make_total_row("instrument", 3.5),
    ]
    res = _result(rows, 3.5)
    assert res.total_pnl == 3.5
    assert reconcile_scale(res, ExplainMethod.WATERFALL) == pytest.approx(2e9 - 3.5)
    assert reconcile_scale(res, ExplainMethod.TAYLOR) == pytest.approx(13.5)              # pv_alive_t1 = 13.5
    big = ValueBreakdown(1e12, 0.0, 0.0)
    res_big = PnLExplainResult(date_t0=datetime(2026, 6, 26), date_t1=datetime(2026, 6, 29), pv_t0=big,
                               pv_alive_t1=big, pv_t1=big, total_pnl=0.0, moves=None, rows=(),
                               unexplained=None, metadata={})
    assert reconcile_scale(res_big, ExplainMethod.WATERFALL) == 1e12
    assert component_sum(rows, ExplainMethod.WATERFALL) == 3.5                            # exactly rounded (fsum)
