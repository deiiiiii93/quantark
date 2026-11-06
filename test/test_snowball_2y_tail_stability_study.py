"""Tests for the 2Y snowball tail-stability study (example/case_studies/snowball_2y_tail_stability)."""
from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

STUDY_DIR = Path(__file__).resolve().parents[1] / "example" / "case_studies" / "snowball_2y_tail_stability"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from quantark.util.calendar import CalendarType, create_calendar  # noqa: E402

from s2ts import HISTORY_DIR, StudyDataError  # noqa: E402
from s2ts.chain import (  # noqa: E402
    build_all_chains,
    build_day_chain,
    contract_expiry,
    load_day_rows,
    parse_cffex_csv,
)

HEADER = "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,今收盘,今结算,前结算,涨跌1,涨跌2,Delta\n"


def sse_calendar(y0: int, y1: int) -> pd.DatetimeIndex:
    cal = create_calendar(CalendarType.CHINA_SSE)
    days = pd.date_range(f"{y0}-01-01", f"{y1}-12-31")
    return pd.DatetimeIndex([d for d in days if cal.is_business_day(datetime(d.year, d.month, d.day))])


def csv_bytes(rows: dict) -> bytes:
    """rows: code -> (close, settle) as strings or numbers."""
    lines = [HEADER] + [f"{c},null,null,null,0,0,1,0,{cl},{st},0,0,0,0\n" for c, (cl, st) in rows.items()]
    return "".join(lines).encode("gbk")


def synth_rows(day_im=("IM2611", "IM2612"), mo_series="MO2709", strikes=(6400, 6500, 6600),
               cs=None, ps=None) -> dict:
    parsed = {c: {"close": 6500.0, "settle": 6500.0} for c in day_im}
    cs = cs or {6400: 680.0, 6500: 610.0, 6600: 545.0}
    ps = ps or {6400: 530.0, 6500: 570.0, 6600: 610.0}
    for k in strikes:
        if cs.get(k) is not None:
            parsed[f"{mo_series}-C-{k}"] = {"close": cs[k], "settle": cs[k]}
        if ps.get(k) is not None:
            parsed[f"{mo_series}-P-{k}"] = {"close": ps[k], "settle": ps[k]}
    return parsed


def test_parse_cffex_skips_non_numeric_rows():
    payload = csv_bytes({"IM2611": (6500.2, 6499.8), "MO2709-C-6600": ("--", "--")})
    assert parse_cffex_csv(payload) == {"IM2611": {"close": 6500.2, "settle": 6499.8}}


def test_parse_cffex_bad_header_raises():
    with pytest.raises(StudyDataError):
        parse_cffex_csv("a,b,c\n1,2,3\n".encode("gbk"))


def test_contract_expiry_third_friday():
    cal = sse_calendar(2026, 2028)
    assert contract_expiry("MO2709", cal) == pd.Timestamp("2027-09-17")
    assert contract_expiry("IM2706", cal) == pd.Timestamp("2027-06-18")


def test_contract_expiry_bad_code():
    with pytest.raises(StudyDataError):
        contract_expiry("XX2706", sse_calendar(2026, 2028))


def test_atm_ignores_strike_with_one_missing_leg():
    cal = sse_calendar(2026, 2027)
    # strike 6500 would minimise |C-P| (40) but its put is missing -> 6600 (|545-610|=65) beats 6400 (150)
    rows = synth_rows(ps={6400: 530.0, 6500: None, 6600: 610.0})
    dc = build_day_chain(date(2026, 10, 21), 6500.0, rows, cal)
    assert dc.mo.strike == 6600
    assert 6500 not in dc.mo_all


def test_atm_no_complete_strike_raises():
    cal = sse_calendar(2026, 2027)
    rows = synth_rows(ps={6400: None, 6500: None, 6600: None})
    with pytest.raises(StudyDataError):
        build_day_chain(date(2026, 10, 21), 6500.0, rows, cal)


def test_im_under_7_days_excluded():
    cal = sse_calendar(2026, 2027)
    # IM2611 expires 2026-11-20; on 2026-11-16 it has 4 days left
    rows = synth_rows(day_im=("IM2611", "IM2612", "IM2703"))
    dc = build_day_chain(date(2026, 11, 16), 6500.0, rows, cal)
    assert [q.code for q in dc.im] == ["IM2612", "IM2703"]
    assert all(q.tenor * 365 >= 7 - 1e-9 for q in dc.im)


def test_missing_day_fails_closed(tmp_path):
    hist, ext = tmp_path / "h", tmp_path / "e"
    (hist / "settlement_csv").mkdir(parents=True)
    ext.mkdir()
    with pytest.raises(StudyDataError) as e:
        load_day_rows(date(2026, 10, 22), hist, ext)
    assert "2026-10-22" in str(e.value) and "01_data.py" in str(e.value)


def test_loader_reads_shared_cache_then_extension(tmp_path):
    hist, ext = tmp_path / "h", tmp_path / "e"
    (hist / "settlement_csv").mkdir(parents=True)
    ext.mkdir()
    (hist / "settlement_csv" / "20261021_1.csv").write_bytes(csv_bytes({"IM2611": (1.0, 2.0)}))
    (ext / "20261022_1.csv").write_bytes(csv_bytes({"IM2611": (3.0, 4.0)}))
    before = sorted(p.name for p in hist.rglob("*"))
    assert load_day_rows(date(2026, 10, 21), hist, ext)["IM2611"]["close"] == 1.0
    assert load_day_rows(date(2026, 10, 22), hist, ext)["IM2611"]["close"] == 3.0
    assert sorted(p.name for p in hist.rglob("*")) == before


def test_build_all_chains_gap_in_spot_raises(tmp_path):
    """A trading day absent from the spot series would fuse two days of P&L."""
    hist, ext = tmp_path / "h", tmp_path / "e"
    (hist / "settlement_csv").mkdir(parents=True)
    ext.mkdir()
    (hist / "csi1000_spot.csv").write_text("date,spot\n2026-10-19,6500\n2026-10-21,6510\n")
    for d in ("20261019", "20261021"):
        (hist / "settlement_csv" / f"{d}_1.csv").write_bytes(csv_bytes({"IM2611": (1.0, 2.0)}))
    with pytest.raises(StudyDataError) as e:
        build_all_chains(hist, ext, sse_calendar(2026, 2028))
    assert "2026-10-20" in str(e.value)


_REAL = Path(HISTORY_DIR) / "settlement_csv" / "20261021_1.csv"


@pytest.mark.skipif(not _REAL.exists(), reason="shared mo_volmodels history cache absent")
def test_day_chain_on_cached_2026_10_21():
    cal = sse_calendar(2026, 2028)
    rows = parse_cffex_csv(_REAL.read_bytes())
    spots = dict(l.split(",") for l in (Path(HISTORY_DIR) / "csi1000_spot.csv").read_text().splitlines()[1:])
    spot = float(spots["2026-10-21"])
    dc = build_day_chain(date(2026, 10, 21), spot, rows, cal)
    assert [q.code for q in dc.im] == ["IM2611", "IM2612", "IM2703", "IM2706"]
    assert dc.mo.series == "MO2709"
    k = dc.mo.strike
    c, p = dc.mo_all[k]
    assert abs(c - p) == min(abs(a - b) for a, b in dc.mo_all.values())
    assert dc.mo.forward == pytest.approx(k + (c - p) / math.exp(-0.014 * dc.mo.tenor), rel=1e-12)
    assert len(dc.mo.strikes_near) == 5
    assert dc.mo.vol > 0
    far = dc.im[-1]
    assert dc.far_carry == pytest.approx(math.log(far.close / spot) / far.tenor, rel=1e-12)
