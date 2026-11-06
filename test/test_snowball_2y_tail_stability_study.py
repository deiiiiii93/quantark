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

from s2ts import HISTORY_DIR, RATE, StudyDataError  # noqa: E402
from s2ts.chain import (  # noqa: E402
    DayChain,
    ImQuote,
    MoPillar,
    build_all_chains,
    build_day_chain,
    contract_expiry,
    load_day_rows,
    parse_cffex_csv,
)
from s2ts.stability import (  # noqa: E402
    day_types,
    multi_strike_forward,
    q2y,
    settlement_chain,
    summarize,
    switch_floor,
)
from s2ts.tails import (  # noqa: E402
    TAILS,
    ConvergingCarryCurve,
    anchor_series,
    build_curve,
    pillar_inputs,
)
from s2ts.books import (  # noqa: E402
    BookA,
    BookB,
    Trade,
    offchain_marks,
    total_cost,
    turnover,
)
from s2ts.measures import (  # noqa: E402
    basis_mitigation,
    holm,
    paired,
    paired_complete,
    per_run,
    terminal_paired,
    verdict,
)
from s2ts.runner import (  # noqa: E402
    OutDirLocked,
    RssGuard,
    RunsFailed,
    coupons_for,
    error_path,
    parity_flag,
    run_all,
    run_one,
    run_path,
)
from s2ts.note import (  # noqa: E402
    DayGreeks,
    NoteLife,
    build_note,
    build_terms,
    daily_greeks,
    inception_dates,
    make_env,
    solve_coupon,
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


# ---------------------------------------------------------------------------
# Task 2: tails, anchor, q(2Y) stability
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402
import importlib.util  # noqa: E402

import numpy as np  # noqa: E402

from quantark.param.curve.carry_curve import CarryCurve  # noqa: E402
from quantark.param.div.dividend_yield import ParallelShiftDividendYield  # noqa: E402
from quantark.param.extrapolation import CarryExtrapolation  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

NODES = [(0.1, -0.004), (0.5, -0.03), (0.9, -0.06)]
TN, BN = NODES[-1]


def test_tails_ids():
    assert TAILS == ("hold_forward", "hold_yield", "zero_carry",
                     "converge_hl025", "converge_hl050", "converge_hl100")


def test_converging_matches_base_inside_last_node():
    conv = ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.5)
    plain = CarryCurve(NODES)
    ts = np.linspace(0.0, TN, 61)
    np.testing.assert_allclose(conv.carry(ts), plain.carry(ts), rtol=1e-14, atol=0.0)
    for t in (0.0, 0.05, 0.3, 0.7, TN):
        assert conv.carry(t) == pytest.approx(plain.carry(t), rel=1e-14, abs=0.0)
        assert isinstance(conv.carry(t), float)
    inner = np.linspace(0.0, TN, 61)[:-1]
    np.testing.assert_allclose(conv.forward_carry(inner), plain.forward_carry(inner), rtol=1e-14, atol=0.0)


def test_converging_continuous_at_last_node():
    conv = ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.5)
    # continuous, not flat: |B(T_n ± h) − B_n| shrinks like h·|slope| (slopes ~0.07/yr)
    lo, hi = conv.carry(TN - 1e-12), conv.carry(TN + 1e-12)
    assert abs(hi - lo) < 1e-12
    for h in (1e-3, 1e-6, 1e-9):
        assert abs(conv.carry(TN + h) - BN) <= 0.1 * h
        assert abs(conv.carry(TN - h) - BN) <= 0.1 * h
    assert abs(conv.carry(TN) - BN) < 1e-15
    # right-continuous forward starts at the last zero carry LEVEL c_n, not the last slope
    assert conv.forward_carry(TN) == pytest.approx(BN / TN, rel=1e-14)


def test_tiny_kappa_reproduces_hold_yield():
    conv = ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=1e9)
    hy = CarryCurve(NODES, extrapolation=CarryExtrapolation.FLAT_CARRY_YIELD)
    assert abs(conv.carry(2.0) - hy.carry(2.0)) < 1e-9
    ts = np.array([1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(conv.carry(ts), hy.carry(ts), rtol=0.0, atol=1e-9)


def test_huge_kappa_forward_carry_is_anchor():
    conv = ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=1e-6)
    assert abs(conv.forward_carry(TN + 0.1) - (-0.05)) < 1e-9
    # B keeps only the vanishing (c_n − b_inf)/κ memory of the start level
    k = math.log(2.0) / 1e-6
    assert conv.carry(2.0) - (BN - 0.05 * (2.0 - TN)) == pytest.approx((BN / TN + 0.05) / k, rel=1e-6)


def test_closed_form_value():
    nodes = [(0.5, -0.03), (0.9, -0.06)]
    b_inf, hl, t = -0.05, 0.5, 2.0
    k = math.log(2.0) / hl
    tau = t - 0.9
    cn = -0.06 / 0.9
    expected = -0.06 + b_inf * tau + (cn - b_inf) * (1.0 - math.exp(-k * tau)) / k
    conv = ConvergingCarryCurve(nodes, b_inf=b_inf, half_life=hl)
    assert conv.carry(t) == pytest.approx(expected, rel=1e-14)
    arr = conv.carry(np.array([0.3, t]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    assert arr[1] == pytest.approx(expected, rel=1e-14)
    fwd = b_inf + (cn - b_inf) * math.exp(-k * tau)
    assert conv.forward_carry(t) == pytest.approx(fwd, rel=1e-14)
    assert conv.forward_carry(np.array([t]))[0] == pytest.approx(fwd, rel=1e-14)
    # interval carry: chord of the closed form; zero-length interval = forward carry
    b1 = -0.06 + b_inf * 0.1 + (cn - b_inf) * (1.0 - math.exp(-k * 0.1)) / k
    assert conv.interval_carry(1.0, t) == pytest.approx((expected - b1) / 1.0, rel=1e-12)
    assert conv.interval_carry(t, t) == pytest.approx(fwd, rel=1e-14)
    assert conv.forward(6500.0, t) == pytest.approx(6500.0 * math.exp(expected), rel=1e-14)


def test_with_nodes_keeps_type_and_params():
    conv = ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.25)
    moved = conv.with_nodes([(t, b + 0.001) for t, b in NODES])
    assert type(moved) is ConvergingCarryCurve
    assert (moved.b_inf, moved.half_life) == (-0.05, 0.25)
    assert moved.extrapolation is CarryExtrapolation.ZERO_FORWARD_CARRY
    assert moved.interpolation is conv.interpolation
    assert moved.carry(TN) == pytest.approx(BN + 0.001, rel=1e-14)
    # the base is ZERO_FORWARD_CARRY, so a parallel q shift takes the exact wrapper path
    view = conv.to_dividend_yield(_flat_rate())
    assert isinstance(view.parallel_shifted(1e-3), ParallelShiftDividendYield)


def test_converging_validates_inputs():
    with pytest.raises(ValidationError):
        ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.0)
    with pytest.raises(ValidationError):
        ConvergingCarryCurve(NODES, b_inf=float("nan"), half_life=0.5)
    with pytest.raises(ValidationError):
        ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.5,
                             extrapolation=CarryExtrapolation.FLAT_FORWARD_CARRY)
    with pytest.raises(ValidationError):
        ConvergingCarryCurve(NODES, b_inf=-0.05, half_life=0.5).carry(-0.1)


def _flat_rate():
    from quantark.param import FlatRateCurve

    return FlatRateCurve(RATE)


# ---- synthetic chains -------------------------------------------------------

def synth_chain(day: date, spot=6500.0, far_close=6300.0, mo_strike=6400.0, mo_series="MO2709",
                mo_tenor=0.86, mo_all=None) -> DayChain:
    im = (ImQuote("IM2612", date(2026, 12, 18), 0.115, 6460.0, 6465.0),
          ImQuote("IM2706", date(2027, 6, 18), 0.6137, far_close, far_close + 6.0))
    mo_all = mo_all or {6200.0: (650.0, 520.0), 6400.0: (560.0, 610.0), 6600.0: (480.0, 720.0),
                        6800.0: (410.0, 840.0), 7000.0: (350.0, 980.0)}
    c, p = mo_all[mo_strike]
    df = math.exp(-RATE * mo_tenor)
    near = tuple((k, mo_all[k][0], mo_all[k][1]) for k in sorted(mo_all))
    mo = MoPillar(mo_series, date(2027, 9, 17), mo_tenor, mo_strike, c, p, mo_strike + (c - p) / df,
                  0.25, near)
    return DayChain(day, spot, im, mo, dict(mo_all), math.log(far_close / spot) / 0.6137)


def _days(n):
    return list(pd.bdate_range("2024-01-02", periods=n).date)


def test_build_curve_tails_and_pillar_inputs():
    dc = synth_chain(date(2026, 11, 5))
    hf = build_curve(dc, "hold_forward", None)
    assert hf.extrapolation is CarryExtrapolation.FLAT_FORWARD_CARRY
    assert build_curve(dc, "hold_yield", None).extrapolation is CarryExtrapolation.FLAT_CARRY_YIELD
    assert build_curve(dc, "zero_carry", None).extrapolation is CarryExtrapolation.ZERO_FORWARD_CARRY
    conv = build_curve(dc, "converge_hl050", -0.05)
    assert isinstance(conv, ConvergingCarryCurve) and conv.half_life == 0.5 and conv.b_inf == -0.05
    expect = [(q.tenor, math.log(q.close / dc.spot)) for q in dc.im] + [
        (dc.mo.tenor, math.log(dc.mo.forward / dc.spot))]
    assert hf.nodes == pytest.approx(expect, rel=1e-14)
    assert build_curve(dc, "hold_forward", None, mo_forward=6500.0).nodes[-1][1] == pytest.approx(
        math.log(6500.0 / dc.spot), rel=1e-14)
    with pytest.raises(ValidationError):
        build_curve(dc, "converge_hl025", None)
    with pytest.raises(ValidationError):
        build_curve(dc, "nope", None)
    labels, mults = pillar_inputs(dc)
    assert labels == ["IM2612", "IM2706", "MO2709 C-P"]
    assert mults == pytest.approx([200.0, 200.0, 100.0 * math.exp(-RATE * dc.mo.tenor)], rel=1e-14)
    assert q2y(hf) == pytest.approx(RATE - hf.carry(2.0) / 2.0, rel=1e-14)


def test_multi_strike_forward_and_settlement_chain():
    dc = synth_chain(date(2026, 11, 5))
    df = math.exp(-RATE * dc.mo.tenor)
    expect = np.mean([k + (c - p) / df for k, c, p in dc.mo.strikes_near])
    assert multi_strike_forward(dc) == pytest.approx(expect, rel=1e-14)
    ks = (6400.0, 6600.0)
    assert multi_strike_forward(dc, ks) == pytest.approx(
        np.mean([k + (dc.mo_all[k][0] - dc.mo_all[k][1]) / df for k in ks]), rel=1e-14)
    st = settlement_chain(dc)
    assert [q.close for q in st.im] == [q.settle for q in dc.im]
    assert st.mo == dc.mo and st.spot == dc.spot
    assert st.far_carry == pytest.approx(math.log(dc.im[-1].settle / dc.spot) / dc.im[-1].tenor, rel=1e-14)


def test_anchor_point_in_time():
    days = _days(80)
    chains = [synth_chain(d, far_close=6300.0 - i) for i, d in enumerate(days)]
    anchors = anchor_series(chains, 63)
    assert min(anchors) == days[62] and days[61] not in anchors
    d = days[70]
    assert anchors[d] == pytest.approx(np.mean([c.far_carry for c in chains[8:71]]), rel=1e-14)
    mutated = chains[:71] + [dataclasses.replace(c, far_carry=c.far_carry + 1.0) for c in chains[71:]]
    again = anchor_series(mutated, 63)
    for day in days[62:71]:
        assert again[day] == anchors[day]
    assert again[days[75]] != anchors[days[75]]


def test_day_types_and_switch_floor_zero_when_no_switch():
    days = _days(4)
    chains = [
        synth_chain(days[0], mo_strike=6400.0),
        synth_chain(days[1], mo_strike=6400.0),
        synth_chain(days[2], mo_strike=6600.0),                       # ATM switch
        synth_chain(days[3], mo_strike=6400.0, mo_series="MO2712"),   # roll (and a switch): roll wins
    ]
    types = day_types(chains)
    assert types == {days[1]: "ordinary", days[2]: "atm_switch", days[3]: "mo_roll"}
    assert switch_floor(chains[0], chains[1], "hold_forward", None) == 0.0
    assert switch_floor(chains[0], chains[1], "converge_hl050", -0.05, multi_strike=True) == 0.0
    # on a switch day the floor is q2y at yesterday's strike minus q2y at today's ATM
    prev, dc = chains[1], chains[2]
    df = math.exp(-RATE * dc.mo.tenor)
    c, p = dc.mo_all[prev.mo.strike]
    f_prev = prev.mo.strike + (c - p) / df
    expect = q2y(build_curve(dc, "hold_yield", None, mo_forward=f_prev)) - q2y(build_curve(dc, "hold_yield", None))
    assert switch_floor(prev, dc, "hold_yield", None) == pytest.approx(expect, rel=1e-12)
    assert expect != 0.0
    with pytest.raises(ValidationError):
        switch_floor(chains[2], chains[3], "hold_forward", None)   # across an MO roll


def test_summarize_in_bp_by_day_type():
    days = _days(6)
    series = {d: 0.05 + 1e-4 * x for d, x in zip(days, [0.0, 1.0, 3.0, 2.0, 2.0, 6.0])}
    types = {days[1]: "ordinary", days[2]: "atm_switch", days[3]: "ordinary",
             days[4]: "mo_roll", days[5]: "atm_switch"}
    s = summarize(series, types)
    deltas = np.array([1.0, 2.0, -1.0, 0.0, 4.0])  # bp of yield
    assert s["all"]["n"] == 5
    assert s["all"]["std"] == pytest.approx(np.std(deltas, ddof=1), rel=1e-9)
    assert s["all"]["mad"] == pytest.approx(np.median(np.abs(deltas - np.median(deltas))), rel=1e-9)
    assert s["all"]["p99_abs"] == pytest.approx(np.percentile(np.abs(deltas), 99), rel=1e-9)
    assert s["atm_switch"]["n"] == 2 and s["ordinary"]["n"] == 2 and s["mo_roll"]["n"] == 1
    assert s["atm_switch"]["std"] == pytest.approx(np.std([2.0, 4.0], ddof=1), rel=1e-9)
    assert s["mo_roll"]["std"] is None


# ---- bucket identity on the 2Y study snapshot (V2) ----------------------------

_CC_DIR = Path(__file__).resolve().parents[1] / "example" / "case_studies" / "snowball_2y_carry_curve"


def _cc_common():
    name = "snowball_2y_cc_common_for_s2ts_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _CC_DIR / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_bucket_report_runs_and_delta_identity_holds():
    from quantark.asset.equity.engine.quad import SnowballQuadEngineV2
    from quantark.asset.equity.param import QuadV2Params
    from quantark.asset.equity.riskmeasures import calculate_carry_curve_buckets

    c = _cc_common()
    snap = c.load_snapshot()
    pricing = c.read_json(c.RESULTS_DIR / "pricing.json")
    s0 = float(snap["spot"])
    base = c.build_curve(snap, "hold_forward")
    far = snap["futures"][-1]
    b_inf = math.log(far["price"] / s0) / c.year_fraction(date.fromisoformat(far["expiry"]))
    curve = ConvergingCarryCurve(base.nodes, b_inf=b_inf, half_life=0.5)
    product = c.build_product(pricing["coupon"]["value"], s0, c.build_terms())
    env = c.make_env(snap, curve.to_dividend_yield(c.rate_curve()))
    rep = calculate_carry_curve_buckets(
        product, env, SnowballQuadEngineV2(params=QuadV2Params()), curve,
        labels=c.pillar_labels(snap), lot_multipliers=c.lot_multipliers(snap),
        spot_bump_rel=1e-3, forward_bump_rel=1e-3,
    )
    identity = rep.pinned_forward_delta + sum(b.forward / s0 * b.delta for b in rep.buckets)
    assert abs(rep.spot_delta - identity) < 0.5
    assert abs(rep.pinned_forward_delta) > 1.0
    # the converging tail reads only the last node (its level), never the last slope
    assert [b.extrapolated_tail for b in rep.buckets] == [False] * (len(rep.buckets) - 1) + [True]


def _stage(name: str, filename: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STUDY_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_stage02_variant_is_paired_on_the_anchor_window():
    st = _stage("s2ts_stage02_stability", "02_stability.py")
    days = _days(70)
    strikes = [6400.0] * 65 + [6600.0, 6600.0, 6400.0, 6400.0, 6400.0]   # switches into days 65 and 67
    chains = [synth_chain(d, far_close=6300.0 - 0.5 * i, mo_strike=k) for i, (d, k) in enumerate(zip(days, strikes))]
    types = day_types(chains)
    anchors = anchor_series(chains, 63)
    v = st.run_variant(chains, types, anchors, multi_strike=False)
    assert v["n_days"] == 8 and v["dates"][0] == days[62].isoformat()
    assert set(v["tails"]) == set(TAILS)
    for t in TAILS:
        assert len(v["tails"][t]["q2y"]) == 8
        assert v["tails"][t]["summary"]["all"]["n"] == 7
        assert v["tails"][t]["summary"]["atm_switch"]["n"] == 2
        assert v["tails"][t]["switch_floor"]["n"] == 2
    assert set(st.VARIANTS) == {"base", "multi_strike", "all_settlement", "anchor_252"}


_STABILITY_JSON = STUDY_DIR / "results" / "stability.json"


@pytest.mark.skipif(not _STABILITY_JSON.exists(), reason="run 02_stability.py first")
def test_stability_json_has_six_tails_by_four_variants():
    import json

    d = json.loads(_STABILITY_JSON.read_text(encoding="utf-8"))
    assert set(d["variants"]) == {"base", "multi_strike", "all_settlement", "anchor_252"}
    for v in d["variants"].values():
        assert set(v["tails"]) == set(TAILS)
        for t in v["tails"].values():
            assert t["summary"]["all"]["std"] > 0 and t["switch_floor"]["n"] > 0
            assert len(t["q2y"]) == v["n_days"]


# ---------------------------------------------------------------------------
# Task 3: the note — terms, coupon, lifecycle, daily greeks
# ---------------------------------------------------------------------------

from s2ts import NOTIONAL  # noqa: E402

S0 = 6500.0
INCEPTION = date(2024, 1, 2)


def _v2():
    from quantark.asset.equity.engine.quad import SnowballQuadEngineV2
    from quantark.asset.equity.param import QuadV2Params

    return SnowballQuadEngineV2(params=QuadV2Params())


def _sse_days(start: date, end: date) -> list:
    cal = create_calendar(CalendarType.CHINA_SSE)
    return [d.date() for d in pd.date_range(start, end)
            if cal.is_business_day(datetime(d.year, d.month, d.day))]


def _env(dc):
    return make_env(dc, build_curve(dc, "hold_forward", None).to_dividend_yield(_flat_rate()))


def _run(path, coupon=0.2, inception=INCEPTION, stop_when_dead=True):
    """Step a NoteLife over ``path`` = [(day, spot)]; returns (life, terms, [(day, NoteDay)])."""
    terms = build_terms(inception)
    life = NoteLife(build_note(coupon, S0, terms), inception)
    out = []
    for day, spot in path:
        dc = synth_chain(day, spot=spot)
        nd = life.step(dc, _env(dc))
        out.append((day, nd))
        if stop_when_dead and not nd.alive:
            break
    return life, terms, out


def test_build_terms_rolls_to_sse_days():
    terms = build_terms(date(2023, 10, 9))
    sse = set(_sse_days(date(2023, 10, 1), date(2025, 12, 31)))
    assert terms.ko_dates[0] == date(2024, 1, 9)            # month 3 = 2024-01-09 (a trading day)
    assert all(d in sse for d in terms.ko_dates) and terms.ko_dates[-1] == terms.maturity_date
    assert len(terms.ko_dates) == 22
    assert terms.maturity_date == date(2025, 10, 9)
    assert terms.ko_times[0] == pytest.approx((date(2024, 1, 9) - date(2023, 10, 9)).days / 365.0, rel=1e-15)
    assert terms.ki_times[0] == pytest.approx(1 / 365.0, rel=1e-15)  # 2023-10-10, first SSE day after
    # a month-3 anniversary on a holiday rolls forward: 2024-02-10 (Spring Festival) -> 2024-02-19
    assert build_terms(date(2023, 11, 10)).ko_dates[0] == date(2024, 2, 19)


def test_coupon_solves_par():
    dc = synth_chain(INCEPTION, spot=S0)
    engine = _v2()
    coupon = solve_coupon(dc, engine)
    assert 0.0 < coupon < 1.0
    pv = engine.price(build_note(coupon, S0, build_terms(INCEPTION)), _env(dc))
    assert abs(pv) <= 1e-6 * NOTIONAL


def test_day0_aged_pv_equals_fresh_pv():
    dc = synth_chain(INCEPTION, spot=S0)
    terms = build_terms(INCEPTION)
    fresh = build_note(0.2, S0, terms)
    life = NoteLife(build_note(0.2, S0, terms), INCEPTION)
    nd = life.step(dc, _env(dc))
    assert nd.alive and nd.cash == 0.0 and not nd.knocked_in and nd.end_reason is None
    aged_pv = _v2().price(nd.product, _env(dc), lifecycle_state=life.lifecycle)
    fresh_pv = _v2().price(fresh, _env(dc))
    assert aged_pv == pytest.approx(fresh_pv, rel=1e-12)


def test_note_knocks_out_on_first_ko_date():
    terms = build_terms(INCEPTION)
    days = _sse_days(INCEPTION, date(2024, 6, 30))
    life, terms, out = _run([(d, S0 if d == INCEPTION else 1.05 * S0) for d in days])
    ko_day = terms.ko_dates[0]
    assert ko_day == date(2024, 4, 2)                         # month 3 (2024-04-02, a trading day)
    assert [d for d, nd in out if not nd.alive] == [ko_day]
    assert out[-2][1].alive and out[-2][1].cash == 0.0
    last = out[-1][1]
    assert last.end_reason == "ko" and last.product is None
    assert last.cash == pytest.approx(0.2 * NOTIONAL * terms.ko_times[0], rel=1e-12)
    assert last.cash > 0
    assert sum(nd.cash for _, nd in out) == last.cash
    with pytest.raises(ValidationError):
        life.step(synth_chain(days[len(out)], spot=S0), _env(synth_chain(days[len(out)], spot=S0)))


def test_note_ki_latches():
    days = _sse_days(INCEPTION, date(2024, 1, 31))[:10]
    hit = [(d, S0 if i == 0 else (0.70 if i == 1 else 0.99) * S0) for i, d in enumerate(days)]
    twin = [(d, S0 if i == 0 else 0.99 * S0) for i, d in enumerate(days)]
    life, _, out = _run(hit)
    life_t, _, out_t = _run(twin)
    assert [nd.knocked_in for _, nd in out] == [False] + [True] * 9
    assert not any(nd.knocked_in for _, nd in out_t)
    assert all(nd.alive and nd.cash == 0.0 for _, nd in out)
    dc = synth_chain(days[-1], spot=0.99 * S0)
    engine = _v2()
    pv_ki = engine.price(out[-1][1].product, _env(dc), lifecycle_state=life.lifecycle)
    pv_tw = engine.price(out_t[-1][1].product, _env(dc), lifecycle_state=life_t.lifecycle)
    assert pv_ki < pv_tw


def test_note_matures_without_ko_no_ki():
    terms = build_terms(INCEPTION)
    days = _sse_days(INCEPTION, terms.maturity_date)
    life, terms, out = _run([(d, S0 if d == INCEPTION else 0.90 * S0) for d in days])
    assert out[-1][0] == terms.maturity_date and len(out) == len(days)
    last = out[-1][1]
    assert not last.alive and last.end_reason == "maturity" and not last.knocked_in
    assert last.cash == pytest.approx(0.2 * NOTIONAL * terms.maturity_years, rel=1e-12)
    assert all(nd.alive and nd.cash == 0.0 for _, nd in out[:-1])


def test_note_matures_after_ki():
    terms = build_terms(INCEPTION)
    days = _sse_days(INCEPTION, terms.maturity_date)
    spots = [S0, 0.70 * S0] + [0.90 * S0] * (len(days) - 2)
    life, terms, out = _run(list(zip(days, spots)))
    last = out[-1][1]
    assert out[-1][0] == terms.maturity_date
    assert not last.alive and last.end_reason == "maturity" and last.knocked_in
    assert last.cash == pytest.approx(-(1.0 - 0.90) * NOTIONAL, rel=1e-12)   # the holder loses
    # every ledger entry is paid on its determination day (no settlement lag): cash == determined amount
    for cf in life.lifecycle.ledger.cashflows:
        assert cf.payment_time == cf.determination_time


@pytest.mark.skipif(not Path("/Users/fuxinyao/quant-ark/example/output/snowball_2y_tail_stability/daily_chain.pkl").exists(),
                    reason="run 01_data.py first")
def test_inception_dates():
    from s2ts.chain import load_chain

    chains = load_chain()
    incs = inception_dates(chains, anchor_series(chains, 63))
    assert incs[0] == date(2023, 10, 9)
    assert 12 <= len(incs) <= 14
    days = [dc.day for dc in chains]
    for d in incs:
        q = (d.month - 1) // 3
        assert d == min(x for x in days if x.year == d.year and (x.month - 1) // 3 == q)
    assert len({(d.year, (d.month - 1) // 3) for d in incs}) == len(incs)


def test_inception_dates_needs_anchor():
    days = _days(120)   # 2024-01-02 .. 2024-06-14
    chains = [synth_chain(d) for d in days]
    incs = inception_dates(chains, anchor_series(chains, 63))
    # Q1 2024 starts before any anchor; Q2 starts 2024-04-01 (index 65 >= 62)
    assert incs == [date(2024, 4, 1)]


@pytest.mark.slow  # ~15-20 s per tail (a full bucket report + 15 V2 prices)
@pytest.mark.parametrize("tail", ["hold_forward", "converge_hl050"])
def test_daily_greeks_match_bucket_report(tail):
    from quantark.asset.equity.riskmeasures import calculate_carry_curve_buckets

    c = _cc_common()
    snap = c.load_snapshot()
    pricing = c.read_json(c.RESULTS_DIR / "pricing.json")
    s0 = float(snap["spot"])
    base = c.build_curve(snap, "hold_forward")
    far = snap["futures"][-1]
    b_inf = math.log(far["price"] / s0) / c.year_fraction(date.fromisoformat(far["expiry"]))
    curve = base if tail == "hold_forward" else ConvergingCarryCurve(base.nodes, b_inf=b_inf, half_life=0.5)
    product = c.build_product(pricing["coupon"]["value"], s0, c.build_terms())
    env = c.make_env(snap, curve.to_dividend_yield(c.rate_curve()))
    mults = c.lot_multipliers(snap)
    rep = calculate_carry_curve_buckets(product, env, _v2(), curve, lot_multipliers=mults,
                                        spot_bump_rel=0.01, forward_bump_rel=1e-3)
    g = daily_greeks(product, env, curve, _v2(), None, mults)
    assert isinstance(g, DayGreeks)
    assert g.pv == pytest.approx(rep.base_pv, rel=1e-12)
    assert g.spot_delta == pytest.approx(rep.spot_delta, rel=1e-12)
    assert g.pinned_delta == pytest.approx(rep.pinned_forward_delta, rel=1e-12, abs=1e-9)
    assert len(g.pillar_delta) == len(rep.buckets) == len(g.pillar_lots)
    for gd, gl, b in zip(g.pillar_delta, g.pillar_lots, rep.buckets):
        assert gd == pytest.approx(b.delta, rel=1e-12, abs=1e-12)
        assert gl == pytest.approx(b.hedge_lots, rel=1e-12, abs=1e-12)
    # holder side: the long note gains when spot rises (a snowball is long the index)
    assert g.spot_delta > 0


# ---------------------------------------------------------------------------
# Task 4: hedge books
# ---------------------------------------------------------------------------

MO_ALL_2709 = {6400.0: (560.0, 610.0), 6600.0: (480.0, 720.0), 6800.0: (410.0, 840.0)}


def book_chain(day, spot=6500.0, im=(("IM2612", 0.115, 6460.0), ("IM2706", 0.6137, 6300.0)),
               mo_series="MO2709", mo_strike=6600.0, mo_all=None, mo_tenor=0.86) -> DayChain:
    """A DayChain with explicit IM (code, tenor, close) and MO pillar series / ATM strike."""
    quotes = tuple(ImQuote(code, date(2027, 6, 18), tenor, close, close + 3.0) for code, tenor, close in im)
    mo_all = dict(mo_all or MO_ALL_2709)
    c, p = mo_all[mo_strike]
    df = math.exp(-RATE * mo_tenor)
    near = tuple((k, mo_all[k][0], mo_all[k][1]) for k in sorted(mo_all))
    mo = MoPillar(mo_series, date(2027, 9, 17), mo_tenor, mo_strike, c, p, mo_strike + (c - p) / df, 0.25, near)
    far = quotes[-1]
    return DayChain(day, spot, quotes, mo, mo_all, math.log(far.close / spot) / far.tenor)


def greeks(spot_delta=0.0, pinned=0.0, deltas=(0.0, 0.0, 0.0)) -> DayGreeks:
    return DayGreeks(0.0, spot_delta, pinned, tuple(deltas), tuple(-d for d in deltas))


def test_linear_note_is_perfectly_hedged():
    """Holder PV = a·(F_far − F_far0): a pure forward. The SHORT note's P&L plus Book A's is zero;
    a sign-flipped target doubles the error (guards the short-desk sign)."""
    a = 1500.0
    days = _days(10)
    path = [(6500.0 + 40.0 * math.sin(i), 6300.0 + 55.0 * math.cos(1.3 * i)) for i in range(10)]
    chains = [book_chain(d, spot=s, im=(("IM2612", 0.115, f + 150.0), ("IM2706", 0.6137, f)))
              for d, (s, f) in zip(days, path)]
    f0 = chains[0].im[-1].close
    pv = [a * (dc.im[-1].close - f0) for dc in chains]
    g = [greeks(spot_delta=a * dc.im[-1].close / dc.spot) for dc in chains]   # holder-side ∂PV/∂S
    book, flipped = BookA(), BookA()
    hedged, hedged_flipped, short_total = 0.0, 0.0, 0.0
    for t, dc in enumerate(chains):
        if t:
            short = -(pv[t] - pv[t - 1])
            short_total += short
            hedged += short + book.pnl(chains[t - 1], dc)
            hedged_flipped += short + flipped.pnl(chains[t - 1], dc)
        book.rebalance(dc, book.target(dc, g[t]))
        flipped.rebalance(dc, {k: -q for k, q in flipped.target(dc, g[t]).items()})
    assert book.positions["IM2706"] == pytest.approx(a / 200.0, rel=1e-12)    # short desk buys
    assert abs(hedged) < 1e-9
    assert abs(short_total) > 1e3
    assert hedged_flipped == pytest.approx(2.0 * short_total, rel=1e-12)


def test_book_a_rolls_to_new_longest():
    d0, d1 = _days(2)
    c0 = book_chain(d0)
    c1 = book_chain(d1, im=(("IM2612", 0.112, 6460.0), ("IM2706", 0.61, 6310.0), ("IM2709", 0.86, 6200.0)))
    book = BookA()
    g = greeks(spot_delta=1000.0)
    book.rebalance(c0, book.target(c0, g))
    assert not book.rolled
    q0 = book.positions["IM2706"]
    assert q0 == pytest.approx(1000.0 * 6500.0 / (200.0 * 6300.0), rel=1e-12)
    trades = book.rebalance(c1, book.target(c1, g))
    assert book.rolled
    assert set(book.positions) == {"IM2709"}
    by = {t.instrument: t for t in trades}
    assert set(by) == {"IM2706", "IM2709"}
    assert by["IM2706"].qty == pytest.approx(-q0) and by["IM2706"].price == 6310.0
    q1 = 1000.0 * 6500.0 / (200.0 * 6200.0)
    assert by["IM2709"].qty == pytest.approx(q1, rel=1e-12)
    for t in trades:
        assert t.cost == pytest.approx(1e-4 * abs(t.qty) * t.price * 200.0, rel=1e-12)
    assert turnover(trades) == pytest.approx(q0 + q1, rel=1e-12)
    assert total_cost(trades) == pytest.approx(sum(t.cost for t in trades), rel=1e-15)


def _b_target(book, dc, mo_delta, im_deltas=(0.0, 0.0), pinned=0.0):
    return book.target(dc, greeks(pinned=pinned, deltas=tuple(im_deltas) + (mo_delta,)))


def test_mo_pair_keeps_opened_strike():
    d0, d1, d2 = _days(3)
    c0 = book_chain(d0, mo_strike=6600.0)
    c1 = book_chain(d1, mo_strike=6400.0)               # ATM moved, same series
    book = BookB()
    book.rebalance(c0, _b_target(book, c0, 50.0))
    m = 100.0 * math.exp(-RATE * 0.86)
    assert book.positions["MO2709@6600"] == pytest.approx(50.0 / m, rel=1e-12)
    trades = book.rebalance(c1, _b_target(book, c1, 80.0))
    assert [t.instrument for t in trades if t.instrument.startswith("MO")] == ["MO2709@6600"]
    assert "MO2709@6400" not in book.positions
    assert book.positions["MO2709@6600"] == pytest.approx(80.0 / m, rel=1e-12)
    assert not book.rolled
    # marked at the HELD strike's C−P, not the ATM pair's
    c2 = book_chain(d2, mo_strike=6400.0, mo_all={**MO_ALL_2709, 6600.0: (500.0, 700.0)})
    assert book.mark(c2)["MO2709@6600"] == pytest.approx(-200.0)
    assert book.pnl(c1, c2) == pytest.approx(80.0 / m * ((500.0 - 700.0) - (480.0 - 720.0)) * 100.0, rel=1e-12)
    # the held strike vanishing from the series fails closed
    c3 = book_chain(d2, mo_strike=6400.0, mo_all={k: v for k, v in MO_ALL_2709.items() if k != 6600.0})
    with pytest.raises(StudyDataError):
        book.mark(c3)


def test_mo_roll_closes_old_pair():
    """Review Focus 3: on the roll day the old pair's overnight P&L is marked at today's settlement of
    the OLD series (off-chain mark), it is closed there, and the new pillar opens from zero."""
    d0, d1 = _days(2)
    c0 = book_chain(d0, mo_strike=6600.0)
    mo_2712 = {6200.0: (700.0, 560.0), 6400.0: (600.0, 640.0)}
    c1 = book_chain(d1, mo_series="MO2712", mo_strike=6400.0, mo_all=mo_2712, mo_tenor=1.0)
    book = BookB()
    book.rebalance(c0, _b_target(book, c0, 50.0))
    q = book.positions["MO2709@6600"]
    with pytest.raises(StudyDataError):
        book.pnl(c0, c1)                                    # no mark for the old series: fail closed
    rows = {"MO2709-C-6600": {"close": 0.0, "settle": 470.0}, "MO2709-P-6600": {"close": 0.0, "settle": 735.0}}
    extra = offchain_marks(rows, book.unmarkable(c1))
    assert extra == {"MO2709@6600": pytest.approx(470.0 - 735.0)}
    assert book.pnl(c0, c1, extra) == pytest.approx(q * ((470.0 - 735.0) - (480.0 - 720.0)) * 100.0, rel=1e-12)
    trades = book.rebalance(c1, _b_target(book, c1, 50.0), extra)
    assert book.rolled
    by = {t.instrument: t for t in trades if t.instrument.startswith("MO")}
    assert by["MO2709@6600"].qty == pytest.approx(-q) and by["MO2709@6600"].price == pytest.approx(-265.0)
    assert by["MO2712@6400"].qty == pytest.approx(50.0 / (100.0 * math.exp(-RATE * 1.0)), rel=1e-12)
    assert set(k for k in book.positions if k.startswith("MO")) == {"MO2712@6400"}


def test_dropped_im_pillar_closed_at_mark():
    d0, d1 = _days(2)
    im0 = (("IM2611", 0.03, 6490.0), ("IM2612", 0.115, 6460.0), ("IM2706", 0.6137, 6300.0))
    c0 = book_chain(d0, im=im0)
    c1 = book_chain(d1, im=(("IM2612", 0.112, 6470.0), ("IM2706", 0.61, 6310.0)))   # IM2611 < 7 days
    book = BookB()
    book.rebalance(c0, _b_target(book, c0, 0.0, im_deltas=(400.0, 200.0, 300.0)))
    assert book.positions["IM2611"] == pytest.approx(2.0)
    assert book.unmarkable(c1) == ["IM2611"]
    extra = offchain_marks({"IM2611": {"close": 6495.0, "settle": 6494.0}}, ["IM2611"])
    pnl = book.pnl(c0, c1, extra)
    expect = 2.0 * 5.0 * 200 + 1.0 * 10.0 * 200 + 1.5 * 10.0 * 200
    assert pnl == pytest.approx(expect, rel=1e-12)
    trades = book.rebalance(c1, _b_target(book, c1, 0.0, im_deltas=(200.0, 300.0)), extra)
    closed = [t for t in trades if t.instrument == "IM2611"][0]
    assert closed.qty == pytest.approx(-2.0) and closed.price == 6495.0
    assert closed.cost == pytest.approx(1e-4 * 2.0 * 6495.0 * 200.0, rel=1e-12)
    assert "IM2611" not in book.positions and book.rolled


def test_book_b_spot_neutral():
    dc = book_chain(date(2026, 11, 5), im=(("IM2612", 0.115, 6460.0), ("IM2703", 0.36, 6380.0),
                                           ("IM2706", 0.6137, 6300.0)))
    deltas = (120.0, -340.0, 2250.0, -6200.0)              # holder-side ∂PV/∂F_i, 3 IM + MO
    pinned = -337.0
    fwd = [q.close for q in dc.im] + [dc.mo.forward]
    spot_delta = pinned + sum(f / dc.spot * d for f, d in zip(fwd, deltas))
    book = BookB()
    tgt = book.target(dc, greeks(spot_delta=spot_delta, pinned=pinned, deltas=deltas))
    df = math.exp(-RATE * dc.mo.tenor)
    exposure = sum(tgt[q.code] * 200.0 * q.close / dc.spot for q in dc.im) + \
        tgt["MO2709@6600"] * 100.0 * df * dc.mo.forward / dc.spot
    assert abs(exposure - spot_delta) < 1e-6
    # the pinned-forward delta sits on the far IM, on top of its pillar lots
    assert tgt["IM2706"] == pytest.approx(2250.0 / 200.0 + pinned * dc.spot / (200.0 * 6300.0), rel=1e-12)
    with pytest.raises(ValidationError):
        book.target(dc, greeks(deltas=(1.0, 2.0, 3.0)))       # one pillar short


def test_mo_cost_40_cny_per_pair():
    dc = book_chain(date(2026, 11, 5))
    book = BookB()
    trades = book.rebalance(dc, {"MO2709@6600": 3.0})
    assert trades == [Trade("MO2709@6600", 3.0, -240.0, 120.0)]
    assert book.premium(trades) == pytest.approx(3.0 * -240.0 * 100.0)
    assert book.value(dc) == pytest.approx(3.0 * -240.0 * 100.0)
    out = book.unwind(dc)
    assert out == [Trade("MO2709@6600", -3.0, -240.0, 120.0)] and book.positions == {}
    assert not book.rolled


def test_book_pnl_split_into_im_margin_and_mo_mtm():
    d0, d1 = _days(2)
    c0 = book_chain(d0)
    c1 = book_chain(d1, im=(("IM2612", 0.112, 6470.0), ("IM2706", 0.61, 6290.0)),
                    mo_all={**MO_ALL_2709, 6600.0: (490.0, 715.0)})
    book = BookB()
    book.rebalance(c0, {"IM2706": 2.0, "MO2709@6600": -1.5})
    im_vm, mo_mtm = book.pnl_split(c0, c1)
    assert im_vm == pytest.approx(2.0 * (6290.0 - 6300.0) * 200.0)
    assert mo_mtm == pytest.approx(-1.5 * ((490.0 - 715.0) - (480.0 - 720.0)) * 100.0)
    assert book.pnl(c0, c1) == pytest.approx(im_vm + mo_mtm, rel=1e-15)


# ---------------------------------------------------------------------------
# Task 5: runner
# ---------------------------------------------------------------------------

import json  # noqa: E402


class FakeEngine:
    """Cheap stand-in for V2 (same ``price`` signature): PV moves with spot and q(1.5), so the
    loop, books and cash account are exercised without quadrature."""

    def __init__(self):
        self.calls = 0

    def price(self, product, env, *, lifecycle_state=None):
        self.calls += 1
        s = float(env.spot_quote.spot)
        q = float(env.get_div_yield(1.5))
        return 4e6 * (s / product.initial_price - 1.0) - 2e7 * (q - 0.05)


DAY_KEYS = {"date", "note_pnl", "pnl_a", "pnl_b", "cost_a", "cost_b", "turn_a", "turn_b", "roll_a", "roll_b",
            "d_far_basis", "d_mo_basis", "pv", "spot_delta", "pinned_delta", "hedged_a", "hedged_b",
            "accrual_a", "accrual_b", "note_cash", "vega", "d_vol"}


def test_daily_vega_sign_and_order_invariance():
    """Vega is priced AFTER the 15 delta prices: switching it off leaves every other field bitwise equal;
    a long snowball holder is short vol."""
    dc = synth_chain(INCEPTION, spot=S0)
    curve = build_curve(dc, "converge_hl050", -0.05)
    env = make_env(dc, curve.to_dividend_yield(_flat_rate()))
    note = build_note(0.2, S0, build_terms(INCEPTION))
    _, mults = pillar_inputs(dc)
    with_vega = daily_greeks(note, env, curve, _v2(), None, mults)
    without = daily_greeks(note, env, curve, _v2(), None, mults, with_vega=False)
    assert without.vega is None
    assert with_vega.vega < 0
    for name in ("pv", "spot_delta", "pinned_delta", "pillar_delta", "pillar_lots"):
        assert getattr(with_vega, name) == getattr(without, name)
    # one vol point = 0.01 absolute vol, central
    from quantark.param import FlatVolSurface

    def pv_at(vol):
        e = make_env(dc, curve.to_dividend_yield(_flat_rate()))
        e.vol_surface = FlatVolSurface(volatility=vol)
        return _v2().price(note, e)

    assert with_vega.vega == pytest.approx((pv_at(dc.mo.vol + 0.01) - pv_at(dc.mo.vol - 0.01)) / 2.0, rel=1e-9)


def test_run_one_records_vega_and_same_expiry_vol_change():
    chains = _path_chains(4, lambda i: S0)
    chains = [dataclasses.replace(dc, mo=dataclasses.replace(dc.mo, vol=0.25 + 0.01 * i)) for i, dc in enumerate(chains)]
    rolled = dataclasses.replace(chains[3], mo=dataclasses.replace(chains[3].mo, series="MO2712",
                                                                   expiry=date(2027, 12, 17)))
    chains[3] = rolled
    old_pair = {"MO2709-C-6400": {"close": 0.0, "settle": 560.0}, "MO2709-P-6400": {"close": 0.0, "settle": 610.0}}
    run = run_one(INCEPTION, "hold_forward", chains, _anchors(chains), 0.2, parity=False, engine=FakeEngine(),
                  rows_loader=lambda day: old_pair)
    d_vols = [d["d_vol"] for d in run["days"]]
    assert d_vols[:2] == [pytest.approx(1.0), pytest.approx(1.0)] and d_vols[2] is None   # roll day: None
    assert all(d["vega"] == 0.0 for d in run["days"][:-1])   # FakeEngine ignores vol
    assert run["day0_vega"] == 0.0


def _path_chains(n, spot_fn, start=INCEPTION):
    days = _sse_days(start, start + pd.Timedelta(days=200).to_pytimedelta())[:n]
    return [synth_chain(d, spot=spot_fn(i), far_close=6300.0 + 20.0 * math.sin(i)) for i, d in enumerate(days)]


def _anchors(chains):
    return {dc.day: -0.05 for dc in chains}


def test_run_one_synthetic_short_path():
    chains = _path_chains(8, lambda i: S0 * (1.0 + 0.01 * math.sin(i)))
    run = run_one(INCEPTION, "converge_hl050", chains, _anchors(chains), 0.2, parity=False, engine=FakeEngine())
    assert run["inception"] == INCEPTION.isoformat() and run["tail"] == "converge_hl050" and run["coupon"] == 0.2
    assert run["end_reason"] == "data_end" and run["truncated"] is True
    assert len(run["days"]) == 7
    assert all(set(d) >= DAY_KEYS for d in run["days"])
    assert all(d["roll_a"] is False and d["d_far_basis"] is not None for d in run["days"])
    for b in ("a", "b"):
        hedged = [d[f"hedged_{b}"] for d in run["days"]]
        for d in run["days"]:
            assert d[f"hedged_{b}"] == pytest.approx(d["note_pnl"] + d[f"pnl_{b}"] + d[f"accrual_{b}"], rel=1e-12)
        # wealth identity: daily hedged P&L sums to the own-PV terminal (gross of costs)
        assert sum(hedged) == pytest.approx(run[f"terminal_own_{b}"], rel=1e-9, abs=1e-6)
        assert run[f"day0_cost_{b}"] > 0
    # traded-price terminal = own-PV terminal less the day-0 own PV grown at r
    growth = 1.0
    for prev, dc in zip(chains, chains[1:]):
        growth *= 1.0 + RATE * (dc.day - prev.day).days / 365.0
    assert run["terminal_own_a"] - run["terminal_a"] == pytest.approx(run["day0_pv"] * growth, rel=1e-9)
    json.dumps(run, allow_nan=False)


def test_run_one_knock_out_unwinds_and_stops():
    chains = _path_chains(80, lambda i: S0 if i == 0 else 1.05 * S0)
    run = run_one(INCEPTION, "hold_forward", chains, _anchors(chains), 0.2, parity=False, engine=FakeEngine())
    assert run["end_reason"] == "ko" and run["truncated"] is False
    last = run["days"][-1]
    assert last["date"] == "2024-04-02" and last["pv"] == 0.0 and last["spot_delta"] is None
    assert last["note_cash"] == pytest.approx(0.2 * NOTIONAL * build_terms(INCEPTION).ko_times[0], rel=1e-12)
    prev_pv = run["days"][-2]["pv"]
    assert last["note_pnl"] == pytest.approx(prev_pv - last["note_cash"], rel=1e-12)
    assert last["turn_a"] > 0                                  # the unwind traded
    for b in ("a", "b"):
        assert sum(d[f"hedged_{b}"] for d in run["days"]) == pytest.approx(run[f"terminal_own_{b}"], rel=1e-9)


def test_same_coupon_and_contract_across_tails():
    chains = _path_chains(4, lambda i: S0 * (1.0 + 0.005 * i))
    runs = [run_one(INCEPTION, t, chains, _anchors(chains), 0.17, parity=False, engine=FakeEngine())
            for t in ("hold_forward", "converge_hl050")]
    assert runs[0]["coupon"] == runs[1]["coupon"] == 0.17
    assert [d["date"] for d in runs[0]["days"]] == [d["date"] for d in runs[1]["days"]]
    assert [d["note_cash"] for d in runs[0]["days"]] == [d["note_cash"] for d in runs[1]["days"]]
    assert runs[0]["day0_pv"] != runs[1]["day0_pv"]


def test_coupons_cached_once_per_inception(tmp_path):
    calls = []

    def solver(dc):
        calls.append(dc.day)
        return 0.1 + 0.01 * len(calls)

    chains = _path_chains(3, lambda i: S0)
    path = tmp_path / "coupons.json"
    first = coupons_for([INCEPTION], chains, path, solver=solver)
    again = coupons_for([INCEPTION], chains, path, solver=solver)
    assert first == again == {INCEPTION: 0.11} and calls == [INCEPTION]
    assert json.loads(path.read_text()) == {INCEPTION.isoformat(): 0.11}


def test_resume_skips_complete_and_ignores_tmp(tmp_path):
    """Review Focus 5: a complete file is skipped, a stale .tmp is never read and is removed."""
    a, b = (INCEPTION, "hold_forward"), (INCEPTION, "zero_carry")
    complete = {"inception": INCEPTION.isoformat(), "tail": "hold_forward", "days": [], "done": True}
    run_path(tmp_path, *a).write_text(json.dumps(complete))
    stale = run_path(tmp_path, *b).with_name(run_path(tmp_path, *b).name + ".tmp")
    stale.write_text("{ half-written")
    called = []

    def job(inception, tail):
        called.append((inception, tail))
        return {"inception": inception.isoformat(), "tail": tail, "days": []}

    paths = run_all([INCEPTION], ["hold_forward", "zero_carry"], tmp_path, workers=0, job=job)
    assert called == [b]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [run_path(tmp_path, *a).name, run_path(tmp_path, *b).name])
    assert json.loads(run_path(tmp_path, *b).read_text()) == {
        "inception": INCEPTION.isoformat(), "tail": "zero_carry", "days": []}
    assert json.loads(run_path(tmp_path, *a).read_text()) == complete
    assert set(paths) == {run_path(tmp_path, *a), run_path(tmp_path, *b)}


def _ok_job(called):
    def job(inception, tail):
        called.append((inception, tail))
        return {"inception": inception.isoformat(), "tail": tail, "days": []}
    return job


def test_failed_run_writes_error_json_and_the_rest_complete(tmp_path):
    """CP1 #1: one failing run never discards the others; it leaves only an .error.json, a summary
    error is raised after all runs, and resume re-runs it (deleting the .error.json first)."""
    tails = ["hold_forward", "zero_carry", "hold_yield"]
    called = []

    def job(inception, tail):
        called.append((inception, tail))
        if tail == "zero_carry":
            raise ValueError("boom in zero_carry")
        return {"inception": inception.isoformat(), "tail": tail, "days": []}

    with pytest.raises(RunsFailed) as e:
        run_all([INCEPTION], tails, tmp_path, workers=0, job=job)
    assert "zero_carry" in str(e.value)
    assert len(called) == 3                                       # every run was attempted
    bad = error_path(tmp_path, INCEPTION, "zero_carry")
    assert bad.exists() and not run_path(tmp_path, INCEPTION, "zero_carry").exists()
    err = json.loads(bad.read_text())
    assert err["tail"] == "zero_carry" and "boom in zero_carry" in err["traceback"]
    for t in ("hold_forward", "hold_yield"):
        assert json.loads(run_path(tmp_path, INCEPTION, t).read_text())["tail"] == t
    again = []
    run_all([INCEPTION], tails, tmp_path, workers=0, job=_ok_job(again))
    assert again == [(INCEPTION, "zero_carry")]
    assert not bad.exists() and run_path(tmp_path, INCEPTION, "zero_carry").exists()


def test_invalid_run_file_is_rerun(tmp_path):
    """CP1 #2: a truncated or incomplete <run>.json is deleted and re-run, never trusted."""
    run_path(tmp_path, INCEPTION, "hold_forward").write_text('{"inception": "2024-01-02", "ta')
    run_path(tmp_path, INCEPTION, "zero_carry").write_text(json.dumps(
        {"inception": INCEPTION.isoformat(), "tail": "zero_carry"}))           # no days list
    run_path(tmp_path, INCEPTION, "hold_yield").write_text(json.dumps(
        {"inception": INCEPTION.isoformat(), "tail": "hold_forward", "days": []}))  # someone else's run
    called = []
    run_all([INCEPTION], ["hold_forward", "zero_carry", "hold_yield"], tmp_path, workers=0, job=_ok_job(called))
    assert sorted(t for _, t in called) == ["hold_forward", "hold_yield", "zero_carry"]
    for t in ("hold_forward", "zero_carry", "hold_yield"):
        assert json.loads(run_path(tmp_path, INCEPTION, t).read_text())["tail"] == t


def test_out_dir_lock(tmp_path):
    """CP1 #3: a live lock refuses to start (and touches nothing); a dead pid's lock is taken over."""
    import os

    lock = tmp_path / ".lock"
    stale_tmp = tmp_path / "x__hold_forward.json.tmp"
    stale_tmp.write_text("{")
    lock.write_text(str(os.getpid()))
    with pytest.raises(OutDirLocked):
        run_all([INCEPTION], ["hold_forward"], tmp_path, workers=0, job=_ok_job([]))
    assert stale_tmp.exists() and lock.read_text() == str(os.getpid())
    lock.write_text("99999999")                                   # no such process
    called = []
    run_all([INCEPTION], ["hold_forward"], tmp_path, workers=0, job=_ok_job(called))
    assert called == [(INCEPTION, "hold_forward")]
    assert not lock.exists() and not stale_tmp.exists()


def test_parity_flag_logic():
    lots = (1.0, -2.0, 3.0)
    assert parity_flag(0.0, 0.6e-4 * NOTIONAL, lots, lots)                       # 0.6 bp
    assert not parity_flag(0.0, 0.4e-4 * NOTIONAL, lots, (1.005, -2.005, 3.0))   # 0.4 bp, 0.005 lots
    assert parity_flag(0.0, 0.0, lots, (1.0, -2.0, 3.02))                         # a lot off by 0.02
    with pytest.raises(ValidationError):
        parity_flag(0.0, 0.0, lots, lots[:2])


def test_rss_guard_trips_and_calls_back():
    import threading

    tripped = threading.Event()
    guard = RssGuard(limit_bytes=1, poll_s=0.01, on_trip=tripped.set)
    guard.start()
    assert tripped.wait(5.0) and guard.tripped.is_set()
    quiet = RssGuard(limit_bytes=1 << 50, poll_s=0.01, on_trip=tripped.clear)
    quiet.start()
    quiet.stop()
    quiet.join(2.0)
    assert not quiet.tripped.is_set() and quiet.rss() > 0


# ---------------------------------------------------------------------------
# Task 6: measures and verdict (synthetic run dicts, no pricing)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from s2ts import NOTIONAL as _N  # noqa: E402

_BP = _N / 1e4


def _row(i, hedged_a, hedged_b=None, note=None, *, d_far=0.0, d_mo=0.0, roll_a=False, roll_b=False,
         d_vol=0.0, vega=0.0, turn=1.0, cost=0.0):
    hb = hedged_a if hedged_b is None else hedged_b
    return {
        "date": f"2024-01-{i + 1:02d}", "note_pnl": (hedged_a if note is None else note) * _BP, "note_cash": 0.0,
        "pv": 0.0, "spot_delta": 0.0, "pinned_delta": 0.0, "vega": vega, "d_vol": d_vol,
        "d_far_basis": d_far, "d_mo_basis": d_mo,
        "hedged_a": hedged_a * _BP, "hedged_b": hb * _BP, "roll_a": roll_a, "roll_b": roll_b,
        "turn_a": turn, "turn_b": 2 * turn, "cost_a": cost * _BP, "cost_b": cost * _BP,
        "pnl_a": 0.0, "pnl_b": 0.0, "vm_a": 0.0, "vm_b": 0.0, "mtm_a": 0.0, "mtm_b": 0.0,
        "accrual_a": 0.0, "accrual_b": 0.0,
    }


def _synth_run(inc, tail, a, b=None, *, term=(0.0, 0.0), truncated=False, day0_pv=0.0, day0_vega=0.0, rows=None):
    b = a if b is None else b
    days = rows if rows is not None else [_row(i, x, y) for i, (x, y) in enumerate(zip(a, b))]
    return {
        "inception": inc, "tail": tail, "coupon": 0.1, "day0_pv": day0_pv, "day0_vega": day0_vega,
        "truncated": truncated, "end_reason": "data_end" if truncated else "knock_out", "live_days": len(days),
        "terminal_a": term[0] * _BP, "terminal_b": term[1] * _BP,
        "terminal_own_a": (term[0] + 5) * _BP, "terminal_own_b": (term[1] + 5) * _BP,
        "cost_total_a": 3 * _BP, "cost_total_b": 4 * _BP, "parity": None, "days": days,
    }


def _alt(n, sd):
    """n values, mean 0, sample std exactly sd (n even): +-sd*sqrt((n-1)/n)."""
    k = sd * math.sqrt((n - 1) / n)
    return [k if i % 2 == 0 else -k for i in range(n)]


def test_std_and_r2_on_known_series():
    h = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
    note = [2.0 * x for x in h]                      # hedged variance = 1/4 of note variance
    out = per_run(_synth_run("2024-01-01", "hold_forward", h, rows=[_row(i, x, note=n) for i, (x, n) in enumerate(zip(h, note))]))
    assert out["a"]["std_bp"] == pytest.approx(float(np.std(h, ddof=1)), rel=1e-12)
    assert out["a"]["r2"] == pytest.approx(0.75, abs=1e-12)
    assert out["a"]["turnover_lots_per_day"] == pytest.approx(1.0)
    assert out["b"]["turnover_lots_per_day"] == pytest.approx(2.0)
    assert out["a"]["cost_bp"] == pytest.approx(3.0) and out["b"]["cost_bp"] == pytest.approx(4.0)


def test_max_drawdown_and_roll_day_abs():
    h = [1.0, 2.0, -4.0, -1.0, 3.0]
    rows = [_row(i, x, roll_a=(i in (2, 3))) for i, x in enumerate(h)]
    out = per_run(_synth_run("2024-01-01", "hold_forward", h, rows=rows))["a"]
    assert out["max_dd_bp"] == pytest.approx(5.0)    # cum: 1,3,-1,-2,1 ; peak 3 -> trough -2
    assert out["roll_day_abs_bp"] == pytest.approx(2.5)


def test_basis_regression_recovers_beta():
    rng = np.random.default_rng(0)
    n = 80
    d_far, d_mo = rng.normal(0, 0.003, n), rng.normal(0, 0.002, n)
    h = 3.0 * 100 * d_far - 1.0 * 100 * d_mo + 1e-6 * rng.normal(size=n)
    rows = [_row(i, h[i], d_far=d_far[i], d_mo=d_mo[i]) for i in range(n)]
    # a roll day with a wild value and None basis must be excluded
    rows.append(_row(n, 500.0, d_far=None, d_mo=None, roll_a=True))
    b = per_run(_synth_run("2024-01-01", "hold_forward", [0.0], rows=rows))["a"]["basis"]
    assert b["beta_far"] == pytest.approx(3.0, abs=1e-4)
    assert b["beta_mo"] == pytest.approx(-1.0, abs=1e-4)
    assert b["basis_r2"] == pytest.approx(1.0, abs=1e-6)


def test_vega_residual_diagnostic():
    # short vega P&L = -vega_{t-1} * d_vol_t ; day 1 uses day0_vega
    rows = [_row(0, -2.0 * 0.5, d_vol=0.5, vega=3.0 * _BP),   # prev = day0_vega = 2 -> -1.0 ; resid 0
            _row(1, -3.0 * 1.0, d_vol=1.0, vega=4.0 * _BP),   # prev vega 3 -> -3 ; resid 0
            _row(2, 7.0, d_vol=None, vega=4.0 * _BP)]         # skipped
    out = per_run(_synth_run("2024-01-01", "hold_forward", [0.0], rows=rows, day0_vega=2.0 * _BP))["a"]
    assert out["std_vega_resid_bp"] == pytest.approx(0.0, abs=1e-9)


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _books(tail_sd, n_inc=13, hf_sd=2.0):
    incs = [f"2024-{i + 1:02d}-01" for i in range(n_inc)]
    hf = [_synth_run(inc, "hold_forward", _alt(10, hf_sd)) for inc in incs]
    tl = [_synth_run(inc, "converge_hl050", _alt(10, tail_sd(i))) for i, inc in enumerate(incs)]
    return {"hold_forward": hf, "converge_hl050": tl}


def test_pooled_std_equal_weight():
    a = [_synth_run("2024-01-01", "converge_hl050", _alt(10, 1.0)), _synth_run("2024-02-01", "converge_hl050", _alt(100, 2.0))]
    hf = [_synth_run("2024-01-01", "hold_forward", _alt(10, 1.0)), _synth_run("2024-02-01", "hold_forward", _alt(100, 2.0))]
    out = paired({"hold_forward": hf, "converge_hl050": a}, "a")
    assert out["converge_hl050"]["pooled_std_bp"] == pytest.approx(math.sqrt(2.5), rel=1e-9)
    assert out["converge_hl050"]["hf_pooled_std_bp"] == pytest.approx(math.sqrt(2.5), rel=1e-9)
    assert out["converge_hl050"]["n"] == 2


def test_paired_ratios_and_t_and_holm():
    rb = _books(lambda i: 1.0 + 0.01 * i)
    rb["converge_hl025"] = rb["converge_hl050"]
    rb["converge_hl100"] = rb["converge_hl050"]
    out = paired(rb, "a")
    r = out["converge_hl050"]
    assert r["n"] == 13 and r["n_lower"] == 13
    assert r["ratios"][0] == pytest.approx(0.25, rel=1e-9)
    assert r["t_log_ratio"] < -5 and r["p_holm"] <= 1.0
    assert r["p_holm"] >= r["p_raw"]


def test_terminal_traded_price_net_of_day0_gap():
    h = _alt(6, 1.0)
    r1 = _synth_run("2024-01-01", "hold_forward", h, term=(2.0, -1.0), day0_pv=0.0)
    r2 = _synth_run("2024-01-01", "converge_hl050", h, term=(2.0, -1.0), day0_pv=100.0 * _BP)
    r2["terminal_own_a"] = r1["terminal_own_a"] + 100.0 * _BP
    assert per_run(r1)["a"]["terminal_bp"] == per_run(r2)["a"]["terminal_bp"] == pytest.approx(2.0)
    assert per_run(r2)["a"]["terminal_own_bp"] == pytest.approx(107.0)


def _verdict_inputs(tail_sd, term_tail, term_hf, n=13, complete=None):
    incs = [f"2024-{i + 1:02d}-01" for i in range(n)]
    complete = n if complete is None else complete
    hf, tl = [], []
    for i, inc in enumerate(incs):
        trunc = i >= complete
        hf.append(_synth_run(inc, "hold_forward", _alt(10, 2.0), term=(term_hf[i % len(term_hf)],) * 2, truncated=trunc))
        tl.append(_synth_run(inc, "converge_hl050", _alt(10, tail_sd(i)), term=(term_tail[i % len(term_tail)],) * 2,
                       truncated=trunc))
    rb = {"hold_forward": hf, "converge_hl050": tl}
    return (paired(rb, "a"), paired(rb, "b"), terminal_paired(rb, "a"), terminal_paired(rb, "b"), rb)


_TERM_LOW, _TERM_HF = [0.5, -0.5, 0.6, -0.6], [3.0, -3.0, 2.0, -2.0]


def test_verdict_requires_both_conditions():
    # lower in only 6/13 inceptions, pooled std lower (huge win in those 6) -> False
    sd = lambda i: 0.1 if i < 6 else 2.5            # noqa: E731
    pa, pb, ta, tb, _ = _verdict_inputs(sd, _TERM_LOW, _TERM_HF)
    assert pa["converge_hl050"]["n_lower"] == 6
    assert pa["converge_hl050"]["pooled_std_bp"] < pa["converge_hl050"]["hf_pooled_std_bp"]
    v = verdict(pa, pb, ta, tb)["converge_hl050"]
    assert v["book_a"] is False and v["daily_ok_a"] is False
    # lower in 7/13 but higher pooled std -> False
    sd = lambda i: 1.9 if i < 7 else 6.0            # noqa: E731
    pa, pb, ta, tb, _ = _verdict_inputs(sd, _TERM_LOW, _TERM_HF)
    assert pa["converge_hl050"]["n_lower"] == 7
    assert pa["converge_hl050"]["pooled_std_bp"] > pa["converge_hl050"]["hf_pooled_std_bp"]
    assert verdict(pa, pb, ta, tb)["converge_hl050"]["book_a"] is False
    # both -> True
    pa, pb, ta, tb, _ = _verdict_inputs(lambda i: 1.0, _TERM_LOW, _TERM_HF)
    v = verdict(pa, pb, ta, tb)["converge_hl050"]
    assert v["book_a"] is True and v["book_b"] is True and v["beats_hold_forward"] is True


def test_verdict_per_book_independent():
    pa, pb, ta, tb, rb = _verdict_inputs(lambda i: 1.0, _TERM_LOW, _TERM_HF)
    # make Book B of the tail worse than hold-forward
    for r in rb["converge_hl050"]:
        for d in r["days"]:
            d["hedged_b"] = d["hedged_b"] * 5.0
    pb = paired(rb, "b")
    v = verdict(pa, pb, ta, tb)["converge_hl050"]
    assert v["book_a"] is True and v["book_b"] is False and v["beats_hold_forward"] is False
    assert v["daily_ok_a"] and not v["daily_ok_b"]


def test_verdict_requires_terminal_agreement():
    pa, pb, ta, tb, _ = _verdict_inputs(lambda i: 1.0, [5.0, -5.0, 4.0, -4.0], _TERM_HF)
    v = verdict(pa, pb, ta, tb)["converge_hl050"]
    assert v["daily_ok_a"] is True and v["terminal_ok_a"] is False
    assert v["book_a"] is False and v["beats_hold_forward"] is False


def test_verdict_not_decidable_under_5_complete():
    pa, pb, ta, tb, _ = _verdict_inputs(lambda i: 1.0, _TERM_LOW, _TERM_HF, complete=4)
    assert ta["converge_hl050"]["n_complete"] == 4
    v = verdict(pa, pb, ta, tb)["converge_hl050"]
    assert v["book_a"] == "not_decidable" and v["book_b"] == "not_decidable"
    assert v["beats_hold_forward"] is False


def test_terminal_excludes_truncated():
    pa, pb, ta, tb, rb = _verdict_inputs(lambda i: 1.0, _TERM_LOW, _TERM_HF)
    base = terminal_paired(rb, "a")["converge_hl050"]["terminal_std_bp"]
    rb["converge_hl050"][0]["truncated"] = True
    rb["converge_hl050"][0]["terminal_a"] = 1e9
    out = terminal_paired(rb, "a")["converge_hl050"]
    assert out["n_complete"] == 12 and out["terminal_std_bp"] < 10 * base


def test_verdict_complete_only_and_vega_diagnostic_not_in_verdict():
    pa, pb, ta, tb, rb = _verdict_inputs(lambda i: 1.0, _TERM_LOW, _TERM_HF)
    v = verdict(pa, pb, ta, tb, paired_complete(rb, "a"), paired_complete(rb, "b"))["converge_hl050"]
    assert v["complete_only"] == {"daily_ok_a": True, "daily_ok_b": True}
    assert "vega_resid" in pa["converge_hl050"] and pa["converge_hl050"]["vega_resid"]["n"] == 13


def test_basis_mitigation_table():
    pa = per_run(_synth_run("2024-01-01", "hold_forward", [1.0, 2.0, 3.0]))
    fake = {"a": {"basis": {"beta_far": -2.0, "beta_mo": 1.0, "basis_r2": 0.5}, "cost_bp": 3.0},
            "b": {"basis": {"beta_far": 1.0, "beta_mo": -0.5, "basis_r2": 0.2}, "cost_bp": 5.0}}
    out = basis_mitigation([{"tail": "hold_forward", "inception": "x", **fake},
                            {"tail": "hold_forward", "inception": "y", **fake}, {"tail": "zero_carry", **pa}])
    t = out["hold_forward"]
    assert t["a"]["abs_beta_far"] == pytest.approx(2.0) and t["b"]["abs_beta_far"] == pytest.approx(1.0)
    assert t["b_minus_a"]["abs_beta_far"] == pytest.approx(-1.0)
    assert t["b_minus_a"]["basis_r2"] == pytest.approx(-0.3) and t["b_minus_a"]["cost_bp"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# T7: report (Bloomberg x Swiss) -- structure and partial runs
# ---------------------------------------------------------------------------

import re  # noqa: E402

_REPORT_SECTIONS = ("verdict", "stability", "hedging", "basis", "curve", "method", "parity")
_REPORT_TAILS = ("hold_forward", "hold_yield", "zero_carry", "converge_hl025", "converge_hl050", "converge_hl100")


def _report_module():
    name = "s2ts_report_04"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STUDY_DIR / "04_report.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _report_runs(skip_tails=(), n_inc=6):
    runs = []
    for i in range(n_inc):
        inc = f"2024-{i + 1:02d}-01"
        for k, tail in enumerate(_REPORT_TAILS):
            if tail in skip_tails:
                continue
            sd = 2.0 if tail == "hold_forward" else 1.0 + 0.1 * k
            run = _synth_run(inc, tail, _alt(10, sd), term=(0.5 + 0.1 * i, -0.5), truncated=(i == n_inc - 1),
                             day0_pv=1000.0 * k, day0_vega=10.0)
            run["parity"] = {"flag": i == 0 and k == 1, "pv_v1": 1.0, "pv_v2": 2.0, "pv_diff_bp": 0.7,
                             "lots_v1": [1.0], "lots_v2": [1.0], "max_lot_diff": 0.0}
            runs.append(run)
    return runs


_REPORT_CURVE = {"day": "2026-11-05", "grid": [0.25, 1.0, 2.0],
                 "q": {t: [0.04, 0.05 + 0.001 * k, 0.06] for k, t in enumerate(_REPORT_TAILS)}}


def _render_report(runs, errors=()):
    from s2ts import html as s2html

    mod = _report_module()
    stability = json.loads((STUDY_DIR / "results" / "stability.json").read_text())
    data = mod.build_data(stability, runs, list(errors), [f"2024-{i + 1:02d}-01" for i in range(6)], _REPORT_CURVE)
    return s2html.render(data), data


def test_report_structure():
    page, data = _render_report(_report_runs(), errors=["2024-02-01__zero_carry"])
    for sid in _REPORT_SECTIONS:
        assert f'<section id="{sid}"' in page, sid
    assert 'id="theme"' in page and 'class="top"' in page and "position: sticky" in page
    assert not re.findall(r'(?:src|href)\s*=\s*["\']?https?:', page)
    assert "http" not in " ".join(re.findall(r'(?:src|href)\s*=\s*["\'][^"\']*', page))
    assert page.count('class="vrow"') == 3                       # one per converging tail
    assert "<svg" in page and "2024-02-01__zero_carry" in page   # failed run listed in #parity
    assert "not a recommendation" in page                        # zero_carry caveat
    assert set(data["verdict"]) == {"converge_hl025", "converge_hl050", "converge_hl100"}


def test_report_handles_partial_runs():
    page, data = _render_report(_report_runs(skip_tails=("converge_hl100",), n_inc=4))
    assert page.count('class="vrow"') == 3
    assert "converge_hl100" in data["missing"]["tails"]
    assert "missing" in page
    for sid in _REPORT_SECTIONS:
        assert f'<section id="{sid}"' in page
    page2, _ = _render_report([r for r in _report_runs() if r["tail"] != "hold_forward"])
    assert "unavailable" in page2                                # no baseline: pairing unavailable, no crash
    page3, _ = _render_report([])
    assert '<section id="verdict"' in page3
