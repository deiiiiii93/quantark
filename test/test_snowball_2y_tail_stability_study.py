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
