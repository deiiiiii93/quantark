"""Tests for the 2Y snowball gamma pin-risk study (example/case_studies/snowball_2y_gamma_pin_sop)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[1] / "example" / "case_studies"
STUDY_DIR = CASE_DIR / "snowball_2y_gamma_pin_sop"
S2TS_DIR = CASE_DIR / "snowball_2y_tail_stability"
for _p in (STUDY_DIR, S2TS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpin  # noqa: E402
from gpin import PREREG_PATH, load_prereg, prereg_hash  # noqa: E402

# ---------------------------------------------------------------------------
# T1 — pre-registration, package constants, research file
# ---------------------------------------------------------------------------

#: id -> (role, barrier, number of grid points); the spec's technique registry.
EXPECTED_TECHNIQUES = {
    "base": ("baseline", "both", 1),
    "base_noclip": ("control", "both", 1),
    "ko_shift": ("candidate", "ko", 3),
    "ko_spread": ("candidate", "ko", 2),
    "ko_bend": ("candidate", "ko", 3),
    "ki_shift": ("candidate", "ki", 2),
    "ko_wrong": ("control", "ko", 1),
    "ki_wrong": ("control", "ki", 1),
    "secant": ("candidate", "both", 2),
    "ki_cap": ("candidate", "ki", 2),
    "surv_hold": ("candidate", "ko", 2),
    "derisk": ("candidate", "ko", 2),
    "freeze": ("candidate", "ko", 2),
    "mo_ko": ("candidate", "ko", 3),
    "mo_ki": ("candidate", "ki", 2),
    "sop_futures": ("composite", "both", 1),
    "sop_overlay": ("composite", "both", 1),
    "band": ("cost", "both", 2),
}


def test_prereg_lists_exactly_the_registry():
    prereg = load_prereg()
    got = {t["id"]: (t["role"], t["barrier"], len(t["grid"])) for t in prereg["techniques"]}
    assert got == EXPECTED_TECHNIQUES
    assert len(prereg["techniques"]) == len(EXPECTED_TECHNIQUES)  # no duplicate ids


def test_prereg_numbers_match_the_spec():
    p = load_prereg()
    assert p["view"] == "seller"
    assert p["split_date"] == "2025-07-01"
    assert p["data"] == {"start": "2023-05-04", "end": "2026-11-06", "anchor_window": 63}
    assert p["books"] == {"ki75": 0.75, "ki85": 0.85, "ki90": 0.90}
    assert p["sample"] == {"ko_prior_gap": 0.02, "ki_near": 0.03, "ko_notes_per_week": 2, "ki_notes_per_month": 12,
                           "lifted_ki_notes_per_week": 3}
    assert p["windows"] == {"ko_days": 10, "ko_band": 0.10, "ki_band": 0.05, "trigger_z": 2.0,
                            "days_per_year": 244, "overlay_ko_days": 5, "overlay_ki_min_days_left": 3}
    assert p["clocks"] == {"primary": "close", "others": ["twap", "next_open"]}
    assert p["im_clip"] == [-0.25, 3.0]
    t = p["tradability"]
    assert (t["desk_notional"], t["desk_notional_sensitivity"]) == (5e8, 1e8)
    assert (t["min_volume"], t["min_open_interest"], t["max_volume_share"], t["series_limit_lots"]) == (100, 500, 0.10, 1200)
    assert (t["mo_cost_pt"], t["mo_cost_pt_sensitivity"], t["im_cost_rate"]) == (0.2, 1.0, 1e-4)
    m = p["models"]
    assert m["ko_shift_sizes"] == [0.005, 0.01, 0.015, 0.02, 0.03]
    assert m["ki_shifts"] == [-0.01, -0.02]
    assert (m["ko_wrong_size"], m["ki_wrong_shift"], m["bump"]) == (0.01, 0.01, 0.01)
    assert m["ladder"] == {"lo": -0.12, "hi": 0.12, "step": 0.01}
    s = p["statistics"]
    assert (s["es_alpha"], s["bootstrap_n"], s["seed"], s["cluster"], s["p_max"]) == (0.10, 10000, 20261112, "iso_week", 0.10)
    v = p["verdict"]
    assert (v["std_ratio_max"], v["terminal_slack_bp"], v["min_events"], v["min_weeks"]) == (1.05, 10.0, 20, 10)
    assert v["whole_life_set"] == "headline of the same book"
    assert (v["ko_book"], v["ki_book"], v["ki_transfer_books"]) == ("ki75", "ki90", ["ki85", "ki75"])
    assert p["amendments"][0]["before_any_ledger_run"] is True
    assert p["selection"] == {"criterion": "es10_event_window", "clock": "close", "slice": "train", "tie": "lower_cost",
                              "ko_book": "ki75", "ki_book": "ki90"}
    assert [h["id"] for h in p["hypotheses"]] == [f"H{i}" for i in range(1, 9)]


def test_prereg_grids_are_the_registered_points():
    grid = {t["id"]: t["grid"] for t in load_prereg()["techniques"]}
    assert grid["ko_shift"] == [{"size": 0.01}, {"size": 0.02}, {"size": 0.03}]
    assert grid["ko_spread"] == [{"e": 0.015}, {"e": 0.03}]
    assert grid["ko_bend"] == [{"e": 0.015, "n_b": 3}, {"e": 0.03, "n_b": 3}, {"e": 0.03, "n_b": 5}]
    assert grid["ki_shift"] == [{"x": -0.01}, {"x": -0.02}]
    assert grid["secant"] == [{"h": 0.03}, {"h": 0.05}]
    assert grid["ki_cap"] == [{"c": 1.0}, {"c": 1.25}]
    assert grid["surv_hold"] == [{"n_s": 1}, {"n_s": 3}]
    assert grid["derisk"] == [{"n_d": 3}, {"n_d": 5}]
    assert grid["freeze"] == [{"n_f": 1}, {"n_f": 2}]
    assert grid["mo_ko"] == [{"steps": 1, "sizing": "payoff"}, {"steps": 2, "sizing": "payoff"},
                             {"steps": 1, "sizing": "scenario"}]
    assert grid["mo_ki"] == [{"steps": 2, "sizing": "payoff"}, {"steps": 2, "sizing": "scenario"}]
    assert grid["band"] == [{"w": 0.02}, {"w": 0.05}]


def test_spread_widths_are_covered_by_priced_shift_models():
    """A spread of width e averages 4 equally spaced shifts 0..e; each non-zero one must be a priced model."""
    p = load_prereg()
    sizes = set(p["models"]["ko_shift_sizes"])
    for tech in p["techniques"]:
        for point in tech["grid"]:
            if "e" in point:
                for k in (1, 2, 3):
                    assert round(point["e"] * k / 3, 6) in sizes, (tech["id"], point)
            if "size" in point:
                assert point["size"] in sizes


def test_prereg_hash_is_stable_and_content_sensitive(tmp_path):
    h = prereg_hash()
    assert h == prereg_hash() and len(h) == 64
    data = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(dict(reversed(list(data.items()))), indent=4), encoding="utf-8")
    assert prereg_hash(reordered) == h  # key order and whitespace do not matter
    data["split_date"] = "2025-07-02"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    assert prereg_hash(changed) != h


def test_prereg_missing_file_fails_closed(tmp_path):
    with pytest.raises(gpin.StudyDataError):
        load_prereg(tmp_path / "nope.json")


def test_s2ts_constants_this_study_depends_on():
    """Spec D12: a later edit to the sibling study must not silently move this study's numbers."""
    import s2ts
    from s2ts import books, note

    assert (s2ts.RATE, s2ts.ACT, s2ts.NOTIONAL, s2ts.IM_MULT, s2ts.MO_MULT) == (0.014, 365.0, 10_000_000.0, 200.0, 100.0)
    assert (s2ts.MIN_TENOR_DAYS, s2ts.ANCHOR_WINDOW) == (7, 63)
    assert (note.KO_PCT, note.KI_PCT, note.LOCKOUT_MONTHS, note.MATURITY_MONTHS) == (1.00, 0.75, 3, 24)
    assert (note.SPOT_BUMP_REL, note.COUPON_TAIL) == (0.01, "hold_forward")
    assert books.IM_COST_RATE == 1e-4 and books.MO_LEG_COST_PT == 0.2
    assert gpin.TAIL == "converge_hl100"
    p = load_prereg()
    assert p["note"]["notional"] == s2ts.NOTIONAL and p["note"]["rate"] == s2ts.RATE
    assert p["note"]["ko_pct"] == note.KO_PCT and p["note"]["ki_pct"] == note.KI_PCT
    assert p["data"]["anchor_window"] == s2ts.ANCHOR_WINDOW
    assert p["tradability"]["im_cost_rate"] == books.IM_COST_RATE
    assert p["tradability"]["mo_cost_pt"] == books.MO_LEG_COST_PT
    assert p["models"]["bump"] == note.SPOT_BUMP_REL


def test_paths_and_ladder():
    assert gpin.STUDY_DIR == STUDY_DIR
    assert gpin.RESULTS_DIR == STUDY_DIR / "results"
    assert gpin.OUTPUT_DIR == Path("/Users/fuxinyao/quant-ark/example/output/snowball_2y_gamma_pin_sop")
    assert len(gpin.LADDER) == 25
    assert gpin.LADDER[0] == -0.12 and gpin.LADDER[12] == 0.0 and gpin.LADDER[-1] == 0.12
    assert gpin.LADDER[13] == 0.01  # exact decimals, not 0.010000000000000002


def test_research_file_is_complete_and_seller_side():
    text = (STUDY_DIR / "RESEARCH.md").read_text(encoding="utf-8")
    for needle in ("Not verified", "Draft SOP", "References", "seller", "before** any backtest"):
        assert needle in text, needle
    for h in range(1, 9):
        assert f"- H{h}." in text
    assert "holder" not in text.lower()
    hyp = {h["id"] for h in load_prereg()["hypotheses"]}
    assert hyp == {f"H{i}" for i in range(1, 9)}


# ---------------------------------------------------------------------------
# T2 — full CFFEX board, lifecycle scan, sample selection
# ---------------------------------------------------------------------------

import math  # noqa: E402
import time  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import pandas as pd  # noqa: E402

from gpin import HISTORY_DIR, CFFEX_EXTENSION_DIR, SPOT_EXTENSION_PATH, StudyDataError  # noqa: E402
from s2ts.chain import contract_expiry, sse_trading_days  # noqa: E402

CFFEX_HEADER = "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,今收盘,今结算,前结算,涨跌1,涨跌2,Delta\n"
_HIST_CSV = Path(HISTORY_DIR) / "settlement_csv"
_REAL_FIRST = _HIST_CSV / "20230504_1.csv"
_REAL_LAST = Path(CFFEX_EXTENSION_DIR) / "20261106_1.csv"
_REAL_DATA = _REAL_FIRST.exists() and _REAL_LAST.exists() and Path(SPOT_EXTENSION_PATH).exists()
needs_real_data = pytest.mark.skipif(not _REAL_DATA, reason="shared history cache / CFFEX extension absent")


def cffex_payload(rows: list) -> bytes:
    """rows: (code, open, high, low, volume, turnover, oi, close, settle) as strings."""
    lines = [CFFEX_HEADER]
    for code, o, h, l, v, t, oi, c, s in rows:
        lines.append(f"{code:<30},{o},{h},{l},{v},{t},{oi},0,{c},{s},0,0,0,0.5\n")
    return "".join(lines).encode("gbk")


def parse_rows(rows):
    from gpin.board import parse_cffex_full

    return parse_cffex_full(cffex_payload(rows))


def test_parse_full_reads_by_name_and_computes_vwap():
    rows = parse_rows([
        ("IM2611", "6415", "6591.4", "6415", "55768", "7267986.532", "99813", "6500", "6503.6"),
        ("MO2611-C-6000", "500", "628", "483.2", "539", "3078.502", "1702", "544.6", "543"),
        ("HO2611-C-3250", "null", "null", "null", "0", "0", "384", "0.6", "0.6"),  # other family: ignored
    ])
    assert set(rows) == {"IM2611", "MO2611-C-6000"}
    im = rows["IM2611"]
    assert (im.open, im.close, im.settle, im.volume, im.open_interest) == (6415.0, 6500.0, 6503.6, 55768, 99813)
    assert im.vwap == pytest.approx(7267986.532 * 1e4 / (55768 * 200.0), rel=1e-14)
    assert im.low <= im.vwap <= im.high
    mo = rows["MO2611-C-6000"]
    assert mo.vwap == pytest.approx(3078.502 * 1e4 / (539 * 100.0), rel=1e-14)


def test_parse_full_null_fields_and_dropped_rows():
    rows = parse_rows([
        ("MO2611-C-6000", "null", "null", "null", "0", "0", "1702", "544.6", "543"),   # did not trade
        ("MO2611-P-6000", "null", "null", "null", "0", "0", "10", "--", "--"),          # no settlement: dropped
        ("IM2611", "6415", "6591.4", "6415", "55768", "7267986.532", "99813", "null", "6503.6"),
    ])
    assert set(rows) == {"MO2611-C-6000", "IM2611"}
    quiet = rows["MO2611-C-6000"]
    assert quiet.open is None and quiet.high is None and quiet.low is None
    assert quiet.volume == 0 and quiet.vwap is None and quiet.settle == 543.0
    assert rows["IM2611"].close is None and rows["IM2611"].settle == 6503.6


def test_parse_full_changed_header_raises():
    from gpin.board import parse_cffex_full

    good = cffex_payload([("IM2611", "1", "2", "0.5", "10", "5", "9", "1", "1")])
    parse_cffex_full(good)
    for old, new in (("今结算", "结算价"), ("成交金额", "成交额"), ("持仓量", "持仓")):
        with pytest.raises(StudyDataError):
            parse_cffex_full(good.decode("gbk").replace(old, new).encode("gbk"))
    with pytest.raises(StudyDataError):
        parse_cffex_full("a,b,c\n1,2,3\n".encode("gbk"))


@needs_real_data
@pytest.mark.parametrize("path", [_REAL_FIRST, _REAL_LAST])
def test_parse_full_on_real_files(path):
    from gpin.board import parse_cffex_full

    rows = parse_cffex_full(path.read_bytes())
    ims = [r for c, r in rows.items() if c.startswith("IM") and r.volume]
    mos = [r for c, r in rows.items() if c.startswith("MO") and r.volume]
    assert ims and mos
    for r in (ims[0], mos[0]):
        assert None not in (r.open, r.close, r.settle, r.vwap) and r.volume > 0 and r.open_interest >= 0
    for r in ims:
        assert r.low <= r.vwap <= r.high, r.code
    # option turnover is not always exactly lots x trade price (measured: ~3% of thin option rows sit up to 6%
    # outside low-high); the day's most active option is inside
    busiest = max(mos, key=lambda r: r.volume)
    assert busiest.low <= busiest.vwap <= busiest.high, busiest.code


def _opt_rows(series, calls, puts):
    rows = []
    for k, px in calls.items():
        rows.append((f"{series}-C-{k}", "1", "2", "1", "10", "5", "100", px, px))
    for k, px in puts.items():
        rows.append((f"{series}-P-{k}", "1", "2", "1", "10", "5", "100", px, px))
    return rows


def test_series_info_forward_by_put_call_parity():
    from gpin.board import build_day_board

    cal = sse_trading_days(2026, 2028)
    day = date(2026, 10, 21)
    rate = load_prereg()["note"]["rate"]
    rows = _opt_rows("MO2709", {6400: "680", 6500: "610", 6600: "545"}, {6400: "530", 6500: "570", 6600: "610"})
    rows += [("IM2611", "1", "2", "1", "10", "5", "9", "6500", "6500")]
    # MO2712: no strike has both legs, so it gets no SeriesInfo (no invented forward)
    rows += _opt_rows("MO2712", {6500: "500"}, {6600: "400"})
    board = build_day_board(day, 6500.0, parse_rows(rows), cal, rate)
    assert set(board.series) == {"MO2709"}
    info = board.series["MO2709"]
    last = contract_expiry("MO2709", cal)
    assert info.last_trading_day == last.date()
    tenor = (last.date() - day).days / 365.0
    assert info.tenor == pytest.approx(tenor, rel=1e-15)
    assert info.discount == pytest.approx(math.exp(-rate * tenor), rel=1e-15)
    # |C-P| = 150, 40, 65 -> K* = 6500
    assert info.forward == pytest.approx(6500 + (610 - 570) / math.exp(-rate * tenor), rel=1e-14)
    assert set(board.mo["MO2712"]) == {("C", 6500.0), ("P", 6600.0)}
    quote = board.mo["MO2709"][("C", 6500.0)]
    assert (quote.settle, quote.close, quote.volume, quote.open_interest) == (610.0, 610.0, 10, 100)
    assert board.spot == 6500.0 and set(board.im) == {"IM2611"}


def test_series_info_ties_take_the_lower_strike():
    from gpin.board import build_day_board

    cal = sse_trading_days(2026, 2028)
    rows = _opt_rows("MO2709", {6500: "600", 6600: "550"}, {6500: "560", 6600: "510"})  # |C-P| = 40 both
    board = build_day_board(date(2026, 10, 21), 6500.0, parse_rows(rows), cal, 0.014)
    info = board.series["MO2709"]
    assert info.forward == pytest.approx(6500 + 40 / info.discount, rel=1e-14)


@pytest.fixture(scope="module")
def real_chains():
    from s2ts.chain import build_all_chains

    if not _REAL_DATA:
        pytest.skip("shared history cache / CFFEX extension absent")
    return build_all_chains(HISTORY_DIR, CFFEX_EXTENSION_DIR, sse_trading_days(2023, 2030), SPOT_EXTENSION_PATH)


@pytest.fixture(scope="module")
def real_boards(real_chains):
    from gpin.board import build_boards

    return build_boards(real_chains, HISTORY_DIR, CFFEX_EXTENSION_DIR)


def test_boards_cover_exactly_the_chain_days_and_reproduce_the_forward(real_chains, real_boards):
    assert list(real_boards) == [dc.day for dc in real_chains]
    for dc in real_chains:
        board = real_boards[dc.day]
        assert board.spot == dc.spot
        assert board.series[dc.mo.series].forward == pytest.approx(dc.mo.forward, rel=1e-12, abs=1e-9)
        assert board.series[dc.mo.series].tenor == pytest.approx(dc.mo.tenor, rel=1e-15)
        assert dc.im[-1].code in board.im and board.im[dc.im[-1].code].close == dc.im[-1].close


def test_build_boards_refuses_a_missing_day(tmp_path):
    from gpin.board import build_boards
    from s2ts.chain import DayChain

    stub = DayChain(date(2026, 10, 21), 6500.0, (), None, {}, 0.0)
    with pytest.raises(StudyDataError):
        build_boards([stub], tmp_path, tmp_path)  # no CSV there


def test_board_pickle_roundtrip(real_boards, tmp_path):
    from gpin.board import load_boards, write_boards

    first = dict(list(real_boards.items())[:5])
    path = tmp_path / "board.pkl"
    write_boards(first, path)
    assert not list(tmp_path.glob("*.tmp"))
    assert load_boards(path) == first
    with pytest.raises(StudyDataError):
        load_boards(tmp_path / "missing.pkl")


# ---- lifecycle scan on synthetic paths (hand-computed) ----

from collections import namedtuple  # noqa: E402
from types import SimpleNamespace  # noqa: E402

Bar = namedtuple("Bar", "day spot")
S0_SYN = 100.0
START = date(2024, 1, 2)


def trading_days(start: date, end: date) -> list:
    cal = sse_trading_days(start.year, end.year)
    return [d.date() for d in cal if start <= d.date() <= end]


def make_path(end: date, overrides: dict = None, base: float = 90.0, start: date = START) -> list:
    """Bars from ``start`` (spot 100) to ``end``; ``base`` everywhere else unless overridden by day."""
    overrides = overrides or {}
    return [Bar(d, S0_SYN if d == start else overrides.get(d, base)) for d in trading_days(start, end)]


def scan_one(path, start: date = START):
    from gpin.events import scan

    scans = scan(path, {start: 0.0})
    return next(s for s in scans if s.inception == start)


def life_outcome(path, inception: date = START):
    """(ko_date, ki_date, end_reason, end_date) from the library's own tracker, no pricing."""
    from s2ts.note import NoteLife, build_note, build_terms, make_env

    life = NoteLife(build_note(0.2, path[0].spot, build_terms(inception)), inception)
    ko = ki = None
    reason, end = "truncated", path[-1].day
    for bar in path:
        if bar.day < inception:
            continue
        dc = SimpleNamespace(day=bar.day, spot=bar.spot, mo=SimpleNamespace(vol=0.2))
        nd = life.step(dc, make_env(dc, None))
        if nd.knocked_in and ki is None:
            ki = bar.day
        if not nd.alive:
            ko, reason, end = (bar.day if nd.end_reason == "ko" else None), nd.end_reason, bar.day
            break
    return ko, ki, reason, end


def assert_scan_matches_library(path):
    sc = scan_one(path)
    ko, ki, reason, end = life_outcome(path)
    assert (sc.ko_date, sc.ki_date, sc.end_reason, sc.end_date) == (ko, ki, reason, end)
    return sc


def ko_dates_of(start: date = START):
    from s2ts.note import build_terms

    return build_terms(start).ko_dates


def test_scan_ko_fires_at_the_first_eligible_observation_on_equality():
    ko1 = ko_dates_of()[0]
    days = trading_days(START, ko1)
    prev = days[-2]
    # a pile of spot above the barrier in the lockout must not fire; S(T) == B fires (>=)
    path = make_path(ko1 + timedelta(days=20), {days[5]: 120.0, prev: 99.0, ko1: 100.0})
    sc = assert_scan_matches_library(path)
    assert sc.ko_date == ko1 and sc.end_reason == "ko" and sc.end_date == ko1 and sc.ki_date is None
    assert sc.ko_observations == 1
    (ev,) = sc.ko_events
    assert (ev.inception, ev.obs_date, ev.fired) == (START, ko1, True)
    assert ev.prior_gap == pytest.approx(0.01, rel=1e-12)
    assert sc.ki_approaches == ()
    assert sc.anchored


def test_scan_survived_pin_observation_and_a_far_one():
    kos = ko_dates_of()
    d1 = trading_days(START, kos[0])
    d2 = trading_days(kos[0], kos[1])
    # obs 1: prior close 98 (gap exactly 2%, a pin), S(T) = 99.99 -> survives
    # obs 2: prior close 95 (gap 5%, not a pin), S(T) = 90
    path = make_path(kos[1] + timedelta(days=5), {d1[-2]: 98.0, kos[0]: 99.99, d2[-2]: 95.0})
    sc = assert_scan_matches_library(path)
    assert sc.ko_observations == 2 and sc.ko_date is None and sc.end_reason == "truncated"
    (ev,) = sc.ko_events
    assert ev.obs_date == kos[0] and not ev.fired
    assert ev.prior_gap == pytest.approx(0.02, rel=1e-9)


def test_scan_knock_in_then_knock_out_and_barrier_equalities():
    kos = ko_dates_of()
    days = trading_days(START, kos[1])
    ki_day = days[10]
    # S == 75 knocks in (<=); the dip is one day, spot recovers, the note survives obs 1 and is called at obs 2
    path = make_path(kos[1] + timedelta(days=3), {ki_day: 75.0, kos[1]: 101.0, trading_days(kos[0], kos[1])[-2]: 99.0})
    sc = assert_scan_matches_library(path)
    assert sc.ki_date == ki_day and sc.ko_date == kos[1] and sc.end_reason == "ko"
    assert sc.ko_observations == 2
    # a spot a hair above the barrier does not knock in
    sc2 = assert_scan_matches_library(make_path(kos[0], {ki_day: 75.0000001}))
    assert sc2.ki_date is None


def test_scan_knock_in_to_maturity_and_ko_on_the_maturity_day():
    from s2ts.note import build_terms

    terms = build_terms(START)
    days = trading_days(START, terms.maturity_date)
    dip = {days[3]: 70.0}
    sc = assert_scan_matches_library(make_path(terms.maturity_date, dip))
    assert (sc.ki_date, sc.ko_date, sc.end_reason, sc.end_date) == (days[3], None, "maturity", terms.maturity_date)
    assert sc.ko_observations == 22 and sc.ko_events == ()
    # the maturity day is also the last KO observation: spot on the barrier is a KO, not a maturity
    sc = assert_scan_matches_library(make_path(terms.maturity_date, {terms.maturity_date: 100.0}))
    assert sc.end_reason == "ko" and sc.ko_date == terms.maturity_date
    # a note with data ending the day before maturity is truncated, and one whose data ends later is not extended
    sc = scan_one(make_path(days[-2]))  # (scan only: the library comparison of a full 2Y path costs ~10 s)
    assert sc.end_reason == "truncated" and sc.end_date == days[-2] and sc.ko_date is None
    assert sc.ko_observations == 21  # observations 1..21; the 22nd (maturity) is not in the data


def test_ki_approach_two_dips_in_one_run_are_one_approach():
    days = trading_days(START, START + timedelta(days=90))
    # run (<= 78.75, i.e. S/K - 1 <= 5% for K = 75): 78, 77, 78, 77, 78 then 90 ends it; 78.75 is on the band
    run = {days[5]: 78.75, days[6]: 78.0, days[7]: 77.0, days[8]: 78.0, days[9]: 77.0, days[10]: 78.0}
    # a second run that touches the 3% line (77.25 -> 77.0 is inside)
    run.update({days[20]: 77.0})
    # a third run that stays between 3% and 5%: no approach
    run.update({days[30]: 78.0, days[31]: 78.5})
    sc = assert_scan_matches_library(make_path(days[-1], run))
    assert [(a.start, a.end, a.knocked_in) for a in sc.ki_approaches] == [
        (days[5], days[10], False), (days[20], days[20], False)]
    assert sc.ki_date is None


def test_ki_approach_ended_by_a_knock_in_is_dated_by_the_knock_in_day():
    days = trading_days(START, START + timedelta(days=90))
    path = make_path(days[-1], {days[5]: 78.0, days[6]: 76.0, days[7]: 74.0, days[8]: 73.0, days[9]: 76.0})
    sc = assert_scan_matches_library(path)
    assert sc.ki_date == days[7]
    assert [(a.start, a.end, a.knocked_in) for a in sc.ki_approaches] == [(days[5], days[7], True)]  # none after the KI
    # a run that touches 3% only on its first day is still an approach; one at the data end is closed there
    sc = assert_scan_matches_library(make_path(days[12], {days[11]: 77.0, days[12]: 78.0}))
    assert [(a.start, a.end, a.knocked_in) for a in sc.ki_approaches] == [(days[11], days[12], False)]


def test_scan_one_record_per_trading_day_with_anchored_flag():
    from gpin.events import scan

    days = trading_days(START, START + timedelta(days=40))
    path = [Bar(d, S0_SYN) for d in days]
    scans = scan(path, {days[2]: 0.0, days[5]: 0.0})
    assert [s.inception for s in scans] == days
    assert [s.inception for s in scans if s.anchored] == [days[2], days[5]]
    assert all(s.end_reason == "truncated" and s.ko_observations == 0 for s in scans)
    assert scans[-1].end_date == days[-1] and scans[-1].ko_events == ()
    gap = path[:3] + path[4:]
    with pytest.raises(StudyDataError):
        scan(gap, {})  # a missing trading day fails closed
    with pytest.raises(StudyDataError):
        scan([Bar(date(2024, 1, 6), 100.0)], {})  # a Saturday is not a trading day


def test_scan_reads_its_thresholds_from_the_preregistration():
    from gpin import events

    p = load_prereg()
    assert (events.KO_PRIOR_GAP, events.KI_NEAR, events.KI_BAND) == (
        p["sample"]["ko_prior_gap"], p["sample"]["ki_near"], p["windows"]["ki_band"])


# ---- slices, week keys, selection ----


def test_slice_of_and_week_key():
    from gpin.events import slice_of, week_key

    assert slice_of(date(2025, 6, 30)) == "train" and slice_of(date(2025, 7, 1)) == "test"
    assert slice_of(date(2023, 5, 4)) == "train" and slice_of(date(2026, 11, 6)) == "test"
    assert week_key(date(2025, 12, 29)) == "2026-W01"      # ISO week-year, not calendar year
    assert week_key(date(2024, 1, 1)) == "2024-W01" and week_key(date(2023, 1, 1)) == "2022-W52"


def fake_scan(inc, *, anchored=True, ko=(), ki=(), n_ko=0, ki_date=None, ki_pct=0.75):
    from gpin.events import KiApproach, KoEvent, NoteScan

    return NoteScan(
        inception=inc, anchored=anchored, ko_date=None, ki_date=ki_date, end_reason="truncated",
        end_date=date(2026, 11, 6), ko_observations=n_ko,
        ko_events=tuple(KoEvent(inc, d, g, f) for d, g, f in ko),
        ki_approaches=tuple(KiApproach(inc, s, e, k) for s, e, k in ki), ki_pct=ki_pct)


def test_select_ko_two_smallest_gaps_per_iso_week_ties_to_earlier_inception():
    from gpin.events import select

    t = date(2025, 3, 5)  # Wednesday of 2025-W10
    scans = [
        fake_scan(date(2024, 1, 4), ko=[(t, 0.010, True)]),
        fake_scan(date(2024, 1, 3), ko=[(t, 0.010, False)]),   # ties with the above; earlier inception wins
        fake_scan(date(2024, 1, 5), ko=[(t, 0.005, True)]),
        fake_scan(date(2024, 1, 8), ko=[(t, 0.019, True)]),
        fake_scan(date(2024, 1, 2), anchored=False, ko=[(t, 0.001, True)]),   # unanchored: never selected
        fake_scan(date(2024, 2, 1), ko=[(date(2025, 3, 12), 0.015, False)]),  # next week, alone
    ]
    sample = select(scans)
    assert sample.pin == (date(2024, 1, 3), date(2024, 1, 5), date(2024, 2, 1))
    assert sample.reasons[date(2024, 1, 5)] == ("ko 2025-W10 rank 1",)
    assert sample.reasons[date(2024, 1, 3)] == ("ko 2025-W10 rank 2",)
    assert date(2024, 1, 2) not in sample.pin and date(2024, 1, 2) not in sample.headline


def test_select_ki_twelve_evenly_spaced_notes_per_month():
    from gpin.events import select

    incs = [date(2024, 1, 2) + timedelta(days=i) for i in range(30)]
    scans = [fake_scan(i, ki=[(date(2024, 2, 1), date(2024, 2, 9), False)]) for i in incs]
    scans += [fake_scan(date(2023, 6, 1), ki=[(date(2024, 3, 4), date(2024, 3, 5), True)], ki_date=date(2024, 3, 5))]
    scans += [fake_scan(date(2023, 6, 2) + timedelta(days=i), ki=[(date(2024, 4, 2), date(2024, 4, 3), False)])
              for i in range(5)]
    sample = select(scans)
    positions = [0, 3, 5, 8, 11, 13, 16, 18, 21, 24, 26, 29]
    feb = [i for i in sample.pin if i >= date(2024, 1, 2)]
    assert feb == [incs[p] for p in positions]
    assert date(2023, 6, 1) in sample.pin                      # a month with fewer than 12 notes keeps all of them
    assert sum(1 for i in sample.pin if date(2023, 6, 2) <= i <= date(2023, 6, 6)) == 5
    assert sample.reasons[incs[3]] == ("ki 2024-02 position 4 of 30",)


def test_select_is_deterministic_headline_is_first_anchored_day_of_a_quarter_and_events_are_complete():
    import random

    from gpin.events import select

    q1 = [date(2024, 1, 2), date(2024, 1, 3)]
    scans = [
        fake_scan(q1[0], anchored=False),
        fake_scan(q1[1]),                        # first day of the quarter is unanchored: no headline for Q1
        fake_scan(date(2024, 4, 1), ko=[(date(2024, 7, 2), 0.01, True), (date(2024, 8, 1), 0.015, False)],
                  ki=[(date(2024, 5, 6), date(2024, 5, 8), False)]),
        fake_scan(date(2024, 4, 2), ko=[(date(2024, 7, 3), 0.02, False)]),
        fake_scan(date(2024, 7, 1)),
    ]
    sample = select(scans)
    assert sample.headline == (date(2024, 4, 1), date(2024, 7, 1))
    # events: every KO pin event and every KI approach of every selected note (headline or pin), nothing else
    got = {(type(e).__name__, e.inception) for e in sample.events}
    assert got == {("KoEvent", date(2024, 4, 1)), ("KiApproach", date(2024, 4, 1)), ("KoEvent", date(2024, 4, 2))}
    assert len([e for e in sample.events if type(e).__name__ == "KoEvent"]) == 3  # two of 04-01 (the week cap does not thin events), one of 04-02
    shuffled = list(scans)
    random.Random(1).shuffle(shuffled)
    assert select(shuffled) == sample
    assert select(scans) == sample
    union = set(sample.headline) | set(sample.pin)
    assert {e.inception for e in sample.events} <= union


# ---- real data: counts of the Decisions block, the headline set, the caps, the library's lifecycle ----


@pytest.fixture(scope="module")
def real_anchors(real_chains):
    from s2ts.tails import anchor_series

    return anchor_series(real_chains, load_prereg()["data"]["anchor_window"])


@pytest.fixture(scope="module")
def real_scans(real_chains, real_anchors):
    from gpin.events import scan

    t0 = time.perf_counter()
    scans = scan(real_chains, real_anchors)
    assert time.perf_counter() - t0 < 60.0  # the pricing-free scan is a few seconds
    return scans


def test_unrestricted_counts_equal_the_decisions_block(real_scans):
    from gpin.events import counts

    n = counts(real_scans)
    assert n["inceptions"] == 852
    assert n["ko_observations"] == 3462        # live KO observations over all 852 inceptions
    assert n["ko_pin_events"] == 307           # prior close within 2% of the barrier
    assert n["ko_pin_fired"] == 145
    assert n["notes_knocked_in"] == 209
    weeks = {week_key_of(e.obs_date) for s in real_scans for e in s.ko_events}
    assert len(weeks) == 79                    # distinct ISO weeks
    assert sum(1 for s in real_scans for e in s.ko_events if not e.fired) == 162


def week_key_of(d):
    from gpin.events import week_key

    return week_key(d)


def test_anchor_restriction_drops_exactly_the_early_days(real_chains, real_anchors, real_scans):
    from gpin.events import counts

    unanchored = [s.inception for s in real_scans if not s.anchored]
    assert len(unanchored) == 62 and unanchored[-1] < min(real_anchors)  # 62 days before the first 63-day anchor
    assert min(real_anchors) == date(2023, 8, 2)
    restricted = counts([s for s in real_scans if s.anchored])
    assert restricted["inceptions"] == 852 - 62


def test_headline_is_the_tail_stability_studys_inception_set(real_chains, real_anchors, real_scans):
    from gpin.events import select
    from s2ts.note import inception_dates

    assert list(select(real_scans).headline) == inception_dates(real_chains, real_anchors)


def test_select_on_real_data_obeys_the_caps_and_is_deterministic(real_scans):
    from gpin.events import KO_NOTES_PER_WEEK, KI_NOTES_PER_MONTH, KoEvent, select, week_key

    sample = select(real_scans)
    again = select(list(reversed(real_scans)))
    assert again == sample
    anchored = {s.inception for s in real_scans if s.anchored}
    assert set(sample.headline) <= anchored and set(sample.pin) <= anchored
    assert set(sample.reasons) == set(sample.pin)
    per_week, per_month = {}, {}
    for inc, why in sample.reasons.items():
        for r in why:
            kind, key = r.split()[0], r.split()[1]
            (per_week if kind == "ko" else per_month).setdefault(key, set()).add(inc)
    assert per_week and per_month
    assert max(len(v) for v in per_week.values()) <= KO_NOTES_PER_WEEK
    assert max(len(v) for v in per_month.values()) <= KI_NOTES_PER_MONTH
    by_inc = {s.inception: s for s in real_scans}
    want = sum(len(by_inc[i].ko_events) + len(by_inc[i].ki_approaches) for i in set(sample.headline) | set(sample.pin))
    assert len(sample.events) == want
    assert all(isinstance(e, KoEvent) is hasattr(e, "obs_date") for e in sample.events)
    assert {week_key(e.obs_date) for e in sample.events if isinstance(e, KoEvent)}


def test_scan_agrees_with_the_library_lifecycle_on_real_notes(real_chains, real_scans):
    """Price-free cross-check on three short real lives: KI then KO, KO only, a truncated knock-in (maturities: the synthetic paths)."""
    by_day = {dc.day: dc for dc in real_chains}

    def shortest(pred):  # lifecycle needs no anchor: every scanned inception is checked, not only the priceable ones
        return min((s for s in real_scans if pred(s)), key=lambda s: (s.end_date - s.inception).days)

    picks = [
        shortest(lambda s: s.ki_date and s.ko_date),
        shortest(lambda s: s.ko_date and not s.ki_date),
        shortest(lambda s: s.end_reason == "truncated" and s.ki_date),
    ]
    assert [p.end_reason for p in picks] == ["ko", "ko", "truncated"]
    for s in picks:
        path = [Bar(d, by_day[d].spot) for d in sorted(by_day) if d >= s.inception]
        assert (s.ko_date, s.ki_date, s.end_reason, s.end_date) == life_outcome(path, s.inception)[:4], s.inception


# ---- results/events.json ----

import importlib.util  # noqa: E402

EVENTS_JSON = STUDY_DIR / "results" / "events.json"


def _load_stage(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_stage"), STUDY_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def book_scans(real_chains, real_anchors, real_scans):
    from gpin.events import BOOKS, scan

    return {b: (real_scans if b == "ki75" else scan(real_chains, real_anchors, ki_pct=lv)) for b, lv in BOOKS.items()}


def test_events_json_matches_a_fresh_scan_and_lists_every_event_per_book(book_scans):
    from gpin.events import BOOKS, select

    stage = _load_stage("02_events.py")
    samples = {b: select(book_scans[b], book=b) for b in BOOKS}
    fresh = json.loads(json.dumps(stage.build_payload(book_scans, samples), sort_keys=True))
    on_disk = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    assert on_disk == fresh                                   # the committed file is the current scan
    assert on_disk["preregistration_hash"] == prereg_hash()
    assert on_disk["split_date"] == "2025-07-01"
    assert on_disk["unrestricted"]["ko_pin_fired"] == 145 and on_disk["unrestricted"]["notes_knocked_in"] == 209
    assert set(on_disk["books"]) == set(BOOKS)
    for b, block in on_disk["books"].items():
        sample = samples[b]
        assert block["ki_pct"] == BOOKS[b]
        ko_rows = block.get("ko_events", [])
        assert len(ko_rows) + len(block["ki_approaches"]) == len(sample.events)
        for row in ko_rows + block["ki_approaches"]:
            assert row["slice"] == ("train" if row["date"] < "2025-07-01" else "test")
            assert row["week"] == week_key_of(date.fromisoformat(row["date"]))
        assert {r["inception"] for r in ko_rows + block["ki_approaches"]} <= set(block["notes"])
        assert all(p["why"] for p in block["pin"])
        sizes = block["sample_sizes"]
        assert sizes["headline_notes"] == len(sample.headline) and sizes["pin_notes"] == len(sample.pin)
        assert sizes["train"]["ki_approaches"] + sizes["test"]["ki_approaches"] == len(block["ki_approaches"])
        r = block["restricted"]
        assert r["train"]["ki_approaches"] + r["test"]["ki_approaches"] == r["ki_approaches"]
    assert "ko_events" in on_disk["books"]["ki75"] and "ko_events" not in on_disk["books"]["ki85"]
    assert "ko_events" not in on_disk["books"]["ki90"]
    assert not any("ko " in w for p in on_disk["books"]["ki90"]["pin"] for w in p["why"])
    # AC4b: the decidable KI test slice
    k90 = on_disk["books"]["ki90"]["restricted"]["test"]
    assert k90["ki_approaches"] >= 20 and k90["ki_weeks"] >= 10
    assert "holder" not in EVENTS_JSON.read_text(encoding="utf-8").lower()


def test_lifted_books_cover_every_week_of_their_restricted_test_universe():
    """The per-week cap cannot drop a week: every ISO week with an approach in the restricted universe has a
    selected note's approach in it, in each slice."""
    on_disk = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    for book in ("ki85", "ki90"):
        blk = on_disk["books"][book]
        for sl in ("train", "test"):
            assert blk["sample_sizes"][sl]["ki_weeks"] == blk["restricted"][sl]["ki_weeks"], (book, sl)
    assert on_disk["books"]["ki90"]["sample_sizes"]["test"]["ki_weeks"] >= 13
    assert all(r.split()[1].count("-W") == 1 for p in on_disk["books"]["ki90"]["pin"] for r in p["why"])


def test_ki75_sample_is_unchanged_by_the_books(real_scans, book_scans):
    """Adding books does not touch the product's scan or sample: it is the same call, the same records."""
    from gpin.events import select

    assert book_scans["ki75"] is real_scans
    assert select(real_scans) == select(real_scans, book="ki75")
    for b in ("ki85", "ki90"):
        assert [(s.inception, s.ko_date, s.ko_events, s.ko_observations) for s in book_scans[b]] == \
            [(s.inception, s.ko_date, s.ko_events, s.ko_observations) for s in real_scans]   # KO side is book-free


# ---------------------------------------------------------------------------
# T3 — hedge models and the pricing pass
# ---------------------------------------------------------------------------

import copy  # noqa: E402

PROBE_INCEPTION = date(2024, 2, 29)
PROBE_DAY = date(2024, 5, 28)       # one day before the 2024-05-29 KO observation, spot 0.06% above the barrier
PROBE_COUPON = 0.15                 # the coupon the spec's probe numbers came from
S0_FIXED = 6000.0


def _models():
    from gpin import models

    return models


def test_model_registry_is_built_from_the_preregistration():
    m = _models()
    assert list(m.MODELS) == ["true", "surv", "ko_s050", "ko_s100", "ko_s150", "ko_s200", "ko_s300", "ko_w100",
                              "ki_m100", "ki_m200", "ki_p100", "kin"]
    window = {k: v.window for k, v in m.MODELS.items()}
    assert window["true"] == "always"
    assert {k for k, w in window.items() if w == "ko"} == {"surv", "ko_s050", "ko_s100", "ko_s150", "ko_s200",
                                                          "ko_s300", "ko_w100"}
    assert {k for k, w in window.items() if w == "ki"} == {"ki_m100", "ki_m200", "ki_p100", "kin"}
    assert [m.MODELS[k].ko_shift for k in ("ko_s050", "ko_s100", "ko_s150", "ko_s200", "ko_s300", "ko_w100")] == \
        [0.005, 0.01, 0.015, 0.02, 0.03, -0.01]
    assert [m.MODELS[k].ki_shift for k in ("ki_m100", "ki_m200", "ki_p100")] == [-0.01, -0.02, 0.01]
    assert m.MODELS["surv"].disable_next_ko and m.MODELS["kin"].force_knocked_in
    assert not m.MODELS["true"].disable_next_ko and not m.MODELS["true"].force_knocked_in


def test_ko_levels_are_hand_computed_for_both_directions():
    from s2ts.note import build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    n = len(terms.ko_dates)
    assert terms.ko_dates[0] == date(2024, 5, 29) and n == 22
    day = date(2024, 5, 28)
    assert m.ko_levels(m.MODELS["true"], terms, day, -1, s0=S0_FIXED) == [S0_FIXED] * n
    for size_id, size in (("ko_s050", 0.005), ("ko_s100", 0.01), ("ko_s300", 0.03)):
        down = m.ko_levels(m.MODELS[size_id], terms, day, -1, s0=S0_FIXED)
        up = m.ko_levels(m.MODELS[size_id], terms, day, +1, s0=S0_FIXED)
        assert down == pytest.approx([S0_FIXED * (1 - size)] * n, rel=1e-15)
        assert up == pytest.approx([S0_FIXED * (1 + size)] * n, rel=1e-15)
    # the wrong-way control moves the booked level the other way
    assert m.ko_levels(m.MODELS["ko_w100"], terms, day, -1, s0=S0_FIXED) == pytest.approx([S0_FIXED * 1.01] * n, rel=1e-15)
    assert m.ko_levels(m.MODELS["ko_w100"], terms, day, +1, s0=S0_FIXED) == pytest.approx([S0_FIXED * 0.99] * n, rel=1e-15)
    # relative levels by default (strike = 1)
    assert m.ko_levels(m.MODELS["ko_s200"], terms, day, -1) == pytest.approx([0.98] * n, rel=1e-15)
    # a model that does not shift the KO barrier ignores the direction (and needs none)
    assert m.ko_levels(m.MODELS["ki_m100"], terms, day, None, s0=S0_FIXED) == [S0_FIXED] * n


def test_surv_disables_exactly_the_next_observation():
    from s2ts.note import build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    off = m.SURV_OFF_MULTIPLE * S0_FIXED
    for day, idx in ((date(2024, 5, 28), 0), (date(2024, 5, 15), 0),
                     (terms.ko_dates[0], 1),                       # the observation day itself is already processed
                     (terms.ko_dates[0] + timedelta(days=1), 1),
                     (terms.ko_dates[4], 5)):
        lv = m.ko_levels(m.MODELS["surv"], terms, day, None, s0=S0_FIXED)
        assert [i for i, x in enumerate(lv) if x == off] == [idx], day
        assert all(x == S0_FIXED for i, x in enumerate(lv) if i != idx)


def test_shift_models_require_a_direction():
    from s2ts.note import build_terms
    from quantark.util.exceptions import ValidationError

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    for mid in ("ko_s100", "ko_w100"):
        for bad in (None, 0):
            with pytest.raises(ValidationError):
                m.ko_levels(m.MODELS[mid], terms, PROBE_DAY, bad, s0=S0_FIXED)


def test_ki_level_is_hand_computed():
    m = _models()
    assert m.ki_level(m.MODELS["true"], S0_FIXED) == pytest.approx(0.75 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m100"], S0_FIXED) == pytest.approx(0.75 * 0.99 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m200"], S0_FIXED) == pytest.approx(0.75 * 0.98 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_p100"], S0_FIXED) == pytest.approx(0.75 * 1.01 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["kin"], S0_FIXED) == pytest.approx(0.75 * S0_FIXED, rel=1e-15)


def test_product_carries_the_levels_and_the_true_product_is_the_note():
    from s2ts.note import build_note, build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    note = build_note(PROBE_COUPON, S0_FIXED, terms)
    true = m.MODELS["true"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, None)
    # build_note books one scalar level, a model one level per observation (the same schedule)
    assert list(true.barrier_config.ko_barrier) == [note.barrier_config.ko_barrier] * 22
    assert true.barrier_config.ki_barrier == note.barrier_config.ki_barrier
    assert true.maturity == note.maturity and true.contract_multiplier == note.contract_multiplier
    shifted = m.MODELS["ko_s100"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, -1)
    assert list(shifted.barrier_config.ko_barrier) == pytest.approx([S0_FIXED * 0.99] * 22, rel=1e-15)
    ki = m.MODELS["ki_m200"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, None)
    assert ki.barrier_config.ki_barrier == pytest.approx(0.75 * 0.98 * S0_FIXED, rel=1e-15)


def test_kin_lifecycle_is_a_forced_copy_and_every_other_model_gets_the_state_itself():
    from s2ts.note import NoteLife, build_note, build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    life = NoteLife(build_note(PROBE_COUPON, S0_FIXED, terms), PROBE_INCEPTION)
    state = life.lifecycle
    assert state.knocked_in is False
    before = copy.deepcopy(state)
    forced = m.MODELS["kin"].lifecycle(state)
    assert forced is not state and forced.knocked_in is True
    assert state.knocked_in is False                               # the tracker's own state is untouched
    assert {k: v for k, v in vars(state).items() if k != 'ledger'} == {k: v for k, v in vars(before).items() if k != 'ledger'}
    for mid, model in m.MODELS.items():
        if mid != "kin":
            assert model.lifecycle(state) is state
    state.knocked_in = True                                        # an already knocked-in state stays so
    assert m.MODELS["kin"].lifecycle(state).knocked_in is True


def test_ko_direction_follows_the_gap_and_is_kept_for_the_observation():
    from gpin.pricing import KoDirection

    kd = KoDirection()
    obs = date(2024, 5, 29)
    assert kd.resolve(obs, 419.0) == -1            # redemption costs the seller: shift the level down
    assert kd.resolve(obs, -50.0) == -1            # kept even when the gap flips sign inside the window
    assert kd.resolve(date(2024, 6, 28), -50.0) == +1
    assert kd.resolve(date(2024, 7, 29), 0.0) == 0   # a zero gap on the first day prices no shift model
    assert kd.resolve(date(2024, 7, 29), 80.0) == 0
    assert kd.resolve(date(2024, 5, 29), 1.0) == -1  # re-entering the window keeps the first day's choice


def test_pricing_fingerprint_covers_pricing_code_only(tmp_path):
    import shutil

    from gpin import pricing

    pkg = tmp_path / "gpin"
    shutil.copytree(STUDY_DIR / "gpin", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    base = pricing.pricing_fingerprint(pkg_dir=pkg, data_digest="d")
    assert base == pricing.pricing_fingerprint(pkg_dir=pkg, data_digest="d")
    for ledger_side in ("black.py", "ledger.py", "techniques.py", "measures.py", "report.py"):
        (pkg / ledger_side).write_text("x = 1\n")
        (pkg / ledger_side).write_text("x = 2\n")
        assert pricing.pricing_fingerprint(pkg_dir=pkg, data_digest="d") == base, ledger_side
    (pkg / "pricing.py").write_text((pkg / "pricing.py").read_text() + "\n# edit\n")
    changed = pricing.pricing_fingerprint(pkg_dir=pkg, data_digest="d")
    assert changed != base
    assert pricing.pricing_fingerprint(pkg_dir=pkg, data_digest="e") != changed


# ---- engine-touching tests: a handful of priced days on the probe note (each solve is about 0.45 s) ----

import numpy as np  # noqa: E402

PROBE_FIRST_WINDOW_DAY = date(2024, 5, 15)     # 10 trading days before the 2024-05-29 observation
PROBE_MID_DAY = date(2024, 4, 10)              # far from both windows
PROBE_KO_DAY = date(2024, 5, 29)
LADDER_CENTRE = 12


@pytest.fixture(scope="module")
def engine():
    from s2ts.runner import v2_engine

    return v2_engine()


@pytest.fixture(scope="module")
def probe_scan(real_scans):
    return next(s for s in real_scans if s.inception == PROBE_INCEPTION)


@pytest.fixture(scope="module")
def probe_states(real_chains, real_anchors, probe_scan, engine):
    from gpin.pricing import price_note

    days = {PROBE_MID_DAY, PROBE_FIRST_WINDOW_DAY, PROBE_DAY}
    states = price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan,
                        engine=engine, price_days=days)
    return states, {r["date"]: r for r in states.days}


def _step_life(real_chains, real_anchors, inception, coupon, until, ki_pct=0.75):
    """Step a NoteLife to ``until`` as the pricing pass does; returns (life, dc, env, nd, curve)."""
    from quantark.param import FlatRateCurve
    from s2ts import RATE
    from s2ts.note import NoteLife, build_note, build_terms, make_env
    from s2ts.tails import build_curve

    i0 = next(i for i, c in enumerate(real_chains) if c.day == inception)
    from gpin.models import book_note

    life = NoteLife(book_note(coupon, real_chains[i0].spot, build_terms(inception), ki_pct), inception)
    for dc in real_chains[i0:]:
        curve = build_curve(dc, gpin.TAIL, real_anchors.get(dc.day))
        env = make_env(dc, curve.to_dividend_yield(FlatRateCurve(RATE)))
        nd = life.step(dc, env)
        if dc.day == until:
            return life, dc, env, nd, curve
    raise AssertionError(f"{until} not reached")


def _bump_delta(states_one, spot):
    """Central 1% bump delta (CNY per index point) of a recorded ladder."""
    s = spot * (1.0 + np.asarray(gpin.LADDER))
    p = states_one["engine_price"]
    return (p[LADDER_CENTRE + 1] - p[LADDER_CENTRE - 1]) / (s[LADDER_CENTRE + 1] - s[LADDER_CENTRE - 1])


def test_probe_gap_ko_direction_and_hedges(probe_states, real_chains):
    from s2ts import NOTIONAL

    _, by = probe_states
    r = by[PROBE_DAY.isoformat()]
    assert r["next_ko"] == "2024-05-29" and r["days_to_ko"] == 1
    assert r["gap_ko"] > 0 and r["ko_dir"] == -1
    # KO cash (369.86 bp, one day of discounting) less the survived price at B, interpolated on the ladder. The
    # spec's "370 - (-49) = 419" read the survived price at the spot, 0.06% above B with a delta of 82% of
    # notional: at B itself it is 5 bp lower, so the gap is 424 bp.
    s0 = next(c.spot for c in real_chains if c.day == PROBE_INCEPTION)
    spots = r["spot"] * (1.0 + np.asarray(gpin.LADDER))
    surv_at_b = float(np.interp(s0, spots, r["states"]["surv"]["engine_price"]))
    cash = 369.86e-4 * NOTIONAL * math.exp(-0.014 / 365.0)
    assert r["gap_ko"] == pytest.approx(cash - surv_at_b, abs=NOTIONAL * 1e-4 * 0.01)   # within 0.01 bp
    assert round(1e4 * r["gap_ko"] / NOTIONAL) == 424 and round(-1e4 * surv_at_b / NOTIONAL) == 54
    # the spec's 193% / 82%: the ladder's point delta at the spot (engine delta x spot / notional)
    point = {m: 100.0 * r["states"][m]["engine_delta"][LADDER_CENTRE] * r["spot"] / NOTIONAL for m in ("true", "surv")}
    assert (round(point["true"]), round(point["surv"])) == (193, 82)
    # the central 1% bump delta (D4's baseline hedge) straddles the KO barrier 0.06% below the spot: 176% for the
    # true note, 82% for the survived branch
    bump = {m: 100.0 * _bump_delta(r["states"][m], r["spot"]) * r["spot"] / NOTIONAL for m in ("true", "surv")}
    assert (round(bump["true"]), round(bump["surv"])) == (176, 82)
    assert set(r["states"]) == {"true", "surv", "ko_s050", "ko_s100", "ko_s150", "ko_s200", "ko_s300", "ko_w100"}


def test_probe_keys_and_units(probe_states):
    states, by = probe_states
    keys = {"date", "spot", "alive", "knocked_in", "engine_note_cash", "end_reason", "next_ko", "days_to_ko",
            "gap_ko", "ko_dir", "states"}
    for r in states.days:
        assert keys <= set(r) <= keys | {"unpriced"}
    one = by[PROBE_DAY.isoformat()]["states"]["true"]
    assert set(one) == {"engine_price", "engine_delta", "engine_gamma"}
    assert all(len(v) == len(gpin.LADDER) == 25 for v in one.values())
    json.dumps(states.to_json("fp"), allow_nan=False)           # JSON-ready, no NaN


def test_models_outside_their_window_have_no_state_and_unpriced_days_are_flagged(probe_states):
    _, by = probe_states
    mid = by[PROBE_MID_DAY.isoformat()]
    assert set(mid["states"]) == {"true"} and mid["gap_ko"] is None and mid["ko_dir"] is None
    assert mid["next_ko"] == "2024-05-29" and mid["days_to_ko"] > 10
    other = by["2024-04-11"]
    assert other["states"] == {} and other["unpriced"] is True and "unpriced" not in mid


def test_ko_dir_is_kept_across_one_observations_window(probe_states):
    _, by = probe_states
    first, last = by[PROBE_FIRST_WINDOW_DAY.isoformat()], by[PROBE_DAY.isoformat()]
    assert first["next_ko"] == last["next_ko"] and first["days_to_ko"] == 10
    assert first["ko_dir"] == last["ko_dir"] == -1
    assert first["gap_ko"] != last["gap_ko"]                      # gap_ko is refreshed every window day


def test_the_pass_ends_on_the_ko_day_with_the_lifecycle_cash(probe_states):
    states, by = probe_states
    ko = by[PROBE_KO_DAY.isoformat()]
    assert ko["alive"] is False and ko["end_reason"] == "ko" and ko["states"] == {}
    assert ko["gap_ko"] is None and ko["next_ko"] is None
    assert states.days[-1] is ko
    assert states.meta["end_reason"] == "ko" and states.meta["ko_date"] == PROBE_KO_DAY.isoformat()


def test_true_ladder_centre_equals_the_engine_price_and_the_daily_greeks_delta(
        probe_run, probe_states, engine):
    from s2ts import NOTIONAL
    from s2ts.note import daily_greeks
    from s2ts.tails import pillar_inputs
    from quantark.util.numerical import is_close

    _, by = probe_states
    r = by[PROBE_DAY.isoformat()]
    a = probe_run.at_day
    life, dc, env, nd, curve = a.life, a.dc, a.env, a.nd, a.curve
    price = float(engine.price(nd.product, env, lifecycle_state=life.lifecycle))
    centre = r["states"]["true"]["engine_price"][LADDER_CENTRE]
    one_millionth_bp = 1e-6 * 1e-4 * NOTIONAL                      # 1e-6 bp of notional, in CNY
    assert is_close(centre, price, abs_tol=one_millionth_bp)
    _, mults = pillar_inputs(dc)
    g = daily_greeks(nd.product, env, curve, engine, life.lifecycle, mults, with_vega=False)
    assert is_close(_bump_delta(r["states"]["true"], r["spot"]), g.spot_delta, rel_tol=1e-9)


def test_true_model_ladder_equals_the_note_built_by_build_note(probe_run, probe_states, engine):
    """The rebuilt-and-aged ``true`` variant is the tracker's own pricing copy: same prices on the whole ladder."""
    from gpin.models import MODELS
    from gpin.pricing import model_ladder
    from s2ts.note import build_terms

    _, by = probe_states
    r = by[PROBE_DAY.isoformat()]
    life, dc, env, nd = probe_run.at_day.life, probe_run.at_day.dc, probe_run.at_day.env, probe_run.at_day.nd
    s0 = probe_run.s0
    spots = dc.spot * (1.0 + np.asarray(gpin.LADDER))
    rebuilt = model_ladder(engine, life, MODELS["true"], build_terms(PROBE_INCEPTION), s0, PROBE_COUPON, PROBE_DAY,
                           PROBE_INCEPTION, env, spots, None)
    assert rebuilt["engine_price"] == pytest.approx(r["states"]["true"]["engine_price"], abs=1e-9)
    assert rebuilt["engine_delta"] == pytest.approx(r["states"]["true"]["engine_delta"], abs=1e-9)


def test_ko_cash_is_what_the_lifecycle_pays(probe_run):
    """``ko_cash`` (first aged record's payoff) equals the cash ``NoteLife`` reports when that observation fires."""
    from gpin.pricing import ko_cash
    from s2ts.note import build_terms

    a = probe_run.at_day
    cash = ko_cash(a.nd.product, a.env, PROBE_KO_DAY, PROBE_DAY, build_terms(PROBE_INCEPTION))
    fired = probe_run.ko
    assert fired.alive is False and fired.end_reason == "ko"
    assert fired.cash == pytest.approx(cash, rel=1e-12)
    assert 1e4 * cash / 1e7 == pytest.approx(369.86, abs=0.01)    # 15% x 90 / 365 of notional


def test_gap_ko_formula_discounts_the_cash_and_interpolates_the_survived_price():
    from gpin.pricing import gap_ko_engine
    from s2ts import RATE

    spots = 100.0 * (1.0 + np.asarray(gpin.LADDER))
    surv = list(np.linspace(-120.0, 360.0, 25))                    # 20 per ladder point, i.e. 20 per index point
    gap = gap_ko_engine(500.0, PROBE_KO_DAY, PROBE_DAY, spots, surv, 100.5)
    discount = math.exp(-RATE * 1 / 365.0)
    assert gap == pytest.approx(500.0 * discount - (-120.0 + 20.0 * (100.5 - 88.0)), rel=1e-12)
    with pytest.raises(StudyDataError):
        gap_ko_engine(500.0, PROBE_KO_DAY, PROBE_DAY, spots, surv, 130.0)


def test_expected_solves_equals_the_solves_made(real_chains, probe_scan, probe_states):
    from gpin.pricing import calendar_for, expected_solves

    states, _ = probe_states
    plan = expected_solves(real_chains, probe_scan, calendar_for(real_chains),
                           price_days={PROBE_MID_DAY, PROBE_FIRST_WINDOW_DAY, PROBE_DAY})
    assert plan["solves"] == states.meta["solves"] == 3 + 2 * 7
    assert plan["priced_days"] == 3 and plan["ko_window_days"] == states.meta["ko_window_days"] == 2
    full = expected_solves(real_chains, probe_scan, calendar_for(real_chains))
    assert (full["priced_days"], full["ko_window_days"], full["solves"]) == (59, 10, 59 + 10 * 7)


def test_the_pass_raises_when_the_tracker_disagrees_with_the_scan(real_chains, real_anchors, probe_scan, engine):
    import dataclasses

    from gpin.pricing import price_note

    for doctored in (dataclasses.replace(probe_scan, ko_date=date(2024, 6, 28)),
                     dataclasses.replace(probe_scan, ki_date=date(2024, 5, 1)),
                     dataclasses.replace(probe_scan, end_reason="maturity"),
                     dataclasses.replace(probe_scan, end_date=date(2024, 6, 3))):
        with pytest.raises(StudyDataError, match="disagrees with the scan"):
            price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, doctored,
                       engine=engine, price_days=set())
    ok = price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan,
                    engine=engine, price_days=set())
    assert ok.meta["solves"] == 0 and len(ok.days) == 60 and all(r["states"] == {} for r in ok.days)


def test_a_horizon_is_checked_only_inside_itself(real_chains, real_anchors, probe_scan, engine):
    import dataclasses

    from gpin.pricing import price_note

    short = price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan,
                       engine=engine, price_days=set(), max_days=20)
    assert len(short.days) == 21 and short.meta["end_reason"] == "truncated"
    inside = dataclasses.replace(probe_scan, ko_date=date(2024, 3, 20))
    with pytest.raises(StudyDataError, match="disagrees with the scan"):
        price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, inside,
                   engine=engine, price_days=set(), max_days=20)


def test_a_second_run_reuses_a_complete_stamped_file_and_a_diagnostic_or_new_fingerprint_does_not(
        real_chains, real_anchors, probe_scan, engine, tmp_path):
    from gpin.pricing import price_one, states_path, write_states_atomic

    kw = dict(engine=engine, price_days=set(), max_days=5)
    first = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw)
    path = states_path(tmp_path, PROBE_INCEPTION, "ki75")
    assert path.name == "ki75__2024-02-29.json" and first["note"] == "ki75:2024-02-29"
    stamp = path.stat().st_mtime_ns
    assert first["reused"] is False and json.loads(path.read_text())["fingerprint"] == "fpA"
    # a run written with max_days / price_days is a diagnostic: never reused, even with the same stamp
    again = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw)
    assert again["reused"] is False and path.stat().st_mtime_ns != stamp
    # a complete file (here: the same record with the diagnostics cleared) is reused under its fingerprint and coupon
    full = json.loads(path.read_text())
    full["meta"].update(max_days=None, price_days=None)
    write_states_atomic(path, full)
    stamp = path.stat().st_mtime_ns
    hit = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw)
    assert hit["reused"] is True and path.stat().st_mtime_ns == stamp
    new_coupon = price_one(PROBE_INCEPTION, PROBE_COUPON + 0.01, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw)
    assert new_coupon["reused"] is False and json.loads(path.read_text())["coupon"] == PROBE_COUPON + 0.01
    write_states_atomic(path, full)
    other = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpB", **kw)
    assert other["reused"] is False and json.loads(path.read_text())["fingerprint"] == "fpB"
    assert not list(tmp_path.glob("*.tmp"))


def test_kin_differs_from_true_before_the_knock_in_equals_it_after_and_spares_the_tracker(
        real_chains, real_anchors, real_scans, engine):
    from gpin.models import MODELS
    from gpin.pricing import model_ladder
    from s2ts.note import build_terms

    # the anchored note that knocks in soonest after its inception (short stepping), at least 5 days in, whose
    # previous close was inside the 5% KI window (a larger one-day drop would leave the KI models unpriced)
    days = [c.day for c in real_chains]
    spot = {c.day: c.spot for c in real_chains}

    def prior_in_window(s):
        return spot[days[days.index(s.ki_date) - 1]] / (0.75 * spot[s.inception]) - 1.0 <= 0.05

    note = min((s for s in real_scans if s.anchored and s.ki_date is not None and (s.ki_date - s.inception).days >= 5
                and prior_in_window(s)), key=lambda s: (s.ki_date - s.inception, s.inception))
    before = days[days.index(note.ki_date) - 1]
    s0 = next(c.spot for c in real_chains if c.day == note.inception)
    terms = build_terms(note.inception)
    for until, equal in ((before, False), (note.ki_date, True)):
        life, dc, env, nd, _ = _step_life(real_chains, real_anchors, note.inception, PROBE_COUPON, until)
        assert nd.knocked_in is equal
        spots = dc.spot * (1.0 + np.asarray(gpin.LADDER))
        true = model_ladder(engine, life, MODELS["true"], terms, s0, PROBE_COUPON, until, note.inception, env, spots, None)
        kin = model_ladder(engine, life, MODELS["kin"], terms, s0, PROBE_COUPON, until, note.inception, env, spots, None)
        diff = max(abs(a - b) for a, b in zip(true["engine_price"], kin["engine_price"]))
        assert (diff == 0.0) is equal, (until, diff)
        assert life.lifecycle.knocked_in is equal                  # the tracker's state is unchanged
    # the KI window day before the knock-in carries the KI models; the knock-in day itself none of them
    from gpin.pricing import price_note

    states = price_note(note.inception, PROBE_COUPON, real_chains, real_anchors, note,
                        engine=engine, price_days={before, note.ki_date}, max_days=days.index(note.ki_date) - days.index(note.inception))
    by = {r["date"]: r for r in states.days}
    assert {"ki_m100", "ki_m200", "ki_p100", "kin"} <= set(by[before.isoformat()]["states"])
    assert not {"ki_m100", "ki_m200", "ki_p100", "kin"} & set(by[note.ki_date.isoformat()]["states"])
    assert by[note.ki_date.isoformat()]["knocked_in"] is True


# ---- 03_pricing.py: the sample, the plan, the pool's failure / lock / resume behaviour ----


@pytest.fixture(scope="module")
def stage3():
    return _load_stage("03_pricing.py")


def test_stage3_plan_is_per_book_and_in_total(stage3, real_chains, book_scans):
    from gpin.events import BOOKS, parse_note_id
    from gpin.pricing import calendar_for, expected_solves

    sample = stage3.load_sample()
    notes = stage3.sample_notes(sample)
    sizes = {b: sample["books"][b]["sample_sizes"]["priced_notes"] for b in BOOKS}
    assert {b: sum(1 for n in notes if parse_note_id(n)[0] == b) for b in BOOKS} == sizes
    assert len(notes) == sum(sizes.values()) and len(set(notes)) == len(notes)
    assert stage3.sample_notes(sample, books=["ki90"]) == [n for n in notes if n.startswith("ki90:")]
    by_book = {b: {s.inception: s for s in book_scans[b]} for b in BOOKS}
    summary = stage3.plan(real_chains, by_book, notes, workers=4)
    cal = calendar_for(real_chains)
    for b in BOOKS:
        per_note = [expected_solves(real_chains, by_book[b][parse_note_id(n)[1]], cal, ki_pct=BOOKS[b])
                    for n in notes if parse_note_id(n)[0] == b]
        z = summary["books"][b]
        assert z["notes"] == sizes[b]
        assert z["solves"] == sum(c["solves"] for c in per_note)
        assert z["priced_days"] == sum(c["priced_days"] for c in per_note)
        assert z["solves"] == z["priced_days"] + 7 * z["ko_window_days"] + 4 * z["ki_window_days"]
        assert z["hours"] == pytest.approx(z["solves"] * stage3.SECONDS_PER_SOLVE / 4 / 3600.0)
        assert 0 < z["live_note_days"] <= z["priced_days"]
    total = summary["total"]
    for key in ("notes", "priced_days", "ko_window_days", "ki_window_days", "solves", "live_note_days"):
        assert total[key] == sum(summary["books"][b][key] for b in BOOKS), key
    assert total["hours"] == pytest.approx(sum(summary["books"][b]["hours"] for b in BOOKS))
    assert total["coupon_solves"] == 3 * len(notes)
    short = stage3.plan(real_chains, by_book, notes[:2], workers=4, max_days=30)
    assert short["total"]["priced_days"] <= 62
    lines = []
    stage3.print_plan(summary, log=lines.append)
    text = "\n".join(lines)
    for b in BOOKS:
        assert f"book {b}" in text
    assert "total" in text and "h wall" in text


def test_stage3_quick_notes_are_two_ki75_notes_and_a_ki90_pin_note(stage3):
    sample = stage3.load_sample()
    picks = stage3.quick_notes(sample)
    assert len(picks) == 3
    assert picks[0] == f"ki75:{sample['books']['ki75']['headline'][0]}" and picks[1] == "ki75:2024-02-29"
    assert "2024-02-29" in {p["inception"] for p in sample["books"]["ki75"]["pin"]}
    book, inception = picks[2].split(":")
    assert book == "ki90" and inception in {p["inception"] for p in sample["books"]["ki90"]["pin"]}


def test_stage3_notes_failed_is_a_quantark_exception(stage3):
    from quantark.util.exceptions import QuantArkException

    assert issubclass(stage3.NotesFailed, QuantArkException)


def test_stage3_pool_records_failures_and_goes_on_and_refuses_a_second_instance(stage3, tmp_path):
    from s2ts.runner import OutDirLocked, _acquire_lock, _release_lock

    notes = ["ki75:2024-02-29", "ki90:2024-02-29", "ki90:2024-03-04"]

    def job(nid):
        if nid == notes[1]:
            raise RuntimeError("boom")
        return {"note": nid, "inception": nid.split(":")[1], "reused": False, "seconds": 1.0, "solves": 7,
                "live_days": 3}

    (tmp_path / "stale.json.tmp").write_text("x")
    lines = []
    with pytest.raises(stage3.NotesFailed, match="1 of 3 notes failed"):
        stage3.run_notes(notes, {}, tmp_path, "fp", workers=0, job=job, log=lines.append)
    err = json.loads((tmp_path / "ki90__2024-02-29.error.json").read_text())
    assert err["note"] == "ki90:2024-02-29" and "boom" in err["error"] and "Traceback" in err["traceback"]
    assert not (tmp_path / "ki75__2024-02-29.error.json").exists() and not (tmp_path / "stale.json.tmp").exists()
    assert not (tmp_path / ".lock").exists()                       # released on exit
    assert any("ki75:2024-02-29" in ln and "7 solves" in ln and "1 s" in ln for ln in lines)   # a line per finished note

    lock = _acquire_lock(tmp_path)                                 # a live instance holds the directory
    try:
        with pytest.raises(OutDirLocked):
            stage3.run_notes(notes, {}, tmp_path, "fp", workers=0, job=job, log=lines.append)
    finally:
        _release_lock(lock)
    # a re-run clears the failed note's error file before it retries
    ok = stage3.run_notes(notes, {}, tmp_path, "fp", workers=0,
                          job=lambda nid: {"note": nid, "inception": nid.split(":")[1], "reused": True, "seconds": 0.0,
                                           "solves": 0, "live_days": 1}, log=lines.append)
    assert [p.name for p in ok] == ["ki75__2024-02-29.json", "ki90__2024-02-29.json", "ki90__2024-03-04.json"]
    assert not list(tmp_path.glob("*.error.json"))


# ---------------------------------------------------------------------------
# T3b — books in the models and the pricing pass
# ---------------------------------------------------------------------------


def test_ki_level_follows_the_books_level():
    m = _models()
    assert m.ki_level(m.MODELS["true"], S0_FIXED, 0.90) == pytest.approx(0.90 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m100"], S0_FIXED, 0.90) == pytest.approx(0.891 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_p100"], S0_FIXED, 0.85) == pytest.approx(0.85 * 1.01 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["true"], S0_FIXED) == pytest.approx(0.75 * S0_FIXED, rel=1e-15)   # the product by default


def test_book_note_is_build_note_at_75_and_carries_the_level_otherwise():
    from s2ts.note import build_note, build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    ref = build_note(PROBE_COUPON, S0_FIXED, terms)
    same = m.book_note(PROBE_COUPON, S0_FIXED, terms, 0.75)
    assert same.barrier_config == ref.barrier_config and same.maturity == ref.maturity
    assert same.contract_multiplier == ref.contract_multiplier
    lifted = m.book_note(PROBE_COUPON, S0_FIXED, terms, 0.90)
    assert lifted.barrier_config.ki_barrier == pytest.approx(0.90 * S0_FIXED, rel=1e-15)
    assert lifted.barrier_config.ko_barrier == ref.barrier_config.ko_barrier
    prod = m.MODELS["ko_s100"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, -1, ki_pct=0.90)
    assert prod.barrier_config.ki_barrier == pytest.approx(0.90 * S0_FIXED, rel=1e-15)
    shifted = m.MODELS["ki_m100"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, None, ki_pct=0.90)
    assert shifted.barrier_config.ki_barrier == pytest.approx(0.891 * S0_FIXED, rel=1e-15)


def test_the_ki_window_follows_the_books_level():
    from gpin.pricing import in_ki_window

    assert in_ki_window(78.75, 100.0, False) and not in_ki_window(78.8, 100.0, False)
    assert in_ki_window(94.5, 100.0, False, ki_pct=0.90) and not in_ki_window(94.6, 100.0, False, ki_pct=0.90)
    assert not in_ki_window(90.0, 100.0, True, ki_pct=0.90)           # a knocked-in note has no KI window


def test_the_ki_window_is_five_percent():
    from gpin import pricing

    assert pricing.KI_BAND == 0.05 == load_prereg()["windows"]["ki_band"]


def test_books_coupons_ki75_equals_solve_coupon_and_higher_levels_pay_more(real_chains, engine):
    from gpin.models import book_note
    from gpin.pricing import solve_book_coupon
    from quantark.param import FlatRateCurve
    from s2ts import NOTIONAL, RATE
    from s2ts.note import build_terms, make_env, solve_coupon
    from s2ts.tails import build_curve

    dc = next(c for c in real_chains if c.day == PROBE_INCEPTION)
    c75 = solve_book_coupon(dc, engine, 0.75)
    assert c75 == solve_coupon(dc, engine)                               # the same method, the same number
    c90 = solve_book_coupon(dc, engine, 0.90)
    assert c90 > c75                                                     # a higher barrier needs a higher coupon
    env = make_env(dc, build_curve(dc, "hold_forward", None).to_dividend_yield(FlatRateCurve(RATE)))
    pv = float(engine.price(book_note(c90, dc.spot, build_terms(dc.day), 0.90), env))
    assert abs(pv) <= 1e-6 * NOTIONAL                                    # reprices to zero under hold-forward


def test_price_note_builds_every_model_off_the_books_level(real_chains, real_anchors, real_scans, engine):
    from gpin.events import scan
    from gpin.models import MODELS
    from gpin.pricing import aged_model_product, price_note
    from s2ts.note import build_terms

    s90 = next(s for s in scan(real_chains, real_anchors, ki_pct=0.90) if s.inception == PROBE_INCEPTION)
    states = price_note(PROBE_INCEPTION, 0.2, real_chains, real_anchors, s90, ki_pct=0.90, engine=engine,
                        price_days={PROBE_MID_DAY}, max_days=44)
    assert states.book == "ki90" and states.ki_pct == 0.90
    out = states.to_json("fp")
    assert (out["book"], out["ki_pct"], out["note"]) == ("ki90", 0.90, "ki90:2024-02-29")
    with pytest.raises(StudyDataError, match="not book"):                # the 75 scan is not the 90 book's
        price_note(PROBE_INCEPTION, 0.2, real_chains, real_anchors,
                   next(s for s in real_scans if s.inception == PROBE_INCEPTION), ki_pct=0.90, engine=engine,
                   price_days=set(), max_days=3)
    life, dc, env, nd, _ = _step_life(real_chains, real_anchors, PROBE_INCEPTION, 0.2, PROBE_MID_DAY, ki_pct=0.90)
    s0 = next(c.spot for c in real_chains if c.day == PROBE_INCEPTION)
    terms = build_terms(PROBE_INCEPTION)
    for mid, mult in (("true", 0.90), ("ki_m100", 0.891), ("kin", 0.90)):
        aged = aged_model_product(life, MODELS[mid], terms, s0, 0.2, PROBE_MID_DAY, PROBE_INCEPTION, env, None,
                                  ki_pct=0.90)
        assert aged.barrier_config.ki_barrier == pytest.approx(mult * s0, rel=1e-12), mid


def test_state_files_are_named_by_book(tmp_path):
    from gpin.pricing import states_path

    assert states_path(tmp_path, date(2024, 2, 29), "ki90").name == "ki90__2024-02-29.json"
    assert states_path(tmp_path, date(2024, 2, 29), "ki75").name == "ki75__2024-02-29.json"


def test_coupon_cache_is_stamped_and_re_solved_when_the_stamp_differs(tmp_path):
    from gpin.pricing import book_coupons

    calls = []

    def solver(book, inception):
        calls.append((book, inception))
        return 0.1 + 0.01 * len(calls)

    path = tmp_path / "coupons.json"
    ids = ["ki75:2024-02-29", "ki90:2024-02-29"]
    first = book_coupons(ids, path, "stampA", solver)
    assert len(calls) == 2 and first["ki75:2024-02-29"] != first["ki90:2024-02-29"]
    assert book_coupons(ids, path, "stampA", solver) == first and len(calls) == 2         # reused
    assert book_coupons(ids[:1], path, "stampA", solver) == {ids[0]: first[ids[0]]}
    again = book_coupons(ids, path, "stampB", solver)                                      # stamp differs: re-solved
    assert len(calls) == 4 and again != first
    assert json.loads(path.read_text())["stamp"] == "stampB"
    path.write_text("{ broken")                                                            # unreadable cache: re-solve
    book_coupons(ids[:1], path, "stampB", solver)
    assert len(calls) == 5


def test_coupon_stamp_covers_data_s2ts_engine_books_and_the_solver_file(tmp_path):
    import shutil

    from gpin import pricing

    s2 = tmp_path / "s2ts"
    shutil.copytree(S2TS_DIR / "s2ts", s2, ignore=shutil.ignore_patterns("__pycache__"))
    pkg = tmp_path / "gpin"
    shutil.copytree(STUDY_DIR / "gpin", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    kw = dict(s2ts_dir=s2, pkg_dir=pkg, data_digest="d")
    base = pricing.coupon_stamp(**kw)
    assert base == pricing.coupon_stamp(**kw)
    assert pricing.coupon_stamp(**{**kw, "data_digest": "e"}) != base
    (s2 / "note.py").write_text((s2 / "note.py").read_text() + "\n# edit\n")
    edited = pricing.coupon_stamp(**kw)
    assert edited != base
    (pkg / "events.py").write_text((pkg / "events.py").read_text() + "\n# edit\n")
    assert pricing.coupon_stamp(**kw) == edited                       # only the solver's file counts
    (pkg / "pricing.py").write_text((pkg / "pricing.py").read_text() + "\n# edit\n")   # an edit of the solver
    assert pricing.coupon_stamp(**kw) != edited
    parts = pricing.coupon_stamp_parts(**kw)
    assert set(parts) == {"data", "s2ts", "engine", "books", "solver"}
    assert parts["books"] == {"ki75": 0.75, "ki85": 0.85, "ki90": 0.90}


def test_the_fingerprint_names_the_quantark_version():
    import quantark

    from gpin import pricing

    assert pricing.library_versions()["quantark"] == quantark.__version__
    assert {"python", "numpy", "scipy"} <= set(pricing.library_versions())


def test_state_file_reuse_needs_fingerprint_coupon_and_a_complete_run(tmp_path):
    from gpin import pricing

    path = tmp_path / "ki90__2024-02-29.json"
    base = {"note": "ki90:2024-02-29", "inception": "2024-02-29", "fingerprint": "A", "coupon": 0.2, "days": [],
            "meta": {"max_days": None, "price_days": None}}
    pricing.write_states_atomic(path, base)
    assert pricing.read_current(path, "ki90:2024-02-29", "A", 0.2) == base
    assert pricing.read_current(path, "ki90:2024-02-29", "B", 0.2) is None            # fingerprint
    assert pricing.read_current(path, "ki90:2024-02-29", "A", 0.2000001) is None      # a re-solved coupon
    assert pricing.read_current(path, "ki75:2024-02-29", "A", 0.2) is None            # another book's note
    for key, value in (("max_days", 5), ("price_days", ["2024-03-01"])):
        pricing.write_states_atomic(path, {**base, "meta": {**base["meta"], key: value}})
        assert pricing.read_current(path, "ki90:2024-02-29", "A", 0.2) is None, key   # a diagnostic run is never reused
    pricing.write_states_atomic(path, {k: v for k, v in base.items() if k != "meta"})
    assert pricing.read_current(path, "ki90:2024-02-29", "A", 0.2) is None            # no provenance, no reuse


# ---------------------------------------------------------------------------
# T3b — tests the checkpoint review asked for (one shared stepped lifecycle where several tests read the same day)
# ---------------------------------------------------------------------------

AGED_INCEPTION = date(2023, 10, 23)
AGED_DAY = date(2024, 3, 22)                 # two KO observations (2024-01-23, 2024-02-23) behind, 2024-03-25 next


@pytest.fixture(scope="module")
def probe_run(real_chains, real_anchors):
    """The probe note stepped once: its state at ``PROBE_DAY`` (a deep copy, so tests may price from it) and the
    KO day's result."""
    from types import SimpleNamespace

    from quantark.param import FlatRateCurve
    from s2ts import RATE
    from s2ts.note import NoteLife, build_terms, make_env
    from s2ts.tails import build_curve
    from gpin.models import book_note

    i0 = next(i for i, c in enumerate(real_chains) if c.day == PROBE_INCEPTION)
    life = NoteLife(book_note(PROBE_COUPON, real_chains[i0].spot, build_terms(PROBE_INCEPTION)), PROBE_INCEPTION)
    out = SimpleNamespace(s0=real_chains[i0].spot)
    for dc in real_chains[i0:]:
        curve = build_curve(dc, gpin.TAIL, real_anchors.get(dc.day))
        env = make_env(dc, curve.to_dividend_yield(FlatRateCurve(RATE)))
        nd = life.step(dc, env)
        if dc.day == PROBE_DAY:
            out.at_day = SimpleNamespace(life=copy.deepcopy(life), dc=dc, env=env, nd=nd, curve=curve)
        if dc.day == PROBE_KO_DAY:
            out.ko = nd
            return out
    raise AssertionError("the probe note did not reach its KO day")


@pytest.fixture(scope="module")
def aged_run(real_chains, real_anchors):
    from types import SimpleNamespace

    life, dc, env, nd, _ = _step_life(real_chains, real_anchors, AGED_INCEPTION, PROBE_COUPON, AGED_DAY)
    return SimpleNamespace(life=life, dc=dc, env=env, nd=nd,
                           s0=next(c.spot for c in real_chains if c.day == AGED_INCEPTION))


def test_an_aged_variant_after_a_ko_observation_keeps_the_ladder_and_disables_the_next_level(aged_run, engine):
    """Guards ``_trim_ko_levels`` and the left-count check: ageing drops the observed records but not the per-
    observation level list, so without the trim the rebuilt variant does not validate."""
    from gpin.models import MODELS, SURV_OFF_MULTIPLE
    from gpin.pricing import aged_model_product, model_ladder, solve_ladder
    from s2ts.note import build_terms

    run = aged_run
    assert run.nd.alive and not run.nd.knocked_in
    terms = build_terms(AGED_INCEPTION)
    spots = run.dc.spot * (1.0 + np.asarray(gpin.LADDER))
    rebuilt = model_ladder(engine, run.life, MODELS["true"], terms, run.s0, PROBE_COUPON, AGED_DAY, AGED_INCEPTION,
                           run.env, spots, None)
    tracker = solve_ladder(engine, run.nd.product, run.env, spots, run.life.lifecycle)
    for key in ("engine_price", "engine_delta"):
        assert rebuilt[key] == pytest.approx(tracker[key], abs=1e-9), key      # the whole ladder
    surv = aged_model_product(run.life, MODELS["surv"], terms, run.s0, PROBE_COUPON, AGED_DAY, AGED_INCEPTION,
                              run.env, None)
    records = list(surv.barrier_config.ko_observation_schedule.records)
    assert len(records) == len(surv.barrier_config.ko_observation_dates) == len(surv.barrier_config.ko_barrier) == 20
    assert records[0].barrier == pytest.approx(SURV_OFF_MULTIPLE * run.s0)    # the first SURVIVING record is the disabled one
    assert all(r.barrier == pytest.approx(run.s0) for r in records[1:])
    assert surv.barrier_config.ko_barrier[0] == pytest.approx(SURV_OFF_MULTIPLE * run.s0)


def test_the_probe_centre_prices_order_by_the_signed_shift(probe_states):
    """``ko_dir = -1`` books the KO level 1% lower: the long note is called sooner and is worth more at a spot that
    sits on the level; the wrong-way control books it higher and is worth less."""
    _, by = probe_states
    r = by[PROBE_DAY.isoformat()]
    assert r["ko_dir"] == -1
    centre = {m: r["states"][m]["engine_price"][LADDER_CENTRE] for m in ("ko_s100", "true", "ko_w100")}
    assert centre["ko_s100"] > centre["true"] > centre["ko_w100"]


def test_ko_dir_is_kept_per_observation_through_price_note_and_reset_for_the_next(
        real_chains, real_anchors, real_scans, engine, monkeypatch):
    """Drives ``price_note`` with a gap whose sign flips inside every observation's window (the engine solves are
    stubbed: the direction is bookkeeping). The first gap of an observation alternates in sign between observations."""
    from gpin import pricing

    state = {"seen": {}}

    def fake_gap(payoff, obs_date, day, spots, surv_price, barrier):
        n = state["seen"].setdefault(obs_date, [])
        first_positive = len(state["seen"]) % 2 == 1          # obs 1: +, -, -, ...; obs 2: -, +, +, ...
        n.append(1)
        positive = first_positive if len(n) == 1 else not first_positive
        return 100.0 if positive else -100.0

    def fake_solve(engine_, product, env, spots, lifecycle_state, label=""):
        zeros = [0.0] * len(spots)
        return {"engine_price": list(zeros), "engine_delta": list(zeros), "engine_gamma": list(zeros)}

    monkeypatch.setattr(pricing, "gap_ko_engine", fake_gap)
    monkeypatch.setattr(pricing, "solve_ladder", fake_solve)
    note = next(s for s in real_scans if s.anchored and s.inception >= date(2023, 9, 1) and s.ko_observations >= 3
                and s.ki_date is None and s.end_reason == "ko")
    states = pricing.price_note(note.inception, PROBE_COUPON, real_chains, real_anchors, note, engine=engine)
    by_obs = {}
    for r in states.days:
        if r["ko_dir"] is not None:
            by_obs.setdefault(r["next_ko"], []).append(r["ko_dir"])
    long_windows = [k for k, v in by_obs.items() if len(v) >= 2]
    assert len(long_windows) >= 2, by_obs
    order = list(state["seen"])                                  # observations in the order the pass met them
    for k, dirs in by_obs.items():
        expected = -1 if order.index(date.fromisoformat(k)) % 2 == 0 else 1   # the sign of that observation's FIRST gap
        assert set(dirs) == {expected}, (k, dirs)                # kept inside the window, set afresh for the next
    assert {-1, 1} <= {d for v in by_obs.values() for d in v}


def test_the_ko_day_record_carries_the_positive_lifecycle_cash(probe_states, probe_run):
    from s2ts import NOTIONAL

    _, by = probe_states
    cash = by[PROBE_KO_DAY.isoformat()]["engine_note_cash"]
    assert cash > 0.0                                             # engine side: the note pays the coupon
    assert cash == probe_run.ko.cash
    assert 1e4 * cash / NOTIONAL == pytest.approx(369.86, abs=0.01)    # 15% x 90 / 365 of notional
    assert cash == pytest.approx(PROBE_COUPON * 90 / 365.0 * NOTIONAL, rel=1e-12)


def test_clean_turns_a_nan_greek_into_null_and_a_nan_price_into_an_error():
    from gpin.pricing import _clean

    ladder = np.array([1.0, float("nan"), 3.0, float("inf")])
    assert _clean(ladder, price=False, label="g") == [1.0, None, 3.0, None]    # a contractual jump stays a gap
    assert _clean(np.array([1.0, 2.0]), price=True, label="p") == [1.0, 2.0]
    for bad in (float("nan"), float("inf")):
        with pytest.raises(StudyDataError, match="non-finite price"):
            _clean(np.array([1.0, bad]), price=True, label="p")


def test_the_probe_one_percent_bump_deltas_are_175_5_and_81_6(probe_states):
    _, by = probe_states
    r = by[PROBE_DAY.isoformat()]
    pct = {m: 100.0 * _bump_delta(r["states"][m], r["spot"]) * r["spot"] / gpin_notional() for m in ("true", "surv")}
    assert pct["true"] == pytest.approx(175.5, abs=0.1)
    assert pct["surv"] == pytest.approx(81.6, abs=0.1)


def gpin_notional() -> float:
    from s2ts import NOTIONAL

    return NOTIONAL


# ---------------------------------------------------------------------------
# T3b — small fixes
# ---------------------------------------------------------------------------


def test_a_short_im_or_mo_row_fails_closed_and_other_families_are_ignored():
    from gpin.board import parse_cffex_full

    good = cffex_payload([("IM2611", "1", "2", "0.5", "10", "5", "9", "1", "1")]).decode("gbk")
    for code in ("IM2611", "MO2611-C-6000"):
        bad = good + f"{code},1,2,0.5,10\n"
        with pytest.raises(StudyDataError, match=code):
            parse_cffex_full(bad.encode("gbk"))
    other = good + "HO2611-C-3250,1,2\n"          # another family with a short row is not our data
    assert set(parse_cffex_full(other.encode("gbk"))) == {"IM2611"}


def test_books_come_from_the_preregistration_and_note_ids_roundtrip():
    from gpin.events import BOOKS, note_id, parse_note_id

    assert BOOKS == load_prereg()["books"] == {"ki75": 0.75, "ki85": 0.85, "ki90": 0.90}
    assert note_id("ki90", date(2024, 2, 29)) == "ki90:2024-02-29"
    assert parse_note_id("ki90:2024-02-29") == ("ki90", date(2024, 2, 29))
    for bad in ("2024-02-29", "ki80:2024-02-29", "ki90:2024-13-01"):
        with pytest.raises(StudyDataError):
            parse_note_id(bad)


def test_scan_ki_level_is_a_parameter_a_90_book_knocks_in_where_the_75_does_not():
    from gpin.events import scan

    days = trading_days(START, START + timedelta(days=60))
    path = make_path(days[-1], {days[5]: 89.0, days[6]: 95.0}, base=95.0)
    s75 = next(s for s in scan(path, {START: 0.0}) if s.inception == START)
    s90 = next(s for s in scan(path, {START: 0.0}, ki_pct=0.90) if s.inception == START)
    assert s75.ki_date is None and s75.ki_pct == 0.75
    assert s90.ki_date == days[5] and s90.ki_pct == 0.90
    assert [(a.start, a.end, a.knocked_in) for a in s90.ki_approaches][-1] == (days[5], days[5], True)
    assert (s90.ko_date, s90.ko_events, s90.end_reason) == (s75.ko_date, s75.ko_events, s75.end_reason)  # KO unchanged
    # 90 x 1.05 = 94.5: a 95 is outside the window of the 90 book, 94.5 is on its edge (tolerant) and 92 is within 3%
    edge = make_path(days[-1], {days[5]: 94.5, days[6]: 92.0}, base=95.0)
    (a,) = [a for a in next(s for s in scan(edge, {START: 0.0}, ki_pct=0.90) if s.inception == START).ki_approaches]
    assert (a.start, a.end, a.knocked_in) == (days[5], days[6], False)


def test_select_lifted_books_take_three_notes_a_week_and_no_ko_pin_set():
    from gpin.events import select

    incs = [date(2024, 1, 2) + timedelta(days=i) for i in range(30)]
    t = date(2025, 3, 5)
    wk6 = (date(2024, 2, 5), date(2024, 2, 9))          # ISO 2024-W06
    wk7 = (date(2024, 2, 12), date(2024, 2, 14))        # ISO 2024-W07
    scans = [fake_scan(i, ki=[(*wk6, False)], ko=[(t, 0.001, True)], ki_pct=0.90) for i in incs]
    scans += [fake_scan(date(2023, 6, 2) + timedelta(days=i), ki=[(*wk7, False)], ki_pct=0.90) for i in range(2)]
    sample = select(scans, book="ki90")
    week6 = [i for i in sample.pin if i >= date(2024, 1, 2)]
    assert week6 == [incs[p] for p in (0, 15, 29)]                       # 3 evenly spaced of 30 in the week
    assert sum(1 for i in sample.pin if date(2023, 6, 2) <= i <= date(2023, 6, 3)) == 2   # fewer than 3: all of them
    assert sample.reasons[incs[15]] == ("ki 2024-W06 position 16 of 30",)
    assert all(r.startswith("ki ") for why in sample.reasons.values() for r in why)
    assert all(type(e).__name__ == "KiApproach" for e in sample.events)       # no KO pin events for a lifted book
    assert {e.inception for e in sample.events} == set(sample.pin) | set(sample.headline)
    ko75 = select([fake_scan(i, ko=[(t, 0.001, True)]) for i in incs[:3]])
    assert [r for why in ko75.reasons.values() for r in why] == ["ko 2025-W10 rank 1", "ko 2025-W10 rank 2"]
    assert any(type(e).__name__ == "KoEvent" for e in ko75.events)
    with pytest.raises(StudyDataError):
        select(scans, book="ki75")                                            # the scans are not that book's


def test_sample_fields_match_contract_c2_positionally():
    import dataclasses

    from gpin.events import Sample

    assert [f.name for f in dataclasses.fields(Sample)] == ["headline", "pin", "events", "reasons"]


def test_the_ko_shift_flag_is_a_tolerant_comparison():
    m = _models()
    assert m.HedgeModel("x", "ko", ko_shift=1e-13).needs_ko_dir is False
    assert m.HedgeModel("x", "ko", ko_shift=0.005).needs_ko_dir is True
