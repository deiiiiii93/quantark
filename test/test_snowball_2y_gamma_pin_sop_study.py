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
    assert p["books"] == {
        "ki75": {"ki_pct": 0.75, "first_ko_month": 3, "maturity_months": 24, "elapsed_months": 0,
                 "ki_below_spot": None, "booking": "sample"},
        "seas6": {"ki_pct": 0.75, "first_ko_month": 1, "maturity_months": 6, "elapsed_months": 18,
                  "ki_below_spot": 0.07, "booking": "weekly"}}
    assert p["sample"] == {"ko_prior_gap": 0.02, "ki_near": 0.03, "ko_notes_per_week": 2, "ki_notes_per_month": 12}
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
    assert v["whole_life_set"] == ("unselected notes of the same book (ki75 headline set; every seas6 booking), "
                                   "day-0 row left out")
    assert (v["ko_book"], v["ki_book"], v["ki_transfer_books"]) == ("ki75", "seas6", ["ki75"])
    assert [a["before_any_ledger_run"] for a in p["amendments"]] == [True, True]
    assert [a["date"] for a in p["amendments"]] == ["2026-11-12", "2026-11-12"]
    assert "ki85" in p["amendments"][1]["text"] and "seas6" in p["amendments"][1]["text"]
    assert "384.2" in p["amendments"][1]["text"] and "8.6" in p["amendments"][1]["text"]
    assert "holder" not in json.dumps(p).lower()
    assert p["selection"] == {"criterion": "es10_event_window", "clock": "close", "slice": "train", "tie": "lower_cost",
                              "ko_book": "ki75", "ki_book": "seas6"}
    h8 = {h["id"]: h["text"] for h in p["hypotheses"]}["H8"]
    assert h8 == ("the ranking of the KI rules on the seasoned book (seas6) is the same as on the 2Y product's own "
                  "approaches (ki75)")
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
    assert "ki85" in text and "knock-in pin is a seasoned-note problem" in text and "seas6" in text
    assert "The ranking of the KI rules on the seasoned book (`seas6`) is the same as on the 2Y" in text
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


def scan_one(path, start: date = START, book: str = "ki75"):
    from gpin.events import scan

    scans = scan(path, {start: 0.0}, book)
    return next(s for s in scans if s.inception == start)


def life_outcome(path, inception: date = START, book: str = "ki75"):
    """(ko_date, ki_date, end_reason, end_date) from the library's own tracker, no pricing."""
    from gpin.events import book_terms
    from gpin.models import book_note
    from s2ts.note import NoteLife, make_env

    life = NoteLife(book_note(0.2, path[0].spot, book_terms(book, inception), book=book), inception)
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
    scans = scan(path, {days[2]: 0.0, days[5]: 0.0}, "ki75")
    assert [s.inception for s in scans] == days
    assert [s.inception for s in scans if s.anchored] == [days[2], days[5]]
    assert all(s.end_reason == "truncated" and s.ko_observations == 0 for s in scans)
    assert scans[-1].end_date == days[-1] and scans[-1].ko_events == ()
    gap = path[:3] + path[4:]
    with pytest.raises(StudyDataError):
        scan(gap, {}, "ki75")  # a missing trading day fails closed
    with pytest.raises(StudyDataError):
        scan([Bar(date(2024, 1, 6), 100.0)], {}, "ki75")  # a Saturday is not a trading day


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


def fake_scan(inc, *, anchored=True, ko=(), ki=(), n_ko=0, ki_date=None, book="ki75"):
    from gpin.events import KiApproach, KoEvent, NoteScan

    return NoteScan(
        inception=inc, anchored=anchored, ko_date=None, ki_date=ki_date, end_reason="truncated",
        end_date=date(2026, 11, 6), ko_observations=n_ko,
        ko_events=tuple(KoEvent(inc, d, g, f) for d, g, f in ko),
        ki_approaches=tuple(KiApproach(inc, s, e, k) for s, e, k in ki), book=book)


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
    sample = select(scans, "ki75")
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
    sample = select(scans, "ki75")
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
    sample = select(scans, "ki75")
    assert sample.headline == (date(2024, 4, 1), date(2024, 7, 1))
    # events: every KO pin event and every KI approach of every selected note (headline or pin), nothing else
    got = {(type(e).__name__, e.inception) for e in sample.events}
    assert got == {("KoEvent", date(2024, 4, 1)), ("KiApproach", date(2024, 4, 1)), ("KoEvent", date(2024, 4, 2))}
    assert len([e for e in sample.events if type(e).__name__ == "KoEvent"]) == 3  # two of 04-01 (the week cap does not thin events), one of 04-02
    shuffled = list(scans)
    random.Random(1).shuffle(shuffled)
    assert select(shuffled, "ki75") == sample
    assert select(scans, "ki75") == sample
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
    scans = scan(real_chains, real_anchors, "ki75")
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

    assert list(select(real_scans, "ki75").headline) == inception_dates(real_chains, real_anchors)


def test_select_on_real_data_obeys_the_caps_and_is_deterministic(real_scans):
    from gpin.events import KO_NOTES_PER_WEEK, KI_NOTES_PER_MONTH, KoEvent, select, week_key

    sample = select(real_scans, "ki75")
    again = select(list(reversed(real_scans)), "ki75")
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


def test_seas6_scan_agrees_with_the_library_lifecycle_on_real_bookings(real_chains, book_scans):
    """Price-free cross-check of the seasoned book's scan against ``NoteLife`` on the shortest real booking of each
    outcome: KO after a knock-in, KO only, a knocked-in maturity, a maturity without a knock-in and a truncated
    knock-in (the pricing pass makes the same comparison for every priced note)."""
    by_day = {dc.day: dc for dc in real_chains}
    scans = book_scans["seas6"]
    picks = []
    for reason, knocked in (("ko", True), ("ko", False), ("maturity", True), ("maturity", False), ("truncated", True)):
        pool = [s for s in scans if s.end_reason == reason and (s.ki_date is not None) is knocked]
        assert pool, (reason, knocked)
        picks.append(min(pool, key=lambda s: ((s.end_date - s.inception).days, s.inception)))
    for s in picks:
        path = [Bar(d, by_day[d].spot) for d in sorted(by_day) if d >= s.inception]
        assert (s.ko_date, s.ki_date, s.end_reason, s.end_date) == life_outcome(path, s.inception, "seas6")[:4], s.inception


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

    return {b: (real_scans if b == "ki75" else scan(real_chains, real_anchors, b)) for b in BOOKS}


def test_events_json_matches_a_fresh_scan_and_lists_every_event_per_book(book_scans):
    from gpin.events import BOOKS, select

    stage = _load_stage("02_events.py")
    samples = {b: select(book_scans[b], b) for b in BOOKS}
    fresh = json.loads(json.dumps(stage.build_payload(book_scans, samples), sort_keys=True))
    on_disk = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    assert on_disk == fresh                                   # the committed file is the current scan
    assert on_disk["preregistration_hash"] == prereg_hash()
    assert on_disk["split_date"] == "2025-07-01"
    assert on_disk["unrestricted"]["ko_pin_fired"] == 145 and on_disk["unrestricted"]["notes_knocked_in"] == 209
    assert set(on_disk["books"]) == set(BOOKS) == {"ki75", "seas6"}
    for b, block in on_disk["books"].items():
        sample = samples[b]
        assert block["ki_pct"] == BOOKS[b].ki_pct
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
    assert "ko_events" in on_disk["books"]["ki75"] and "ko_events" not in on_disk["books"]["seas6"]
    assert on_disk["books"]["seas6"]["pin"] == []
    assert "holder" not in EVENTS_JSON.read_text(encoding="utf-8").lower()


def test_seas6_samples_every_booking_and_its_test_slice_is_decidable():
    """AC4b: one booking per ISO week, every one priced and listed as headline (none picked for a pin), and on real
    data the test slice holds at least 20 approaches in at least 10 distinct weeks."""
    on_disk = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    blk = on_disk["books"]["seas6"]
    z = blk["sample_sizes"]
    assert z["headline_notes"] == z["priced_notes"] == len(blk["notes"]) == blk["restricted"]["inceptions"] == 168
    assert z["pin_notes"] == 0
    for sl in ("train", "test"):
        assert z[sl]["ki_approaches"] == blk["restricted"][sl]["ki_approaches"]       # nothing is dropped by selection
        assert z[sl]["ki_weeks"] == blk["restricted"][sl]["ki_weeks"]
    assert z["test"]["ki_approaches"] >= 20 and z["test"]["ki_weeks"] >= 10
    assert z["train"]["ki_approaches"] >= 20 and z["train"]["ki_weeks"] >= 10
    weeks = [date.fromisoformat(d).isocalendar()[:2] for d in blk["notes"]]
    assert len(set(weeks)) == len(weeks) == 168                                         # one booking per ISO week


def test_ki75_sample_is_unchanged_by_the_books(real_scans, book_scans):
    """Adding the seasoned book does not touch the product's scan or sample: pinned to the counts of the committed
    events.json of the lifted-books task (13 headline notes, 125 KO pin reasons, 42 KI reasons, 150 KO events, 80 KI
    approaches over 142 pin notes), and the ki75 records are the library-pinned ones (``real_scans``)."""
    from gpin.events import KoEvent, select

    assert book_scans["ki75"] is real_scans
    sample = select(real_scans, "ki75")
    reasons = [r for why in sample.reasons.values() for r in why]
    assert len(sample.headline) == 13 and len(sample.pin) == 142
    assert sum(r.startswith("ko ") for r in reasons) == 125 and sum(r.startswith("ki ") for r in reasons) == 42
    assert sum(isinstance(e, KoEvent) for e in sample.events) == 150
    assert sum(not isinstance(e, KoEvent) for e in sample.events) == 80
    assert all(s.book == "ki75" for s in real_scans)
    on_disk = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))["books"]["ki75"]
    assert (len(on_disk["headline"]), len(on_disk["ko_events"]), len(on_disk["ki_approaches"]), len(on_disk["notes"])) == \
        (13, 150, 80, 153)


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
    assert m.ko_levels(m.MODELS["true"], terms, day, -1, s0=S0_FIXED, book="ki75") == [S0_FIXED] * n
    for size_id, size in (("ko_s050", 0.005), ("ko_s100", 0.01), ("ko_s300", 0.03)):
        down = m.ko_levels(m.MODELS[size_id], terms, day, -1, s0=S0_FIXED, book="ki75")
        up = m.ko_levels(m.MODELS[size_id], terms, day, +1, s0=S0_FIXED, book="ki75")
        assert down == pytest.approx([S0_FIXED * (1 - size)] * n, rel=1e-15)
        assert up == pytest.approx([S0_FIXED * (1 + size)] * n, rel=1e-15)
    # the wrong-way control moves the booked level the other way
    assert m.ko_levels(m.MODELS["ko_w100"], terms, day, -1, s0=S0_FIXED, book="ki75") == pytest.approx([S0_FIXED * 1.01] * n, rel=1e-15)
    assert m.ko_levels(m.MODELS["ko_w100"], terms, day, +1, s0=S0_FIXED, book="ki75") == pytest.approx([S0_FIXED * 0.99] * n, rel=1e-15)
    # relative levels by default (strike = 1)
    assert m.ko_levels(m.MODELS["ko_s200"], terms, day, -1, book="ki75") == pytest.approx([0.98] * n, rel=1e-15)
    # a model that does not shift the KO barrier ignores the direction (and needs none)
    assert m.ko_levels(m.MODELS["ki_m100"], terms, day, None, s0=S0_FIXED, book="ki75") == [S0_FIXED] * n


def test_surv_disables_exactly_the_next_observation():
    from s2ts.note import build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    off = m.SURV_OFF_MULTIPLE * S0_FIXED
    for day, idx in ((date(2024, 5, 28), 0), (date(2024, 5, 15), 0),
                     (terms.ko_dates[0], 1),                       # the observation day itself is already processed
                     (terms.ko_dates[0] + timedelta(days=1), 1),
                     (terms.ko_dates[4], 5)):
        lv = m.ko_levels(m.MODELS["surv"], terms, day, None, s0=S0_FIXED, book="ki75")
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
                m.ko_levels(m.MODELS[mid], terms, PROBE_DAY, bad, s0=S0_FIXED, book="ki75")


def test_ki_level_is_hand_computed():
    m = _models()
    assert m.ki_level(m.MODELS["true"], S0_FIXED, book="ki75") == pytest.approx(0.75 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m100"], S0_FIXED, book="ki75") == pytest.approx(0.75 * 0.99 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m200"], S0_FIXED, book="ki75") == pytest.approx(0.75 * 0.98 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_p100"], S0_FIXED, book="ki75") == pytest.approx(0.75 * 1.01 * S0_FIXED, rel=1e-15)
    assert m.ki_level(m.MODELS["kin"], S0_FIXED, book="ki75") == pytest.approx(0.75 * S0_FIXED, rel=1e-15)


def test_product_carries_the_levels_and_the_true_product_is_the_note():
    from s2ts.note import build_note, build_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    note = build_note(PROBE_COUPON, S0_FIXED, terms)
    true = m.MODELS["true"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, None, book="ki75")
    # build_note books one scalar level, a model one level per observation (the same schedule)
    assert list(true.barrier_config.ko_barrier) == [note.barrier_config.ko_barrier] * 22
    assert true.barrier_config.ki_barrier == note.barrier_config.ki_barrier
    assert true.maturity == note.maturity and true.contract_multiplier == note.contract_multiplier
    shifted = m.MODELS["ko_s100"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, -1, book="ki75")
    assert list(shifted.barrier_config.ko_barrier) == pytest.approx([S0_FIXED * 0.99] * 22, rel=1e-15)
    ki = m.MODELS["ki_m200"].product(terms, S0_FIXED, PROBE_COUPON, PROBE_DAY, None, book="ki75")
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
                        engine=engine, price_days=days, book="ki75")
    return states, {r["date"]: r for r in states.days}


def _step_life(real_chains, real_anchors, inception, coupon, until, book="ki75"):
    """Step a NoteLife to ``until`` as the pricing pass does; returns (life, dc, env, nd, curve)."""
    from quantark.param import FlatRateCurve
    from s2ts import RATE
    from s2ts.note import NoteLife, make_env
    from s2ts.tails import build_curve

    i0 = next(i for i, c in enumerate(real_chains) if c.day == inception)
    from gpin.events import book_terms
    from gpin.models import book_note

    life = NoteLife(book_note(coupon, real_chains[i0].spot, book_terms(book, inception), book=book), inception)
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
                           PROBE_INCEPTION, env, spots, None, book="ki75")
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
                           price_days={PROBE_MID_DAY, PROBE_FIRST_WINDOW_DAY, PROBE_DAY}, book="ki75")
    assert plan["solves"] == states.meta["solves"] == 3 + 2 * 7
    assert plan["priced_days"] == 3 and plan["ko_window_days"] == states.meta["ko_window_days"] == 2
    full = expected_solves(real_chains, probe_scan, calendar_for(real_chains), book="ki75")
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
                       engine=engine, price_days=set(), book="ki75")
    ok = price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan,
                    engine=engine, price_days=set(), book="ki75")
    assert ok.meta["solves"] == 0 and len(ok.days) == 60 and all(r["states"] == {} for r in ok.days)


def test_a_horizon_is_checked_only_inside_itself(real_chains, real_anchors, probe_scan, engine):
    import dataclasses

    from gpin.pricing import price_note

    short = price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan,
                       engine=engine, price_days=set(), max_days=20, book="ki75")
    assert len(short.days) == 21 and short.meta["end_reason"] == "truncated"
    inside = dataclasses.replace(probe_scan, ko_date=date(2024, 3, 20))
    with pytest.raises(StudyDataError, match="disagrees with the scan"):
        price_note(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, inside,
                   engine=engine, price_days=set(), max_days=20, book="ki75")


def test_a_second_run_reuses_a_complete_stamped_file_and_a_diagnostic_or_new_fingerprint_does_not(
        real_chains, real_anchors, probe_scan, engine, tmp_path):
    from gpin.pricing import price_one, states_path, write_states_atomic

    kw = dict(engine=engine, price_days=set(), max_days=5)
    first = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw, book="ki75")
    path = states_path(tmp_path, PROBE_INCEPTION, "ki75")
    assert path.name == "ki75__2024-02-29.json" and first["note"] == "ki75:2024-02-29"
    stamp = path.stat().st_mtime_ns
    assert first["reused"] is False and json.loads(path.read_text())["fingerprint"] == "fpA"
    # a run written with max_days / price_days is a diagnostic: never reused, even with the same stamp
    again = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw, book="ki75")
    assert again["reused"] is False and path.stat().st_mtime_ns != stamp
    # a complete file (here: the same record with the diagnostics cleared) is reused under its fingerprint and coupon
    full = json.loads(path.read_text())
    full["meta"].update(max_days=None, price_days=None)
    write_states_atomic(path, full)
    stamp = path.stat().st_mtime_ns
    hit = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw, book="ki75")
    assert hit["reused"] is True and path.stat().st_mtime_ns == stamp
    new_coupon = price_one(PROBE_INCEPTION, PROBE_COUPON + 0.01, real_chains, real_anchors, probe_scan, tmp_path, "fpA", **kw, book="ki75")
    assert new_coupon["reused"] is False and json.loads(path.read_text())["coupon"] == PROBE_COUPON + 0.01
    write_states_atomic(path, full)
    other = price_one(PROBE_INCEPTION, PROBE_COUPON, real_chains, real_anchors, probe_scan, tmp_path, "fpB", **kw, book="ki75")
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
        true = model_ladder(engine, life, MODELS["true"], terms, s0, PROBE_COUPON, until, note.inception, env, spots, None, book="ki75")
        kin = model_ladder(engine, life, MODELS["kin"], terms, s0, PROBE_COUPON, until, note.inception, env, spots, None, book="ki75")
        diff = max(abs(a - b) for a, b in zip(true["engine_price"], kin["engine_price"]))
        assert (diff == 0.0) is equal, (until, diff)
        assert life.lifecycle.knocked_in is equal                  # the tracker's state is unchanged
    # the KI window day before the knock-in carries the KI models; the knock-in day itself none of them
    from gpin.pricing import price_note

    states = price_note(note.inception, PROBE_COUPON, real_chains, real_anchors, note,
                        engine=engine, price_days={before, note.ki_date}, max_days=days.index(note.ki_date) - days.index(note.inception), book="ki75")
    by = {r["date"]: r for r in states.days}
    assert {"ki_m100", "ki_m200", "ki_p100", "kin"} <= set(by[before.isoformat()]["states"])
    assert not {"ki_m100", "ki_m200", "ki_p100", "kin"} & set(by[note.ki_date.isoformat()]["states"])
    assert by[note.ki_date.isoformat()]["knocked_in"] is True


# ---------------------------------------------------------------------------
# T3c — the seasoned book's payoffs, proved on the library lifecycle and on the QUAD V2 engine
# ---------------------------------------------------------------------------

SEAS_COUPON = 0.20
SEAS_S0 = 100.0
SEAS_ELAPSED = 1.5          # 18 months


def seasoned_life(path, coupon: float = SEAS_COUPON, inception: date = START):
    """Step the seas6 note booked at ``inception`` (spot 100) through ``path`` with ``s2ts.note.NoteLife`` (the
    library tracker, no pricing): ``(reason, end_day, cash, ki_day, steps)`` with ``cash`` the holder-side cash of
    the day the note ended (``steps``: ``[(day, NoteDay)]``)."""
    from gpin.events import book_terms
    from gpin.models import book_note
    from s2ts.note import NoteLife, make_env

    life = NoteLife(book_note(coupon, SEAS_S0, book_terms("seas6", inception), book="seas6"), inception)
    ki_day, steps = None, []
    for bar in path:
        dc = SimpleNamespace(day=bar.day, spot=bar.spot, mo=SimpleNamespace(vol=0.2))
        nd = life.step(dc, make_env(dc, None))
        steps.append((bar.day, nd))
        if nd.knocked_in and ki_day is None:
            ki_day = bar.day
        if not nd.alive:
            return nd.end_reason, bar.day, nd.cash, ki_day, steps
    return "truncated", path[-1].day, 0.0, ki_day, steps


def test_seas6_note_is_priced_off_the_strike_with_the_unit_at_the_strike():
    from gpin.events import book_levels, book_terms
    from gpin.models import book_note
    from s2ts import NOTIONAL

    terms = book_terms("seas6", START)
    note = book_note(SEAS_COUPON, SEAS_S0, terms, book="seas6")
    strike, ko, ki = book_levels("seas6", SEAS_S0)
    assert note.strike == note.initial_price == pytest.approx(strike, rel=1e-15)
    assert note.contract_multiplier == pytest.approx(NOTIONAL / strike, rel=1e-15)
    assert note.barrier_config.ko_barrier == pytest.approx(ko, rel=1e-15)
    assert note.barrier_config.ki_barrier == pytest.approx(ki, rel=1e-15)
    assert note.barrier_config.ko_rate == SEAS_COUPON and note.maturity == terms.maturity_years
    assert note.accrual_config.accrual_factors == pytest.approx([SEAS_ELAPSED + t for t in terms.ko_times], rel=1e-15)
    assert note.accrual_config.is_annualized_rebate is False
    assert note.payoff_config.rebate_rate == pytest.approx(SEAS_COUPON * (SEAS_ELAPSED + terms.maturity_years), rel=1e-15)


def test_seas6_pays_the_seasoned_ko_cash_on_the_library_lifecycle():
    """AC4b: a fired KO pays ``coupon x (1.5 + t) x NOTIONAL``, ``t`` the years from the booking day."""
    from gpin.events import book_terms
    from s2ts import NOTIONAL

    terms = book_terms("seas6", START)
    for i in (0, 2, 4):
        ko = terms.ko_dates[i]
        reason, end, cash, ki, _ = seasoned_life(seas6_path(terms.maturity_date, {ko: 125.0}))
        assert (reason, end, ki) == ("ko", ko, None)
        assert cash == pytest.approx(SEAS_COUPON * (SEAS_ELAPSED + (ko - START).days / 365.0) * NOTIONAL, rel=1e-12)
        assert cash == pytest.approx(SEAS_COUPON * (SEAS_ELAPSED + terms.ko_times[i]) * NOTIONAL, rel=1e-12)
    # a knocked-in note that later knocks out pays the KO cash as usual
    ko3 = terms.ko_dates[2]
    days = trading_days(START, terms.maturity_date)
    reason, end, cash, ki, _ = seasoned_life(seas6_path(terms.maturity_date, {days[3]: 90.0, ko3: 125.0}))
    assert (reason, end, ki) == ("ko", ko3, days[3])
    assert cash == pytest.approx(SEAS_COUPON * (SEAS_ELAPSED + terms.ko_times[2]) * NOTIONAL, rel=1e-12)


def test_seas6_pays_coupon_over_the_full_24_months_at_a_maturity_without_knock_in():
    """AC4b: never knocked in, the maturity pays ``coupon x (1.5 + T) x NOTIONAL`` (about ``coupon x 2.0``)."""
    from gpin.events import book_terms
    from s2ts import NOTIONAL

    terms = book_terms("seas6", START)
    reason, end, cash, ki, _ = seasoned_life(seas6_path(terms.maturity_date, {terms.maturity_date: 110.0}))
    assert (reason, end, ki) == ("maturity", terms.maturity_date, None)
    assert cash == pytest.approx(SEAS_COUPON * (SEAS_ELAPSED + terms.maturity_years) * NOTIONAL, rel=1e-12)
    assert cash == pytest.approx(SEAS_COUPON * 2.0 * NOTIONAL, rel=2e-3)
    # the same cash as a KO on the maturity day (the last observation)
    reason, _, ko_cash_at_maturity, _, _ = seasoned_life(seas6_path(terms.maturity_date, {terms.maturity_date: 130.0}))
    assert reason == "ko" and ko_cash_at_maturity == pytest.approx(cash, rel=1e-12)


def test_seas6_pays_the_put_loss_on_the_strike_at_a_knocked_in_maturity():
    """AC4b: knocked in and not knocked out, the maturity pays ``max(0, strike - S_T) x NOTIONAL / strike`` to the
    seller (negative holder-side cash); a knocked-in note whose last close is at or above the strike is a KO."""
    from gpin.events import book_levels, book_terms
    from s2ts import NOTIONAL

    terms = book_terms("seas6", START)
    strike, _, _ = book_levels("seas6", SEAS_S0)
    days = trading_days(START, terms.maturity_date)
    for s_t in (100.0, 93.0, 120.0, 60.0):
        reason, end, cash, ki, _ = seasoned_life(seas6_path(terms.maturity_date, {days[10]: 93.0, terms.maturity_date: s_t}))
        assert (reason, end, ki) == ("maturity", terms.maturity_date, days[10])
        assert cash == pytest.approx(-(strike - s_t) / strike * NOTIONAL, rel=1e-12)
        assert cash < 0.0
    # a knocked-in note finishing above the strike is called, not put
    reason, _, cash, _, _ = seasoned_life(seas6_path(terms.maturity_date, {days[10]: 93.0, terms.maturity_date: 130.0}))
    assert reason == "ko" and cash > 0.0


def _seasoned_engine_price(engine, until: date, last_spot: float, force_ki: bool, extra: dict = None):
    """QUAD V2 price (holder side) of the seas6 note booked on ``START``, valued on ``until`` after the library
    lifecycle was stepped there, with the last close at ``last_spot``."""
    from quantark.param.div.dividend_yield import ContinuousDividendYield
    from gpin.events import book_terms
    from gpin.models import MODELS, book_note
    from s2ts.note import NoteLife, make_env

    terms = book_terms("seas6", START)
    life = NoteLife(book_note(SEAS_COUPON, SEAS_S0, terms, book="seas6"), START)
    nd = None
    overrides = dict(extra or {})
    overrides[until] = last_spot
    for bar in seas6_path(until, overrides):
        dc = SimpleNamespace(day=bar.day, spot=bar.spot, mo=SimpleNamespace(vol=0.2))
        nd = life.step(dc, make_env(dc, ContinuousDividendYield(0.02)))
        assert nd.alive
    state = MODELS["kin"].lifecycle(life.lifecycle) if force_ki else life.lifecycle
    return float(engine.price(nd.product, make_env(dc, ContinuousDividendYield(0.02)), lifecycle_state=state)), terms


def test_quad_v2_prices_the_seasoned_ko_cash_one_day_before_a_certain_ko(engine):
    """The engine reads the same contract: with the spot far above the KO level the day before the first observation
    the price is the discounted seasoned KO cash."""
    from gpin.events import book_terms
    from s2ts import NOTIONAL, RATE

    terms = book_terms("seas6", START)
    ko1 = terms.ko_dates[0]
    prev = trading_days(START, ko1)[-2]
    price, _ = _seasoned_engine_price(engine, prev, 200.0, force_ki=False)
    cash = SEAS_COUPON * (SEAS_ELAPSED + terms.ko_times[0]) * NOTIONAL
    discount = math.exp(-RATE * (ko1 - prev).days / 365.0)
    assert price == pytest.approx(cash * discount, rel=1e-3)
    assert 1e4 * cash / NOTIONAL > 3000.0                       # a seasoned cash (about 30% of notional), not 1 month's


def test_quad_v2_prices_the_seasoned_maturity_payments_one_day_before_maturity(engine):
    """Never knocked in, mid-range spot the day before maturity: the discounted ``coupon x (1.5 + T) x NOTIONAL``;
    forced knocked in, the discounted put loss on the strike."""
    from gpin.events import book_levels, book_terms
    from s2ts import NOTIONAL, RATE

    terms = book_terms("seas6", START)
    prev = trading_days(START, terms.maturity_date)[-2]
    discount = math.exp(-RATE * (terms.maturity_date - prev).days / 365.0)
    price, _ = _seasoned_engine_price(engine, prev, 110.0, force_ki=False)
    cash = SEAS_COUPON * (SEAS_ELAPSED + terms.maturity_years) * NOTIONAL
    assert price == pytest.approx(cash * discount, rel=2e-3)
    strike, _, _ = book_levels("seas6", SEAS_S0)
    price_ki, _ = _seasoned_engine_price(engine, prev, 100.0, force_ki=True)
    loss = (100.0 * math.exp((RATE - 0.02) * (terms.maturity_date - prev).days / 365.0) - strike) / strike * NOTIONAL
    assert price_ki == pytest.approx(loss * discount, rel=5e-3)
    assert price_ki < 0.0


# ---- 00_ki_gap_probe.py (AC4c): the KI gap and the hedge step by barrier and remaining life ----

KI_GAP_JSON = STUDY_DIR / "results" / "ki_gap_probe.json"


@pytest.fixture(scope="module")
def probe_stage():
    return _load_stage("00_ki_gap_probe.py")


def test_ki_gap_probe_json_pins_the_75_row_to_the_nearest_bp_against_a_recomputation(probe_stage, engine):
    """AC4c: the committed rows at 24 and 3 months left for the 75% barrier equal a recomputation to the nearest bp (and
    the hedge step to the nearest point); the fixed inputs are the pre-stated ones."""
    data = json.loads(KI_GAP_JSON.read_text(encoding="utf-8"))
    assert data["inputs"]["coupon"] == 0.20 and data["inputs"]["vol"] == 0.25 and data["inputs"]["q"] == 0.05
    assert data["inputs"]["above_barrier"] == 0.03 and data["inputs"]["notional"] == 1e7
    assert data["inputs"]["r"] == load_prereg()["note"]["rate"]
    assert data["barriers"] == [0.75, 0.85, 0.90] and data["months_left"] == [1, 3, 6, 12, 18, 24]
    rows = {(r["ki_pct"], r["months_left"]): r for r in data["rows"]}
    assert len(rows) == len(data["rows"]) == 18
    for months in (24, 3):
        fresh = probe_stage.probe_cell(engine, 0.75, months)
        stored = rows[(0.75, months)]
        assert round(stored["gap_bp"]) == round(fresh["gap_bp"]), months
        assert round(stored["hedge_step_pct"]) == round(fresh["hedge_step_pct"]), months
        assert stored["gap_bp"] == pytest.approx(fresh["gap_bp"], rel=1e-9)
    assert round(rows[(0.75, 24)]["gap_bp"]) == 9 and round(rows[(0.75, 3)]["gap_bp"]) == 384
    assert "seller" in data["seller_view"] and "holder" not in json.dumps(data).lower()


def test_ki_gap_probe_gap_falls_as_life_lengthens_and_as_the_barrier_rises():
    data = json.loads(KI_GAP_JSON.read_text(encoding="utf-8"))
    gap = {(r["ki_pct"], r["months_left"]): r["gap_bp"] for r in data["rows"]}
    step = {(r["ki_pct"], r["months_left"]): r["hedge_step_pct"] for r in data["rows"]}
    for k in data["barriers"]:
        for a, b in zip(data["months_left"], data["months_left"][1:]):
            assert gap[(k, a)] >= gap[(k, b)] and step[(k, a)] >= step[(k, b)], (k, a, b)
    assert all(gap[(0.75, a)] > gap[(0.75, b)] for a, b in zip(data["months_left"], data["months_left"][1:]))
    for m in data["months_left"]:
        for lo, hi in zip(data["barriers"], data["barriers"][1:]):
            assert gap[(lo, m)] >= gap[(hi, m)], (lo, hi, m)
    assert all(gap[(0.75, m)] > gap[(0.90, m)] for m in data["months_left"])
    assert all(g >= 0.0 for g in gap.values())
    assert gap[(0.90, 24)] < 0.05 < gap[(0.75, 3)]


def test_ki_gap_probe_uses_the_lifecycle_state_the_engine_reads(probe_stage, engine):
    """The knocked-in side is priced with a lifecycle-state copy forced to knocked-in (as the ``kin`` model), never a
    product attribute: the product the two sides share has no ``_otc_lifecycle_knocked_in``."""
    alive, kin, spots = probe_stage.price_both(engine, 0.75, 3)
    assert alive["price"][1] > kin["price"][1]
    import inspect

    assert "_otc_lifecycle_knocked_in" not in inspect.getsource(probe_stage)


def _gap_bp_at_three_percent_above_ki(true_ladder, kin_ladder, spot, ki_level) -> float:
    """``true`` minus ``kin`` engine price (bp of notional) read on the ladder at 3% above the KI barrier."""
    from s2ts import NOTIONAL

    spots = spot * (1.0 + np.asarray(gpin.LADDER))
    target = 1.03 * ki_level
    assert spots[0] <= target <= spots[-1]
    return 1.0e4 * float(np.interp(target, spots, true_ladder["engine_price"])
                         - np.interp(target, spots, kin_ladder["engine_price"])) / NOTIONAL


def test_the_ki_pin_is_a_property_of_a_short_remaining_life(real_chains, real_anchors, book_scans, engine):
    """AC4b on real notes. A seas6 booking (6 months left) with the spot 3% above its KI barrier has a ``true`` minus
    ``kin`` engine price above 100 bp of notional (the pin exists); a real ki75 note in the first month of its life
    (23 months left) on its first KI-window day, 3% above the barrier, has under 30 bp."""
    from gpin.events import book_levels, book_terms
    from gpin.models import MODELS
    from gpin.pricing import model_ladder, price_note, solve_book_coupon

    rec = next(s for s in book_scans["seas6"] if s.inception >= date(2024, 3, 1))
    dc0 = next(c for c in real_chains if c.day == rec.inception)
    coupon = solve_book_coupon(dc0, engine, "seas6")
    life, dc, env, nd, _ = _step_life(real_chains, real_anchors, rec.inception, coupon, rec.inception, book="seas6")
    terms = book_terms("seas6", rec.inception)
    spots = dc.spot * (1.0 + np.asarray(gpin.LADDER))
    ladders = {m: model_ladder(engine, life, MODELS[m], terms, dc.spot, coupon, rec.inception, rec.inception, env, spots,
                               None, book="seas6") for m in ("true", "kin")}
    _, _, ki = book_levels("seas6", dc.spot)
    seasoned_gap = _gap_bp_at_three_percent_above_ki(ladders["true"], ladders["kin"], dc.spot, ki)
    assert seasoned_gap > 100.0, seasoned_gap

    young = min(((a.start - s.inception).days, s.inception, a.start) for s in book_scans["ki75"] if s.anchored
                for a in s.ki_approaches)
    age, inception, day = young
    assert age <= 31                                              # the first month of its life
    note = next(s for s in book_scans["ki75"] if s.inception == inception)
    days = [c.day for c in real_chains]
    states = price_note(inception, PROBE_COUPON, real_chains, real_anchors, note, book="ki75", engine=engine,
                        price_days={day}, max_days=days.index(day) - days.index(inception))
    rec_day = states.days[-1]
    assert rec_day["date"] == day.isoformat() and {"true", "kin"} <= set(rec_day["states"])
    spot0 = next(c.spot for c in real_chains if c.day == inception)
    young_gap = _gap_bp_at_three_percent_above_ki(rec_day["states"]["true"], rec_day["states"]["kin"], rec_day["spot"],
                                                  0.75 * spot0)
    assert young_gap < 30.0, young_gap
    assert young_gap < seasoned_gap / 3.0


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
    assert stage3.sample_notes(sample, books=["seas6"]) == [n for n in notes if n.startswith("seas6:")]
    assert sizes == {"ki75": 153, "seas6": 168}
    by_book = {b: {s.inception: s for s in book_scans[b]} for b in BOOKS}
    summary = stage3.plan(real_chains, by_book, notes, workers=4)
    cal = calendar_for(real_chains)
    for b in BOOKS:
        per_note = [expected_solves(real_chains, by_book[b][parse_note_id(n)[1]], cal, book=b)
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


def test_stage3_quick_notes_are_two_ki75_notes_and_a_seas6_note_that_enters_its_ki_window_early(stage3):
    sample = stage3.load_sample()
    picks = stage3.quick_notes(sample)
    assert len(picks) == 3
    assert picks[0] == f"ki75:{sample['books']['ki75']['headline'][0]}" and picks[1] == "ki75:2024-02-29"
    assert "2024-02-29" in {p["inception"] for p in sample["books"]["ki75"]["pin"]}
    book, inception = picks[2].split(":")
    blk = sample["books"]["seas6"]
    assert book == "seas6" and inception in blk["notes"]
    starts = [r["start"] for r in blk["ki_approaches"] if r["inception"] == inception]
    assert starts and (date.fromisoformat(min(starts)) - date.fromisoformat(inception)).days <= stage3.QUICK_CALENDAR_DAYS
    # the earliest booking that does so
    early = sorted(r["inception"] for r in blk["ki_approaches"]
                   if (date.fromisoformat(r["start"]) - date.fromisoformat(r["inception"])).days <= stage3.QUICK_CALENDAR_DAYS)
    assert inception == early[0]


def test_stage3_notes_failed_is_a_quantark_exception(stage3):
    from quantark.util.exceptions import QuantArkException

    assert issubclass(stage3.NotesFailed, QuantArkException)


def test_stage3_pool_records_failures_and_goes_on_and_refuses_a_second_instance(stage3, tmp_path):
    from s2ts.runner import OutDirLocked, _acquire_lock, _release_lock

    notes = ["ki75:2024-02-29", "seas6:2024-02-29", "seas6:2024-03-04"]

    def job(nid):
        if nid == notes[1]:
            raise RuntimeError("boom")
        return {"note": nid, "inception": nid.split(":")[1], "reused": False, "seconds": 1.0, "solves": 7,
                "live_days": 3}

    (tmp_path / "stale.json.tmp").write_text("x")
    lines = []
    with pytest.raises(stage3.NotesFailed, match="1 of 3 notes failed"):
        stage3.run_notes(notes, {}, tmp_path, "fp", workers=0, job=job, log=lines.append)
    err = json.loads((tmp_path / "seas6__2024-02-29.error.json").read_text())
    assert err["note"] == "seas6:2024-02-29" and "boom" in err["error"] and "Traceback" in err["traceback"]
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
    assert [p.name for p in ok] == ["ki75__2024-02-29.json", "seas6__2024-02-29.json", "seas6__2024-03-04.json"]
    assert not list(tmp_path.glob("*.error.json"))


# ---------------------------------------------------------------------------
# T3b — books in the models and the pricing pass
# ---------------------------------------------------------------------------


def test_levels_follow_the_books_for_both_directions_and_the_models():
    from gpin.events import book_levels

    m = _models()
    strike, ko, ki = book_levels("seas6", S0_FIXED)
    assert (strike, ko) == (pytest.approx(S0_FIXED / 1.07 / 0.75, rel=1e-15), pytest.approx(S0_FIXED / 1.07 / 0.75, rel=1e-15))
    assert m.ki_level(m.MODELS["true"], S0_FIXED, book="seas6") == pytest.approx(S0_FIXED / 1.07, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m100"], S0_FIXED, book="seas6") == pytest.approx(0.99 * S0_FIXED / 1.07, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_m200"], S0_FIXED, book="seas6") == pytest.approx(0.98 * S0_FIXED / 1.07, rel=1e-15)
    assert m.ki_level(m.MODELS["ki_p100"], S0_FIXED, book="seas6") == pytest.approx(1.01 * S0_FIXED / 1.07, rel=1e-15)
    assert m.ki_level(m.MODELS["kin"], S0_FIXED, book="seas6") == pytest.approx(S0_FIXED / 1.07, rel=1e-15)
    from gpin.events import book_terms

    terms = book_terms("seas6", date(2024, 1, 2))
    n = len(terms.ko_dates)
    day = date(2024, 1, 31)
    assert m.ko_levels(m.MODELS["true"], terms, day, None, S0_FIXED, book="seas6") == pytest.approx([ko] * n, rel=1e-15)
    for direction in (-1, +1):
        shifted = m.ko_levels(m.MODELS["ko_s200"], terms, day, direction, S0_FIXED, book="seas6")
        assert shifted == pytest.approx([ko * (1.0 + direction * 0.02)] * n, rel=1e-15)
        wrong = m.ko_levels(m.MODELS["ko_w100"], terms, day, direction, S0_FIXED, book="seas6")
        assert wrong == pytest.approx([ko * (1.0 - direction * 0.01)] * n, rel=1e-15)
    surv = m.ko_levels(m.MODELS["surv"], terms, day, None, S0_FIXED, book="seas6")
    assert [i for i, x in enumerate(surv) if x > ko * 10] == [0]                 # exactly the next observation
    assert surv[0] == pytest.approx(m.SURV_OFF_MULTIPLE * strike, rel=1e-15)
    assert surv[1:] == pytest.approx([ko] * (n - 1), rel=1e-15)
    later = m.ko_levels(m.MODELS["surv"], terms, terms.ko_dates[1], None, S0_FIXED, book="seas6")
    assert [i for i, x in enumerate(later) if x > ko * 10] == [2]


def test_book_note_is_build_note_at_75_and_the_models_carry_the_seasoned_contract():
    from s2ts.note import build_note, build_terms

    from gpin.events import book_levels, book_terms

    m = _models()
    terms = build_terms(PROBE_INCEPTION)
    ref = build_note(PROBE_COUPON, S0_FIXED, terms)
    same = m.book_note(PROBE_COUPON, S0_FIXED, terms, book="ki75")
    assert same.barrier_config == ref.barrier_config and same.maturity == ref.maturity
    assert same.payoff_config == ref.payoff_config and same.accrual_config == ref.accrual_config
    assert (same.strike, same.initial_price, same.contract_multiplier) == (ref.strike, ref.initial_price, ref.contract_multiplier)
    assert same.accrual_config.accrual_factors is None                    # the 2Y product has no seasoning
    # a seasoned model product: the book's levels per observation and the seasoned accrual factors
    st = book_terms("seas6", PROBE_INCEPTION)
    strike, ko, ki = book_levels("seas6", S0_FIXED)
    prod = m.MODELS["ko_s100"].product(st, S0_FIXED, PROBE_COUPON, date(2024, 3, 5), -1, book="seas6")
    assert list(prod.barrier_config.ko_barrier) == pytest.approx([ko * 0.99] * 6, rel=1e-15)
    assert prod.barrier_config.ki_barrier == pytest.approx(ki, rel=1e-15) and prod.strike == pytest.approx(strike, rel=1e-15)
    assert prod.accrual_config.accrual_factors == pytest.approx([1.5 + t for t in st.ko_times], rel=1e-15)
    shifted = m.MODELS["ki_m100"].product(st, S0_FIXED, PROBE_COUPON, date(2024, 3, 5), None, book="seas6")
    assert shifted.barrier_config.ki_barrier == pytest.approx(0.99 * ki, rel=1e-15)


def test_the_windows_follow_the_books_levels():
    from gpin.pricing import in_ki_window, in_ko_window

    assert in_ki_window(78.75, 100.0, False, book="ki75") and not in_ki_window(78.8, 100.0, False, book="ki75")
    ki = 100.0 / 1.07                                           # seas6 booked at 100: KI 93.458, strike = KO 124.61
    assert in_ki_window(ki * 1.05, 100.0, False, book="seas6") and not in_ki_window(ki * 1.0501, 100.0, False, book="seas6")
    assert in_ki_window(ki * 0.9, 100.0, False, book="seas6")
    assert not in_ki_window(ki * 1.01, 100.0, True, book="seas6")           # a knocked-in note has no KI window
    ko = ki / 0.75
    assert in_ko_window(ko * 0.9, 100.0, 10, book="seas6") and not in_ko_window(ko * 0.8999, 100.0, 10, book="seas6")
    assert in_ko_window(ko * 1.1, 100.0, 1, book="seas6") and not in_ko_window(ko * 1.11, 100.0, 1, book="seas6")
    assert not in_ko_window(ko, 100.0, 11, book="seas6") and not in_ko_window(ko, 100.0, None, book="seas6")
    assert in_ko_window(100.0, 100.0, 5, book="ki75") and not in_ko_window(100.0, 100.0, 5, book="seas6")


def test_the_ki_window_is_five_percent():
    from gpin import pricing

    assert pricing.KI_BAND == 0.05 == load_prereg()["windows"]["ki_band"]


def test_book_coupons_ki75_equals_solve_coupon_and_seas6_uses_the_ki75_coupon_of_its_booking_day(real_chains, engine):
    from gpin.models import book_note
    from gpin.pricing import solve_book_coupon
    from quantark.param import FlatRateCurve
    from s2ts import NOTIONAL, RATE
    from s2ts.note import build_terms, make_env, solve_coupon
    from s2ts.tails import build_curve

    dc = next(c for c in real_chains if c.day == PROBE_INCEPTION)
    c75 = solve_book_coupon(dc, engine, "ki75")
    assert c75 == solve_coupon(dc, engine)                               # the same method, the same number
    assert solve_book_coupon(dc, engine, "seas6") == c75                 # not solved again for the seasoned contract
    env = make_env(dc, build_curve(dc, "hold_forward", None).to_dividend_yield(FlatRateCurve(RATE)))
    pv = float(engine.price(book_note(c75, dc.spot, build_terms(dc.day), book="ki75"), env))
    assert abs(pv) <= 1e-6 * NOTIONAL                                    # reprices to zero under hold-forward


@pytest.fixture(scope="module")
def seas_run(real_chains, real_anchors, book_scans):
    """A real seasoned booking stepped (library lifecycle) to the day after its second KO observation: its aged
    state, plus the booking's scan record and terms."""
    from types import SimpleNamespace

    from gpin.events import book_terms

    rec = next(s for s in book_scans["seas6"] if s.end_reason == "maturity" and s.ki_date is None
               and s.inception >= date(2024, 3, 1))
    terms = book_terms("seas6", rec.inception)
    day = next(c.day for c in real_chains if c.day > terms.ko_dates[1])
    life, dc, env, nd, _ = _step_life(real_chains, real_anchors, rec.inception, PROBE_COUPON, day, book="seas6")
    return SimpleNamespace(scan=rec, terms=terms, day=day, life=life, dc=dc, env=env, nd=nd,
                           s0=next(c.spot for c in real_chains if c.day == rec.inception))


def test_the_aged_seasoned_variant_keeps_the_seasoned_cash_and_equals_the_trackers_note(seas_run, engine):
    """Ageing past two KO observations takes the first two accrual factors and levels off the front: the surviving
    KO records still pay ``coupon x (1.5 + t) x NOTIONAL`` with ``t`` from the booking day, and the rebuilt ``true``
    variant prices like the tracker's own pricing copy."""
    from gpin.models import MODELS, SURV_OFF_MULTIPLE
    from gpin.pricing import aged_model_product, model_ladder, solve_ladder
    from s2ts import NOTIONAL

    r = seas_run
    assert r.nd.alive and len(r.terms.ko_dates) == 6
    aged = aged_model_product(r.life, MODELS["true"], r.terms, r.s0, PROBE_COUPON, r.day, r.scan.inception, r.env, None,
                              book="seas6")
    records = aged.resolve_ko_observations(r.env)
    assert len(records) == 4
    for rec, t in zip(records, r.terms.ko_times[2:]):
        assert rec.payoff == pytest.approx(PROBE_COUPON * (1.5 + t) * NOTIONAL, rel=1e-12)
    spots = r.dc.spot * (1.0 + np.asarray(gpin.LADDER))
    rebuilt = model_ladder(engine, r.life, MODELS["true"], r.terms, r.s0, PROBE_COUPON, r.day, r.scan.inception, r.env,
                           spots, None, book="seas6")
    tracker = solve_ladder(engine, r.nd.product, r.env, spots, r.life.lifecycle)
    for key in ("engine_price", "engine_delta"):
        assert rebuilt[key] == pytest.approx(tracker[key], abs=1e-9), key
    surv = aged_model_product(r.life, MODELS["surv"], r.terms, r.s0, PROBE_COUPON, r.day, r.scan.inception, r.env, None,
                              book="seas6")
    sr = list(surv.barrier_config.ko_observation_schedule.records)
    strike = r.s0 / 1.07 / 0.75
    assert len(sr) == len(surv.barrier_config.ko_barrier) == len(surv.accrual_config.accrual_factors) == 4
    assert sr[0].barrier == pytest.approx(SURV_OFF_MULTIPLE * strike) and all(x.barrier == pytest.approx(strike) for x in sr[1:])


def test_price_note_builds_every_model_off_the_seasoned_book(real_chains, real_anchors, book_scans, real_scans, engine):
    from gpin.models import MODELS
    from gpin.pricing import aged_model_product, price_note
    from gpin.events import book_levels, book_terms

    rec = next(s for s in book_scans["seas6"] if s.inception >= date(2024, 3, 1))
    states = price_note(rec.inception, 0.2, real_chains, real_anchors, rec, book="seas6", engine=engine,
                        price_days={rec.inception}, max_days=3)
    assert states.book == "seas6" and states.ki_pct == 0.75
    out = states.to_json("fp")
    spot0 = next(c.spot for c in real_chains if c.day == rec.inception)
    strike, ko, ki = book_levels("seas6", spot0)
    assert (out["book"], out["ki_pct"], out["note"]) == ("seas6", 0.75, f"seas6:{rec.inception}")
    assert (out["spot0"], out["strike"], out["ko_level"], out["ki_level"]) == (spot0, strike, ko, ki)
    assert out["spec"] == {"id": "seas6", "ki_pct": 0.75, "first_ko_month": 1, "maturity_months": 6, "elapsed_months": 18,
                           "ki_below_spot": 0.07, "booking": "weekly"}
    assert {"days", "coupon", "inception", "fingerprint", "meta"} <= set(out)           # the keys the ledger reads
    with pytest.raises(StudyDataError, match="not book"):                # the 75 scan is not the seasoned book's
        price_note(rec.inception, 0.2, real_chains, real_anchors,
                   next(s for s in real_scans if s.inception == rec.inception), book="seas6", engine=engine,
                   price_days=set(), max_days=3)
    life, dc, env, nd, _ = _step_life(real_chains, real_anchors, rec.inception, 0.2, rec.inception, book="seas6")
    terms = book_terms("seas6", rec.inception)
    for mid, mult in (("true", 1.0), ("ki_m100", 0.99), ("kin", 1.0)):
        aged = aged_model_product(life, MODELS[mid], terms, spot0, 0.2, rec.inception, rec.inception, env, None,
                                  book="seas6")
        assert aged.barrier_config.ki_barrier == pytest.approx(mult * ki, rel=1e-12), mid


def test_state_files_are_named_by_book(tmp_path):
    from gpin.pricing import states_path

    assert states_path(tmp_path, date(2024, 2, 29), "seas6").name == "seas6__2024-02-29.json"
    assert states_path(tmp_path, date(2024, 2, 29), "ki75").name == "ki75__2024-02-29.json"


def test_coupon_cache_is_stamped_and_re_solved_when_the_stamp_differs(tmp_path):
    from gpin.pricing import book_coupons

    calls = []

    def solver(book, inception):
        calls.append((book, inception))
        return 0.1 + 0.01 * len(calls)

    path = tmp_path / "coupons.json"
    ids = ["ki75:2024-02-29", "seas6:2024-02-29"]
    first = book_coupons(ids, path, "stampA", solver)
    assert len(calls) == 2 and first["ki75:2024-02-29"] != first["seas6:2024-02-29"]
    assert book_coupons(ids, path, "stampA", solver) == first and len(calls) == 2         # reused
    assert book_coupons(ids[:1], path, "stampA", solver) == {ids[0]: first[ids[0]]}
    again = book_coupons(ids, path, "stampB", solver)                                      # stamp differs: re-solved
    assert len(calls) == 4 and again != first
    assert json.loads(path.read_text())["stamp"] == "stampB"
    path.write_text("{ broken")                                                            # unreadable cache: re-solve
    book_coupons(ids[:1], path, "stampB", solver)
    assert len(calls) == 5


def test_coupon_cache_is_written_after_each_solved_coupon(tmp_path):
    """A run that dies on the third coupon keeps the first two on disk."""
    from gpin.pricing import book_coupons

    ids = [f"ki75:2024-02-{d:02d}" for d in (27, 28, 29)]
    calls = []

    def solver(book, inception):
        calls.append(inception)
        if len(calls) == 3:
            raise RuntimeError("killed")
        return 0.1 * len(calls)

    path = tmp_path / "coupons.json"
    with pytest.raises(RuntimeError):
        book_coupons(ids, path, "stampA", solver)
    assert json.loads(path.read_text()) == {"stamp": "stampA", "coupons": {ids[0]: 0.1, ids[1]: 0.2}}
    assert not list(tmp_path.glob("*.tmp"))
    resumed = book_coupons(ids[:2], path, "stampA", solver)                                # nothing is solved again
    assert resumed == {ids[0]: 0.1, ids[1]: 0.2} and len(calls) == 3


def test_coupon_stamp_is_the_full_pricing_fingerprint(tmp_path):
    import shutil

    from gpin import pricing

    s2 = tmp_path / "s2ts"
    shutil.copytree(S2TS_DIR / "s2ts", s2, ignore=shutil.ignore_patterns("__pycache__"))
    pkg = tmp_path / "gpin"
    shutil.copytree(STUDY_DIR / "gpin", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    kw = dict(s2ts_dir=s2, pkg_dir=pkg, data_digest="d")
    base = pricing.coupon_stamp(**kw)
    assert base == pricing.pricing_fingerprint(**kw)
    assert pricing.coupon_stamp(**{**kw, "data_digest": "e"}) != base
    for name in ("models.py", "pricing.py", "events.py"):       # an edit to any pricing-side file re-solves the coupons
        before = pricing.coupon_stamp(**kw)
        (pkg / name).write_text((pkg / name).read_text() + "\n# edit\n")
        assert pricing.coupon_stamp(**kw) != before, name
    (s2 / "note.py").write_text((s2 / "note.py").read_text() + "\n# edit\n")
    assert pricing.coupon_stamp(**kw) != pricing.pricing_fingerprint(**{**kw, "s2ts_dir": S2TS_DIR / "s2ts"})


def test_the_fingerprint_names_the_quantark_version():
    import quantark

    from gpin import pricing

    assert pricing.library_versions()["quantark"] == quantark.__version__
    assert {"python", "numpy", "scipy"} <= set(pricing.library_versions())


def test_state_file_reuse_needs_fingerprint_coupon_and_a_complete_run(tmp_path):
    from gpin import pricing

    path = tmp_path / "seas6__2024-02-29.json"
    base = {"note": "seas6:2024-02-29", "inception": "2024-02-29", "fingerprint": "A", "coupon": 0.2, "days": [],
            "meta": {"max_days": None, "price_days": None}}
    pricing.write_states_atomic(path, base)
    assert pricing.read_current(path, "seas6:2024-02-29", "A", 0.2) == base
    assert pricing.read_current(path, "seas6:2024-02-29", "B", 0.2) is None            # fingerprint
    assert pricing.read_current(path, "seas6:2024-02-29", "A", 0.2000001) is None      # a re-solved coupon
    assert pricing.read_current(path, "ki75:2024-02-29", "A", 0.2) is None            # another book's note
    for key, value in (("max_days", 5), ("price_days", ["2024-03-01"])):
        pricing.write_states_atomic(path, {**base, "meta": {**base["meta"], key: value}})
        assert pricing.read_current(path, "seas6:2024-02-29", "A", 0.2) is None, key   # a diagnostic run is never reused
    pricing.write_states_atomic(path, {k: v for k, v in base.items() if k != "meta"})
    assert pricing.read_current(path, "seas6:2024-02-29", "A", 0.2) is None            # no provenance, no reuse


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
    life = NoteLife(book_note(PROBE_COUPON, real_chains[i0].spot, build_terms(PROBE_INCEPTION), book="ki75"), PROBE_INCEPTION)
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
                           run.env, spots, None, book="ki75")
    tracker = solve_ladder(engine, run.nd.product, run.env, spots, run.life.lifecycle)
    for key in ("engine_price", "engine_delta"):
        assert rebuilt[key] == pytest.approx(tracker[key], abs=1e-9), key      # the whole ladder
    surv = aged_model_product(run.life, MODELS["surv"], terms, run.s0, PROBE_COUPON, AGED_DAY, AGED_INCEPTION,
                              run.env, None, book="ki75")
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
    states = pricing.price_note(note.inception, PROBE_COUPON, real_chains, real_anchors, note, engine=engine, book="ki75")
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
    import dataclasses

    from gpin.events import BOOKS, BookSpec, note_id, parse_note_id

    assert [f.name for f in dataclasses.fields(BookSpec)] == [
        "id", "ki_pct", "first_ko_month", "maturity_months", "elapsed_months", "ki_below_spot", "booking"]
    assert BOOKS == {"ki75": BookSpec("ki75", 0.75, 3, 24, 0, None, "sample"),
                     "seas6": BookSpec("seas6", 0.75, 1, 6, 18, 0.07, "weekly")}
    assert {k: dataclasses.asdict(v) for k, v in BOOKS.items()} == {
        k: {"id": k, **v} for k, v in load_prereg()["books"].items()}
    assert note_id("seas6", date(2024, 2, 29)) == "seas6:2024-02-29"
    assert parse_note_id("seas6:2024-02-29") == ("seas6", date(2024, 2, 29))
    for bad in ("2024-02-29", "ki80:2024-02-29", "ki90:2024-02-29", "seas6:2024-13-01"):
        with pytest.raises(StudyDataError):
            parse_note_id(bad)


def test_ki75_terms_and_levels_are_the_product_s():
    from gpin.events import book_levels, book_terms
    from s2ts.note import build_terms

    assert book_terms("ki75", START) == build_terms(START)
    assert book_levels("ki75", 6000.0) == (6000.0, 6000.0, pytest.approx(0.75 * 6000.0, rel=1e-15))
    with pytest.raises(StudyDataError):
        book_terms("ki90", START)
    with pytest.raises(StudyDataError):
        book_levels("ki90", 100.0)


def test_seas6_terms_are_six_monthly_observations_and_levels_follow_the_booking_spot():
    """AC4b, hand-computed: booked 2024-01-02, KO on the month anniversaries 1..6 rolled to the next SSE trading day
    (2024-03-02 is a Saturday, 2024-05-02 is inside the Labour Day closure), the sixth is the maturity; KI daily after
    the booking day; the KI level 7% below the spot, the strike and KO level at KI / 0.75."""
    from gpin.events import book_levels, book_terms

    terms = book_terms("seas6", START)
    assert terms.ko_dates == (date(2024, 2, 2), date(2024, 3, 4), date(2024, 4, 2), date(2024, 5, 6),
                              date(2024, 6, 3), date(2024, 7, 2))
    assert terms.maturity_date == date(2024, 7, 2) == terms.ko_dates[-1] and terms.inception == START
    assert terms.maturity_years == (date(2024, 7, 2) - START).days / 365.0
    assert terms.ko_times == tuple((d - START).days / 365.0 for d in terms.ko_dates)
    assert terms.ki_dates == tuple(trading_days(date(2024, 1, 3), date(2024, 7, 2)))
    assert START not in terms.ki_dates and terms.ki_times[0] == 1 / 365.0
    strike, ko, ki = book_levels("seas6", 6000.0)
    assert ki == pytest.approx(6000.0 / 1.07, rel=1e-15)
    assert strike == ko == pytest.approx(ki / 0.75, rel=1e-15)
    assert ki / strike == pytest.approx(0.75, rel=1e-15)


def seas6_path(end: date, overrides: dict = None):
    """A flat 100 booking path: the KI level is 93.457 (7% below), the strike and KO level 124.61."""
    return make_path(end, overrides, base=100.0)


def test_seas6_scan_knock_in_maturity_without_knock_in_and_ko_on_hand_computed_dates():
    from gpin.events import book_terms

    terms = book_terms("seas6", START)
    maturity = terms.maturity_date
    days = trading_days(START, maturity)
    # never touches either level: the note matures on the sixth observation, no KI approach (100 / 93.46 - 1 = 7%)
    sc = scan_one(seas6_path(maturity + timedelta(days=5)), START, "seas6")
    assert (sc.ko_date, sc.ki_date, sc.end_reason, sc.end_date, sc.ko_observations) == (None, None, "maturity", maturity, 6)
    assert sc.ki_approaches == () and sc.book == "seas6" and sc.anchored
    # a knock-in on one close (93.0 <= 93.457), recovered: knocked in, to maturity
    sc = scan_one(seas6_path(maturity, {days[10]: 93.0}), START, "seas6")
    assert (sc.ko_date, sc.ki_date, sc.end_reason, sc.end_date) == (None, days[10], "maturity", maturity)
    assert [(a.start, a.end, a.knocked_in) for a in sc.ki_approaches] == [(days[10], days[10], True)]
    # a hair above the level does not knock in; it is a near approach (within 3% and inside the 5% band)
    sc = scan_one(seas6_path(maturity, {days[10]: 93.46}), START, "seas6")
    assert sc.ki_date is None and [(a.start, a.end, a.knocked_in) for a in sc.ki_approaches] == [(days[10], days[10], False)]
    # a fired KO at the first observation (month 1): 125 >= 124.61; 124.0 on the day before is a miss, not a pin event
    ko1 = terms.ko_dates[0]
    prev = trading_days(START, ko1)[-2]
    sc = scan_one(seas6_path(maturity, {ko1: 125.0, prev: 124.0}), START, "seas6")
    assert (sc.ko_date, sc.ki_date, sc.end_reason, sc.end_date, sc.ko_observations) == (ko1, None, "ko", ko1, 1)
    assert sc.ko_events == ()                                  # a seasoned book records no KO pin events
    # knocked in, then called at the third observation (a knocked-in note can still knock out)
    ko3 = terms.ko_dates[2]
    sc = scan_one(seas6_path(maturity, {days[3]: 90.0, ko3: 125.0}), START, "seas6")
    assert (sc.ko_date, sc.ki_date, sc.end_reason, sc.end_date, sc.ko_observations) == (ko3, days[3], "ko", ko3, 3)
    # spot at or above the KO level on the maturity day is a KO, not a maturity
    sc = scan_one(seas6_path(maturity, {maturity: 130.0}), START, "seas6")
    assert (sc.end_reason, sc.ko_date) == ("ko", maturity)
    # a path that ends before the maturity is truncated
    sc = scan_one(seas6_path(days[-3]), START, "seas6")
    assert sc.end_reason == "truncated" and sc.end_date == days[-3] and sc.ko_observations == 5


def test_seas6_scan_books_the_first_anchored_chain_day_of_each_iso_week():
    from gpin.events import scan

    days = trading_days(START, START + timedelta(days=40))
    path = [Bar(d, S0_SYN) for d in days]
    by_week = {}
    for d in days:
        by_week.setdefault(d.isocalendar()[:2], []).append(d)
    weeks = sorted(by_week)
    # week 1: the Monday is not anchored, so the Tuesday is booked; week 2: the Monday; week 3: nothing anchored
    anchors = {by_week[weeks[0]][1]: 0.0, by_week[weeks[0]][2]: 0.0, by_week[weeks[1]][0]: 0.0, by_week[weeks[1]][3]: 0.0,
               by_week[weeks[3]][4]: 0.0}
    scans = scan(path, anchors, "seas6")
    assert [s.inception for s in scans] == [by_week[weeks[0]][1], by_week[weeks[1]][0], by_week[weeks[3]][4]]
    assert all(s.anchored and s.book == "seas6" for s in scans)
    with pytest.raises(StudyDataError):
        scan(path[:3] + path[4:], anchors, "seas6")           # a missing trading day still fails closed
    with pytest.raises(StudyDataError):
        scan(path, anchors, "ki90")                            # an unknown book


def test_select_seas6_lists_every_booking_as_headline_with_no_pin_set_and_no_ko_events():
    from gpin.events import select

    incs = [date(2024, 1, 2) + timedelta(days=7 * i) for i in range(5)]
    t = date(2025, 3, 5)
    ap = (date(2024, 2, 5), date(2024, 2, 9))
    scans = [fake_scan(i, ki=[(*ap, i == incs[2])], ko=[(t, 0.001, True)], book="seas6") for i in incs]
    scans.append(fake_scan(date(2024, 3, 1), anchored=False, book="seas6"))
    sample = select(scans, "seas6")
    assert sample.headline == tuple(incs) and sample.pin == () and sample.reasons == {}
    assert [type(e).__name__ for e in sample.events] == ["KiApproach"] * 5                 # no KO pin events
    assert select(list(reversed(scans)), "seas6") == sample
    with pytest.raises(StudyDataError):
        select(scans, "ki75")                                                                # not that book's scans
    with pytest.raises(StudyDataError):
        select([fake_scan(incs[0])], "seas6")


def test_sample_fields_match_contract_c2_positionally():
    import dataclasses

    from gpin.events import Sample

    assert [f.name for f in dataclasses.fields(Sample)] == ["headline", "pin", "events", "reasons"]


def test_the_ko_shift_flag_is_a_tolerant_comparison():
    m = _models()
    assert m.HedgeModel("x", "ko", ko_shift=1e-13).needs_ko_dir is False
    assert m.HedgeModel("x", "ko", ko_shift=0.005).needs_ko_dir is True


# ---------------------------------------------------------------------------
# T4 — Black-76 helper, technique interface, ledger
# ---------------------------------------------------------------------------

from scipy.stats import norm as _norm  # noqa: E402

from quantark.util.exceptions import ValidationError  # noqa: E402


def _series(days_to_last=73, forward=5000.0, rate=0.014, last=None, anchor=date(2024, 1, 2)):
    from gpin.board import SeriesInfo

    tenor = days_to_last / 365.0
    return SeriesInfo(last or anchor + timedelta(days=days_to_last), tenor, forward, math.exp(-rate * tenor))


def _bs(info, strike, vol, call=True):
    """Independent Black-76 on the forward: price, forward delta, forward gamma."""
    s = vol * math.sqrt(info.tenor)
    d1 = (math.log(info.forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        price = info.discount * (info.forward * _norm.cdf(d1) - strike * _norm.cdf(d2))
        delta = _norm.cdf(d1)
    else:
        price = info.discount * (strike * _norm.cdf(-d2) - info.forward * _norm.cdf(-d1))
        delta = _norm.cdf(d1) - 1.0
    gamma = _norm.pdf(d1) / (info.forward * s)
    return price, delta, gamma


@pytest.mark.parametrize("call", [True, False])
def test_black_value_and_implied_vol_round_trip(call):
    from gpin import black

    info = _series()
    cp = "C" if call else "P"
    price, _, _ = _bs(info, 5100.0, 0.23, call)
    assert black.value(0.23, cp, 5100.0, info) == pytest.approx(price, rel=1e-12)
    assert black.implied_vol(price, cp, 5100.0, info) == pytest.approx(0.23, abs=1e-9)


def test_black_implied_vol_is_none_when_it_cannot_be_inverted():
    from gpin import black

    info = _series()
    intrinsic = info.discount * (5000.0 - 4500.0)
    assert black.implied_vol(intrinsic - 1.0, "C", 4500.0, info) is None        # below the no-arbitrage bound
    assert black.implied_vol(info.discount * 5000.0 + 1.0, "C", 4500.0, info) is None  # above forward x discount
    assert black.implied_vol(0.0, "C", 4500.0, info) is None
    assert black.implied_vol(10.0, "C", 5100.0, _series(days_to_last=0)) is None  # zero tenor


def test_black_index_delta_and_gamma_scale_the_forward_with_spot():
    from gpin import black

    info = _series()
    spot, strike, vol = 4950.0, 5100.0, 0.21
    _, dfwd, gfwd = _bs(info, strike, vol, True)
    assert black.index_delta(vol, "C", strike, info, spot) == pytest.approx(info.discount * dfwd * info.forward / spot, rel=1e-12)
    _, pfwd, _ = _bs(info, strike, vol, False)
    assert black.index_delta(vol, "P", strike, info, spot) == pytest.approx(info.discount * pfwd * info.forward / spot, rel=1e-12)
    assert black.index_gamma(vol, "C", strike, info, spot) == pytest.approx(
        info.discount * gfwd * (info.forward / spot) ** 2, rel=1e-12)
    # delta and gamma are the derivatives of the price when the forward moves with spot
    h = 0.5

    def price_at(s):
        return black.value_at(vol, "C", strike, info, spot, s, info.tenor)

    assert price_at(spot) == pytest.approx(black.value(vol, "C", strike, info), rel=1e-12)
    assert (price_at(spot + h) - price_at(spot - h)) / (2 * h) == pytest.approx(
        black.index_delta(vol, "C", strike, info, spot), rel=1e-6)
    assert (price_at(spot + h) - 2 * price_at(spot) + price_at(spot - h)) / h ** 2 == pytest.approx(
        black.index_gamma(vol, "C", strike, info, spot), rel=1e-4)


def test_black_value_at_uses_the_time_left_and_the_scaled_forward():
    from gpin import black
    from s2ts import RATE

    info = _series()
    left = 30 / 365.0
    scenario = 5200.0
    spot = 5000.0
    moved = type(info)(info.last_trading_day, left, info.forward * scenario / spot, math.exp(-RATE * left))
    assert black.value_at(0.2, "P", 5000.0, info, spot, scenario, left) == pytest.approx(
        black.value(0.2, "P", 5000.0, moved), rel=1e-12)
    # no time left: the payoff
    assert black.value_at(0.2, "C", 5000.0, info, spot, 5300.0, 0.0) == pytest.approx(info.forward * 5300.0 / spot - 5000.0)


# ---- technique interface (C6) ----

def _ladder_state(price0, delta, gamma, spot):
    """A recorded ``{engine_price, engine_delta, engine_gamma}`` ladder of a quadratic-in-spot price (the 1% bump
    delta is exactly ``delta`` when ``gamma`` is 0)."""
    s = [spot * (1.0 + x) for x in gpin.LADDER]
    return {"engine_price": [price0 + delta * (v - spot) + 0.5 * gamma * (v - spot) ** 2 for v in s],
            "engine_delta": [delta + gamma * (v - spot) for v in s],
            "engine_gamma": [gamma] * len(s)}


def test_target_is_a_frozen_value_with_an_optional_filled_delta():
    from gpin.techniques import Target

    t = Target(1200.0, {("MO2401", "C", 5000.0): 2.0})
    assert t.note_delta == 1200.0 and t.note_delta_if_filled is None and t.legs == {("MO2401", "C", 5000.0): 2.0}
    assert Target(1.0, {}, note_delta_if_filled=2.0).note_delta_if_filled == 2.0
    with pytest.raises(Exception):
        t.note_delta = 5.0


def test_hedge_delta_is_the_central_bump_delta_at_the_ladder_centre():
    from gpin.techniques import hedge_delta

    spot = 5000.0
    st = _ladder_state(100.0, 1234.0, 0.0, spot)
    assert hedge_delta(st, spot) == pytest.approx(1234.0, rel=1e-12)                 # models.bump = 1%: indices 11 and 13
    curved = _ladder_state(100.0, 1234.0, 0.5, spot)
    curved["engine_price"][15] += 1.0e5                                               # skews the wide bump only
    s = [spot * (1.0 + x) for x in gpin.LADDER]
    expect = (curved["engine_price"][13] - curved["engine_price"][11]) / (s[13] - s[11])
    assert hedge_delta(curved, spot) == pytest.approx(expect, rel=1e-12)
    wide = (curved["engine_price"][15] - curved["engine_price"][9]) / (s[15] - s[9])   # a 3% bump: indices 9 and 15
    assert hedge_delta(curved, spot, bump=0.03) == pytest.approx(wide, rel=1e-12)
    assert hedge_delta(curved, spot, bump=0.03) != hedge_delta(curved, spot)


# ---- synthetic worlds for the ledger tests: a recorded note, boards and chains fabricated by hand ----

WORLD_START = date(2024, 1, 2)       # a Tuesday; every world day is a consecutive weekday


def _weekdays(n, start=WORLD_START):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _series_info(day, last, forward, rate=0.014):
    from gpin.board import SeriesInfo

    tenor = (last - day).days / 365.0
    return SeriesInfo(last, tenor, forward, math.exp(-rate * tenor))


def _day(day, spot, price, delta, *, gamma=0.0, alive=True, cash=0.0, knocked_in=False, next_ko=None,
         im=None, mo=None, vol=0.2, end_reason=None, extra=None):
    """One world day. ``price`` / ``delta`` / ``gamma``: the recorded engine ladder of the long note (raw engine
    values). ``im``: ``{code: dict(close, open=None, vwap=None, volume=0, listed=True)}``. ``mo``:
    ``{series: dict(last=date, forward=float, quotes={(cp, strike): (settle, close, volume, oi)})}``."""
    return dict(day=day, spot=spot, price=price, delta=delta, gamma=gamma, alive=alive, cash=cash,
                knocked_in=knocked_in, next_ko=next_ko, im=im or {}, mo=mo or {}, vol=vol, end_reason=end_reason,
                extra=extra or {})


def _world(days, *, coupon=0.1, book="ki75", n_records=None):
    """``(states, boards, chains)`` for the world days: ``states`` in the shape of ``NoteStates.to_json``; only the first
    ``n_records`` days get a record (the pricing pass stops at the note's end, the chain goes on)."""
    from gpin.board import DayBoard, ImBar, OptionQuote
    from s2ts.chain import DayChain, ImQuote, MoPillar

    records, boards, chains = [], {}, []
    for i, w in enumerate(days):
        d = w["day"]
        rec = {"date": d.isoformat(), "spot": w["spot"], "alive": w["alive"], "knocked_in": w["knocked_in"],
               "engine_note_cash": w["cash"], "end_reason": w["end_reason"],
               "next_ko": None if w["next_ko"] is None else w["next_ko"].isoformat(),
               "days_to_ko": None if w["next_ko"] is None else (w["next_ko"] - d).days,
               "gap_ko": None, "ko_dir": None, "states": {}}
        if w["alive"]:
            rec["states"]["true"] = _ladder_state(w["price"], w["delta"], w["gamma"], w["spot"])
        rec.update(w["extra"])
        if n_records is None or i < n_records:
            records.append(rec)
        listed, bars = [], {}
        for code, b in w["im"].items():
            bars[code] = ImBar(b.get("open"), b["close"], b["close"], b.get("vwap"), b.get("volume", 0))
            if b.get("listed", True):
                listed.append(ImQuote(code, d + timedelta(days=60), 60 / 365.0, b["close"], b["close"]))
        mo, infos = {}, {}
        for series, s in w["mo"].items():
            infos[series] = _series_info(d, s["last"], s["forward"])
            mo[series] = {k: OptionQuote(q[0], q[1], None, None, q[2], q[3]) for k, q in s["quotes"].items()}
        boards[d] = DayBoard(d, w["spot"], bars, mo, infos)
        pillar = MoPillar("MO0000", d + timedelta(days=60), 60 / 365.0, w["spot"], 1.0, 1.0, w["spot"], w["vol"], ())
        chains.append(DayChain(d, w["spot"], tuple(listed), pillar, {}, 0.0))
    # the book's contract in index points, as ``NoteStates.to_json`` writes it: the 2Y product is struck at its booking
    # spot with the KI level at 75%; the seasoned book has its KI level 7% below the booking spot and strike = KI / 0.75
    spot0 = days[0]["spot"]
    ki_level = spot0 / 1.07 if book == "seas6" else 0.75 * spot0
    strike = ki_level / 0.75 if book == "seas6" else spot0
    states = {"book": book, "ki_pct": 0.75, "inception": days[0]["day"].isoformat(), "coupon": coupon,
              "days": records, "meta": {}, "spot0": spot0, "strike": strike, "ko_level": strike, "ki_level": ki_level}
    return states, boards, chains


def test_day_context_is_frozen_and_checks_its_dates():
    from gpin.techniques import DayContext, NoteInfo

    states, boards, _ = _world([_day(WORLD_START, 5000.0, 1.0, 100.0, im={"IM2401": dict(close=5000.0)})])
    rec = states["days"][0]
    note = NoteInfo("ki75", WORLD_START, 0.1, 0.75, 1.0, 5000.0, 5000.0, 5000.0, 3750.0)
    ctx = DayContext.build(WORLD_START, 0, 5000.0, 0.2, rec, boards[WORLD_START], note, {"IM2401": 5.0},
                           {("MO2401", "C", 5000.0): 1.0}, None)
    assert ctx.alive is True and ctx.knocked_in is False and ctx.gap_ko is None and ctx.ko_dir is None
    assert ctx.state("true")["engine_price"][12] == 1.0 and ctx.state("surv") is None
    assert ctx.held_im == {"IM2401": 5.0} and ctx.prev is None and ctx.board.day == WORLD_START
    with pytest.raises(Exception):
        ctx.spot = 1.0                                               # frozen
    with pytest.raises(TypeError):
        ctx.record["alive"] = False                                  # the record is read-only
    with pytest.raises(TypeError):
        ctx.held_im["IM2401"] = 0.0                                  # so are the positions
    with pytest.raises(TypeError):
        ctx.state("true")["engine_price"][0] = 0.0                   # and the ladders (tuples)
    # a record or a board of another day is refused; so is a previous context that is not earlier
    other = _weekdays(2)[1]
    with pytest.raises(ValidationError):
        DayContext.build(other, 1, 5000.0, 0.2, rec, boards[WORLD_START], note, {}, {}, ctx)
    with pytest.raises(ValidationError):
        DayContext.build(WORLD_START, 1, 5000.0, 0.2, rec, boards[WORLD_START], note, {}, {}, ctx)


def test_base_targets_the_true_bump_delta_every_day_and_noclip_only_differs_in_the_flag():
    from gpin.techniques import Base, BaseNoClip, DayContext, NoteInfo, Technique

    states, boards, _ = _world([_day(WORLD_START, 5000.0, 1.0, 750.0, gamma=0.0)])
    note = NoteInfo("ki75", WORLD_START, 0.1, 0.75, 1.0, 5000.0, 5000.0, 5000.0, 3750.0)
    ctx = DayContext.build(WORLD_START, 0, 5000.0, 0.2, states["days"][0], boards[WORLD_START], note, {}, {}, None)
    base, noclip = Base(), BaseNoClip()
    assert isinstance(base, Technique) and isinstance(noclip, Technique)
    assert (base.id, base.family, base.params, base.clip) == ("base", "baseline", {}, True)
    assert (noclip.id, noclip.family, noclip.params, noclip.clip) == ("base_noclip", "control", {}, False)
    for tech in (base, noclip):
        t = tech.target(ctx)
        assert t.note_delta == pytest.approx(750.0, rel=1e-12) and dict(t.legs) == {} and t.note_delta_if_filled is None


# ---- ledger core (C7), close clock, IM only ----

LEDGER_ROW_KEYS = ["date", "seller_pv", "position_delta", "position_gamma", "note_cash", "note_pnl", "im_vm", "mo_mtm",
                   "fill_pnl", "accrual", "cost", "hedged", "cash", "hedge_ratio", "im_lots", "legs", "phi", "clipped",
                   "unfilled", "flags"]
IM_COST = 1e-4
IM_MULT_ = 200.0
ACT_ = 365.0
RATE_ = 0.014
NOTIONAL_ = 10_000_000.0
CASH_TOL = 1e-6


def _run(states, tech, boards, chains, mode="close", *, mo_cost_pt=0.2, desk_notional=NOTIONAL_ * 50, until=None):
    from gpin.ledger import run_technique

    return run_technique(states, tech, boards, chains, mode=mode, mo_cost_pt=mo_cost_pt, desk_notional=desk_notional,
                         until=until)


def _by_date(run):
    return {r["date"]: r for r in run.rows}


def _check_identity(run, boards=None):
    """The C7 identity: sum of ``hedged`` = ``cash_end + seller_pv_end + option value_end`` (flat book: no option value)."""
    last = run.rows[-1]
    value_end = 0.0
    assert not last["legs"], "the identity helper handles a flat option book at the end"
    total = sum(r["hedged"] for r in run.rows)
    assert total == pytest.approx(last["cash"] + last["seller_pv"] + value_end, abs=CASH_TOL)


def _world_a():
    """Three days, the note survives; IM close 5000 / 5100 / 5000; the last day is the data end."""
    d = _weekdays(3)
    return _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, im={"IM2403": dict(close=5000.0)}),
        _day(d[1], 5100.0, 95000.0, 1200.0, im={"IM2403": dict(close=5100.0)}),
        _day(d[2], 5000.0, 105000.0, 1000.0, im={"IM2403": dict(close=5000.0)}),
    ])


def test_close_clock_baseline_on_a_surviving_note_matches_hand_computed_rows():
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    run = _run(states, Base(), boards, chains)
    assert len(run.rows) == 3
    for r in run.rows:
        assert list(r) == LEDGER_ROW_KEYS
        assert "holder" not in "".join(r)
    r0, r1, r2 = run.rows
    dt = 1.0 / ACT_
    # day 0: sold at 0, so note_pnl is the seller PV; 5 lots (1000 x 5000 / (200 x 5000)) opened at 5000
    cost0 = IM_COST * 5 * 5000.0 * IM_MULT_
    assert (r0["date"], r0["seller_pv"], r0["position_delta"], r0["note_cash"]) == (WORLD_START.isoformat(), -100000.0, -1000.0, 0.0)
    assert r0["position_gamma"] == 0.0
    assert r0["note_pnl"] == -100000.0 and r0["im_vm"] == 0.0 and r0["mo_mtm"] == 0.0 and r0["accrual"] == 0.0
    assert r0["cost"] == pytest.approx(cost0, abs=CASH_TOL) and r0["hedged"] == pytest.approx(-100000.0 - cost0, abs=CASH_TOL)
    assert r0["cash"] == pytest.approx(-cost0, abs=CASH_TOL)
    assert r0["im_lots"] == pytest.approx(5.0) and r0["hedge_ratio"] == pytest.approx(5 * 200 * 5000.0 / NOTIONAL_)
    assert (r0["phi"], r0["clipped"], r0["unfilled"], r0["legs"], r0["flags"]) == (1.0, False, {}, {}, [])
    # day 1: 5 lots earn 5 x 100 x 200; the target becomes 1200 x 5100 / (200 x 5100) = 6 lots, one more at 5100
    accrual1 = r0["cash"] * RATE_ * dt
    cost1 = IM_COST * 1 * 5100.0 * IM_MULT_
    assert r1["im_vm"] == pytest.approx(5 * 100.0 * 200.0) and r1["note_pnl"] == pytest.approx(-95000.0 + 100000.0)
    assert r1["accrual"] == pytest.approx(accrual1) and r1["cost"] == pytest.approx(cost1)
    assert r1["hedged"] == pytest.approx(5000.0 + 100000.0 + accrual1 - cost1)
    assert r1["cash"] == pytest.approx(r0["cash"] + accrual1 + 100000.0 - cost1)
    assert r1["im_lots"] == pytest.approx(6.0) and r1["fill_pnl"] == 0.0
    # day 2: the data end, the note is alive: 6 lots earn 6 x (-100) x 200, then everything is marked flat at the
    # close. The flatten is a mark, not a trade: it costs nothing
    accrual2 = r1["cash"] * RATE_ * dt
    assert r2["im_vm"] == pytest.approx(-120000.0) and r2["note_pnl"] == pytest.approx(-105000.0 + 95000.0)
    assert r2["cost"] == 0.0 and r2["im_lots"] == 0.0 and r2["hedge_ratio"] == 0.0
    assert r2["hedged"] == pytest.approx(-10000.0 - 120000.0 + accrual2)
    assert "data_end" in r2["flags"]
    _check_identity(run)


def test_a_fired_ko_ends_the_rows_on_the_ko_day_and_pays_the_seller_side_cash():
    from gpin.techniques import Base

    d = _weekdays(3)
    states, boards, chains = _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, im={"IM2403": dict(close=5000.0)}),
        _day(d[1], 5100.0, 95000.0, 1200.0, im={"IM2403": dict(close=5100.0)}),
        _day(d[2], 5000.0, 0.0, 0.0, alive=False, cash=700000.0, end_reason="ko", im={"IM2403": dict(close=5000.0)}),
    ])
    run = _run(states, Base(), boards, chains)
    assert [r["date"] for r in run.rows] == [x.isoformat() for x in d]          # flat on the KO day: no row after
    r2 = run.rows[2]
    assert r2["note_cash"] == -700000.0                                           # the seller pays the KO coupon
    assert r2["seller_pv"] == 0.0 and r2["position_delta"] == 0.0 and r2["im_lots"] == 0.0
    assert r2["note_pnl"] == pytest.approx((0.0 - (-95000.0)) - 700000.0)
    assert "data_end" not in r2["flags"]
    assert r2["cash"] == pytest.approx(run.rows[1]["cash"] + r2["accrual"] - 700000.0 + r2["im_vm"] - r2["cost"])
    _check_identity(run)


def test_an_im_roll_closes_the_old_contract_and_opens_the_new_one_marking_from_the_board():
    from gpin.techniques import Base

    d = _weekdays(5)

    def im(**codes):
        return {c: dict(close=v[0], listed=v[1]) for c, v in codes.items()}

    states, boards, chains = _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, im=im(IM2403=(5000.0, True))),
        _day(d[1], 5000.0, 100000.0, 1000.0, im=im(IM2403=(5100.0, True))),
        _day(d[2], 5000.0, 100000.0, 1000.0, im=im(IM2403=(5050.0, True))),
        # day 3: the old contract is no longer in the chain (under 7 days), its close is only on the board
        _day(d[3], 5000.0, 100000.0, 1000.0, im=im(IM2403=(5030.0, False), IM2404=(5100.0, True))),
        _day(d[4], 5000.0, 100000.0, 1000.0, im=im(IM2404=(5120.0, True))),
    ])
    run = _run(states, Base(), boards, chains)
    lots = [1000.0 * 5000.0 / (200.0 * f) for f in (5000.0, 5100.0, 5050.0, 5100.0)]
    rows = run.rows
    assert [r["im_lots"] for r in rows[:4]] == pytest.approx(lots)
    assert rows[1]["im_vm"] == pytest.approx(lots[0] * 100.0 * 200.0)
    assert rows[2]["im_vm"] == pytest.approx(lots[1] * -50.0 * 200.0)
    assert rows[3]["im_vm"] == pytest.approx(lots[2] * -20.0 * 200.0)             # the board's 5030 marks the old contract
    cost3 = IM_COST * lots[2] * 5030.0 * 200.0 + IM_COST * lots[3] * 5100.0 * 200.0   # old closed at its close, new opened
    assert rows[3]["cost"] == pytest.approx(cost3) and "roll" in rows[3]["flags"]
    assert rows[4]["im_vm"] == pytest.approx(lots[3] * 20.0 * 200.0)              # the new contract is held from day 3
    assert rows[4]["im_lots"] == 0.0 and "data_end" in rows[4]["flags"]
    _check_identity(run)


def test_the_run_carries_its_identity_and_serialises():
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    run = _run(states, Base(), boards, chains, mo_cost_pt=1.0, desk_notional=1e8)
    assert (run.book, run.inception, run.technique, run.params, run.mode) == ("ki75", WORLD_START, "base", {}, "close")
    assert (run.mo_cost_pt, run.desk_notional, run.coupon) == (1.0, 1e8, 0.1)
    payload = json.loads(json.dumps(run.to_json(), allow_nan=False))
    assert payload["technique"] == "base" and payload["mode"] == "close" and payload["desk_notional"] == 1e8
    assert payload["inception"] == WORLD_START.isoformat() and len(payload["rows"]) == 3
    assert "holder" not in json.dumps(payload)


# ---- the three clocks (D7) ----

def _world_twap():
    """Four days; IM vwap / volume given except on day 2, which did not trade (vwap None, volume 0)."""
    d = _weekdays(4)

    def im(close, vwap, volume, open_=None):
        return {"IM2403": dict(close=close, vwap=vwap, volume=volume, open=open_)}

    return _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, im=im(5000.0, 4990.0, 1000)),
        _day(d[1], 5100.0, 95000.0, 1200.0, im=im(5100.0, 5080.0, 1000)),
        _day(d[2], 5000.0, 105000.0, 1000.0, im=im(5000.0, None, 0)),
        _day(d[3], 5000.0, 105000.0, 1000.0, im=im(5000.0, 5000.0, 1000)),
    ])


def test_twap_fills_im_at_the_vwap_and_at_the_close_when_nothing_traded():
    from gpin.techniques import Base

    states, boards, chains = _world_twap()
    run = _run(states, Base(), boards, chains, "twap")
    r0, r1, r2, r3 = run.rows
    # day 0: 5 lots bought at the VWAP 4990, marked at the close 5000: fill_pnl = 5 x (5000 - 4990) x 200
    assert r0["fill_pnl"] == pytest.approx(5 * 10.0 * 200.0) and r0["im_vm"] == pytest.approx(10000.0)
    assert r0["cost"] == pytest.approx(IM_COST * 5 * 4990.0 * 200.0)
    # day 1: 5 lots held earn 5 x 100 x 200; one lot bought at 5080: fill_pnl = 1 x (5100 - 5080) x 200
    assert r1["fill_pnl"] == pytest.approx(1 * 20.0 * 200.0)
    assert r1["im_vm"] == pytest.approx(5 * 100.0 * 200.0 + 4000.0)
    assert r1["cost"] == pytest.approx(IM_COST * 1 * 5080.0 * 200.0)
    # day 2: zero volume, so the VWAP is not used: the lot sold fills at the close
    assert r2["fill_pnl"] == 0.0 and r2["cost"] == pytest.approx(IM_COST * 1 * 5000.0 * 200.0)
    assert r2["im_vm"] == pytest.approx(6 * -100.0 * 200.0)
    # day 3: the data end, flat at the day's mark (the close), not at the VWAP
    assert r3["fill_pnl"] == 0.0 and r3["im_lots"] == 0.0 and "data_end" in r3["flags"]
    _check_identity(run)


def test_clocks_give_the_same_end_of_day_positions_on_non_fixing_days():
    from gpin.techniques import Base

    states, boards, chains = _world_twap()
    lots = {m: [r["im_lots"] for r in _run(states, Base(), boards, chains, m).rows] for m in ("close", "twap", "next_open")}
    assert lots["close"] == lots["twap"] == lots["next_open"] == pytest.approx([5.0, 6.0, 5.0, 0.0])
    # the fills differ: only the twap row has a fill part
    assert [r["fill_pnl"] for r in _run(states, Base(), boards, chains, "close").rows] == [0.0] * 4
    assert _run(states, Base(), boards, chains, "twap").rows[0]["fill_pnl"] != 0.0


def _ko_world(n_extra=1, last_open=4930.0):
    """d0, d1 alive (the KO observation is d2), the KO fires on d2 (dead record, cash 700000); ``n_extra`` chain days
    follow it (no record). IM close 5000 / 5100 / 4900 / 4950, the day after the KO opens at ``last_open``."""
    d = _weekdays(3 + n_extra)
    days = [
        _day(d[0], 5000.0, 100000.0, 1000.0, next_ko=d[2], im={"IM2403": dict(close=5000.0)}),
        _day(d[1], 5100.0, 95000.0, 1200.0, next_ko=d[2], im={"IM2403": dict(close=5100.0)}),
        _day(d[2], 4900.0, 0.0, 0.0, alive=False, cash=700000.0, end_reason="ko", im={"IM2403": dict(close=4900.0)}),
    ]
    for k in range(n_extra):
        days.append(_day(d[3 + k], 4950.0, 0.0, 0.0, alive=False,
                         im={"IM2403": dict(close=4950.0, open=last_open if k == 0 else 4950.0)}))
    return _world(days, n_records=3)


def test_next_open_carries_the_hedge_through_the_ko_and_flattens_at_the_next_open():
    from gpin.techniques import Base

    states, boards, chains = _ko_world()
    run = _run(states, Base(), boards, chains, "next_open")
    assert len(run.rows) == 4                                                   # a T+1 row after the fired KO
    r1, r2, r3 = run.rows[1:]
    lots1 = 1200.0 * 5100.0 / (200.0 * 5100.0)
    assert r1["im_lots"] == pytest.approx(lots1)
    # the KO day: no IM trade at the fixing close, the hedge is carried and earns the close-to-close move
    assert r2["im_lots"] == pytest.approx(lots1) and r2["cost"] == 0.0 and "fixing_deferred" in r2["flags"]
    assert r2["im_vm"] == pytest.approx(lots1 * (4900.0 - 5100.0) * 200.0)
    assert r2["hedge_ratio"] == pytest.approx(lots1 * 200.0 * 4900.0 / NOTIONAL_)
    assert r2["note_cash"] == -700000.0 and r2["seller_pv"] == 0.0
    # T+1: the held lots earn (open - close) x 200, the flat target fills at the open
    assert r3["im_vm"] == pytest.approx(lots1 * (4930.0 - 4900.0) * 200.0) and r3["im_lots"] == 0.0
    assert r3["cost"] == pytest.approx(IM_COST * lots1 * 4930.0 * 200.0)
    assert r3["note_pnl"] == 0.0 and r3["note_cash"] == 0.0 and r3["seller_pv"] == 0.0
    assert "deferred_fill" in r3["flags"] and "last_day_fixing" not in r3["flags"]
    _check_identity(run)
    # under the close clock the same world is flat on the KO day, three rows
    assert len(_run(states, Base(), boards, chains, "close").rows) == 3


def test_next_open_a_knock_in_day_is_a_fixing_day_and_the_pending_trade_fills_at_the_open():
    from gpin.techniques import Base

    d = _weekdays(5)

    def im(close, open_=None):
        return {"IM2403": dict(close=close, open=open_)}

    states, boards, chains = _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, im=im(5000.0)),
        _day(d[1], 5000.0, 100000.0, 1000.0, im=im(5000.0)),
        _day(d[2], 5000.0, 90000.0, 1500.0, knocked_in=True, im=im(5000.0)),         # the knock-in day
        _day(d[3], 5000.0, 90000.0, 1200.0, knocked_in=True, im=im(5100.0, open_=5040.0)),
        _day(d[4], 5000.0, 90000.0, 1200.0, knocked_in=True, im=im(5100.0)),
    ])
    run = _run(states, Base(), boards, chains, "next_open")
    r2, r3 = run.rows[2], run.rows[3]
    assert r2["im_lots"] == pytest.approx(5.0) and "fixing_deferred" in r2["flags"] and r2["cost"] == 0.0
    pend = 1500.0 * 5000.0 / (200.0 * 5000.0)                                       # 7.5 lots, sized at the day-2 close
    own = 1200.0 * 5000.0 / (200.0 * 5100.0)                                        # day 3's own target at its close
    # 5 lots held earn 5 x 100 x 200; 2.5 more bought at the open 5040 earn 2.5 x (5100 - 5040) x 200; the own trade
    # at the close has no fill part
    assert r3["fill_pnl"] == pytest.approx(2.5 * 60.0 * 200.0)
    assert r3["im_vm"] == pytest.approx(5 * 100.0 * 200.0 + 2.5 * 60.0 * 200.0)
    assert r3["cost"] == pytest.approx(IM_COST * 2.5 * 5040.0 * 200.0 + IM_COST * abs(own - pend) * 5100.0 * 200.0)
    assert r3["im_lots"] == pytest.approx(own) and "deferred_fill" in r3["flags"]
    _check_identity(run)


def test_a_fixing_on_the_last_data_day_fills_at_that_close_and_is_flagged():
    from gpin.techniques import Base

    # the KO fires on the last chain day: no next open exists
    states, boards, chains = _ko_world(n_extra=0)
    run = _run(states, Base(), boards, chains, "next_open")
    r2 = run.rows[-1]
    assert len(run.rows) == 3 and r2["im_lots"] == 0.0 and "last_day_fixing" in r2["flags"]
    assert r2["im_vm"] == pytest.approx(1200.0 * 5100.0 / (200.0 * 5100.0) * (4900.0 - 5100.0) * 200.0)
    assert r2["cost"] == pytest.approx(IM_COST * 6.0 * 4900.0 * 200.0)             # 6 lots sold at the fixing close
    _check_identity(run)
    # an alive note whose observation is the last data day is flattened at that close, flagged both ways
    d = _weekdays(3)
    states, boards, chains = _world([
        _day(d[0], 5000.0, 100000.0, 1000.0, next_ko=d[2], im={"IM2403": dict(close=5000.0)}),
        _day(d[1], 5000.0, 100000.0, 1000.0, next_ko=d[2], im={"IM2403": dict(close=5000.0)}),
        _day(d[2], 5000.0, 100000.0, 1000.0, next_ko=d[2] + timedelta(days=90), im={"IM2403": dict(close=5000.0)}),
    ])
    last = _run(states, Base(), boards, chains, "next_open").rows[-1]
    assert "last_day_fixing" in last["flags"] and "data_end" in last["flags"] and last["im_lots"] == 0.0


# ---- option legs: fills, tradability, limits, expiry (D8) ----

SERIES = "MO2403"
KEY_C5000 = (SERIES, "C", 5000.0)
STR_C5000 = "MO2403:C:5000"
LEG_FORWARD = 5000.0


class _Scripted:
    """A test technique: ``legs_fn(ctx) -> dict`` of desired legs; ``note_delta`` constant (and the filled variant)."""

    clip = True

    def __init__(self, legs_fn, note_delta=1000.0, if_filled=None):
        self.id, self.family, self.params = "scripted", "test", {}
        self.legs_fn, self.note_delta, self.if_filled = legs_fn, note_delta, if_filled
        self.seen = []

    def target(self, ctx):
        from gpin.techniques import Target

        self.seen.append(ctx)
        return Target(self.note_delta, self.legs_fn(ctx), self.if_filled)


def _px(day, last, cp, strike, vol, forward=LEG_FORWARD):
    """The settlement of an option at an implied vol ``vol`` on ``day`` (a Black-76 price, library function)."""
    from gpin import black

    return black.value(vol, cp, strike, _series_info(day, last, forward))


def _leg_world(n_days, last, quotes_by_day, *, delta=1000.0, spots=None, records=None, alive_until=None, extra_days=0):
    """``n_days`` days of a constant-delta note (spot 5000, IM 5000); ``quotes_by_day[i]`` = ``{(cp, strike):
    (settle, close, volume, oi)}`` of series ``SERIES`` on day i (``None``: the series is not on the board)."""
    d = _weekdays(n_days + extra_days)
    days = []
    for i, day in enumerate(d):
        q = quotes_by_day[i] if i < len(quotes_by_day) else None
        mo = {} if q is None else {SERIES: dict(last=last if last is not None else d[-1] + timedelta(days=60),
                                                 forward=LEG_FORWARD, quotes=q)}
        dead = alive_until is not None and i >= alive_until
        days.append(_day(day, 5000.0 if spots is None else spots[i], 100000.0, delta, alive=not dead,
                         cash=0.0 if not dead else 0.0, end_reason="ko" if dead else None,
                         im={"IM2403": dict(close=5000.0)}, mo=mo))
    return _world(days, n_records=records if records is not None else n_days), d


def _leg_delta(day, last, cp, strike, vol, lots, spot=5000.0, forward=LEG_FORWARD):
    """CNY per index point of ``lots`` options: Black-76 index delta x 100."""
    from gpin import black

    return lots * 100.0 * black.index_delta(vol, cp, strike, _series_info(day, last, forward), spot)


def _hold_when(days_with_leg, lots):
    return lambda ctx: {KEY_C5000: lots} if ctx.day in days_with_leg else {}


def test_a_leg_opened_and_closed_marks_at_the_settlement_and_shifts_the_im_target():
    from gpin.techniques import Base

    n = 5
    d = _weekdays(n)
    last = d[0] + timedelta(days=60)
    vols = [0.20, 0.22, 0.19, 0.21, 0.20]
    quotes = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, vols[i]), None, 5000, 10000)} for i in range(n)]
    (states, boards, chains), d = _leg_world(n, last, quotes)
    tech = _Scripted(_hold_when({d[1], d[2]}, 2.0))
    run = _run(states, tech, boards, chains, "close", desk_notional=NOTIONAL_)         # scale 1: no cut binds
    rows = run.rows
    s = [q[("C", 5000.0)][0] for q in quotes]
    # day 1: two calls opened at the settlement, cost 0.2 points x 100 x 2 lots, premium 2 x s1 x 100
    assert rows[1]["legs"] == {STR_C5000: 2.0} and rows[1]["phi"] == 1.0 and rows[1]["unfilled"] == {}
    assert rows[1]["mo_mtm"] == 0.0 and rows[1]["fill_pnl"] == 0.0
    assert rows[1]["cost"] == pytest.approx(0.2 * 100.0 * 2.0 + _im_cost_of_change(rows, 1))
    leg1 = _leg_delta(d[1], last, "C", 5000.0, 0.22, 2.0)
    assert rows[1]["im_lots"] == pytest.approx((1000.0 - leg1) * 5000.0 / (200.0 * 5000.0))       # the IM target nets the leg
    # day 2: held 2 earn 2 x (s2 - s1) x 100
    assert rows[2]["mo_mtm"] == pytest.approx(2.0 * (s[2] - s[1]) * 100.0)
    leg2 = _leg_delta(d[2], last, "C", 5000.0, 0.19, 2.0)
    assert rows[2]["im_lots"] == pytest.approx((1000.0 - leg2) * 5000.0 / (200.0 * 5000.0))
    # day 3: closed at the settlement: it earns the day's move, the premium comes back
    assert rows[3]["legs"] == {} and rows[3]["mo_mtm"] == pytest.approx(2.0 * (s[3] - s[2]) * 100.0)
    assert rows[3]["cost"] == pytest.approx(0.2 * 100.0 * 2.0 + _im_cost_of_change(rows, 3))
    assert rows[3]["im_lots"] == pytest.approx(1000.0 * 5000.0 / (200.0 * 5000.0))
    # cash: premium paid on day 1, received on day 3
    c1 = rows[0]["cash"] + rows[1]["accrual"] + rows[1]["im_vm"] - 2.0 * s[1] * 100.0 - rows[1]["cost"]
    assert rows[1]["cash"] == pytest.approx(c1)
    c3 = rows[2]["cash"] + rows[3]["accrual"] + rows[3]["im_vm"] + 2.0 * s[3] * 100.0 - rows[3]["cost"]
    assert rows[3]["cash"] == pytest.approx(c3)
    assert sum(r["mo_mtm"] for r in rows) == pytest.approx(2.0 * (s[3] - s[1]) * 100.0)
    _check_identity(run)


def _im_cost_of_change(rows, i):
    """IM cost of the day's lot change at the close 5000 (every test world's IM close is 5000)."""
    return IM_COST * abs(rows[i]["im_lots"] - rows[i - 1]["im_lots"]) * 5000.0 * 200.0


@pytest.mark.parametrize("quote, why", [
    ((None, None, 99, 10000), "volume"),
    ((None, None, 5000, 499), "open_interest"),
    ((None, None, 5000, None), "open_interest"),
    (("itm", None, 5000, 10000), "implied_vol"),
])
def test_a_leg_is_not_opened_without_volume_open_interest_or_an_implied_vol(quote, why):
    from gpin.techniques import Base

    d = _weekdays(3)
    last = d[0] + timedelta(days=60)
    ok = _px(d[1], last, "C", 5000.0, 0.2)
    settle = 300.0 if quote[0] == "itm" else ok                    # a settlement below the 4500 call's intrinsic: no vol
    strike = 4500.0 if quote[0] == "itm" else 5000.0
    q = [{("C", strike): (_px(d[i], last, "C", strike, 0.2), None, 5000, 10000)} for i in range(3)]
    q[1] = {("C", strike): (settle, None, quote[2], quote[3])}
    (states, boards, chains), d = _leg_world(3, last, q)
    key = (SERIES, "C", strike)
    run = _run(states, _Scripted(lambda ctx: {key: 2.0} if ctx.day_index >= 1 else {}), boards, chains, desk_notional=NOTIONAL_)
    r1 = run.rows[1]
    assert r1["legs"] == {} and r1["unfilled"] == {f"{SERIES}:C:{strike:g}": 2.0} and r1["phi"] == 0.0
    assert "refused_open" in r1["flags"] and r1["cost"] == pytest.approx(_im_cost_of_change(run.rows, 1))
    assert r1["im_lots"] == pytest.approx(5.0)                                  # no overlay: the plain note target


def test_the_tradability_thresholds_are_inclusive():
    d = _weekdays(3)
    last = d[0] + timedelta(days=60)
    q = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, 100, 500)} for i in range(3)]
    (states, boards, chains), d = _leg_world(3, last, q)
    run = _run(states, _Scripted(lambda ctx: {KEY_C5000: 0.05} if ctx.day_index >= 1 else {}), boards, chains, desk_notional=NOTIONAL_)   # a twentieth of a lot, scale 1
    assert run.rows[1]["legs"] == {STR_C5000: 0.05} and run.rows[1]["unfilled"] == {}


def test_a_trade_above_the_volume_share_is_cut_and_the_half_filled_overlay_blends_the_im_target():
    d = _weekdays(4)
    last = d[0] + timedelta(days=60)
    q = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, 1000, 10000)} for i in range(4)]
    (states, boards, chains), d = _leg_world(4, last, q)
    # desk scale 100: 2 note-lots are 200 desk lots, a tenth of the day's 1000 lots is 100: half is filled
    tech = _Scripted(lambda ctx: {KEY_C5000: 2.0} if ctx.day_index >= 1 else {}, note_delta=1000.0, if_filled=200.0)
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_ * 100)
    r1 = run.rows[1]
    assert r1["legs"] == {STR_C5000: 1.0} and r1["unfilled"] == {STR_C5000: 1.0}
    assert r1["phi"] == pytest.approx(0.5) and "volume_cut" in r1["flags"]
    leg = _leg_delta(d[1], last, "C", 5000.0, 0.2, 1.0)
    assert r1["im_lots"] == pytest.approx((0.5 * 200.0 + 0.5 * 1000.0 - leg) * 5000.0 / (200.0 * 5000.0))
    assert r1["cost"] == pytest.approx(0.2 * 100.0 * 1.0 + _im_cost_of_change(run.rows, 1))     # only the filled lot pays
    # day 2: the second half is bought too (the day's cap is per day), the overlay is whole and phi is 1
    assert run.rows[2]["legs"] == {STR_C5000: 2.0} and run.rows[2]["phi"] == 1.0
    assert run.rows[2]["unfilled"] == {}


def test_a_series_position_is_cut_to_1200_lots_per_side_at_desk_scale():
    d = _weekdays(3)
    last = d[0] + timedelta(days=60)
    strikes = [("C", 5000.0), ("P", 5000.0), ("C", 5100.0)]
    q = [{k: (_px(d[i], last, k[0], k[1], 0.2), None, 20000, 10000) for k in strikes} for i in range(3)]
    (states, boards, chains), d = _leg_world(3, last, q)
    # scale 100. One leg of 15 note-lots (1500 desk): 12 fill, 3 do not
    run = _run(states, _Scripted(lambda ctx: {KEY_C5000: 15.0} if ctx.day_index >= 1 else {}), boards, chains, desk_notional=NOTIONAL_ * 100)
    r1 = run.rows[1]
    assert r1["legs"] == {STR_C5000: 12.0} and r1["unfilled"] == {STR_C5000: 3.0} and "series_limit" in r1["flags"]
    # a long call and a short put are the same side of the series (both gain when the index rises): 8 + 8 -> 8 + 4;
    # a short call is the other side and is not counted against it
    desired = {KEY_C5000: 8.0, (SERIES, "P", 5000.0): -8.0, (SERIES, "C", 5100.0): -10.0}
    run = _run(states, _Scripted(lambda ctx: dict(desired) if ctx.day_index >= 1 else {}), boards, chains, desk_notional=NOTIONAL_ * 100)
    r1 = run.rows[1]
    assert r1["legs"] == {STR_C5000: 8.0, "MO2403:P:5000": -4.0, "MO2403:C:5100": -10.0}
    assert r1["unfilled"] == {"MO2403:P:5000": -4.0}


def _cut_close_world():
    d = _weekdays(4)
    last = d[0] + timedelta(days=60)
    vols = [20000, 20000, 500, 20000]                              # day 2: only 500 lots trade, a tenth is 50 desk lots
    q = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, vols[i], 10000)} for i in range(4)]
    return _leg_world(3, last, q + [q[3]], alive_until=2, records=3, extra_days=1)


def test_a_cut_close_leaves_the_leg_on_the_book_keeps_its_delta_in_the_im_target_and_rows_continue():
    (states, boards, chains), d = _cut_close_world()
    last = d[0] + timedelta(days=60)
    tech = _Scripted(lambda ctx: {KEY_C5000: 1.0} if ctx.day_index >= 1 else {})
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_ * 100)
    rows = run.rows
    assert len(rows) == 4 and rows[1]["legs"] == {STR_C5000: 1.0}
    # day 2: the note is dead (KO), the target is flat; 50 of the 100 desk lots close, half a lot stays
    assert rows[2]["legs"] == {STR_C5000: 0.5} and rows[2]["unfilled"] == {STR_C5000: -0.5}
    leg = _leg_delta(d[2], last, "C", 5000.0, 0.2, 0.5)
    assert rows[2]["im_lots"] == pytest.approx((0.0 - leg) * 5000.0 / (200.0 * 5000.0))       # the leg's delta stays hedged
    assert rows[2]["im_lots"] != 0.0
    # day 3 (no record: the note is gone): the rest closes, the IM position goes
    assert rows[3]["legs"] == {} and rows[3]["im_lots"] == 0.0 and rows[3]["seller_pv"] == 0.0
    assert rows[3]["mo_mtm"] == pytest.approx(0.5 * (_px(d[3], last, "C", 5000.0, 0.2) - _px(d[2], last, "C", 5000.0, 0.2)) * 100.0)
    _check_identity(run)


@pytest.mark.parametrize("traded", [True, False])
def test_on_its_last_trading_day_a_leg_exits_at_the_close_if_it_traded_else_at_the_settlement(traded):
    d = _weekdays(4)
    last = d[2]
    s1 = _px(d[1], last, "C", 5000.0, 0.2)
    settle2, close2 = 130.0, 128.0
    q = [None,
         {("C", 5000.0): (s1, None, 20000, 10000)},
         {("C", 5000.0): (settle2, close2 if traded else None, 800 if traded else 0, 10000)},
         None]
    (states, boards, chains), d = _leg_world(4, last, q)
    tech = _Scripted(lambda ctx: {KEY_C5000: 2.0} if ctx.day == d[1] else {})
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_)
    r2 = run.rows[2]
    exit_price = close2 if traded else settle2
    assert r2["legs"] == {} and ("expiry_exit" if traded else "expiry_settled") in r2["flags"]
    # the held 2 earn the day's settlement move; at the close the exit also earns (settle - close) per option sold
    fill = -2.0 * (settle2 - exit_price) * 100.0
    assert r2["fill_pnl"] == pytest.approx(fill)
    assert r2["mo_mtm"] == pytest.approx(2.0 * (settle2 - s1) * 100.0 + fill)
    cost = (0.2 * 100.0 * 2.0 if traded else 0.0) + _im_cost_of_change(run.rows, 2)
    assert r2["cost"] == pytest.approx(cost)
    c2 = run.rows[1]["cash"] + r2["accrual"] + r2["im_vm"] + 2.0 * exit_price * 100.0 - r2["cost"]
    assert r2["cash"] == pytest.approx(c2)
    _check_identity(run)


def test_a_missing_mark_for_a_held_leg_fails_closed():
    d = _weekdays(4)
    last = d[0] + timedelta(days=60)
    q = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, 20000, 10000)} for i in range(2)] + [None, None]
    (states, boards, chains), d = _leg_world(4, last, q)
    with pytest.raises(StudyDataError):
        _run(states, _Scripted(lambda ctx: {KEY_C5000: 1.0} if ctx.day_index >= 1 else {}), boards, chains, desk_notional=NOTIONAL_)


# ---- the D9 clip, next_open legs, until, fail-closed data, no look-ahead through the context ----

def test_the_d9_clip_binds_at_3_and_is_counted_and_base_noclip_is_not_clipped():
    from gpin.techniques import Base, BaseNoClip

    d = _weekdays(4)
    # delta 6800 CNY per point at spot 5000 is a hedge ratio h = 6800 x 5000 / 10 mio = 3.4
    states, boards, chains = _world([_day(x, 5000.0, 100000.0, 6800.0, im={"IM2403": dict(close=5000.0)}) for x in d])
    run = _run(states, Base(), boards, chains)
    assert [r["clipped"] for r in run.rows] == [True, True, True, False]            # the data-end flatten is not a hedge
    assert run.clip_count == 3
    assert run.rows[0]["im_lots"] == pytest.approx(6000.0 * 5000.0 / (200.0 * 5000.0))      # clipped to h = 3.0: 30 lots
    assert run.rows[0]["hedge_ratio"] == pytest.approx(3.0)
    free = _run(states, BaseNoClip(), boards, chains)
    assert [r["clipped"] for r in free.rows] == [False] * 4 and free.clip_count == 0
    assert free.rows[0]["im_lots"] == pytest.approx(34.0) and free.rows[0]["hedge_ratio"] == pytest.approx(3.4)
    assert free.technique == "base_noclip" and free.family == "control"
    # the lower bound is -0.25
    states, boards, chains = _world([_day(x, 5000.0, 100000.0, -1000.0, im={"IM2403": dict(close=5000.0)}) for x in d])
    short = _run(states, Base(), boards, chains)
    assert short.rows[0]["clipped"] and short.rows[0]["hedge_ratio"] == pytest.approx(-0.25)
    assert short.rows[0]["im_lots"] == pytest.approx(-2.5)


def test_next_open_a_fixing_day_defers_the_leg_adjustment_to_the_next_settlement():
    n = 5
    d = _weekdays(n)
    last = d[0] + timedelta(days=60)
    vols = [0.2, 0.2, 0.2, 0.21, 0.2]
    q = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, vols[i]), None, 20000, 10000)} for i in range(n)]
    days = []
    for i, day in enumerate(d):
        days.append(_day(day, 5000.0, 100000.0, 1000.0, knocked_in=i >= 2, im={"IM2403": dict(close=5000.0, open=5000.0)},
                         mo={SERIES: dict(last=last, forward=LEG_FORWARD, quotes=q[i])}))
    states, boards, chains = _world(days)
    tech = _Scripted(lambda ctx: {KEY_C5000: 2.0} if ctx.day_index >= 2 else {})
    run = _run(states, tech, boards, chains, "next_open", desk_notional=NOTIONAL_)
    r2, r3 = run.rows[2], run.rows[3]
    assert r2["legs"] == {} and "fixing_deferred" in r2["flags"] and r2["cost"] == 0.0
    s3 = q[3][("C", 5000.0)][0]
    assert r3["legs"] == {STR_C5000: 2.0} and "deferred_fill" in r3["flags"]
    assert r3["mo_mtm"] == 0.0                                                      # bought today at the settlement
    assert r3["cost"] >= 0.2 * 100.0 * 2.0
    assert r3["cash"] == pytest.approx(r2["cash"] + r3["accrual"] + r3["im_vm"] - 2.0 * s3 * 100.0 - r3["cost"])
    leg3 = _leg_delta(d[3], last, "C", 5000.0, 0.21, 2.0)
    assert r3["im_lots"] == pytest.approx((1000.0 - leg3) * 5000.0 / (200.0 * 5000.0))
    _check_identity(run)


def test_until_stops_the_run_and_marks_everything_flat_with_a_flag():
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    d = [c.day for c in chains]
    run = _run(states, Base(), boards, chains, until=d[1])
    assert [r["date"] for r in run.rows] == [d[0].isoformat(), d[1].isoformat()]
    assert "until" in run.rows[1]["flags"] and run.rows[1]["im_lots"] == 0.0
    full = _run(states, Base(), boards, chains)
    assert run.rows[0] == full.rows[0]                                              # nothing before the cut changes
    assert run.rows[1]["note_pnl"] == full.rows[1]["note_pnl"]
    assert run.rows[1]["im_vm"] == full.rows[1]["im_vm"]
    assert run.to_json()["until"] == d[1].isoformat()
    _check_identity(run)


def test_the_ledger_fails_closed_on_missing_data():
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    unpriced = json.loads(json.dumps(states))
    unpriced["days"][1]["states"] = {}
    with pytest.raises(StudyDataError):
        _run(unpriced, Base(), boards, chains)
    skewed = json.loads(json.dumps(states))
    skewed["days"][1]["date"] = "2024-02-01"
    with pytest.raises(StudyDataError):
        _run(skewed, Base(), boards, chains)
    with pytest.raises(StudyDataError):
        _run(states, Base(), {k: v for k, v in boards.items() if k != chains[1].day}, chains)
    # a held IM contract that neither the chain nor the board can mark
    d = _weekdays(3)
    states, boards, chains = _world([
        _day(d[0], 5000.0, 1.0, 1000.0, im={"IM2403": dict(close=5000.0)}),
        _day(d[1], 5000.0, 1.0, 1000.0, im={"IM2404": dict(close=5000.0)}),
        _day(d[2], 5000.0, 1.0, 1000.0, im={"IM2404": dict(close=5000.0)}),
    ])
    with pytest.raises(StudyDataError):
        _run(states, Base(), boards, chains)
    with pytest.raises(ValidationError):
        _run(states, Base(), boards, chains, "weekly")


def test_the_context_a_technique_sees_is_dated_at_or_before_the_decision_day():
    from gpin.techniques import Base

    class Spy(_Scripted):
        pass

    states, boards, chains = _world_a()
    spy = Spy(lambda ctx: {})
    _run(states, spy, boards, chains)
    assert [c.day for c in spy.seen] == [chains[0].day, chains[1].day]            # the final day is flattened, not decided
    for c in spy.seen:
        assert c.board.day == c.day and c.record["date"] == c.day.isoformat()
        assert c.prev is None or c.prev.day < c.day
    assert spy.seen[1].prev is spy.seen[0] and spy.seen[0].held_im == {} and spy.seen[1].held_im == pytest.approx({"IM2403": 5.0})
    # changing any state, board or chain dated after day 1 leaves the rows up to day 1 untouched
    base = _run(states, Base(), boards, chains)
    changed, boards2, chains2 = _world([
        _day(chains[0].day, 5000.0, 100000.0, 1000.0, im={"IM2403": dict(close=5000.0)}),
        _day(chains[1].day, 5100.0, 95000.0, 1200.0, im={"IM2403": dict(close=5100.0)}),
        _day(chains[2].day, 4000.0, 777.0, 4321.0, im={"IM2403": dict(close=4100.0)}),
    ])
    alt = _run(changed, Base(), boards2, chains2)
    assert alt.rows[:2] == base.rows[:2] and alt.rows[2] != base.rows[2]


# ---- seller view on the real probe (AC15) and the baseline against the stored tail-stability run (AC10) ----

def _probe_two_day_states(probe_states):
    """The probe note's records of the close before the KO observation and of the KO day, as a two-day note (the
    first day plays the inception: the ledger only needs consecutive recorded days)."""
    states, by_day = probe_states
    out = json.loads(json.dumps(states.to_json()))
    out["days"] = [r for r in out["days"] if r["date"] in (PROBE_DAY.isoformat(), PROBE_KO_DAY.isoformat())]
    assert len(out["days"]) == 2
    return out


def test_probe_ledger_rows_are_the_sellers_negative_delta_positive_hedge_and_negative_ko_cash(
        probe_states, real_chains, real_boards):
    from gpin.techniques import Base

    states = _probe_two_day_states(probe_states)
    run = _run(states, Base(), real_boards, real_chains, "close")
    day, ko = run.rows
    ko_rec = states["days"][1]
    assert day["date"] == PROBE_DAY.isoformat() and ko["date"] == PROBE_KO_DAY.isoformat()
    spot = real_boards[PROBE_DAY].spot
    # the seller is short the note: negative position delta, a long IM hedge of equal size
    assert day["position_delta"] < 0 and day["im_lots"] > 0
    assert day["position_delta"] == pytest.approx(-175.5e-2 * NOTIONAL_ / spot, rel=2e-3)    # the 175.5% bump delta
    assert day["hedge_ratio"] * NOTIONAL_ / spot == pytest.approx(-day["position_delta"], rel=1e-12)
    assert day["seller_pv"] == pytest.approx(-states["days"][0]["states"]["true"]["engine_price"][12])
    assert day["position_gamma"] == pytest.approx(-states["days"][0]["states"]["true"]["engine_gamma"][12])
    # the KO fires: the seller pays the coupon and the redemption leaves the book
    assert ko_rec["engine_note_cash"] > 0 and ko["note_cash"] == pytest.approx(-ko_rec["engine_note_cash"])
    assert ko["note_cash"] < 0 and ko["seller_pv"] == 0.0 and ko["im_lots"] == 0.0
    # the same under next_open: the hedge is carried through the fixing and flattened at the next IM open
    nxt = _run(states, Base(), real_boards, real_chains, "next_open")
    assert len(nxt.rows) == 3 and "fixing_deferred" in nxt.rows[1]["flags"] and nxt.rows[1]["im_lots"] > 0
    assert nxt.rows[2]["im_lots"] == 0.0 and nxt.rows[2]["note_pnl"] == 0.0 and "deferred_fill" in nxt.rows[2]["flags"]


def test_a_knocked_in_maturity_pays_the_seller_a_positive_cash():
    from gpin.techniques import Base

    d = _weekdays(3)
    states, boards, chains = _world([
        _day(d[0], 4000.0, -300000.0, -500.0, knocked_in=True, next_ko=d[2], im={"IM2403": dict(close=4000.0)}),
        _day(d[1], 4000.0, -310000.0, -500.0, knocked_in=True, next_ko=d[2], im={"IM2403": dict(close=4000.0)}),
        # maturity, knocked in: the holder of the note receives a negative amount (it loses): engine cash < 0
        _day(d[2], 3900.0, 0.0, 0.0, alive=False, cash=-1200000.0, knocked_in=True, end_reason="maturity",
             im={"IM2403": dict(close=3900.0)}),
    ])
    run = _run(states, Base(), boards, chains)
    assert run.rows[2]["note_cash"] == 1200000.0                                          # the seller receives the loss
    assert run.rows[0]["position_delta"] == 500.0 and run.rows[0]["im_lots"] < 0         # long note delta -> short IM hedge


EXISTING_RUNS = Path("/Users/fuxinyao/quant-ark/example/output/snowball_2y_tail_stability/runs")
AC10_INCEPTION = date(2023, 10, 9)
AC10_DAYS = 30


@pytest.fixture(scope="module")
def ac10_case(real_chains, real_anchors, real_scans, engine):
    """The stored tail-stability run of ``AC10_INCEPTION`` (tail converge_hl100) and the same note's first 30 days
    priced here (about 30 solves) under the stored coupon at the 75% book."""
    from types import SimpleNamespace

    from gpin.pricing import price_note

    path = EXISTING_RUNS / f"{AC10_INCEPTION.isoformat()}__converge_hl100.json"
    if not path.exists():
        pytest.skip("the tail-stability study's stored runs are absent")
    stored = json.loads(path.read_text(encoding="utf-8"))
    scan_rec = next(s for s in real_scans if s.inception == AC10_INCEPTION)
    states = price_note(AC10_INCEPTION, stored["coupon"], real_chains, real_anchors, scan_rec, engine=engine,
                        max_days=AC10_DAYS, book="ki75")
    return SimpleNamespace(stored=stored, states=states)


def test_baseline_under_close_reproduces_the_stored_note_pnl_and_book_a_pnl_row_by_row(
        ac10_case, real_chains, real_boards):
    from gpin.techniques import Base

    run = _run(ac10_case.states, Base(), real_boards, real_chains, "close")
    stored_days = ac10_case.stored["days"][:AC10_DAYS]
    mine = run.rows[1:AC10_DAYS + 1]                         # the stored rows start on day 1
    assert [r["date"] for r in mine] == [r["date"] for r in stored_days]
    d_note = max(abs(m["note_pnl"] - s["note_pnl"]) for m, s in zip(mine, stored_days))
    d_im = max(abs(m["im_vm"] + m["mo_mtm"] - s["pnl_a"]) for m, s in zip(mine, stored_days))
    print(f"AC10 max |note_pnl diff| = {d_note:.3e} CNY, max |im pnl diff| = {d_im:.3e} CNY")
    assert d_note < 1e-6 and d_im < 1e-6
    assert all(m["mo_mtm"] == 0.0 and m["legs"] == {} for m in mine)
    assert run.clip_count == 0                                # the clip does not bind on these days
    assert "data_end" in mine[-1]["flags"]


# ---- AC7: one synthetic path with an IM roll, a leg opened and closed, a surviving note and a fired KO, three clocks ----

def _combined_world(fire_ko):
    """Six days. IM2403 until day 2, IM2404 from day 3 (IM2403 is then on the board only). A call is wanted on days 1
    and 2 and closed on day 3. ``fire_ko``: the observation is day 4 and the KO fires (day 5 is the next chain day);
    else the note survives to the data end (day 5)."""
    d = _weekdays(6)
    last = d[0] + timedelta(days=60)
    vols = [0.2, 0.21, 0.22, 0.2, 0.2, 0.2]
    spots = [5000.0, 5050.0, 4980.0, 5020.0, 4950.0, 4900.0]
    ims = [
        {"IM2403": dict(close=5000.0, open=4995.0, vwap=4998.0, volume=900)},
        {"IM2403": dict(close=5060.0, open=5010.0, vwap=5040.0, volume=900)},
        {"IM2403": dict(close=4990.0, open=5050.0, vwap=5000.0, volume=900)},
        {"IM2403": dict(close=5030.0, listed=False), "IM2404": dict(close=5100.0, open=5040.0, vwap=5090.0, volume=700)},
        {"IM2404": dict(close=5020.0, open=5110.0, vwap=5060.0, volume=700)},
        {"IM2404": dict(close=4960.0, open=5000.0, vwap=4970.0, volume=700)},
    ]
    days = []
    for i, day in enumerate(d):
        q = {("C", 5000.0): (_px(day, last, "C", 5000.0, vols[i], forward=5000.0), None, 20000, 10000)}
        mo = {SERIES: dict(last=last, forward=5000.0, quotes=q)}
        dead = fire_ko and i >= 4
        days.append(_day(day, spots[i], 100000.0 - 500.0 * i, 800.0 + 100.0 * i, alive=not dead,
                         cash=900000.0 if (dead and i == 4) else 0.0, end_reason="ko" if dead else None,
                         next_ko=d[4] if (fire_ko and i < 4) else None, im=ims[i], mo=mo))
    return _world(days, n_records=5 if fire_ko else 6), d


@pytest.mark.parametrize("fire_ko", [False, True])
def test_the_identity_holds_in_all_three_clocks_with_a_roll_a_leg_and_a_fired_ko_or_a_survivor(fire_ko):
    from gpin.techniques import Base

    (states, boards, chains), d = _combined_world(fire_ko)
    tech_for = lambda: _Scripted(lambda ctx: {KEY_C5000: 1.0} if ctx.day_index in (1, 2) else {}, note_delta=900.0)
    runs = {m: _run(states, tech_for(), boards, chains, m, desk_notional=NOTIONAL_) for m in ("close", "twap", "next_open")}
    for mode, run in runs.items():
        assert any("roll" in r["flags"] for r in run.rows), mode
        assert [r["legs"] for r in run.rows[:4]] == [{}, {STR_C5000: 1.0}, {STR_C5000: 1.0}, {}], mode
        assert run.rows[-1]["im_lots"] == 0.0 and not run.rows[-1]["legs"], mode
        _check_identity(run)
    # fills differ between clocks, yet the end-of-day positions agree on every non-fixing day
    nonfixing = 4 if fire_ko else 6
    for k in range(nonfixing):
        assert runs["close"].rows[k]["im_lots"] == pytest.approx(runs["twap"].rows[k]["im_lots"]), k
        assert runs["close"].rows[k]["im_lots"] == pytest.approx(runs["next_open"].rows[k]["im_lots"]), k
        assert runs["close"].rows[k]["legs"] == runs["twap"].rows[k]["legs"] == runs["next_open"].rows[k]["legs"], k
    assert any(r["fill_pnl"] != 0.0 for r in runs["twap"].rows) and all(r["fill_pnl"] == 0.0 for r in runs["close"].rows)
    if fire_ko:
        # close: flat on the fixing day; next_open: the hedge is carried through it and flattened the day after
        assert len(runs["close"].rows) == 5 and runs["close"].rows[4]["im_lots"] == 0.0
        assert len(runs["next_open"].rows) == 6 and runs["next_open"].rows[4]["im_lots"] != 0.0
        assert runs["next_open"].rows[5]["im_lots"] == 0.0
        assert runs["next_open"].rows[5]["im_vm"] == pytest.approx(
            runs["next_open"].rows[4]["im_lots"] * (5000.0 - 5020.0) * 200.0)           # held lots x (open - close) x 200
        assert runs["close"].rows[4]["note_cash"] == -900000.0


# ---------------------------------------------------------------------------
# T5 — techniques: ledger-side fixes, the registry, every family, overlay construction, composites
# ---------------------------------------------------------------------------

# ---- the note's levels come from the state file (a seasoned note is not struck at its booking spot) ----

def _spy_note(states, boards, chains):
    spy = _Scripted(lambda ctx: {})
    _run(states, spy, boards, chains)
    return spy.seen[0].note


def test_note_levels_are_read_from_the_state_file_for_a_seasoned_note():
    d = _weekdays(3)
    days = [_day(x, 6420.0, 100000.0, 1000.0, im={"IM2403": dict(close=6400.0)}) for x in d]
    states, boards, chains = _world(days, book="seas6")
    # seas6 booked at 6420: KI level 7% below the booking spot = 6420 / 1.07 = 6000, strike = KO level = 6000 / 0.75 = 8000
    assert (states["strike"], states["ko_level"], states["ki_level"]) == pytest.approx((8000.0, 8000.0, 6000.0))
    note = _spy_note(states, boards, chains)
    assert note.book == "seas6" and note.s0 == 6420.0
    assert (note.strike, note.ko_level, note.ki_level) == pytest.approx((8000.0, 8000.0, 6000.0))
    assert note.ko_level == pytest.approx(1.2461 * 6420.0, rel=1e-4) and note.ki_level == pytest.approx(0.9346 * 6420.0, rel=1e-4)
    # the 2Y product is struck at its booking spot
    states, boards, chains = _world(days)
    note = _spy_note(states, boards, chains)
    assert (note.strike, note.ko_level, note.ki_level) == pytest.approx((6420.0, 6420.0, 4815.0))


@pytest.mark.parametrize("missing", ["strike", "ko_level", "ki_level"])
def test_a_state_without_its_levels_fails_closed(missing):
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    del states[missing]
    with pytest.raises(StudyDataError, match=missing):
        _run(states, Base(), boards, chains)


# ---- a held leg at or below its intrinsic bound: the zero-vol limit of the Black-76 delta, gamma 0 ----

def test_black_greeks_at_a_settlement_use_the_zero_vol_limit_at_or_below_the_intrinsic_bound():
    from gpin import black

    info = _series()                                    # forward 5000, 73 days
    spot = 4950.0
    unit = info.discount * 5000.0 / spot                # discount x forward / spot
    intrinsic = info.discount * 500.0                   # the 4500 call and the 5500 put are 500 points in the money
    # a settlement with an implied vol: Black-76 at that vol
    price, _, _ = _bs(info, 5100.0, 0.23, True)
    delta, gamma = black.greeks_at_settle(price, "C", 5100.0, info, spot)
    assert delta == pytest.approx(black.index_delta(0.23, "C", 5100.0, info, spot), rel=1e-9)
    assert gamma == pytest.approx(black.index_gamma(0.23, "C", 5100.0, info, spot), rel=1e-6)
    # below and at the intrinsic bound, in the money: +/- discount x forward / spot, gamma 0
    assert black.implied_vol(intrinsic - 1.0, "C", 4500.0, info) is None
    assert black.greeks_at_settle(intrinsic - 1.0, "C", 4500.0, info, spot) == (pytest.approx(unit, rel=1e-12), 0.0)
    assert black.greeks_at_settle(intrinsic, "C", 4500.0, info, spot) == (pytest.approx(unit, rel=1e-12), 0.0)
    assert black.greeks_at_settle(intrinsic - 1.0, "P", 5500.0, info, spot) == (pytest.approx(-unit, rel=1e-12), 0.0)
    # out of the money with a settlement at the bound 0: delta 0, gamma 0
    assert black.greeks_at_settle(0.0, "C", 5500.0, info, spot) == (0.0, 0.0)
    assert black.greeks_at_settle(0.0, "P", 4500.0, info, spot) == (0.0, 0.0)
    # it is the limit of the Black-76 delta as the vol goes to 0
    assert black.index_delta(1e-4, "C", 4500.0, info, spot) == pytest.approx(unit, rel=1e-9)
    assert black.index_delta(1e-4, "P", 5500.0, info, spot) == pytest.approx(-unit, rel=1e-9)
    assert black.index_delta(1e-4, "C", 5500.0, info, spot) == pytest.approx(0.0, abs=1e-12)
    # any other failed inversion has no greeks: above the upper bound, no time left, exactly at the money
    assert black.greeks_at_settle(info.discount * 5000.0 + 1.0, "C", 4500.0, info, spot) is None
    assert black.greeks_at_settle(intrinsic - 1.0, "C", 4500.0, _series(days_to_last=0), spot) is None
    assert black.greeks_at_settle(0.0, "C", 5000.0, info, spot) is None


def _itm_leg_world(settle_day2):
    """Four days; a 4500 call (forward 5000) is opened on day 1 at a 20% vol; ``settle_day2`` is its day-2 settlement."""
    d = _weekdays(4)
    last = d[0] + timedelta(days=60)
    q = [{("C", 4500.0): (_px(d[i], last, "C", 4500.0, 0.2), None, 20000, 10000)} for i in range(4)]
    q[2] = {("C", 4500.0): (settle_day2(_series_info(d[2], last, LEG_FORWARD)), None, 20000, 10000)}
    (states, boards, chains), d = _leg_world(4, last, q)
    tech = _Scripted(lambda ctx: {(SERIES, "C", 4500.0): 2.0} if ctx.day_index >= 1 else {})
    return states, boards, chains, tech, d, last


def test_a_held_leg_below_its_intrinsic_bound_is_hedged_with_the_zero_vol_delta():
    states, boards, chains, tech, d, last = _itm_leg_world(lambda info: info.discount * 500.0 - 2.0)
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_)
    r2 = run.rows[2]
    info = _series_info(d[2], last, LEG_FORWARD)
    assert r2["legs"] == {"MO2403:C:4500": 2.0} and "refused_open" not in r2["flags"]
    leg = 2.0 * 100.0 * info.discount * 5000.0 / 5000.0                   # lots x 100 x discount x forward / spot
    assert r2["im_lots"] == pytest.approx((1000.0 - leg) * 5000.0 / (200.0 * 5000.0), rel=1e-12)
    _check_identity(run)


def test_a_held_leg_with_any_other_failed_inversion_still_fails_closed():
    # a settlement above discount x forward: no vol, and not the intrinsic bound
    states, boards, chains, tech, _, _ = _itm_leg_world(lambda info: info.discount * 5000.0 + 5.0)
    with pytest.raises(StudyDataError, match="implied vol"):
        _run(states, tech, boards, chains, desk_notional=NOTIONAL_)


# ---- the registry (C6): one technique per pre-registered grid point, composites from a selection ----

#: every registered id without the composites, hand-written from the spec's registry table and its grids
T5_IDS = [
    "base", "base_noclip",
    "ko_shift@0.01", "ko_shift@0.02", "ko_shift@0.03",
    "ko_spread@0.015", "ko_spread@0.03",
    "ko_bend@0.015,3", "ko_bend@0.03,3", "ko_bend@0.03,5",
    "ki_shift@-0.01", "ki_shift@-0.02",
    "ko_wrong", "ki_wrong",
    "secant@0.03", "secant@0.05",
    "ki_cap@1.0", "ki_cap@1.25",
    "surv_hold@1", "surv_hold@3",
    "derisk@3", "derisk@5",
    "freeze@1", "freeze@2",
    "mo_ko@1,payoff", "mo_ko@2,payoff", "mo_ko@1,scenario",
    "mo_ki@2,payoff", "mo_ki@2,scenario",
    "band@0.02", "band@0.05",
]
T5_SELECTION = {"sop_futures": {"ko": "derisk@5", "ki": "ki_cap@1.25"},
                "sop_overlay": {"ko": "mo_ko@1,scenario", "ki": "mo_ki@2,payoff"}}
T5_ALL_IDS = T5_IDS + ["sop_futures", "sop_overlay"]


def _registry(selection=T5_SELECTION):
    from gpin.techniques import build_registry

    return build_registry(load_prereg(), selection)


def test_registry_has_one_technique_per_grid_point_with_stable_ids():
    from gpin.techniques import Base, BaseNoClip, Technique, build_registry

    reg = build_registry(load_prereg())
    assert list(reg) == T5_IDS                                  # the pre-registration's order; no composite without a selection
    assert all(isinstance(t, Technique) and t.id == key for key, t in reg.items())
    assert type(reg["base"]) is Base and type(reg["base_noclip"]) is BaseNoClip
    assert [k for k, t in reg.items() if not t.clip] == ["base_noclip"]
    # id, row, family and the grid point as params
    rows = {t["id"]: t for t in load_prereg()["techniques"]}
    for key, tech in reg.items():
        row = rows[tech.row]
        assert key.split("@")[0] == tech.row and tech.family == row["family"] and tech.params in row["grid"]
    assert reg["ko_bend@0.03,5"].params == {"e": 0.03, "n_b": 5} and reg["ko_bend@0.03,5"].family == "1 barrier bending"
    assert reg["mo_ko@1,scenario"].params == {"steps": 1, "sizing": "scenario"}
    assert reg["ki_shift@-0.02"].params == {"x": -0.02} and reg["ki_cap@1.0"].params == {"c": 1.0}
    assert reg["ko_wrong"].params == {} and reg["ko_wrong"].family == "control"
    # the number of grid points per row is the pre-registration's
    for row_id, (_, _, n_points) in EXPECTED_TECHNIQUES.items():
        if row_id not in ("sop_futures", "sop_overlay"):
            assert sum(1 for t in reg.values() if t.row == row_id) == n_points, row_id


def test_grid_ids_are_the_values_in_the_grids_key_order():
    from gpin.techniques import grid_id

    assert grid_id("base", {}) == "base"
    assert grid_id("ko_shift", {"size": 0.02}) == "ko_shift@0.02"
    assert grid_id("ko_bend", {"e": 0.015, "n_b": 3}) == "ko_bend@0.015,3"
    assert grid_id("mo_ko", {"steps": 2, "sizing": "payoff"}) == "mo_ko@2,payoff"
    assert grid_id("ki_cap", {"c": 1.0}) == "ki_cap@1.0" and grid_id("ki_shift", {"x": -0.01}) == "ki_shift@-0.01"


def test_composites_exist_only_with_a_selection_and_a_selection_must_name_registered_members():
    from gpin.techniques import build_registry

    prereg = load_prereg()
    assert "sop_futures" not in build_registry(prereg) and "sop_overlay" not in build_registry(prereg)
    reg = build_registry(prereg, T5_SELECTION)
    assert list(reg) == T5_ALL_IDS
    assert reg["sop_futures"].family == "SOP composite" and reg["sop_overlay"].row == "sop_overlay"
    assert reg["sop_futures"].params == {"ko": "derisk@5", "ki": "ki_cap@1.25"}
    assert reg["sop_overlay"].params == {"futures": "sop_futures", "ko": "mo_ko@1,scenario", "ki": "mo_ki@2,payoff"}
    bad = [
        {"sop_futures": {"ko": "derisk@5"}, "sop_overlay": T5_SELECTION["sop_overlay"]},                    # a slot missing
        {"sop_futures": T5_SELECTION["sop_futures"]},                                                         # a composite missing
        {"sop_futures": {"ko": "derisk@4", "ki": "ki_cap@1.25"}, "sop_overlay": T5_SELECTION["sop_overlay"]},   # not a grid point
        {"sop_futures": {"ko": "ki_cap@1.0", "ki": "ki_cap@1.25"}, "sop_overlay": T5_SELECTION["sop_overlay"]},  # not a KO row
        {"sop_futures": {"ko": "mo_ko@1,payoff", "ki": "ki_cap@1.25"}, "sop_overlay": T5_SELECTION["sop_overlay"]},
        {"sop_futures": T5_SELECTION["sop_futures"], "sop_overlay": {"ko": "mo_ki@2,payoff", "ki": "mo_ki@2,payoff"}},
        {"sop_futures": T5_SELECTION["sop_futures"], "sop_overlay": {"ko": "mo_ko@1,payoff", "ki": "ki_shift@-0.01"}},
    ]
    for selection in bad:
        with pytest.raises(ValidationError):
            build_registry(prereg, selection)


def test_module_registry_follows_the_selection_file(tmp_path):
    from gpin import techniques as tq

    # no results/selection.json yet: the composites are undefined, not defaulted
    assert tq.SELECTION_PATH == gpin.RESULTS_DIR / "selection.json"
    expected = T5_ALL_IDS if tq.SELECTION_PATH.exists() else T5_IDS
    assert list(tq.REGISTRY) == expected
    assert tq.composite_selection(tmp_path / "selection.json") is None
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"composites": T5_SELECTION, "hashes": {}}), encoding="utf-8")
    assert tq.composite_selection(path) == T5_SELECTION
    assert list(tq.build_registry(load_prereg(), tq.composite_selection(path))) == T5_ALL_IDS
    path.write_text(json.dumps({"picks": {}}), encoding="utf-8")
    with pytest.raises(StudyDataError):                          # a selection file without the composites' members
        tq.composite_selection(path)


def test_an_unknown_row_or_an_unpriced_shift_fails_closed():
    from gpin.techniques import build_registry

    prereg = json.loads(json.dumps(load_prereg()))
    prereg["techniques"].append({"id": "magic", "family": "x", "role": "candidate", "barrier": "ko", "grid": [{}]})
    with pytest.raises(ValidationError, match="magic"):
        build_registry(prereg)
    prereg = json.loads(json.dumps(load_prereg()))
    next(t for t in prereg["techniques"] if t["id"] == "ko_shift")["grid"].append({"size": 0.025})   # no ko_s250 model
    with pytest.raises(ValidationError, match="0.025"):
        build_registry(prereg)
    prereg = json.loads(json.dumps(load_prereg()))
    next(t for t in prereg["techniques"] if t["id"] == "secant")["grid"].append({"h": 0.035})       # not a ladder point
    with pytest.raises(ValidationError):
        build_registry(prereg)


# ---- hand-computed targets, one family at a time, on hand-made day contexts ----

T5_DAYS = _weekdays(12)            # 2024-01-02 .. 2024-01-17: also consecutive SSE trading days
T5_DAY = T5_DAYS[4]                # 2024-01-08


def _t5_note(*, book="ki75", s0=5000.0, strike=5000.0, ko=5000.0, ki=3750.0):
    from gpin.techniques import NoteInfo

    return NoteInfo(book, WORLD_START, 0.1, 0.75, 1.0, s0, strike, ko, ki)


def _t5_record(day, spot, deltas, *, n=None, gap_ko=None, ko_dir=None, knocked_in=False, prices=None, ladders=None):
    """A day record with a linear ladder of slope ``deltas[model]`` (level ``prices[model]`` at the spot) per model;
    ``ladders`` replaces whole ladders. ``n``: trading days to the next KO observation (dated ``n`` weekdays ahead)."""
    prices = prices or {}
    states = {m: _ladder_state(prices.get(m, 0.0), d, 0.0, spot) for m, d in deltas.items()}
    states.update(ladders or {})
    next_ko = None if n is None else _weekdays(n + 1, day)[-1]
    return {"date": day.isoformat(), "spot": spot, "alive": True, "knocked_in": knocked_in, "engine_note_cash": 0.0,
            "end_reason": None, "next_ko": None if next_ko is None else next_ko.isoformat(), "days_to_ko": n,
            "gap_ko": gap_ko, "ko_dir": ko_dir, "states": states}


def _t5_board(day, spot, *, im=None, mo=None):
    """``im``: ``{code: close}``; ``mo``: ``{series: dict(last, forward, quotes={(cp, strike): (settle, volume, oi)})}``."""
    from gpin.board import DayBoard, ImBar, OptionQuote

    bars = {code: ImBar(None, close, close, None, 0) for code, close in (im or {}).items()}
    quotes, infos = {}, {}
    for series, s in (mo or {}).items():
        infos[series] = _series_info(day, s["last"], s["forward"])
        quotes[series] = {k: OptionQuote(q[0], None, None, None, q[1], q[2]) for k, q in s["quotes"].items()}
    return DayBoard(day, spot, bars, quotes, infos)


def _t5_ctx(record, *, vol=0.2, board=None, note=None, held_im=None, held_legs=None, prev=None, index=5):
    from gpin.techniques import DayContext

    day = date.fromisoformat(record["date"])
    board = board if board is not None else _t5_board(day, record["spot"])
    return DayContext.build(day, index, record["spot"], vol, record, board, note or _t5_note(), held_im or {},
                            held_legs or {}, prev)


#: hedge deltas of the hand-made KO-window day: the true note and every KO model
KO_DELTAS = {"true": 1000.0, "surv": 450.0, "ko_s050": 960.0, "ko_s100": 900.0, "ko_s150": 800.0, "ko_s200": 700.0,
             "ko_s300": 600.0, "ko_w100": 1100.0}
#: of the hand-made KI-window day
KI_DELTAS = {"true": 2000.0, "ki_m100": 1700.0, "ki_m200": 1500.0, "ki_p100": 2300.0, "kin": 1200.0}


def _ko_ctx(n=3, spot=5100.0, deltas=None, *, ko_dir=-1, gap_ko=400000.0, **kw):
    """A KO-window day: barrier 5000, flat vol 20%; z = ln(S / 5000) / (0.2 / sqrt(244) x sqrt(n))."""
    return _t5_ctx(_t5_record(T5_DAY, spot, KO_DELTAS if deltas is None else deltas, n=n, gap_ko=gap_ko, ko_dir=ko_dir), **kw)


def _ki_ctx(spot=3900.0, deltas=None, *, n=40, knocked_in=False, **kw):
    """A KI-window day: KI barrier 3750, spot 4% above it; the next KO observation is far."""
    return _t5_ctx(_t5_record(T5_DAY, spot, KI_DELTAS if deltas is None else deltas, n=n, knocked_in=knocked_in), **kw)


def _far_ctx(spot=4400.0, delta=1000.0, **kw):
    """A day outside both windows (spot 12% below the KO barrier and 17% above the KI barrier): only the true state."""
    return _t5_ctx(_t5_record(T5_DAY, spot, {"true": delta}, n=3), **kw)


def _delta_only(target, value):
    assert target.note_delta == pytest.approx(value, rel=1e-12)
    assert dict(target.legs) == {} and target.note_delta_if_filled is None and tuple(target.flags) == ()


def test_window_and_trigger_predicates_use_the_pre_registered_numbers_and_agree_with_the_pricing_pass():
    from gpin import pricing
    from gpin import techniques as tq

    s = tq.Settings.from_prereg(load_prereg())
    assert (s.ko_days, s.ko_band, s.ki_band, s.trigger_z, s.days_per_year) == (10, 0.10, 0.05, 2.0, 244)
    assert (s.overlay_ko_days, s.overlay_ki_min_days_left, s.bump, s.scenario_cap) == (5, 3, 0.01, 3.0)
    assert (s.min_volume, s.min_open_interest) == (100, 500)
    # KO window: at most 10 trading days ahead and within 10% of the barrier (inclusive)
    for n, spot, expect in [(10, 5000.0, True), (11, 5000.0, False), (3, 5500.0, True), (3, 5500.5, False),
                            (3, 4500.0, True), (3, 4499.0, False), (None, 5000.0, False)]:
        ctx = _t5_ctx(_t5_record(T5_DAY, spot, {"true": 1.0}, n=n))
        assert tq.in_ko_window(ctx, s) is expect, (n, spot)
        assert pricing.in_ko_window(spot, 5000.0, n, book="ki75") is expect
    # KI window: not knocked in and at most 5% above the KI barrier 3750 (inclusive)
    for spot, ki, expect in [(3937.5, False, True), (3938.0, False, False), (3700.0, False, True), (3800.0, True, False)]:
        ctx = _t5_ctx(_t5_record(T5_DAY, spot, {"true": 1.0}, n=40, knocked_in=ki))
        assert tq.in_ki_window(ctx, s) is expect, (spot, ki)
        assert pricing.in_ki_window(spot, 5000.0, ki, book="ki75") is expect
    # z = ln(S / B) / (vol / sqrt(244) x sqrt(max(n, 1))); the trigger is |z| <= 2 inside the KO window
    ctx = _ko_ctx(n=3, spot=5200.0)
    z = math.log(5200.0 / 5000.0) / (0.2 / math.sqrt(244.0) * math.sqrt(3.0))
    assert z == pytest.approx(1.7686, abs=1e-4) and tq.ko_z(ctx, s) == pytest.approx(z, rel=1e-12)
    assert tq.ko_trigger(ctx, s) is True
    assert tq.ko_trigger(_ko_ctx(n=3, spot=5250.0), s) is False            # z = 2.20: in the window, outside the trigger
    assert tq.ko_trigger(_ko_ctx(n=3, spot=4800.0), s) is True             # z = -1.84
    assert tq.ko_trigger(_ko_ctx(n=3, spot=4750.0), s) is False            # z = -2.31
    assert tq.ko_trigger(_ko_ctx(n=11, spot=5000.0), s) is False           # z = 0 but outside the window
    with pytest.raises(ValidationError):
        tq.ko_z(_ko_ctx(n=3, spot=5200.0, vol=0.0), s)


def test_family_1_barrier_shift_and_spread_hand_computed():
    reg = _registry()
    ctx = _ko_ctx(n=8)
    _delta_only(reg["base"].target(ctx), 1000.0)
    _delta_only(reg["ko_shift@0.01"].target(ctx), 900.0)                   # d(ko_s100)
    _delta_only(reg["ko_shift@0.02"].target(ctx), 700.0)                   # d(ko_s200)
    _delta_only(reg["ko_shift@0.03"].target(ctx), 600.0)                   # d(ko_s300)
    _delta_only(reg["ko_wrong"].target(ctx), 1100.0)                       # d(ko_w100)
    # the digital booked as a spread of width e: the mean over the shifts 0, e/3, 2e/3, e (shift 0 is the true note)
    _delta_only(reg["ko_spread@0.015"].target(ctx), (1000.0 + 960.0 + 900.0 + 800.0) / 4.0)       # 915
    _delta_only(reg["ko_spread@0.03"].target(ctx), (1000.0 + 900.0 + 700.0 + 600.0) / 4.0)        # 800
    ki = _ki_ctx()
    _delta_only(reg["ki_shift@-0.01"].target(ki), 1700.0)                  # d(ki_m100)
    _delta_only(reg["ki_shift@-0.02"].target(ki), 1500.0)                  # d(ki_m200)
    _delta_only(reg["ki_wrong"].target(ki), 2300.0)                        # d(ki_p100)


def test_family_1_bending_is_the_spread_only_near_the_observation_and_under_the_trigger():
    reg = _registry()
    spread3, spread15 = 800.0, 915.0
    _delta_only(reg["ko_bend@0.03,3"].target(_ko_ctx(n=3)), spread3)
    _delta_only(reg["ko_bend@0.03,3"].target(_ko_ctx(n=4)), 1000.0)        # n > n_b: the baseline
    _delta_only(reg["ko_bend@0.03,5"].target(_ko_ctx(n=5)), spread3)
    _delta_only(reg["ko_bend@0.03,5"].target(_ko_ctx(n=6)), 1000.0)
    _delta_only(reg["ko_bend@0.015,3"].target(_ko_ctx(n=1, spot=5050.0)), spread15)
    _delta_only(reg["ko_bend@0.03,3"].target(_ko_ctx(n=3, spot=5250.0)), 1000.0)   # z = 2.20: outside the trigger
    _delta_only(reg["ko_bend@0.03,3"].target(_ko_ctx(n=3, spot=5200.0)), spread3)  # z = 1.77
    # the plain spread and the shift do not look at the trigger
    _delta_only(reg["ko_spread@0.03"].target(_ko_ctx(n=3, spot=5250.0)), spread3)
    _delta_only(reg["ko_shift@0.02"].target(_ko_ctx(n=3, spot=5250.0)), 700.0)


def test_family_2_futures_only_rules_hand_computed():
    reg = _registry()
    # secant: (V(S(1+h)) - V(S(1-h))) / (2hS) on the true ladder. Lift the +3% point by 30,000 CNY: the 3% secant
    # gains 30,000 / (2 x 0.03 x 5100) = 98.04 per point, the 1% bump delta and the 5% secant are unchanged
    lifted = _ladder_state(0.0, 1000.0, 0.0, 5100.0)
    lifted["engine_price"][15] += 30000.0
    rec = _t5_record(T5_DAY, 5100.0, KO_DELTAS, n=8, gap_ko=1.0, ko_dir=-1, ladders={"true": lifted})
    _delta_only(reg["base"].target(_t5_ctx(rec)), 1000.0)
    _delta_only(reg["secant@0.03"].target(_t5_ctx(rec)), 1000.0 + 30000.0 / (2 * 0.03 * 5100.0))
    _delta_only(reg["secant@0.05"].target(_t5_ctx(rec)), 1000.0)
    # the same in the KI window, and outside both windows (the baseline)
    lifted_ki = _ladder_state(0.0, 2000.0, 0.0, 3900.0)
    lifted_ki["engine_price"][17] -= 39000.0                                # the +5% point
    rec = _t5_record(T5_DAY, 3900.0, KI_DELTAS, n=40, ladders={"true": lifted_ki})
    _delta_only(reg["secant@0.05"].target(_t5_ctx(rec)), 2000.0 - 39000.0 / (2 * 0.05 * 3900.0))    # 1900
    _delta_only(reg["secant@0.03"].target(_t5_ctx(rec)), 2000.0)
    far = _ladder_state(0.0, 1000.0, 0.0, 4400.0)
    far["engine_price"][15] += 30000.0
    _delta_only(reg["secant@0.03"].target(_t5_ctx(_t5_record(T5_DAY, 4400.0, {}, n=3, ladders={"true": far}))), 1000.0)
    # ki_cap: min(d(true), c x d(kin)) in the KI window
    _delta_only(reg["ki_cap@1.0"].target(_ki_ctx()), 1200.0)               # min(2000, 1200)
    _delta_only(reg["ki_cap@1.25"].target(_ki_ctx()), 1500.0)              # min(2000, 1.25 x 1200)
    low = dict(KI_DELTAS, true=1100.0)
    _delta_only(reg["ki_cap@1.0"].target(_ki_ctx(deltas=low)), 1100.0)     # the cap does not bind
    # surv_hold: d(surv) on the last n_s closes before the observation under the trigger
    _delta_only(reg["surv_hold@1"].target(_ko_ctx(n=1, spot=5050.0)), 450.0)
    _delta_only(reg["surv_hold@1"].target(_ko_ctx(n=2, spot=5050.0)), 1000.0)
    _delta_only(reg["surv_hold@3"].target(_ko_ctx(n=3)), 450.0)
    _delta_only(reg["surv_hold@3"].target(_ko_ctx(n=4)), 1000.0)
    _delta_only(reg["surv_hold@3"].target(_ko_ctx(n=3, spot=5250.0)), 1000.0)      # outside the trigger


def test_family_4_derisk_is_a_linear_blend_to_the_survived_branch():
    reg = _registry()
    # n_d = 5: weight on d(surv) = (5 - n) / 4: n = 5 -> 1000, 4 -> 862.5, 3 -> 725, 2 -> 587.5, 1 -> 450
    for n, expect in [(6, 1000.0), (5, 1000.0), (4, 862.5), (3, 725.0), (2, 587.5), (1, 450.0)]:
        _delta_only(reg["derisk@5"].target(_ko_ctx(n=n, spot=5020.0)), expect)
    # n_d = 3: 3 -> 1000, 2 -> 725, 1 -> 450
    for n, expect in [(4, 1000.0), (3, 1000.0), (2, 725.0), (1, 450.0)]:
        _delta_only(reg["derisk@3"].target(_ko_ctx(n=n, spot=5020.0)), expect)
    _delta_only(reg["derisk@5"].target(_ko_ctx(n=2, spot=5250.0)), 1000.0)         # z = 2.7 at n = 2: no trigger


# ---- settled rules: the baseline when a model state is missing, when ko_dir is 0, on a knock-in day ----

NEEDS_KO_MODEL = ["ko_shift@0.01", "ko_shift@0.02", "ko_shift@0.03", "ko_spread@0.015", "ko_spread@0.03",
                  "ko_bend@0.015,3", "ko_bend@0.03,3", "ko_bend@0.03,5", "ko_wrong", "surv_hold@1", "surv_hold@3",
                  "derisk@3", "derisk@5", "mo_ko@1,payoff", "mo_ko@2,payoff", "mo_ko@1,scenario"]
NEEDS_KI_MODEL = ["ki_shift@-0.01", "ki_shift@-0.02", "ki_wrong", "ki_cap@1.0", "ki_cap@1.25", "mo_ki@2,payoff",
                  "mo_ki@2,scenario"]
SHIFT_IDS = ["ko_shift@0.01", "ko_shift@0.02", "ko_shift@0.03", "ko_spread@0.015", "ko_spread@0.03", "ko_bend@0.015,3",
             "ko_bend@0.03,3", "ko_bend@0.03,5", "ko_wrong"]


@pytest.mark.parametrize("tid", NEEDS_KO_MODEL)
def test_a_ko_technique_returns_the_baseline_when_its_model_has_no_state(tid):
    # one day before the observation, at the barrier, under the trigger: yet only the true model was recorded
    ctx = _t5_ctx(_t5_record(T5_DAY, 5010.0, {"true": 1000.0}, n=1))
    _delta_only(_registry()[tid].target(ctx), 1000.0)


@pytest.mark.parametrize("tid", NEEDS_KI_MODEL)
def test_a_ki_technique_returns_the_baseline_when_its_model_has_no_state_and_on_a_knock_in_day(tid):
    tech = _registry()[tid]
    _delta_only(tech.target(_t5_ctx(_t5_record(T5_DAY, 3900.0, {"true": 2000.0}, n=40))), 2000.0)
    # the knock-in day: the close is below the barrier, the note is knocked in, no KI model is priced
    _delta_only(tech.target(_t5_ctx(_t5_record(T5_DAY, 3700.0, {"true": 2000.0}, n=40, knocked_in=True))), 2000.0)
    # even a (contradictory) record that still carries KI states is not used once the note is knocked in
    _delta_only(tech.target(_ki_ctx(spot=3700.0, knocked_in=True)), 2000.0)


@pytest.mark.parametrize("tid", SHIFT_IDS)
def test_a_shift_technique_returns_the_baseline_when_ko_dir_is_zero(tid):
    tech = _registry()[tid]
    # ko_dir = 0: no shift model was priced for this observation (surv is)
    ctx = _t5_ctx(_t5_record(T5_DAY, 5010.0, {"true": 1000.0, "surv": 450.0}, n=1, gap_ko=0.0, ko_dir=0))
    _delta_only(tech.target(ctx), 1000.0)
    _delta_only(tech.target(_ko_ctx(n=1, spot=5010.0, ko_dir=0, gap_ko=0.0)), 1000.0)   # even if shift states were there


@pytest.mark.parametrize("tid", T5_ALL_IDS)
def test_every_technique_returns_the_baseline_outside_both_windows(tid):
    _delta_only(_registry()[tid].target(_far_ctx()), 1000.0)


def test_techniques_read_only_models_the_pricing_pass_prices():
    from gpin.models import MODELS

    reg = _registry()
    for tech in reg.values():
        assert set(tech.models) <= set(MODELS), tech.id
    assert reg["ko_shift@0.02"].models == ("ko_s200",) and reg["ko_wrong"].models == ("ko_w100",)
    assert reg["ko_spread@0.015"].models == ("true", "ko_s050", "ko_s100", "ko_s150")
    assert reg["ko_bend@0.03,5"].models == ("true", "ko_s100", "ko_s200", "ko_s300")
    assert reg["ki_shift@-0.01"].models == ("ki_m100",) and reg["ki_wrong"].models == ("ki_p100",)
    assert reg["ki_cap@1.0"].models == ("kin",) and reg["surv_hold@1"].models == ("surv",) == reg["derisk@3"].models
    assert reg["mo_ki@2,payoff"].models == ("ki_m200", "kin")
    assert reg["base"].models == () == reg["freeze@1"].models == reg["band@0.02"].models == reg["mo_ko@1,payoff"].models


# ---- the delta carried (freeze, band): the IM held at the board's closes plus the legs held ----

def _t5_world(specs, *, strike=5000.0, ki=3750.0, book="ki75"):
    """``(states, boards, chains)`` from day specs ``dict(day, spot, deltas, n, gap_ko, ko_dir, knocked_in, prices, vol,
    im={code: close}, mo)``: records by ``_t5_record``, boards by ``_t5_board``, one chain row per day whose IM quotes
    are the board's closes (the last one is the far contract the ledger hedges with)."""
    from s2ts.chain import DayChain, ImQuote, MoPillar

    records, boards, chains = [], {}, []
    for sp in specs:
        d, spot = sp["day"], sp["spot"]
        records.append(_t5_record(d, spot, sp["deltas"], n=sp.get("n"), gap_ko=sp.get("gap_ko"), ko_dir=sp.get("ko_dir"),
                                  knocked_in=sp.get("knocked_in", False), prices=sp.get("prices")))
        boards[d] = _t5_board(d, spot, im=sp["im"], mo=sp.get("mo"))
        quotes = tuple(ImQuote(code, d + timedelta(days=60), 60 / 365.0, close, close) for code, close in sp["im"].items())
        pillar = MoPillar("MO0000", d + timedelta(days=60), 60 / 365.0, spot, 1.0, 1.0, spot, sp.get("vol", 0.2), ())
        chains.append(DayChain(d, spot, quotes, pillar, {}, 0.0))
    states = {"book": book, "ki_pct": 0.75, "inception": specs[0]["day"].isoformat(), "coupon": 0.1, "days": records,
              "meta": {}, "spot0": specs[0]["spot"], "strike": strike, "ko_level": strike, "ki_level": ki}
    return states, boards, chains


def _ov_mo(day, lasts, strikes, forward, vol_of=lambda k: 0.2, volume=5000, oi=10000):
    """A hand-made MO board: per series (``lasts``: name -> last trading day) the call and the put of every strike, settled
    at the Black-76 price of ``vol_of(strike)`` on ``forward``, with ``volume`` lots traded and ``oi`` lots open."""
    return {name: dict(last=last, forward=forward,
                       quotes={(cp, k): (_px(day, last, cp, k, vol_of(k), forward), volume, oi)
                               for k in strikes for cp in "CP"})
            for name, last in lasts.items()}


def test_the_delta_carried_is_the_im_held_at_the_boards_closes_plus_the_legs_held():
    from gpin import techniques as tq

    # 7 lots of IM2403 at 5080 and 2 lots of IM2406 at 5050, spot 5100: (7 x 5080 + 2 x 5050) x 200 / 5100 per index point
    board = _t5_board(T5_DAY, 5100.0, im={"IM2403": 5080.0, "IM2406": 5050.0, "IM2409": 5000.0})
    ctx = _ko_ctx(n=3, spot=5100.0, board=board, held_im={"IM2403": 7.0})
    assert tq.carried_delta(ctx) == pytest.approx(7.0 * 200.0 * 5080.0 / 5100.0, rel=1e-12)
    ctx = _ko_ctx(n=3, spot=5100.0, board=board, held_im={"IM2403": 7.0, "IM2406": 2.0})
    assert tq.carried_delta(ctx) == pytest.approx((7.0 * 5080.0 + 2.0 * 5050.0) * 200.0 / 5100.0, rel=1e-12)
    assert tq.carried_delta(_ko_ctx(n=3, spot=5100.0, board=board)) == 0.0
    with pytest.raises(StudyDataError, match="IM2412"):                    # a held contract the board cannot mark
        tq.carried_delta(_ko_ctx(n=3, spot=5100.0, board=board, held_im={"IM2412": 1.0}))
    # a leg held: 3 calls of 100 CNY per point at their Black-76 index delta (discount x N(d1) x forward / spot)
    last = date(2024, 2, 16)
    mo = _ov_mo(T5_DAY, {"MO2402": last}, [5000.0, 5100.0], 5090.0, lambda k: 0.22)
    board = _t5_board(T5_DAY, 5100.0, im={"IM2403": 5080.0}, mo=mo)
    info = _series_info(T5_DAY, last, 5090.0)
    leg = 3.0 * 100.0 * info.discount * _bs(info, 5100.0, 0.22, True)[1] * 5090.0 / 5100.0
    ctx = _ko_ctx(n=3, spot=5100.0, board=board, held_im={"IM2403": 7.0}, held_legs={("MO2402", "C", 5100.0): 3.0})
    assert tq.carried_delta(ctx) == pytest.approx(7.0 * 200.0 * 5080.0 / 5100.0 + leg, rel=1e-9)
    # d1 = (ln(5090 / 5100) + 0.5 x 0.0719 ^ 2) / 0.0719 = 0.0087, N(d1) = 0.5035, x discount 0.9985 x 5090 / 5100
    assert leg == pytest.approx(150.5, abs=0.1)
    with pytest.raises(StudyDataError, match="MO2402:C:5200"):              # a held leg the board cannot mark
        tq.carried_delta(_ko_ctx(n=3, spot=5100.0, board=board, held_legs={("MO2402", "C", 5200.0): 1.0}))


def test_family_4_freeze_holds_the_delta_carried_on_the_last_closes_before_the_observation():
    reg = _registry()
    board = _t5_board(T5_DAY, 5050.0, im={"IM2403": 5030.0})
    held = {"IM2403": 5.2}
    carried = 5.2 * 200.0 * 5030.0 / 5050.0                                # 1035.88 per index point
    assert carried == pytest.approx(1035.88, abs=0.01)
    kw = dict(spot=5050.0, board=board, held_im=held)
    _delta_only(reg["freeze@1"].target(_ko_ctx(n=1, **kw)), carried)       # the last close before the observation
    _delta_only(reg["freeze@1"].target(_ko_ctx(n=2, **kw)), 1000.0)
    _delta_only(reg["freeze@2"].target(_ko_ctx(n=2, **kw)), carried)
    _delta_only(reg["freeze@2"].target(_ko_ctx(n=1, **kw)), carried)
    _delta_only(reg["freeze@2"].target(_ko_ctx(n=3, **kw)), 1000.0)
    # outside the trigger (n = 1, spot 5200: z = ln(1.04) / 0.0128 = 3.06): the baseline
    far = _t5_board(T5_DAY, 5200.0, im={"IM2403": 5180.0})
    _delta_only(reg["freeze@1"].target(_ko_ctx(n=1, spot=5200.0, board=far, held_im=held)), 1000.0)
    # nothing held: nothing is traded
    _delta_only(reg["freeze@1"].target(_ko_ctx(n=1, spot=5050.0)), 0.0)
    # it needs no hedge model
    ctx = _t5_ctx(_t5_record(T5_DAY, 5050.0, {"true": 1000.0}, n=1), board=board, held_im=held)
    _delta_only(reg["freeze@1"].target(ctx), carried)


def _im_only_world(deltas, ns, closes, spot=5000.0):
    days = _weekdays(len(deltas))
    return _t5_world([dict(day=days[i], spot=spot, deltas={"true": deltas[i]}, n=ns[i], im={"IM2403": closes[i]})
                      for i in range(len(deltas))])


def test_freeze_makes_no_im_trade_in_the_ledger_on_the_last_closes_before_the_observation():
    reg = _registry()
    # spot 5000 = the barrier every day; the observation is day 4; the hedge delta and the IM close move every day
    deltas = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    closes = [5000.0, 5010.0, 5030.0, 4990.0, 5000.0, 5000.0]
    states, boards, chains = _im_only_world(deltas, [4, 3, 2, 1, 20, 19], closes)
    lots = [deltas[i] * 5000.0 / (200.0 * closes[i]) for i in range(6)]    # the baseline's lots: delta x S / (200 x F)
    base = _run(states, reg["base"], boards, chains)
    assert [r["im_lots"] for r in base.rows] == pytest.approx(lots[:5] + [0.0])
    two = _run(states, reg["freeze@2"], boards, chains)
    assert [r["im_lots"] for r in two.rows] == pytest.approx([lots[0], lots[1], lots[1], lots[1], lots[4], 0.0])
    assert two.rows[2]["cost"] == 0.0 and two.rows[3]["cost"] == 0.0       # no trade on the two frozen closes
    assert two.rows[1]["cost"] > 0.0 and two.rows[4]["cost"] > 0.0
    one = _run(states, reg["freeze@1"], boards, chains)
    assert [r["im_lots"] for r in one.rows] == pytest.approx([lots[0], lots[1], lots[2], lots[2], lots[4], 0.0])
    assert one.rows[3]["cost"] == 0.0 and one.rows[2]["cost"] > 0.0
    _check_identity(two)


def test_band_trades_im_only_beyond_the_band_outside_the_windows_and_never_inside_one():
    reg = _registry()
    # outside both windows at spot 4400: notional delta = 10 mio / 4400 = 2272.7 per point; 2% of it 45.5, 5% of it 113.6
    board = _t5_board(T5_DAY, 4400.0, im={"IM2403": 4400.0})

    def far(lots):
        return _far_ctx(board=board, held_im={"IM2403": lots})

    # 4.7 lots carry 940: the baseline 1000 is 60 away, beyond the 2% band and inside the 5% band
    _delta_only(reg["band@0.02"].target(far(4.7)), 1000.0)
    _delta_only(reg["band@0.05"].target(far(4.7)), 940.0)
    # 4.8 lots carry 960 (40 away): no trade under either; 4.4 lots carry 880 (120 away): both trade; so does 5.3 (60 above)
    _delta_only(reg["band@0.02"].target(far(4.8)), 960.0)
    _delta_only(reg["band@0.05"].target(far(4.8)), 960.0)
    _delta_only(reg["band@0.02"].target(far(4.4)), 1000.0)
    _delta_only(reg["band@0.05"].target(far(4.4)), 1000.0)
    _delta_only(reg["band@0.02"].target(far(5.3)), 1000.0)
    _delta_only(reg["band@0.05"].target(far(5.3)), 1060.0)
    # no band inside the KO window nor inside the KI window: the baseline, however close the hedge carried is
    ko_board = _t5_board(T5_DAY, 5100.0, im={"IM2403": 5100.0})
    _delta_only(reg["band@0.05"].target(_ko_ctx(n=8, board=ko_board, held_im={"IM2403": 4.99})), 1000.0)
    ki_board = _t5_board(T5_DAY, 3900.0, im={"IM2403": 3900.0})
    _delta_only(reg["band@0.05"].target(_ki_ctx(board=ki_board, held_im={"IM2403": 9.99})), 2000.0)


def test_band_makes_no_im_trade_in_the_ledger_inside_the_band():
    reg = _registry()
    # no observation ahead and far from the KI barrier; spot and IM at 5000: 2% of the notional delta 2000 is 40
    deltas = [1000.0, 1010.0, 1099.0, 1090.0, 1095.0]
    states, boards, chains = _im_only_world(deltas, [None] * 5, [5000.0] * 5)
    narrow = _run(states, reg["band@0.02"], boards, chains)
    assert [r["im_lots"] for r in narrow.rows] == pytest.approx([5.0, 5.0, 5.495, 5.495, 0.0])
    assert narrow.rows[1]["cost"] == 0.0 and narrow.rows[3]["cost"] == 0.0 and narrow.rows[2]["cost"] > 0.0
    wide = _run(states, reg["band@0.05"], boards, chains)                  # 5% is 100: the 99 move is inside it
    assert [r["im_lots"] for r in wide.rows] == pytest.approx([5.0, 5.0, 5.0, 5.0, 0.0])
    assert [r["cost"] for r in wide.rows[1:4]] == [0.0, 0.0, 0.0]
    # IM cost before the final flatten, 1 bp of 200 x 5000 per lot traded: the baseline trades 0.05 + 0.445 + 0.045 lots
    base = _run(states, reg["base"], boards, chains)
    cost = [sum(r["cost"] for r in run.rows[1:4]) for run in (wide, narrow, base)]
    assert cost == pytest.approx([0.0, 49.5, 54.0])


# ---- overlay construction (AC13) on a hand-made board: legs, signs, the expiry payoff across the barrier ----

OV_STRIKES = [4700.0 + 100.0 * i for i in range(7)]                        # 4700 .. 5300, 100 points apart
#: MO2401 ends before the observation of 2024-01-11 (three trading days after 2024-01-08); MO2402 is the first that does not
OV_LASTS = {"MO2401": date(2024, 1, 10), "MO2402": date(2024, 2, 16), "MO2403": date(2024, 3, 15)}
OV = "MO2402"
OV_FORWARD = 4940.0


def _ov_vol(k):
    """A mild skew: 21.2% at 4700, 20% at 5000, 18.8% at 5300."""
    return 0.20 + 0.00004 * (5000.0 - k)


def _ov_board_mo(lasts=None, strikes=None, forward=OV_FORWARD):
    return _ov_mo(T5_DAY, lasts or OV_LASTS, strikes or OV_STRIKES, forward, _ov_vol)


def _mo_ko_ctx(spot, gap_ko, *, n=3, mo=None, **kw):
    """A KO-window day (barrier 5000 unless ``note`` says otherwise, flat vol 20%) with the hand-made MO board."""
    mo = _ov_board_mo() if mo is None else mo
    return _ko_ctx(n=n, spot=spot, gap_ko=gap_ko, board=_t5_board(T5_DAY, spot, mo=mo), **kw)


def _expiry_payoff(legs, s):
    """CNY paid at expiry by ``legs`` (lots per 10 mio, 100 CNY per index point) with the index at ``s``."""
    return sum(lots * 100.0 * (max(s - k, 0.0) if cp == "C" else max(k - s, 0.0)) for (_, cp, k), lots in legs.items())


def _assert_payoff_steps(legs, lo, hi, amount):
    """The expiry payoff is flat below ``lo``, moves by ``amount`` between ``lo`` and ``hi`` monotonically, and is flat
    beyond ``hi`` (the spread pays its full width between its strikes and nothing beyond them)."""
    assert _expiry_payoff(legs, lo - 700.0) == pytest.approx(_expiry_payoff(legs, lo), abs=1e-6)
    assert _expiry_payoff(legs, hi + 700.0) == pytest.approx(_expiry_payoff(legs, hi), abs=1e-6)
    assert _expiry_payoff(legs, hi) - _expiry_payoff(legs, lo) == pytest.approx(amount, rel=1e-12)
    grid = [lo + (hi - lo) * i / 10.0 for i in range(11)]
    steps = [_expiry_payoff(legs, b) - _expiry_payoff(legs, a) for a, b in zip(grid, grid[1:])]
    assert all(x * amount > 0.0 for x in steps)


def _overlay(target, legs, base=1000.0):
    """An overlay target: the legs, both note deltas = d(true), nothing refused."""
    assert dict(target.legs) == {k: pytest.approx(v, rel=1e-12) for k, v in legs.items()}
    assert target.note_delta == pytest.approx(base, rel=1e-12)
    assert target.note_delta_if_filled == pytest.approx(base, rel=1e-12) and tuple(target.flags) == ()


def test_mo_ko_for_a_positive_gap_is_a_spread_rising_across_the_barrier_from_below():
    reg = _registry()
    one, two = reg["mo_ko@1,payoff"], reg["mo_ko@2,payoff"]
    gap = 400000.0                                                          # the seller loses 400 bp on redemption
    # K2 = 5000 (the highest listed strike at or below B = 5000), K1 = 4900; lots = 400000 / (100 points x 100) = 40.
    # spot 4950 between the strikes: long the call spread
    calls = {(OV, "C", 4900.0): 40.0, (OV, "C", 5000.0): -40.0}
    t = one.target(_mo_ko_ctx(4950.0, gap))
    _overlay(t, calls)
    _assert_payoff_steps(t.legs, 4900.0, 5000.0, gap)                      # full width paid by B, nothing beyond it
    assert _expiry_payoff(t.legs, 4900.0) == 0.0 and _expiry_payoff(t.legs, 5000.0) == pytest.approx(gap)
    _overlay(one.target(_mo_ko_ctx(4850.0, gap)), calls)                   # both strikes above spot
    _overlay(one.target(_mo_ko_ctx(5000.0, gap)), calls)                   # spot on K2 is "between them"
    # spot 5050, both strikes below it: short the put spread on the same strikes (short the 5000 put, long the 4900 put)
    puts = {(OV, "P", 5000.0): -40.0, (OV, "P", 4900.0): 40.0}
    t = one.target(_mo_ko_ctx(5050.0, gap))
    _overlay(t, puts)
    _assert_payoff_steps(t.legs, 4900.0, 5000.0, gap)
    assert _expiry_payoff(t.legs, 5000.0) == 0.0 and _expiry_payoff(t.legs, 4900.0) == pytest.approx(-gap)
    # two steps: K1 = 4800, width 200, 20 lots
    t = two.target(_mo_ko_ctx(4950.0, gap))
    _overlay(t, {(OV, "C", 4800.0): 20.0, (OV, "C", 5000.0): -20.0})
    _assert_payoff_steps(t.legs, 4800.0, 5000.0, gap)
    # a barrier between two strikes (5030): K2 is still 5000, so the width is paid before the barrier
    _overlay(one.target(_mo_ko_ctx(4950.0, gap, note=_t5_note(ko=5030.0))), calls)
    # the series: the first whose last trading day is on or after the observation (here the observation day itself)
    mo = _ov_board_mo({"MO2401": date(2024, 1, 11), "MO2402": date(2024, 2, 16)})
    _overlay(one.target(_mo_ko_ctx(4950.0, gap, mo=mo)), {("MO2401", "C", 4900.0): 40.0, ("MO2401", "C", 5000.0): -40.0})
    # the strike step is the listed spacing AT the barrier: 100 between 5000 and 5100, though 50 below 5000
    uneven = [4800.0, 4850.0, 4900.0, 4950.0, 5000.0, 5100.0, 5200.0]
    _overlay(one.target(_mo_ko_ctx(4925.0, gap, mo=_ov_board_mo(strikes=uneven))), calls)
    # ... and 50 for a barrier at 4980: K2 = 4950, K1 = 4900, 80 lots
    t = one.target(_mo_ko_ctx(4925.0, gap, mo=_ov_board_mo(strikes=uneven), note=_t5_note(ko=4980.0)))
    _overlay(t, {(OV, "C", 4900.0): 80.0, (OV, "C", 4950.0): -80.0})
    _assert_payoff_steps(t.legs, 4900.0, 4950.0, gap)


def test_mo_ko_for_a_negative_gap_is_the_mirrored_spread_falling_across_the_barrier_from_above():
    reg = _registry()
    one, two = reg["mo_ko@1,payoff"], reg["mo_ko@2,payoff"]
    gap = -300000.0                                                         # the seller GAINS 300 bp on redemption
    # K1 = 5000 (the lowest listed strike at or above B), K2 = 5100; 30 lots. Spot 4950, both strikes above it: short
    # the call spread
    calls = {(OV, "C", 5000.0): -30.0, (OV, "C", 5100.0): 30.0}
    t = one.target(_mo_ko_ctx(4950.0, gap))
    _overlay(t, calls)
    _assert_payoff_steps(t.legs, 5000.0, 5100.0, gap)                      # nothing below B, the full width lost by K2
    assert _expiry_payoff(t.legs, 5000.0) == 0.0 and _expiry_payoff(t.legs, 5100.0) == pytest.approx(gap)
    _overlay(one.target(_mo_ko_ctx(5000.0, gap)), calls)                   # spot on K1: both strikes at or above spot
    # spot 5050 between the strikes, and 5150 above both: long the put spread (long the 5100 put, short the 5000 put)
    puts = {(OV, "P", 5100.0): 30.0, (OV, "P", 5000.0): -30.0}
    t = one.target(_mo_ko_ctx(5050.0, gap))
    _overlay(t, puts)
    _assert_payoff_steps(t.legs, 5000.0, 5100.0, gap)
    assert _expiry_payoff(t.legs, 5100.0) == 0.0 and _expiry_payoff(t.legs, 5000.0) == pytest.approx(-gap)
    _overlay(one.target(_mo_ko_ctx(5150.0, gap)), puts)
    t = two.target(_mo_ko_ctx(4950.0, gap))
    _overlay(t, {(OV, "C", 5000.0): -15.0, (OV, "C", 5200.0): 15.0})
    _assert_payoff_steps(t.legs, 5000.0, 5200.0, gap)
    # a barrier between two strikes (5030): K1 = 5100, the width is lost only above the barrier
    _overlay(one.target(_mo_ko_ctx(5050.0, gap, note=_t5_note(ko=5030.0))),
             {(OV, "C", 5100.0): -30.0, (OV, "C", 5200.0): 30.0})
    # the day's gap decides the direction, not the observation's fixed ko_dir
    _overlay(one.target(_mo_ko_ctx(4950.0, gap, ko_dir=-1)), calls)
    _overlay(one.target(_mo_ko_ctx(4950.0, gap, ko_dir=0)), calls)
    _overlay(one.target(_mo_ko_ctx(4950.0, 400000.0, ko_dir=1)), {(OV, "C", 4900.0): 40.0, (OV, "C", 5000.0): -40.0})


def test_a_barrier_within_one_listed_step_beyond_the_listing_is_built_and_further_out_is_refused():
    from gpin import techniques as tq

    tech = _registry()["mo_ko@1,payoff"]
    held = {(OV, "P", 4800.0): 7.0}

    def refused(spot, gap, strikes, ko, held_legs=None):
        t = tech.target(_mo_ko_ctx(spot, gap, mo=_ov_board_mo(strikes=strikes), note=_t5_note(ko=ko), held_legs=held_legs))
        assert tuple(t.flags) == ("overlay_no_strike",) and dict(t.legs) == dict(held_legs or {}), (ko, gap)

    top = [4700.0, 4800.0, 4900.0, 5000.0]                                  # nothing listed above 5000; last step 100
    calls = {(OV, "C", 4900.0): 40.0, (OV, "C", 5000.0): -40.0}
    # gap > 0 needs only strikes at or below B. B on the last strike, 60 points above it and exactly one step above it:
    # K2 = 5000, the spacing is the last listed interval's
    for ko in (5000.0, 5060.0, 5100.0):
        _overlay(tech.target(_mo_ko_ctx(4950.0, 400000.0, mo=_ov_board_mo(strikes=top), note=_t5_note(ko=ko))), calls)
    # B more than one listed step above the listing (130 points): no strike near the barrier, refused, what is held stays
    refused(4950.0, 400000.0, top, 5130.0)
    refused(4950.0, 400000.0, top, 5130.0, held)
    # gap < 0 needs K1 at or above B and K2 above it: with B = 5000 on the last strike K2 = 5100 is not listed; with B
    # above the last strike there is no K1
    for ko in (5000.0, 5060.0, 5130.0):
        refused(4950.0, -300000.0, top, ko)
    # mirrored at the bottom of the listing (first step 100): B 40 points below the lowest strike and exactly one step
    # below it are built for gap < 0 (K1 = 5100, the lowest listed strike at or above B)
    bottom = [5100.0, 5200.0, 5300.0]
    puts_or_calls = {(OV, "C", 5100.0): -30.0, (OV, "C", 5200.0): 30.0}
    for ko in (5060.0, 5000.0):
        t = tech.target(_mo_ko_ctx(5080.0, -300000.0, mo=_ov_board_mo(strikes=bottom), note=_t5_note(ko=ko)))
        _overlay(t, puts_or_calls)
    # B more than one listed step below the listing (120 points): refused
    refused(5080.0, -300000.0, bottom, 4980.0)
    refused(5080.0, -300000.0, bottom, 4980.0, held)
    for ko in (5060.0, 4980.0):                                             # gap > 0 has no K2 at or below B there
        refused(5080.0, 400000.0, bottom, ko)
    # the function itself: (cp, K1, K2, sign of the K1 leg)
    assert tq.spread_strikes(1.0, 5060.0, 4950.0, top, 2) == ("C", 4800.0, 5000.0, 1.0)
    assert tq.spread_strikes(1.0, 5100.0, 4950.0, top, 1) == ("C", 4900.0, 5000.0, 1.0)       # one step: inclusive
    assert tq.spread_strikes(1.0, 5100.5, 4950.0, top, 1) is None
    assert tq.spread_strikes(-1.0, 5060.0, 5080.0, bottom, 2) == ("C", 5100.0, 5300.0, -1.0)
    assert tq.spread_strikes(-1.0, 5000.0, 5080.0, bottom, 1) == ("C", 5100.0, 5200.0, -1.0)
    assert tq.spread_strikes(-1.0, 4999.5, 5080.0, bottom, 1) is None
    assert tq.spread_strikes(-1.0, 5060.0, 4950.0, top, 1) is None and tq.spread_strikes(1.0, 5060.0, 5080.0, bottom, 1) is None
    assert tq.spread_strikes(1.0, 5000.0, 4950.0, [5000.0], 1) is None and tq.spread_strikes(1.0, 5000.0, 4950.0, [], 1) is None


def test_mo_ko_holds_legs_only_on_the_last_five_closes_under_the_trigger_and_is_flat_otherwise():
    tech = _registry()["mo_ko@1,payoff"]
    held = {(OV, "C", 4900.0): 40.0, (OV, "C", 5000.0): -40.0}
    _overlay(tech.target(_mo_ko_ctx(4950.0, 400000.0, n=5)), held)
    # one close before the observation of 2024-01-09: MO2401 (ending 2024-01-10) is now the first series that reaches it
    _overlay(tech.target(_mo_ko_ctx(4950.0, 400000.0, n=1)),
             {("MO2401", "C", 4900.0): 40.0, ("MO2401", "C", 5000.0): -40.0})
    _delta_only(tech.target(_mo_ko_ctx(4950.0, 400000.0, n=6, held_legs=held)), 1000.0)           # too early
    _delta_only(tech.target(_mo_ko_ctx(5250.0, 400000.0, n=3, held_legs=held)), 1000.0)           # z = 2.2: no trigger
    _delta_only(tech.target(_mo_ko_ctx(4950.0, 0.0, n=3, held_legs=held)), 1000.0)                # no gap, no overlay
    # the observation day itself (the note survived): the next observation is a month away, the target is flat
    _delta_only(tech.target(_mo_ko_ctx(4950.0, None, n=21, held_legs=held)), 1000.0)


def test_an_overlay_is_refused_and_recorded_and_the_legs_held_stay_when_a_leg_cannot_be_opened():
    tech = _registry()["mo_ko@1,payoff"]
    held = {(OV, "C", 4900.0): 12.0, (OV, "C", 5000.0): -12.0}
    other = {(OV, "P", 5100.0): 7.0}

    def refused(mo, flag, held_legs):
        t = tech.target(_mo_ko_ctx(4950.0, 400000.0, mo=mo, held_legs=held_legs))
        assert tuple(t.flags) == (flag,) and dict(t.legs) == dict(held_legs or {})         # nothing new, what is held stays
        assert t.note_delta == pytest.approx(1000.0)
        assert t.note_delta_if_filled == (pytest.approx(1000.0) if held_legs else None)

    def quotes(mo):
        return mo[OV]["quotes"]

    # no series whose last trading day is on or after the observation
    for held_legs in (None, other):
        refused(_ov_board_mo({"MO2401": date(2024, 1, 10)}), "overlay_no_series", held_legs)
    # a missing strike: K1 = 4900 is not listed at all; only its call is missing; no strike is listed at or below B
    mo = _ov_board_mo(strikes=[4700.0, 4800.0, 5000.0, 5100.0])
    refused(mo, "overlay_no_strike", None)
    refused(mo, "overlay_no_strike", other)
    mo = _ov_board_mo()
    del quotes(mo)[("C", 4900.0)]
    refused(mo, "overlay_no_strike", other)
    refused(_ov_board_mo(strikes=[5100.0, 5200.0, 5300.0]), "overlay_no_strike", other)
    refused(_ov_board_mo(strikes=[5000.0]), "overlay_no_strike", other)    # one strike: no spacing, no K1
    # a leg without an implied vol: the 4900 call settles under its intrinsic value (forward 4940)
    mo = _ov_board_mo()
    quotes(mo)[("C", 4900.0)] = (1.0, 5000, 10000)
    refused(mo, "overlay_no_vol", held)
    # a leg that may not be opened under D8: volume below 100 lots, open interest below 500 lots or unknown
    for volume, oi in [(99, 10000), (5000, 499), (5000, None)]:
        mo = _ov_board_mo()
        quotes(mo)[("C", 5000.0)] = (quotes(mo)[("C", 5000.0)][0], volume, oi)
        refused(mo, "overlay_not_openable", held)
        refused(mo, "overlay_not_openable", None)
    # the thresholds are inclusive
    mo = _ov_board_mo()
    for key in (("C", 4900.0), ("C", 5000.0)):
        quotes(mo)[key] = (quotes(mo)[key][0], 100, 500)
    want = {(OV, "C", 4900.0): 40.0, (OV, "C", 5000.0): -40.0}
    _overlay(tech.target(_mo_ko_ctx(4950.0, 400000.0, mo=mo, held_legs=held)), want)
    # a leg held at or beyond the wanted size is only reduced, which needs a settlement and nothing else
    for key in (("C", 4900.0), ("C", 5000.0)):
        quotes(mo)[key] = (quotes(mo)[key][0], 0, 0)
    big = {(OV, "C", 4900.0): 50.0, (OV, "C", 5000.0): -50.0}
    _overlay(tech.target(_mo_ko_ctx(4950.0, 400000.0, mo=mo, held_legs=big)), want)
    # ... but one leg short of it must be increased, and may not be: everything stays
    lopsided = {(OV, "C", 4900.0): 50.0, (OV, "C", 5000.0): -30.0}
    refused(mo, "overlay_not_openable", lopsided)
    # a leg held on the other side counts as an opening
    refused(mo, "overlay_not_openable", {(OV, "C", 4900.0): -50.0, (OV, "C", 5000.0): -50.0})


def _scenario_lots(gap, level, sd, legs, info, spot, left):
    """Lots for which the spread's value at ``left`` years to its last trading day moves by ``gap`` between ``level x
    (1 - sd)`` and ``level x (1 + sd)``; ``legs``: ``{(cp, strike): sign}``, each at its skew vol of the day, the forward
    moved in proportion to spot, discounted at 1.4%. Written here from the Black-76 formula, not with ``gpin.black``."""
    def value(s):
        forward = info.forward * s / spot
        total = 0.0
        for (cp, k), sign in legs.items():
            if left == 0.0:
                px = max(forward - k, 0.0) if cp == "C" else max(k - forward, 0.0)
            else:
                moved = type(info)(info.last_trading_day, left, forward, math.exp(-0.014 * left))
                px = _bs(moved, k, _ov_vol(k), cp == "C")[0]
            total += sign * px
        return total

    return abs(gap) / (abs(value(level * (1.0 + sd)) - value(level * (1.0 - sd))) * 100.0)


def test_scenario_sizing_matches_a_hand_computed_lot_count_and_the_three_times_cap():
    tech = _registry()["mo_ko@1,scenario"]
    gap, spot = 400000.0, 4950.0
    sd = 0.2 / math.sqrt(244.0)                                            # one daily sigma: 1.28%
    spread = {("C", 4900.0): 1.0, ("C", 5000.0): -1.0}

    def lots_of(lasts):
        t = tech.target(_mo_ko_ctx(spot, gap, mo=_ov_board_mo(lasts)))
        (series,) = {k[0] for k in t.legs}
        assert t.legs[(series, "C", 4900.0)] == pytest.approx(-t.legs[(series, "C", 5000.0)], rel=1e-12)
        assert tuple(t.flags) == () and t.note_delta_if_filled == pytest.approx(1000.0)
        return t.legs[(series, "C", 4900.0)]

    # 1. the series ends on the observation day: the scenario values are payoffs. The forward moves with spot:
    #    B(1 - sd) = 4935.98 -> forward 4926.01, 26.01 points in the money on the 4900 call; B(1 + sd) = 5064.02 ->
    #    forward 5053.79, above both strikes: the full 100. lots = 400000 / ((100 - 26.01) x 100) = 54.06
    f_dn = OV_FORWARD * 5000.0 * (1.0 - sd) / spot
    f_up = OV_FORWARD * 5000.0 * (1.0 + sd) / spot
    assert f_dn == pytest.approx(4926.01, abs=0.01) and f_up > 5000.0
    by_hand = gap / ((100.0 - (f_dn - 4900.0)) * 100.0)
    assert by_hand == pytest.approx(54.06, abs=0.01)
    assert lots_of({"MO2401": date(2024, 1, 11)}) == pytest.approx(by_hand, rel=1e-9)
    # 2. the series ends four days after the observation: Black-76 values with 4 days left at each leg's vol of today
    last = date(2024, 1, 15)
    expect = _scenario_lots(gap, 5000.0, sd, spread, _series_info(T5_DAY, last, OV_FORWARD), spot, 4 / 365.0)
    assert expect == pytest.approx(96.5, abs=0.1)                           # more than the payoff's 40 lots, under the cap
    assert lots_of({"MO2401": last}) == pytest.approx(expect, rel=1e-6)
    # 3. a series ending 36 days after the observation moves too little (254 lots): capped at 3 x the payoff lots
    last = date(2024, 2, 16)
    assert _scenario_lots(gap, 5000.0, sd, spread, _series_info(T5_DAY, last, OV_FORWARD), spot, 36 / 365.0) > 120.0
    assert lots_of({"MO2402": last}) == pytest.approx(120.0, rel=1e-12)
    # the mirrored spread of a negative gap is sized the same way
    last = date(2024, 1, 15)
    t = tech.target(_mo_ko_ctx(spot, -300000.0, mo=_ov_board_mo({"MO2401": last})))
    mirrored = {("C", 5000.0): -1.0, ("C", 5100.0): 1.0}
    expect = _scenario_lots(300000.0, 5000.0, sd, mirrored, _series_info(T5_DAY, last, OV_FORWARD), spot, 4 / 365.0)
    assert expect == pytest.approx(76.6, abs=0.1)                           # under the cap of 3 x 30
    assert dict(t.legs) == {("MO2401", "C", 5000.0): pytest.approx(-expect, rel=1e-6),
                            ("MO2401", "C", 5100.0): pytest.approx(expect, rel=1e-6)}
    assert t.note_delta_if_filled == pytest.approx(1000.0) and tuple(t.flags) == ()


# ---- the knock-in overlay: the same construction across K, the gap between the not-knocked-in and the knocked-in note ----

KI_STRIKES = [3500.0 + 50.0 * i for i in range(11)]                        # 3500 .. 4000, 50 points apart
KI_LASTS = {"MO2401": date(2024, 1, 19), "MO2402": date(2024, 2, 16)}
KI_FORWARD = 3890.0


def _mo_ki_ctx(spot=3900.0, *, lasts=None, mo=None, knocked_in=False, deltas=None, **kw):
    """A KI-window day (KI barrier 3750). The ``ki_m200`` ladder is 100,000 at spot with slope 1500 and its -4% point
    lifted by 39,000; the ``kin`` ladder is -300,000 at spot with slope 1200. At K = 3750 (150 below the spot 3900,
    between the ladder points 3744 and 3783): ki_m200 = 100000 - 1500 x 150 + 39000 x 33 / 39 = -92000, kin = -300000
    - 1200 x 150 = -480000: the gap is 388,000."""
    lifted = _ladder_state(100000.0, 1500.0, 0.0, spot)
    lifted["engine_price"][8] += 39000.0
    rec = _t5_record(T5_DAY, spot, KI_DELTAS if deltas is None else deltas, n=40, knocked_in=knocked_in,
                     prices={"kin": -300000.0}, ladders=None if deltas is not None else {"ki_m200": lifted})
    mo = _ov_mo(T5_DAY, lasts or KI_LASTS, KI_STRIKES, KI_FORWARD, _ov_vol) if mo is None else mo
    return _t5_ctx(rec, board=_t5_board(T5_DAY, spot, mo=mo), **kw)


def test_mo_ki_is_the_rising_spread_below_the_knock_in_barrier_sized_to_the_knock_in_gap():
    reg = _registry()
    tech = reg["mo_ki@2,payoff"]
    gap = 388000.0
    # K2 = 3750 (the highest listed strike at or below K), two steps of 50 down: K1 = 3650; both below spot: short the
    # put spread; lots = 388000 / (100 points x 100) = 38.8
    legs = {("MO2401", "P", 3750.0): -38.8, ("MO2401", "P", 3650.0): 38.8}
    t = tech.target(_mo_ki_ctx())
    _overlay(t, legs, base=2000.0)
    _assert_payoff_steps(t.legs, 3650.0, 3750.0, gap)                      # the seller gives the knock-in gain back below K
    assert _expiry_payoff(t.legs, 3750.0) == 0.0 and _expiry_payoff(t.legs, 3650.0) == pytest.approx(-gap)
    # the hedge it also follows is the baseline's (the bump delta of the unlifted true ladder)
    assert t.note_delta == pytest.approx(2000.0)
    # the front series needs three trading days left (2024-01-09, -10, -11): with two it is rolled to the next series
    _overlay(tech.target(_mo_ki_ctx(lasts={"MO2401": date(2024, 1, 11), "MO2402": date(2024, 2, 16)})), legs, base=2000.0)
    rolled = {("MO2402", "P", 3750.0): -38.8, ("MO2402", "P", 3650.0): 38.8}
    t = tech.target(_mo_ki_ctx(lasts={"MO2401": date(2024, 1, 10), "MO2402": date(2024, 2, 16)}, held_legs=legs))
    _overlay(t, rolled, base=2000.0)                                       # the old series' legs are not wanted any more
    # flat on a knock-in and on leaving the window (5.3% above the barrier), whatever is held
    _delta_only(tech.target(_mo_ki_ctx(spot=3700.0, knocked_in=True, held_legs=legs)), 2000.0)
    _delta_only(tech.target(_mo_ki_ctx(spot=3950.0, held_legs=legs)), 2000.0)
    # refused and recorded, the legs held stay: no series with three trading days left; the rolled series not listed at K1
    t = tech.target(_mo_ki_ctx(lasts={"MO2401": date(2024, 1, 10)}, held_legs=legs))
    assert tuple(t.flags) == ("overlay_no_series",) and dict(t.legs) == legs and t.note_delta_if_filled == pytest.approx(2000.0)
    mo = _ov_mo(T5_DAY, {"MO2401": date(2024, 1, 10), "MO2402": date(2024, 2, 16)}, KI_STRIKES, KI_FORWARD, _ov_vol)
    del mo["MO2402"]["quotes"][("P", 3650.0)]
    t = tech.target(_mo_ki_ctx(mo=mo, held_legs=legs))
    assert tuple(t.flags) == ("overlay_no_strike",) and dict(t.legs) == legs


def test_mo_ki_scenario_sizing_values_the_spread_at_the_next_daily_observation():
    tech = _registry()["mo_ki@2,scenario"]
    sd = 0.2 / math.sqrt(244.0)
    spread = {("P", 3750.0): -1.0, ("P", 3650.0): 1.0}
    # the next knock-in observation is the next trading day, 2024-01-09. A front series ending 2024-01-12 (four trading
    # days left today) has 3 days left then
    last = date(2024, 1, 12)
    info = _series_info(T5_DAY, last, KI_FORWARD)
    expect = _scenario_lots(388000.0, 3750.0, sd, spread, info, 3900.0, 3 / 365.0)
    assert expect == pytest.approx(105.2, abs=0.1)                          # between the payoff's 38.8 lots and the cap 116.4
    t = tech.target(_mo_ki_ctx(lasts={"MO2401": last, "MO2402": date(2024, 2, 16)}))
    assert dict(t.legs) == {("MO2401", "P", 3750.0): pytest.approx(-expect, rel=1e-6),
                            ("MO2401", "P", 3650.0): pytest.approx(expect, rel=1e-6)}
    assert t.note_delta_if_filled == pytest.approx(2000.0) and tuple(t.flags) == ()
    # with 10 days left then (MO2401 ending 2024-01-19) the spread moves too little (169 lots): capped at 3 x 38.8
    info = _series_info(T5_DAY, date(2024, 1, 19), KI_FORWARD)
    assert _scenario_lots(388000.0, 3750.0, sd, spread, info, 3900.0, 10 / 365.0) > 3.0 * 38.8
    _overlay(tech.target(_mo_ki_ctx()), {("MO2401", "P", 3750.0): -116.4, ("MO2401", "P", 3650.0): 116.4}, base=2000.0)


def test_the_knock_in_gap_is_interpolated_on_the_ladder_at_the_barrier():
    from gpin import techniques as tq

    # ki_m200 minus kin, each read at K = 3750 by linear interpolation between the ladder points 3744 and 3783
    assert tq.ki_gap(_mo_ki_ctx()) == pytest.approx(388000.0, rel=1e-12)
    # straight ladders: (0 - 1500 x 150) - (-300000 - 1200 x 150) = 255000
    assert tq.ki_gap(_mo_ki_ctx(deltas=dict(KI_DELTAS))) == pytest.approx(255000.0, rel=1e-12)
    # the sign is the ladders': both at 0 at spot gives -1500 x 150 + 1200 x 150 = -45000
    assert tq.ki_gap(_ki_ctx()) == pytest.approx(-45000.0, rel=1e-12)
    assert tq.ki_gap(_t5_ctx(_t5_record(T5_DAY, 3900.0, {"true": 2000.0}, n=40))) is None      # no KI state
    with pytest.raises(StudyDataError):                                     # a barrier off the ladder (15% above spot)
        tq.ki_gap(_ki_ctx(note=_t5_note(ki=4485.0)))


# ---- composites: the futures rule by window, the overlay on top of it ----

def test_sop_futures_applies_the_ko_member_in_the_ko_window_and_the_ki_member_in_the_ki_window():
    sop = _registry()["sop_futures"]                                        # derisk@5 at KO, ki_cap@1.25 at KI
    _delta_only(sop.target(_ko_ctx(n=3, spot=5020.0)), 725.0)              # derisk@5 at n = 3
    _delta_only(sop.target(_ko_ctx(n=8, spot=5020.0)), 1000.0)             # derisk@5 before n_d: the baseline
    _delta_only(sop.target(_ki_ctx()), 1500.0)                             # min(2000, 1.25 x 1200)
    _delta_only(sop.target(_far_ctx()), 1000.0)
    # both windows active (a note with its KI barrier at 4800, spot 5020): the KO rule applies, the KI rule does not
    both = dict(KO_DELTAS, kin=600.0, ki_m100=900.0, ki_m200=800.0, ki_p100=1200.0)
    note = _t5_note(ki=4800.0)
    reg = _registry()
    ctx = _ko_ctx(n=3, spot=5020.0, deltas=both, note=note)
    _delta_only(reg["ki_cap@1.25"].target(ctx), 750.0)                     # what the KI rule alone would do
    _delta_only(sop.target(ctx), 725.0)
    ctx = _ko_ctx(n=8, spot=5020.0, deltas=both, note=note)
    _delta_only(sop.target(ctx), 1000.0)                                   # the KO rule's baseline, not the KI cap
    # the secant with one bump per barrier type
    sec = _registry({"sop_futures": {"ko": "secant@0.03", "ki": "secant@0.05"},
                     "sop_overlay": T5_SELECTION["sop_overlay"]})["sop_futures"]
    lifted = _ladder_state(0.0, 1000.0, 0.0, 5100.0)
    lifted["engine_price"][15] += 30000.0                                   # the +3% point
    lifted["engine_price"][17] += 51000.0                                   # the +5% point
    rec = _t5_record(T5_DAY, 5100.0, KO_DELTAS, n=8, gap_ko=1.0, ko_dir=-1, ladders={"true": lifted})
    _delta_only(sec.target(_t5_ctx(rec)), 1000.0 + 30000.0 / (2 * 0.03 * 5100.0))
    lifted = _ladder_state(0.0, 2000.0, 0.0, 3900.0)
    lifted["engine_price"][15] += 30000.0
    lifted["engine_price"][17] -= 39000.0
    rec = _t5_record(T5_DAY, 3900.0, KI_DELTAS, n=40, ladders={"true": lifted})
    _delta_only(sec.target(_t5_ctx(rec)), 2000.0 - 39000.0 / (2 * 0.05 * 3900.0))
    # a member that holds the hedge carried
    frz = _registry({"sop_futures": {"ko": "freeze@1", "ki": "ki_shift@-0.01"},
                     "sop_overlay": T5_SELECTION["sop_overlay"]})["sop_futures"]
    board = _t5_board(T5_DAY, 5050.0, im={"IM2403": 5030.0})
    _delta_only(frz.target(_ko_ctx(n=1, spot=5050.0, board=board, held_im={"IM2403": 5.2})), 5.2 * 200.0 * 5030.0 / 5050.0)
    _delta_only(frz.target(_ki_ctx()), 1700.0)


def test_sop_overlay_takes_the_futures_rule_as_note_delta_and_the_true_delta_for_the_filled_overlay():
    reg = _registry()                                                       # mo_ko@1,scenario and mo_ki@2,payoff
    sop = reg["sop_overlay"]
    # KO trigger day, n = 3: the futures rule is derisk@5 (725), the legs are mo_ko@1,scenario's, the filled part is
    # hedged at d(true) = 1000
    ctx = _mo_ko_ctx(4950.0, 400000.0, mo=_ov_board_mo({"MO2401": date(2024, 1, 11)}))
    t = sop.target(ctx)
    assert t.note_delta == pytest.approx(725.0, rel=1e-12) and t.note_delta_if_filled == pytest.approx(1000.0, rel=1e-12)
    assert dict(t.legs) == dict(reg["mo_ko@1,scenario"].target(ctx).legs) and len(t.legs) == 2 and tuple(t.flags) == ()
    assert t.legs[("MO2401", "C", 4900.0)] == pytest.approx(54.06, abs=0.01)
    # KI window day: ki_cap@1.25 (1500), mo_ki@2,payoff's legs, d(true) = 2000
    t = sop.target(_mo_ki_ctx())
    assert t.note_delta == pytest.approx(1500.0, rel=1e-12) and t.note_delta_if_filled == pytest.approx(2000.0, rel=1e-12)
    assert dict(t.legs) == {("MO2401", "P", 3750.0): pytest.approx(-38.8), ("MO2401", "P", 3650.0): pytest.approx(38.8)}
    # no overlay wanted (KO window, six closes ahead): nothing is covered, so the futures rule is the whole target
    shift = _registry({"sop_futures": {"ko": "ko_shift@0.02", "ki": "ki_cap@1.25"},
                       "sop_overlay": T5_SELECTION["sop_overlay"]})["sop_overlay"]
    _delta_only(shift.target(_mo_ko_ctx(4950.0, 400000.0, n=6)), 700.0)
    t = shift.target(_mo_ko_ctx(4950.0, 400000.0, n=5))
    assert t.note_delta == pytest.approx(700.0) and t.note_delta_if_filled == pytest.approx(1000.0) and len(t.legs) == 2
    # a refused overlay: recorded, the legs held stay and are hedged at d(true); with none held the futures rule is all
    held = {(OV, "C", 4900.0): 12.0, (OV, "C", 5000.0): -12.0}
    mo = _ov_board_mo(strikes=[4700.0, 4800.0, 5000.0, 5100.0])
    t = shift.target(_mo_ko_ctx(4950.0, 400000.0, mo=mo, held_legs=held))
    assert tuple(t.flags) == ("overlay_no_strike",) and dict(t.legs) == held
    assert t.note_delta == pytest.approx(700.0) and t.note_delta_if_filled == pytest.approx(1000.0)
    t = shift.target(_mo_ko_ctx(4950.0, 400000.0, mo=mo))
    assert tuple(t.flags) == ("overlay_no_strike",) and dict(t.legs) == {} and t.note_delta_if_filled is None
    assert t.note_delta == pytest.approx(700.0)


# ---- the techniques through the ledger: refusals in the row, no look-ahead for every registered id (AC8) ----

AC8_SPOTS = {
    "ko": [5000.0, 4980.0, 5020.0, 5040.0, 4990.0, 4960.0, 5010.0, 4985.0, 4990.0, 4970.0, 4950.0, 4940.0],
    "ki": [4100.0, 3980.0, 3930.0, 3900.0, 3880.0, 3920.0, 3950.0, 3890.0, 3800.0, 3760.0, 3740.0, 3700.0],
}
#: the day's gap_ko of the KO world's first eight days (the observation is day 8); the sign flips on day 6
AC8_GAPS = [400e3, 410e3, 395e3, 405e3, 400e3, 380e3, -60e3, 390e3]
AC8_KO_LASTS = {"MO2401": date(2024, 1, 19), "MO2402": date(2024, 2, 16)}


def _ac8_specs(kind, alt_from=None, drop=None):
    """Twelve day specs of a hand-made note (strike and KO level 5000, KI level 3750).

    ``ko``: the note sits at its KO barrier, the observation is day 8 (survived), every KO model is recorded on days
    0..7 and only the true model after. ``ki``: the note falls into its KI window on day 2, leaves it on day 6, comes
    back and knocks in on day 10; the KI models are recorded on the window days. From day ``alt_from`` on every number
    is different (spot, flat vol, every ladder, the gap and its sign, IM closes, option settlements and volumes).
    ``drop``: ``(day index, (cp, strike))`` of an option taken off the front series' board that day."""
    days, specs = T5_DAYS, []
    for i, day in enumerate(days):
        alt = alt_from is not None and i >= alt_from
        spot = AC8_SPOTS[kind][i] * (1.012 if alt else 1.0)
        scale = (1.0 + 0.01 * i) * (1.6 if alt else 1.0)
        sp = dict(day=day, spot=spot, vol=0.31 if alt else 0.2, im={"IM2403": spot - (53.0 if alt else 20.0)})
        opt_vol = 0.26 if alt else 0.2
        volume = 700 if alt else 5000
        if kind == "ko":
            in_window = i < 8
            sp.update(n=8 - i if in_window else 28 - i, ko_dir=-1 if in_window else None,
                      gap_ko=(AC8_GAPS[i] * (-1.3 if alt else 1.0)) if in_window else None,
                      deltas={m: d * scale for m, d in KO_DELTAS.items()} if in_window else {"true": 1000.0 * scale},
                      mo=_ov_mo(day, AC8_KO_LASTS, OV_STRIKES, spot - 10.0, lambda k: opt_vol, volume))
        else:
            in_window = i in (2, 3, 4, 5, 7, 8, 9)
            sp.update(n=40 - i, knocked_in=i >= 10,
                      deltas={m: d * scale for m, d in KI_DELTAS.items()} if in_window else {"true": 2000.0 * scale},
                      prices={"ki_m200": 100000.0 * scale, "kin": -300000.0 * scale},
                      mo=_ov_mo(day, KI_LASTS, KI_STRIKES, spot - 8.0, lambda k: opt_vol, volume))
        if drop is not None and drop[0] == i:
            del sp["mo"]["MO2401"]["quotes"][drop[1]]
        specs.append(sp)
    return specs


class _Recording:
    """A technique wrapped to record every target it returns."""

    def __init__(self, tech):
        self.tech, self.id, self.family, self.params, self.clip = tech, tech.id, tech.family, tech.params, tech.clip
        self.seen = []

    def target(self, ctx):
        t = self.tech.target(ctx)
        self.seen.append((ctx.day, t.note_delta, dict(t.legs), t.note_delta_if_filled, tuple(t.flags)))
        return t


@pytest.fixture(scope="module")
def ac8_worlds():
    """``{(kind, alt_from): (states, boards, chains)}``: the two worlds and their variants altered from a day on."""
    out = {}
    for kind in ("ko", "ki"):
        for alt_from in (None, 3, 5, 7, 9, 11):
            out[(kind, alt_from)] = _t5_world(_ac8_specs(kind, alt_from))
    return out


def _recorded(tid, world):
    rec = _Recording(_registry()[tid])
    run = _run(*([world[0], rec] + list(world[1:])))
    return rec.seen, run.rows


@pytest.mark.parametrize("tid", T5_ALL_IDS)
def test_no_look_ahead_for_every_registered_technique(tid, ac8_worlds):
    acted = False
    for kind in ("ko", "ki"):
        seen, rows = _recorded(tid, ac8_worlds[(kind, None)])
        base_seen, _ = _recorded("base", ac8_worlds[(kind, None)])
        assert [s[0] for s in seen] == T5_DAYS[:11]                        # a decision on every day but the last (flattened)
        acted = acted or seen != base_seen
        for alt_from in (3, 5, 7, 9, 11):
            alt_seen, alt_rows = _recorded(tid, ac8_worlds[(kind, alt_from)])
            # everything dated from ``alt_from`` on differs; the targets and the rows before it do not
            assert alt_seen[:alt_from] == seen[:alt_from] and alt_rows[:alt_from] == rows[:alt_from], (kind, alt_from)
            assert alt_seen[alt_from:] != seen[alt_from:] or alt_from == 11
            assert alt_rows[alt_from:] != rows[alt_from:]
    assert acted or tid in ("base", "base_noclip")                          # the worlds do exercise the technique


def test_the_ac8_worlds_exercise_the_overlays_with_cut_fills_and_both_gap_signs(ac8_worlds):
    seen, rows = _recorded("mo_ko@1,payoff", ac8_worlds[("ko", None)])
    legs = [s[2] for s in seen]
    assert legs[2] == {} and legs[8] == {}                                  # six closes ahead; the observation day
    # day 3, spot 5040 above both strikes, gap 405,000: the short put spread, 40.5 lots
    assert legs[3] == {("MO2401", "P", 5000.0): pytest.approx(-40.5), ("MO2401", "P", 4900.0): pytest.approx(40.5)}
    # day 4, spot 4990 between the strikes: the call spread
    assert legs[4] == {("MO2401", "C", 4900.0): pytest.approx(40.0), ("MO2401", "C", 5000.0): pytest.approx(-40.0)}
    # day 6, the gap is -60,000 and spot 5010 is between K1 = 5000 and K2 = 5100: the long put spread, 6 lots
    assert legs[6] == {("MO2401", "P", 5100.0): pytest.approx(6.0), ("MO2401", "P", 5000.0): pytest.approx(-6.0)}
    # at the 500 mio desk a fill is cut to a tenth of the day's 5000 lots (10 lots per 10 mio): phi = 10 / 40.5
    assert rows[3]["phi"] == pytest.approx(10.0 / 40.5) and "volume_cut" in rows[3]["flags"]
    assert rows[3]["legs"] == {"MO2401:P:5000": pytest.approx(-10.0), "MO2401:P:4900": pytest.approx(10.0)}
    seen, rows = _recorded("mo_ki@2,payoff", ac8_worlds[("ki", None)])
    legs = [s[2] for s in seen]
    assert [bool(x) for x in legs] == [False, False, True, True, True, True, False, True, True, True, False]
    assert set(legs[3]) == {("MO2401", "P", 3750.0), ("MO2401", "P", 3650.0)}
    assert rows[10]["legs"] != {} and rows[11]["legs"] == {}                # a cut close stays on the book after the knock-in


def test_a_refused_overlay_is_recorded_in_the_ledger_row_and_the_legs_held_stay():
    tech = _registry()["mo_ko@1,payoff"]
    # day 3 (spot 5040) wants the short put spread, but the 4900 put is not on the board: refused, recorded, nothing asked
    states, boards, chains = _t5_world(_ac8_specs("ko", drop=(3, ("P", 4900.0))))
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_)
    assert "overlay_no_strike" in run.rows[3]["flags"] and run.rows[3]["legs"] == {}
    assert run.rows[3]["unfilled"] == {} and run.rows[3]["phi"] == 1.0
    assert "overlay_no_strike" not in run.rows[4]["flags"] and len(run.rows[4]["legs"]) == 2      # the next day it is bought
    assert all("overlay_no_strike" not in r["flags"] for i, r in enumerate(run.rows) if i != 3)
    # legs held stay through a refusal. Days 4..6 at spot 4990 want the 4900 / 5000 call spread (40, 38, 38 lots: bought,
    # then reduced); day 7 wants 39, an increase, and the 5000 call trades only 50 lots that day
    specs = _ac8_specs("ko")
    for i in (5, 6):
        specs[i]["spot"], specs[i]["gap_ko"], specs[i]["im"] = specs[4]["spot"], 380e3, dict(specs[4]["im"])
        specs[i]["mo"] = _ov_mo(specs[i]["day"], AC8_KO_LASTS, OV_STRIKES, specs[4]["spot"] - 10.0)
    q = specs[7]["mo"]["MO2401"]["quotes"]
    q[("C", 5000.0)] = (q[("C", 5000.0)][0], 50, 10000)
    states, boards, chains = _t5_world(specs)
    run = _run(states, tech, boards, chains, desk_notional=NOTIONAL_)
    spread = {"MO2401:C:4900": pytest.approx(38.0), "MO2401:C:5000": pytest.approx(-38.0)}
    assert run.rows[6]["legs"] == spread and run.rows[6]["flags"] == []
    assert "overlay_not_openable" in run.rows[7]["flags"] and run.rows[7]["legs"] == spread
    assert run.rows[7]["unfilled"] == {} and run.rows[7]["phi"] == 1.0
    assert run.rows[8]["legs"] == {} and "overlay_not_openable" not in run.rows[8]["flags"]     # flat after the observation
    _check_identity(run)


def test_the_no_look_ahead_check_catches_a_technique_that_reads_tomorrow(ac8_worlds):
    """A negative control: a technique handed the whole recorded note (which no registered technique is) and hedging
    with TOMORROW's delta fails the comparison the AC8 test makes."""
    from gpin.techniques import Target, hedge_delta

    class Cheat:
        id, family, params, clip = "cheat", "test", {}, True

        def __init__(self, states):
            self.days = states["days"]

        def target(self, ctx):
            nxt = self.days[min(ctx.day_index + 1, len(self.days) - 1)]
            return Target(hedge_delta(nxt["states"]["true"], nxt["spot"]))

    def seen(world):
        rec = _Recording(Cheat(world[0]))
        _run(world[0], rec, *world[1:])
        return rec.seen

    whole, altered = seen(ac8_worlds[("ko", None)]), seen(ac8_worlds[("ko", 5)])
    assert whole[:4] == altered[:4] and whole[:5] != altered[:5]           # day 4's target already carries day 5


def test_trading_day_helpers_follow_the_sse_calendar():
    from gpin import techniques as tq

    assert tq.trading_days_between(date(2024, 1, 8), date(2024, 1, 11)) == 3       # the 9th, 10th and 11th
    assert tq.trading_days_between(date(2024, 1, 8), date(2024, 1, 8)) == 0
    assert tq.trading_days_between(date(2023, 12, 28), date(2024, 1, 3)) == 3      # across the year end: New Year's Day is closed
    assert tq.trading_days_between(date(2024, 2, 8), date(2024, 2, 19)) == 1       # the Spring Festival week is closed
    assert tq.next_trading_day(date(2024, 1, 5)) == date(2024, 1, 8)
    assert tq.next_trading_day(date(2023, 12, 29)) == date(2024, 1, 2)
    assert tq.next_trading_day(date(2024, 2, 8)) == date(2024, 2, 19)


def test_target_flags_default_to_none_and_reach_the_ledger_row_on_the_decision_day():
    from gpin.techniques import Target

    assert Target(1.0).flags == () and Target(1.0, {}, None, ("overlay_no_strike",)).flags == ("overlay_no_strike",)

    class Flagging(_Scripted):
        def target(self, ctx):
            return Target(1000.0, {}, None, ("overlay_no_vol", "overlay_no_vol") if ctx.day_index == 1 else ())

    states, boards, chains = _world_a()
    run = _run(states, Flagging(lambda ctx: {}), boards, chains)
    assert [r["flags"] for r in run.rows[:2]] == [[], ["overlay_no_vol"]]          # written once, on its day


def test_windows_and_overlays_use_the_notes_levels_not_its_booking_spot():
    """A seasoned note booked at 6420: KI level 6000, strike and KO level 8000 (the state file's levels)."""
    reg = _registry()
    note = _t5_note(book="seas6", s0=6420.0, strike=8000.0, ko=8000.0, ki=6000.0)
    # at the booking spot, three closes before an observation: 20% below the KO level and 7% above the KI level. Neither
    # window is open, whatever states the record carries
    every = {**KI_DELTAS, **KO_DELTAS}                                      # the true delta is the KO fixture's 1000
    ctx = _t5_ctx(_t5_record(T5_DAY, 6420.0, every, n=3, gap_ko=4e5, ko_dir=-1), note=note)
    for tid in ("ko_shift@0.02", "surv_hold@3", "derisk@5", "ki_shift@-0.01", "ki_cap@1.0", "secant@0.03", "sop_futures"):
        _delta_only(reg[tid].target(ctx), 1000.0)
    # 4% above the KI level 6000: the KI rules act
    ctx = _t5_ctx(_t5_record(T5_DAY, 6240.0, KI_DELTAS, n=3), note=note)
    _delta_only(reg["ki_shift@-0.01"].target(ctx), 1700.0)
    _delta_only(reg["ki_cap@1.0"].target(ctx), 1200.0)
    # 1% below the KO level 8000: the KO rules act, and the overlay is struck around 8000
    strikes = [7700.0 + 100.0 * i for i in range(7)]
    mo = _ov_mo(T5_DAY, {"MO2402": date(2024, 2, 16)}, strikes, 7910.0)
    ctx = _t5_ctx(_t5_record(T5_DAY, 7920.0, KO_DELTAS, n=3, gap_ko=6.4e6, ko_dir=-1), note=note,
                  board=_t5_board(T5_DAY, 7920.0, mo=mo))
    _delta_only(reg["ko_shift@0.02"].target(ctx), 700.0)
    _delta_only(reg["surv_hold@3"].target(ctx), 450.0)
    # a knocked-in seasoned note back at its KO level: the whole accrued coupon is the digital, 6.4 mio / (100 x 100)
    _overlay(reg["mo_ko@1,payoff"].target(ctx), {("MO2402", "C", 7900.0): 640.0, ("MO2402", "C", 8000.0): -640.0})


# ---------------------------------------------------------------------------
# T6 — measures, training selection, verdict, gamma carry, the ledger driver
# ---------------------------------------------------------------------------

# ---- ES10 and the week-clustered bootstrap (D10, AC11) ----


def test_es_is_the_mean_of_the_k_smallest_values_with_k_the_ceiling_of_a_tenth():
    from gpin.measures import es

    seven = [3.0, -8.0, 1.0, 12.0, -2.0, 5.0, 0.5]
    assert es(seven, 0.10) == -8.0                                           # n = 7: k = max(1, ceil(0.7)) = 1
    twenty_five = [float(v) for v in range(25, 0, -1)]                       # 25 .. 1
    assert es(twenty_five, 0.10) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0, rel=1e-15)    # k = ceil(2.5) = 3
    thirty = [float(v) for v in range(30)]
    assert es(thirty, 0.10) == pytest.approx(1.0, rel=1e-15)                 # k = 3 exactly, not ceil(3.0000000000000004)
    assert es([4.0], 0.10) == 4.0
    assert es([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(1.5, rel=1e-15)
    assert es(seven) == -8.0                                                 # the pre-registered alpha is the default
    with pytest.raises(ValidationError):
        es([], 0.10)
    with pytest.raises(ValidationError):
        es([1.0, float("nan")], 0.10)


def _reference_bootstrap(tech, base, weeks, n, seed, alpha=0.10):
    """The bootstrap written as a loop over replicates, with the documented draw: clusters in sorted order, replicate
    ``i`` takes the clusters ``default_rng(seed).integers(0, C, size=(n, C))[i]`` with their multiplicity."""
    from gpin.measures import es

    clusters = sorted(set(weeks.values()))
    members = {c: [e for e in sorted(tech) if weeks[e] == c] for c in clusters}
    draws = np.random.default_rng(seed).integers(0, len(clusters), size=(n, len(clusters)))
    hits = 0
    for row in draws:
        picked = [e for c in row for e in members[clusters[c]]]
        hits += es([tech[e] for e in picked], alpha) - es([base[e] for e in picked], alpha) <= 0.0
    return hits / n


def test_cluster_bootstrap_identical_arms_give_one_and_a_dominating_arm_gives_zero():
    from gpin.measures import cluster_bootstrap

    rng = np.random.default_rng(7)
    base = {f"e{i}": float(v) for i, v in enumerate(rng.normal(0.0, 50.0, 37))}
    weeks = {e: f"2025-W{30 + i % 11:02d}" for i, e in enumerate(base)}
    assert cluster_bootstrap(base, dict(base), weeks, n=500) == 1.0          # the difference is 0 on every replicate
    better = {e: v + 5.0 for e, v in base.items()}                           # ES10 is 5 higher on every resample
    assert cluster_bootstrap(better, base, weeks, n=500) == 0.0
    assert cluster_bootstrap(base, better, weeks, n=500) == 1.0


def test_cluster_bootstrap_equals_a_replicate_loop_and_a_two_cluster_case_worked_by_hand():
    from gpin.measures import cluster_bootstrap

    rng = np.random.default_rng(11)
    base = {f"e{i:02d}": float(v) for i, v in enumerate(rng.normal(0.0, 40.0, 29))}
    tech = {e: v + float(d) for (e, v), d in zip(base.items(), rng.normal(3.0, 30.0, 29))}
    sizes = [1, 1, 2, 3, 5, 8, 9]                                            # unequal clusters: 29 events in 7 weeks
    weeks = {}
    for (e, _), w in zip(base.items(), [w for w, size in enumerate(sizes) for _ in range(size)]):
        weeks[e] = f"2025-W{40 + w:02d}"
    p = cluster_bootstrap(tech, base, weeks, n=400, seed=5)
    assert p == _reference_bootstrap(tech, base, weeks, 400, 5)
    assert 0.0 < p < 1.0
    assert cluster_bootstrap(tech, base, weeks, n=400, seed=5) == p           # fixed seed: the same number again
    # two one-event weeks; ES10 of two events is their minimum. Resamples (a, a), (a, b), (b, a), (b, b): the technique's
    # minimum minus the baseline's is -1, +2, +2, +2: p is the share of (a, a) draws, a quarter in expectation
    tech2, base2, weeks2 = {"a": 9.0, "b": 12.0}, {"a": 10.0, "b": 7.0}, {"a": "2025-W30", "b": "2025-W31"}
    draws = np.random.default_rng(20261112).integers(0, 2, size=(10_000, 2))
    expected = float(np.mean((draws == 0).all(axis=1)))
    p2 = cluster_bootstrap(tech2, base2, weeks2)                             # n = 10,000 and seed 20261112 by default
    assert p2 == expected and abs(p2 - 0.25) < 0.02


def test_cluster_bootstrap_counts_a_difference_of_rounding_noise_as_no_gain():
    """A technique that never acts on the events equals the baseline to rounding (its rows are summed in another
    order): every replicate is a tie, so p is 1, not the share of replicates on which the noise happens to be positive."""
    from gpin.measures import cluster_bootstrap

    rng = np.random.default_rng(3)
    base = {f"e{i}": float(v) for i, v in enumerate(rng.normal(0.0, 50.0, 40))}
    weeks = {e: f"2025-W{30 + i % 13:02d}" for i, e in enumerate(base)}
    noisy = {e: v * (1.0 + 1.0e-13 * (-1.0) ** i) for i, (e, v) in enumerate(base.items())}
    assert noisy != base
    assert cluster_bootstrap(noisy, base, weeks, n=500) == 1.0
    assert cluster_bootstrap(base, noisy, weeks, n=500) == 1.0


def test_cluster_bootstrap_fails_closed_on_unpaired_arms_or_a_missing_week():
    from gpin.measures import cluster_bootstrap

    with pytest.raises(ValidationError):
        cluster_bootstrap({"a": 1.0}, {"b": 1.0}, {"a": "w", "b": "w"})
    with pytest.raises(ValidationError):
        cluster_bootstrap({"a": 1.0}, {"a": 1.0}, {})
    with pytest.raises(ValidationError):
        cluster_bootstrap({}, {}, {})


# ---- events, event windows, whole-life statistics, gamma carry (D10, AC11) on hand-made rows ----

T6_CAL = _weekdays(40, date(2025, 5, 5))          # 2025-05-05 (a Monday) .. 2025-06-27: the measures' trading calendar
T6_NOTE = ("ki75", T6_CAL[0])


def _t6_row(day, gross, *, cost=0.0, hedge_ratio=1.0, clipped=False, flags=(), legs=None, phi=1.0, unfilled=None,
            position_gamma=-2.0, im_lots=10.0):
    """One ledger row with the keys the measures read: ``hedged = gross - cost`` (CNY)."""
    return {"date": day.isoformat(), "hedged": gross - cost, "cost": cost, "hedge_ratio": hedge_ratio,
            "clipped": clipped, "flags": list(flags), "legs": dict(legs or {}), "phi": phi,
            "unfilled": dict(unfilled or {}), "position_gamma": position_gamma, "im_lots": im_lots}


def _t6_run(rows, *, note=T6_NOTE, technique="base", mode="close"):
    from gpin.ledger import Run

    return Run(note[0], note[1], 0.1, technique, "baseline", {}, mode, 0.2, 1e-4, 5e8, None, rows=list(rows))


def _t6_event(kind, day, *, start=None, hit=False, note=T6_NOTE, slice_="train"):
    from gpin.events import week_key
    from gpin.measures import Event

    return Event(kind, note[0], note[1], day, start, week_key(day), slice_, hit)


def _bp(cny):
    return cny / NOTIONAL_ * 1.0e4


def _t6_rows(last, overrides=None):
    """Rows on ``T6_CAL[0..last]`` with ``gross_i = 1000 x (i + 1)`` CNY (so every sum of a window is recognisable);
    ``overrides[i]`` are keyword arguments of that day's row."""
    return [_t6_row(T6_CAL[i], 1000.0 * (i + 1), **(overrides or {}).get(i, {})) for i in range(last + 1)]


def test_events_have_stable_ids_and_are_read_from_the_events_file():
    from gpin.measures import Event, note_events

    ko = _t6_event("ko", T6_CAL[12], hit=True)
    ki = _t6_event("ki", T6_CAL[8], start=T6_CAL[5])
    assert (ko.id, ki.id) == ("ko|ki75:2025-05-05|2025-05-21", "ki|ki75:2025-05-05|2025-05-15")
    assert ko.note == "ki75:2025-05-05"
    with pytest.raises(ValidationError):
        Event("ki", "ki75", T6_CAL[0], T6_CAL[8], None, "2025-W20", "train", False)       # an approach has a first day
    with pytest.raises(ValidationError):
        Event("xx", "ki75", T6_CAL[0], T6_CAL[8], None, "2025-W20", "train", False)
    sample = {"books": {
        "ki75": {"ko_events": [{"date": "2025-05-21", "fired": True, "inception": "2025-05-05", "prior_gap": 0.01,
                                "slice": "train", "week": "2025-W21"},
                               {"date": "2025-07-02", "fired": False, "inception": "2025-06-02", "prior_gap": 0.01,
                                "slice": "test", "week": "2025-W27"}],
                 "ki_approaches": [{"date": "2025-05-15", "inception": "2025-05-05", "knocked_in": False,
                                    "slice": "train", "start": "2025-05-12", "week": "2025-W20"}]},
        "seas6": {"ko_events": [], "ki_approaches": []}}}
    got = note_events(sample, "ki75:2025-05-05")
    assert got == [Event("ki", "ki75", T6_CAL[0], date(2025, 5, 15), date(2025, 5, 12), "2025-W20", "train", False),
                   Event("ko", "ki75", T6_CAL[0], date(2025, 5, 21), None, "2025-W21", "train", True)]   # by date
    assert note_events(sample, "seas6:2025-05-05") == []
    # the committed file: 150 KO pin events and 80 approaches on ki75, 176 approaches on seas6, every id distinct
    real = json.loads(EVENTS_JSON.read_text(encoding="utf-8"))
    ids = [e.id for book in ("ki75", "seas6") for inc in real["books"][book]["notes"]
           for e in note_events(real, f"{book}:{inc}")]
    assert len(ids) == len(set(ids)) == 150 + 80 + 176


def test_event_windows_sum_exactly_the_d10_rows_for_a_survived_ko_a_fired_ko_and_an_approach():
    from gpin.measures import event_windows, window_dates

    survived = _t6_event("ko", T6_CAL[12])                    # window: 10 trading days ahead .. the day after
    fired = _t6_event("ko", T6_CAL[33], hit=True)             # the rows end on the KO day: no T+1 row
    approach = _t6_event("ki", T6_CAL[19], start=T6_CAL[16])
    assert window_dates(survived, T6_CAL, 10) == (T6_CAL[2], T6_CAL[13])
    assert window_dates(fired, T6_CAL, 10) == (T6_CAL[23], T6_CAL[34])
    assert window_dates(approach, T6_CAL, 10) == (T6_CAL[16], T6_CAL[20])
    # costs: 500 on the day before the survived window (outside), 300 at its first close (the entry trade), 200 at T + 1
    run = _t6_run(_t6_rows(33, {1: dict(cost=500.0), 2: dict(cost=300.0), 13: dict(cost=200.0)}))
    got = event_windows(run, [survived, fired, approach], T6_CAL, 10)
    assert set(got) == {survived.id, fired.id, approach.id}
    assert got[survived.id] == pytest.approx(_bp(sum(1000.0 * (i + 1) for i in range(2, 14)) - 300.0 - 200.0), rel=1e-12)
    assert got[fired.id] == pytest.approx(_bp(sum(1000.0 * (i + 1) for i in range(23, 34))), rel=1e-12)   # T+1 counts 0
    assert got[approach.id] == pytest.approx(_bp(sum(1000.0 * (i + 1) for i in range(16, 21))), rel=1e-12)
    # under next_open the fired KO has its T+1 row (the hedge carried through the fixing): it is inside the window
    longer = _t6_run(_t6_rows(34), mode="next_open")
    assert event_windows(longer, [fired], T6_CAL, 10)[fired.id] == pytest.approx(
        _bp(sum(1000.0 * (i + 1) for i in range(23, 35))), rel=1e-12)
    # the pre-registered 10 days are the default, and so is the SSE calendar (no holiday falls in this window)
    assert event_windows(run, [survived], T6_CAL) == {survived.id: got[survived.id]}
    assert event_windows(run, [survived]) == {survived.id: got[survived.id]}


def test_event_windows_leave_out_the_day_0_row_count_missing_rows_as_zero_and_refuse_another_notes_event():
    from gpin.measures import event_windows, window_dates

    early = _t6_event("ko", T6_CAL[4])                       # fewer than 10 trading days after the booking day
    assert window_dates(early, T6_CAL, 10) == (T6_CAL[0], T6_CAL[5])
    run = _t6_run(_t6_rows(8))
    assert event_windows(run, [early], T6_CAL, 10)[early.id] == pytest.approx(
        _bp(sum(1000.0 * (i + 1) for i in range(1, 6))), rel=1e-12)                       # row 0 is not in it
    last = _t6_event("ko", T6_CAL[-1])                       # the calendar ends on T: there is no trading day after
    assert window_dates(last, T6_CAL, 10) == (T6_CAL[-11], T6_CAL[-1])
    gone = _t6_event("ki", T6_CAL[30], start=T6_CAL[28])     # every row of the window is missing: 0, not an error
    assert event_windows(run, [gone], T6_CAL, 10) == {gone.id: 0.0}
    with pytest.raises(ValidationError):
        event_windows(run, [_t6_event("ko", T6_CAL[4], note=("ki75", T6_CAL[1]))], T6_CAL, 10)
    with pytest.raises(ValidationError):
        window_dates(_t6_event("ko", date(2025, 5, 10)), T6_CAL, 10)                      # a Saturday: not a trading day


def test_life_stats_and_whole_life_leave_day_0_out_of_the_std_and_truncated_notes_out_of_the_terminal():
    from gpin.measures import life_stats, whole_life

    a = _t6_run([_t6_row(T6_CAL[i], g, cost=c, im_lots=lots, clipped=cl)
                 for i, (g, c, lots, cl) in enumerate([(50000.0, 100.0, 10.0, False), (3000.0, 0.0, 10.0, False),
                                                       (-1000.0, 50.0, 12.0, True), (2000.0, 20.0, 0.0, False)])])
    b = _t6_run([_t6_row(T6_CAL[3 + i], g) for i, g in enumerate([-20000.0, 500.0, 700.0])], note=("ki75", T6_CAL[3]))
    la, lb = life_stats(a, True), life_stats(b, False)
    daily_a = [_bp(3000.0), _bp(-1050.0), _bp(1980.0)]
    assert (la["n"], la["rows"], la["completed"]) == (3, 4, True)
    assert la["sum_bp"] == pytest.approx(sum(daily_a), rel=1e-12)
    assert la["sumsq_bp"] == pytest.approx(sum(v * v for v in daily_a), rel=1e-12)
    assert la["terminal_bp"] == pytest.approx(_bp(49900.0) + sum(daily_a), rel=1e-12)     # day 0 stays in the terminal
    assert la["cost_bp"] == pytest.approx(_bp(170.0), rel=1e-12) and la["clips"] == 1
    assert la["im_lots_traded"] == pytest.approx(10.0 + 0.0 + 2.0 + 12.0, rel=1e-12)
    assert (la["first"], la["last"]) == (T6_CAL[0].isoformat(), T6_CAL[3].isoformat())
    whole = whole_life([{"life": la}, {"life": lb}])
    daily = daily_a + [_bp(500.0), _bp(700.0)]
    assert (whole["n_notes"], whole["n_days"], whole["n_completed"]) == (2, 5, 1)
    assert whole["std_bp"] == pytest.approx(float(np.std(daily, ddof=1)), rel=1e-10)
    assert whole["mean_terminal_bp"] == pytest.approx(la["terminal_bp"], rel=1e-12)        # the truncated note is left out
    assert whole["mean_cost_bp"] == pytest.approx(_bp(170.0) / 2.0, rel=1e-12)
    assert whole_life([la, lb]) == whole                                                  # the life blocks themselves
    only_truncated = whole_life([lb])
    assert only_truncated["mean_terminal_bp"] is None and only_truncated["n_completed"] == 0
    with pytest.raises(ValidationError):
        whole_life([])


def test_life_stats_count_the_overlay_days_the_fill_ratio_and_the_flags():
    from gpin.measures import life_stats

    legs = {"MO2507:C:5000": 2.0}
    run = _t6_run([
        _t6_row(T6_CAL[0], 0.0),
        _t6_row(T6_CAL[1], 0.0, legs=legs, phi=0.25, unfilled={"MO2507:C:5000": 6.0}, flags=("volume_cut",)),
        _t6_row(T6_CAL[2], 0.0, legs=legs, phi=1.0),
        _t6_row(T6_CAL[3], 0.0, flags=("overlay_no_strike", "roll")),
        _t6_row(T6_CAL[4], 0.0, unfilled={"MO2507:C:5000": 2.0}, phi=0.0, flags=("refused_open",)),
    ])
    life = life_stats(run, False)
    assert (life["leg_days"], life["overlay_days"], life["phi_lt1_days"]) == (2, 3, 2)
    assert life["phi_sum"] == pytest.approx(1.25, rel=1e-12) and life["phi_min"] == 0.0
    assert life["flags"] == {"volume_cut": 1, "overlay_no_strike": 1, "roll": 1, "refused_open": 1}
    assert life_stats(_t6_run(_t6_rows(3)), False)["phi_min"] is None                     # no overlay day at all


T6_SERIES = "MO2506"
T6_LAST = date(2025, 6, 20)


def _t6_board(day, spot, quotes=None, forward=None):
    mo = None if quotes is None else {T6_SERIES: dict(last=T6_LAST, forward=forward or spot, quotes=quotes)}
    return _t5_board(day, spot, im={"IM2509": spot}, mo=mo)


def test_gamma_carry_is_the_net_gamma_at_the_close_before_the_fixing_and_the_hedge_carried_into_it():
    from gpin import black
    from gpin.measures import gamma_carry

    event = _t6_event("ko", T6_CAL[12])
    t1, spot, forward = T6_CAL[11], 4990.0, 4975.0
    info = _series_info(t1, T6_LAST, forward)
    quotes = {("C", k): (_px(t1, T6_LAST, "C", k, vol, forward), 5000, 9000) for k, vol in ((4900.0, 0.22), (5000.0, 0.2))}
    boards = {t1: _t6_board(t1, spot, quotes, forward)}
    legs = {"MO2506:C:4900": 40.0, "MO2506:C:5000": -40.0}
    ratios = {2: 1.2, 5: 2.4, 11: 0.9, 12: 0.1, 13: 0.0}
    rows = _t6_rows(13, {i: dict(hedge_ratio=h) for i, h in ratios.items()})
    rows[11] = _t6_row(t1, 12000.0, hedge_ratio=0.9, legs=legs, position_gamma=-35.0)
    got = gamma_carry(_t6_run(rows), [event, _t6_event("ki", T6_CAL[8], start=T6_CAL[6])], boards, T6_CAL, 10)
    assert set(got) == {event.id}                             # KO events only
    g = got[event.id]
    # Black-76 gamma under the proportional forward: discount x pdf(d1) / (F sigma sqrt(T)) x (F / S)^2, x 100 a lot
    def gamma(k, vol):
        s = vol * math.sqrt(info.tenor)
        d1 = (math.log(forward / k) + 0.5 * s * s) / s
        return info.discount * _norm.pdf(d1) / (forward * s) * (forward / spot) ** 2
    legs_gamma = 100.0 * (40.0 * gamma(4900.0, 0.22) - 40.0 * gamma(5000.0, 0.2))
    assert legs_gamma == pytest.approx(100.0 * 40.0 * (black.index_gamma(0.22, "C", 4900.0, info, spot)
                                                         - black.index_gamma(0.2, "C", 5000.0, info, spot)), rel=1e-9)
    assert g["date"] == t1.isoformat() and g["note_gamma"] == -35.0
    assert g["legs_gamma"] == pytest.approx(legs_gamma, rel=1e-6)
    assert g["net_gamma"] == pytest.approx(-35.0 + legs_gamma, rel=1e-6)
    assert g["net_gamma_1pct_bp"] == pytest.approx(_bp((-35.0 + legs_gamma) * (0.01 * spot) ** 2), rel=1e-6)
    assert g["hedge_into_fixing"] == 0.9                      # the hedge carried at the T-1 close
    assert g["hedge_max"] == 2.4                              # the largest in the window (rows 2 .. 13)
    # no legs: the note's gamma alone, and no board is needed; a missing T-1 row or a null gamma gives nulls
    plain = gamma_carry(_t6_run(_t6_rows(13, {11: dict(position_gamma=-7.0)})), [event], {}, T6_CAL, 10)[event.id]
    assert (plain["note_gamma"], plain["legs_gamma"], plain["net_gamma"]) == (-7.0, 0.0, -7.0)
    assert plain["net_gamma_1pct_bp"] is None                 # no board, no spot to scale with
    cut = gamma_carry(_t6_run(_t6_rows(5)), [event], {}, T6_CAL, 10)[event.id]
    assert (cut["note_gamma"], cut["net_gamma"], cut["hedge_into_fixing"]) == (None, None, None)
    # a leg held without a mark on the board fails closed
    with pytest.raises(StudyDataError):
        gamma_carry(_t6_run(rows), [event], {t1: _t6_board(t1, spot)}, T6_CAL, 10)


def test_summarise_run_stores_per_event_what_the_report_needs_and_flags_the_events_left_out_under_next_open():
    from gpin.measures import event_windows, summarise_run

    ko = _t6_event("ko", T6_CAL[12])
    ki = _t6_event("ki", T6_CAL[19], start=T6_CAL[16])
    rows = _t6_rows(20, {2: dict(cost=300.0, clipped=True), 5: dict(clipped=True, flags=("roll",)),
                           12: dict(flags=("fixing_deferred",)), 17: dict(flags=("overlay_no_series",)),
                           19: dict(flags=("last_day_fixing",)), 1: dict(clipped=True)})
    run = _t6_run(rows, mode="next_open")
    s = summarise_run(run, [ko, ki], T6_CAL, {}, completed=False, ko_days=10)
    assert set(s) == {"events", "life"} and set(s["events"]) == {ko.id, ki.id}
    windows = event_windows(run, [ko, ki], T6_CAL, 10)
    e = s["events"][ko.id]
    assert e["pnl_bp"] == windows[ko.id] and e["cost_bp"] == pytest.approx(_bp(300.0), rel=1e-12)
    assert (e["clips"], e["left_out"], e["flags"]) == (2, False, {"roll": 1, "fixing_deferred": 1})
    assert e["days"][0] == [T6_CAL[2].isoformat(), pytest.approx(_bp(3000.0 - 300.0)), 1.0] and len(e["days"]) == 12
    assert e["gamma"]["hedge_into_fixing"] == 1.0 and e["gamma"]["note_gamma"] == -2.0
    k = s["events"][ki.id]
    assert k["left_out"] is True and k["flags"] == {"overlay_no_series": 1, "last_day_fixing": 1} and "gamma" not in k
    assert s["life"]["n"] == 20 and s["life"]["completed"] is False
    json.dumps(s, allow_nan=False)                            # JSON-ready


# ---- the verdict (D10, AC11) on hand-made event P&L ----


def _t6_arms(n_events=24, n_weeks=12):
    """``n_events`` events spread over ``n_weeks`` weeks (event i in week i mod n_weeks); the baseline loses ``-100 +
    10 i`` bp on event i, so with 24 events its ES10 is the mean of -100, -90, -80 and its worst event -100."""
    base = {f"e{i:02d}": -100.0 + 10.0 * i for i in range(n_events)}
    weeks = {e: f"2025-W{30 + i % n_weeks:02d}" for i, e in enumerate(base)}
    return base, weeks


def _t6_life(std, terminal):
    return {"n_notes": 13, "n_days": 4000, "std_bp": std, "n_completed": 9, "mean_terminal_bp": terminal,
            "mean_cost_bp": 1.0, "mean_im_lots_traded": 2.0}


def _t6_rules(**kw):
    from dataclasses import replace
    from gpin.measures import Rules

    return replace(Rules.from_prereg(load_prereg()), **{"bootstrap_n": 2000, **kw})


def test_rules_are_the_pre_registered_numbers():
    from gpin.measures import Rules

    r = Rules.from_prereg(load_prereg())
    assert (r.es_alpha, r.bootstrap_n, r.seed, r.p_max) == (0.10, 10000, 20261112, 0.10)
    assert (r.std_ratio_max, r.terminal_slack_bp, r.min_events, r.min_weeks, r.ko_days) == (1.05, 10.0, 20, 10, 10)


def test_verdict_win_needs_the_tail_and_the_whole_life():
    from gpin.measures import verdict

    base, weeks = _t6_arms()
    tech = {e: v + 30.0 for e, v in base.items()}
    v = verdict(tech, base, weeks, _t6_life(10.4, -5.0), _t6_life(10.0, 0.0), _t6_rules())
    assert (v["n_events"], v["n_weeks"], v["decidable"]) == (24, 12, True)
    assert v["es10"] == pytest.approx(-60.0) and v["es10_base"] == pytest.approx(-90.0)   # k = ceil(2.4) = 3
    assert (v["worst"], v["worst_base"], v["p"]) == (-70.0, -100.0, 0.0)
    assert v["mean"] == pytest.approx(45.0) and v["mean_base"] == pytest.approx(15.0)
    assert (v["es_higher"], v["worst_not_lower"], v["tail_pass"]) == (True, True, True)
    assert v["std_ratio"] == pytest.approx(1.04) and v["terminal_diff_bp"] == pytest.approx(-5.0)
    assert (v["std_pass"], v["terminal_pass"], v["life_pass"], v["win"]) == (True, True, True, True)
    # the bounds are inclusive: 1.05 x the baseline's std and exactly 10 bp below its terminal P&L still pass
    edge = verdict(tech, base, weeks, _t6_life(10.5, -10.0), _t6_life(10.0, 0.0), _t6_rules())
    assert (edge["life_pass"], edge["win"]) == (True, True)


def test_verdict_fails_on_the_tail():
    from gpin.measures import cluster_bootstrap, verdict

    base, weeks = _t6_arms()
    good_life = (_t6_life(10.0, 0.0), _t6_life(10.0, 0.0))
    # ES10 higher (-101, -30, -20 against -100, -90, -80) and p below 0.10 (the technique is only behind on resamples
    # that draw week 30 three times or more, 7% of them), but the worst event is 1 bp lower: that alone fails the tail
    tech = {e: v + 60.0 for e, v in base.items()}
    tech["e00"] = -101.0
    v = verdict(tech, base, weeks, *good_life, _t6_rules())
    assert v["es10"] == pytest.approx(-151.0 / 3.0) and (v["es_higher"], v["worst_not_lower"]) == (True, False)
    assert 0.04 < v["p"] < 0.10 and (v["worst"], v["worst_base"]) == (-101.0, -100.0)
    assert (v["tail_pass"], v["life_pass"], v["win"]) == (False, True, False)
    # ES10 not higher: the same events
    same = verdict(dict(base), base, weeks, *good_life, _t6_rules())
    assert (same["es_higher"], same["worst_not_lower"], same["p"], same["tail_pass"], same["win"]) == (
        False, True, 1.0, False, False)
    # ES10 higher and the worst event higher, but the gain sits in one week (events 0 and 12, both week 30): every
    # resample without that week shows no gain, (11 / 12) ** 12 = 35% of them
    one_week = dict(base)
    one_week["e00"], one_week["e12"] = -70.0, 50.0
    v = verdict(one_week, base, weeks, *good_life, _t6_rules())
    assert v["es10"] == pytest.approx((-90.0 - 80.0 - 70.0) / 3.0) and v["worst"] == -90.0
    assert v["p"] == cluster_bootstrap(one_week, base, weeks, n=2000) and abs(v["p"] - (11.0 / 12.0) ** 12) < 0.03
    assert (v["es_higher"], v["worst_not_lower"], v["tail_pass"], v["win"]) == (True, True, False, False)


def test_verdict_fails_on_the_whole_life():
    from gpin.measures import verdict

    base, weeks = _t6_arms()
    tech = {e: v + 30.0 for e, v in base.items()}
    noisy = verdict(tech, base, weeks, _t6_life(10.6, 0.0), _t6_life(10.0, 0.0), _t6_rules())
    assert (noisy["tail_pass"], noisy["std_pass"], noisy["terminal_pass"], noisy["life_pass"], noisy["win"]) == (
        True, False, True, False, False)
    costly = verdict(tech, base, weeks, _t6_life(9.0, -10.5), _t6_life(10.0, 0.0), _t6_rules())
    assert (costly["tail_pass"], costly["std_pass"], costly["terminal_pass"], costly["life_pass"], costly["win"]) == (
        True, True, False, False, False)
    # no completed note: the terminal criterion cannot be evaluated, so neither can the win
    unknown = verdict(tech, base, weeks, _t6_life(9.0, None), _t6_life(10.0, None), _t6_rules())
    assert (unknown["terminal_pass"], unknown["life_pass"], unknown["win"]) == (None, None, None)


def test_verdict_is_not_decidable_with_too_few_events_or_too_few_weeks():
    from gpin.measures import verdict

    lives = (_t6_life(10.0, 0.0), _t6_life(10.0, 0.0))
    for n_events, n_weeks, decidable in ((19, 12, False), (24, 9, False), (20, 10, True)):
        base, weeks = _t6_arms(n_events, n_weeks)
        tech = {e: v + 30.0 for e, v in base.items()}
        v = verdict(tech, base, weeks, *lives, _t6_rules())
        assert (v["n_events"], v["n_weeks"], v["decidable"]) == (n_events, n_weeks, decidable)
        assert v["tail_pass"] is True and v["life_pass"] is True         # the measures are still reported
        assert v["win"] is (True if decidable else None)
    # no event at all: nothing to measure, not decidable
    none = verdict({}, {}, {}, *lives, _t6_rules())
    assert (none["n_events"], none["n_weeks"], none["decidable"], none["es10"], none["p"], none["tail_pass"],
            none["win"]) == (0, 0, False, None, None, None, None)
    with pytest.raises(ValidationError):
        verdict({"a": 1.0}, {"b": 1.0}, {"a": "w", "b": "w"}, *lives, _t6_rules())


# ---- training selection (AC12) on hand-made training tables ----


def _t6_windows(es_by_id):
    """``{technique id: {event: bp}}`` over three events whose worst is the given number (ES10 of three = the worst)."""
    return {tid: {"e1": v, "e2": v + 10.0, "e3": v + 20.0} for tid, v in es_by_id.items()}


T6_KO_ES = {
    "base": -40.0,
    "ko_shift@0.01": -50.0, "ko_shift@0.02": -30.0, "ko_shift@0.03": -40.0,
    "ko_spread@0.015": -45.0, "ko_spread@0.03": -45.0,                    # a tie: the lower cost wins
    "ko_bend@0.015,3": -45.0, "ko_bend@0.03,3": -45.0, "ko_bend@0.03,5": -46.0,   # a tie on both: the grid's order
    "secant@0.03": -60.0, "secant@0.05": -20.0,
    "surv_hold@1": -25.0, "surv_hold@3": -35.0,
    "derisk@3": -70.0, "derisk@5": -65.0,                                 # the winner does not beat the baseline
    "freeze@1": -80.0, "freeze@2": -85.0,
    "mo_ko@1,payoff": -22.0, "mo_ko@2,payoff": -28.0, "mo_ko@1,scenario": -21.0,
}
T6_KI_ES = {
    "base": -20.0,
    "ki_shift@-0.01": -15.0, "ki_shift@-0.02": -12.0,
    "secant@0.03": -10.0, "secant@0.05": -90.0,                           # the other h than at the KO barrier
    "ki_cap@1.0": -12.0, "ki_cap@1.25": -30.0,
    "mo_ki@2,payoff": -5.0, "mo_ki@2,scenario": -7.0,
}


def _t6_costs(**overrides):
    costs = {"ko": {tid: 1.0 for tid in T6_KO_ES}, "ki": {tid: 1.0 for tid in T6_KI_ES}}
    costs["ko"].update({"ko_spread@0.015": 3.0, "ko_spread@0.03": 2.0})
    for barrier, block in overrides.items():
        costs[barrier].update(block)
    return costs


def test_select_on_train_picks_the_highest_training_es10_per_row_and_barrier_with_the_tie_rule():
    from gpin.measures import select_on_train

    windows = {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(T6_KI_ES)}
    sel = select_on_train(load_prereg(), windows, _t6_costs())
    assert {row: p["id"] for row, p in sel["picks"]["ko"].items()} == {
        "ko_shift": "ko_shift@0.02", "ko_spread": "ko_spread@0.03", "ko_bend": "ko_bend@0.015,3",
        "secant": "secant@0.05", "surv_hold": "surv_hold@1", "derisk": "derisk@5", "freeze": "freeze@1",
        "mo_ko": "mo_ko@1,scenario"}
    assert {row: p["id"] for row, p in sel["picks"]["ki"].items()} == {
        "ki_shift": "ki_shift@-0.02", "secant": "secant@0.03", "ki_cap": "ki_cap@1.0", "mo_ki": "mo_ki@2,payoff"}
    pick = sel["picks"]["ko"]["ko_shift"]
    assert (pick["es10"], pick["cost_bp"], pick["n_events"], pick["beats_base"]) == (-30.0, 1.0, 3, True)
    assert pick["grid"] == {"ko_shift@0.01": {"es10": -50.0, "cost_bp": 1.0},
                            "ko_shift@0.02": {"es10": -30.0, "cost_bp": 1.0},
                            "ko_shift@0.03": {"es10": -40.0, "cost_bp": 1.0}}
    # a row whose winner does not beat the baseline is still carried to the test with that winner
    assert (sel["picks"]["ko"]["derisk"]["es10"], sel["picks"]["ko"]["derisk"]["beats_base"]) == (-65.0, False)
    assert sel["picks"]["ko"]["ko_shift"]["beats_base"] is True
    assert sel["base"] == {"ko": {"es10": -40.0, "n_events": 3}, "ki": {"es10": -20.0, "n_events": 3}}
    assert sel["rows"] == {b: list(rows) for b, rows in sel["picks"].items()}
    # composite members: the winner across the futures-side rows of each barrier; the overlay rows' own picks
    assert sel["composites"] == {"sop_futures": {"ko": "secant@0.05", "ki": "secant@0.03"},
                                 "sop_overlay": {"ko": "mo_ko@1,scenario", "ki": "mo_ki@2,payoff"}}
    assert set(_registry(sel["composites"])) >= {"sop_futures", "sop_overlay"}        # the registry accepts it


def test_select_on_train_breaks_a_tie_across_rows_by_the_lower_cost_then_by_the_registered_order():
    from gpin.measures import select_on_train

    ki = dict(T6_KI_ES)
    ki["ki_cap@1.0"] = -10.0                                # ties secant@0.03 for the futures-side KI member
    windows = {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(ki)}
    cheaper = select_on_train(load_prereg(), windows, _t6_costs(ki={"ki_cap@1.0": 0.5}))
    assert cheaper["composites"]["sop_futures"]["ki"] == "ki_cap@1.0"
    # the same cost too: the row listed first in the composite's members (ki_shift, ki_cap, secant)
    assert select_on_train(load_prereg(), windows, _t6_costs())["composites"]["sop_futures"]["ki"] == "ki_cap@1.0"
    dearer = select_on_train(load_prereg(), windows, _t6_costs(ki={"ki_cap@1.0": 1.5}))
    assert dearer["composites"]["sop_futures"]["ki"] == "secant@0.03"


def test_select_on_train_fails_closed_on_missing_runs_unpaired_events_and_an_empty_training_slice():
    from gpin.measures import select_on_train

    windows = {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(T6_KI_ES)}
    missing = {"ko": {k: v for k, v in windows["ko"].items() if k != "freeze@2"}, "ki": windows["ki"]}
    with pytest.raises(ValidationError):
        select_on_train(load_prereg(), missing, _t6_costs())
    unpaired = {"ko": dict(windows["ko"]), "ki": windows["ki"]}
    unpaired["ko"]["freeze@2"] = {"e1": 1.0, "e2": 2.0}
    with pytest.raises(ValidationError):
        select_on_train(load_prereg(), unpaired, _t6_costs())
    empty = {"ko": {tid: {} for tid in T6_KO_ES}, "ki": windows["ki"]}
    with pytest.raises(StudyDataError):
        select_on_train(load_prereg(), empty, _t6_costs())
    # a partial set of states (a smoke run) may have no event of a barrier type: said so, and its rows have NO pick (no
    # grid point stands in for one); such a selection is marked and the test slice refuses it
    partial = select_on_train(load_prereg(), empty, _t6_costs(), allow_empty=True)
    assert partial["empty"] == ["ko"]
    assert partial["picks"]["ko"]["ko_shift"] == {
        "id": None, "es10": None, "cost_bp": None, "n_events": 0, "beats_base": None,
        "grid": {tid: {"es10": None, "cost_bp": None} for tid in ("ko_shift@0.01", "ko_shift@0.02", "ko_shift@0.03")}}
    assert all(p["id"] is None for p in partial["picks"]["ko"].values()) and partial["rows"]["ko"][0] == "ko_shift"
    assert partial["picks"]["ki"]["ki_cap"]["id"] == "ki_cap@1.0" and partial["base"]["ko"] == {"es10": None, "n_events": 0}
    # a composite needs a member for both barriers: with one barrier empty there is none, and none is written
    assert partial["composites"] == {}
    assert select_on_train(load_prereg(), windows, _t6_costs())["empty"] == []


def test_evaluated_ids_are_the_baselines_the_controls_the_cost_rows_the_picks_and_the_composites():
    from gpin.measures import evaluated_ids, select_on_train

    sel = select_on_train(load_prereg(), {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(T6_KI_ES)}, _t6_costs())
    assert evaluated_ids(load_prereg(), sel) == [
        "base", "base_noclip", "ko_shift@0.02", "ko_spread@0.03", "ko_bend@0.015,3", "ki_shift@-0.02", "ko_wrong",
        "ki_wrong", "secant@0.03", "secant@0.05", "ki_cap@1.0", "surv_hold@1", "derisk@5", "freeze@1",
        "mo_ko@1,scenario", "mo_ki@2,payoff", "sop_futures", "sop_overlay", "band@0.02", "band@0.05"]


# ---- a hand-made market straddling the split: a 2Y note at its KO barrier and a seasoned note at its KI barrier ----

T6_DAYS = _weekdays(40, date(2025, 6, 2))          # 2025-06-02 (a Monday) .. 2025-07-25; day 21 is the split date
T6_SPLIT, T6_UNTIL = date(2025, 7, 1), date(2025, 6, 30)
T6_SPOTS = [5080.0, 5080.0, 5070.0, 5060.0, 5075.0, 5085.0, 4930.0, 4900.0, 4830.0, 4920.0,      # 0 .. 9
            5090.0, 4990.0, 5050.0, 5060.0, 5070.0, 5080.0, 5090.0, 5100.0, 5080.0, 5060.0,      # 10 .. 19
            5050.0, 5040.0, 5030.0, 4920.0, 4880.0, 4820.0, 4690.0, 4950.0, 4960.0, 4980.0,      # 20 .. 29
            4990.0, 5020.0, 5030.0, 5035.0, 5040.0, 5030.0, 5020.0, 5025.0, 5030.0, 5040.0]      # 30 .. 39
T6_KO_OBS = (11, 31)                               # the 2Y note's observations: 06-17 survived, 07-15 fired
T6_KI_LEVEL = 4700.0                               # the seasoned note's KI level: window at or below 4935
T6_KI_RUNS = ((6, 9), (23, 26))                    # its approaches: 06-10 .. 06-13, and 07-03 .. 07-08 ending in a knock-in
T6_LASTS = {"MO2507": date(2025, 8, 1), "MO2508": date(2025, 8, 15)}
T6_STRIKES = [4400.0 + 100.0 * i for i in range(10)]                        # 4400 .. 5300


def _t6_specs(kind):
    specs = []
    for i, (day, spot) in enumerate(zip(T6_DAYS, T6_SPOTS)):
        sp = dict(day=day, spot=spot, im={"IM2509": spot - 20.0}, mo=_ov_mo(day, T6_LASTS, T6_STRIKES, spot - 10.0))
        if kind == "ko":
            obs = next(o for o in T6_KO_OBS + (60,) if o > i)
            in_window = obs - i <= 10 and obs in T6_KO_OBS
            sp.update(n=obs - i, ko_dir=-1 if in_window else None, gap_ko=4.0e5 if in_window else None,
                      deltas=dict(KO_DELTAS) if in_window else {"true": 1000.0},
                      prices={m: 150000.0 + 1000.0 * (spot - 5000.0) for m in KO_DELTAS})
        else:
            in_window = any(a <= i <= b for a, b in T6_KI_RUNS) and i != 26
            sp.update(n=70 - i, knocked_in=i >= 26, deltas=dict(KI_DELTAS) if in_window else {"true": 2000.0},
                      prices={"true": 400000.0 + 2000.0 * (spot - 5000.0), "ki_m200": 100000.0, "kin": -300000.0})
        specs.append(sp)
    return specs


def _t6_market():
    """``(ko_states, ki_states, boards, chains)``: the 2Y note ``ki75:2025-06-02`` (strike and KO level 5000; it
    survives 06-17 and is redeemed on 07-15) and the seasoned note ``seas6:2025-06-02`` (KI level 4700; it knocks in on
    07-08 and is alive at the data end), on one market."""
    from gpin.board import DayBoard, ImBar

    ko_states, boards, chains = _t5_world(_t6_specs("ko"))
    ki_states, _, _ = _t5_world(_t6_specs("ki"), strike=T6_KI_LEVEL / 0.75, ki=T6_KI_LEVEL, book="seas6")
    # IM bars with an open and a VWAP, so the twap and next_open clocks have their fills
    boards = {d: DayBoard(d, b.spot, {c: ImBar(bar.close - 6.0, bar.close, bar.close, bar.close - 2.0, 1000)
                                      for c, bar in b.im.items()}, b.mo, b.series) for d, b in boards.items()}
    dead = ko_states["days"][31]
    dead.update(alive=False, engine_note_cash=250000.0, end_reason="ko", states={}, next_ko=None, days_to_ko=None)
    ko_states["days"] = ko_states["days"][:32]
    return ko_states, ki_states, boards, chains


def _t6_sample():
    """The events file of the hand-made market, in the shape of ``results/events.json``."""
    def iso(i):
        return T6_DAYS[i].isoformat()

    def ko(i, fired):
        return {"date": iso(i), "fired": fired, "inception": iso(0), "prior_gap": 0.01,
                "slice": "train" if T6_DAYS[i] < T6_SPLIT else "test", "week": week_key_of(T6_DAYS[i])}

    def ki(a, b, knocked):
        return {"date": iso(b), "inception": iso(0), "knocked_in": knocked, "start": iso(a),
                "slice": "train" if T6_DAYS[b] < T6_SPLIT else "test", "week": week_key_of(T6_DAYS[b])}

    note = {"end_date": iso(39), "end_reason": "truncated", "ki_date": None, "ko_date": None, "ko_observations": 0}
    return {"split_date": "2025-07-01", "books": {
        "ki75": {"headline": [iso(0)], "pin": [], "notes": {iso(0): dict(note)},
                 "ko_events": [ko(11, False), ko(31, True)], "ki_approaches": []},
        "seas6": {"headline": [iso(0)], "pin": [], "notes": {iso(0): dict(note)},
                  "ko_events": [], "ki_approaches": [ki(6, 9, False), ki(23, 26, True)]}}}


@pytest.fixture(scope="module")
def t6_market():
    return _t6_market()


def _t6_plan(modes=("close",)):
    from gpin.measures import primary_config

    return [(primary_config(load_prereg()), tuple(modes))]


def test_configs_are_the_pre_registered_desk_and_cost_and_one_at_a_time_sensitivities():
    from gpin.measures import Config, primary_config, sensitivity_configs

    prereg = load_prereg()
    primary = primary_config(prereg)
    assert (primary, primary.key) == (Config(5.0e8, 0.2), "desk500_cost0.2")
    assert sensitivity_configs(prereg, [5.0e8], [0.2]) == []
    assert [c.key for c in sensitivity_configs(prereg, [5.0e8, 1.0e8], [0.2, 1.0])] == [
        "desk100_cost0.2", "desk500_cost1"]                               # never both changed at once


def test_run_note_keeps_per_technique_and_clock_the_events_the_life_and_the_gap_observations(t6_market):
    from gpin.measures import (event_windows, gap_observations, life_stats, note_completed, note_events, run_note,
                               summarise_run)

    ko_states, ki_states, boards, chains = t6_market
    reg = _registry()
    techs = [reg["base"], reg["surv_hold@1"], reg["mo_ko@1,payoff"]]
    events = note_events(_t6_sample(), "ki75:2025-06-02")
    note = run_note(ko_states, techs, boards, chains, events, plan=_t6_plan(("close", "next_open")))
    assert (note["note"], note["book"], note["inception"], note["until"]) == ("ki75:2025-06-02", "ki75", "2025-06-02", None)
    assert (note["completed"], note["end_reason"], note["last_record"]) == (True, "ko", "2025-07-15")
    assert note["events"] == {e.id: e.to_json() for e in events} and len(events) == 2
    assert set(note["runs"]) == {"desk500_cost0.2"} and set(note["runs"]["desk500_cost0.2"]) == {"close", "next_open"}
    assert list(note["runs"]["desk500_cost0.2"]["close"]) == ["base", "surv_hold@1", "mo_ko@1,payoff"]
    # one cell is the summary of that run
    run = _run(ko_states, reg["surv_hold@1"], boards, chains)
    days = [c.day for c in chains]
    got = note["runs"]["desk500_cost0.2"]["close"]["surv_hold@1"]
    assert got == summarise_run(run, events, days, boards, completed=True)
    assert got["life"] == life_stats(run, True) and got["life"]["last"] == "2025-07-15"
    assert {eid: e["pnl_bp"] for eid, e in got["events"].items()} == event_windows(run, events, days)
    # the technique does act: the survived-branch delta into each fixing is not the baseline's hedge
    base = note["runs"]["desk500_cost0.2"]["close"]["base"]
    for e in events:
        assert got["events"][e.id]["gamma"]["hedge_into_fixing"] < base["events"][e.id]["gamma"]["hedge_into_fixing"]
    # under next_open the redeemed note has its T+1 row
    assert note["runs"]["desk500_cost0.2"]["next_open"]["base"]["life"]["last"] == "2025-07-16"
    # the overlay holds legs into the fixing: their gamma is in the net gamma
    overlay = note["runs"]["desk500_cost0.2"]["close"]["mo_ko@1,payoff"]["events"][events[0].id]
    assert overlay["leg_days"] > 0 and overlay["gamma"]["legs_gamma"] != 0.0
    assert overlay["gamma"]["net_gamma"] == pytest.approx(overlay["gamma"]["note_gamma"] + overlay["gamma"]["legs_gamma"])
    # gap observations: the first KO-window day of each observation
    assert note["gap_observations"] == gap_observations(ko_states, None) == [
        {"obs": "2025-06-17", "date": "2025-06-03", "gap_ko_bp": 400.0, "ko_dir": -1},
        {"obs": "2025-07-15", "date": "2025-07-01", "gap_ko_bp": 400.0, "ko_dir": -1}]
    assert note_completed(ko_states, None) and not note_completed(ko_states, T6_UNTIL)
    assert not note_completed(ki_states, None)
    json.dumps(note, allow_nan=False)


def test_run_note_on_the_training_slice_stops_at_the_last_day_before_the_split(t6_market):
    from gpin.measures import note_events, run_note

    ko_states, ki_states, boards, chains = t6_market
    reg = _registry()
    for states, nid in ((ko_states, "ki75:2025-06-02"), (ki_states, "seas6:2025-06-02")):
        events = [e for e in note_events(_t6_sample(), nid) if e.slice == "train"]
        assert len(events) == 1
        note = run_note(states, [reg["base"], reg["ki_cap@1.0"]], boards, chains, events,
                        plan=_t6_plan(("close", "twap", "next_open")), until=T6_UNTIL)
        assert (note["until"], note["completed"]) == ("2025-06-30", False)
        assert [g["obs"] for g in note["gap_observations"]] == (["2025-06-17"] if states is ko_states else [])
        for clock, cells in note["runs"]["desk500_cost0.2"].items():
            for tid, cell in cells.items():
                assert cell["life"]["last"] == "2025-06-30" and cell["life"]["rows"] == 21
                assert all(d[0] < "2025-07-01" for e in cell["events"].values() for d in e["days"])
        # an event dated on or after the split is refused: no test-dated P&L is ever computed on the training slice
        with pytest.raises(ValidationError):
            run_note(states, [reg["base"]], boards, chains, note_events(_t6_sample(), nid), plan=_t6_plan(),
                     until=T6_UNTIL)


def test_gap_observations_belong_to_the_slice_of_the_observation_date():
    """An observation whose KO window opens before the split and which is dated after it is a test-slice observation:
    a training file does not hold it (nor any other date after the last training day)."""
    from gpin.measures import gap_observations

    def rec(day, obs, gap, ko_dir=-1):
        return {"date": day, "next_ko": obs, "gap_ko": gap, "ko_dir": ko_dir}

    states = {"days": [rec("2025-06-02", "2025-06-17", None, None), rec("2025-06-03", "2025-06-17", 2.5e5),
                       rec("2025-06-04", "2025-06-17", -1.0e5, -1), rec("2025-06-26", "2025-07-08", -3.0e4, 1),
                       rec("2025-06-27", "2025-07-08", 5.0e4, 1), rec("2025-07-09", "2025-08-08", 0.0, 0)]}
    first = {"obs": "2025-06-17", "date": "2025-06-03", "gap_ko_bp": 250.0, "ko_dir": -1}
    assert gap_observations(states, T6_UNTIL) == [first]
    assert gap_observations(states, None) == [first, {"obs": "2025-07-08", "date": "2025-06-26", "gap_ko_bp": -30.0, "ko_dir": 1},
                                              {"obs": "2025-08-08", "date": "2025-07-09", "gap_ko_bp": 0.0, "ko_dir": 0}]


def test_run_note_refuses_an_event_dated_after_the_notes_last_recorded_day(t6_market):
    """A short diagnostic state file (``03_pricing.py --quick``) ends before its note's events: their windows would
    hold no row and count 0, which is not a measurement."""
    from gpin.measures import measurable_events, note_events, run_note

    ko_states, _, boards, chains = t6_market
    short = dict(ko_states, days=ko_states["days"][:8])                    # 06-02 .. 06-11, the note alive
    events = note_events(_t6_sample(), "ki75:2025-06-02")
    with pytest.raises(ValidationError, match="last recorded day"):
        run_note(short, [_registry()["base"]], boards, chains, events, plan=_t6_plan())
    assert measurable_events(short, events) == [] and measurable_events(ko_states, events) == events
    note = run_note(short, [_registry()["base"]], boards, chains, [], plan=_t6_plan())
    assert (note["completed"], note["last_record"], note["events"]) == (False, "2025-06-11", {})


def test_merge_notes_adds_the_composites_runs_and_refuses_anything_else(t6_market):
    from gpin.measures import merge_notes, note_events, run_note

    ko_states, _, boards, chains = t6_market
    reg = _registry()
    events = note_events(_t6_sample(), "ki75:2025-06-02")
    first = run_note(ko_states, [reg["base"]], boards, chains, events, plan=_t6_plan())
    second = run_note(ko_states, [reg["sop_futures"]], boards, chains, events, plan=_t6_plan())
    merged = merge_notes(first, second)
    assert list(merged["runs"]["desk500_cost0.2"]["close"]) == ["base", "sop_futures"]
    assert list(first["runs"]["desk500_cost0.2"]["close"]) == ["base"]                 # the inputs are not changed
    with pytest.raises(ValidationError):
        merge_notes(first, first)                                                      # the same technique twice
    other = dict(second, until="2025-06-30")
    with pytest.raises(ValidationError):
        merge_notes(first, other)


# ---- the three hashes and the guard of the test slice (D11, AC12) ----


def test_the_code_hash_covers_exactly_the_ledger_side_files(tmp_path):
    import hashlib
    from gpin import measures

    assert measures.CODE_FILES == ("black.py", "ledger.py", "techniques.py", "measures.py")
    h = hashlib.sha256()
    for name in measures.CODE_FILES:
        h.update(name.encode("utf-8") + b"\0" + (STUDY_DIR / "gpin" / name).read_bytes() + b"\0")
    assert measures.code_hash() == h.hexdigest()
    # a copy with one ledger-side file changed hashes differently; a pricing-side file is not in it
    for name in measures.CODE_FILES + ("pricing.py",):
        (tmp_path / name).write_bytes((STUDY_DIR / "gpin" / name).read_bytes())
    assert measures.code_hash(tmp_path) == measures.code_hash()
    (tmp_path / "pricing.py").write_text("# changed\n")
    assert measures.code_hash(tmp_path) == measures.code_hash()
    (tmp_path / "black.py").write_bytes((tmp_path / "black.py").read_bytes() + b"\n")
    assert measures.code_hash(tmp_path) != measures.code_hash()
    f = tmp_path / "events.json"
    f.write_text("{}")
    assert measures.file_hash(f) == hashlib.sha256(b"{}").hexdigest()


def test_check_selection_refuses_a_missing_file_each_wrong_hash_and_a_partial_selection(tmp_path):
    from gpin.measures import SelectionRefused, check_selection

    current = {"prereg_hash": "p" * 64, "events_hash": "e" * 64, "code_hash": "c" * 64, "inputs_hash": "i" * 64}
    path = tmp_path / "selection.json"
    with pytest.raises(SelectionRefused, match="does not exist"):
        check_selection(path, current)
    good = {**current, "partial": False, "picks": {}, "composites": {}}
    path.write_text(json.dumps(good))
    assert check_selection(path, current) == good
    names = ("pre-registration", "events file", "ledger-side code", "recorded inputs")
    for key, word in zip(("prereg_hash", "events_hash", "code_hash", "inputs_hash"), names):
        path.write_text(json.dumps({**good, key: "0" * 64}))
        with pytest.raises(SelectionRefused, match=word) as err:
            check_selection(path, current)
        assert sum(w in str(err.value) for w in names) == 1 and key in str(err.value)
    # a selection made before the fourth hash existed carries none: refused
    path.write_text(json.dumps({k: v for k, v in good.items() if k != "inputs_hash"}))
    with pytest.raises(SelectionRefused, match="recorded inputs"):
        check_selection(path, current)
    path.write_text(json.dumps({k: v for k, v in good.items() if k != "code_hash"}))
    with pytest.raises(SelectionRefused, match="ledger-side code"):
        check_selection(path, current)
    path.write_text(json.dumps({**good, "partial": True}))
    with pytest.raises(SelectionRefused, match="partial"):
        check_selection(path, current)
    assert check_selection(path, current, allow_partial=True)["partial"] is True
    assert issubclass(SelectionRefused, StudyDataError)


# ---- aggregation: the training selection and the verdict tables from stored note files (hand-made) ----

T6_CFG, T6_CFG_DESK = "desk500_cost0.2", "desk100_cost0.2"


def _t6_cell(note, events, pnl, daily, *, completed=True, left_out=(), cost=0.5, clips=1, hedge=1.5, flags=None,
             legs=False):
    """One stored run: ``pnl[event id]`` bp per event of ``events``, ``daily`` CNY rows (the first is day 0)."""
    from gpin.measures import life_stats

    cells = {}
    for e in events:
        cells[e.id] = {"pnl_bp": pnl[e.id], "cost_bp": cost, "clips": clips, "leg_days": 2 if legs else 0,
                       "phi_min": 0.5 if legs else None, "flags": dict(flags or {}), "left_out": e.id in left_out,
                       "days": []}
        if e.kind == "ko":
            cells[e.id]["gamma"] = {"date": "2025-07-01", "note_gamma": -10.0, "legs_gamma": 4.0 if legs else 0.0,
                                    "net_gamma": -6.0 if legs else -10.0, "net_gamma_1pct_bp": -3.0 if legs else -5.0,
                                    "hedge_into_fixing": hedge, "hedge_max": hedge + 0.5}
    rows = [_t6_row(T6_CAL[i], g, legs={"MO2507:C:5000": 1.0} if legs and i == 1 else None, phi=0.5 if legs and i == 1 else 1.0,
                    flags=("volume_cut",) if legs and i == 1 else ()) for i, g in enumerate(daily)]
    return {"events": cells, "life": life_stats(_t6_run(rows, note=(note[0], T6_CAL[0])), completed)}


def _t6_note_file(note, events, cell_fn, tids, cells_of, *, gaps=()):
    """A stored note file: ``cell_fn(tid, clock, config) -> keyword arguments of _t6_cell`` for every technique of every
    ``(config, clocks)`` of ``cells_of``."""
    from gpin.events import note_id

    return {"note": note_id(*note), "book": note[0], "inception": note[1].isoformat(), "until": None,
            "completed": True, "end_reason": "ko", "last_record": "2025-12-31",
            "events": {e.id: e.to_json() for e in events}, "gap_observations": list(gaps),
            "runs": {cfg: {clock: {tid: _t6_cell(note, events, **cell_fn(tid, clock, cfg)) for tid in tids}
                           for clock in clocks} for cfg, clocks in cells_of.items()}}


def _t6_selection():
    from gpin.measures import select_on_train

    return select_on_train(load_prereg(), {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(T6_KI_ES)}, _t6_costs())


def test_training_selection_reads_the_training_events_of_the_selection_books_under_the_primary_clock():
    from gpin.measures import select_on_train, training_selection

    prereg = load_prereg()
    tids = list(_registry(None))                                           # every grid point, no composite
    ko_note, ki_note, other = ("ki75", date(2024, 1, 2)), ("seas6", date(2024, 1, 2)), ("ki75", date(2024, 1, 3))
    days = [date(2025, 3, 3), date(2025, 3, 10), date(2025, 3, 17)]
    ko_events = [_t6_event("ko", d, note=ko_note) for d in days]
    ki_events = [_t6_event("ki", d, start=d, note=ki_note) for d in days]
    stray = [_t6_event("ki", days[0], start=days[0], note=other)]            # a ki75 approach: not the KI selection book

    def cells(es_by_id, events, wild=False):
        def fn(tid, clock, cfg):
            v = es_by_id.get(tid, es_by_id["base"])
            if clock != "close" or wild:
                v = 1.0e6 if tid == "ko_shift@0.01" else -1.0e6           # another clock or book must not be read
            cost = {"ko_spread@0.015": 3.0, "ko_spread@0.03": 2.0}.get(tid, 1.0)
            return dict(pnl={e.id: v + 10.0 * k for k, e in enumerate(events)}, daily=[0.0, 1000.0], cost=cost)
        return fn

    both = {T6_CFG: ("close", "twap")}
    notes = [_t6_note_file(ko_note, ko_events, cells(T6_KO_ES, ko_events), tids, both),
             _t6_note_file(ki_note, ki_events, cells(T6_KI_ES, ki_events), tids, both),
             _t6_note_file(other, stray, cells(T6_KI_ES, stray, wild=True), tids, both)]
    got = training_selection(notes, prereg)
    want = select_on_train(prereg, {"ko": _t6_windows(T6_KO_ES), "ki": _t6_windows(T6_KI_ES)}, _t6_costs())
    assert got["composites"] == want["composites"] and got["base"] == want["base"] and got["empty"] == []
    assert {b: {r: p["id"] for r, p in rows.items()} for b, rows in got["picks"].items()} == {
        b: {r: p["id"] for r, p in rows.items()} for b, rows in want["picks"].items()}
    assert got["picks"]["ko"]["ko_spread"]["cost_bp"] == pytest.approx(2.0)   # the mean event-window cost breaks the tie
    assert got["events"] == {"ko": {"n": 3, "weeks": 3, "book": "ki75"}, "ki": {"n": 3, "weeks": 3, "book": "seas6"}}
    # a test-dated event in a training file is refused outright
    late_event = _t6_event("ko", date(2025, 7, 2), note=ko_note, slice_="test")
    late = _t6_note_file(ko_note, [late_event], lambda tid, clock, cfg: dict(pnl={late_event.id: 0.0}, daily=[0.0, 1.0]),
                         tids, both)
    with pytest.raises(ValidationError):
        training_selection([late, notes[1]], prereg)
    # no training event of a barrier type: fails closed unless the run is a smoke run on a partial set of states
    with pytest.raises(StudyDataError):
        training_selection(notes[:1], prereg)
    assert training_selection(notes[:1], prereg, allow_empty=True)["empty"] == ["ki"]


def _t6_book():
    """Stored test-slice files of a hand-made sample: 12 ``ki75`` pin notes with two KO pin events each (24 events in
    12 weeks; event i loses the baseline ``-100 + 10 i`` bp), 2 ``ki75`` headline notes without events, 24 ``seas6``
    bookings with one approach each, and two ``ki75`` approaches (the transfer check)."""
    from gpin.measures import evaluated_ids

    prereg, sel = load_prereg(), _t6_selection()
    tids = evaluated_ids(prereg, sel)
    monday = date(2025, 7, 7)

    def day_of(i):
        return monday + timedelta(days=7 * (i % 12) + i // 12)

    pins = [("ki75", d) for d in _weekdays(12, date(2024, 1, 2))]
    heads = [("ki75", date(2023, 10, 9)), ("ki75", date(2023, 11, 1))]
    seas = [("seas6", date(2024, 2, 5) + timedelta(days=7 * j)) for j in range(24)]
    index, files = {}, []
    cells_of = {T6_CFG: ("close", "twap", "next_open"), T6_CFG_DESK: ("close",)}

    def gain(tid, clock, cfg, kind, i):
        if kind == "ki":
            return 30.0 if tid == "ki_cap@1.0" else 0.0
        if tid == "ko_spread@0.03":                                        # the whole gain sits in one week
            return {0: 30.0, 12: 30.0}.get(i, 0.0)
        if tid == "derisk@5" and clock == "next_open":
            return 0.0
        if tid == "mo_ko@1,scenario":
            return 60.0 if cfg == T6_CFG_DESK else 30.0
        return 30.0 if tid in ("surv_hold@1", "derisk@5", "freeze@1", "ko_wrong", "sop_futures") else 0.0

    def cell_fn(note, events, daily, wild=1.0):
        def fn(tid, clock, cfg):
            pnl = {}
            for e in events:
                kind, i = index[e.id]
                pnl[e.id] = (-5000.0 if e.slice == "train" else -100.0 + 10.0 * i + gain(tid, clock, cfg, kind, i))
            scale = wild * (2.0 if tid == "freeze@1" else 1.0)
            out = dict(pnl=pnl, daily=[daily[0]] + [scale * g for g in daily[1:]],
                       left_out=[e.id for e in events if clock == "next_open" and index[e.id] == ("ko", 23)],
                       hedge=0.8 if tid == "surv_hold@1" else 1.5, legs=tid.startswith("mo_ko"))
            if tid == "mo_ko@1,scenario":
                out["flags"] = {"overlay_no_strike": 1}
            return out
        return fn

    for j, note in enumerate(pins):
        events = [_t6_event("ko", day_of(i), note=note, slice_="test", hit=i % 2 == 0) for i in (2 * j, 2 * j + 1)]
        for e, i in zip(events, (2 * j, 2 * j + 1)):
            index[e.id] = ("ko", i)
        train = _t6_event("ko", date(2025, 3, 3), note=note)               # a training event: not a test number
        index[train.id] = ("ko", 0)
        events.append(train)
        if j < 2:
            approach = _t6_event("ki", day_of(j), start=day_of(j), note=note, slice_="test")
            index[approach.id] = ("ki", j)
            events.append(approach)
        gaps = [{"obs": day_of(2 * j).isoformat(), "date": "2025-06-20", "gap_ko_bp": 400.0 if j else -3.0, "ko_dir": -1}]
        files.append(_t6_note_file(note, events, cell_fn(note, events, [50000.0, 3000.0, -1000.0], wild=100.0), tids,
                                   cells_of, gaps=gaps))
    for note in heads:
        files.append(_t6_note_file(note, [], cell_fn(note, [], [50000.0, 3000.0, -1000.0, 2000.0, -2500.0]), tids,
                                   cells_of))
    for i, note in enumerate(seas):
        e = _t6_event("ki", day_of(i), start=day_of(i), note=note, slice_="test", hit=i % 3 == 0)
        index[e.id] = ("ki", i)
        gaps = [{"obs": "2025-08-01", "date": "2025-07-20", "gap_ko_bp": 0.0, "ko_dir": 0}] if i == 0 else []
        files.append(_t6_note_file(note, [e], cell_fn(note, [e], [-80000.0, 1000.0, -1000.0]), tids, cells_of, gaps=gaps))
    sample = {"books": {
        "ki75": {"headline": [n[1].isoformat() for n in heads], "pin": [{"inception": n[1].isoformat()} for n in pins]},
        "seas6": {"headline": [n[1].isoformat() for n in seas], "pin": []}}}
    last_ko = next(eid for eid, v in index.items() if v == ("ko", 23))
    return files, sample, sel, tids, last_ko


@pytest.fixture(scope="module")
def t6_tables():
    from gpin.measures import tables

    files, sample, sel, tids, last_ko = _t6_book()
    return tables(files, load_prereg(), sample, sel, slice_="test"), tids, last_ko


def test_tables_give_the_ko_verdict_on_test_events_with_the_whole_life_of_the_headline_notes(t6_tables):
    from s2ts.measures import holm

    out, tids, _ = t6_tables
    assert (out["view"], out["slice"], out["clocks"]) == ("seller", "test", ["close", "twap", "next_open"])
    assert out["primary"] == {"clock": "close", "config": T6_CFG}
    assert out["notes"] == {"n": 38, "ki75": 14, "seas6": 24}
    t = out["tables"]["close"]["ko"]
    assert (t["book"], t["barrier"], t["slice"], t["n_events"], t["n_weeks"], t["decidable"]) == (
        "ki75", "ko", "test", 24, 12, True)
    assert (t["whole_life"], t["left_out"]) == ({"set": "headline", "n_notes": 2}, [])
    rows = t["rows"]
    assert "ki_shift@-0.02" not in rows and "ki_wrong" not in rows and "mo_ki@2,payoff" not in rows
    assert list(rows) == t["order"] == [tid for tid in tids if tid in rows] and len(rows) == 16 and out["order"] == tids
    win = rows["surv_hold@1"]
    assert win["es10"] == pytest.approx(-60.0) and win["es10_base"] == pytest.approx(-90.0) and win["p"] == 0.0
    assert (win["eligible"], win["tail_pass"], win["life_pass"], win["win"], win["robust"]) == (True,) * 5
    assert (win["row"], win["role"], win["clips"], win["mean_cost_bp"]) == ("surv_hold", "candidate", 24, 0.5)
    # the whole life is the two headline notes': the pin notes' hundred-fold daily P&L is not in it
    daily = [_bp(g) for g in (3000.0, -1000.0, 2000.0, -2500.0)] * 2
    assert win["std_bp"] == pytest.approx(float(np.std(daily, ddof=1)), rel=1e-10) and win["std_ratio"] == pytest.approx(1.0)
    assert win["mean_terminal_bp"] == pytest.approx(_bp(51500.0), rel=1e-12) and win["n_completed"] == 2
    # passes under close and twap but not under next_open: a win that is not robust
    assert (rows["derisk@5"]["win"], rows["derisk@5"]["robust"]) == (True, False)
    assert out["tables"]["next_open"]["ko"]["rows"]["derisk@5"]["tail_pass"] is False
    # the tail passes, the daily std is twice the baseline's: fails on the whole life
    fz = rows["freeze@1"]
    assert (fz["tail_pass"], fz["std_pass"], fz["life_pass"], fz["win"], fz["robust"]) == (True, False, False, False, False)
    assert fz["std_ratio"] == pytest.approx(2.0)
    # same events as the baseline: fails on the tail
    assert (rows["ko_shift@0.02"]["tail_pass"], rows["ko_shift@0.02"]["win"]) == (False, False)
    # controls, cost rows, the baseline and the other barrier's pick of a row are reported, never eligible
    for tid in ("base", "base_noclip", "ko_wrong", "band@0.02", "band@0.05", "secant@0.03"):
        assert (rows[tid]["eligible"], rows[tid]["win"], rows[tid]["robust"], rows[tid]["p_holm"]) == (False, None, None, None)
    assert rows["ko_wrong"]["tail_pass"] is True and rows["ko_wrong"]["p"] == 0.0
    # Holm across the ten eligible rows, beside the raw p
    eligible = [tid for tid, r in rows.items() if r["eligible"]]
    assert eligible == ["ko_shift@0.02", "ko_spread@0.03", "ko_bend@0.015,3", "secant@0.05", "surv_hold@1", "derisk@5",
                        "freeze@1", "mo_ko@1,scenario", "sop_futures", "sop_overlay"]
    assert [rows[tid]["p_holm"] for tid in eligible] == holm([rows[tid]["p"] for tid in eligible])
    spread = rows["ko_spread@0.03"]
    assert 0.25 < spread["p"] < 0.45 and spread["p_holm"] > spread["p"] and spread["win"] is False
    assert t["n_eligible"] == 10 and t["wins"] == ["surv_hold@1", "derisk@5", "mo_ko@1,scenario", "sop_futures"]


def test_tables_leave_out_the_last_day_fixing_under_next_open_and_filter_the_slice(t6_tables):
    out, _, last_ko = t6_tables
    t = out["tables"]["next_open"]["ko"]
    assert (t["n_events"], t["n_weeks"], t["left_out"]) == (23, 12, [last_ko])
    # without event 23 (+130 for the baseline) nothing changes in the tail of 23 events: k = ceil(2.3) = 3
    assert t["rows"]["surv_hold@1"]["es10"] == pytest.approx(-60.0) and t["rows"]["base"]["worst"] == -100.0
    assert out["tables"]["twap"]["ko"]["n_events"] == 24
    # the training events of the same files (-5000 bp each) are in no test table
    assert all(r["worst"] > -200.0 for c in out["tables"].values() for r in c["ko"]["rows"].values())
    assert out["tables"]["close"]["ko"]["rows"]["base"]["robust"] is None
    assert "robust" not in out["tables"]["twap"]["ko"]["rows"]["surv_hold@1"]            # only the primary clock's table


def test_tables_give_the_ki_verdict_on_the_seasoned_book_and_the_ki75_transfer_without_a_verdict(t6_tables):
    out, _, _ = t6_tables
    t = out["tables"]["close"]["ki"]
    assert (t["book"], t["barrier"], t["n_events"], t["n_weeks"], t["decidable"]) == ("seas6", "ki", 24, 12, True)
    assert t["whole_life"] == {"set": "headline", "n_notes": 24}
    rows = t["rows"]
    assert "surv_hold@1" not in rows and "mo_ko@1,scenario" not in rows and "ko_wrong" not in rows
    assert [tid for tid, r in rows.items() if r["eligible"]] == [
        "ki_shift@-0.02", "secant@0.03", "ki_cap@1.0", "mo_ki@2,payoff", "sop_futures", "sop_overlay"]
    assert (rows["ki_cap@1.0"]["win"], rows["ki_cap@1.0"]["robust"], rows["ki_shift@-0.02"]["win"]) == (True, True, False)
    assert rows["secant@0.05"]["eligible"] is False and rows["ki_wrong"]["win"] is None
    # the same picks on the 2Y product's own approaches: measures only
    tr = out["tables"]["close"]["ki_transfer"]
    assert (tr["book"], tr["n_events"], tr["n_weeks"], tr["verdict"]) == ("ki75", 2, 2, False)
    assert set(tr["rows"]) == set(rows)
    cap = tr["rows"]["ki_cap@1.0"]
    assert (cap["es10"], cap["es10_base"], cap["worst"], cap["worst_base"]) == (-70.0, -100.0, -70.0, -100.0)
    assert not {"win", "p", "p_holm", "tail_pass", "life_pass", "robust", "decidable"} & set(cap)


def test_tables_carry_the_sensitivities_the_pin_set_the_gamma_carry_and_the_descriptive_extras(t6_tables):
    out, tids, _ = t6_tables
    sens = out["sensitivities"]
    assert set(sens) == {T6_CFG_DESK} and set(sens[T6_CFG_DESK]) == {"close"}
    assert sens[T6_CFG_DESK]["close"]["ko"]["rows"]["mo_ko@1,scenario"]["es10"] == pytest.approx(-30.0)
    assert out["tables"]["close"]["ko"]["rows"]["mo_ko@1,scenario"]["es10"] == pytest.approx(-60.0)
    # the pin set's whole life is printed separately and decides nothing
    pin = out["whole_life_pin"]["close"]
    assert set(pin) == set(tids) and pin["base"]["n_notes"] == 12
    assert pin["base"]["std_bp"] == pytest.approx(float(np.std([_bp(3.0e5), _bp(-1.0e5)] * 12, ddof=1)), rel=1e-10)
    # gamma carry on the KO test events, per technique
    g = out["gamma_carry"]["close"]
    assert g["base"] == {"n_events": 24, "mean_note_gamma": -10.0, "mean_legs_gamma": 0.0, "mean_net_gamma": -10.0,
                         "mean_net_gamma_1pct_bp": -5.0, "mean_hedge_into_fixing": 1.5, "mean_hedge_max": 2.0,
                         "max_hedge_max": 2.0}
    assert g["surv_hold@1"]["mean_hedge_into_fixing"] == pytest.approx(0.8)
    assert g["mo_ko@1,scenario"]["mean_net_gamma"] == pytest.approx(-6.0)
    assert g["mo_ko@1,scenario"]["mean_legs_gamma"] == pytest.approx(4.0)
    assert out["gamma_carry"]["next_open"]["base"]["n_events"] == 23
    x = out["extras"]
    assert x["gap_observations"] == {"ki75": {"n": 12, "positive": 11, "negative": 1, "zero": 0, "non_positive": 1},
                                     "seas6": {"n": 1, "positive": 0, "negative": 0, "zero": 1, "non_positive": 1}}
    assert x["left_out"] == {"close": {"ko": 0, "ki": 0}, "twap": {"ko": 0, "ki": 0}, "next_open": {"ko": 1, "ki": 0}}
    ov = x["overlay"]["close"]
    assert set(ov) == {"mo_ko@1,scenario"}                                     # the only technique that held a leg
    assert ov["mo_ko@1,scenario"] == {"leg_days": 38, "overlay_days": 38, "phi_lt1_days": 38, "phi_mean": 0.5,
                                      "phi_min": 0.5, "flags": {"volume_cut": 38}}
    assert x["flags"]["close"]["mo_ko@1,scenario"] == {"volume_cut": 38} and x["flags"]["close"]["base"] == {}
    assert x["event_flags"]["close"]["ko"]["mo_ko@1,scenario"] == {"overlay_no_strike": 24}
    band = x["band"]["ki75"]
    assert set(band) == {"base", "band@0.02", "band@0.05"} and band["base"]["n_notes"] == 2
    assert band["band@0.02"]["mean_cost_bp"] == pytest.approx(0.0) and "mean_im_lots_traded" in band["band@0.05"]
    assert out["techniques"]["secant@0.05"] == {"row": "secant", "family": "2 futures-only", "role": "candidate",
                                                "barrier": "both", "eligible": ["ko"]}
    assert out["techniques"]["sop_futures"]["eligible"] == ["ko", "ki"] and out["techniques"]["ko_wrong"]["eligible"] == []
    text = json.dumps(out, allow_nan=False)
    assert "holder" not in text.lower()


def test_tables_fail_closed_on_files_that_do_not_hold_the_same_runs():
    from gpin.measures import tables

    files, sample, sel, tids, _ = _t6_book()
    del files[3]["runs"][T6_CFG]["close"]["freeze@1"]
    with pytest.raises(ValidationError):
        tables(files, load_prereg(), sample, sel, slice_="test")
    with pytest.raises(ValidationError):
        tables([], load_prereg(), sample, sel, slice_="test")


# ---- the ledger driver: slice separation and the hash guard (D11, AC12), on the hand-made market in tmp dirs ----

import pickle  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

T6_NOTE_IDS = ["ki75:2025-06-02", "seas6:2025-06-02"]
T6_FP = "f" * 64                                   # the pricing fingerprint of the hand-made state files


def _t6_stamp(files, states=None):
    """The current four hashes of the hand-made driver inputs."""
    from gpin import measures

    states = states or files.states
    return measures.stamp(files.events, files.board, files.chain,
                          {nid: states / (nid.replace(":", "__") + ".json") for nid in T6_NOTE_IDS})


@pytest.fixture(scope="module")
def stage4():
    return _load_stage("04_ledger.py")


@pytest.fixture(scope="module")
def t6_files(tmp_path_factory, t6_market):
    """The hand-made market as the driver's inputs: a states directory (with the junk a real one holds), the events
    file and the board and chain pickles."""
    root = tmp_path_factory.mktemp("t6_driver")
    ko_states, ki_states, boards, chains = t6_market
    states = root / "states"
    states.mkdir()
    # as the pricing pass writes them: each state file names its note and carries the pricing fingerprint
    (states / "ki75__2025-06-02.json").write_text(json.dumps(dict(ko_states, note="ki75:2025-06-02", fingerprint=T6_FP)))
    (states / "seas6__2025-06-02.json").write_text(json.dumps(dict(ki_states, note="seas6:2025-06-02", fingerprint=T6_FP)))
    (states / ".lock").write_text("1")
    (states / "ki75__2025-06-03.error.json").write_text("{}")
    events = root / "events.json"
    events.write_text(json.dumps(_t6_sample()))
    with open(root / "board.pkl", "wb") as fh:
        pickle.dump(boards, fh)
    with open(root / "chain.pkl", "wb") as fh:
        pickle.dump(chains, fh)
    return SimpleNamespace(root=root, states=states, events=events, board=root / "board.pkl", chain=root / "chain.pkl")


def _t6_argv(files, slice_, out, *extra, states=None, events=None):
    return ["--slice", slice_, "--states-dir", str(states or files.states), "--out-dir", str(out),
            "--events", str(events or files.events), "--board", str(files.board), "--chain", str(files.chain),
            "--workers", "1", *extra]


class _LedgerSpy:
    """Wraps ``gpin.measures.run_technique``: what ``until`` every run was given and the last date it produced."""

    def __init__(self):
        from gpin import measures

        self.measures, self.real, self.calls = measures, measures.run_technique, []

    def __enter__(self):
        def spy(states, technique, boards, chains, **kw):
            run = self.real(states, technique, boards, chains, **kw)
            self.calls.append((technique.id, kw["mode"], kw["until"], max(r["date"] for r in run.rows)))
            return run

        self.measures.run_technique = spy
        return self

    def __exit__(self, *exc):
        self.measures.run_technique = self.real


@pytest.fixture(scope="module")
def t6_train(stage4, t6_files):
    out = t6_files.root / "out"
    tracked = [STUDY_DIR / "results" / name for name in ("selection.json", "backtest.json")]
    before = [p.exists() for p in tracked]
    with _LedgerSpy() as spy:
        rc = stage4.main(_t6_argv(t6_files, "train", out))
    assert [p.exists() for p in tracked] == before                     # the tracked results are not this run's
    return SimpleNamespace(out=out, rc=rc, calls=spy.calls)


def test_the_train_slice_writes_the_selection_with_the_picks_and_the_four_hashes(t6_train, t6_files):
    from gpin import measures
    from gpin.techniques import build_registry, composite_selection

    assert t6_train.rc == 0
    path = t6_train.out / "selection.json"                               # the selection follows --out-dir
    sel = json.loads(path.read_text())
    current = _t6_stamp(t6_files)
    assert sorted(current) == ["code_hash", "events_hash", "inputs_hash", "prereg_hash"]
    assert {k: sel[k] for k in current} == current
    assert current["prereg_hash"] == prereg_hash() and current["code_hash"] == measures.code_hash()
    assert (sel["view"], sel["slice"], sel["split_date"], sel["until"], sel["clock"], sel["config"]) == (
        "seller", "train", "2025-07-01", "2025-06-30", "close", "desk500_cost0.2")
    assert (sel["partial"], sel["empty"], sel["notes"]) == (False, [], T6_NOTE_IDS)
    assert sel["events"] == {"ko": {"n": 1, "weeks": 1, "book": "ki75"}, "ki": {"n": 1, "weeks": 1, "book": "seas6"}}
    # the file is written with sorted keys: the rows' registered order is kept beside the picks
    assert sel["rows"] == {"ko": ["ko_shift", "ko_spread", "ko_bend", "secant", "surv_hold", "derisk", "freeze", "mo_ko"],
                           "ki": ["ki_shift", "secant", "ki_cap", "mo_ki"]}
    assert {b: sorted(rows) for b, rows in sel["picks"].items()} == {b: sorted(rows) for b, rows in sel["rows"].items()}
    for barrier, rows in sel["picks"].items():
        for row, pick in rows.items():
            # the highest training ES10 of the row (grid points equal to rounding are a tie: lower cost, then order)
            assert pick["id"] in pick["grid"]
            assert pick["es10"] == pytest.approx(max(g["es10"] for g in pick["grid"].values()), rel=1e-9)
            assert pick["n_events"] == 1 and isinstance(pick["beats_base"], bool)
    # the composites block is what the registry reads
    assert composite_selection(path) == sel["composites"]
    reg = build_registry(load_prereg(), composite_selection(path))
    assert reg["sop_futures"].params == sel["composites"]["sop_futures"]
    assert sel["composites"]["sop_overlay"] == {"ko": sel["picks"]["ko"]["mo_ko"]["id"],
                                                "ki": sel["picks"]["ki"]["mo_ki"]["id"]}
    assert "holder" not in path.read_text().lower()


def test_the_train_slice_writes_no_row_dated_on_or_after_the_split(t6_train):
    # every ledger run was cut at the last trading day before the split, and none produced a later row
    assert len(t6_train.calls) == 2 * 33 * 3                             # 2 notes x (31 grid points + 2 composites) x 3 clocks
    assert {c[2] for c in t6_train.calls} == {T6_UNTIL}
    assert max(c[3] for c in t6_train.calls) == "2025-06-30"
    assert {c[0] for c in t6_train.calls} >= {"sop_futures", "sop_overlay", "base", "band@0.05"}
    runs = t6_train.out / "runs" / "train"
    names = sorted(p.name for p in runs.iterdir())
    assert names == ["ki75__2025-06-02.json", "manifest.json", "seas6__2025-06-02.json", "tables.json"]
    assert not (t6_train.out / "runs" / "test").exists() and not (t6_train.out / "backtest.json").exists()
    for p in runs.iterdir():                                             # no date of the test slice anywhere in the files
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", p.read_text())
        assert dates and max(dates) <= ("2025-07-01" if p.name in ("manifest.json", "tables.json") else "2025-06-30"), p.name
    manifest = json.loads((runs / "manifest.json").read_text())
    assert (manifest["slice"], manifest["until"], manifest["split_date"], manifest["partial"]) == (
        "train", "2025-06-30", "2025-07-01", False)
    assert manifest["notes"] == T6_NOTE_IDS and manifest["configs"] == {"desk500_cost0.2": ["close", "twap", "next_open"]}
    assert len(manifest["techniques"]) == 33 and manifest["techniques"][-2:] == ["sop_futures", "sop_overlay"]
    note = json.loads((runs / "ki75__2025-06-02.json").read_text())
    assert (note["slice"], note["until"], note["completed"]) == ("train", "2025-06-30", False)
    assert (note["end_reason"], note["last_record"]) == (None, "2025-06-30")   # how the note ends later is not written
    assert list(note["events"]) == ["ko|ki75:2025-06-02|2025-06-17"]     # the 07-15 observation is not in a training file
    assert note["stamp"] == manifest["stamp"]
    cell = note["runs"]["desk500_cost0.2"]["close"]["sop_overlay"]
    assert cell["life"]["last"] == "2025-06-30" and set(cell["events"]) == set(note["events"])
    tables = json.loads((runs / "tables.json").read_text())
    assert (tables["slice"], tables["tables"]["close"]["ko"]["n_events"], tables["tables"]["close"]["ki"]["n_events"]) == (
        "train", 1, 1)
    assert len(tables["tables"]["close"]["ko"]["rows"]) == 2 + 3 + 2 + 3 + 1 + 2 + 2 + 2 + 2 + 3 + 2 + 2


def _t6_tampered(t6_train, tmp_path, **changes):
    sel = json.loads((t6_train.out / "selection.json").read_text())
    sel.update(changes)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(sel))
    return path


def test_the_test_slice_refuses_without_a_selection_and_with_each_hash_wrong(stage4, t6_files, t6_train, tmp_path,
                                                                              capsys, monkeypatch):
    def refused(argv, word):
        out = Path(argv[argv.index("--out-dir") + 1])
        with _LedgerSpy() as spy:
            rc = stage4.main(argv)
        err = capsys.readouterr().err
        assert rc != 0 and "REFUSED" in err and word in err, err
        assert spy.calls == [] and not out.exists()                      # no ledger run, nothing written
        return err

    refused(_t6_argv(t6_files, "test", tmp_path / "a"), "does not exist")
    names = ("the pre-registration", "the events file", "the ledger-side code", "the recorded inputs")
    for key, word in zip(("prereg_hash", "events_hash", "code_hash", "inputs_hash"), names):
        sel = _t6_tampered(t6_train, tmp_path, **{key: "0" * 64})
        err = refused(_t6_argv(t6_files, "test", tmp_path / "b", "--selection", str(sel)), word)
        assert sum(w in err for w in names) == 1
    # the files themselves changed after the selection was made: the events file ...
    good = str(t6_train.out / "selection.json")
    events = tmp_path / "events.json"
    events.write_text(t6_files.events.read_text() + "\n")
    refused(_t6_argv(t6_files, "test", tmp_path / "c", "--selection", good, events=events), "the events file")
    # ... the ledger-side code, the pre-registration
    from gpin import measures

    with monkeypatch.context() as m:
        m.setattr(measures, "code_hash", lambda directory=None: "1" * 64)
        refused(_t6_argv(t6_files, "test", tmp_path / "d", "--selection", good), "the ledger-side code")
    with monkeypatch.context() as m:
        m.setattr(measures, "prereg_hash", lambda path=None: "2" * 64)
        refused(_t6_argv(t6_files, "test", tmp_path / "e", "--selection", good), "the pre-registration")


def test_the_test_slice_exits_non_zero_from_the_command_line_without_a_selection(t6_files, tmp_path):
    import os

    env = dict(os.environ, PYTHONPATH=str(STUDY_DIR.parents[2]))
    done = subprocess.run([sys.executable, str(STUDY_DIR / "04_ledger.py"), *_t6_argv(t6_files, "test", tmp_path / "out")],
                          capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 2 and "REFUSED" in done.stderr and "does not exist" in done.stderr
    assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def t6_test(stage4, t6_files, t6_train):
    """The test slice on the hand-made market, with both sensitivities, into its own out-dir (the selection is the
    training run's)."""
    out = t6_files.root / "out_test"
    with _LedgerSpy() as spy:
        rc = stage4.main(_t6_argv(t6_files, "test", out, "--selection", str(t6_train.out / "selection.json"),
                                  "--desk-notional", "5e8", "1e8", "--mo-cost-pt", "0.2", "1.0"))
    return SimpleNamespace(out=out, rc=rc, calls=spy.calls)


def test_the_test_slice_runs_the_selected_techniques_over_the_full_data_and_writes_the_backtest(t6_test, t6_train):
    from gpin.measures import evaluated_ids

    assert t6_test.rc == 0
    sel = json.loads((t6_train.out / "selection.json").read_text())
    ids = evaluated_ids(load_prereg(), sel)
    assert {c[2] for c in t6_test.calls} == {None}                       # no cut: the full data
    assert {c[0] for c in t6_test.calls} == set(ids)
    assert len(t6_test.calls) == 2 * len(ids) * (3 + 2)                  # three clocks, and two sensitivities under close
    back = json.loads((t6_test.out / "backtest.json").read_text())       # the payload follows --out-dir too
    assert (back["view"], back["slice"], back["clocks"], back["partial"]) == ("seller", "test",
                                                                             ["close", "twap", "next_open"], False)
    assert back["stamp"] == {k: sel[k] for k in ("prereg_hash", "events_hash", "code_hash", "inputs_hash")}
    assert back["selection"]["composites"] == sel["composites"]
    assert back["selection"]["picks"] == {b: {row: p["id"] for row, p in rows.items()} for b, rows in sel["picks"].items()}
    assert set(back["sensitivities"]) == {"desk100_cost0.2", "desk500_cost1"}
    for clock in back["clocks"]:
        for barrier, book in (("ko", "ki75"), ("ki", "seas6")):
            t = back["tables"][clock][barrier]
            assert (t["book"], t["slice"], t["n_events"], t["decidable"], t["wins"]) == (book, "test", 1, False, [])
            assert all(r["win"] is None and r["es10"] is not None for r in t["rows"].values())
    ko = back["tables"]["close"]["ko"]
    assert back["order"] == ids and ko["order"] == [tid for tid in ids if tid in ko["rows"]] and len(ko["order"]) == len(ko["rows"])
    # the redeemed 2Y note: one KO test event, a fired one, with its gamma carry
    assert back["gamma_carry"]["close"]["base"]["n_events"] == 1
    assert back["extras"]["gap_observations"]["ki75"] == {"n": 2, "positive": 2, "negative": 0, "zero": 0, "non_positive": 0}
    text = (t6_test.out / "backtest.json").read_text()
    assert "holder" not in text.lower() and "NaN" not in text
    runs = t6_test.out / "runs" / "test"
    manifest = json.loads((runs / "manifest.json").read_text())
    assert (manifest["slice"], manifest["until"], manifest["techniques"]) == ("test", None, ids)
    assert manifest["configs"] == {"desk500_cost0.2": ["close", "twap", "next_open"], "desk100_cost0.2": ["close"],
                                   "desk500_cost1": ["close"]}
    note = json.loads((runs / "ki75__2025-06-02.json").read_text())
    assert (note["completed"], note["end_reason"], len(note["events"])) == (True, "ko", 2)
    assert note["runs"]["desk500_cost0.2"]["next_open"]["base"]["life"]["last"] == "2025-07-16"
    assert "holder" not in json.dumps(note).lower()
    # the training run's files are untouched by the test slice
    assert not (t6_test.out / "runs" / "train").exists() and not (t6_test.out / "selection.json").exists()


def test_a_partial_set_of_states_needs_the_flag_marks_the_selection_and_is_refused_by_a_full_test_slice(
        stage4, t6_files, tmp_path, capsys):
    states = tmp_path / "states"
    states.mkdir()
    shutil.copy(t6_files.states / "seas6__2025-06-02.json", states)
    out = tmp_path / "out"
    assert stage4.main(_t6_argv(t6_files, "train", out, states=states)) != 0      # a sample note has no state file
    assert "ki75:2025-06-02" in capsys.readouterr().err and not out.exists()
    assert stage4.main(_t6_argv(t6_files, "train", out, "--partial", "--mode", "close", states=states)) == 0
    sel = json.loads((out / "selection.json").read_text())
    assert (sel["partial"], sel["empty"], sel["notes"]) == (True, ["ko"], ["seas6:2025-06-02"])
    assert sel["picks"]["ko"]["ko_shift"]["es10"] is None and sel["picks"]["ki"]["ki_cap"]["es10"] is not None
    assert sel["picks"]["ko"]["ko_shift"]["id"] is None and sel["composites"] == {}       # no pick, no composite
    train_manifest = json.loads((out / "runs" / "train" / "manifest.json").read_text())
    assert train_manifest["configs"] == {"desk500_cost0.2": ["close"]}
    assert len(train_manifest["techniques"]) == 31 and "sop_futures" not in train_manifest["techniques"]
    capsys.readouterr()
    assert stage4.main(_t6_argv(t6_files, "test", out, "--mode", "close", states=states)) != 0
    assert "partial" in capsys.readouterr().err and not (out / "backtest.json").exists()
    assert stage4.main(_t6_argv(t6_files, "test", out, "--partial", "--mode", "close", states=states)) == 0
    back = json.loads((out / "backtest.json").read_text())
    assert (back["partial"], back["clocks"], back["notes"]["n"]) == (True, ["close"], 1)
    assert back["tables"]["close"]["ko"]["n_events"] == 0 and back["tables"]["close"]["ko"]["rows"]["base"]["es10"] is None
    # the test slice of a partial selection runs what was picked: no KO row's grid point, no composite
    ran = json.loads((out / "runs" / "test" / "manifest.json").read_text())["techniques"]
    ki = sel["picks"]["ki"]
    assert ran == ["base", "base_noclip", ki["ki_shift"]["id"], "ko_wrong", "ki_wrong", ki["secant"]["id"],
                   ki["ki_cap"]["id"], ki["mo_ki"]["id"], "band@0.02", "band@0.05"]
    assert back["selection"]["composites"] == {} and back["selection"]["picks"]["ko"]["ko_shift"] is None
    ko_table = back["tables"]["close"]["ko"]
    assert ko_table["n_eligible"] == 0 and sorted(ko_table["rows"]) == sorted(
        ["base", "base_noclip", "ko_wrong", sel["picks"]["ki"]["secant"]["id"], "band@0.02", "band@0.05"])
    assert back["tables"]["close"]["ki"]["n_eligible"] == 4                                  # four KI rows, no composite
    # every note is there but one state file is a short diagnostic one (as --quick writes): its later events cannot be
    # measured. A full run refuses; a smoke run leaves those events out and says how many
    short = json.loads((t6_files.states / "ki75__2025-06-02.json").read_text())
    short["days"] = short["days"][:8]
    (states / "ki75__2025-06-02.json").write_text(json.dumps(short))
    out2 = tmp_path / "out2"
    capsys.readouterr()
    assert stage4.main(_t6_argv(t6_files, "train", out2, "--mode", "close", states=states)) != 0
    assert "last recorded day" in capsys.readouterr().err and not out2.exists()
    assert stage4.main(_t6_argv(t6_files, "train", out2, "--partial", "--mode", "close", states=states)) == 0
    sel = json.loads((out2 / "selection.json").read_text())
    assert (sel["partial"], sel["empty"], sel["notes"]) == (True, ["ko"], T6_NOTE_IDS)
    manifest = json.loads((out2 / "runs" / "train" / "manifest.json").read_text())
    assert manifest["events_not_measurable"] == {"ki75:2025-06-02": 1}


def test_the_train_slice_runs_only_the_notes_booked_before_the_split(stage4, t6_files, t6_market, tmp_path):
    """A note booked in the test slice has no training-dated row: the training slice does not run it."""
    ko_states, _, _, _ = t6_market
    states = tmp_path / "states"
    shutil.copytree(t6_files.states, states)
    late = dict(json.loads(json.dumps(ko_states)), fingerprint=T6_FP)
    late["days"] = [dict(r, next_ko=None, days_to_ko=None, gap_ko=None, ko_dir=None) for r in late["days"][21:31]]
    late["inception"] = late["days"][0]["date"]
    assert late["inception"] == "2025-07-01"
    (states / "ki75__2025-07-01.json").write_text(json.dumps(late))
    sample = _t6_sample()
    sample["books"]["ki75"]["notes"]["2025-07-01"] = dict(sample["books"]["ki75"]["notes"]["2025-06-02"])
    events = tmp_path / "events.json"
    events.write_text(json.dumps(sample))
    out = tmp_path / "out"
    with _LedgerSpy() as spy:
        assert stage4.main(_t6_argv(t6_files, "train", out, "--mode", "close", states=states, events=events)) == 0
    manifest = json.loads((out / "runs" / "train" / "manifest.json").read_text())
    assert manifest["notes"] == T6_NOTE_IDS and manifest["partial"] is False
    assert not (out / "runs" / "train" / "ki75__2025-07-01.json").exists()
    assert len(spy.calls) == 2 * 33 and max(c[3] for c in spy.calls) == "2025-06-30"
    assert json.loads((out / "selection.json").read_text())["notes"] == T6_NOTE_IDS


def test_the_driver_resolves_its_paths_and_never_sends_a_partial_run_to_the_tracked_results(stage4, tmp_path):
    from gpin import OUTPUT_DIR, RESULTS_DIR, STATES_DIR

    plain = stage4.resolve(stage4.parse_args(["--slice", "train"]))
    assert (plain.states_dir, plain.runs_dir) == (STATES_DIR, OUTPUT_DIR / "runs" / "train")
    assert (plain.selection, plain.backtest) == (RESULTS_DIR / "selection.json", RESULTS_DIR / "backtest.json")
    assert (plain.workers, plain.modes, plain.partial) == (4, ("close", "twap", "next_open"), False)
    scratch = stage4.resolve(stage4.parse_args(["--slice", "test", "--out-dir", str(tmp_path), "--mode", "twap"]))
    assert (scratch.runs_dir, scratch.selection, scratch.backtest) == (
        tmp_path / "runs" / "test", tmp_path / "selection.json", tmp_path / "backtest.json")
    assert scratch.modes == ("close", "twap")                            # the primary clock always runs
    named = stage4.resolve(stage4.parse_args(["--slice", "test", "--out-dir", str(tmp_path), "--selection", "x.json",
                                              "--backtest", "y.json", "--workers", "9"]))
    assert (named.selection, named.backtest, named.workers) == (Path("x.json"), Path("y.json"), 4)   # at most 4 workers
    with pytest.raises(StudyDataError):
        stage4.resolve(stage4.parse_args(["--slice", "train", "--partial"]))
    with pytest.raises(StudyDataError):
        stage4.resolve(stage4.parse_args(["--slice", "test", "--partial", "--out-dir", str(tmp_path),
                                          "--backtest", str(RESULTS_DIR / "backtest.json")]))


def test_the_bootstrap_defaults_are_the_pre_registered_replicates_and_seed():
    import inspect
    from gpin.measures import cluster_bootstrap, default_rules

    params = inspect.signature(cluster_bootstrap).parameters
    stats = load_prereg()["statistics"]
    assert (params["n"].default, params["seed"].default) == (stats["bootstrap_n"], stats["seed"]) == (10_000, 20261112)
    assert (default_rules().bootstrap_n, default_rules().seed, default_rules().es_alpha) == (10_000, 20261112, 0.10)


# ---------------------------------------------------------------------------
# CP2 fixes — ledger side
# ---------------------------------------------------------------------------

# ---- the fill ratio of a refusal day is the last sized one, not 1 ----

PHI_SELECTION = {"sop_futures": {"ko": "ko_shift@0.02", "ki": "ki_cap@1.25"},
                 "sop_overlay": {"ko": "mo_ko@1,payoff", "ki": "mo_ki@2,payoff"}}
PHI_LAST = date(2024, 1, 19)                                               # MO2401's last trading day
PHI_SPOT, PHI_FORWARD, PHI_IM = 4990.0, 4980.0, 4970.0


def _phi_world(thin_days):
    """Six closes at spot 4990 under the KO barrier 5000, the observation five closes after day 1; every KO model is
    recorded (d(true) = 1000, d(ko_s200) = 700) and the gap is 400,000: from day 1 the overlay wants 40 lots of the 4900
    / 5000 call spread. 6000 lots trade in every option, so the 500 mio desk (50 x the note) fills 12 lots a day, and
    its series limit is 24 lots per side. On ``thin_days`` the 5000 call trades 50 lots: it may not be opened."""
    days = _weekdays(6)
    specs = []
    for i, day in enumerate(days):
        mo = _ov_mo(day, AC8_KO_LASTS, OV_STRIKES, PHI_FORWARD, volume=6000)
        if i in thin_days:
            q = mo["MO2401"]["quotes"]
            q[("C", 5000.0)] = (q[("C", 5000.0)][0], 50, 10000)
        specs.append(dict(day=day, spot=PHI_SPOT, n=6 - i, gap_ko=400e3, ko_dir=-1, deltas=dict(KO_DELTAS),
                          im={"IM2403": PHI_IM}, mo=mo))
    return _t5_world(specs), days


def _phi_im_lots(day, lots, phi, futures=700.0, true=1000.0):
    """IM lots of ``phi x d(true) + (1 - phi) x futures rule - delta of `lots` call spreads held``."""
    legs = (_leg_delta(day, PHI_LAST, "C", 4900.0, 0.2, lots, PHI_SPOT, PHI_FORWARD)
            + _leg_delta(day, PHI_LAST, "C", 5000.0, 0.2, -lots, PHI_SPOT, PHI_FORWARD))
    return (phi * true + (1.0 - phi) * futures - legs) * PHI_SPOT / (200.0 * PHI_IM)


def test_a_refusal_day_with_a_partly_filled_overlay_keeps_the_last_sized_fill_ratio():
    sop = _registry(PHI_SELECTION)["sop_overlay"]
    (states, boards, chains), d = _phi_world(thin_days={2})
    rows = _run(states, sop, boards, chains).rows
    spread = lambda lots: {"MO2401:C:4900": pytest.approx(lots), "MO2401:C:5000": pytest.approx(-lots)}   # noqa: E731
    # day 0: six closes ahead, no overlay yet: the futures rule alone (ko_shift@0.02: 700)
    assert rows[0]["legs"] == {} and rows[0]["phi"] == 1.0
    assert rows[0]["im_lots"] == pytest.approx(700.0 * PHI_SPOT / (200.0 * PHI_IM), rel=1e-12)
    # day 1: 12 of the 40 lots fill (a tenth of 6000 lots at the desk's 50 x): phi = 0.3
    assert rows[1]["legs"] == spread(12.0) and rows[1]["phi"] == pytest.approx(0.3) and "volume_cut" in rows[1]["flags"]
    assert rows[1]["im_lots"] == pytest.approx(_phi_im_lots(d[1], 12.0, 0.3), rel=1e-9)
    # day 2: the overlay is refused (the 5000 call trades 50 lots); the 12 lots held stay. They still cover 30% of the
    # digital: the IM target is 0.3 x d(true) + 0.7 x the futures rule - the held legs' delta, and the row says 0.3
    assert "overlay_not_openable" in rows[2]["flags"] and rows[2]["legs"] == spread(12.0) and rows[2]["unfilled"] == {}
    assert rows[2]["phi"] == pytest.approx(0.3)
    assert rows[2]["im_lots"] == pytest.approx(_phi_im_lots(d[2], 12.0, 0.3), rel=1e-9)
    assert rows[2]["im_lots"] != pytest.approx(_phi_im_lots(d[2], 12.0, 1.0), rel=1e-3)    # not the whole note at d(true)
    # day 3: a normal sized day again: 12 more lots fill and phi is recomputed (24 of 40)
    assert "overlay_not_openable" not in rows[3]["flags"] and rows[3]["legs"] == spread(24.0)
    assert rows[3]["phi"] == pytest.approx(0.6)
    assert rows[3]["im_lots"] == pytest.approx(_phi_im_lots(d[3], 24.0, 0.6), rel=1e-9)
    _check_identity(_run(states, sop, boards, chains))


def test_a_refusal_keeps_the_latest_sized_fill_ratio_and_nothing_held_keeps_the_present_rule():
    sop = _registry(PHI_SELECTION)["sop_overlay"]
    # refused on day 1 with nothing held (nothing was ever sized): no leg is wanted, phi is 1 and the futures rule is all
    (states, boards, chains), d = _phi_world(thin_days={1, 4})
    rows = _run(states, sop, boards, chains).rows
    assert "overlay_not_openable" in rows[1]["flags"] and rows[1]["legs"] == {} and rows[1]["phi"] == 1.0
    assert rows[1]["im_lots"] == pytest.approx(700.0 * PHI_SPOT / (200.0 * PHI_IM), rel=1e-12)
    # days 2 and 3 fill 12 lots each (phi 0.3, then 0.6); the refusal of day 4 carries the latest, 0.6
    assert (rows[2]["phi"], rows[3]["phi"]) == (pytest.approx(0.3), pytest.approx(0.6))
    assert "overlay_not_openable" in rows[4]["flags"] and rows[4]["phi"] == pytest.approx(0.6)
    assert rows[4]["im_lots"] == pytest.approx(_phi_im_lots(d[4], 24.0, 0.6), rel=1e-9)
    # the same under next_open on days that are not fixings
    late = _run(states, sop, boards, chains, "next_open").rows
    assert [r["phi"] for r in late[:5]] == [r["phi"] for r in rows[:5]]
    assert [r["im_lots"] for r in late[:5]] == [r["im_lots"] for r in rows[:5]]
    # an overlay alone (both note deltas are d(true)) is hedged the same whatever phi says; its row carries it too
    solo = _run(states, _registry()["mo_ko@1,payoff"], boards, chains).rows
    assert solo[4]["phi"] == pytest.approx(0.6)
    assert solo[4]["im_lots"] == pytest.approx(_phi_im_lots(d[4], 24.0, 0.6, futures=1000.0), rel=1e-9)


def test_a_closed_overlay_clears_the_carried_fill_ratio_so_a_residual_leg_covers_nothing_on_a_later_refusal():
    from gpin.techniques import Target

    n = 5
    d = _weekdays(n)
    last = d[0] + timedelta(days=60)
    volumes = [6000, 6000, 500, 6000, 6000]                   # day 2: a close is cut to 50 desk lots = 1 lot per note
    quotes = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, volumes[i], 10000)} for i in range(n)]
    (states, boards, chains), d = _leg_world(n, last, quotes)

    class _Closing:
        """Sizes an overlay on day 1, closes it on day 2, and on day 3 is refused a new one while a residual is held."""

        clip = True
        id, family, params = "closing", "test", {}

        def target(self, ctx):
            if ctx.day_index == 1:
                return Target(700.0, {KEY_C5000: 40.0}, 1000.0)
            if ctx.day_index == 3:
                return Target(700.0, dict(ctx.held_legs), 1000.0, ("overlay_not_openable",))
            return Target(700.0, {})

    rows = _run(states, _Closing(), boards, chains).rows
    assert rows[1]["legs"] == {STR_C5000: pytest.approx(12.0)} and rows[1]["phi"] == pytest.approx(0.3)
    # day 2: the technique wants no leg (no refusal): the overlay is closed; the thin market leaves 11 lots on the book
    assert rows[2]["legs"] == {STR_C5000: pytest.approx(11.0)} and "volume_cut" in rows[2]["flags"]
    # day 3: a refusal with that residual held. It was sized for an overlay the technique has closed, so it covers
    # nothing of the digital now: phi is 0 (not the stale 0.3, not 1) and the whole note follows the futures rule;
    # the residual's own delta is still hedged
    assert "overlay_not_openable" in rows[3]["flags"] and rows[3]["legs"] == {STR_C5000: pytest.approx(11.0)}
    assert rows[3]["phi"] == 0.0
    assert rows[3]["im_lots"] == pytest.approx((700.0 - _leg_delta(d[3], last, "C", 5000.0, 0.2, 11.0)) / 200.0, rel=1e-9)
    _check_identity(_run(states, _Closing(), boards, chains))


# ---- the flatten at the data end and at `until` is a mark, not a trade: it costs nothing ----

@pytest.mark.parametrize("mode", ["close", "twap", "next_open"])
def test_the_flatten_at_the_data_end_and_at_until_costs_nothing_and_keeps_the_identity(mode):
    n = 4
    d = _weekdays(n)
    last = d[0] + timedelta(days=60)
    quotes = [{("C", 5000.0): (_px(d[i], last, "C", 5000.0, 0.2), None, 5000, 10000)} for i in range(n)]
    (states, boards, chains), d = _leg_world(n, last, quotes)
    tech = _Scripted(lambda ctx: {KEY_C5000: 2.0} if ctx.day_index >= 1 else {})
    for until, flag, k in ((None, "data_end", 3), (d[2], "until", 2)):
        run = _run(states, tech, boards, chains, mode, desk_notional=NOTIONAL_, until=until)
        end, before = run.rows[k], run.rows[k - 1]
        assert len(run.rows) == k + 1 and flag in end["flags"]
        assert before["legs"] == {STR_C5000: 2.0} and before["im_lots"] != 0.0     # an IM hedge and a leg are carried in
        assert end["legs"] == {} and end["im_lots"] == 0.0 and end["cost"] == 0.0  # marked flat, nothing charged
        # the leg is marked out at the day's settlement: its value comes back as cash
        s_end = quotes[k][("C", 5000.0)][0]
        assert end["cash"] == pytest.approx(before["cash"] + end["accrual"] + end["im_vm"] + 2.0 * s_end * 100.0, abs=CASH_TOL)
        assert end["hedged"] == pytest.approx(end["note_pnl"] + end["im_vm"] + end["mo_mtm"] + end["accrual"], abs=CASH_TOL)
        assert sum(r["cost"] for r in run.rows[:k]) > 0.0                           # the trades before it do cost
        _check_identity(run)


# ---- the note's book and KI multiple come from the state file, never from a default ----

@pytest.mark.parametrize("missing", ["book", "ki_pct"])
def test_a_state_without_its_book_or_its_ki_multiple_fails_closed(missing):
    from gpin.techniques import Base

    states, boards, chains = _world_a()
    del states[missing]
    with pytest.raises(StudyDataError, match=missing):
        _run(states, Base(), boards, chains)


# ---- no inline numeric tolerance in the ledger-side code ----

def test_a_bump_must_be_a_ladder_step_by_the_repos_comparison():
    import re
    from gpin.techniques import hedge_delta

    state = _ladder_state(100.0, 1234.0, 0.0, 5000.0)
    assert hedge_delta(state, 5000.0, bump=0.01) == pytest.approx(1234.0, rel=1e-12)
    assert hedge_delta(state, 5000.0, bump=0.03 * (1.0 + 1e-13)) == pytest.approx(1234.0, rel=1e-12)   # rounding noise
    # half a billionth of an index fraction off a 1% step is not the step (relative 5e-8, the repo's tolerance is 1e-9)
    with pytest.raises(ValidationError):
        hedge_delta(state, 5000.0, bump=0.01 + 5e-10)
    with pytest.raises(ValidationError):
        hedge_delta(state, 5000.0, bump=0.015)
    # and no tolerance literal is written inline in the technique or the ledger code
    for name in ("techniques.py", "ledger.py"):
        source = (STUDY_DIR / "gpin" / name).read_text(encoding="utf-8")
        assert re.findall(r"\b\d+(?:\.\d+)?[eE]-\d+\b", source) == [], name


# ---------------------------------------------------------------------------
# CP3 fix round — T6 findings
# ---------------------------------------------------------------------------

# ---- finding 4: the vectorised tie is the repo's symmetric is_close ----

#: A pair found by search: |a - b| is inside 1e-9 x max(|a|, |b|) (the repo's is_close) and outside 1e-9 x |b|
#: (numpy.isclose with b as its reference). The two bounds are 1e-18 relative apart, below the spacing of doubles, so
#: such pairs are rare: about one in a hundred thousand candidates one ulp either side of the bound.
CP3_PAIR = (1.9489436749377653, 1.9489436729888217)


def test_the_vectorised_tie_is_the_repos_symmetric_is_close_element_by_element():
    from quantark.util.numerical import is_close
    from gpin.measures import close_array

    a, b = CP3_PAIR
    assert is_close(a, b) and is_close(b, a) and not bool(np.isclose(a, b, rtol=1e-9, atol=0.0))   # the forms differ here
    left = np.array([a, b, 0.0, 0.0, 5.0, -3.0, 1.0e6, -250.0, 1.0, 1.0e-300])
    right = np.array([b, a, 0.0, 1.0e-12, 5.0 * (1.0 + 5.0e-10), -3.0 * (1.0 + 2.0e-9), 1.0e6 + 1.0e-4, 250.0, 1.0 + 1.1e-9,
                      0.0])
    got = close_array(left, right)
    assert got.dtype == bool and got.tolist() == [is_close(float(x), float(y)) for x, y in zip(left, right)]
    assert got.tolist() == [True, True, True, False, True, False, True, False, False, False]
    assert close_array(right, left).tolist() == got.tolist()                                    # symmetric


def test_cluster_bootstrap_ties_with_the_symmetric_form():
    """One event: every replicate is that event. The technique's ES is above the baseline's by less than the repo's
    relative tolerance of the LARGER of the two: a tie, so no gain on any replicate."""
    from gpin.measures import cluster_bootstrap

    a, b = CP3_PAIR
    weeks = {"e": "2025-W30"}
    assert cluster_bootstrap({"e": a}, {"e": b}, weeks, n=50) == 1.0
    assert cluster_bootstrap({"e": b}, {"e": a}, weeks, n=50) == 1.0
    assert cluster_bootstrap({"e": a * (1.0 + 1.1e-9)}, {"e": a}, weeks, n=50) == 0.0            # beyond the tolerance: a gain


# ---- finding 5: a barrier without a training event has no pick and no composite ----


def test_a_selection_with_an_empty_barrier_has_no_pick_for_it_and_is_evaluated_without_it():
    from gpin.measures import evaluated_ids, select_on_train, selection_composites
    from gpin.techniques import build_registry

    prereg = load_prereg()
    windows = {"ko": {tid: {} for tid in T6_KO_ES}, "ki": _t6_windows(T6_KI_ES)}
    partial = select_on_train(prereg, windows, _t6_costs(), allow_empty=True)
    # no composite can be built, so the registry is built without them rather than with an invented member
    assert selection_composites(partial) is None
    assert not {"sop_futures", "sop_overlay"} & set(build_registry(prereg, selection_composites(partial)))
    assert evaluated_ids(prereg, partial) == [
        "base", "base_noclip", "ki_shift@-0.02", "ko_wrong", "ki_wrong", "secant@0.03", "ki_cap@1.0", "mo_ki@2,payoff",
        "band@0.02", "band@0.05"]
    full = _t6_selection()
    assert selection_composites(full) == full["composites"] and len(evaluated_ids(prereg, full)) == 20
    # a block that names only one of the two composites is not usable either (the registry wants both or none)
    assert selection_composites({"composites": {"sop_futures": full["composites"]["sop_futures"]}}) is None
    # a selection that is not partial never has a row without a pick
    broken = json.loads(json.dumps(full))
    broken["picks"]["ko"]["freeze"]["id"] = None
    with pytest.raises(ValidationError):
        evaluated_ids(prereg, broken)


QUICK_STATES = gpin.OUTPUT_DIR / "states_quick"


@pytest.mark.skipif(not (QUICK_STATES.is_dir() and gpin.BOARD_PATH.exists() and gpin.CHAIN_PATH.exists()),
                    reason="the quick states, the board or the chain are absent")
def test_the_train_slice_on_the_quick_states_records_no_ko_pick_and_no_composite(stage4, tmp_path):
    """``states_quick`` holds three 31-day notes: one seasoned approach, no KO event. The training slice of such a
    smoke run must not record a KO grid point as if it had been picked, nor composites built on one."""
    from gpin.measures import evaluated_ids, selection_composites
    from gpin.techniques import build_registry, composite_selection

    out = tmp_path / "out"
    rc = stage4.main(["--slice", "train", "--partial", "--mode", "close", "--workers", "1",
                      "--states-dir", str(QUICK_STATES), "--out-dir", str(out)])
    assert rc == 0
    sel = json.loads((out / "selection.json").read_text())
    assert (sel["partial"], sel["empty"], sel["until"]) == (True, ["ko"], "2025-06-30")
    assert sel["events"]["ko"]["n"] == 0 and sel["events"]["ki"]["n"] == 1
    assert {row: p["id"] for row, p in sel["picks"]["ko"].items()} == dict.fromkeys(sel["rows"]["ko"])
    assert all(p["id"] in p["grid"] and p["es10"] is not None for p in sel["picks"]["ki"].values())
    assert sel["composites"] == {} and composite_selection(out / "selection.json") == {}
    assert selection_composites(sel) is None
    prereg = load_prereg()
    manifest = json.loads((out / "runs" / "train" / "manifest.json").read_text())
    assert manifest["techniques"] == list(build_registry(prereg)) and len(manifest["techniques"]) == 31
    ids = evaluated_ids(prereg, sel)
    assert not [tid for tid in ids if tid.split("@")[0] in sel["rows"]["ko"] and tid.split("@")[0] != "secant"]
    assert "sop_futures" not in ids and "sop_overlay" not in ids and {"base", "ki_wrong", "band@0.05"} <= set(ids)
    tables = json.loads((out / "runs" / "train" / "tables.json").read_text())
    assert tables["tables"]["close"]["ko"]["n_eligible"] == 0 and tables["tables"]["close"]["ki"]["n_eligible"] == 4
    assert max(re.findall(r"\d{4}-\d{2}-\d{2}", (out / "runs" / "train" / "ki75__2024-02-29.json").read_text())) < "2025-07-01"


# ---- finding 1: what enters the training selection is decided in the hashed code, not in the driver ----


def test_training_scope_is_the_last_chain_day_before_the_split_the_notes_booked_by_then_and_the_training_slice(t6_market):
    from gpin.measures import TrainingScope, training_scope

    _, _, _, chains = t6_market                                           # 2025-06-02 .. 2025-07-25, weekdays
    prereg = load_prereg()
    notes = ["ki75:2025-06-02", "seas6:2025-06-30", "ki75:2025-07-01", "seas6:2025-07-07", "ki75:2025-06-16"]
    scope = training_scope(chains, notes, prereg)
    assert scope == TrainingScope(until=date(2025, 6, 30), notes=("ki75:2025-06-02", "seas6:2025-06-30", "ki75:2025-06-16"),
                                  event_slice="train", split_date=date(2025, 7, 1))
    # the cut-off is a day of the chain: without 06-30 and 06-27 on it, the last day before the split is 06-26
    thin = [c for c in chains if c.day not in (date(2025, 6, 30), date(2025, 6, 27))]
    assert training_scope(thin, notes, prereg).until == date(2025, 6, 26)
    assert training_scope(thin, notes, prereg).notes == ("ki75:2025-06-02", "ki75:2025-06-16")
    # the split and the slice are the pre-registration's
    moved = dict(prereg, split_date="2025-06-18", selection=dict(prereg["selection"], slice="blind"))
    assert training_scope(chains, notes, moved) == TrainingScope(date(2025, 6, 17), ("ki75:2025-06-02", "ki75:2025-06-16"),
                                                                 "blind", date(2025, 6, 18))
    with pytest.raises(StudyDataError, match="no day before the split"):
        training_scope([c for c in chains if c.day >= date(2025, 7, 1)], notes, prereg)
    with pytest.raises(StudyDataError, match="booked"):
        training_scope(chains, ["ki75:2025-07-01"], prereg)


def test_the_train_driver_takes_its_cut_off_its_notes_and_its_event_slice_from_training_scope(
        stage4, t6_files, tmp_path, monkeypatch, capsys):
    from gpin import measures

    real = measures.training_scope
    seen = []

    def scoped(**changes):
        def fake(chains, notes, prereg):
            seen.append(list(notes))
            return measures.TrainingScope(**{**real(chains, notes, prereg)._asdict(), **changes})
        return fake

    # an earlier cut-off: every run stops there and the files say so
    monkeypatch.setattr(measures, "training_scope", scoped(until=date(2025, 6, 23)))
    out = tmp_path / "a"
    with _LedgerSpy() as spy:
        assert stage4.main(_t6_argv(t6_files, "train", out, "--mode", "close")) == 0
    assert seen[-1] == T6_NOTE_IDS                                         # it was asked about every state file found
    assert {c[2] for c in spy.calls} == {date(2025, 6, 23)} and max(c[3] for c in spy.calls) == "2025-06-23"
    sel = json.loads((out / "selection.json").read_text())
    manifest = json.loads((out / "runs" / "train" / "manifest.json").read_text())
    assert (sel["until"], manifest["until"], sel["slice"], sel["split_date"]) == ("2025-06-23", "2025-06-23", "train",
                                                                              "2025-07-01")
    # fewer notes: only those it names are run, though both state files are there
    monkeypatch.setattr(measures, "training_scope", scoped(notes=("seas6:2025-06-02",)))
    out = tmp_path / "b"
    with _LedgerSpy() as spy:
        assert stage4.main(_t6_argv(t6_files, "train", out, "--mode", "close", "--partial")) == 0
    assert json.loads((out / "runs" / "train" / "manifest.json").read_text())["notes"] == ["seas6:2025-06-02"]
    assert len(spy.calls) == 31 and not (out / "runs" / "train" / "ki75__2025-06-02.json").exists()
    # another event slice: the driver hands those events to the run, which refuses anything dated after its cut-off
    monkeypatch.setattr(measures, "training_scope", scoped(event_slice="test"))
    out = tmp_path / "c"
    capsys.readouterr()
    assert stage4.main(_t6_argv(t6_files, "train", out, "--mode", "close")) != 0
    assert "cannot be measured on a run that stops there" in capsys.readouterr().err and not out.exists()
    # a slice no event belongs to: nothing is measured in either pass, so (as a smoke run) no barrier has a pick
    monkeypatch.setattr(measures, "training_scope", scoped(event_slice="none"))
    out = tmp_path / "d"
    assert stage4.main(_t6_argv(t6_files, "train", out, "--mode", "close", "--partial")) == 0
    sel = json.loads((out / "selection.json").read_text())
    assert (sel["slice"], sel["empty"], sel["composites"]) == ("none", ["ko", "ki"], {})
    assert json.loads((out / "runs" / "train" / "ki75__2025-06-02.json").read_text())["events"] == {}
    assert json.loads((out / "runs" / "train" / "tables.json").read_text())["slice"] == "none"
    # and the driver has no split-date logic of its own left
    src = (STUDY_DIR / "04_ledger.py").read_text()
    assert "SPLIT_DATE" not in src and "slice_of" not in src and ".day <" not in src and "training_scope(" in src


# ---- finding 2: the fourth hash, over the recorded inputs (board, chain, the state files' pricing fingerprints) ----


def test_inputs_hash_is_over_the_board_and_chain_digests_and_the_sorted_state_fingerprints(tmp_path):
    import hashlib
    from gpin import measures

    board, chain = tmp_path / "board.pkl", tmp_path / "chain.pkl"
    board.write_bytes(b"board bytes")
    chain.write_bytes(b"chain bytes")
    fps = {"seas6:2024-01-08": "b" * 64, "ki75:2024-02-29": "a" * 64}
    canonical = json.dumps({"board": hashlib.sha256(b"board bytes").hexdigest(),
                            "chain": hashlib.sha256(b"chain bytes").hexdigest(),
                            "states": [["ki75:2024-02-29", "a" * 64], ["seas6:2024-01-08", "b" * 64]]},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    want = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert measures.inputs_hash(board, chain, fps) == want
    assert measures.inputs_hash(board, chain, dict(reversed(list(fps.items())))) == want       # the order does not matter
    assert measures.inputs_hash(board, chain, {**fps, "ki75:2024-02-29": "c" * 64}) != want    # a re-priced note
    assert measures.inputs_hash(board, chain, {"ki75:2024-02-29": "a" * 64}) != want           # another set of notes
    chain.write_bytes(b"chain bytes.")
    assert measures.inputs_hash(board, chain, fps) != want
    chain.write_bytes(b"chain bytes")
    board.write_bytes(b"board bytes.")
    assert measures.inputs_hash(board, chain, fps) != want
    with pytest.raises(StudyDataError):
        measures.inputs_hash(tmp_path / "absent.pkl", chain, fps)


def test_state_fingerprints_are_read_from_the_state_files_and_fail_closed(tmp_path):
    from gpin import measures

    a, b = tmp_path / "ki75__2024-02-29.json", tmp_path / "seas6__2024-01-08.json"
    a.write_text(json.dumps({"note": "ki75:2024-02-29", "fingerprint": "a" * 64, "days": []}))
    b.write_text(json.dumps({"note": "seas6:2024-01-08", "fingerprint": "b" * 64, "days": []}))
    files = {"ki75:2024-02-29": a, "seas6:2024-01-08": b}
    assert measures.state_fingerprints(files) == {"ki75:2024-02-29": "a" * 64, "seas6:2024-01-08": "b" * 64}
    b.write_text(json.dumps({"note": "seas6:2024-01-08", "days": []}))                         # not stamped
    with pytest.raises(StudyDataError, match="fingerprint"):
        measures.state_fingerprints(files)
    b.write_text(json.dumps({"note": "seas6:2024-01-15", "fingerprint": "b" * 64, "days": []}))  # another note's file
    with pytest.raises(StudyDataError, match="seas6:2024-01-15"):
        measures.state_fingerprints(files)


def test_the_stamp_is_four_hashes(t6_files):
    from gpin import measures

    files = {nid: t6_files.states / (nid.replace(":", "__") + ".json") for nid in T6_NOTE_IDS}
    got = measures.stamp(t6_files.events, t6_files.board, t6_files.chain, files)
    assert got == {"prereg_hash": prereg_hash(), "events_hash": measures.file_hash(t6_files.events),
                   "code_hash": measures.code_hash(),
                   "inputs_hash": measures.inputs_hash(t6_files.board, t6_files.chain, dict.fromkeys(T6_NOTE_IDS, T6_FP))}
    assert measures.CODE_FILES == ("black.py", "ledger.py", "techniques.py", "measures.py")    # the code hash is unchanged


def _t6_copy_inputs(t6_files, tmp_path, *, board=None, chain=None, fingerprint=None):
    """The driver's inputs with one of them changed: a byte appended to the board or the chain file, or another
    pricing fingerprint in one state file."""
    states = tmp_path / "states"
    shutil.copytree(t6_files.states, states)
    files = SimpleNamespace(root=tmp_path, states=states, events=t6_files.events, board=t6_files.board, chain=t6_files.chain)
    for name, change in (("board", board), ("chain", chain)):
        if change:
            path = tmp_path / f"{name}.pkl"
            path.write_bytes(getattr(t6_files, name).read_bytes() + b"\n")
            setattr(files, name, path)
    if fingerprint:
        path = states / "seas6__2025-06-02.json"
        path.write_text(json.dumps(dict(json.loads(path.read_text()), fingerprint=fingerprint)))
    return files


@pytest.mark.parametrize("change", [dict(board=True), dict(chain=True), dict(fingerprint="e" * 64)],
                         ids=["board", "chain", "state fingerprint"])
def test_the_test_slice_refuses_when_a_recorded_input_changed_after_the_selection(stage4, t6_files, t6_train, tmp_path,
                                                                                  capsys, change):
    """01_data.py or 03_pricing.py re-run after the selection was made: the selection must be made again."""
    files = _t6_copy_inputs(t6_files, tmp_path, **change)
    out = tmp_path / "out"
    capsys.readouterr()
    with _LedgerSpy() as spy:
        rc = stage4.main(_t6_argv(files, "test", out, "--selection", str(t6_train.out / "selection.json")))
    err = capsys.readouterr().err
    assert rc != 0 and "REFUSED" in err and "the recorded inputs" in err and "inputs_hash" in err, err
    assert sum(w in err for w in ("the pre-registration", "the events file", "the ledger-side code")) == 0
    assert spy.calls == [] and not out.exists()                             # no ledger run, nothing created
    # the same copy with nothing changed is accepted (the guard reads content, not paths)
    same = _t6_copy_inputs(t6_files, tmp_path / "same")
    assert _t6_stamp(same) == _t6_stamp(t6_files) and _t6_stamp(files) != _t6_stamp(t6_files)
    assert {k for k in _t6_stamp(files) if _t6_stamp(files)[k] != _t6_stamp(t6_files)[k]} == {"inputs_hash"}


def test_the_fourth_hash_is_in_every_file_and_the_inputs_are_hashed_once_per_run(stage4, t6_files, t6_train, t6_test,
                                                                                 tmp_path, monkeypatch):
    from gpin import measures

    current = _t6_stamp(t6_files)
    sel = json.loads((t6_train.out / "selection.json").read_text())
    assert sel["inputs_hash"] == current["inputs_hash"]
    for run in (t6_train.out / "runs" / "train", t6_test.out / "runs" / "test"):
        for name in ("manifest.json", "ki75__2025-06-02.json", "seas6__2025-06-02.json"):
            assert json.loads((run / name).read_text())["stamp"] == current, (run.name, name)
    assert json.loads((t6_train.out / "runs" / "train" / "tables.json").read_text())["stamp"] == current
    assert json.loads((t6_test.out / "backtest.json").read_text())["stamp"] == current
    # one run: the board, the chain and the events file are each hashed once and every state fingerprint read once,
    # however many notes and techniques run (the per-note jobs do not hash anything)
    hashed, read = [], []
    real_hash, real_fps = measures.file_hash, measures.state_fingerprints
    monkeypatch.setattr(measures, "file_hash", lambda path: hashed.append(Path(path).name) or real_hash(path))
    monkeypatch.setattr(measures, "state_fingerprints", lambda files: read.append(sorted(files)) or real_fps(files))
    assert stage4.main(_t6_argv(t6_files, "train", tmp_path / "out", "--mode", "close")) == 0
    assert sorted(hashed) == ["board.pkl", "chain.pkl", "events.json"] and read == [T6_NOTE_IDS]


# ---- finding 3: one stamp across the note files, and a stored payload checked against the current stamp ----


def test_tables_refuse_note_files_that_do_not_carry_one_stamp():
    from gpin.measures import tables

    files, sample, sel, _, _ = _t6_book()
    prereg = load_prereg()
    stamp = {"prereg_hash": "p" * 64, "events_hash": "e" * 64, "code_hash": "c" * 64, "inputs_hash": "i" * 64}
    for f in files:
        f["stamp"] = dict(stamp)
    assert tables(files, prereg, sample, sel, slice_="test")["stamp"] == stamp            # one stamp: carried through
    files[5]["stamp"] = dict(stamp, code_hash="d" * 64)                                   # a file of another run
    with pytest.raises(ValidationError) as err:
        tables(files, prereg, sample, sel, slice_="test")
    assert files[0]["note"] in str(err.value) and files[5]["note"] in str(err.value) and "stamp" in str(err.value)
    del files[5]["stamp"]                                                                 # a file without one, beside stamped ones
    with pytest.raises(ValidationError, match="stamp"):
        tables(files, prereg, sample, sel, slice_="test")
    for f in files:                                                                       # none stamped (in memory): allowed
        f.pop("stamp", None)
    assert "stamp" not in tables(files, prereg, sample, sel, slice_="test")


def test_check_payload_refuses_a_stored_payload_or_manifest_that_is_not_the_current_runs(t6_files, t6_train, t6_test):
    from gpin.measures import PayloadRefused, check_payload

    current = _t6_stamp(t6_files)
    sel = json.loads((t6_train.out / "selection.json").read_text())
    back = json.loads((t6_test.out / "backtest.json").read_text())
    manifest = json.loads((t6_test.out / "runs" / "test" / "manifest.json").read_text())
    train_manifest = json.loads((t6_train.out / "runs" / "train" / "manifest.json").read_text())
    for payload in (back, manifest, train_manifest):
        assert check_payload(payload, sel, current) is None                               # the run's own files pass
    names = {"prereg_hash": "the pre-registration", "events_hash": "the events file", "code_hash": "the ledger-side code",
             "inputs_hash": "the recorded inputs"}
    for key, word in names.items():
        # a stale payload or manifest: written under another stamp than the current files'
        for payload in (back, manifest):
            stale = dict(payload, stamp=dict(payload["stamp"], **{key: "0" * 64}))
            with pytest.raises(PayloadRefused, match=word) as err:
                check_payload(stale, sel, current)
            assert key in str(err.value) and sum(w in str(err.value) for w in names.values()) == 1
        # the files moved on after the payload was written
        with pytest.raises(PayloadRefused, match=word):
            check_payload(back, sel, dict(current, **{key: "1" * 64}))
        # the payload is not of this selection
        with pytest.raises(PayloadRefused, match="selection"):
            check_payload(back, dict(sel, **{key: "2" * 64}), current)
    with pytest.raises(PayloadRefused, match="no stamp"):
        check_payload({k: v for k, v in back.items() if k != "stamp"}, sel, current)
    # a payload whose picks or composites are not the selection's (written from another training run)
    other = json.loads(json.dumps(back))
    other["selection"]["composites"]["sop_futures"]["ko"] = "freeze@2"
    with pytest.raises(PayloadRefused, match="composites"):
        check_payload(other, sel, current)
    other = json.loads(json.dumps(back))
    other["selection"]["picks"]["ki"]["ki_cap"] = "ki_cap@none"
    with pytest.raises(PayloadRefused, match="picks"):
        check_payload(other, sel, current)
    assert issubclass(PayloadRefused, StudyDataError)


def test_training_selection_refuses_note_files_that_do_not_carry_one_stamp():
    from gpin.measures import training_selection

    prereg = load_prereg()
    tids = list(_registry(None))
    ko_note, ki_note = ("ki75", date(2024, 1, 2)), ("seas6", date(2024, 1, 2))
    day = date(2025, 3, 3)
    ko_e, ki_e = _t6_event("ko", day, note=ko_note), _t6_event("ki", day, start=day, note=ki_note)
    both = {T6_CFG: ("close",)}
    notes = [_t6_note_file(n, [e], lambda tid, clock, cfg, e=e: dict(pnl={e.id: -1.0}, daily=[0.0, 1.0]), tids, both)
             for n, e in ((ko_note, ko_e), (ki_note, ki_e))]
    notes[0]["stamp"], notes[1]["stamp"] = {"code_hash": "a"}, {"code_hash": "a"}
    assert training_selection(notes, prereg)["events"]["ko"]["n"] == 1
    notes[1]["stamp"] = {"code_hash": "b"}
    with pytest.raises(ValidationError, match="stamp"):
        training_selection(notes, prereg)


def test_measures_lists_the_samples_state_files_so_the_report_can_compute_the_current_stamp(t6_files, tmp_path):
    """Which state files a run reads is decided beside the stamp (the report needs the same list to check a stored
    payload): the sample's notes, selected by note id, every one of them unless the run is partial."""
    from gpin import measures

    sample = _t6_sample()
    files = measures.state_files(t6_files.states, sample, False)            # the lock and the error file are not notes
    assert files == {nid: t6_files.states / (nid.replace(":", "__") + ".json") for nid in T6_NOTE_IDS}
    assert list(files) == T6_NOTE_IDS                                        # books in registry order, by inception
    assert measures.stamp(t6_files.events, t6_files.board, t6_files.chain, files) == _t6_stamp(t6_files)
    some = tmp_path / "states"
    some.mkdir()
    shutil.copy(t6_files.states / "seas6__2025-06-02.json", some)
    (some / "ki75__2031-01-06.json").write_text("{}")                        # a state file of no note of the sample
    with pytest.raises(StudyDataError, match="ki75:2025-06-02"):
        measures.state_files(some, sample, False)
    assert list(measures.state_files(some, sample, True)) == ["seas6:2025-06-02"]
    with pytest.raises(StudyDataError, match="no state file"):
        measures.state_files(tmp_path / "absent", sample, True)
