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
    states = {"book": book, "ki_pct": 0.75, "inception": days[0]["day"].isoformat(), "coupon": coupon,
              "days": records, "meta": {}}
    return states, boards, chains


def test_day_context_is_frozen_and_checks_its_dates():
    from gpin.techniques import DayContext, NoteInfo

    states, boards, _ = _world([_day(WORLD_START, 5000.0, 1.0, 100.0, im={"IM2401": dict(close=5000.0)})])
    rec = states["days"][0]
    note = NoteInfo("ki75", WORLD_START, 0.1, 0.75, 1.0, 5000.0)
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
    note = NoteInfo("ki75", WORLD_START, 0.1, 0.75, 1.0, 5000.0)
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
    # day 2: the data end, the note is alive: 6 lots earn 6 x (-100) x 200, then everything is closed at the close
    accrual2 = r1["cash"] * RATE_ * dt
    cost2 = IM_COST * 6 * 5000.0 * IM_MULT_
    assert r2["im_vm"] == pytest.approx(-120000.0) and r2["note_pnl"] == pytest.approx(-105000.0 + 95000.0)
    assert r2["cost"] == pytest.approx(cost2) and r2["im_lots"] == 0.0 and r2["hedge_ratio"] == 0.0
    assert r2["hedged"] == pytest.approx(-10000.0 - 120000.0 + accrual2 - cost2)
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
                        max_days=AC10_DAYS)
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
