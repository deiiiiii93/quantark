"""2Y snowball flat-q vs CarryCurve case study (spec .ship/2026-11-05-snowball-2y-carry-curve).

Fixtures only (no live dashboard DB, no network): a sqlite built from the
2026-11-05 rows and a GBK excerpt of the CFFEX daily statistics CSV.
"""
from __future__ import annotations

import importlib.util
import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "example" / "case_studies" / "snowball_2y_carry_curve"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "snowball_2y_carry_curve"
CSV = FIXTURES / "cffex_20261105_excerpt.csv"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load("snowball_2y_cc_common", "_common.py")


# 2026-11-05 rows of the dashboard's latest_daily_basis view (columns the study reads)
DB_ROWS = [
    (1318449, 1, "IM2611", 6647.2, "2026-11-20", "edb3", None, None),
    (1318695, 2, "IM2612", 6605.0, "2026-12-18", "dc97", None, None),
    (1318941, 3, "IM2703", 6456.0, "2027-03-19", "1f8a", None, None),
    (1319187, 4, "IM2706", 6286.4, "2027-06-18", "0236", None, None),
    (1319397, 5, "SYN_FUT_20270917.SH", 6224.57713433221, "2027-09-17", "bbc2", "MO2709C6600", "MO2709P6600"),
]


def _make_db(path: Path, rows=DB_ROWS) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE latest_daily_basis (revision_id INTEGER, provider TEXT, index_code TEXT, "
        "futures_family TEXT, slot INTEGER, trading_date TEXT, spot REAL, contract_code TEXT, "
        "futures REAL, expiry_date TEXT, content_hash TEXT, call_option_contract TEXT, "
        "put_option_contract TEXT, observation_status TEXT)"
    )
    for rid, slot, code, fut, exp, h, c, p in rows:
        con.execute(
            "INSERT INTO latest_daily_basis VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, "akshare", "000852.SH", "IM", slot, "2026-11-05", 6650.1383, code, fut, exp, h, c, p, "close"),
        )
    con.commit()
    con.close()
    return path


# --------------------------------------------------------------------------- T3 snapshot (AC7)


def test_parse_cffex_csv_reads_close_and_settle(common):
    rows = common.parse_cffex_csv(CSV.read_bytes())
    assert rows["IM2611"] == {"close": 6647.2, "settle": 6624.2}
    assert rows["IM2706"] == {"close": 6286.4, "settle": 6266.8}
    assert rows["MO2709-C-6600"] == {"close": 464.6, "settle": 464.6}
    assert rows["MO2709-P-6600"] == {"close": 851.8, "settle": 840.0}


def test_read_dashboard_chain(common, tmp_path):
    chain = common.read_dashboard_chain(_make_db(tmp_path / "b.sqlite3"), "2026-11-05")
    assert chain["spot"] == 6650.1383
    assert [f["contract"] for f in chain["futures"]] == ["IM2611", "IM2612", "IM2703", "IM2706"]
    assert [f["price"] for f in chain["futures"]] == [6647.2, 6605.0, 6456.0, 6286.4]
    assert chain["futures"][0]["revision_id"] == 1318449
    assert chain["synthetic"]["call_contract"] == "MO2709C6600"
    assert chain["synthetic"]["expiry"] == "2027-09-17"


def test_snapshot_values_and_mo_cross_check(common, tmp_path):
    chain = common.read_dashboard_chain(_make_db(tmp_path / "b.sqlite3"), "2026-11-05")
    snap = common.build_snapshot(chain, CSV.read_bytes(), source_url="fixture")
    assert snap["valuation_date"] == "2026-11-05"
    assert snap["spot"] == 6650.1383
    pair = snap["option_pair"]
    assert (pair["strike"], pair["call_settle"], pair["put_settle"]) == (6600.0, 464.6, 840.0)
    assert (pair["call_close"], pair["put_close"]) == (464.6, 851.8)
    assert [f["settle"] for f in snap["futures"]] == [6624.2, 6582.4, 6434.6, 6266.8]
    T = (316) / 365.0
    f_mo = 6600.0 + (464.6 - 840.0) * math.exp(0.014 * T)
    assert snap["mo_forward"] == pytest.approx(f_mo, rel=1e-12)
    assert abs(f_mo / snap["dashboard_syn_fut"]["price"] - 1.0) < 0.005
    vol = snap["vol"]
    assert vol["strike"] == 6200.0                       # nearest grid strike to F ≈ 6220
    assert vol["value"] == pytest.approx(0.2762, abs=5e-4)
    assert vol["value"] == pytest.approx(0.5 * (vol["call_iv"] + vol["put_iv"]), rel=1e-12)
    assert len(snap["cffex_csv"]["sha256"]) == 64


def test_missing_rows_fail_closed(common, tmp_path):
    db = _make_db(tmp_path / "short.sqlite3", DB_ROWS[:3] + DB_ROWS[4:])
    with pytest.raises(common.StudyDataError):
        common.read_dashboard_chain(db, "2026-11-05")
    with pytest.raises(common.StudyDataError):
        common.read_dashboard_chain(_make_db(tmp_path / "b.sqlite3"), "2026-11-06")
    chain = common.read_dashboard_chain(_make_db(tmp_path / "c.sqlite3"), "2026-11-05")
    no_put = b"\r\n".join(
        line for line in CSV.read_bytes().split(b"\r\n") if not line.startswith(b"MO2709-P-6600")
    )
    with pytest.raises(common.StudyDataError):
        common.build_snapshot(chain, no_put, source_url="fixture")


def test_committed_snapshot_matches_fixture_values(common):
    snap = common.load_snapshot()
    assert snap["spot"] == 6650.1383
    assert [f["price"] for f in snap["futures"]] == [6647.2, 6605.0, 6456.0, 6286.4]
    assert snap["option_pair"]["call_settle"] == 464.6 and snap["option_pair"]["put_settle"] == 840.0


# --------------------------------------------------------------------------- T4 pricing (AC8, AC9)


def test_terms_schedule(common):
    terms = common.build_terms()
    assert terms.maturity_date.isoformat() == "2028-11-06"       # 2028-11-05 is a Sunday
    assert len(terms.ko_times) == 22                              # months 3..24
    assert terms.ko_dates[0].isoformat() == "2027-02-15"          # 2027-02-05 falls in the Spring Festival closure
    assert terms.ko_times[-1] == terms.maturity_years
    assert terms.ki_times[0] == 1.0 / 365.0 and terms.ki_times[-1] == terms.maturity_years
    assert all(b > a for a, b in zip(terms.ki_times, terms.ki_times[1:]))


def test_solve_coupon_affine_and_fails_closed(common):
    sol = common.solve_coupon(lambda c: 3.0e6 * (c - 0.2))
    assert sol.coupon == pytest.approx(0.2, abs=1e-14) and sol.annuity == pytest.approx(3.0e6)
    with pytest.raises(common.ValidationError):
        common.solve_coupon(lambda c: 3.0e6 * (c - 0.2) + 1e6 * c * c)   # not affine


def test_tails_share_nodes_flat_matches_forward_and_leverage(common):
    snap = common.load_snapshot()
    T = common.build_terms().maturity_years
    curves = {t: common.build_curve(snap, t) for t in common.TAILS}
    nodes = curves["hold_forward"].nodes
    for curve in curves.values():
        assert curve.nodes == nodes
        q = common.flat_q(curve, T)
        assert common.flat_forward_matches(curve, q, T, snap["spot"])
    lev = {t: common.tail_leverage(c, T) for t, c in curves.items()}
    t4, t5 = nodes[3][0], nodes[4][0]
    assert lev["hold_forward"][3] == pytest.approx(-(T - t5) / (t5 - t4), rel=1e-6)
    assert lev["hold_forward"][4] == pytest.approx(1.0 + (T - t5) / (t5 - t4), rel=1e-6)
    assert lev["hold_yield"][4] == pytest.approx(T / t5, rel=1e-6)
    assert lev["zero_carry"][4] == pytest.approx(1.0, rel=1e-6)


def test_pricing_results_meet_acceptance(common):
    res = common.read_json(common.RESULTS_DIR / "pricing.json")
    assert abs(res["coupon"]["pv"]) <= 1e-6 * res["notional"]                        # AC8
    eng = res["engine"]
    assert eng["quad_readout"] == "transition"
    assert eng["quad_grid_fine_resolved"] >= 2 * eng["quad_grid_resolved"]          # AC9: really doubled
    assert set(res["tails"]) == set(common.TAILS)
    for tail in res["tails"].values():
        assert tail["flat_forward_matches_curve"] is True                            # AC8
        for model in ("curve", "flat"):
            cell = tail[model]
            assert abs(cell["pv_bp_fine_grid"] - cell["pv_bp"]) <= 1.0               # AC9
            assert abs(cell["pv_bp_pde"] - cell["pv_bp"]) <= 5.0
            ev = cell["events"]
            assert 0.0 <= ev["p_ko_total"] <= 1.0 and 0.0 <= ev["p_ki_ever"] <= 1.0
            assert 0.0 < ev["expected_life_years"] <= res["terms"]["maturity_years"]
        assert abs(tail["gap_bp_pde"] - tail["gap_bp"]) <= 2.0
    assert set(res["marks"]) == {"mo_close", "all_settle"}


# --------------------------------------------------------------------------- T5 Greeks (AC5, AC10)


def test_forward_interval_bumps_partition_to_parallel(common):
    import numpy as np

    snap = common.load_snapshot()
    T = common.build_terms().maturity_years
    curve = common.build_curve(snap, "hold_forward")
    view = curve.to_dividend_yield(common.rate_curve())
    edges = common.interval_edges(curve, T) + [math.inf]
    ts = np.array([0.0, 0.01, 0.2, 0.7, 0.9, 1.5, T, 3.0])
    total = np.zeros_like(ts)
    for a, b in zip(edges, edges[1:]):
        wrapped = common.ForwardIntervalBumpDividendYield(view, a, b, 0.01)
        bumped = wrapped.get_yield(ts)
        total += bumped - view.get_yield(ts)
        for t, v in zip(ts[1:], bumped[1:]):                       # scalar path == array path
            assert wrapped.get_yield(float(t)) == pytest.approx(v, abs=1e-15)
    np.testing.assert_allclose(total[1:], 0.01, atol=1e-15)
    assert total[0] == 0.0
    with pytest.raises(common.ValidationError):
        common.ForwardIntervalBumpDividendYield(view, 1.0, 0.5, 0.01)


def test_node_yield_bump_moves_only_that_node(common):
    snap = common.load_snapshot()
    curve = common.build_curve(snap, "hold_forward")
    bumped = common.node_yield_bumped(curve, {2: 0.01})
    view, view_b = (x.to_dividend_yield(common.rate_curve()) for x in (curve, bumped))
    for i, t in enumerate(curve.tenors):
        expect = 0.01 if i == 2 else 0.0
        assert float(view_b.get_yield(t)) - float(view.get_yield(t)) == pytest.approx(expect, abs=1e-15)


def test_greeks_results_meet_acceptance(common):
    res = common.read_json(common.RESULTS_DIR / "greeks.json")
    notional = res["notional"]
    assert res["engine"]["quad_readout"] == "transition"
    assert set(res["tails"]) == set(common.TAILS)
    for name, tail in res["tails"].items():
        cur, half = tail["curve"], tail["curve_half_bumps"]
        # AC5: chain rule within 2% of the spot delta, every tail
        assert abs(cur["delta_identity_residual"]) <= 0.02 * abs(cur["spot_delta"])
        if name != "zero_carry":
            assert abs(cur["rho_sum_residual"]) <= 0.01 * abs(cur["parallel_rho_q"])
        else:
            # the flat-B tail cannot carry q+s: the residual IS the tail interval's exposure
            tail_interval = tail["forward_interval_rho"][-1]["rho_q"]
            assert cur["rho_sum_residual"] == pytest.approx(tail_interval, rel=0.01)
        # AC5: halving every bump moves each bucket by <= 2% (or <= 0.1 bp of notional)
        for b, hb in zip(cur["buckets"], half["buckets"]):
            for key, scale in (("delta", 0.01 * b["forward"]), ("rho_q", 1.0)):
                diff = abs(b[key] - hb[key])
                assert diff <= 0.02 * abs(b[key]) or 1e4 * diff * scale / notional <= 0.1
        # AC10: forward-interval buckets partition the parallel rho
        assert tail["forward_interval_sum"] == pytest.approx(cur["parallel_rho_q"], rel=0.005)
        assert len(tail["forward_interval_rho"]) == len(cur["buckets"]) + 1
        names = [s["scenario"] for s in tail["scenarios"]]
        assert len(names) == len(cur["buckets"]) + 3
        for s in tail["scenarios"]:
            assert s["missed_bp"] == pytest.approx(s["dpv_curve_bp"] - s["dpv_flat_bp"], abs=1e-9)
        lots = [b["hedge_lots"] for b in cur["buckets"]]
        assert all(isinstance(x, float) for x in lots)
    # the default tail hedges the unquoted 2Y exposure as an IM2706 / MO2709 calendar spread
    hf = res["tails"]["hold_forward"]["curve"]["buckets"]
    assert hf[3]["hedge_lots"] * hf[4]["hedge_lots"] < 0.0


# --------------------------------------------------------------------------- T6 report (AC11)

REPORT_SECTION_IDS = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")


def test_report_builds_offline_with_required_chrome(common, tmp_path):
    import re

    report = _load("snowball_2y_cc_report", "04_report.py")
    out = tmp_path / "report.html"
    report.build_report(out)
    assert out.exists()
    page = out.read_text(encoding="utf-8")
    # frozen navigator + light/dark switcher (docs/UI_LANGUAGE.md)
    for needle in ("position: sticky", "data-theme", "localStorage", "prefers-color-scheme", "IntersectionObserver"):
        assert needle in page, needle
    # opens offline: no network asset in any src/href
    assert not re.search(r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", page, flags=re.I)
    for sid in REPORT_SECTION_IDS:
        assert f'id="{sid}"' in page, sid
        assert f'href="#{sid}"' in page, sid
    assert "not hedgeable with IM futures" in page
    # true minus sign in figures, never a negative zero
    assert "−0.00<" not in page and "-0.00<" not in page
