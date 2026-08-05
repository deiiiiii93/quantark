"""Tests for example/mo_volmodels/11_pde_convergence_gate.py.

Harness unit tests (date substitution, term-sheet builder, tolerance/gate
logic on synthetic numbers incl. bias detection, JSON schemas, deterministic
hashing) plus ONE --quick end-to-end run asserting the pipeline completes and
writes all artifacts, and that a rerun reproduces the identical evidence hash.
"""

import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/11_pde_convergence_gate.py"
HISTORY_DIR = ROOT / "example/mo_volmodels/data/history"

spec = importlib.util.spec_from_file_location("pde_convergence_gate_11", MODULE_PATH)
gate11 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate11  # dataclass processing resolves cls.__module__ here
spec.loader.exec_module(gate11)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weekday_calendar(start: date, end: date) -> gate11.TradingCalendar:
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return gate11.TradingCalendar(days)


def _cell(date_s, case, *, level="medium", passed=True, signed=0.1, pde=10.0, notional=100.0):
    return {
        "date": date_s,
        "case": case,
        "variant": "heston",
        "level": level,
        "grid": {"n_x": 1, "n_v": 1, "n_t": 1},
        "pde_price": pde,
        "mc_ref": 10.0,
        "mc_se": 0.01,
        "notional": notional,
        "signed_diff_pct": signed,
        "tol_pct": 0.25,
        "passed": passed,
        "error": None if pde is not None else "boom",
    }


def _delta_row(date_s, case, *, s0=100.0, signed_diff=0.0, passed=True):
    """A Task-6 delta row (see gate11.decide_route / detect_delta_bias)."""
    return {
        "date": date_s,
        "case": case,
        "variant": "heston",
        "level": "medium",
        "s0": s0,
        "signed_diff": signed_diff,
        "passed": passed,
    }


# Neutral delta evidence for decide_route tests whose PV cells are the thing
# under test: passes cleanly and carries zero mean bias, so it never itself
# routes the variant to "mc".
_PASSING_DELTAS = [_delta_row("2023-05-15", "full"), _delta_row("2024-02-08", "full")]


def _all_pass_medium_fine():
    medium = [_cell("2023-05-15", "full", signed=0.1, pde=10.10),
              _cell("2023-05-15", "decayed", signed=-0.1, pde=9.90),
              _cell("2024-02-08", "full", signed=0.12, pde=10.12),
              _cell("2024-02-08", "decayed", signed=-0.08, pde=9.92)]
    fine = [_cell("2023-05-15", "full", level="fine", signed=0.05, pde=10.15),
            _cell("2023-05-15", "decayed", level="fine", signed=-0.05, pde=9.95),
            _cell("2024-02-08", "full", level="fine", signed=0.06, pde=10.18),
            _cell("2024-02-08", "decayed", level="fine", signed=-0.04, pde=9.96)]
    return medium, fine


# ---------------------------------------------------------------------------
# add_months / calendar
# ---------------------------------------------------------------------------

def test_add_months_basic_and_clamp():
    assert gate11.add_months(date(2023, 5, 15), 3) == date(2023, 8, 15)
    assert gate11.add_months(date(2023, 11, 15), 36) == date(2026, 11, 15)
    assert gate11.add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # leap clamp
    assert gate11.add_months(date(2023, 12, 15), 2) == date(2024, 2, 15)


def test_trading_calendar_extension_and_ranges():
    cal = _weekday_calendar(date(2023, 5, 4), date(2023, 5, 19))
    # next_trading_day snaps weekend to Monday inside the CSV range
    assert cal.next_trading_day(date(2023, 5, 13)) == date(2023, 5, 15)
    # beyond the CSV end it extends with weekdays
    assert cal.next_trading_day(date(2023, 5, 21)) == date(2023, 5, 22)
    assert cal.is_trading_day(date(2023, 6, 5))  # Monday, beyond range
    assert not cal.is_trading_day(date(2023, 6, 4))  # Sunday
    between = cal.trading_days_between(date(2023, 5, 15), date(2023, 5, 19))
    assert between == [date(2023, 5, 16), date(2023, 5, 17), date(2023, 5, 18), date(2023, 5, 19)]


# ---------------------------------------------------------------------------
# Sample-date resolution / substitution
# ---------------------------------------------------------------------------

def test_resolve_sample_dates_substitution():
    admitted = [date(2024, 2, 5), date(2024, 2, 8), date(2024, 2, 19)]
    requested = [date(2024, 2, 19), date(2024, 2, 7)]
    out = gate11.resolve_sample_dates(admitted, requested)
    assert out[0] == {"requested": "2024-02-19", "date": "2024-02-19", "substituted": False}
    # 2024-02-07 is not admitted; nearest admitted is 2024-02-08 (1d away vs 2d)
    assert out[1] == {"requested": "2024-02-07", "date": "2024-02-08", "substituted": True}


def test_resolve_sample_dates_tie_break_and_dedupe():
    admitted = [date(2024, 2, 5), date(2024, 2, 9)]
    out = gate11.resolve_sample_dates(admitted, [date(2024, 2, 7)])
    assert out[0]["date"] == "2024-02-05"  # earlier wins on ties
    # several requests mapping to the SAME admitted date are deduped
    admitted2 = [date(2024, 2, 5), date(2024, 2, 20)]
    dup = gate11.resolve_sample_dates(
        admitted2, [date(2024, 2, 6), date(2024, 2, 7), date(2024, 2, 8)]
    )
    assert len(dup) == 1
    assert dup[0]["date"] == "2024-02-05"


def test_resolve_sample_dates_requires_admitted():
    with pytest.raises(ValueError):
        gate11.resolve_sample_dates([], [date(2024, 1, 2)])


# ---------------------------------------------------------------------------
# Term-sheet builder
# ---------------------------------------------------------------------------

def test_terms_builder_ko_ki_structure():
    cal = _weekday_calendar(date(2023, 5, 2), date(2026, 6, 1))
    terms = gate11.build_snowball_terms(date(2023, 5, 15), cal)
    assert terms.maturity_date == date(2026, 5, 15)
    assert abs(terms.maturity_years - 1096 / 365.0) < 1e-12
    # 34 monthly KO dates, first at month 3, last at maturity
    assert len(terms.ko_times) == 34
    first_ko_expected = (date(2023, 8, 15) - date(2023, 5, 15)).days / 365.0
    assert abs(terms.ko_times[0] - first_ko_expected) < 1e-12
    assert abs(terms.ko_times[-1] - terms.maturity_years) < 1e-12
    assert all(a < b for a, b in zip(terms.ko_times, terms.ko_times[1:]))
    # KO spacing is ~monthly (28-31 days +/- up to 2 days of weekend snapping)
    gaps = [b - a for a, b in zip(terms.ko_times, terms.ko_times[1:])]
    assert all(25 / 365.0 <= g <= 35 / 365.0 for g in gaps)
    # daily-discrete KI: every trading day in (inception, maturity]
    assert 700 <= len(terms.ki_times) <= 800
    assert all(0.0 < t <= terms.maturity_years + 1e-12 for t in terms.ki_times)
    assert all(a < b for a, b in zip(terms.ki_times, terms.ki_times[1:]))
    # every KO date is also a KI (trading) date
    assert set(terms.ko_times) <= set(terms.ki_times)


def test_decay_terms_shifts_and_drops():
    cal = _weekday_calendar(date(2023, 5, 2), date(2026, 6, 1))
    terms = gate11.build_snowball_terms(date(2023, 5, 15), cal)
    decay = date(2025, 5, 15)
    shifted = gate11.decay_terms(terms, decay)
    t_shift = (decay - terms.inception).days / 365.0
    assert abs(shifted.maturity_years - (terms.maturity_years - t_shift)) < 1e-12
    assert shifted.valuation == decay
    assert shifted.inception == terms.inception
    assert all(t > 0.0 for t in shifted.ko_times + shifted.ki_times)
    # remaining KO dates are the original ones after the decay date
    expected_ko = sum(1 for t in terms.ko_times if t > t_shift)
    assert len(shifted.ko_times) == expected_ko == 12
    assert shifted.maturity_date == terms.maturity_date
    with pytest.raises(ValueError):
        gate11.decay_terms(terms, terms.inception)


def test_pick_decay_date_bounds():
    cal = _weekday_calendar(date(2023, 5, 2), date(2026, 6, 1))
    terms = gate11.build_snowball_terms(date(2023, 5, 15), cal)
    admitted = [date(2025, 5, 14), date(2025, 5, 15), date(2026, 7, 22)]
    # latest admitted <= inception+2Y with remainder in [0.5, 2.0]
    assert gate11.pick_decay_date(admitted, terms) == date(2025, 5, 15)
    # if only a too-late date is available the remainder shrinks below 0.5y -> skip
    assert gate11.pick_decay_date([date(2026, 5, 10)], terms) is None
    # if nothing is after inception -> skip
    assert gate11.pick_decay_date([date(2023, 5, 15)], terms) is None
    # recent inception: everything leaves ~3y remaining (> 2.0) -> skip
    terms_recent = gate11.build_snowball_terms(date(2026, 7, 15), _weekday_calendar(date(2026, 7, 1), date(2029, 8, 1)))
    assert gate11.pick_decay_date([date(2026, 7, 22)], terms_recent) is None


def test_build_snowball_product_from_terms():
    cal = _weekday_calendar(date(2023, 5, 2), date(2026, 6, 1))
    terms = gate11.build_snowball_terms(date(2023, 5, 15), cal)
    s0 = 6585.91
    product = gate11.build_snowball_product(terms, s0)
    bc = product.barrier_config
    assert product.initial_price == pytest.approx(s0)
    assert product.strike == pytest.approx(s0)
    assert product.maturity == pytest.approx(terms.maturity_years)
    assert bc.ko_barrier == pytest.approx(1.03 * s0)
    assert bc.ki_barrier == pytest.approx(0.75 * s0)
    assert bc.ko_rate == pytest.approx(0.15)
    assert len(bc.ko_observation_dates) == 34
    assert abs(bc.ko_observation_dates[0] - terms.ko_times[0]) < 1e-12
    assert len(bc.ki_observation_dates) == len(terms.ki_times)
    assert bc.ki_continuous is False
    from quantark.util.enum import ObservationType

    assert bc.ki_observation_type == ObservationType.DISCRETE
    assert product.payoff_config.include_principal is False


# ---------------------------------------------------------------------------
# PDE ladder / event-key collisions
# ---------------------------------------------------------------------------

def test_pde_ladder_full_and_quick():
    ladder = dict((lvl, (nx, nv, nt)) for lvl, nx, nv, nt in gate11.pde_ladder(3.0, quick=False))
    assert ladder["coarse"] == (90, 36, 96)  # stage-08 grid
    assert ladder["medium"] == (200, 60, 1200)  # ceil(400*T): dt <= 1/400y
    assert ladder["fine"] == (300, 90, 2400)
    decayed = dict((lvl, (nx, nv, nt)) for lvl, nx, nv, nt in gate11.pde_ladder(1.0, quick=False))
    assert decayed["medium"][2] == 400
    assert decayed["fine"][2] == 800
    quick = gate11.pde_ladder(3.0, quick=True)
    assert [lvl for lvl, *_ in quick] == ["coarse", "medium", "fine"]
    assert max(nt for *_rest, nt in quick) <= 64


def test_event_key_collision_count():
    # daily KI times over 3y
    ki = [(i + 1) / 365.0 for i in range(3 * 365)]
    assert gate11.event_key_collision_count(ki, 3.0, 1200) == 0  # dt < 1 day
    coarse = gate11.event_key_collision_count(ki, 3.0, 96)
    assert coarse > 900  # ~11 events per step bucket at the stage-08 grid


# ---------------------------------------------------------------------------
# Tolerance / bias / route logic on synthetic numbers
# ---------------------------------------------------------------------------

def test_gate_tolerance_and_cell_pass():
    assert gate11.gate_tolerance_pct(0.05, 0.25) == pytest.approx(0.25)  # floor binds
    assert gate11.gate_tolerance_pct(0.20, 0.25) == pytest.approx(0.40)  # 2*SE binds
    assert gate11.gate_cell_passed(0.24, 0.05, 0.25) is True
    assert gate11.gate_cell_passed(-0.26, 0.05, 0.25) is False
    assert gate11.gate_cell_passed(0.39, 0.20, 0.25) is True


def test_detect_systematic_bias():
    biased, info = gate11.detect_systematic_bias([0.13] * 8, 0.25)
    assert biased is True
    assert info["sign_fraction"] == pytest.approx(1.0)
    assert info["median_abs_pct"] == pytest.approx(0.13)


def test_detect_systematic_bias_fraction_threshold():
    # 7/8 same sign is below the 0.9 threshold -> not biased
    biased, _ = gate11.detect_systematic_bias([-0.2] * 7 + [0.05], 0.25)
    assert biased is False
    # alternating signs -> not biased even with large magnitudes
    biased, _ = gate11.detect_systematic_bias([0.3, -0.3] * 5, 0.25)
    assert biased is False
    # same sign but tiny magnitude (< 0.5*TOL) -> not biased
    biased, _ = gate11.detect_systematic_bias([0.05] * 10, 0.25)
    assert biased is False
    # fewer than 4 cells -> not biased
    biased, info = gate11.detect_systematic_bias([0.5, 0.5, 0.5], 0.25)
    assert biased is False
    assert info["n_cells"] == 3


def test_decide_route_all_pass():
    medium, fine = _all_pass_medium_fine()
    out = gate11.decide_route(medium, fine, _PASSING_DELTAS, 0.25)
    assert out["route"] == "pde"
    assert out["medium_pass"] and out["fine_pass"] and not out["biased"]
    assert out["drift_max_pct"] <= 0.25
    assert out["delta_pass"] is True and out["delta_biased"] is False


def test_decide_route_medium_failure_is_mc():
    medium = [_cell("2023-05-15", "full", passed=False, signed=0.9),
              _cell("2023-05-15", "decayed"),
              _cell("2024-02-08", "full"),
              _cell("2024-02-08", "decayed")]
    fine = [_cell(d, c, level="fine") for d, c in
            [("2023-05-15", "full"), ("2023-05-15", "decayed"), ("2024-02-08", "full"), ("2024-02-08", "decayed")]]
    out = gate11.decide_route(medium, fine, _PASSING_DELTAS, 0.25)
    assert out["route"] == "mc"
    assert "medium" in out["rationale"]


def test_decide_route_bias_is_mc_even_inside_tolerance():
    # every diff +0.13% (inside tol 0.25) but systematically one-sided
    medium = [_cell(f"2023-0{i}-15", "full", signed=0.13) for i in range(1, 5)] + \
             [_cell(f"2023-0{i}-15", "decayed", signed=0.14) for i in range(1, 5)]
    fine = [_cell(c["date"], c["case"], level="fine", signed=0.13, pde=10.10) for c in medium]
    out = gate11.decide_route(medium, fine, _PASSING_DELTAS, 0.25)
    assert out["route"] == "mc"
    assert out["biased"] is True
    assert "bias" in out["rationale"]


def test_decide_route_fine_drift_is_mc():
    medium = [_cell(f"2023-0{i}-15", "full", signed=0.05 * (1 if i % 2 else -1), pde=10.0) for i in range(1, 5)] + \
             [_cell(f"2023-0{i}-15", "decayed", signed=0.05 * (1 if i % 2 else -1), pde=10.0) for i in range(1, 5)]
    fine = [_cell(c["date"], c["case"], level="fine", signed=0.04, pde=10.4) for c in medium]
    out = gate11.decide_route(medium, fine, _PASSING_DELTAS, 0.25)
    assert out["route"] == "mc"
    assert out["drift_max_pct"] > 0.25
    assert "drift" in out["rationale"]


def test_decide_route_error_cell_is_mc():
    medium = [_cell("2023-05-15", "full", passed=False, signed=None, pde=None),
              _cell("2023-05-15", "decayed"),
              _cell("2024-02-08", "full"),
              _cell("2024-02-08", "decayed")]
    fine = [_cell(d, c, level="fine") for d, c in
            [("2023-05-15", "full"), ("2023-05-15", "decayed"), ("2024-02-08", "full"), ("2024-02-08", "decayed")]]
    out = gate11.decide_route(medium, fine, _PASSING_DELTAS, 0.25)
    assert out["route"] == "mc"


def test_decide_route_empty_cells_is_mc():
    out = gate11.decide_route([], [], [], 0.25)
    assert out["route"] == "mc"
    assert "no cells evaluated" in out["rationale"]
    assert out["medium_pass"] is False and out["fine_pass"] is False
    assert out["delta_pass"] is False
    fine_only = [_cell("2023-05-15", "full", level="fine")]
    out2 = gate11.decide_route([], fine_only, [], 0.25)
    assert out2["route"] == "mc"
    assert "no cells evaluated" in out2["rationale"]
    medium_only = [_cell("2023-05-15", "full")]
    out3 = gate11.decide_route(medium_only, [], [], 0.25)
    assert out3["route"] == "mc"


def test_decide_route_delta_cell_failure_is_mc():
    """Every PV cell passes, but one delta row exceeds half a contract."""
    medium, fine = _all_pass_medium_fine()
    deltas = [_delta_row("2023-05-15", "full", passed=False),
              _delta_row("2024-02-08", "full", passed=True)]
    out = gate11.decide_route(medium, fine, deltas, 0.25)
    assert out["route"] == "mc"
    assert out["delta_pass"] is False
    assert "delta disagreement" in out["rationale"]


def test_decide_route_delta_bias_is_mc_even_though_each_cell_passes():
    """Each delta cell individually passes, but they're all one-sided."""
    medium, fine = _all_pass_medium_fine()
    s0 = 6733.97
    q = gate11.delta_quantum_per_unit(s0)
    deltas = [
        _delta_row(f"2023-0{i}-15", "full", s0=s0, signed_diff=0.2 * q, passed=True)
        for i in range(1, 9)
    ]
    assert all(d["passed"] for d in deltas)  # confirms the bias is NOT a cell failure
    out = gate11.decide_route(medium, fine, deltas, 0.25)
    assert out["route"] == "mc"
    assert out["delta_pass"] is True
    assert out["delta_biased"] is True
    assert "delta bias" in out["rationale"]


def test_decide_route_empty_delta_rows_is_mc_even_when_pv_all_passes():
    """Empty delta evidence must not admit PDE -- mirrors empty PV evidence."""
    medium, fine = _all_pass_medium_fine()
    out = gate11.decide_route(medium, fine, [], 0.25)
    assert out["route"] == "mc"
    assert out["delta_pass"] is False
    assert "no delta evidence" in out["rationale"]


def test_decide_route_delegates_adi_greeks_without_weakening_pv_gate():
    medium, fine = _all_pass_medium_fine()
    failing_delta = [_delta_row("2023-05-15", "full", passed=False)]

    delegated = gate11.decide_route(
        medium,
        fine,
        failing_delta,
        0.25,
        require_delta=False,
    )
    assert delegated["route"] == "pde"
    assert delegated["delta_pass"] is False
    assert delegated["delta_required"] is False
    assert "delegated" in delegated["rationale"]

    failing_medium = list(medium)
    failing_medium[0] = dict(failing_medium[0], passed=False)
    assert gate11.decide_route(
        failing_medium,
        fine,
        _PASSING_DELTAS,
        0.25,
        require_delta=False,
    )["route"] == "mc"


# ---------------------------------------------------------------------------
# Non-finite engine prices -> recorded error cells (never a crashed run)
# ---------------------------------------------------------------------------


class _StubMCEngine:
    def __init__(self, price, se=0.01):
        self._price = price
        self._se = se

    def price(self, product, env):
        return self._price

    def get_last_result(self):
        return types.SimpleNamespace(num_paths=8, batches_used=1)

    def get_last_std_error(self):
        return self._se


class _StubPDEEngine:
    def __init__(self, price):
        self._price = price

    def price(self, product, env):
        return self._price


def _tiny_terms():
    cal = _weekday_calendar(date(2023, 5, 2), date(2023, 12, 1))
    return gate11.build_snowball_terms(
        date(2023, 5, 15), cal, maturity_months=6, lockout_months=1
    )


def _tiny_cfg():
    return {
        "quick": True,
        "mc": {"paths_per_batch": 8, "batches": 1, "substeps_per_interval": 1},
        "seed": 1,
        "tol_abs_pct": 0.25,
    }


def _stub_gate_pairs(mc_price, pde_price):
    """Two variants, both fully stubbed -- exercises _evaluate_case's error
    handling without needing a real PricingEnvironment or calibrated model."""
    pair = gate11.GatePair(
        production="pde_2d_stub", reference="mc_stub",
        build_production=lambda model, grid: _StubPDEEngine(pde_price),
        build_reference=lambda model, grid: _StubMCEngine(mc_price),
        reference_is_mc=True,
        surface_vol_mode="full_grid",  # matches heston/heston_slv's real mode
    )
    return {"heston": pair, "heston_slv": pair}


def _evaluate_with_stubs(monkeypatch, mc_price, pde_price):
    pairs = _stub_gate_pairs(mc_price, pde_price)
    monkeypatch.setattr(gate11, "GATE_PAIRS", pairs)
    monkeypatch.setattr(gate11, "VARIANTS", tuple(pairs))
    return gate11._evaluate_case(
        date_iso="2023-05-15",
        case_name="decayed",  # skips the delta path (needs a real env)
        terms=_tiny_terms(),
        s0_inception=100.0,
        envs={"full_grid": None},  # engines are stubbed; env is only passed through
        models={"heston": object(), "heston_slv": object()},
        cfg=_tiny_cfg(),
    )


def _assert_cells_survive_assembly(cells):
    payload = {
        "schema_version": 1,
        "study": "pde_convergence_gate",
        "config": {},
        "dates": [],
        "cells": cells,
        "sanity": {},
    }
    gate11.validate_gate_payload(payload)
    json.dumps(payload, allow_nan=False)  # the old crash site


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pde_price_becomes_error_cells(monkeypatch, bad):
    out = _evaluate_with_stubs(monkeypatch, mc_price=1.0, pde_price=bad)
    cells = out["cells"]
    assert len(cells) == 2 * 3  # 2 variants x 3 levels
    for c in cells:
        assert c["passed"] is False
        assert c["pde_price"] is None
        assert c["signed_diff_pct"] is None
        assert c["error"] and "non-finite production price" in c["error"]
        assert c["mc_ref"] == 1.0  # reference itself is fine here
    _assert_cells_survive_assembly(cells)


def test_non_finite_mc_reference_becomes_error_cells(monkeypatch):
    out = _evaluate_with_stubs(monkeypatch, mc_price=float("nan"), pde_price=0.9)
    for c in out["cells"]:
        assert c["passed"] is False
        assert c["mc_ref"] is None and c["mc_se"] is None and c["tol_pct"] is None
        assert c["signed_diff_pct"] is None
        assert c["error"] and "non-finite reference price" in c["error"]
        assert c["pde_price"] == 0.9  # finite PDE price kept as evidence
    for ref in out["mc_refs"].values():
        assert ref["price"] is None and ref["std_error"] is None
        assert ref["error"] and "non-finite reference price" in ref["error"]
    _assert_cells_survive_assembly(out["cells"])


# ---------------------------------------------------------------------------
# Deterministic hashing / schemas
# ---------------------------------------------------------------------------

def test_strip_timings_and_canonical_hash():
    base = {"a": 1, "timings": {"seconds": 1.0}, "b": [{"c": 2, "timings": {"x": 3}}]}
    other = {"a": 1, "timings": {"seconds": 99.0}, "b": [{"c": 2, "timings": {"x": -4}}]}
    assert gate11.canonical_sha256(base) == gate11.canonical_sha256(other)
    changed = {"a": 2, "b": [{"c": 2}]}
    assert gate11.canonical_sha256(base) != gate11.canonical_sha256(changed)
    assert "timings" not in json.dumps(gate11.strip_timings(base))


def test_validate_gate_payload_roundtrip_and_failures():
    good = {
        "schema_version": 1,
        "study": "pde_convergence_gate",
        "config": {},
        "dates": [],
        "cells": [
            {
                "date": "2023-05-15", "case": "full", "variant": "heston",
                "level": "medium", "grid": {}, "pde_price": 1.0, "mc_ref": 1.0,
                "mc_se": 0.1, "notional": 100.0, "signed_diff_pct": 0.0,
                "tol_pct": 0.25, "passed": True, "error": None,
            }
        ],
        "sanity": {},
    }
    gate11.validate_gate_payload(good)
    bad = json.loads(json.dumps(good))
    bad["cells"][0]["pde_price"] = None
    with pytest.raises(ValueError):
        gate11.validate_gate_payload(bad)  # null price without error tag
    bad = json.loads(json.dumps(good))
    bad["cells"][0]["pde_price"] = None
    bad["cells"][0]["mc_ref"] = None
    bad["cells"][0]["mc_se"] = None
    bad["cells"][0]["signed_diff_pct"] = None
    bad["cells"][0]["error"] = "boom"
    gate11.validate_gate_payload(bad)  # error cells may carry nulls
    bad = json.loads(json.dumps(good))
    bad["cells"][0]["level"] = "ultra"
    with pytest.raises(ValueError):
        gate11.validate_gate_payload(bad)


def test_validate_decision_payload(monkeypatch):
    # Scope VARIANTS to the two used here; the schema rules under test
    # (route enum, params-key presence, rationale) don't depend on the count.
    monkeypatch.setattr(gate11, "VARIANTS", ("heston", "heston_slv"))
    good = {
        "schema_version": 1,
        "study": "pde_convergence_gate_decision",
        "variants": {
            "heston": {
                "route": "pde",
                "pde_params": {"n_x": 1},
                "gate": {
                    "delta_authority": "stage16",
                    "delta_required": False,
                },
                "rationale": "ok",
            },
            "heston_slv": {
                "route": "mc",
                "gate": {
                    "delta_authority": "stage16",
                    "delta_required": False,
                },
                "mc_params": {
                    "engine": "HestonSLVQESnowballMCEngine",
                    "method": "randomized_quasi",
                    "seed": 1,
                    "paths_per_batch": 8192,
                    "batches": 16,
                    "rqmc_min_batches": 16,
                    "rqmc_max_batches": 16,
                    "rqmc_target_std": 1e-12,
                    "substeps_per_interval": 1,
                    "note": "REFERENCE-quality config used for the gate",
                },
                "rationale": "no",
            },
        },
        "tolerance": {},
        "evidence_sha256": "abc",
    }
    gate11.validate_decision_payload(good)
    # the RQMC pinning keys are part of the decision schema contract
    pinned = {"rqmc_min_batches", "rqmc_max_batches", "rqmc_target_std", "note"}
    assert pinned <= set(good["variants"]["heston_slv"]["mc_params"])
    bad = json.loads(json.dumps(good))
    bad["variants"]["heston"]["route"] = "quad"
    with pytest.raises(ValueError):
        gate11.validate_decision_payload(bad)
    bad = json.loads(json.dumps(good))
    del bad["variants"]["heston_slv"]
    with pytest.raises(ValueError):
        gate11.validate_decision_payload(bad)


# ---------------------------------------------------------------------------
# End-to-end --quick smoke on one real date
# ---------------------------------------------------------------------------

def _run_quick(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--quick",
            "--out-dir",
            str(out_dir),
            "--history-dir",
            str(HISTORY_DIR),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.skipif(
    not (HISTORY_DIR / "surface_manifest.json").is_file(),
    reason="frozen surface history not present",
)
def test_quick_end_to_end(tmp_path):
    out1 = tmp_path / "run1"
    proc = _run_quick(out1)
    assert proc.returncode == 0, f"quick run failed:\n{proc.stdout}\n{proc.stderr}"

    gate_path = out1 / "pde_convergence_gate.json"
    decision_path = out1 / "gate_decision.json"
    report_path = out1 / "gate_report.md"
    for path in (gate_path, decision_path, report_path):
        assert path.is_file(), f"missing {path.name}"

    gate = json.loads(gate_path.read_text())
    decision = json.loads(decision_path.read_text())
    gate11.validate_gate_payload(gate)
    gate11.validate_decision_payload(decision)

    # one quick date, both cases present, 3 levels x every study variant x 2 cases
    assert [d["date"] for d in gate["dates"]] == ["2023-05-15"]
    assert gate["dates"][0]["substituted"] is False
    full = gate["dates"][0]["terms"]["full"]
    assert full["n_ko"] == 34 and full["n_ki"] > 600
    assert gate["dates"][0]["terms"]["decayed"] is not None
    n_variants = len(gate11.VARIANTS)
    assert n_variants == 6  # the six-variant pair table this gate now covers
    combos = {(c["case"], c["variant"], c["level"]) for c in gate["cells"]}
    assert len(combos) == 2 * n_variants * 3
    assert len(gate["cells"]) == 2 * n_variants * 3
    for c in gate["cells"]:
        if gate11.GATE_PAIRS[c["variant"]].reference_is_mc:
            assert c["mc_se"] > 0.0
        else:
            assert c["mc_se"] is None  # deterministic reference (QUAD / finer PDE)
        assert c["tol_pct"] >= 0.25
    # calibrated params recorded per date -- present for every variant, but
    # only the vol models the calibrator fits (heston/heston_slv/localvol)
    # carry a non-None record; the BSM variants price off the raw surface.
    assert set(gate["dates"][0]["calibration"]) == set(gate11.VARIANTS)
    for variant in gate11.VARIANTS:
        record = gate["dates"][0]["calibration"][variant]
        if variant in gate11._CALIBRATED_VARIANTS:
            assert record is not None, variant
        else:
            assert record is None, variant
    assert {"v0", "kappa", "theta", "sigma", "rho"} <= set(
        gate["dates"][0]["calibration"]["heston"]
    )
    # decision sanity
    assert set(decision["variants"]) == set(gate11.VARIANTS)
    for variant, row in decision["variants"].items():
        assert row["route"] in ("pde", "mc")
        assert row["rationale"]
        mc = row["mc_params"]
        if gate11.GATE_PAIRS[variant].reference_is_mc:
            # RQMC pinning keys are part of the emitted decision contract
            assert {"rqmc_min_batches", "rqmc_max_batches", "rqmc_target_std", "note"} <= set(mc)
            assert mc["rqmc_min_batches"] == mc["batches"]
            assert mc["rqmc_max_batches"] == mc["batches"]
            assert mc["rqmc_target_std"] > 0.0
            assert "REFERENCE" in mc["note"] and "Phase 4" in mc["note"]
        else:
            assert mc["kind"] == "deterministic"
    assert decision["quick_mode"] is True
    # evidence hash matches the canonical hash of the gate payload
    assert decision["evidence_sha256"] == gate11.canonical_sha256(gate)
    assert "heston" in report_path.read_text()

    # determinism: rerun reproduces the identical evidence hash
    out2 = tmp_path / "run2"
    proc2 = _run_quick(out2)
    assert proc2.returncode == 0, f"quick rerun failed:\n{proc2.stdout}\n{proc2.stderr}"
    gate2 = json.loads((out2 / "pde_convergence_gate.json").read_text())
    decision2 = json.loads((out2 / "gate_decision.json").read_text())
    assert gate11.canonical_sha256(gate2) == gate11.canonical_sha256(gate)
    assert decision2["evidence_sha256"] == decision["evidence_sha256"]
