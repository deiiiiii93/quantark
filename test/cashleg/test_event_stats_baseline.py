"""Cash-leg baseline gate (unified event statistics spec D9, AC5 / AC19).

``event_stats_baseline/baseline.json`` was banked at the base commit (``BASE_SHA``) by
``event_stats_baseline/generate.py``: per (engine, product) case, ``price()``, the event route's npv,
and every leg's value on the event route (``price_with_events`` -> ``leg.value``) and on the folded
route (``folded_trade_value`` TOTAL per leg; DRILLDOWN per leg, total and npv). Combinations that
raised at base are banked as expected raises (exception class name).

Each record is one test. A banked value must be reproduced within ``test/golden_compare.py``'s
cross-arch tolerance (the bank is ARM64, CI is x86_64): relative ``GOLDEN_REL_TOL`` and an absolute
floor of ``GOLDEN_REL_TOL`` times the record's nominal cash scale. A banked raise must still raise
the same exception class.

A later task that moves a record on purpose lists it in ``EXPECTED_MOVES`` with one of the D9
causes and a check that asserts the move against that cause; nothing else may move.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import pytest

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL, close

HERE = Path(__file__).resolve().parent
BANK_DIR = HERE / "event_stats_baseline"
BANK = BANK_DIR / "baseline.json"


def _load_cases():
    """``event_stats_baseline/cases.py`` under a unique name (``cases`` is taken by other test dirs, and
    a ``cashleg`` package on ``sys.path`` would shadow the legacy ``cashleg`` import alias)."""
    name = "_cashleg_event_stats_baseline_cases"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BANK_DIR / "cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cases = _load_cases()

#: spec D9's causes a banked value may move for (landing 2 asserts each move against its cause)
D9_CAUSES: Dict[str, str] = {
    "a": "KI-after-KO removal",
    "b": "coupon cash out of KO cash",
    "c": "fixed defects (T2-T4)",
    "d": "maturity probability direct instead of 1 - sum(KO) - ki (moves by probability_residual)",
    "e": "KI stream on its own axis (AT_KI legs pay on KI dates, or stop raising)",
    "f": "MC statistics on price() paths (T3, D14)",
    "g": "REBASED now supported",
}


@dataclass(frozen=True)
class MovedBy:
    """A record that moves on purpose: ``cause`` is a ``D9_CAUSES`` key; ``check(live, banked, record)``
    asserts the live outcome against that cause (it must fail if the move has any other source)."""

    cause: str
    check: Callable[[dict, dict, dict], None]


#: record id -> MovedBy. Empty at base: every banked record must reproduce.
EXPECTED_MOVES: Dict[str, MovedBy] = {}


def _bank():
    if not BANK.exists():
        return None
    with BANK.open() as fh:
        return json.load(fh)


BANKED = _bank()
RECORDS = {} if BANKED is None else {r["id"]: r for r in BANKED["records"]}


# --- live recomputation (cached per worker: one event route / one drilldown per case) -------------

@lru_cache(maxsize=None)
def _case(case_id: str):
    meta = BANKED["cases"][case_id]
    return cases.Case(meta["engine"], meta["product"])


@lru_cache(maxsize=None)
def _event(case_id: str):
    return cases.event_route(_case(case_id))


@lru_cache(maxsize=None)
def _drilldown(case_id: str):
    return cases.fold_drilldown(_case(case_id), list(BANKED["cases"][case_id]["drilldown_set"]))


def live_outcome(record: dict) -> dict:
    """Recompute one record exactly as ``generate.py`` banked it."""
    case_id, leg, route = record["case"], record["leg"], record["route"]
    if route == "price":
        return _event(case_id)["price"]
    if route == "event_npv":
        return _event(case_id)["event_npv"]
    if route == "event":
        return _event(case_id)["legs"][leg]
    if route == "fold_total":
        return cases.fold_total(_case(case_id), leg)
    if route == "fold_drilldown_total":
        return _drilldown(case_id)["total"]
    if route == "fold_drilldown_npv":
        return _drilldown(case_id)["npv"]
    if route == "fold_drilldown":
        if leg in BANKED["cases"][case_id]["drilldown_set"]:
            return _drilldown(case_id)["per_leg"][leg]
        return cases.fold_drilldown(_case(case_id), [leg])["per_leg"][leg]
    raise AssertionError(f"unknown route {route!r}")


def assert_reproduced(live: dict, banked: dict, record: dict) -> None:
    """A banked raise raises the same class; a banked value is reproduced within golden tolerance."""
    if "raises" in banked:
        assert "raises" in live, (
            f"{record['id']}: banked a {banked['raises']} at base, now returns {live.get('value')!r}"
        )
        assert live["raises"] == banked["raises"], (
            f"{record['id']}: banked {banked['raises']} at base, now raises {live['raises']}: {live.get('message')}"
        )
        return
    assert "value" in live, (
        f"{record['id']}: banked {banked['value']!r} at base, now raises {live['raises']}: {live.get('message')}"
    )
    abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
    assert close(float(live["value"]), float(banked["value"]), rel_tol=GOLDEN_REL_TOL, abs_tol=abs_tol), (
        f"{record['id']}: live {live['value']!r} != banked {banked['value']!r} "
        f"(rel {GOLDEN_REL_TOL}, abs {abs_tol})"
    )


# --- the gate --------------------------------------------------------------------------------------

def test_the_bank_exists_at_base():
    assert BANKED is not None, f"{BANK} is not banked; run event_stats_baseline/generate.py at base"
    assert BANKED["header"]["base_sha"] == cases.BASE_SHA
    assert RECORDS


def test_the_bank_covers_every_case_leg_and_route():
    """The bank and ``cases.py`` describe the same matrix (a case or leg added later is banked first)."""
    assert BANKED is not None, "baseline.json is not banked"
    expected = set()
    for case in cases.all_cases():
        expected |= {f"{case.id}__-__{route}" for route in
                     ("price", "event_npv", "fold_drilldown_total", "fold_drilldown_npv")}
        for spec in case.legs():
            expected |= {f"{case.id}__{spec.name}__{route}" for route in ("event", "fold_total", "fold_drilldown")}
    assert set(RECORDS) == expected


def test_expected_moves_name_banked_records_and_d9_causes():
    for rid, move in EXPECTED_MOVES.items():
        assert rid in RECORDS, f"EXPECTED_MOVES names {rid!r}, which is not a banked record"
        assert move.cause in D9_CAUSES, f"{rid}: cause {move.cause!r} is not a D9 cause"


# banking order keeps one case's records adjacent, so a worker mostly reuses its cached case solves
@pytest.mark.parametrize("record_id", list(RECORDS) or ["<bank missing>"])
def test_record_reproduces_or_moves_by_its_cause(record_id):
    assert BANKED is not None, f"{BANK} is not banked"
    record = RECORDS[record_id]
    live = live_outcome(record)
    move = EXPECTED_MOVES.get(record_id)
    if move is not None:
        move.check(live, record["outcome"], record)
        return
    assert_reproduced(live, record["outcome"], record)
