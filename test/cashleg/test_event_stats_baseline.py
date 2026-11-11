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
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pytest

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL, close

HERE = Path(__file__).resolve().parent
BANK_DIR = HERE / "event_stats_baseline"
BANK = BANK_DIR / "baseline.json"


def _load(stem: str):
    """``event_stats_baseline/<stem>.py`` under a unique name (``cases`` is taken by other test dirs,
    and a ``cashleg`` package on ``sys.path`` would shadow the legacy ``cashleg`` import alias)."""
    name = f"_cashleg_event_stats_baseline_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BANK_DIR / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cases = _load("cases")
rebank_guard = _load("rebank_guard")

#: spec D9's causes a banked value may move for (landing 2 asserts each move against its cause)
D9_CAUSES: Dict[str, str] = {
    "a": "KI-after-KO removal",
    "b": "coupon cash out of KO cash",
    "c": "fixed defects (T2-T4)",
    "d": "maturity probability direct instead of 1 - sum(KO) - ki (moves by probability_residual)",
    "e": "KI stream on its own axis (AT_KI legs pay on KI dates, or stop raising)",
    "f": "MC statistics on price() paths (T3, D14)",
    "g": "REBASED now supported",
    "h": "engine or state newly producing native statistics (DCN PDE, terminated DCN/V2 states, "
         "QUAD V1 Phoenix KO-at-valuation coupon stream)",
    "i": "npv := price() where the marched stats pv differed by round-off (Heston ADI ~2 ULP)",
}


@dataclass(frozen=True)
class MovedBy:
    """A record that moves on purpose: ``cause`` is a ``D9_CAUSES`` key; ``check(live, banked, record)``
    asserts the live outcome against that cause (it must fail if the move has any other source)."""

    cause: str
    check: Callable[[dict, dict, dict], None]


def _bank():
    if not BANK.exists():
        return None
    with BANK.open() as fh:
        return json.load(fh)


BANKED = _bank()
RECORDS = {} if BANKED is None else {r["id"]: r for r in BANKED["records"]}


# --- moves (T6: EventDistribution on per-stream axes) ----------------------------------------------

def _live_distribution(record: dict):
    case = _case(record["case"])
    engine, product, env = case.fresh()
    return engine.price_with_events(product, env, **case.kwargs()).event_distribution, env


def _leg(record: dict):
    return next(spec.leg for spec in _case(record["case"]).legs() if spec.name == record["leg"])


def _ki_stream_on_its_own_axis(live: dict, banked: dict, record: dict) -> None:
    """Cause (e). At base the distribution dropped the KI stream (its length differed from the KO grid), so
    the AT_KI leg raised. Now the KI stream sits on its own axis, paid at the KI observation (D8), and the leg
    values exactly ``amount · Σ_k P_k · DF(t_k + offset)`` on it."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    dist, env = _live_distribution(record)
    times = np.asarray(dist.stream_times[EventType.KI], dtype=float)
    probability = np.asarray(dist.probabilities[EventType.KI], dtype=float)
    np.testing.assert_array_equal(np.asarray(dist.payment_times_for(EventType.KI)), times)       # D8
    assert times.size != base["n_event_times"] or not np.array_equal(times, base["event_times"]), (
        f"{record['id']}: the KI stream sits on the KO grid; base would not have dropped it")
    leg = _leg(record)
    offset = float(leg.payment_offset_days) / 365.0
    expected = float(leg.amount) * math.fsum(
        float(p) * env.get_discount_factor(float(t) + offset) for t, p in zip(times, probability))
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != Σ P·DF on the KI axis {expected!r}")


def _maturity_moves_by_the_residual(live: dict, banked: dict, record: dict) -> None:
    """Cause (d). ``from_statistics`` absorbs the statistics' ``probability_residual`` into MATURITY_NO_KO
    (spec D16); at base the converter kept the producer's direct pre-maturity probability (its normalisation
    ignored a drift under ``Tolerance.PROBABILITY``). With the absorption undone on the live distribution the
    leg gives back its banked value, and the move itself is more than round-off."""
    from dataclasses import replace

    from quantark.cashleg import EventType
    from quantark.util.numerical import Tolerance

    assert "value" in banked and "value" in live, (live, banked)
    dist, env = _live_distribution(record)
    residual = dist.probability_residual
    assert residual != 0.0 and abs(residual) <= Tolerance.PROBABILITY, residual
    leg = _leg(record)
    assert leg.value(dist, env, cases.UNIT) == float(live["value"])
    undone = replace(dist, probabilities={
        **dist.probabilities,
        EventType.MATURITY_NO_KO: dist.probabilities[EventType.MATURITY_NO_KO] - residual,
    })
    tol = GOLDEN_REL_TOL * abs(float(banked["value"]))
    assert close(leg.value(undone, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: undoing the absorbed residual does not give back the banked value")
    assert abs(float(live["value"]) - float(banked["value"])) > tol / 100.0, (
        f"{record['id']}: the move is round-off, not the residual")


#: T6: the KI stream that base dropped (every AT_KI event record that raised for it and now values)
_KI_AXIS_MOVES = (
    "mc_snowball__snowball_cki", "mc_phoenix__phoenix_mem_instant", "mc_phoenix__phoenix_nomem_instant",
    "mc_phoenix__phoenix_mem_expiry", "mc_phoenix__phoenix_nomem_expiry", "mc_lv_snowball__snowball_cki",
    "mc_lv_phoenix__phoenix_mem_instant", "mc_heston_snowball__snowball_cki", "mc_heston_phoenix__phoenix_mem_expiry",
    "mc_snowball__snowball_cki_knocked_in", "quad_v1_snowball__snowball_cki_knocked_in",
    "pde_snowball__snowball_cki_knocked_in", "adi_heston_snowball__snowball_cki_knocked_in",
    "mc_phoenix__phoenix_mem_instant_knocked_in", "quad_v1_phoenix__phoenix_mem_instant_knocked_in",
    "pde_phoenix__phoenix_mem_instant_knocked_in",
    "mc_snowball__snowball_cki_ko_at_valuation", "mc_snowball__snowball_cki_knocked_in_ko_at_valuation",
    "pde_snowball__snowball_cki_knocked_in_ko_at_valuation", "quad_v1_snowball__snowball_cki_knocked_in_ko_at_valuation",
    "mc_phoenix__phoenix_mem_instant_ko_at_valuation", "mc_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
    "pde_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
    "quad_v1_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
    "mc_snowball__snowball_reverse",
    "mc_snowball__snowball_cki_knocked_in_no_ko_after_ki", "pde_snowball__snowball_cki_knocked_in_no_ko_after_ki",
)
#: T6: D16 absorbs the PDE KO-reset buckets' march residual (-6.9e-10 .. -2.8e-9) into MATURITY_NO_KO; every
#: event record it moves by more than 1% of its tolerance (two of them beyond the tolerance itself)
_RESIDUAL_MOVES = tuple(
    f"{case}__{leg}__event"
    for case in ("pde_ko_reset__ko_reset_absolute_cki", "pde_ko_reset__ko_reset_absolute",
                 "pde_lv_ko_reset__ko_reset_absolute")
    for leg in ("fixed_at_maturity_no_ko", "fixed_at_maturity_any", "autocallable_margin_nmp_both",
                "autocallable_backend_premium_no_ko")
)

#: record id -> MovedBy; every other banked record must reproduce.
EXPECTED_MOVES: Dict[str, MovedBy] = {
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _ki_stream_on_its_own_axis) for case in _KI_AXIS_MOVES},
    **{rid: MovedBy("d", _maturity_moves_by_the_residual) for rid in _RESIDUAL_MOVES},
}


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


def test_rebanks_name_d9_causes_and_banked_cases():
    """A family re-banked in place (``generate.py --rebank``) names its commit, its D9 cause and its cases."""
    assert BANKED is not None, "baseline.json is not banked"
    for rebank in BANKED["header"].get("rebanks", []):
        assert rebank["cause"] in D9_CAUSES, rebank
        assert rebank["at_sha"] and rebank["cases"], rebank
        assert set(rebank["cases"]) <= set(BANKED["cases"]), rebank


def test_rebanks_list_every_record_they_moved():
    """Each re-bank lists the records it moved with old and new outcome, so a move absorbed in place
    stays visible: never a ``price`` record, ``event_npv`` only under the npv causes, and the chain of
    moves ends at the banked outcome."""
    assert BANKED is not None, "baseline.json is not banked"
    latest = {}
    for rebank in BANKED["header"].get("rebanks", []):
        assert "moved" in rebank, f"re-bank at {rebank['at_sha']} does not list its moved records"
        for move in rebank["moved"]:
            rid = move["id"]
            assert rid in RECORDS, move
            assert RECORDS[rid]["case"] in rebank["cases"], move
            assert move["route"] != "price", f"{rid}: a re-bank absorbed a price() move"
            if move["route"] == "event_npv":
                assert rebank["cause"] in rebank_guard.NPV_MOVE_CAUSES, move
            assert move["old"] != move["new"], move
            if rid in latest:
                assert move["old"] == latest[rid], f"{rid}: move chain broken"
            latest[rid] = move["new"]
    for rid, outcome in latest.items():
        assert RECORDS[rid]["outcome"] == outcome, f"{rid}: banked outcome is not the last re-banked one"


def _rec(rid, route, value):
    return dict(id=rid, route=route, outcome=dict(value=value))


def test_rebank_guard_lists_moves_and_keeps_unmoved_records_unlisted():
    banked = [_rec("a", "price", 1.0), _rec("b", "event_npv", 2.0), _rec("c", "event", 3.0)]
    fresh = [_rec("a", "price", 1.0), _rec("b", "event_npv", 2.5), _rec("c", "event", 3.0)]
    merged, moved = rebank_guard.merge_rebank(banked, fresh, cause="f")
    assert [r["outcome"]["value"] for r in merged] == [1.0, 2.5, 3.0]
    assert moved == [dict(id="b", route="event_npv", old=dict(value=2.0), new=dict(value=2.5))]


def test_rebank_guard_refuses_a_moved_price():
    banked = [_rec("a", "price", 1.0)]
    with pytest.raises(rebank_guard.RebankRefused, match="price"):
        rebank_guard.merge_rebank(banked, [_rec("a", "price", 1.0 + 1e-15)], cause="f")


def test_rebank_guard_refuses_an_npv_move_outside_the_npv_causes():
    banked = [_rec("b", "event_npv", 2.0)]
    with pytest.raises(rebank_guard.RebankRefused, match="event_npv"):
        rebank_guard.merge_rebank(banked, [_rec("b", "event_npv", 2.1)], cause="c")


def test_rebank_guard_refuses_a_record_not_in_the_bank():
    with pytest.raises(rebank_guard.RebankRefused, match="not in the bank"):
        rebank_guard.merge_rebank([_rec("a", "price", 1.0)], [_rec("z", "event", 1.0)], cause="f")


def test_every_mc_case_is_rebanked_on_price_paths():
    """T3 (spec D14, AC5): the MC rows were re-banked once statistics came from price()'s own paths."""
    assert BANKED is not None, "baseline.json is not banked"
    mc_cases = {cid for cid, meta in BANKED["cases"].items() if "mc" in meta["engine_kind"].split("_")}
    rebanked = {cid for rebank in BANKED["header"].get("rebanks", []) if rebank["cause"] == "f"
                for cid in rebank["cases"]}
    assert mc_cases and mc_cases <= rebanked


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
