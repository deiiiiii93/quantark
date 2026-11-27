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
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
    "d": "maturity outcome direct instead of 1 - sum(KO) - ki (moves by probability_residual) and on its own "
         "determination/payment row (KO-reset ABSOLUTE: maturity row + F5 survival instead of the legacy "
         "one-maturity layout; probabilities unchanged)",
    "e": "KI stream on its own axis (AT_KI legs pay on KI dates, or stop raising)",
    "f": "MC statistics on price() paths (T3, D14)",
    "g": "REBASED now supported",
    "h": "engine or state newly producing native statistics (DCN PDE, terminated DCN/V2 states, "
         "QUAD V1 Phoenix KO-at-valuation coupon stream)",
    "i": "npv := price() where the marched stats pv differed by round-off (Heston ADI ~2 ULP)",
    "k": "PDE knocked-in edge row honours the annualized-KI leg (D26): edge-reaching annualized-KI PDE / ADI "
         "price, event_npv and folded records",
    "l": "PDE knocked-in edge row honours the airbag (D27): partially protected PDE / ADI contracts whose airbag "
         "changes the knocked-in payoff on the knock-in edge cell (no banked case is one: "
         "test_no_banked_case_is_in_the_d27_airbag_class)",
    "m": "D29 termination survival: settled notes leave after their settlement (a maturity-type termination is alive "
         "at its determination and gone right after, past the note's last node too)",
    "n": "D30 KO-reset EXPIRY schedule-1 pay date: under EXPIRY coupon pay a first-schedule knock-out is paid at "
         "schedule 1's maturity (plus the terminal delay), not the product maturity, on PDE 1D / LV PDE / ADI, QUAD V1, "
         "QUAD V2 and the fold structure (no banked case is one: "
         "test_no_banked_case_is_in_the_d30_ko_reset_expiry_class)",
    "o": "D28 partial-protection edge row: partially protected PDE / ADI contracts whose floor does not bind on the "
         "knocked-in knock-in edge cell read that row off the payoff (no banked case is partially protected: "
         "test_no_banked_case_is_in_the_d28_partial_protection_class)",
    "p": "D31 MC knocked-in spot at schedule-2 end: the KO-reset MC kernel (BSM, LocalVol, Heston, Heston-SLV MC) "
         "reads an ABSOLUTE (carried knock-in included) knocked-in payoff's spot at schedule 2's end node, where it "
         "settles, not the grid's last node (schedule 1's end when that runs on past schedule 2) (no banked case was "
         "one; replaced by cause (u), run 9, whose census is test_the_run_nine_exercise_readout_class_is_pinned)",
    "q": "D33 settlement precedence: a KO-reset note that ends with schedule 1 (the never-knocked-in maturity; under "
         "EXPIRY also a schedule-1 knock-out) is paid on the product's explicit settlement_date when booked, the "
         "settlement delay only without one, on PDE 1D / LV PDE / ADI, QUAD V1, QUAD V2 and the fold structure (no "
         "banked case is one: test_no_banked_case_is_in_the_d33_settlement_precedence_class)",
    "s": "QUAD V1 exact first step (sub-resolution run, 0.12.0, owner-signed 2026-11-23): a valuation-to-first-event "
         "step spanning fewer than 13 cells is priced at the spot on the pre-event surfaces instead of the lattice "
         "readout. Checked by recomputing every QUAD V1 record with the exact first step off "
         "(``_without_exact_first_step``): that must reproduce the bank under every other cause, so the first step is "
         "the only new source of a move; ``test_quad_v1_first_step_moves_are_bounded`` measures the moves themselves",
    "t": "QUAD V1 KO-reset knock-outs by the cell-average projection (0.16.0 D1, QUAD numerics run 5): under the "
         "default event_projection both schedules' knock-outs are Snowball's dual-cell projection instead of the "
         "nodal cosine smoothing. Checked by recomputing every QUAD V1 record with the previous nodal knock-outs "
         "(``_legacy_ko_reset_knock_outs``, the 0.15 code path bit for bit) on top of cause (s)'s first step off: that "
         "must reproduce the bank under every other cause, so D1 is the only new source of a move; "
         "``test_ko_reset_v1_projection_moves_are_bounded`` measures the moves themselves",
    "u": "run 9 knocked-in KO-reset put at the exercise date (owner ruling 2026-11-27, replaces D31): a knocked-in "
         "KO-reset not knocked out is a live put to the PRODUCT'S EXERCISE DATE -- every engine reads V1's spot at "
         "the exercise date and pays it at the product terminal (the explicit settlement_date when booked, else the "
         "exercise date plus the settlement delay), ABSOLUTE, carried knock-in and REBASED alike -- and annualized "
         "knocked-in amounts accrue from the origin to the exercise date (O2). Only notes whose schedule 2 can end "
         "before the exercise move; in the bank that is the MC REBASED case (an early knock-in ends schedule 2 "
         "before exercise 2.0): its price and event_npv, checked against the price's own paths "
         "(``_put_at_exercise_on_the_same_paths``); census test_the_run_nine_exercise_readout_class_is_pinned",
    "r": "D34 intraday twin carries the D33 payment: a numeric KO-reset with first_schedule_settlement_time (the "
         "intraday twin of a dated D33-class note) is paid there; only intraday prices of D33-class notes move (no "
         "banked case carries it: test_no_banked_case_is_in_the_d34_intraday_twin_class)",
}


@dataclass(frozen=True)
class MovedBy:
    """A record that moves on purpose: ``cause`` is a ``D9_CAUSES`` key; ``check(live, banked, record)``
    asserts the live outcome against that cause (it must fail if the move has any other source).

    ``within_tolerance``: the move is pinned by its check but smaller than the bank tolerance (the record would also
    reproduce), so the AC19 census does not require it to break reproduction; every other listed record must."""

    cause: str
    check: Callable[[dict, dict, dict], None]
    within_tolerance: bool = False


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


def _carried_knock_in_is_past(live: dict, banked: dict, record: dict) -> None:
    """Cause (e), "stop raising", pinned to the definition. At base the distribution dropped the KI stream of a note
    already knocked in (one row against the KO grid), so the AT_KI leg raised. The knock-in is a past event (T23
    ruling R1, DESIGN §4.9c): cash it decided is paid or a receivable, never a flow of the remaining life. Pinned
    without the conversion under test:

    * the product carries the knock-in (``_otc_lifecycle_knocked_in``, the flag the engines read);
    * the statistics keep DESIGN §4.5's carried row: KI table exactly [(0, 1)], P(knock in) = 1, unflagged;
    * the distribution holds the KI stream on its own axis, empty, marked ``knock_in_carried``, and the leg is
      exactly 0 (a stand-in ``amount · DF(offset)`` -- the knock-in counted as a future event -- fails it)."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    assert record["features"].get("knocked_in_at_valuation") is True, f"{record['id']}: not a carried knock-in"
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    case = _case(record["case"])
    engine, product, env = case.fresh()
    assert getattr(product, "_otc_lifecycle_knocked_in", False) is True
    stats = engine.calculate_event_stats(product, env)
    np.testing.assert_array_equal(np.asarray(stats.ki_table.times, dtype=float), [0.0])
    np.testing.assert_array_equal(np.asarray(stats.ki_table.probability, dtype=float), [1.0])
    assert stats.knock_in.probability == 1.0 and stats.flag("ki_table") is None
    dist, env = _live_distribution(record)
    assert dist.knock_in_carried
    assert np.asarray(dist.stream_times[EventType.KI]).size == 0
    assert np.asarray(dist.probabilities[EventType.KI]).size == 0
    assert EventType.KI not in dist.flagged_streams, dist.flagged_streams
    assert float(live["value"]) == 0.0, f"{record['id']}: live {live['value']!r} != 0 (the knock-in is past)"


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


#: T6 (CP2): the carried knock-in base dropped: every AT_KI event record of a note already knocked in that
#: raised for it and now values. Legacy KI-ever tables (MC continuous KI, KO at valuation without a knock-in)
#: are not carried and reproduce their banked raise until the MC producers migrate.
_KI_AXIS_MOVES = (
    "mc_snowball__snowball_cki_knocked_in", "quad_v1_snowball__snowball_cki_knocked_in",
    "pde_snowball__snowball_cki_knocked_in", "adi_heston_snowball__snowball_cki_knocked_in",
    "mc_phoenix__phoenix_mem_instant_knocked_in", "quad_v1_phoenix__phoenix_mem_instant_knocked_in",
    "pde_phoenix__phoenix_mem_instant_knocked_in",
    "mc_snowball__snowball_cki_knocked_in_ko_at_valuation",
    "pde_snowball__snowball_cki_knocked_in_ko_at_valuation", "quad_v1_snowball__snowball_cki_knocked_in_ko_at_valuation",
    "mc_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
    "pde_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
    "quad_v1_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation",
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
#: T23 census: the two residual moves beyond the bank tolerance (measured); the other ten are pinned within it
_RESIDUAL_MOVES_BEYOND_TOLERANCE = ("pde_ko_reset__ko_reset_absolute_cki__fixed_at_maturity_no_ko__event",
                                    "pde_ko_reset__ko_reset_absolute_cki__fixed_at_maturity_any__event")

# --- moves (T9: MC Snowball / KO-reset produce EventStatistics) ------------------------------------------
#
# Every check below decides the outcome from the price's own paths (``mc_same_path_reference``: a plain price()
# with a spy on the payoff kernel, outcomes decided from the definitions in plain NumPy), never from the
# statistics object it checks, and compares the leg against that independent number.


def _same_path():
    from test_asset.equity.engine_mc import mc_same_path_reference

    return mc_same_path_reference


def _leg_on_rows(leg, env, times, probability) -> float:
    offset = float(leg.payment_offset_days) / 365.0
    return leg.sign() * float(leg.amount) * float(sum(
        p * env.get_discount_factor(float(t) + offset) for t, p in zip(times, probability)))


def _first_knock_in_rows(record: dict, *, before_ko: bool = True):
    """(row times, P per row, env) of the first knock-in on the price's own paths: before any knock-out
    (``before_ko``) or ever (the legacy KI-ever table). Continuous: trading days of the window, mass by
    brute-force step overlap (D18); discrete: the product's KI rows; a knock-out at valuation: no row."""
    same_path = _same_path()
    case = _case(record["case"])
    engine, product, env = case.fresh()
    family = BANKED["cases"][record["case"]]["family"]
    if family == "dcn":
        # T11: the DCN's own paths and contract rules (daily closes; never monitored after a knock-out)
        ref = same_path.dcn_paths(engine, product, env, **case.kwargs())
        if ref["carried"]:
            return np.array([0.0]), np.array([1.0]), env
        ctx, first = ref["ctx"], ref["first_col"]
        counts = np.bincount(ref["ki_col"][ref["ki_col"] >= 0], minlength=ctx.times.size)[first:]
        return np.asarray(ctx.times[first:], dtype=float), counts / ref["n"], env
    if family.startswith("ko_reset"):
        ref = same_path.ko_reset_paths(engine, product, env)
        mask = ref["ki_first"] if before_ko else (ref["ki_index"] >= 0)
        knocked_out_now = False
    else:
        # T10: Phoenix by its own per-path reference (same first-KO / first-KI / KO-wins-a-tie decisions)
        ref = (same_path.phoenix_paths if family == "phoenix" else same_path.snowball_paths)(engine, product, env)
        mask = ref["ki_first"] if before_ko else ref["ki_ever"]
        knocked_out_now = bool(np.all(ref["ko_time"] == 0.0))
    n = ref["n"]
    if ref.get("carried"):
        return np.array([0.0]), np.array([1.0]), env
    if before_ko and knocked_out_now:
        return np.zeros(0), np.zeros(0), env
    first = np.asarray(ref["ki_index"])[mask]
    steps = np.asarray(ref["step_ends"], dtype=float)
    if ref["continuous"]:
        days = same_path.numeric_trading_days(float(steps[-1]), env.bus_days_in_year)
        return days, same_path.day_overlap(steps, np.bincount(first, minlength=steps.size) / n, days), env
    return steps, np.bincount(first, minlength=steps.size) / n, env


def _knock_in_on_its_own_axis_before_any_ko(live: dict, banked: dict, record: dict) -> None:
    """Cause (e). Base dropped the KI stream (a KI-ever table off the KO grid, or KO-reset, which emitted
    none), so the AT_KI leg raised. Now the KI table is on its own axis (continuous: trading days, D18) and holds
    first knock-ins before any knock-out: the leg equals ``sign · amount · Σ P_row · DF(t_row + offset)`` with
    the rows decided on the price's own paths here (a knock-out at valuation leaves no row: 0)."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    times, probability, env = _first_knock_in_rows(record)
    dist, _ = _live_distribution(record)
    np.testing.assert_allclose(np.asarray(dist.stream_times[EventType.KI], dtype=float), times, rtol=0, atol=1e-15)
    assert EventType.KI not in dist.flagged_streams, dist.flagged_streams
    expected = _leg_on_rows(_leg(record), env, times, probability)
    if times.size == 0:                                     # no knock-in row: the leg is exactly 0
        assert float(live["value"]) == 0.0, live
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != same-path KI-before-KO value {expected!r}")


def _knock_in_after_ko_removed(live: dict, banked: dict, record: dict) -> None:
    """Cause (a). Discrete KI on the KO dates: base kept the legacy KI-ever table on the KO grid. On the price's
    own paths the banked value is the leg on the first knock-in EVER (a knock-in after a knock-out included), and
    the live value the leg on the first knock-in before any knock-out."""
    assert "value" in banked and "value" in live, (live, banked)
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" in base["streams"], f"{record['id']}: base had no KI stream"
    leg = _leg(record)
    times, ever, env = _first_knock_in_rows(record, before_ko=False)
    tol = GOLDEN_REL_TOL * abs(float(banked["value"]))
    assert close(_leg_on_rows(leg, env, times, ever), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: the banked value is not the KI-ever leg on the same paths")
    times, before, env = _first_knock_in_rows(record)
    expected = _leg_on_rows(leg, env, times, before)
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"]))
    assert abs(float(live["value"]) - float(banked["value"])) > tol, f"{record['id']}: no KI after a KO to remove"


def _ko_reset_maturity_on_its_own_date(live: dict, banked: dict, record: dict) -> None:
    """Cause (d), MC (T9) and QUAD V1 (T17) KO-reset ABSOLUTE. Base read the legacy statistics through the one-maturity
    layout:
    MATURITY_NO_KO = 1 - ΣKO - ki at the last KO record (2y), paid with the knocked-in settlement, and survival
    1 - ΣKO. Now the maturity row is computed directly on its own determination (the first schedule's end, 1y,
    paid at its own settlement) and the survival drops that mass there (F5). The probabilities themselves do not
    move (same paths). Undoing the layout on the live distribution gives back the banked value."""
    from dataclasses import replace

    from quantark.cashleg import EventType

    assert "value" in banked and "value" in live, (live, banked)
    dist, env = _live_distribution(record)
    base = BANKED["cases"][record["case"]]["event_distribution"]
    for name, total in base["probability_totals"].items():
        live_total = float(np.sum(np.asarray(dist.probabilities[EventType[name]], dtype=float)))
        assert abs(live_total - total) <= 1e-12, f"{record['id']}: {name} {live_total!r} != banked {total!r}"
    ko_times = np.asarray(dist.stream_times[EventType.KO], dtype=float)
    np.testing.assert_array_equal(ko_times, np.asarray(base["event_times"], dtype=float))
    survival = 1.0 - np.cumsum(np.asarray(dist.probabilities[EventType.KO], dtype=float))
    last = float(ko_times[-1])
    undone = replace(
        dist, survival_times=np.r_[0.0, ko_times], survival_probability=np.r_[1.0, survival],
        payment_times={**dist.payment_times,
                       EventType.MATURITY_NO_KO: dist.payment_times[EventType.MATURITY_WITH_KI]},
        determination_times={EventType.MATURITY_NO_KO: last, EventType.MATURITY_WITH_KI: last},
        terminal_times=np.zeros(0), terminal_probability=np.zeros(0),       # base had no D29 terminal steps
    )
    leg = _leg(record)
    assert leg.value(dist, env, cases.UNIT) == float(live["value"])
    tol = GOLDEN_REL_TOL * abs(float(banked["value"]))
    assert close(leg.value(undone, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: undoing the one-maturity layout does not give back the banked value")
    assert abs(float(live["value"]) - float(banked["value"])) > tol, f"{record['id']}: the move is round-off"
    if record["id"] in _D29_MOVES:
        # D29 moves this leg too (T23 follow-up 2): live is D29's leg from probabilities alone (MC: the paths)
        expected = _leg_by_rule(record["case"], record["leg"], d29=True)
        bound = (1e-12 * float(record["scale"]) if "mc" in record["engine_kind"]
                 else max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"])))
        assert abs(float(live["value"]) - expected) <= bound, f"{record['id']}: != D29 leg {expected!r}"


def _rebased_distribution_on_the_same_paths(record: dict):
    """The live REBASED distribution against the price's own paths (the kernel's outcome, determination and
    settlement per path): every KO and knocked-in settlement row's probability, the never-knocked-in maturity,
    the survival at every axis time; the KO axis is every first-schedule date and every knock-in date + offset."""
    from quantark.cashleg import EventType

    dist, env = _live_distribution(record)
    case = _case(record["case"])
    engine, product, _ = case.fresh()
    ref = _same_path().ko_reset_paths(engine, product, env)
    n = ref["n"]
    assert dist.rebased and abs(dist.probability_residual) <= 1e-12
    for event_type, events in ((EventType.KO, ("ko_pre", "ko_post")), (EventType.MATURITY_WITH_KI,
                                                                        ("ki_settlement",))):
        times = np.asarray(dist.stream_times[event_type], dtype=float)
        pays = np.asarray(dist.payment_times[event_type], dtype=float)
        probability = np.asarray(dist.probabilities[event_type], dtype=float)
        in_event = np.isin(ref["event"], events)
        covered = 0
        for t, p in set(zip(times.tolist(), pays.tolist())):
            rows = (times == t) & (pays == p)
            count = np.count_nonzero(in_event & (ref["det"] == t) & (ref["settle"] == p))
            covered += count
            assert abs(float(probability[rows].sum()) - count / n) <= 1e-15, (event_type, t, p)
        assert covered == np.count_nonzero(in_event), f"{event_type}: a path off the axis"
    assert abs(float(dist.probabilities[EventType.MATURITY_NO_KO]) - np.count_nonzero(ref["v0"]) / n) <= 1e-15
    # survival by spec D29: a knock-out leaves at its row; a maturity-type termination (maturity, knocked-in
    # settlement) is alive at its determination and gone after it
    knocked_out = np.isin(ref["event"], ("ko_pre", "ko_post"))
    for t, survival in zip(dist.survival_times, dist.survival_probability):
        gone = (knocked_out & (ref["det"] <= t)) | (~knocked_out & (ref["det"] < t))
        assert abs(survival - (1.0 - np.count_nonzero(gone) / n)) <= 1e-14, (t, survival)
    grid = ref["grid"]
    candidates = set(np.asarray(grid["pre_times"], dtype=float).tolist())
    for ki_time in np.asarray(grid["ki_times"], dtype=float):
        candidates |= set((ki_time + np.asarray(grid["post_times"], dtype=float)).tolist())
    assert set(np.asarray(dist.stream_times[EventType.KO], dtype=float).tolist()) == candidates
    return dist, env


def _rebased_now_supported(live: dict, banked: dict, record: dict) -> None:
    """Cause (g). Base refused REBASED on the event route (``NotImplementedError``: offsets). Now the statistics
    carry the absolute termination distribution (D13): the npv is price() bitwise; a leg's value is the leg on
    the distribution, which is checked row by row against the price's own paths; an AutocallableCashLeg booked
    on the contract's dates refuses the candidate axis (T6 ruling: no interpolation), a different refusal."""
    assert banked.get("raises") == "NotImplementedError", banked
    if record["route"] == "event_npv":
        price = RECORDS[f"{record['case']}__-__price"]["outcome"]
        assert live == {"value": price["value"]}, (live, price)
        return
    dist, env = _rebased_distribution_on_the_same_paths(record)
    leg = _leg(record)
    if record["leg"].startswith("autocallable_"):
        assert live.get("raises") == "ValidationError", live
        assert "schedule" in live["message"] or "coupon stream" in live["message"], live
        return
    assert live.get("value") == leg.value(dist, env, cases.UNIT), live


_T9_KI_AXIS = ("mc_snowball__snowball_cki", "mc_heston_snowball__snowball_cki", "mc_lv_snowball__snowball_cki",
               "mc_snowball__snowball_reverse", "mc_snowball__snowball_cki_ko_at_valuation",
               "mc_snowball__ko_reset_absolute", "mc_snowball__ko_reset_absolute_cki")
_T9_KI_AFTER_KO = ("mc_snowball__snowball_dki", "mc_heston_snowball__snowball_dki")
_T9_KO_RESET_MATURITY = tuple(
    f"{case}__{leg}__event"
    for case in ("mc_snowball__ko_reset_absolute", "mc_snowball__ko_reset_absolute_cki")
    for leg in ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_any")
)
_T9_REBASED = ("mc_snowball__ko_reset_rebased__-__event_npv",) + tuple(
    f"mc_snowball__ko_reset_rebased__{leg}__event"
    for leg in ("deterministic", "accrual_truncate_enter", "accrual_truncate_complete",
                "accrual_truncate_at_maturity", "accrual_full_schedule", "fixed_at_ko", "fixed_at_ki",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_with_ki", "fixed_at_maturity_any",
                "autocallable_margin_nmp_both", "autocallable_backend_premium_no_ko",
                "autocallable_rebate_with_ki", "autocallable_backend_interest_coupon")
)

# --- moves (T10: MC Phoenix and the vol-model Phoenix engines produce EventStatistics) -------------------------
#
# Every Phoenix record in the bank is a continuous-KI contract: base kept MC's KI-ever table, off the KO grid, so the
# distribution dropped the KI stream and the AT_KI leg raised. Coupon, KO and maturity legs reproduce (the coupon
# cash was already the price's since T2; the coupon probability, KO rows and maturity branches are the same path
# counts). Cause (e), checked by ``_knock_in_on_its_own_axis_before_any_ko`` against ``phoenix_paths``.
_T10_KI_AXIS = ("mc_phoenix__phoenix_mem_instant", "mc_phoenix__phoenix_nomem_instant",
                "mc_phoenix__phoenix_mem_expiry", "mc_phoenix__phoenix_nomem_expiry",
                "mc_phoenix__phoenix_mem_instant_ko_at_valuation", "mc_lv_phoenix__phoenix_mem_instant",
                "mc_heston_phoenix__phoenix_mem_expiry")

# --- moves (T11: MC DCN and the vol-model DCN engines produce EventStatistics) -----------------------------------
#
# Base read the DCN MC statistics through the legacy converter: no KI stream (an AT_KI leg raised), no COUPON stream
# (a COUPON-basis AutocallableCashLeg raised), and a knocked-out lifecycle state got the maturity-only trivial
# distribution (its legs valued as if the note ran to maturity). KO, maturity and fold-route records reproduce (same
# paths, same counts, the same DCN axis, spec D12).

def _dcn_coupon_stream_now_native(live: dict, banked: dict, record: dict) -> None:
    """Cause (h). Base had no COUPON stream on DCN MC (the legacy statistics' coupon probability was flagged), so a
    COUPON-basis AutocallableCashLeg raised. Now the coupon table is native: P(the row's fixed coupon is paid) on the
    monthly rows, paid at the coupon payment. The leg is recomputed here by its own formula from the coupons and the
    two survivor branches counted on the price's own paths (``dcn_paths``)."""
    from quantark.cashleg import EventType
    from quantark.cashleg.autocallable_leg import PvFormula

    assert banked.get("raises") == "ValidationError" and "coupon stream" in banked.get("message", ""), banked
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "COUPON" not in base["streams"], f"{record['id']}: base had a COUPON stream; the move is not cause (h)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    case = _case(record["case"])
    engine, product, env = case.fresh()
    ref = _same_path().dcn_paths(engine, product, env, **case.kwargs())
    n = ref["n"]
    leg = _leg(record)
    assert leg.accrual_basis.value == "coupon" and leg.pv_formula is PvFormula.NORMAL
    coupon = ref["coupon_paid"].sum(axis=0) / n
    np.testing.assert_allclose(np.asarray(leg.observation_schedule, dtype=float),
                               ref["ctx"].times[ref["ctx"].obs_cols], rtol=0, atol=1e-15)
    contingent = float(np.sum(np.asarray(leg.accrual_factors) * coupon
                              * np.array([env.get_discount_factor(float(t)) for t in leg.settlement_schedule])))
    terminal = {EventType.MATURITY_NO_KO: np.count_nonzero(ref["maturity"]) / n,
                EventType.MATURITY_WITH_KI: np.count_nonzero(ref["ki_settlement"]) / n}
    contingent += float(leg.terminal_accrual_factor) * sum(terminal[e] for e in leg.terminal_events) * float(
        env.get_discount_factor(float(leg.terminal_settlement_time)))
    expected = leg.sign() * float(leg.notional) * float(leg.rate) * contingent
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != same-path coupon-basis value {expected!r}")


def _dcn_terminated_state_has_no_event(live: dict, banked: dict, record: dict) -> None:
    """Cause (h), AC17. A knocked-out lifecycle state: base had no statistics and handed legs the maturity-only
    trivial distribution, so an event-dependent leg valued the note as if it ran to maturity -- the banked value is
    exactly that leg on ``EventDistribution.trivial(T)``. Now the statistics are the receivables only (unpaid; their
    rows are the ledger's flows) or ``None`` (settled), the distribution is the ended note's and the leg refuses it."""
    from quantark.cashleg import EventDistribution

    assert "value" in banked, banked
    assert record["features"].get("lifecycle", "").startswith("terminated_"), record["id"]
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert base["streams"] == ["MATURITY_NO_KO"] and base["probability_totals"] == {"MATURITY_NO_KO": 1.0}, base
    case = _case(record["case"])
    engine, product, env = case.fresh()
    trivial = EventDistribution.trivial(max(float(product.get_maturity(env)), 0.0))
    tol = GOLDEN_REL_TOL * max(abs(float(banked["value"])), 1.0)
    assert close(_leg(record).value(trivial, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: the banked value is not the leg on the trivial distribution")
    assert live.get("raises") == "ValidationError" and "terminated note" in live.get("message", ""), live
    dist, _ = _live_distribution(record)
    assert dist.terminated and dist.probabilities == {}
    state = case.kwargs()["lifecycle_state"]
    stats = case.fresh()[0].calculate_event_stats(product, env, lifecycle_state=state)
    pending = state.ledger.pending(state.valuation_point)
    if not pending:
        assert stats is None
        return
    assert stats.terminated and stats.receivables.cash_to_pay.tolist() == [f.amount for f in pending]
    assert stats.receivables.payment_dates == tuple(f.payment_date for f in pending)


_T11_DCN_CASES = ("mc_dcn__dcn", "mc_dcn__dcn_seller", "mc_dcn__dcn_receivable_alive", "mc_lv_dcn__dcn",
                  "mc_heston_dcn__dcn")
_T11_TERMINATED = tuple(
    f"mc_dcn__{case}__{leg}__event"
    for case in ("dcn_ko_unpaid", "dcn_ko_settled")
    for leg in ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_any")
)

# --- moves (T12: PDE 1D Snowball / Phoenix / KO-reset produce EventStatistics) ---------------------------------
#
# A grid engine has no same-path reference: each check pins the move to the product's own valuation-date decision
# or to the legacy converter's definitional layout undone on the live distribution, never to the statistics object
# it checks.


def _knocked_out_at_valuation_has_no_knock_in_row(live: dict, banked: dict, record: dict) -> None:
    """Cause (e), PDE 1D. Base carried no KI stream (the legacy PDE table was empty and the converter dropped it), so
    the AT_KI leg raised. A note knocked out at valuation without a carried knock-in has, by the product's own
    valuation-date decision (DESIGN §4.5), no knock-in at all: the KI stream exists on its own axis and is empty, so
    the leg is exactly 0. The decision is read from the product, independent of the statistics."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv", "quad_v2", "quad_v1"), record["engine_kind"]
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    _, product, env = _case(record["case"]).fresh()
    carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    decision = product.decide_observations_at_valuation(env.spot, env, knocked_in=carried)
    assert decision.knocked_out and not decision.knocked_in, f"{record['id']}: not knocked out at valuation alone"
    assert "value" in live, f"{record['id']}: still raises {live}"
    dist, _ = _live_distribution(record)
    assert np.asarray(dist.stream_times[EventType.KI]).size == 0, dist.stream_times[EventType.KI]
    assert EventType.KI not in dist.flagged_streams, dist.flagged_streams
    assert float(live["value"]) == 0.0, f"{record['id']}: live {live['value']!r} != 0 (no knock-in row)"


def _knocked_in_settlement_marched_directly(live: dict, banked: dict, record: dict) -> None:
    """Cause (d), PDE 1D, a note knocked in at valuation. Base converted the legacy statistics by the one-maturity
    split ``(1 - ΣKO - ki, ki)`` with ``ki`` clipped to ``1 - ΣKO`` -- on a knocked-in note the marched settlement unit
    exceeds ``1 - ΣKO`` by the march's discount residual, so base's MATURITY_WITH_KI was ``1 - ΣKO``. Now it is the
    marched column itself and the residual is carried (D16; MATURITY_NO_KO is exactly 0 and cannot absorb it). With
    the legacy split undone on the live distribution the leg gives back its banked value; the move is the residual,
    more than round-off."""
    from dataclasses import replace

    from quantark.cashleg import EventType
    from quantark.util.numerical import Tolerance

    assert "value" in banked and "value" in live, (live, banked)
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv") and record["features"].get("knocked_in_at_valuation")
    dist, env = _live_distribution(record)
    residual = dist.probability_residual
    assert residual < 0.0 and abs(residual) <= Tolerance.PROBABILITY, residual
    assert dist.probabilities[EventType.MATURITY_NO_KO] == 0.0
    case = _case(record["case"])
    engine, product, env = case.fresh()
    stats = engine.calculate_event_stats(product, env)
    assert dist.probabilities[EventType.MATURITY_WITH_KI] == stats.ki_settlement.probability
    assert stats.maturity.probability == 0.0                       # a knocked-in note never matures not knocked in
    leg = _leg(record)
    assert leg.value(dist, env, cases.UNIT) == float(live["value"])
    ko_total = float(np.sum(dist.probabilities[EventType.KO]))
    legacy = replace(dist, probabilities={**dist.probabilities, EventType.MATURITY_WITH_KI: max(0.0, 1.0 - ko_total)})
    tol = GOLDEN_REL_TOL * abs(float(banked["value"]))
    assert close(leg.value(legacy, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: the legacy split undone does not give back the banked value")
    assert abs(float(live["value"]) - float(banked["value"])) > tol / 100.0, (
        f"{record['id']}: the move is round-off, not the residual")


# --- moves (T13: the PDE 1D forward adjoint pass fills the KI table) -------------------------------------------
#
# Base had no PDE KI stream, so every PDE AT_KI leg raised. Each check pins the new stream to the product's own KI
# schedule, to independent backward numbers (the disable_ko_after_ki twin's KI settlement; discrete rows: the twins
# cut to their first k KI dates) and, on flat vol, to an RQMC run row by row.

#: the discrete KI rows' MC reference tolerance: 4 batch SEs (floored at the binomial SE of the PDE's own row) plus
#: 1e-3, the stated allowance for the MC's own time discretisation (daily steps; local vol Euler) -- a row moved to
#: another date (the legacy KO-grid fallback, an equal split) is off by ~1e-2
_KI_ROW_MC_ALLOWANCE = 1e-3


def _mc_knock_in_rows(product, env):
    """(rows, SEs, paths) of P(first KI at each discrete KI date before any KO) by RQMC, 16 independent scrambles x
    4096 paths, on the record's own product and (flat) market."""
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.util.enum.engine_enums import MonteCarloMethod

    params = MCParams(num_paths=4096, seed=13, rqmc_min_batches=16, rqmc_max_batches=16, rqmc_target_std=1e-12,
                      rqmc_paths_mode="per_batch", time_steps=504)
    engine = SnowballMCEngine(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI)
    stats = engine.calculate_event_stats(product, env)
    return (np.asarray(stats.ki_table.probability, dtype=float),
            np.asarray(stats.standard_errors["ki_table.probability"], dtype=float), 16 * 4096)


def _ki_schedule_cut(product, k, env):
    """``product`` with its KI schedule cut to its first ``k`` observations and ``disable_ko_after_ki`` set: its
    knocked-in settlement is P(first KI at one of those k dates before any KO) of the original note."""
    from dataclasses import replace

    from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule

    bc = product.barrier_config
    schedule = (None if bc.ki_observation_schedule is None
                else ObservationSchedule(records=list(bc.ki_observation_schedule.records[:k])))
    dates = None if bc.ki_observation_dates is None else list(bc.ki_observation_dates[:k])
    product.barrier_config = replace(bc, ki_observation_schedule=schedule, ki_observation_dates=dates,
                                     disable_ko_after_ki=True)
    product.validate()
    assert len(product.resolve_ki_observations(env)) == k
    return product


def _pde_knock_in_before_ko_on_its_own_axis(live: dict, banked: dict, record: dict) -> None:
    """Cause (e), PDE 1D (T13), native Heston / SLV ADI (T14), QUAD V2 (T16) and QUAD V1 (T17). Base carried no KI
    stream (the legacy PDE / ADI table was empty and the converter dropped it; QUAD V2's and QUAD V1's legacy statistics
    had no table), so the AT_KI leg raised. Now the forward adjoint pass fills the KI table (the PDE 1D pass; the ADI
    pass, ``pde/vol_model_adjoint.py``; the QUAD V2 pass, ``quad/v2/adjoint.py``; the QUAD V1 pass, the transpose of
    its backward step, ``quad/v1_adjoint.py``). Pinned without the statistics object it checks:

    * the axis is the product's own: its KI observations up to the knock-in horizon (discrete), or one numeric
      trading day ``k / bus_days_in_year`` per row up to the horizon (continuous, AC13);
    * the total is P(KI before any KO), checked against an independent backward number: the knocked-in settlement
      of the note's ``disable_ko_after_ki`` twin (its knocked-in surface never knocks out, so every first knock-in
      before a knock-out settles knocked in; same not-knocked-in march, a same-instant tie to the knock-out), within
      the march's discount residual (T12 RESIDUAL_BOUND 1e-8; the ADI discounts exactly: round-off). A KI-ever table
      (a knock-in after a knock-out counted -- the ADI's removed KO-independent monitor) or the note's own settlement
      (no knock-out after a knock-in) fails it;
    * discrete KI, row by row: the first k rows' sum against the KI settlement of the twin cut to its first k KI
      observations, within 1e-8 (a row moved to another date fails it, even on a KI schedule that is the KO grid;
      CP4: possible since the statistics settle the knock-in-side edge knocked in only where the price's row does);
      on flat vol also each row against an RQMC run within 4 batch SEs plus a stated 1e-3 allowance;
    * the leg is ``sign · amount · Σ P_row · DF(t_row + offset)`` on those rows (D8)."""
    from dataclasses import replace

    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.cashleg import EventType
    from quantark.util.enum import ObservationType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv", "pde_adi", "pde_adi_slv", "quad_v2", "quad_v1"), (
        record["engine_kind"])
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    case = _case(record["case"])
    engine, product, env = case.fresh()
    horizon = float(product.get_pre_maturity_time(env) if isinstance(product, KnockOutResetSnowballOption)
                    else product.get_maturity(env))
    bc = product.barrier_config
    if bc.ki_continuous or bc.ki_observation_type == ObservationType.CONTINUOUS:
        times = _same_path().numeric_trading_days(horizon, env.bus_days_in_year)
    else:
        times = np.array([r.observation_time for r in product.resolve_ki_observations(env)
                          if r.observation_time <= horizon + 1e-10], dtype=float)
    dist, _ = _live_distribution(record)
    np.testing.assert_allclose(np.asarray(dist.stream_times[EventType.KI], dtype=float), times, rtol=0, atol=1e-15)
    assert EventType.KI not in dist.flagged_streams, dist.flagged_streams
    probability = np.asarray(dist.probabilities[EventType.KI], dtype=float)
    twin = case.fresh()[1]
    twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
    twin.validate()
    settles = case.fresh()[0].calculate_event_stats(twin, env).ki_settlement.probability
    assert abs(float(probability.sum()) - settles) <= 1e-8, (
        f"{record['id']}: P(KI before KO) {probability.sum()!r} != the twin's KI settlement {settles!r}")
    own = engine.calculate_event_stats(product, env).ki_settlement.probability
    assert float(probability.sum()) - own > 1e-4, f"{record['id']}: the KI stream is the note's own settlement"
    continuous = bc.ki_continuous or bc.ki_observation_type == ObservationType.CONTINUOUS
    if not continuous:
        # discrete rows one by one against backward numbers (CP4): the first k rows' sum is the KI settlement of the
        # twin cut to its first k KI observations (no knock-out after a knock-in; its knock-in-side edge has no event
        # after its last observation, exactly as the price's row) -- measured <= 2.2e-10 on every discrete record
        cumulative = np.cumsum(probability)
        for k in range(1, times.size + 1):
            cut = _ki_schedule_cut(case.fresh()[1], k, env)
            settles_k = case.fresh()[0].calculate_event_stats(cut, env).ki_settlement.probability
            assert abs(cumulative[k - 1] - settles_k) <= 1e-8, (
                f"{record['id']}: Σ first {k} KI rows {cumulative[k - 1]!r} != the cut twin's KI settlement "
                f"{settles_k!r}")
    if record["engine_kind"] in ("pde_1d", "quad_v2", "quad_v1") and not continuous:
        # and against MC on the same product: a row moved to another date fails. Flat vol only: on this skewed
        # market the LV PDE and LV MC disagree on every outcome (first KO 0.426 vs 0.387, measured, backward columns
        # included -- not the forward pass), so the LV record is pinned by the cut twins above
        mc, se, paths = _mc_knock_in_rows(case.fresh()[1], env)
        floor = np.sqrt(np.clip(probability, 0.0, 1.0) * (1.0 - np.clip(probability, 0.0, 1.0)) / paths)
        excess = np.abs(probability - mc) - 4.0 * np.maximum(se, floor) - _KI_ROW_MC_ALLOWANCE
        assert np.all(excess <= 0.0), f"{record['id']}: KI rows {probability} vs MC {mc} (SE {se})"
    expected = _leg_on_rows(_leg(record), env, times, probability)
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != the leg on the KI rows {expected!r}")


_T13_KI_AXIS = ("pde_snowball__snowball_cki", "pde_snowball__snowball_dki", "pde_snowball__snowball_reverse",
                "pde_phoenix__phoenix_mem_instant", "pde_phoenix__phoenix_nomem_instant",
                "pde_phoenix__phoenix_mem_expiry", "pde_phoenix__phoenix_nomem_expiry",
                "pde_ko_reset__ko_reset_absolute", "pde_ko_reset__ko_reset_absolute_cki",
                "pde_lv_snowball__snowball_cki", "pde_lv_phoenix__phoenix_mem_instant",
                "pde_lv_ko_reset__ko_reset_absolute")

#: T14: the native ADI forward adjoint pass (Heston / SLV; the carried-KI ADI record moved with T6's _KI_AXIS_MOVES)
_T14_KI_AXIS = ("adi_heston_snowball__snowball_cki", "adi_heston_snowball__snowball_dki",
                "adi_heston_phoenix__phoenix_mem_instant", "adi_heston_phoenix__phoenix_nomem_expiry",
                "adi_heston_ko_reset__ko_reset_absolute", "adi_slv_snowball__snowball_cki")

# --- moves (T14b: the knocked-in edge row honours the annualized-KI leg, spec D26) -----------------------------------
#
# Base's knocked-in Dirichlet row on the knock-in edge omitted the annualized knock-in leg's tenor (KO-reset: the
# post-KI tenor, 2y here), and the not-knocked-in surface copies that row, so every price-carrying record of a PDE / ADI
# KO-reset whose paths reach the edge moved. Event-route leg values (probabilities and the legs' own cash) do not.

#: the wide-domain reference: num_std 16 (1D; the ADI default is 8) on 4x the points, the same spacing as the bank's
#: grid, so the edge lies out of the paths' reach
_WIDE_NUM_STD, _WIDE_FACTOR = 16.0, 4
#: on the wide domain the base row and the D26 row agree to this (relative to the record's scale): the edge is out of
#: reach (measured <= 6.1e-8 absolute on a 100 notional, T14b report)
_WIDE_ROW_INDEPENDENCE = 1e-8


class _BaseKnockInRow:
    """Within the block, the PDE solvers write base's knocked-in edge row (the D26 branch off: the knock-in leg reads
    as not annualized to ``SnowballPDESolver._boundary_rows_v1``, and the D27 airbag and D28 partial-floor gates are
    shut; the product and every other leg are untouched)."""

    def __enter__(self):
        from quantark.asset.equity.engine.pde import snowball_pde_solver as sps

        self._module, self._real = sps, sps.leg_annualized
        self._airbag = sps.SnowballPDESolver.__dict__["_airbag_moves_knock_in_edge"]
        self._floor = sps.SnowballPDESolver.__dict__["_partial_floor_unbound_on_knock_in_edge"]
        sps.leg_annualized = lambda product, leg: False if leg == "knock_in" else self._real(product, leg)
        sps.SnowballPDESolver._airbag_moves_knock_in_edge = staticmethod(lambda product, env, s_vec: False)
        sps.SnowballPDESolver._partial_floor_unbound_on_knock_in_edge = staticmethod(lambda product, env, s_vec: False)
        return self

    def __exit__(self, *exc):
        self._module.leg_annualized = self._real
        self._module.SnowballPDESolver._airbag_moves_knock_in_edge = self._airbag
        self._module.SnowballPDESolver._partial_floor_unbound_on_knock_in_edge = self._floor
        return False


class _WideCase:
    """A bank case whose engine prices on the wide domain (everything else is the case's own)."""

    def __init__(self, case):
        self._case = case

    def __getattr__(self, name):
        return getattr(self._case, name)

    def fresh(self):
        from quantark.asset.equity.engine.pde import GridConfig
        from quantark.asset.equity.engine.pde.grid import GridBinder

        engine, product, env = self._case.fresh()
        if hasattr(engine, "n_x"):                                   # Heston / SLV ADI: the S axis binder
            engine.n_x = _WIDE_FACTOR * engine.n_x
            engine._layer_binder_2d = GridBinder("standard", GridConfig(points=engine.n_x, num_std=_WIDE_NUM_STD),
                                                 cache_enabled=engine._is_cache_enabled())
            return engine, product, env
        points = {"fast": 200, "standard": 400, "high": 800}[engine.params.accuracy]
        wide = type(engine)(params=replace(engine.params, grid=GridConfig(num_std=_WIDE_NUM_STD,
                                                                          points=_WIDE_FACTOR * points)))
        return wide, product, env


def _route_on(case, record: dict) -> dict:
    """``record``'s route recomputed on ``case`` (uncached: ``live_outcome`` without its per-case caches)."""
    route = record["route"]
    if route in ("price", "event_npv"):
        return cases.event_route(case)[route]
    if route == "fold_total":
        return cases.fold_total(case, record["leg"])
    if route in ("fold_drilldown_total", "fold_drilldown_npv"):
        drilldown = cases.fold_drilldown(case, list(BANKED["cases"][record["case"]]["drilldown_set"]))
        return drilldown["total" if route == "fold_drilldown_total" else "npv"]
    raise AssertionError(f"{record['id']}: route {route!r} carries no price")


class _ContractTermsKnockInRow:
    """Within the block, the knocked-in lower row is written from the contract terms alone, independently of the
    payoff-read row under test: principal x DF + participation x multiplier x tenor x (S x DF_q - K x DF), the tenor
    the knock-in leg is annualized over (KO-reset: the post-KI tenor). Only for the bank's KO-reset (standard, no
    protection, no airbag, nothing folded)."""

    def __enter__(self):
        from quantark.asset.equity.engine.pde.core.statistics.rows import EdgeRow, RowTerm
        from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

        own = SnowballPDESolver._boundary_rows_v1

        def rows(solver, t_idx, tau, product, env, s_vec):
            lower, upper = own(solver, t_idx, tau, product, env, s_vec)
            assert lower.branch == "v1_lower_none_slope" and solver._fold is None, lower.branch
            assert product.airbag_config.airbag_barrier is None
            df, df_div = solver._get_asymptotic_discount_factors(env, tau)
            principal = product.initial_price * product.contract_multiplier
            k = (product.payoff_config.participation_rate * product.contract_multiplier
                 * product._resolve_post_contract_tenor(env))
            v = principal * df + k * (float(s_vec[0]) * df_div - product.strike * df)
            return EdgeRow(v, lower.branch, (RowTerm("M", v),), lower.copy_from_ki), upper

        self._cls, self._own = SnowballPDESolver, own
        SnowballPDESolver._boundary_rows_v1 = rows
        return self

    def __exit__(self, *exc):
        self._cls._boundary_rows_v1 = self._own
        return False


#: the price under the contract-terms row (``_ContractTermsKnockInRow``) vs live: the two rows differ by round-off
#: only; measured <= 8.9e-16 relative to the record's scale (T14b report)
_ROW_REIMPLEMENTED = 1e-12
#: a non-price record moves by exactly its case's price move: the PDE is linear in its edge rows and folded cash
#: enters base's row and the D26 row alike; measured <= 2.4e-15 relative to the record's scale (T14b report)
_MOVE_IS_THE_PRICE_MOVE = 1e-12


def _annualized_ki_edge_row(live: dict, banked: dict, record: dict) -> None:
    """Cause (k), D26. Pinned without the statistics or the row under test:

    * the product's knock-in leg is annualized and the engine is a PDE 1D / ADI route;
    * the move is the row alone: with base's row written back the record reproduces its banked value (bank
      tolerance), and the move itself exceeds that tolerance;
    * the price record: (i) a reprice whose knocked-in edge row is written from the contract terms
      (``_ContractTermsKnockInRow``) gives the live value; (ii) a wide-domain reprice (``_WideCase``), where base's row
      and the D26 row agree (the edge is out of reach), is nearer the live value than the banked one;
    * any other record (event npv, folded) moves by its case's price move (the PDE is linear in its edge rows)."""
    from quantark.asset.equity.product.option.accrual import leg_annualized

    assert "value" in banked and "value" in live, (live, banked)
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv", "pde_adi", "pde_adi_slv"), record["engine_kind"]
    case = _case(record["case"])
    assert leg_annualized(case.fresh()[1], "knock_in"), f"{record['id']}: the knock-in leg is not annualized"
    scale = float(record["scale"])
    tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * scale)
    move = float(live["value"]) - float(banked["value"])
    with _BaseKnockInRow():
        base_row = _route_on(case, record)
    assert close(float(base_row["value"]), float(banked["value"]), rel_tol=GOLDEN_REL_TOL, abs_tol=tol), (
        f"{record['id']}: base's row gives {base_row['value']!r}, not the banked {banked['value']!r}")
    assert abs(move) > tol, f"{record['id']}: the move is round-off"
    if record["route"] != "price":
        price = RECORDS[f"{record['case']}__-__price"]
        price_move = float(_event(record["case"])["price"]["value"]) - float(price["outcome"]["value"])
        assert abs(move - price_move) <= _MOVE_IS_THE_PRICE_MOVE * scale, (
            f"{record['id']}: moved {move!r}, its case's price moved {price_move!r}")
        return
    with _ContractTermsKnockInRow():
        terms = float(_route_on(case, record)["value"])
    assert abs(float(live["value"]) - terms) <= _ROW_REIMPLEMENTED * scale, (
        f"{record['id']}: live {live['value']!r} != the contract-terms row's price {terms!r}")
    with _BaseKnockInRow():
        wide_base_row = float(_route_on(_WideCase(case), record)["value"])
    wide = float(_route_on(_WideCase(case), record)["value"])
    assert abs(wide - wide_base_row) <= _WIDE_ROW_INDEPENDENCE * scale, (
        f"{record['id']}: the wide reference still reaches the edge ({wide!r} vs {wide_base_row!r})")
    assert abs(float(live["value"]) - wide) < abs(float(banked["value"]) - wide), (
        f"{record['id']}: live {live['value']!r} is not nearer the wide reference {wide!r} than the banked "
        f"{banked['value']!r}")


# --- moves (T15: DCN PDE statistics from the forward adjoint of the price's own march) ---------------------------
#
# At base no DCN PDE engine produced statistics: ``price_with_events`` handed every leg the maturity-only trivial
# distribution (KO-paying legs valued the note as if it ran to maturity; KI / KO / coupon-stream legs raised). Each
# check pins the live value to a number the forward pass never computed: the folded solve banked at base (the fold
# route marches each leg's cash inside the price on the same grid and never read statistics), or the backward march
# of unit claims with the engine's own step and event primitives (``backward_claims_1d`` / ``backward_claims_adi``).


def _dcn_pde_base_was_trivial(banked: dict, record: dict, *, raised: bool) -> None:
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert base["streams"] == ["MATURITY_NO_KO"] and base["probability_totals"] == {"MATURITY_NO_KO": 1.0}, base
    if raised:
        assert banked.get("raises") == "ValidationError", banked
        return
    from quantark.cashleg import EventDistribution

    case = _case(record["case"])
    _, product, env = case.fresh()
    trivial = EventDistribution.trivial(max(float(product.get_maturity(env)), 0.0))
    tol = GOLDEN_REL_TOL * max(abs(float(banked["value"])), 1.0)
    assert close(_leg(record).value(trivial, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: the banked value is not the leg on the trivial distribution")


def _dcn_pde_event_is_the_folded_leg(live: dict, banked: dict, record: dict) -> None:
    """Cause (h). The event route now values the leg on the native statistics; on the same grid the folded solve
    (``fold_drilldown`` per leg, banked at base and unmoved) marches the same cash inside the price, so the two are
    equal (measured <= 4e-10 absolute on scales 2e3..1e6, the ADI's far-edge rows included)."""
    _dcn_pde_base_was_trivial(banked, record, raised="raises" in banked)
    folded = RECORDS[f"{record['case']}__{record['leg']}__fold_drilldown"]["outcome"]
    assert "value" in folded and "value" in live, (folded, live)
    abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
    assert close(float(live["value"]), float(folded["value"]), rel_tol=GOLDEN_REL_TOL, abs_tol=abs_tol), (
        f"{record['id']}: live {live['value']!r} != the folded leg {folded['value']!r} banked at base")
    assert not close(float(live["value"]), float(banked.get("value", np.nan)), rel_tol=GOLDEN_REL_TOL,
                     abs_tol=abs_tol), f"{record['id']}: did not move"


@lru_cache(maxsize=None)
def _dcn_pde_claims(case_id: str) -> dict:
    """The backward unit claims of a DCN PDE case (``pde/dcn_statistics.py``: the engine's own step and events)."""
    from quantark.asset.equity.engine.pde.dcn_statistics import backward_claims_1d, backward_claims_adi

    case = _case(case_id)
    engine, product, env = case.fresh()
    if BANKED["cases"][case_id]["engine_kind"] == "dcn_pde_adi":
        n_obs = len(product.schedule.monthly)
        names = ["ko_total", "maturity", "ki_settlement", "ki_total", "ki_time"] + [f"coupon_{j}" for j in range(n_obs)]
        return backward_claims_adi(engine, product, env, names=names)
    return backward_claims_1d(engine, product, env, **case.kwargs())


def _dcn_pde_knock_in_by_backward_claims(live: dict, banked: dict, record: dict) -> None:
    """Cause (h). The AT_KI leg raised at base (no KI stream); it now values on the per-date knock-in table: its axis
    is the DCN's daily KI dates (exact, on the price's ACT/365F axis), its total and its time-weighted total equal
    the backward knock-in claims (a first knock-in before any knock-out, KO winning a same-date tie) to 1e-10, and the
    leg is ``sign x amount x Σ P DF(t + offset)`` on those rows."""
    from quantark.cashleg import EventType
    from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context

    _dcn_pde_base_was_trivial(banked, record, raised=True)
    assert "requires knock_in" in banked.get("message", ""), banked
    assert "value" in live, live
    dist, env = _live_distribution(record)
    _, product, _ = _case(record["case"]).fresh()
    ctx = build_dcn_grid_context(product, env)
    first = 0 if ctx.valuation_monitored else 1
    times = np.asarray(dist.stream_times[EventType.KI], dtype=float)
    probability = np.asarray(dist.probabilities[EventType.KI], dtype=float)
    np.testing.assert_array_equal(times, ctx.times[first:])
    claims = _dcn_pde_claims(record["case"])
    assert abs(float(np.sum(probability)) - claims["ki_total"]) <= 1e-10
    assert abs(float(np.sum(times * probability)) - claims["ki_time"]) <= 1e-10
    assert close(float(live["value"]), _leg_on_rows(_leg(record), env, times, probability), rel_tol=1e-12,
                 abs_tol=1e-12 * float(record["scale"]))


def _dcn_pde_coupon_stream_by_backward_claims(live: dict, banked: dict, record: dict) -> None:
    """Cause (h). The COUPON-basis leg raised at base (no coupon stream); it now values on the coupon table. The leg
    is recomputed by its own formula from the backward unit claims of each row's fixed coupon and the two survivor
    branches (never the statistics), to round-off -- the never-knocked-in branch carrying the claims' own probability
    residual, as ``from_statistics`` absorbs it (spec D16)."""
    from quantark.cashleg import EventType
    from quantark.cashleg.autocallable_leg import PvFormula

    _dcn_pde_base_was_trivial(banked, record, raised=True)
    assert "coupon stream" in banked.get("message", ""), banked
    assert "value" in live, live
    case = _case(record["case"])
    _, product, env = case.fresh()
    claims = _dcn_pde_claims(record["case"])
    leg = _leg(record)
    assert leg.accrual_basis.value == "coupon" and leg.pv_formula is PvFormula.NORMAL
    n_obs = len(product.schedule.monthly)
    coupon = np.array([claims[f"coupon_{j}"] for j in range(n_obs)])
    contingent = float(np.sum(np.asarray(leg.accrual_factors) * coupon
                              * np.array([env.get_discount_factor(float(t)) for t in leg.settlement_schedule])))
    # D16: the residual 1 - (ΣKO + maturity + KI settlement) rides on MATURITY_NO_KO
    terminal = {EventType.MATURITY_NO_KO: 1.0 - claims["ko_total"] - claims["ki_settlement"],
                EventType.MATURITY_WITH_KI: claims["ki_settlement"]}
    contingent += float(leg.terminal_accrual_factor) * sum(terminal[e] for e in leg.terminal_events) * float(
        env.get_discount_factor(float(leg.terminal_settlement_time)))
    expected = leg.sign() * float(leg.notional) * float(leg.rate) * contingent
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != backward-claim coupon-basis value {expected!r}")


_T15_DCN_CASES = ("pde_dcn__dcn", "pde_dcn__dcn_seller", "pde_lv_dcn__dcn", "pde_dcn__dcn_receivable_alive",
                  "adi_heston_dcn__dcn")
_T15_FOLDED = tuple(
    f"{case}__{leg}__event"
    for case in _T15_DCN_CASES
    for leg in ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity", "fixed_at_ko",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_with_ki", "fixed_at_maturity_any",
                "autocallable_margin_nmp_both", "autocallable_backend_premium_no_ko", "autocallable_rebate_with_ki")
)
_T15_TERMINATED = tuple(
    f"pde_dcn__{case}__{leg}__event"
    for case in ("dcn_ko_unpaid", "dcn_ko_settled")
    for leg in ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_any")
)


_FOLD_LEGS = ("deterministic", "accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
              "accrual_full_schedule", "fixed_at_ko", "fixed_at_maturity_no_ko", "fixed_at_maturity_with_ki",
              "fixed_at_maturity_any")
_FOLD_AUTOCALLABLE = ("autocallable_margin_nmp_both", "autocallable_backend_premium_no_ko",
                      "autocallable_rebate_with_ki")
#: measured: every price-carrying record of the three PDE 1D KO-reset cases; on the ADI (default width 8, the edge
#: barely reached: price moves 6.7e-4) its drilldown total and three autocallable fold totals stay within tolerance
_T14B_EDGE_ROW = (
    *(f"{case}__-__{route}"
      for case in ("pde_ko_reset__ko_reset_absolute", "pde_ko_reset__ko_reset_absolute_cki",
                   "pde_lv_ko_reset__ko_reset_absolute")
      for route in ("price", "event_npv", "fold_drilldown_total", "fold_drilldown_npv")),
    *(f"{case}__{leg}__fold_total"
      for case in ("pde_ko_reset__ko_reset_absolute", "pde_ko_reset__ko_reset_absolute_cki",
                   "pde_lv_ko_reset__ko_reset_absolute")
      for leg in _FOLD_LEGS + _FOLD_AUTOCALLABLE),
    *(f"adi_heston_ko_reset__ko_reset_absolute__-__{route}" for route in ("price", "event_npv", "fold_drilldown_npv")),
    *(f"adi_heston_ko_reset__ko_reset_absolute__{leg}__fold_total" for leg in _FOLD_LEGS),
)

_T12_KO_AT_VALUATION = ("pde_snowball__snowball_cki_ko_at_valuation", "pde_phoenix__phoenix_mem_instant_ko_at_valuation")

# --- moves (T16: QUAD V2 EventStatistics, its forward adjoint pass, terminated lifecycle states) --------------------


def _v2_terminated_state_has_no_event(live: dict, banked: dict, record: dict) -> None:
    """Cause (h), AC17, QUAD V2. Base refused any statistics of a terminated lifecycle state (``NotImplementedError``),
    so its whole event route raised. Now the statistics are the receivables alone (unpaid: the ledger's pending
    flows) or ``None`` (settled), the distribution is the ended note's: ``event_npv`` is ``price()`` (its own banked
    record, bitwise unchanged), a leg that reads events refuses the ended note, a leg that reads none is its value on
    any distribution (the maturity-only trivial one here)."""
    from quantark.cashleg import EventDistribution

    assert banked.get("raises") == "NotImplementedError" and "terminated V2" in banked.get("message", ""), banked
    assert record["engine_kind"] == "quad_v2", record["engine_kind"]
    assert record["features"].get("lifecycle", "").startswith("terminated_"), record["id"]
    case = _case(record["case"])
    engine, product, env = case.fresh()
    state = case.kwargs()["lifecycle_state"]
    dist, _ = _live_distribution(record)
    assert dist.terminated and dist.probabilities == {}
    stats = case.fresh()[0].calculate_event_stats(product, env, lifecycle_state=state)
    pending = state.ledger.pending(state.valuation_point)
    price = engine.price(product, env, lifecycle_state=state)
    if not pending:
        assert stats is None
    else:
        assert stats.terminated and stats.receivables.cash_to_pay.tolist() == [f.amount for f in pending]
        assert stats.receivables.payment_dates == tuple(f.payment_date for f in pending)
        assert stats.pv == price and abs(stats.reconciliation_error) <= 1e-12 * max(1.0, abs(price))
    if record["route"] == "event_npv":
        banked_price = RECORDS[f"{record['case']}__-__price"]["outcome"]["value"]
        assert float(live["value"]) == price, f"{record['id']}: live {live['value']!r} != price() {price!r}"
        assert close(price, float(banked_price), rel_tol=GOLDEN_REL_TOL, abs_tol=GOLDEN_ABS_TOL)
        return
    leg = _leg(record)
    trivial = EventDistribution.trivial(max(float(product.get_maturity(env)), 0.0))
    try:
        leg.value(EventDistribution.ended(), env, cases.UNIT)
    except Exception as refused:    # noqa: BLE001 -- the leg reads events: it must refuse the ended note
        assert "terminated note" in str(refused), refused
        assert live.get("raises") == "ValidationError" and "terminated note" in live.get("message", ""), live
        return
    expected = leg.value(trivial, env, cases.UNIT)
    assert float(live["value"]) == expected, f"{record['id']}: a leg reading no event {live!r} != {expected!r}"


_T16_KI_AXIS = ("quad_v2_snowball__snowball_cki", "quad_v2_snowball__snowball_dki", "quad_v2_snowball__snowball_reverse",
                "quad_v2_phoenix__phoenix_mem_instant", "quad_v2_phoenix__phoenix_nomem_instant",
                "quad_v2_phoenix__phoenix_mem_expiry", "quad_v2_phoenix__phoenix_nomem_expiry",
                "quad_v2_ko_reset__ko_reset_absolute", "quad_v2_ko_reset__ko_reset_absolute_cki",
                "quad_v2_snowball__snowball_cki_receivable_alive")
_T16_CARRIED = ("quad_v2_snowball__snowball_cki_knocked_in", "quad_v2_phoenix__phoenix_mem_instant_knocked_in",
                "quad_v2_snowball__snowball_cki_knocked_in_ko_at_valuation",
                "quad_v2_phoenix__phoenix_mem_instant_knocked_in_ko_at_valuation")
_T16_KO_AT_VALUATION = ("quad_v2_snowball__snowball_cki_ko_at_valuation",
                        "quad_v2_phoenix__phoenix_mem_instant_ko_at_valuation")
_T16_TERMINATED = tuple(
    f"quad_v2_snowball__{case}__{leg}"
    for case in ("snowball_cki_ko_unpaid", "snowball_cki_ko_settled")
    for leg in ("-__event_npv", *(f"{name}__event" for name in (
        "deterministic", "accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
        "accrual_full_schedule", "fixed_at_ko", "fixed_at_ki", "fixed_at_maturity_no_ko", "fixed_at_maturity_with_ki",
        "fixed_at_maturity_any", "autocallable_margin_nmp_both", "autocallable_backend_premium_no_ko",
        "autocallable_rebate_with_ki", "autocallable_backend_interest_coupon")))
)
_T12_KNOCKED_IN_SETTLEMENT = tuple(
    f"{case}__{leg}__event"
    for case in ("pde_snowball__snowball_cki_knocked_in", "pde_snowball__snowball_cki_knocked_in_no_ko_after_ki",
                 "pde_phoenix__phoenix_mem_instant_knocked_in")
    for leg in ("fixed_at_maturity_with_ki", "fixed_at_maturity_any")
)

# --- moves (T17: QUAD V1 EventStatistics, the transpose of its backward step) --------------------------------------


def _v1_decided_coupon_stream(live: dict, banked: dict, record: dict) -> None:
    """Cause (h), QUAD V1 Phoenix knocked out at valuation. Base emitted no COUPON stream for a note knocked out at the
    valuation instant (its legacy decided statistics had no coupon row), so a coupon-basis leg raised. Now the coupon
    decided with the knock-out is its own row (AC11): pinned to the product's own valuation decision (one row at the
    decided record, probability 1 exactly when the decision pays its coupon) and, for the value, to the banked value of
    the same leg on the same contract by the three other engines at base (QUAD V2, PDE 1D, MC: the decided event is
    deterministic, so every engine agrees)."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "coupon stream" in banked.get("message", ""), banked
    assert record["engine_kind"] == "quad_v1" and record["features"].get("ko_at_valuation"), record["id"]
    assert "value" in live, f"{record['id']}: still raises {live}"
    _, product, env = _case(record["case"]).fresh()
    carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    decision = product.decide_observations_at_valuation(env.spot, env, knocked_in=carried)
    assert decision.knocked_out, record["id"]
    dist, _ = _live_distribution(record)
    times = np.asarray(dist.stream_times[EventType.COUPON], dtype=float)
    expected = np.where(np.isclose(times, float(decision.ko_record.observation_time), rtol=0.0, atol=1e-12),
                        1.0 if decision.coupon_index is not None else 0.0, 0.0)
    np.testing.assert_array_equal(np.asarray(dist.probabilities[EventType.COUPON], dtype=float), expected)
    for other in ("quad_v2_phoenix", "pde_phoenix", "mc_phoenix"):
        rid = record["id"].replace("quad_v1_phoenix", other, 1)
        value = float(RECORDS[rid]["outcome"]["value"])
        abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
        assert close(float(live["value"]), value, rel_tol=GOLDEN_REL_TOL, abs_tol=abs_tol), (
            f"{record['id']}: live {live['value']!r} != {other} banked {value!r}")


_T17_KI_AXIS = ("quad_v1_snowball__snowball_cki", "quad_v1_snowball__snowball_dki",
                "quad_v1_phoenix__phoenix_mem_instant", "quad_v1_phoenix__phoenix_nomem_instant",
                "quad_v1_phoenix__phoenix_mem_expiry", "quad_v1_phoenix__phoenix_nomem_expiry",
                "quad_v1_ko_reset__ko_reset_absolute")
_T17_KO_AT_VALUATION = ("quad_v1_snowball__snowball_cki_ko_at_valuation",
                        "quad_v1_phoenix__phoenix_mem_instant_ko_at_valuation")
_T17_DECIDED_COUPON = tuple(
    f"quad_v1_phoenix__{case}__autocallable_backend_interest_coupon__event"
    for case in ("phoenix_mem_instant_ko_at_valuation", "phoenix_mem_instant_knocked_in_ko_at_valuation")
)
#: the maturity row on its own determination (the first schedule's end) and the F5 survival (cause d, as T9 for MC)
_T17_KO_RESET_MATURITY = tuple(
    f"quad_v1_ko_reset__ko_reset_absolute__{leg}__event"
    for leg in ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity",
                "fixed_at_maturity_no_ko", "fixed_at_maturity_any")
)



# --- moves (T23 follow-up 2, spec D29: termination survival) ------------------------------------------------------
#
# A maturity-type termination (the never-knocked-in maturity, the knocked-in settlement) is alive at its own
# determination and gone right after it, past the note's last node too; a knock-out leaves at its own row. Two
# classes move: KO-reset ABSOLUTE, whose first schedule's maturity (1.0) the tables dropped AT its time (F5) -- a
# COMPLETE_PERIOD period ending there now counts the matured notes; and the aged DCN, whose legs run past the note's
# end (0.58y) -- base held the survival flat there and paid ~1.4y of accrual on a matured note.
#
# Each check recomputes the leg from probabilities alone -- the engine's own KO rows and terminal rows (grid
# engines) or the price's own paths (MC: ``ko_reset_paths``, ``dcn_paths``) -- never from a survival column or the
# EventDistribution under test, under both rules: base's (F5 drop at the maturity, survival flat past the last node)
# and D29's. The banked value must be base's leg, the move must be D29 less base, and on MC the path oracle gives
# the live value itself.

#: the ABSOLUTE KO-reset F5 class and the aged DCN, by case: the accrual legs whose survival reads move
_D29_LEGS = {
    **{case: ("accrual_truncate_complete",) for case in (
        "mc_snowball__ko_reset_absolute", "mc_snowball__ko_reset_absolute_cki", "quad_v1_ko_reset__ko_reset_absolute",
        "quad_v2_ko_reset__ko_reset_absolute", "quad_v2_ko_reset__ko_reset_absolute_cki",
        "pde_ko_reset__ko_reset_absolute", "pde_ko_reset__ko_reset_absolute_cki", "pde_lv_ko_reset__ko_reset_absolute",
        "adi_heston_ko_reset__ko_reset_absolute")},
    **{case: ("accrual_truncate_enter", "accrual_truncate_complete", "accrual_truncate_at_maturity")
       for case in ("mc_dcn__dcn_receivable_alive", "pde_dcn__dcn_receivable_alive")},
}


@lru_cache(maxsize=None)
def _d29_sources(case_id: str):
    """(node times, KO mass per node, maturity-type terminations [(time, mass)], maturity payment time, F5) from
    probabilities only: the engine's own statistics rows (grid), the price's own paths (MC)."""
    case = _case(case_id)
    engine, product, env = case.fresh()
    stats = engine.calculate_event_stats(product, env, **case.kwargs())
    obs = stats.observations
    nodes = np.unique(np.asarray(obs.times, dtype=float))
    maturity_pay = float(stats.maturity.payment_time)
    kind = BANKED["cases"][case_id]["engine_kind"]
    if kind == "dcn_mc":
        ref = _same_path().dcn_paths(case.fresh()[0], product, env, **case.kwargs())
        ctx, n = ref["ctx"], ref["n"]
        ko_t = np.asarray(ctx.times[ctx.obs_cols], dtype=float)[ref["ko_row"][ref["ko_row"] >= 0]]
        ko_mass = np.array([np.count_nonzero(ko_t == t) for t in nodes]) / n
        end = float(ctx.times[int(ctx.obs_cols[-1])])
        terminals = [(end, np.count_nonzero(ref["maturity"]) / n), (end, np.count_nonzero(ref["ki_settlement"]) / n)]
        source = "dcn_paths"
    elif kind == "mc":
        ref = _same_path().ko_reset_paths(case.fresh()[0], product, env)
        n, det, event = ref["n"], ref["det"], ref["event"]
        knocked = np.isin(event, ("ko_pre", "ko_post"))
        ko_mass = np.array([np.count_nonzero(knocked & (det == t)) for t in nodes]) / n
        terminals = [(float(t), np.count_nonzero((event == e) & (det == t)) / n)
                     for e in ("maturity", "ki_settlement") for t in np.unique(det[event == e])]
        source = "ko_reset_paths"
    else:
        times = np.asarray(obs.times, dtype=float)
        ko_mass = np.array([float(np.sum(obs.probability[times == t])) for t in nodes])
        terminals = [(float(stats.maturity.time), float(stats.maturity.probability)),
                     (float(stats.ki_settlement.time), float(stats.ki_settlement.probability))]
        source = "statistics rows"
    f5 = (stats.product_family == "ko_reset" and nodes.size > 0
          and float(stats.maturity.time) < float(nodes[-1]) - 1e-10)
    return nodes, ko_mass, tuple(terminals), maturity_pay, float(stats.maturity.time) if f5 else None, source, env


def _survival_by_rule(nodes, ko_mass, terminals, f5_time, t: float, *, d29: bool, entering: bool) -> float:
    """Survival at ``t`` from probabilities alone: the knock-out part linear between nodes (the distribution's
    interpolation), flat past the last node; base drops the F5 maturity mass AT its time and holds the rest past the
    end; D29 drops every maturity-type mass right after its time (``entering``: at it), past the end too."""
    if t <= 0.0:
        return 1.0
    grid = np.r_[0.0, nodes] if nodes.size == 0 or nodes[0] > 1e-10 else nodes
    mass = np.r_[0.0, ko_mass] if grid.size != nodes.size else ko_mass
    ko_part = np.interp(t, grid, 1.0 - np.cumsum(mass))
    if not d29:
        if f5_time is None:
            return float(ko_part)
        f5_mass = sum(m for tt, m in terminals if tt == f5_time)
        node_t = grid[min(np.searchsorted(grid, t - 1e-12), grid.size - 1)]
        dropped = np.where(grid >= f5_time - 1e-12, f5_mass, 0.0)
        return float(ko_part - np.interp(t, grid, dropped)) if node_t != t else float(
            ko_part - (f5_mass if t >= f5_time - 1e-12 else 0.0))
    gone = sum(m for tt, m in terminals if (tt <= t + 1e-10 if entering else tt < t - 1e-10))
    return float(ko_part - gone)


def _leg_by_rule(case_id: str, leg_name: str, *, d29: bool) -> float:
    nodes, ko_mass, terminals, maturity_pay, f5_time, _, env = _d29_sources(case_id)
    leg = next(spec.leg for spec in _case(case_id).legs() if spec.name == leg_name)
    starts, ends = np.asarray(leg.schedule.period_starts), np.asarray(leg.schedule.period_ends)
    from quantark.cashleg import PaymentConvention, SurvivalBasis

    entering = leg.survival_basis is SurvivalBasis.ENTER_PERIOD
    reads = starts if entering else ends
    alive = np.array([_survival_by_rule(nodes, ko_mass, terminals, f5_time, float(t), d29=d29,
                                        entering=entering and d29) for t in reads])
    pays = (np.full(starts.size, maturity_pay) if leg.payment_convention is PaymentConvention.AT_MATURITY
            else np.asarray(leg.schedule.payment_times, dtype=float))
    dfs = np.array([env.get_discount_factor(float(t)) for t in pays])
    return leg.sign() * leg.rate * leg.base.resolve(cases.UNIT) * float(np.sum((ends - starts) * alive * dfs))


def _settled_notes_leave_after_their_settlement(live: dict, banked: dict, record: dict) -> None:
    """Cause (m), D29. From probabilities alone (``_d29_sources``): the banked value is base's rule and the live
    value moved by exactly D29 less base -- a fold record by its leg's move, a drilldown total by the sum of its
    moved legs' moves; an event record (and an MC fold leg) is D29's leg itself."""
    assert "value" in banked and "value" in live, (live, banked)
    case_id, route = record["case"], record["route"]
    legs = (_D29_LEGS[case_id] if route == "fold_drilldown_total"
            else (record["leg"],))
    assert all(leg in _D29_LEGS[case_id] for leg in legs), record["id"]
    delta = _d29_delta(record)
    abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
    move = float(live["value"]) - float(banked["value"])
    assert abs(move - delta) <= _d29_tol(record, delta), (
        f"{record['id']}: moved {move!r}, D29 less base from probabilities {delta!r}")
    if route in ("event", "fold_drilldown") or (route == "fold_total" and "mc" in record["engine_kind"]):
        leg = record["leg"]
        expected = _leg_by_rule(case_id, leg, d29=True)
        value = float(live["value"])
        if route == "fold_total":
            value -= float(_event(case_id)["event_npv"]["value"])
        # MC: the price's own paths give the same numbers as the distribution (round-off); grid: the statistics
        # rows' KO part, within the bank tolerance (the table's survival column is the march's own)
        tol = 1e-12 * float(record["scale"]) if "mc" in record["engine_kind"] else _d29_tol(record, expected)
        assert abs(value - expected) <= tol, f"{record['id']}: {value!r} != D29 leg {expected!r}"
        base = _leg_by_rule(case_id, leg, d29=False)
        if route != "event" or "mc" in record["engine_kind"]:
            assert abs((value - move) - base) <= tol, f"{record['id']}: banked is not base's leg {base!r}"


#: a grid engine's folded solve marches the leg's cash with its own probabilities, not the statistics rows': the
#: two agree to <= 4.2e-8 relative (measured, DCN PDE; KO-reset PDE 4e-9); the bank tolerance is the floor
_D29_FOLD_REL = 1e-6


def _d29_delta(record: dict) -> float:
    """D29 less base, from probabilities alone, for ``record``'s leg (a drilldown total: every moved leg)."""
    legs = _D29_LEGS[record["case"]] if record["route"] == "fold_drilldown_total" else (record["leg"],)
    return sum(_leg_by_rule(record["case"], leg, d29=True) - _leg_by_rule(record["case"], leg, d29=False)
               for leg in legs)


def _d29_tol(record: dict, size: float) -> float:
    abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
    if "mc" in record["engine_kind"] or record["route"] == "event":
        return abs_tol
    return abs_tol + _D29_FOLD_REL * abs(size)


def _edge_row_and_settled_notes(live: dict, banked: dict, record: dict) -> None:
    """Causes (k) and (m) on one folded record (a T14b edge-row record whose accrual leg D29 also moves): with base's
    knocked-in edge row written back the route gives the banked value plus the D29 move alone, and the live record
    moved by its case's price move (k) plus that D29 move (m), the D29 move from probabilities alone."""
    assert "value" in banked and "value" in live, (live, banked)
    case = _case(record["case"])
    delta = _d29_delta(record)
    with _BaseKnockInRow():
        base_row = float(_route_on(case, record)["value"])
    tol = _d29_tol(record, delta)
    assert abs(base_row - delta - float(banked["value"])) <= tol, (
        f"{record['id']}: base's row gives {base_row!r}, banked {banked['value']!r}, D29 move {delta!r}")
    price = RECORDS[f"{record['case']}__-__price"]
    price_move = float(_event(record["case"])["price"]["value"]) - float(price["outcome"]["value"])
    move = float(live["value"]) - float(banked["value"])
    assert abs(move - price_move - delta) <= tol + _MOVE_IS_THE_PRICE_MOVE * float(record["scale"]), (
        f"{record['id']}: moved {move!r} = price move {price_move!r} + D29 {delta!r}?")
    assert abs(delta) > tol, f"{record['id']}: no D29 move"


def _dcn_pde_event_is_the_folded_leg_after_d29(live: dict, banked: dict, record: dict) -> None:
    """Causes (h) and (m): the aged DCN's event-route leg (no native statistics at base, cause h) equals the folded
    leg live -- the folded record itself moved by D29 and is pinned under (m) by probabilities -- and is D29's leg
    from the statistics rows."""
    _dcn_pde_base_was_trivial(banked, record, raised="raises" in banked)
    folded = _drilldown(record["case"])["per_leg"][record["leg"]]
    abs_tol = max(GOLDEN_ABS_TOL, GOLDEN_REL_TOL * float(record["scale"]))
    assert close(float(live["value"]), float(folded["value"]), rel_tol=GOLDEN_REL_TOL,
                 abs_tol=abs_tol + _D29_FOLD_REL * abs(float(folded["value"]))), (
        f"{record['id']}: live {live['value']!r} != the folded leg {folded['value']!r}")
    expected = _leg_by_rule(record["case"], record["leg"], d29=True)
    assert abs(float(live["value"]) - expected) <= abs_tol, f"{record['id']}: != D29 leg {expected!r}"


_D29_MOVES = tuple(
    rid for case, legs in _D29_LEGS.items() for rid in (
        f"{case}__-__fold_drilldown_total",
        *(f"{case}__{leg}__{route}" for leg in legs for route in ("event", "fold_total", "fold_drilldown")))
)

# --- moves (run 9: the knocked-in KO-reset put at the exercise date, cause u) ---------------------------------------
#
# Owner ruling 2026-11-27 (replaces D31): a knocked-in KO-reset not knocked out is a live put to the product's exercise
# date, read there and paid at the product terminal; annualized knocked-in amounts accrue origin -> exercise (O2). The
# banked REBASED note (``cases._ko_reset``: offsets to 1y after the knock-in, maturity 2.0) read an early knock-in's
# put at its own rebased end (knock-in + 1y), on the tenor to it, and settled it there; now at 2.0, on the tenor 2.0.
# Only MC prices REBASED (the grid engines' REBASED cases are banked raises that still raise); every banked ABSOLUTE
# KO-reset has its exercise on schedule 2's end, which run 9 leaves bitwise.
_RUN9_PUT_AT_EXERCISE = ("mc_snowball__ko_reset_rebased__-__price", "mc_snowball__ko_reset_rebased__-__event_npv")


def _rebased_put_readings(case_id: str):
    """(run 9 price, pre-run-9 price) of an MC REBASED KO-reset, both from the price's own paths (a spy on the kernel):
    every path that is not a knocked-in settlement keeps the kernel's cash and settlement (run 9 does not touch it);
    a knocked-in path's V1 is ``get_maturity_payoff_v1`` read off its path

    - run 9: at the exercise node, on the tenor to the exercise date (origin -> exercise, O2), paid at the product
      terminal (``resolve_terminal_timing``);
    - before: at the path's own rebased end (its knock-in record + schedule 2's last offset), on the tenor to that
      end, paid at that end's terminal timing (``resolve_terminal_timing_at``),

    each discounted to its payment and averaged as ``price()`` averages."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing, resolve_terminal_timing_at
    from quantark.util.enum.option_enums import PostKOScheduleMode

    same_path = _same_path()
    case = _case(case_id)
    engine, product, env = case.fresh()
    assert product.post_ko_mode is PostKOScheduleMode.REBASED and not case.kwargs(), case_id
    calls = same_path._spy(engine, "_compute_payoffs_ko_reset")
    try:
        engine.price(product, env)
    finally:
        del engine.__dict__["_compute_payoffs_ko_reset"]
    assert len(calls) == 1, f"{case_id}: a single-batch price makes one kernel call, got {len(calls)}"
    args, kwargs, (payoffs, settle, _stats, _extra) = calls[0]
    call = {**dict(zip(("product", "pricing_env", "paths", "grid"), args)), **kwargs}
    paths, grid = call["paths"], call["grid"]
    ref = same_path.ko_reset_paths(*case.fresh())
    v1 = np.asarray(ref["v1"], dtype=bool)
    assert v1.any(), f"{case_id}: no knocked-in settlement on the price's paths"

    terminal = resolve_terminal_timing(product, env)
    exercise_col = int(grid["exercise_idx"]) + 1
    assert float(grid["all_times"][exercise_col - 1]) == float(terminal.determination_time)
    tenor = float(product._resolve_post_contract_tenor(env))
    new_cash, new_pay = np.array(payoffs, dtype=float), np.array(settle, dtype=float)
    old_cash, old_pay = new_cash.copy(), new_pay.copy()
    last = int(np.argmax(np.asarray(grid["post_times"], dtype=float)))
    for p in np.flatnonzero(v1):
        new_cash[p] = product.get_maturity_payoff_v1(float(paths[p, exercise_col]), env, contract_tenor=tenor)
        new_pay[p] = float(terminal.payment_time)
        k = int(ref["ki_index"][p])
        end = float(grid["post_times_by_ki"][k][last])
        end_col = int(grid["post_indices_by_ki"][k][last]) + 1
        old_cash[p] = product.get_maturity_payoff_v1(float(paths[p, end_col]), env, contract_tenor=end)
        old_pay[p] = float(resolve_terminal_timing_at(product, env, determination_time=end).payment_time)
    assert np.any(old_pay[v1] < new_pay[v1]), f"{case_id}: no knocked-in path ends schedule 2 before the exercise"

    def price(cash, pay):
        return float(np.mean(cash * np.array([env.get_discount_factor(float(t)) for t in pay])))

    return price(new_cash, new_pay), price(old_cash, old_pay)


def _put_at_exercise_on_the_same_paths(live: dict, banked: dict, record: dict) -> None:
    """Cause (u), run 9. The live price is the run-9 reading of the price's own paths (V1 at the exercise date, on the
    tenor to it, paid at the product terminal); the pre-run-9 reading of the same paths (V1 at each path's rebased
    end) gives back the banked price, so the knocked-in readout, its tenor and its settlement are the only source of
    the move. The event_npv record also carries cause (g): base refused REBASED on the event route
    (``NotImplementedError``), and its npv is ``price()`` bitwise."""
    price_record = RECORDS[f"{record['case']}__-__price"]
    if record["route"] == "event_npv":
        assert banked.get("raises") == "NotImplementedError", banked            # cause (g)
        live_price = live_outcome(price_record)
        assert live == {"value": live_price["value"]}, (live, live_price)
    else:
        assert record["route"] == "price" and "value" in banked, (record["id"], banked)
    banked_price = float(price_record["outcome"]["value"])
    run9, before = _rebased_put_readings(record["case"])
    value = float(live["value"])
    assert abs(value - run9) <= 1e-12 * float(record["scale"]), (
        f"{record['id']}: live {value!r} != the put at exercise on the price's paths {run9!r}")
    tol = GOLDEN_REL_TOL * abs(banked_price)
    assert close(before, banked_price, rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: the pre-run-9 reading {before!r} does not give back the banked price {banked_price!r}")
    assert abs(value - banked_price) > tol, f"{record['id']}: the move is round-off"


#: record id -> MovedBy; every other banked record must reproduce.
EXPECTED_MOVES: Dict[str, MovedBy] = {
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _carried_knock_in_is_past) for case in _KI_AXIS_MOVES},
    **{rid: MovedBy("d", _maturity_moves_by_the_residual, within_tolerance=rid not in _RESIDUAL_MOVES_BEYOND_TOLERANCE)
       for rid in _RESIDUAL_MOVES},
    # T9: MC Snowball / KO-reset statistics
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knock_in_on_its_own_axis_before_any_ko) for case in _T9_KI_AXIS},
    **{f"{case}__fixed_at_ki__event": MovedBy("a", _knock_in_after_ko_removed) for case in _T9_KI_AFTER_KO},
    **{rid: MovedBy("d", _ko_reset_maturity_on_its_own_date) for rid in _T9_KO_RESET_MATURITY},
    **{rid: MovedBy("g", _rebased_now_supported) for rid in _T9_REBASED if rid not in _RUN9_PUT_AT_EXERCISE},
    # T10: MC Phoenix statistics
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knock_in_on_its_own_axis_before_any_ko) for case in _T10_KI_AXIS},
    # T11: MC DCN statistics
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knock_in_on_its_own_axis_before_any_ko) for case in _T11_DCN_CASES},
    **{f"{case}__autocallable_backend_interest_coupon__event": MovedBy("h", _dcn_coupon_stream_now_native)
       for case in _T11_DCN_CASES},
    **{rid: MovedBy("h", _dcn_terminated_state_has_no_event) for rid in _T11_TERMINATED},
    # T12: PDE 1D statistics
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knocked_out_at_valuation_has_no_knock_in_row)
       for case in _T12_KO_AT_VALUATION},
    **{rid: MovedBy("d", _knocked_in_settlement_marched_directly) for rid in _T12_KNOCKED_IN_SETTLEMENT},
    # T13: the PDE 1D forward adjoint pass
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _pde_knock_in_before_ko_on_its_own_axis) for case in _T13_KI_AXIS},
    # T14: the native ADI forward adjoint pass
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _pde_knock_in_before_ko_on_its_own_axis) for case in _T14_KI_AXIS},
    # T14b: the knocked-in edge row honours the annualized-KI leg (D26)
    **{rid: MovedBy("k", _annualized_ki_edge_row) for rid in _T14B_EDGE_ROW},
    # T15: DCN PDE statistics (the forward adjoint of the price's own march)
    **{rid: MovedBy("h", _dcn_pde_event_is_the_folded_leg) for rid in _T15_FOLDED},
    **{f"{case}__fixed_at_ki__event": MovedBy("h", _dcn_pde_knock_in_by_backward_claims) for case in _T15_DCN_CASES},
    **{f"{case}__autocallable_backend_interest_coupon__event": MovedBy("h", _dcn_pde_coupon_stream_by_backward_claims)
       for case in _T15_DCN_CASES},
    **{rid: MovedBy("h", _dcn_terminated_state_has_no_event) for rid in _T15_TERMINATED},
    # T16: QUAD V2 statistics (the forward adjoint of its recursion; terminated lifecycle states, AC17)
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _pde_knock_in_before_ko_on_its_own_axis) for case in _T16_KI_AXIS},
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _carried_knock_in_is_past) for case in _T16_CARRIED},
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knocked_out_at_valuation_has_no_knock_in_row)
       for case in _T16_KO_AT_VALUATION},
    **{rid: MovedBy("h", _v2_terminated_state_has_no_event) for rid in _T16_TERMINATED},
    # T17: QUAD V1 statistics (the transpose of its backward step; the direct maturity column)
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _pde_knock_in_before_ko_on_its_own_axis) for case in _T17_KI_AXIS},
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knocked_out_at_valuation_has_no_knock_in_row)
       for case in _T17_KO_AT_VALUATION},
    **{rid: MovedBy("h", _v1_decided_coupon_stream) for rid in _T17_DECIDED_COUPON},
    **{rid: MovedBy("d", _ko_reset_maturity_on_its_own_date) for rid in _T17_KO_RESET_MATURITY},
    # T23 follow-up 2: D29 termination survival (event records already moved by cause d keep cause d)
    **{rid: MovedBy("m", _settled_notes_leave_after_their_settlement) for rid in _D29_MOVES
       if rid not in _T9_KO_RESET_MATURITY and rid not in _T17_KO_RESET_MATURITY and rid not in _T14B_EDGE_ROW
       and rid not in _T15_FOLDED},
    **{rid: MovedBy("k", _edge_row_and_settled_notes) for rid in _D29_MOVES if rid in _T14B_EDGE_ROW},
    **{rid: MovedBy("h", _dcn_pde_event_is_the_folded_leg_after_d29) for rid in _D29_MOVES if rid in _T15_FOLDED},
    # run 9: the knocked-in KO-reset put at the exercise date (the event_npv record's check also pins cause g)
    **{rid: MovedBy("u", _put_at_exercise_on_the_same_paths) for rid in _RUN9_PUT_AT_EXERCISE},
}
#: T23 census: records D29 moves on top of an earlier cause (their checks pin both)
D29_ALSO = frozenset(rid for rid in _D29_MOVES if rid in EXPECTED_MOVES and EXPECTED_MOVES[rid].cause != "m")


# --- live recomputation (cached per worker: one event route / one drilldown per case) -------------

@contextmanager
def _legacy_ko_reset_knock_outs(case_id: str):
    """Cause (t): QUAD V1 records are recomputed with the KO-reset knock-outs on the previous nodal rule
    (``_projected_knock_outs`` False: the code path before 0.16.0 D1, bit for bit; Snowball / Phoenix never read it)."""
    if not str(BANKED["cases"][case_id]["engine"]).startswith("quad_v1"):
        yield
        return
    from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine

    saved = KOResetSnowballQuadEngine.__dict__["_projected_knock_outs"]
    KOResetSnowballQuadEngine._projected_knock_outs = lambda self: False
    try:
        yield
    finally:
        KOResetSnowballQuadEngine._projected_knock_outs = saved


@contextmanager
def _without_exact_first_step(case_id: str):
    """Cause (s): QUAD V1 records are recomputed with the exact first step off (``_FIRST_STEP_S_F = 0``: the lattice
    readout every first step it resolves used before the sub-resolution run, bit for bit), so every other cause's
    check keeps comparing like with like and the first step is the only new source of a QUAD V1 move. Cause (t)'s
    legacy KO-reset knock-outs ride along (``_legacy_ko_reset_knock_outs``): the bank predates both."""
    if not str(BANKED["cases"][case_id]["engine"]).startswith("quad_v1"):
        yield
        return
    from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine

    saved = SnowballQuadEngine._FIRST_STEP_S_F
    SnowballQuadEngine._FIRST_STEP_S_F = 0.0
    try:
        with _legacy_ko_reset_knock_outs(case_id):
            yield
    finally:
        SnowballQuadEngine._FIRST_STEP_S_F = saved


@lru_cache(maxsize=None)
def _case(case_id: str):
    meta = BANKED["cases"][case_id]
    return cases.Case(meta["engine"], meta["product"])


@lru_cache(maxsize=None)
def _event(case_id: str):
    with _without_exact_first_step(case_id):
        return cases.event_route(_case(case_id))


@lru_cache(maxsize=None)
def _drilldown(case_id: str):
    with _without_exact_first_step(case_id):
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
        with _without_exact_first_step(case_id):
            return cases.fold_total(_case(case_id), leg)
    if route == "fold_drilldown_total":
        return _drilldown(case_id)["total"]
    if route == "fold_drilldown_npv":
        return _drilldown(case_id)["npv"]
    if route == "fold_drilldown":
        if leg in BANKED["cases"][case_id]["drilldown_set"]:
            return _drilldown(case_id)["per_leg"][leg]
        with _without_exact_first_step(case_id):
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


def test_no_banked_case_is_in_the_d27_airbag_class():
    """Cause (l), D27, pins no record, because no banked case can reach the airbag edge row: its gate
    (``SnowballPDESolver._airbag_moves_knock_in_edge``) fires only on a partially protected contract with an airbag,
    and no banked product is one. Every banked PDE / ADI price therefore reproduces through T14c (or moves by an
    earlier cause); a case added to the bank in that class would have to be pinned here under (l)."""
    from quantark.util.enum import ProtectionType

    assert BANKED is not None, "baseline.json is not banked"
    assert not any(move.cause == "l" for move in EXPECTED_MOVES.values())
    in_class = []
    for case_id in BANKED["cases"]:
        _, product, _ = _case(case_id).fresh()
        airbag, payoff = getattr(product, "airbag_config", None), getattr(product, "payoff_config", None)
        if airbag is not None and airbag.airbag_barrier is not None and payoff.protection_type == ProtectionType.PARTIAL:
            in_class.append(case_id)
    assert in_class == [], f"in the D27 class, pin under cause (l): {in_class}"


def test_no_banked_case_is_in_the_d28_partial_protection_class():
    """Cause (o), D28, pins no record, because no banked case can reach the partial-protection edge row: its gate
    (``SnowballPDESolver._partial_floor_unbound_on_knock_in_edge``) fires only on a partially protected contract, and
    no banked product is one (the census reads the protection off each case's product, not the gate under test). Every
    banked PDE / ADI price therefore reproduces through T25c (or moves by an earlier cause); a case added to the bank
    with partial protection would have to be pinned here under (o)."""
    from quantark.util.enum import ProtectionType

    assert BANKED is not None, "baseline.json is not banked"
    assert not any(move.cause == "o" for move in EXPECTED_MOVES.values())
    payoff_products, in_class = [], []
    for case_id in BANKED["cases"]:
        _, product, _ = _case(case_id).fresh()
        payoff = getattr(product, "payoff_config", None)
        if payoff is None or not hasattr(payoff, "protection_type"):
            continue
        payoff_products.append(case_id)
        if payoff.protection_type == ProtectionType.PARTIAL:
            in_class.append(case_id)
    assert payoff_products, "the bank holds products with a payoff protection: the census must see them"
    assert in_class == [], f"partially protected, pin under cause (o): {in_class}"


def test_no_banked_case_is_in_the_d30_ko_reset_expiry_class():
    """Cause (n), D30, pins no record, because no banked case is in its class: a KO-reset under EXPIRY coupon pay whose
    first schedule ends before the product maturity (the only contracts whose payment time moves). Every banked
    KO-reset pays INSTANT (``cases._ko_reset``: the default ``AccrualConfig``), so each banked record reproduces
    through T25b (or moves by an earlier cause); a case added to the bank in that class would have to be pinned here
    under (n), its expected value the base value with the schedule-1 knock-out cash re-discounted from the product
    terminal's payment to schedule 1's (independent of the engine under test)."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.util.enum import CouponPayType

    assert BANKED is not None, "baseline.json is not banked"
    assert not any(move.cause == "n" for move in EXPECTED_MOVES.values())
    ko_resets, in_class = [], []
    for case_id in BANKED["cases"]:
        _, product, env = _case(case_id).fresh()
        if not isinstance(product, KnockOutResetSnowballOption):
            continue
        ko_resets.append(case_id)
        if product.accrual_config.coupon_pay_type != CouponPayType.EXPIRY:
            continue
        pre = [r.observation_time for r in product._resolve_ko_schedule(product.barrier_config, env)[0]]
        if pre and max(pre) < float(resolve_terminal_timing(product, env).determination_time):
            in_class.append(case_id)
    assert ko_resets, "the bank holds KO-reset cases: the census must see them"
    assert in_class == [], f"in the D30 class, pin under cause (n): {in_class}"


def test_the_run_nine_exercise_readout_class_is_pinned():
    """Cause (u), run 9 (replaces the D31 census): the class is a KO-reset whose knocked-in note can end schedule 2
    before the exercise date -- ABSOLUTE (a carried knock-in included): schedule 2's last observation before it;
    REBASED: a knock-in whose rebased end (the knock-in record plus schedule 2's last offset) is before it. Only such a
    note's knocked-in readout, tenor and settlement move (a note whose exercise is schedule 2's end reads and pays the
    put there before and after). Read off each case's contract: every banked ABSOLUTE KO-reset is out of the class
    (exercise on schedule 2's end); every banked REBASED KO-reset is in it, and is either a banked raise on the price
    (the grid engines: nothing to move) or the MC case whose price and event_npv are pinned under (u) -- exactly the
    cause (u) records. A case added to the bank in the class would have to be pinned here under (u), its expected value
    the put at exercise recomputed from the price's paths (``_rebased_put_readings``), never the value under test."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.util.enum.option_enums import PostKOScheduleMode
    from quantark.util.numerical import is_close

    assert BANKED is not None, "baseline.json is not banked"
    pinned = {rid for rid, move in EXPECTED_MOVES.items() if move.cause == "u"}
    assert pinned == set(_RUN9_PUT_AT_EXERCISE), pinned
    absolute, in_class, expected = [], [], set()
    for case_id in BANKED["cases"]:
        _, product, env = _case(case_id).fresh()
        if not isinstance(product, KnockOutResetSnowballOption):
            continue
        exercise = float(resolve_terminal_timing(product, env).determination_time)
        post = [float(t) for t in product.get_post_ko_observation_profile(env)["observation_times"]]
        if product.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            absolute.append(case_id)
            ends = [float(product.get_post_maturity_time(env))]
        else:
            ki = [float(t) for t in product.get_ki_observation_profile(env)["observation_times"]]
            ends = [t + max(post) for t in ki]
        if not any(end < exercise and not is_close(end, exercise) for end in ends):
            assert product.post_ko_mode == PostKOScheduleMode.ABSOLUTE, case_id
            continue
        in_class.append(case_id)
        assert product.post_ko_mode == PostKOScheduleMode.REBASED, f"in the run-9 class, pin under cause (u): {case_id}"
        if "raises" not in RECORDS[f"{case_id}__-__price"]["outcome"]:
            expected |= {f"{case_id}__-__price", f"{case_id}__-__event_npv"}
    assert absolute and in_class, "the bank holds ABSOLUTE and REBASED KO-reset cases: the census must see them"
    assert expected == pinned, f"the run-9 class prices {sorted(expected)}, pinned {sorted(pinned)}"


def test_no_banked_case_is_in_the_d33_settlement_precedence_class():
    """Cause (q), D33, pins no record, because no banked case is in its class: a KO-reset with an explicit
    ``settlement_date`` whose first schedule ends before the product maturity (the only contracts whose payment time
    moves; one-maturity notes already paid their maturity on the settlement date). Every banked KO-reset is on the
    numeric clock (``cases._ko_reset``), which cannot carry a settlement date (clock rule), so each banked record
    reproduces through T25b-fu (or moves by an earlier cause); a case added to the bank in that class would have to be
    pinned here under (q), its expected value the base value with the never-knocked-in maturity (and, under EXPIRY, the
    schedule-1 knock-out) cash re-discounted from ``pre + (settlement - T)`` to the settlement date (the ratio of the
    two discount factors, independent of the engine under test)."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    assert BANKED is not None, "baseline.json is not banked"
    assert not any(move.cause == "q" for move in EXPECTED_MOVES.values())
    ko_resets, in_class = [], []
    for case_id in BANKED["cases"]:
        _, product, env = _case(case_id).fresh()
        if not isinstance(product, KnockOutResetSnowballOption):
            continue
        ko_resets.append(case_id)
        if getattr(product, "settlement_date", None) is None:
            continue
        pre = [r.observation_time for r in product._resolve_ko_schedule(product.barrier_config, env)[0]]
        if pre and max(pre) < float(resolve_terminal_timing(product, env).determination_time):
            in_class.append(case_id)
    assert ko_resets, "the bank holds KO-reset cases: the census must see them"
    assert in_class == [], f"in the D33 class, pin under cause (q): {in_class}"


def test_no_banked_case_is_in_the_d34_intraday_twin_class():
    """Cause (r), D34, pins no record, because no banked case is in its class: a numeric KO-reset carrying
    ``first_schedule_settlement_time``, the explicit first-schedule payment the float-time intraday twin of a dated
    D33-class note carries (only such a contract's payment moves; without it every engine reads exactly the request it
    read before). The bank prices no intraday twin and no banked KO-reset carries the time, so each banked record
    reproduces through T25g; a case added to the bank in that class would have to be pinned here under (r), its
    expected value the dated note's on the same engine on the twin's clock (``test_ko_reset_d33_twin``)."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    assert BANKED is not None, "baseline.json is not banked"
    assert not any(move.cause == "r" for move in EXPECTED_MOVES.values())
    ko_resets, in_class = [], []
    for case_id in BANKED["cases"]:
        _, product, _env = _case(case_id).fresh()
        if not isinstance(product, KnockOutResetSnowballOption):
            continue
        ko_resets.append(case_id)
        if getattr(product, "first_schedule_settlement_time", None) is not None:
            in_class.append(case_id)
    assert ko_resets, "the bank holds KO-reset cases: the census must see them"
    assert in_class == [], f"in the D34 class, pin under cause (r): {in_class}"


def test_expected_moves_name_banked_records_and_d9_causes():
    for rid, move in EXPECTED_MOVES.items():
        assert rid in RECORDS, f"EXPECTED_MOVES names {rid!r}, which is not a banked record"
        assert move.cause in D9_CAUSES, f"{rid}: cause {move.cause!r} is not a D9 cause"


# banking order keeps one case's records adjacent, so a worker mostly reuses its cached case solves
@pytest.mark.parametrize("record_id", list(RECORDS) or ["<bank missing>"])
def test_record_reproduces_or_moves_by_its_cause(record_id):
    assert BANKED is not None, f"{BANK} is not banked"
    with _without_exact_first_step(RECORDS[record_id]["case"]):     # cause (s): checks compare like with like
        _record_reproduces_or_moves_by_its_cause(record_id)


def _record_reproduces_or_moves_by_its_cause(record_id):
    record = RECORDS[record_id]
    live = live_outcome(record)
    move = EXPECTED_MOVES.get(record_id)
    if move is not None:
        # AC19 census: a listed record really moved (it no longer reproduces its banked outcome), so the moved set
        # is exactly EXPECTED_MOVES -- a stale entry whose check passed on an unmoved record fails here
        if not move.within_tolerance:
            with pytest.raises(AssertionError):
                assert_reproduced(live, record["outcome"], record)
        move.check(live, record["outcome"], record)
        return
    assert_reproduced(live, record["outcome"], record)


QUAD_V1_CASES = sorted(cid for cid in (BANKED or {}).get("cases", {})
                       if str(BANKED["cases"][cid]["engine"]).startswith("quad_v1"))


@pytest.mark.parametrize("case_id", QUAD_V1_CASES or ["<none>"])
def test_quad_v1_first_step_moves_are_bounded(case_id):
    """Cause (s), measured: with the exact first step on, a QUAD V1 case's price moves by at most 2e-4 relative and
    every event-route leg by at most 5e-3 relative (owner-signed; S3b measured 7.8e-5 / 4.0e-3). Both sides keep cause
    (t)'s legacy KO-reset knock-outs, so this measures the first step alone."""
    assert QUAD_V1_CASES, "no QUAD V1 case is banked"
    off = _event(case_id)
    with _legacy_ko_reset_knock_outs(case_id):
        on = cases.event_route(_case(case_id))
    if "raises" in off["price"]:
        assert "raises" in on["price"], (case_id, on["price"])
        return
    p_off, p_on = float(off["price"]["value"]), float(on["price"]["value"])
    assert abs(p_on - p_off) <= 2e-4 * max(1.0, abs(p_off)), (case_id, p_off, p_on)
    for leg, outcome in off["legs"].items():
        if "raises" in outcome:
            assert "raises" in on["legs"][leg], (case_id, leg)
            continue
        old, new = float(outcome["value"]), float(on["legs"][leg]["value"])
        assert abs(new - old) <= 5e-3 * max(1.0, abs(old)), (case_id, leg, old, new)
    # the fold routes too (final review: S3b's largest move, 4.0e-3, is a KO-reset fold_drilldown leg)
    drill_set = list(BANKED["cases"][case_id]["drilldown_set"])
    if drill_set:
        with _legacy_ko_reset_knock_outs(case_id):
            d_off, d_on = _drilldown(case_id), cases.fold_drilldown(_case(case_id), drill_set)
        for key in ("total", "npv"):
            if "value" in d_off[key]:
                old, new = float(d_off[key]["value"]), float(d_on[key]["value"])
                assert abs(new - old) <= 5e-3 * max(1.0, abs(old)), (case_id, key, old, new)
        for leg, outcome in d_off["per_leg"].items():
            if "value" in outcome:
                old, new = float(outcome["value"]), float(d_on["per_leg"][leg]["value"])
                assert abs(new - old) <= 5e-3 * max(1.0, abs(old)), (case_id, "drilldown", leg, old, new)


def test_the_exact_first_step_moves_some_quad_v1_price():
    """Cause (s) is live: the check above is not comparing the readout with itself."""
    moved = [cid for cid in QUAD_V1_CASES if "value" in _event(cid)["price"]
             and _first_step_on_price(cid) != _event(cid)["price"]["value"]]
    assert moved, "no QUAD V1 price moved with the exact first step on"


def _first_step_on_price(case_id):
    with _legacy_ko_reset_knock_outs(case_id):
        return cases.event_route(_case(case_id))["price"]["value"]


def _route_values(event: dict, drilldown: dict) -> dict:
    out = {("price",): event["price"]}
    out.update({("event", leg): outcome for leg, outcome in event["legs"].items()})
    if drilldown is not None:
        out.update({("drilldown", key): drilldown[key] for key in ("total", "npv")})
        out.update({("drilldown", leg): outcome for leg, outcome in drilldown["per_leg"].items()})
    return out


@pytest.mark.parametrize("case_id", QUAD_V1_CASES or ["<none>"])
def test_ko_reset_v1_projection_moves_are_bounded(case_id):
    """Cause (t), measured (both sides with the exact first step on): a QUAD V1 KO-reset case's price moves by at most
    2e-4 relative and every event-route and fold-drilldown value by at most 5e-3 relative (the cause (s) bounds;
    measured on the banked ko_reset_absolute at 401 nodes: price 94.542085178648 -> 94.52639841532303, 1.66e-4; worst
    leg fixed_at_maturity_no_ko 2.82e-3, event and drilldown); Snowball and Phoenix cases do not move at all."""
    assert QUAD_V1_CASES, "no QUAD V1 case is banked"
    drill_set = list(BANKED["cases"][case_id]["drilldown_set"])
    with _legacy_ko_reset_knock_outs(case_id):
        old = _route_values(cases.event_route(_case(case_id)),
                            cases.fold_drilldown(_case(case_id), drill_set) if drill_set else None)
    new = _route_values(cases.event_route(_case(case_id)),
                        cases.fold_drilldown(_case(case_id), drill_set) if drill_set else None)
    ko_reset = BANKED["cases"][case_id]["engine"] == "quad_v1_ko_reset"
    for key, outcome in old.items():
        if "raises" in outcome:
            assert new[key].get("raises") == outcome["raises"], (case_id, key)
            continue
        before, after = float(outcome["value"]), float(new[key]["value"])
        if not ko_reset:
            assert after == before, (case_id, key, before, after)
            continue
        bound = 2e-4 if key == ("price",) else 5e-3
        assert abs(after - before) <= bound * max(1.0, abs(before)), (case_id, key, before, after)


def test_the_projection_moves_the_ko_reset_quad_v1_price():
    """Cause (t) is live: the check above is not comparing the legacy knock-outs with themselves."""
    case_id = "quad_v1_ko_reset__ko_reset_absolute"
    assert _first_step_on_price(case_id) != cases.event_route(_case(case_id))["price"]["value"]


def test_rebanks_absorbed_no_move_without_a_cause_check():
    """AC19: a re-bank (``generate.py --rebank``) that absorbed a moved record in place would leave the move with no
    cause check (the guard checks only price / event_npv). Every record a re-bank lists as moved must therefore be
    pinned by an ``EXPECTED_MOVES`` check against the old outcome; the only re-bank (T3, cause f) moved none."""
    assert BANKED is not None, "baseline.json is not banked"
    absorbed = [move["id"] for rebank in BANKED["header"].get("rebanks", []) for move in rebank.get("moved", [])]
    assert absorbed == [], f"re-banked moves with no cause check: {absorbed}"
