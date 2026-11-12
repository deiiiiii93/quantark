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


def _carried_knock_in_on_its_own_axis(live: dict, banked: dict, record: dict) -> None:
    """Cause (e), pinned to the definition. At base the distribution dropped the KI stream of a note already
    knocked in (one row against the KO grid), so the AT_KI leg raised. The carried knock-in is the KI table's
    single row at t = 0 with probability 1 (DESIGN §4.5), on its own axis, paid at the KI time (D8): the stream
    must be exactly [(0, 1)], unflagged, and the leg exactly ``amount · DF(offset)``. Any other probabilities
    fail (a KI-ever table is never carried; those records reproduce their banked raise)."""
    from quantark.cashleg import EventType

    assert banked.get("raises") == "ValidationError" and "requires knock_in" in banked.get("message", ""), banked
    assert record["features"].get("knocked_in_at_valuation") is True, f"{record['id']}: not a carried knock-in"
    base = BANKED["cases"][record["case"]]["event_distribution"]
    assert "KI" not in base["streams"], f"{record['id']}: base had a KI stream; the move is not cause (e)"
    assert "value" in live, f"{record['id']}: still raises {live}"
    dist, env = _live_distribution(record)
    np.testing.assert_array_equal(np.asarray(dist.stream_times[EventType.KI], dtype=float), [0.0])
    np.testing.assert_array_equal(np.asarray(dist.probabilities[EventType.KI], dtype=float), [1.0])
    np.testing.assert_array_equal(np.asarray(dist.payment_times_for(EventType.KI), dtype=float), [0.0])   # D8
    assert EventType.KI not in dist.flagged_streams, dist.flagged_streams
    leg = _leg(record)
    expected = float(leg.amount) * env.get_discount_factor(float(leg.payment_offset_days) / 365.0)
    assert close(float(live["value"]), expected, rel_tol=1e-12, abs_tol=1e-12 * float(record["scale"])), (
        f"{record['id']}: live {live['value']!r} != amount · DF(offset) {expected!r}")


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
    """Cause (d), MC KO-reset ABSOLUTE. Base read the legacy MC statistics through the one-maturity layout:
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
    )
    leg = _leg(record)
    assert leg.value(dist, env, cases.UNIT) == float(live["value"])
    tol = GOLDEN_REL_TOL * abs(float(banked["value"]))
    assert close(leg.value(undone, env, cases.UNIT), float(banked["value"]), rel_tol=0.0, abs_tol=tol), (
        f"{record['id']}: undoing the one-maturity layout does not give back the banked value")
    assert abs(float(live["value"]) - float(banked["value"])) > tol, f"{record['id']}: the move is round-off"


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
    for t, survival in zip(dist.survival_times, dist.survival_probability):
        assert abs(survival - (1.0 - np.count_nonzero(ref["det"] <= t) / n)) <= 1e-14
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
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv"), record["engine_kind"]
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
    """Cause (e), PDE 1D (T13) and native Heston / SLV ADI (T14). Base carried no KI stream (the legacy PDE / ADI table
    was empty and the converter dropped it), so the AT_KI leg raised. Now the forward adjoint pass fills the KI table
    (the PDE 1D pass; the ADI pass, ``pde/vol_model_adjoint.py``). Pinned without the statistics object it checks:

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
    assert record["engine_kind"] in ("pde_1d", "pde_1d_lv", "pde_adi", "pde_adi_slv"), record["engine_kind"]
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
    if record["engine_kind"] == "pde_1d" and not continuous:
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

_T12_KO_AT_VALUATION = ("pde_snowball__snowball_cki_ko_at_valuation", "pde_phoenix__phoenix_mem_instant_ko_at_valuation")
_T12_KNOCKED_IN_SETTLEMENT = tuple(
    f"{case}__{leg}__event"
    for case in ("pde_snowball__snowball_cki_knocked_in", "pde_snowball__snowball_cki_knocked_in_no_ko_after_ki",
                 "pde_phoenix__phoenix_mem_instant_knocked_in")
    for leg in ("fixed_at_maturity_with_ki", "fixed_at_maturity_any")
)

#: record id -> MovedBy; every other banked record must reproduce.
EXPECTED_MOVES: Dict[str, MovedBy] = {
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _carried_knock_in_on_its_own_axis) for case in _KI_AXIS_MOVES},
    **{rid: MovedBy("d", _maturity_moves_by_the_residual) for rid in _RESIDUAL_MOVES},
    # T9: MC Snowball / KO-reset statistics
    **{f"{case}__fixed_at_ki__event": MovedBy("e", _knock_in_on_its_own_axis_before_any_ko) for case in _T9_KI_AXIS},
    **{f"{case}__fixed_at_ki__event": MovedBy("a", _knock_in_after_ko_removed) for case in _T9_KI_AFTER_KO},
    **{rid: MovedBy("d", _ko_reset_maturity_on_its_own_date) for rid in _T9_KO_RESET_MATURITY},
    **{rid: MovedBy("g", _rebased_now_supported) for rid in _T9_REBASED},
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
