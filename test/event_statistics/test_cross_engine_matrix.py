"""The full cross-engine matrix of the unified event statistics (spec AC20 full matrix, AC21; DESIGN §5; T24).

Slow: ``@pytest.mark.slow`` AND skipped unless ``QUANTARK_EVENT_STATS_MATRIX=1``. Run it (about 4 minutes on six
workers; see the T24 report for the measured wall, workers and peak RSS)::

    QUANTARK_EVENT_STATS_MATRIX=1 env PYTHONPATH=<wt>:<wt>/test python -m pytest -n 6 \\
        test/event_statistics/test_cross_engine_matrix.py

Per entry of ``matrix_cases.ENTRIES`` (a DESIGN §5 contract in a model group), every engine of the group against
the group's MC reference (16 independent PSEUDO runs, recomputed live), every number of ``matrix_fields_of``:

- grid engine at L0: ``|grid - MC| <= K x SE + allowance + floor`` with the banked measured allowance
  (``matrix_allowances.json``, ``bank_matrix.py``: the envelope ``max(|L0 - L1|, |L0 - L2|)`` of two refinements, the
  three levels banked as the evidence); the live L0 reproduces the bank (``golden_compare``), so the allowance belongs
  to the engine that runs;
- the alternative MC (RQMC with ``ALT_BATCHES`` scrambles, or batched PSEUDO DCN): ``|alt - MC| <= K x
  sqrt(SE_ref^2 + SE_alt^2) + floor`` with ``SE_alt`` the engine's own batch-mean error (spec AC21) where it reports
  one; on a derived column (cumulative KI table, survival) the reference's error scaled to the alternative's path
  count, and on the cumulative KI table at least the cumulative row errors (sd of a sum <= sum of sds);
- SE: the reference's run spread / sqrt(runs), floored on probabilities by the binomial ``sqrt(p (1 - p) / paths)``
  of the larger estimate and on a probability-weighted cash by that floor times the cash per unit probability (a row
  the reference never visits has a spread of 0);
- continuous-KI E[KI time]: the AC13 / D18 convention allowance (grid: one trading day + one MC step; alternative
  MC: one MC step); gated only when the reference has at least two knock-in paths (spec AC21);
- axes (KI table times, REBASED offsets, the termination axis) equal; receivables (deterministic) equal at round-off;
- K as in the smoke (4). Engine discrepancies found by the matrix are listed in ``FINDINGS``: the listed fields must
  fail the gate (strict: a fix makes the test fail until the finding is removed), every other field must pass.

The cheap AC21 checks of the reference estimator (``pool_runs``) and the bank's coverage run in the default suite.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from event_statistics import matrix_cases as M
from golden_compare import close

warnings.filterwarnings("ignore", category=DeprecationWarning)

MATRIX = os.environ.get("QUANTARK_EVENT_STATS_MATRIX") == "1"
BANK_PATH = Path(__file__).resolve().parent / "matrix_allowances.json"
BANK = json.loads(BANK_PATH.read_text())
LOG = os.environ.get("QUANTARK_EVENT_STATS_MATRIX_LOG")
#: the gate's multiple of the standard error (the smoke's K)
K = 4.0
#: round-off floor relative to the cash scale (max(1, |pv|))
FLOOR = 1e-10

#: engine discrepancies the matrix found (T24): (entry, engine) -> (fields expected to fail, finding). Not absorbed
#: into an allowance; price() unchanged. A field listed here that passes fails the test (the finding is resolved).
FINDINGS = {
    ("kr_abs_expiry|bs", engine): (
        ("pv", "total_discounted_cash", "observations.discounted_cash"),
        "KO-reset EXPIRY coupon pay: the MC pays a first-schedule knock-out at the first schedule's maturity (1.0, "
        "the note's end under the KO-reset rule 2), PDE 1D / QUAD V2 / QUAD V1 at the product maturity (2.0); "
        "price() differs by ~2.1 (MC 93.45 +- 0.09 vs 91.34)")
    for engine in ("pde_1d", "quad_v2", "quad_v1")
}

PROBABILITY_LIKE = ("probability", "survival", "ki_cdf", "ki_survival", "probability_no_ko")


def _probability_like(path: str) -> bool:
    return path.split(".")[-1] in PROBABILITY_LIKE


@lru_cache(maxsize=None)
def _reference(key: str) -> M.Reference:
    name, group = key.split("|")
    return M.mc_reference(name, group)


def _cash_scale(reference: M.Reference) -> float:
    return max(1.0, abs(float(reference.mean["pv"][0])))


def _floored_se(path, value, reference, base_se, fields):
    """The batch SE with the binomial floor on probabilities and on probability-weighted cash.

    Cash: a row's cash is an indicator times a bounded amount, so its floor is the indicator's binomial error (the
    rarer side of the reference's and the engine's probability on the row, as on the probability gate) times the
    REFERENCE's cash per unit probability -- on a row the reference visits its own ``|mean| / p``, on a row it never
    visits the largest such unit over the visited rows of the same column. The engine's own cash never sets the
    scale, so a row the reference never visits holds the engine's cash to the probability gate's tolerance times a
    reference cash scale (CP8 minor i: scaling by the engine's value made that gate vacuous)."""
    se = np.asarray(base_se, dtype=float)
    mean = np.asarray(reference.mean[path], dtype=float)
    n = float(reference.paths)
    if _probability_like(path):
        return np.maximum(se, _binomial(value, mean, n))
    head, _, column = path.rpartition(".")
    probability = f"{head}.probability"
    if column in ("discounted_cash", "cash_to_pay") and probability in reference.mean:
        p_ref = np.clip(np.asarray(reference.mean[probability], dtype=float), 0.0, 1.0)
        if p_ref.shape == mean.shape:
            visited = p_ref * n >= 0.5
            unit = np.zeros_like(mean)
            unit[visited] = np.abs(mean[visited]) / p_ref[visited]
            unit[~visited] = float(np.max(unit[visited])) if np.any(visited) else 0.0
            p_engine = np.clip(np.asarray(fields.get(probability, p_ref), dtype=float), 0.0, 1.0)
            if p_engine.shape != p_ref.shape:
                p_engine = p_ref
            q = np.maximum(np.minimum(p_ref, 1.0 - p_ref), np.minimum(p_engine, 1.0 - p_engine))
            q = np.maximum(q, 1.0 / n)
            return np.maximum(se, np.sqrt(q * (1.0 - q) / n) * unit)
    return se


def _binomial(value, mean, n):
    """``sqrt(q (1 - q) / n)`` with ``q`` the rarer side of the larger-distance estimate: a probability near 0 or near 1
    (a survival) that the reference saw on no path has a spread of 0, the grid's small mass is still a rare event."""
    p = np.clip(np.stack([np.abs(np.asarray(value, dtype=float)), np.abs(mean)]), 0.0, 1.0)
    q = np.max(np.minimum(p, 1.0 - p), axis=0)
    return np.sqrt(q * (1.0 - q) / n)


def _continuous(name: str) -> bool:
    barrier = getattr(M.CONTRACTS[name].product(), "barrier_config", None)
    return barrier is not None and bool(barrier.ki_continuous)


def _day(name: str, group: str) -> float:
    return 1.0 / float(M.market(name, group)[1].bus_days_in_year)


def _receivables_offset(fields, reference):
    """An engine fed a lifecycle state the reference does not take (QUAD V2 vs the equity MC): its pending receivables
    sit in pv and the cash totals (DESIGN §4.9) -- compare the remaining life."""
    if fields["receivables.discounted_cash"].size and not reference.mean["receivables.discounted_cash"].size:
        fields = dict(fields)
        fields["pv"] = fields["pv"] - fields["receivables.discounted_cash"].sum()
        fields["total_discounted_cash"] = fields["total_discounted_cash"] - fields["receivables.discounted_cash"].sum()
        fields["total_cash_to_pay"] = fields["total_cash_to_pay"] - fields["receivables.cash_to_pay"].sum()
        fields["receivables.discounted_cash"] = np.zeros(0)
        fields["receivables.cash_to_pay"] = np.zeros(0)
    return fields


def _gate(key, engine, fields, reference, *, se_of, allowance_of, convention):
    """Compare every reference number; ``[(path, row, z, ok, detail)]`` and the max z net of the allowance."""
    scale = _cash_scale(reference)
    rows = []
    for path in M.AXES:
        if path in reference.mean:
            assert path in fields, (engine, path)
            np.testing.assert_allclose(fields[path], reference.mean[path], rtol=0, atol=1e-12, err_msg=f"{engine} {path}")
    for path in M.DETERMINISTIC:
        np.testing.assert_allclose(fields[path], reference.mean[path], rtol=1e-12, atol=FLOOR * scale,
                                   err_msg=f"{engine} {path}")
    missing = set(reference.mean) - set(fields) - {"knock_in.expected_time"}
    assert not missing, (engine, missing)
    for path, mc in reference.mean.items():
        if path in M.AXES or path in M.DETERMINISTIC:
            continue
        if path == "knock_in.expected_time":
            if reference.se.get(path) is None:
                continue                      # spec AC21: undefined under two knock-in paths, not gated
            assert path in fields, (engine, "expected_time missing")
        value = np.asarray(fields[path], dtype=float)
        assert value.shape == mc.shape, (engine, path, value.shape, mc.shape)
        se = se_of(path, value)
        extra = allowance_of(path) + (convention if path == "knock_in.expected_time" else 0.0)
        gap = np.abs(value - mc)
        ok = gap <= K * se + extra + FLOOR * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0.0, np.maximum(gap - extra, 0.0) / se, np.where(gap - extra > FLOOR * scale, np.inf, 0.0))
        for i in range(value.size):
            rows.append((path, i, float(z[i]), bool(ok[i]),
                         f"{engine} {path}[{i}]: value {value[i]!r} mc {mc[i]!r} se {se[i]:.3g} allowance "
                         f"{np.broadcast_to(extra, value.shape)[i]:.3g}"))
    return rows


def _judge(key, engine, rows, log):
    expected, finding = FINDINGS.get((key, engine), ((), None))
    failures = [r for r in rows if not r[3] and r[0] not in expected]
    finding_fields = {r[0] for r in rows if not r[3] and r[0] in expected}
    z_max = max((r[2] for r in rows if r[0] not in expected), default=0.0)
    worst = max((r for r in rows if r[0] not in expected), key=lambda r: r[2], default=None)
    log.append({"entry": key, "engine": engine, "numbers": len(rows), "z_max": z_max,
                "worst": None if worst is None else f"{worst[0]}[{worst[1]}]",
                "failures": [r[4] for r in failures], "finding": finding,
                "finding_fields_failing": sorted(finding_fields)})
    problems = [r[4] for r in failures]
    if finding is not None and finding_fields != set(expected):
        problems.append(f"{engine}: finding fields {sorted(set(expected) - finding_fields)} pass now -- resolved? "
                        f"({finding})")
    return problems


@pytest.mark.parametrize("key", [f"{n}|{g}" for n, g in M.ENTRIES if not M.CONTRACTS[n].terminated])
@pytest.mark.slow
@pytest.mark.skipif(not MATRIX, reason="full matrix: set QUANTARK_EVENT_STATS_MATRIX=1 (slow)")
def test_every_engine_agrees_with_the_mc_reference(key):
    name, group = key.split("|")
    reference = _reference(key)
    assert reference.runs == len(M.SEEDS) and reference.paths == len(M.SEEDS) * M.MC_PATHS
    banked = BANK[key]
    convention_grid = (_day(name, group) * 2.0) if _continuous(name) else 0.0
    convention_mc = _day(name, group) if _continuous(name) else 0.0
    log, problems = [], []
    for engine in M.grid_engines(name, group):
        fields = M.matrix_fields_of(M.engine_statistics(name, group, engine, "L0"))
        bank = banked["engines"][engine]
        assert set(fields) == set(bank["L0"]), (engine, set(fields) ^ set(bank["L0"]))
        for path, value in fields.items():
            assert close(np.asarray(value).tolist(), bank["L0"][path]), (engine, path, value, bank["L0"][path])
        fields = _receivables_offset(fields, reference)
        allowance = bank["allowance"]
        rows = _gate(key, engine, fields, reference,
                     se_of=lambda path, value: _floored_se(path, value, reference, reference.se[path], fields),
                     allowance_of=lambda path: np.asarray(allowance[path], dtype=float),
                     convention=convention_grid)
        problems += _judge(key, engine, rows, log)
    if M.has_alt(name, group):
        stats = M.alt_statistics(name, group)
        fields = M.matrix_fields_of(stats)
        errors = stats.standard_errors
        assert errors is not None, ("the alternative MC must report batch errors (AC21)", dict(stats.unavailable))
        size = stats.ki_table.size
        # a derived column the engine reports no error for: the cumulative KI table is bounded by the cumulative row
        # errors (the standard deviation of a sum is at most the sum of the standard deviations); any other by the
        # reference's error scaled to the alternative's path count
        proxy_scale = math.sqrt(reference.paths / float(M.ALT_BATCHES * M.ALT_PATHS))
        row_errors = errors.get("ki_table.probability")
        cumulative = None if row_errors is None or size == 0 else \
            np.cumsum(np.asarray(row_errors, dtype=float))[M.cdf_rows(size)]

        def se_alt(path, value):
            ref_se = np.asarray(reference.se[path], dtype=float)
            alt = errors.get(path)
            if alt is None:
                alt = ref_se * proxy_scale
                if path in ("ki_cdf", "ki_survival") and cumulative is not None:
                    alt = np.maximum(alt, cumulative)
            return _floored_se(path, value, reference, np.sqrt(ref_se ** 2 + np.asarray(alt, dtype=float) ** 2),
                               fields)

        rows = _gate(key, "alt", fields, reference, se_of=se_alt,
                     allowance_of=lambda path: 0.0, convention=convention_mc)
        problems += _judge(key, "alt", rows, log)
    if LOG:
        with open(LOG, "a") as handle:
            for line in log:
                handle.write(json.dumps(line) + "\n")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("key", [f"{n}|{g}" for n, g in M.ENTRIES if M.CONTRACTS[n].terminated])
@pytest.mark.slow
@pytest.mark.skipif(not MATRIX, reason="full matrix: set QUANTARK_EVENT_STATS_MATRIX=1 (slow)")
def test_a_terminated_unpaid_note_reports_the_same_receivables_on_every_engine(key):
    """A terminated note with unpaid cash (spec AC17): statistics with only the receivables, pv == price(), the same
    receivables on every engine taking the state, and their discounted cash the amount x DF(payment)."""
    name, group = key.split("|")
    contract = M.CONTRACTS[name]
    seen = []
    product, env = M.market(name, group)
    state = contract.lifecycle()
    pending = state.ledger.pending(state.valuation_point)
    engines = list(M.grid_engines(name, group)) + (["alt"] if M.has_alt(name, group) else [])
    family = contract.family
    if "mc" in contract.lifecycle_engines:
        engines.append("mc")
    for engine in engines:
        if engine == "alt":
            stats = M.alt_statistics(name, group)
            make = lambda: M.GROUPS[group].alt[family](product)
        elif engine == "mc":
            make = lambda: M.GROUPS[group].mc[family](M.SEEDS[0])
            stats = make().calculate_event_stats(product, env, lifecycle_state=contract.lifecycle())
        else:
            stats = M.engine_statistics(name, group, engine)
            make = M.GROUPS[group].grid[family][engine]["L0"]
        assert stats is not None and stats.terminated, engine
        assert stats.pv == make().price(product, env, lifecycle_state=contract.lifecycle()), engine
        assert stats.observations.times.size == 0 and stats.maturity.probability == 0.0
        assert stats.receivables.cash_to_pay.tolist() == [f.amount for f in pending], engine
        expected = [f.amount * env.get_discount_factor(
            (f.payment_date - env.valuation_date).days / 365.0) for f in pending]
        np.testing.assert_allclose(stats.receivables.discounted_cash, expected, rtol=1e-12)
        seen.append(stats)
    assert seen, key
    for other in seen[1:]:
        np.testing.assert_array_equal(other.receivables.discounted_cash, seen[0].receivables.discounted_cash)
        assert other.receivables.payment_dates == seen[0].receivables.payment_dates


@pytest.mark.slow
@pytest.mark.skipif(not MATRIX, reason="full matrix: set QUANTARK_EVENT_STATS_MATRIX=1 (slow)")
@pytest.mark.parametrize("key", ["sb_cki|bs", "ph_mem_expiry_dki|bs", "dcn_base|dcn_bs"])
def test_the_alternative_mc_batch_errors_are_calibrated(key):
    """Spec AC21 on the engines: the batch-mean standard errors an MC engine reports (RQMC scrambles, batched PSEUDO)
    match the spread of its estimates over independent re-runs (16 seeds, 100003 apart): the mean reported error over the observed
    spread within [0.6, 1.6] on pv, P(KI) and E[KI time] (delta method), and the ratio's error is None with an
    "undefined:" reason only under two knock-in paths."""
    name, group = key.split("|")
    contract = M.CONTRACTS[name]
    product, env = M.market(name, group)
    family = contract.family
    estimates, reported = {k: [] for k in ("pv", "knock_in.probability", "knock_in.expected_time")}, \
        {k: [] for k in ("pv", "knock_in.probability", "knock_in.expected_time")}
    # seeds 100003 apart: a batched RQMC run scrambles batch b with seed + b and a batched PSEUDO (DCN) run streams
    # batch b from seed + 1000 b, so seeds 1 or 1000 apart share batches and are not independent re-runs (measured on
    # sb_cki: reported / spread 1.65 with consecutive seeds); 100003 collides with neither for 8 batches
    for seed in range(21, 21 + 100003 * 16, 100003):
        stats = M.GROUPS[group].alt[family](product, seed=seed).calculate_event_stats(product, env)
        se = stats.standard_errors
        assert se is not None
        estimates["pv"].append(stats.pv)
        reported["pv"].append(float(se["total_discounted_cash"]))
        estimates["knock_in.probability"].append(stats.knock_in.probability)
        reported["knock_in.probability"].append(float(se["knock_in.probability"]))
        assert "knock_in.expected_time" in se, dict(stats.unavailable)
        estimates["knock_in.expected_time"].append(stats.knock_in.expected_time)
        reported["knock_in.expected_time"].append(float(se["knock_in.expected_time"]))
    for path in estimates:
        spread = float(np.std(estimates[path], ddof=1))
        ratio = float(np.mean(reported[path])) / spread
        assert 0.6 <= ratio <= 1.6, (path, ratio, spread, np.mean(reported[path]))


# --- default suite: the reference estimator (spec AC21) and the bank -------------------------------------------------

def _run(p, t, pv=1.0):
    return {"pv": np.array([pv]), "knock_in.probability": np.array([p]), "knock_in.expected_time": np.array([t]),
            "ki_table.times": np.array([0.25, 0.5])}


def test_reference_errors_come_from_independent_runs_and_the_ratio_by_the_delta_method():
    runs = [_run(0.30, 0.40, 100.0), _run(0.32, 0.42, 101.0), _run(0.28, 0.39, 99.5), _run(0.31, 0.41, 100.4)]
    ref = M.pool_runs(runs, paths_per_run=1000)
    pv = np.array([100.0, 101.0, 99.5, 100.4])
    assert ref.runs == 4 and ref.paths == 4000
    assert math.isclose(float(ref.se["pv"][0]), float(np.std(pv, ddof=1)) / 2.0, rel_tol=1e-12)
    p = np.array([0.30, 0.32, 0.28, 0.31])
    t = np.array([0.40, 0.42, 0.39, 0.41])
    ratio = float(np.sum(p * t) / np.sum(p))
    assert math.isclose(float(ref.mean["knock_in.expected_time"][0]), ratio, rel_tol=1e-14)
    delta = math.sqrt(4 / 3 * float(np.sum((p * t - ratio * p) ** 2))) / float(np.sum(p))
    assert math.isclose(float(ref.se["knock_in.expected_time"][0]), delta, rel_tol=1e-12)
    np.testing.assert_array_equal(ref.mean["ki_table.times"], [0.25, 0.5])
    assert ref.unavailable == {}


def test_reference_expected_time_error_is_undefined_under_two_knock_in_paths():
    """One knock-in path over all runs (1 / 1000 in one run): the ratio is defined, its error is not (AC21)."""
    runs = [_run(0.0, 0.0), _run(0.001, 0.75), _run(0.0, 0.0)]
    runs[0].pop("knock_in.expected_time")
    runs[2].pop("knock_in.expected_time")
    ref = M.pool_runs(runs, paths_per_run=1000)
    assert ref.knock_in_paths == pytest.approx(1.0)
    assert ref.mean["knock_in.expected_time"][0] == pytest.approx(0.75)
    assert ref.se["knock_in.expected_time"] is None
    assert ref.unavailable["knock_in.expected_time"].startswith("undefined:")
    none = M.pool_runs([_run(0.0, 0.0), _run(0.0, 0.0)], paths_per_run=1000)
    assert "knock_in.expected_time" not in none.mean
    assert none.unavailable["knock_in.expected_time"].startswith("undefined:")
    with pytest.raises(ValueError):
        M.pool_runs([_run(0.3, 0.4)], paths_per_run=1000)


def test_an_mc_engine_with_one_knock_in_path_has_no_expected_time_error():
    """The engine side of AC21: a batched run (RQMC, independent scrambles) with exactly one knock-in path before any
    knock-out reports E[KI time] but no standard error for it, with an "undefined:" reason."""
    from quantark.asset.equity.engine.mc import SnowballMCEngine
    from quantark.asset.equity.param import MCParams

    product = M.ONE_KNOCK_IN_PATH["product"]()
    params = MCParams(num_paths=M.ONE_KNOCK_IN_PATH["paths"], seed=M.ONE_KNOCK_IN_PATH["seed"], rqmc_min_batches=4,
                      rqmc_max_batches=4, rqmc_target_std=1e-12, rqmc_paths_mode="per_batch")
    engine = SnowballMCEngine(params=params, method=M.RQMC)
    stats = engine.calculate_event_stats(product, M.routes.flat_env())
    records = engine._last_stats_records
    assert sum(float(r["ki_before_ko_count"]) for r in records) == 1.0
    assert stats.standard_errors is not None and len(records) == 4
    assert stats.knock_in.expected_time is not None and stats.knock_in.expected_time > 0.0
    assert "knock_in.expected_time" not in stats.standard_errors
    assert stats.unavailable["standard_errors.knock_in.expected_time"].startswith("undefined:")


def _banked_reference(key: str) -> M.Reference:
    r = BANK[key]["mc_reference"]
    return M.Reference(mean={k: np.asarray(v, dtype=float) for k, v in r["mean"].items()},
                       se={k: (None if v is None else np.asarray(v, dtype=float)) for k, v in r["se"].items()},
                       runs=r["runs"], paths=r["paths"], knock_in_paths=r["knock_in_paths"],
                       unavailable=dict(r["unavailable"]))


BANKED_GRID = [(key, engine) for key, record in BANK.items() if not key.startswith("_")
               for engine in record.get("engines", {})]


@pytest.mark.parametrize("key, engine", BANKED_GRID)
def test_the_refined_level_agrees_with_the_mc_reference_on_banked_numbers(key, engine):
    """CP8 ruling b: the matrix gates L0 with the envelope allowance; this gates the twice-refined L2 against the same
    banked MC reference with only its own last refinement step ``|L1 - L2|`` as allowance -- so a large L0 allowance
    cannot hide an engine whose refined numbers disagree with the MC. Banked numbers only: no solve, default suite.
    (A refinement step is not an error bound -- non-monotone convergence exists, T24 DCN Heston -- but L2 agreeing
    with an independent MC within K SE plus that step is the evidence the L0 allowance is discretisation.)"""
    name, group = key.split("|")
    reference = _banked_reference(key)
    record = BANK[key]["engines"][engine]
    fields = _receivables_offset({k: np.asarray(v, dtype=float) for k, v in record["L2"].items()}, reference)
    step = {k: np.abs(np.asarray(record["L1"][k], dtype=float) - np.asarray(record["L2"][k], dtype=float))
            for k in record["allowance"]}
    convention = 2.0 * _day(name, group) if _continuous(name) else 0.0
    rows = _gate(key, engine, fields, reference,
                 se_of=lambda path, value: _floored_se(path, value, reference, reference.se[path], fields),
                 allowance_of=lambda path: step[path], convention=convention)
    problems = _judge(key, engine, rows, [])
    assert not problems, "\n".join(problems)


def test_the_matrix_bank_covers_every_entry_and_engine():
    keys = {f"{n}|{g}" for n, g in M.ENTRIES}
    assert {k for k in BANK if not k.startswith("_")} == keys
    for key in keys:
        name, group = key.split("|")
        expected = set() if M.CONTRACTS[name].terminated else set(M.grid_engines(name, group))
        assert set(BANK[key]["engines"]) == expected, key
        for engine, record in BANK[key]["engines"].items():
            assert set(record["allowance"]) == set(record["L0"]) - set(M.AXES), (key, engine)
            for level in M.LEVELS:
                assert set(record[level]) == set(record["L0"]), (key, engine, level)
        if not M.CONTRACTS[name].terminated:
            assert BANK[key]["mc_reference"]["runs"] == len(M.SEEDS)
    assert BANK["_provenance"]["script"] == "test/event_statistics/bank_matrix.py"


def test_the_matrix_is_slow_and_opt_in():
    """AC20: the full matrix is ``slow`` AND needs QUANTARK_EVENT_STATS_MATRIX=1 (the default suite skips it)."""
    for test in (test_every_engine_agrees_with_the_mc_reference,
                 test_a_terminated_unpaid_note_reports_the_same_receivables_on_every_engine,
                 test_the_alternative_mc_batch_errors_are_calibrated):
        marks = {m.name for m in getattr(test, "pytestmark", [])}
        assert {"slow", "skipif"} <= marks, (test.__name__, marks)
