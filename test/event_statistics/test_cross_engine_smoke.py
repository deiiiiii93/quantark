"""Fast cross-engine smoke of the unified event statistics (spec AC20 fast, R3: one contract, one meaning per field,
across engines; MC as the reference, DESIGN §5; T22).

Per case of ``cases.CASES`` (flat Snowball continuous KI, flat memory-EXPIRY Phoenix, flat KO-reset ABSOLUTE,
local-vol Snowball, Heston Snowball, DCN flat / local vol / Heston), every grid engine's number against the MC
reference, field by field (pv, the KO table, the coupon table, maturity, knocked-in settlement, P(KI before KO),
expected loss, expected lifetime, the cumulative knock-in table, E[KI time]):

    |grid - MC| <= K x SE + allowance + ABS_FLOOR

- SE from the reference's independent runs; where the reference saw no event in a row (a batch spread of 0 on a
  1e-6 probability) the binomial floor sqrt(p (1 - p) / paths) of the larger of the two estimates;
- allowance: the grid engine's banked discretisation allowance (``allowances.json``; the live L0 number must
  reproduce the banked one within ``golden_compare`` tolerances, so the allowance belongs to the engine that runs);
- the continuous-KI expected time adds the convention allowance of spec AC13 / D18: the MC books a crossing at its
  step's end, the grid engines at the crossing interval's (day's) midpoint -- one trading day plus one MC step;
- E[KI time] is gated only above two knock-in paths (spec AC21).

The engines' AC8 bounds, AC9 and AC10: ``test_engine_invariants.py``; the full DESIGN §5 matrix: T24.
"""
from __future__ import annotations

import json
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from event_statistics.cases import CASES, LEVELS, MC_PATHS, SEEDS, fields_of, grid_statistics, mc_reference
from golden_compare import GOLDEN_REL_TOL, close

warnings.filterwarnings("ignore", category=DeprecationWarning)

BANK = json.loads((Path(__file__).resolve().parent / "allowances.json").read_text())
#: the gate's multiple of the reference's standard error (DESIGN §5: k x SE); measured max |z| net of the allowance
#: over every field of every case: see the T22 report (<= 3.0)
K = 4.0
#: round-off floor of the gate (a probability or cash identically zero on both sides)
ABS_FLOOR = 1e-10


@lru_cache(maxsize=None)
def _reference(name):
    return mc_reference(CASES[name])


def _se(name, path, grid_value, reference):
    """The reference's batch SE with the binomial floor on probabilities (a row with no MC event has a batch spread of
    0): the floor uses the larger of the grid and the MC estimate, on the reference's total path count."""
    se = np.asarray(reference.se[path], dtype=float)
    if path.endswith("probability") or path == "ki_cdf":
        p = np.clip(np.maximum(np.abs(grid_value), np.abs(reference.mean[path])), 0.0, 1.0)
        se = np.maximum(se, np.sqrt(p * (1.0 - p) / (MC_PATHS * len(SEEDS))))
    return se


def _continuous(case) -> bool:
    barrier = getattr(case.product(), "barrier_config", None)
    return barrier is not None and bool(barrier.ki_continuous)


def _mc_step(case) -> float:
    """The MC reference's step on a continuous knock-in: one trading day (``continuous_ki_step_count``)."""
    return 1.0 / float(case.env().bus_days_in_year)


ENGINE_CASES = [(name, engine) for name, case in CASES.items() for engine in case.grid]


@pytest.mark.parametrize("name, engine", ENGINE_CASES)
def test_the_live_engine_is_the_banked_one(name, engine):
    """The smoke's L0 statistics reproduce the bank (golden_compare tolerances: ARM64 bank, x86_64 CI)."""
    live = fields_of(grid_statistics(CASES[name], engine, "L0"))
    banked = BANK[name][engine]["L0"]
    assert set(live) == set(banked), (set(live) ^ set(banked))
    for path, value in live.items():
        assert close(value.tolist(), banked[path]), (path, value, banked[path])


@pytest.mark.parametrize("name", list(CASES))
def test_every_engine_agrees_with_the_mc_reference(name):
    case = CASES[name]
    reference = _reference(name)
    failures, z_max = [], 0.0
    for engine in case.grid:
        stats = grid_statistics(case, engine, "L0")
        grid = fields_of(stats)
        allowance = BANK[name][engine]["allowance"]
        np.testing.assert_allclose(grid["ki_table.times"], reference.mean["ki_table.times"], rtol=0, atol=1e-12)
        assert set(reference.mean) - {"knock_in.expected_time"} <= set(grid), set(reference.mean) - set(grid)
        for path, mc in reference.mean.items():
            if path == "ki_table.times":
                continue
            if path == "knock_in.expected_time" and reference.knock_in_paths < 2.0:
                continue
            value = grid[path]
            se = _se(name, path, value, reference)
            extra = np.asarray(allowance[path], dtype=float)
            if path == "knock_in.expected_time" and _continuous(case):
                extra = extra + 1.0 / float(case.env().bus_days_in_year) + _mc_step(case)
            gap = np.abs(value - mc)
            ok = gap <= K * se + extra + ABS_FLOOR
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(se > 0.0, np.maximum(gap - extra, 0.0) / se, 0.0)
            z_max = max(z_max, float(np.max(z)) if z.size else 0.0)
            if not np.all(ok):
                k = int(np.argmax(gap - (K * se + extra)))
                failures.append(f"{engine} {path}[{k}]: grid {value[k]!r} mc {mc[k]!r} se {se[k]:.3g} "
                                f"allowance {extra[k]:.3g}")
    assert not failures, "\n".join(failures)
    assert z_max < K


@pytest.mark.parametrize("name", list(CASES))
def test_the_reference_has_independent_batches_and_resolves_every_field(name):
    """The reference is what DESIGN §5 asks: independent runs (distinct seeds; a positive spread on every outcome
    probability that is neither 0 nor 1), and its standard errors are small enough to gate (each outcome probability's
    SE below 2e-3)."""
    reference = _reference(name)
    assert reference.runs == len(SEEDS) >= 8
    for path in ("observations.probability", "maturity.probability", "ki_settlement.probability",
                 "knock_in.probability"):
        mean, se = reference.mean[path], reference.se[path]
        inside = (mean > 1e-3) & (mean < 1.0 - 1e-3)
        assert np.all(se[inside] > 0.0) and np.all(se < 2e-3), (path, se)


@pytest.mark.parametrize("name, engine", ENGINE_CASES)
def test_banked_residuals_are_bounded_and_shrink_under_refinement(name, engine):
    """D16 on the smoke's engines: the banked residuals at L0 / L1 / L2. A residual above round-off (PDE 1D and DCN PDE
    probability residual, the V1 EXPIRY reconciliation) does not grow with refinement and is smaller at L2 than at
    L0; a round-off one stays round-off at every level."""
    residuals = BANK[name][engine]["residuals"]
    for column in (0, 1):
        values = [abs(residuals[level][column]) for level in LEVELS]
        if values[0] <= 1e-12:
            assert max(values) <= 1e-12, values
            continue
        assert values[1] <= values[0] * (1.0 + 1e-6) and values[2] < values[0], (column, values)


def test_the_bank_covers_every_case_and_engine():
    assert {k for k in BANK if not k.startswith("_")} == set(CASES)
    for name, case in CASES.items():
        assert set(BANK[name]) == set(case.grid)
        for engine in case.grid:
            assert set(BANK[name][engine]["allowance"]) == set(BANK[name][engine]["L0"]) - {"ki_table.times"}
    assert BANK["_provenance"]["script"] == "test/event_statistics/bank_allowances.py"
    assert math.isclose(GOLDEN_REL_TOL, 1e-9)
