# Per-date PDE Solve (Design B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price all six cells of the simulated-path snowball study with one exact per-date PDE solve per state, then run the study on the banked 40 paths (stage 1) and on 2,000 paths (stage 2).

**Architecture:** Design B is the existing exact `RepricingPricer` driven by the study's PDE engine config: each state is one solve from maturity back to its date with that date's real dividend object. The only library change removes the second `_solve` that `price()` + `calculate_greeks()` cost today; which of two mechanisms does it is decided by a standalone runtime-patched demo (Task 1) and the user (Task 2). The study gains a `per_date` provider, `--history-end`, `--paths-dir`, `--spot-range`, an exact-QUAD check on 40 paths, and day-0 book marks.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest (+xdist), quantark PDE engine (`SnowballPDESolver`), `quantark.backtest.simulation`.

**Spec:** `docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md` (commit `f283d459`). Read it before starting; this plan argues from it.

## Global Constraints

- Branch `feat/simulated-path-backtest`, main checkout `/Users/fuxinyao/quant-ark` (the banked outputs and market caches live here). Other sessions share this checkout and its git index: **commit by pathspec only** (`git commit -F msg -- <paths>`), never `git add -A`, never `git add example/` wholesale.
- Never stage: anything under `output/`; `example/standalone_rho/` (real client bookings); `docs/superpowers/plans/2026-09-15-intraday-pricing-design.md` (a peer's WIP). Files under `docs/` need `git add -f`.
- After any full-suite run: `git checkout -- example/mo_volmodels/data/` before staging (test runs churn two tracked sample files).
- Canonical `quantark.*` imports only. Library comparisons use `quantark/util/numerical/` helpers; tests asserting bitwise identity use `==` / `np.array_equal` deliberately.
- Exact semantics by default; approximations are opt-in only. No fallbacks: when a required value is missing, raise `ValidationError` naming it.
- Never import or instantiate a `*MCEngine` from PDE engine code.
- Commit message trailer: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Tests: `.venv/bin/python -m pytest <path>` (xdist on by default via `pytest.ini`); `-n0` for serial. A nonexistent test path voids the whole invocation ("no tests ran") — that is a failed run, not a pass.
- Shared machine (14 cores; a peer fleet may hold 6): timings are compared only back-to-back in one window. Long runs: `nohup caffeinate -i -m -s`. Kill spawn pools by process group, never `pkill -f`; verify by CPU against wall time.
- Tables shipped under `example/snowball_simulated_paths/data/` carry no absolute path of this machine.
- Study constants that must not change: mesh `SURFACE_POINTS = 1601`, `SURFACE_STEPS_PER_DAY = 16.0`, `SURFACE_MAX_STEPS = 8000`, `SURFACE_MAX_POINTS = 2000`, `SURFACE_SPOT_RANGE = (0.40, 1.60)`; fair coupon under `term_flat_q` with QUAD on a 401-point grid; banked coupon 37.8254%.

## File Map

| File | Responsibility | Tasks |
|---|---|---|
| `example/snowball_simulated_paths/phase0_single_solve.py` (new) | Standalone runtime-patched demo and decision matrix | 1 |
| `docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md` | Records the Phase 0 decision | 2 |
| `quantark/util/fingerprint.py` (new, 3A only) | Structural object rendering for keys | 3A |
| `quantark/backtest/simulation/pricing/repricing.py` | Uses the util (3A) or one-call PDE path (3B) | 3A / 3B |
| `quantark/asset/equity/engine/pde/snowball_pde_solver.py`, `phoenix_pde_solver.py`, `ko_reset_snowball_pde_solver.py` (4A only) | One-entry solve memo | 4A |
| `test/simulation/test_util_fingerprint.py` (new, 3A), `test/simulation/test_solve_memo.py` (new, 4A), `test/simulation/test_repricing.py` (3B) | Library tests | 3A / 4A / 3B |
| `example/snowball_simulated_paths/01_build_paths.py` | `--history-end`, `load_paths(paths_dir=)` | 5, 7 |
| `example/snowball_simulated_paths/02_ensemble_fleet.py`, `_sim_common.py` | `per_date`, defaults, check oracle, `--spot-range`, per-run batching, day-0 mark, `--paths-dir` | 6, 7 |
| `example/snowball_simulated_paths/03_report.py` | Provider labels, day-0 table, caveat | 8 |
| `example/snowball_simulated_paths/README.md` | Design B text, commands, results | 9, 13 |
| `test/test_snowball_simulated_paths_study.py` | Study tests | 5–8 |

Task order: 1 → 2 → (3A → 4A) or (3B) → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13.

---

### Task 1: Phase 0 standalone single-solve demo

**Files:**
- Create: `example/snowball_simulated_paths/phase0_single_solve.py`

**Interfaces:**
- Consumes: `S02.cell_config`, `S02.study_terms`, `S01.load_paths` (existing); `C.engine_config`, `C.Q.build_product`; `run_ensemble`; `RepricingPricer`; `SnowballPDESolver`.
- Produces: `output/snowball_simulated_paths/phase0/decision_matrix.md` and `decision_matrix.json`. No library code.

- [ ] **Step 1: Write the demo script**

```python
"""Phase 0: one PDE solve per state instead of two -- a standalone, runtime-patched demo.

    .venv/bin/python example/snowball_simulated_paths/phase0_single_solve.py
    .venv/bin/python example/snowball_simulated_paths/phase0_single_solve.py --skip-suites

Exact repricing on the PDE engine calls ``engine.price`` and then
``engine.calculate_greeks`` per state, and each runs ``_solve``.  Two
candidates remove the second solve.  Both are monkeypatches applied inside
one process; no library file changes.

  (a) memo    a one-entry solve memo on ``SnowballPDESolver``: inside
              ``price``/``calculate_greeks``, an identical (solver type,
              product, environment, params, cache strategy, extra nodes) and
              the SAME injected session objects reuse the last result; any
              other ``_solve`` clears it.  The key INCLUDES lifecycle
              attributes: the solver reads ``_otc_lifecycle_knocked_in``,
              which the state cache's ``_canonical`` deliberately skips.
  (b) greeks  ``RepricingPricer._price_env`` takes the mark from
              ``calculate_greeks()["price"]`` and never calls ``price``.

Evidence (spec 3.2): one study path bitwise against the two-call baseline;
the conformance suite (replay-oracle parity) under each arm; the replay
goldens under (a); memo miss probes; seconds per state back-to-back.
Writes output/snowball_simulated_paths/phase0/decision_matrix.{md,json}.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sim_common as C  # noqa: E402


def _stage(name: str):
    spec = importlib.util.spec_from_file_location(
        f"snowball_simulated_paths_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S02 = _stage("02_ensemble_fleet")

import quantark.backtest.simulation.pricing.repricing as repricing  # noqa: E402
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver  # noqa: E402
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.backtest.replay.engine_factory import create_pricing_engine  # noqa: E402
from quantark.backtest.replay.market import SignedDividendYield  # noqa: E402
from quantark.backtest.simulation import CacheConfig, day_chain, dividend_yield_for_day, run_ensemble  # noqa: E402
from quantark.backtest.simulation.hedge import day_active_contract  # noqa: E402
from quantark.backtest.simulation.pricing.cache import StateCache  # noqa: E402
from quantark.backtest.simulation.pricing.repricing import RepricingPricer  # noqa: E402
from quantark.backtest.simulation.results import BOOL_COLUMNS, FLOAT_COLUMNS  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

MODEL = "term_flat_q"
HEDGE = "front"
OUT = C.DEFAULT_OUT_DIR / "phase0"
SUITES = {
    "conformance": ["test/simulation/test_conformance.py"],
    "replay_goldens": ["test/test_replay_goldens.py"],
}
SESSION_ATTRS = ("_session_grids", "_session_step_coefficients", "_session_matrix_pack",
                 "_session_banded_pack", "_frozen_base_layout")
COUNTS = {"solves": 0, "memo_hits": 0}
_ORIGINAL_SOLVE = SnowballPDESolver.__dict__["_solve"]
_ORIGINAL_PRICE = SnowballPDESolver.__dict__["price"]
_ORIGINAL_GREEKS = SnowballPDESolver.__dict__["calculate_greeks"]


# -- the arms -----------------------------------------------------------------

def counting_solve(self, product, pricing_env):
    COUNTS["solves"] += 1
    return _ORIGINAL_SOLVE(self, product, pricing_env)


def structural_key(obj: Any) -> str:
    """``repricing._canonical`` with lifecycle attributes KEPT (the solver reads them)."""
    saved = repricing._STATE_PREFIXES
    repricing._STATE_PREFIXES = ()
    try:
        return repricing._canonical(obj, set())
    finally:
        repricing._STATE_PREFIXES = saved


def memo_key(solver, product, pricing_env) -> str:
    return repr((type(solver).__qualname__, structural_key(product), structural_key(pricing_env),
                 structural_key(solver._params_cache_key()), solver._resolve_cache_strategy(),
                 tuple(getattr(solver, "_extra_time_nodes", ()))))


def memo_solve(self, product, pricing_env):
    if not getattr(self, "_demo_memo_on", False):
        self._demo_memo = None                      # any other solve invalidates the memo
        return counting_solve(self, product, pricing_env)
    key = memo_key(self, product, pricing_env)
    refs = tuple(getattr(self, name, None) for name in SESSION_ATTRS)
    memo = getattr(self, "_demo_memo", None)
    if memo is not None and memo[0] == key and all(x is y for x, y in zip(memo[1], refs)):
        COUNTS["memo_hits"] += 1
        return memo[2]
    self._demo_memo = None
    result = counting_solve(self, product, pricing_env)
    self._demo_memo = (key, refs, result)
    return result


def _flagged(method):
    def wrapper(self, product, pricing_env):
        previous = getattr(self, "_demo_memo_on", False)
        self._demo_memo_on = True
        try:
            return method(self, product, pricing_env)
        finally:
            self._demo_memo_on = previous
    return wrapper


def greeks_only_price_env(self, date, *, knocked_in, key, spot, vol, rate, div_yield, basis_yield, label):
    product = self.aged_product(date, knocked_in=knocked_in)
    self._seed_engine(self._engine, key)
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name=self.underlying),
        vol_surface=FlatVolSurface(volatility=vol), rate_curve=FlatRateCurve(rate=rate),
        div_yield=div_yield, basis_yield=basis_yield, valuation_date=pd.Timestamp(date).to_pydatetime(),
    )
    try:
        greeks = self._engine.calculate_greeks(product, env)
    except Exception as exc:
        raise ValidationError(f"pricing failed at {label}: spot={spot!r}, vol={vol!r}, knocked_in={knocked_in}: {exc}") from exc
    if "price" not in greeks:
        raise ValidationError(f"{type(self._engine).__name__}.calculate_greeks returned no price")
    self._engine_calls += 1
    return float(greeks["price"]), float(greeks["delta"]), float(greeks["gamma"])


ARMS = {
    "baseline": [(SnowballPDESolver, "_solve", counting_solve)],
    "a_memo": [(SnowballPDESolver, "_solve", memo_solve),
               (SnowballPDESolver, "price", _flagged(_ORIGINAL_PRICE)),
               (SnowballPDESolver, "calculate_greeks", _flagged(_ORIGINAL_GREEKS))],
    "b_greeks": [(SnowballPDESolver, "_solve", counting_solve),
                 (RepricingPricer, "_price_env", greeks_only_price_env)],
}


@contextmanager
def arm(name: str):
    patches = ARMS[name]
    saved = [(owner, attr, owner.__dict__[attr]) for owner, attr, _ in patches]
    for owner, attr, value in patches:
        setattr(owner, attr, value)
    COUNTS.update(solves=0, memo_hits=0)
    try:
        yield
    finally:
        for owner, attr, value in saved:
            setattr(owner, attr, value)


# -- the study path -----------------------------------------------------------

def study_setup(path: int):
    bootstrap, _, _ = S02.S01.load_paths(C.DEFAULT_OUT_DIR)
    coupon = float(C.read_json(C.DEFAULT_OUT_DIR / "coupon.json")["coupon"]["coupon"])
    terms = S02.study_terms(bootstrap.dates, maturity_months=C.Q.MATURITY_MONTHS, lockout_months=C.Q.LOCKOUT_MONTHS)
    s0 = float(bootstrap.spot[0, 0])
    product = C.Q.build_product(terms, s0, coupon)
    cfg = S02.cell_config(product, MODEL, HEDGE, provider="exact", cost_bp=C.COST_BP, workers=1,
                          batch_paths=None, quad_grid=401)
    pde = C.engine_config(MODEL, "pde", quad_grid=401, s0=s0)
    cfg = dataclasses.replace(cfg, engine_config=pde, metadata={**cfg.metadata, "engine": "pde", "provider": "per_date"})
    return bootstrap, bootstrap.take([path]), product, cfg, pde, s0


def run_arm(name: str, cfg, batch) -> Dict[str, Any]:
    with arm(name):
        started = time.perf_counter()
        results = run_ensemble(cfg, batch)
        seconds = time.perf_counter() - started
        counts = dict(COUNTS)
    states = int(results.manifest["engine_calls"])
    return {"results": results, "seconds": seconds, "states": states, **counts,
            "solves_per_state": counts["solves"] / states, "seconds_per_state": seconds / states}


def bitwise(a, b) -> Dict[str, Any]:
    differ = [n for n in FLOAT_COLUMNS if not np.array_equal(getattr(a.cube, n), getattr(b.cube, n), equal_nan=True)]
    differ += [n for n in BOOL_COLUMNS if not np.array_equal(getattr(a.cube, n), getattr(b.cube, n))]
    worst = {n: float(np.nanmax(np.abs(getattr(a.cube, n) - getattr(b.cube, n)))) for n in ("product_mtm", "delta", "gamma")}
    return {"identical": not differ and a.cube.active_contract == b.cube.active_contract
            and pd.DataFrame(a.trades).equals(pd.DataFrame(b.trades)),
            "columns_differ": differ, "max_abs_gap": worst}


def env_on_day(paths, d: int, *, spot=None, vol=None, rate=None, div=None, when=None) -> PricingEnvironment:
    chain = day_chain(paths, d, multiplier=C.Q.FUTURES_MULTIPLIER)
    code, _ = day_active_contract(chain, FuturesRollPolicy(roll_days_before_expiry=C.Q.ROLL_DAYS_BEFORE_EXPIRY), None)
    s, r = float(paths.spot[0, d]), float(paths.rate[0, d])
    if div is None:
        div = dividend_yield_for_day(chain, 0, spot=s, rate=r, engine_config=C.engine_config(MODEL, "quad", quad_grid=401),
                                     active_contract=code, curve_tenors=paths.tenor_grid, curve_carry=paths.carry[0, d])
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=s if spot is None else spot, asset_name=C.Q.UNDERLYING_NAME),
        vol_surface=FlatVolSurface(volatility=float(paths.atm_vol[0, d]) if vol is None else vol),
        rate_curve=FlatRateCurve(rate=r if rate is None else rate), div_yield=div,
        valuation_date=(pd.Timestamp(paths.dates[d]) if when is None else when).to_pydatetime(),
    )


def memo_probes(one, product, pde, s0) -> Dict[str, Any]:
    """Under arm (a): the unchanged repeat must HIT; each input changed alone must MISS (a real solve)."""
    d = 60
    date = pd.Timestamp(one.dates[d])
    pricer = RepricingPricer(product, engine_config=pde, start_date=one.dates[0], underlying=C.Q.UNDERLYING_NAME,
                             cache=StateCache(CacheConfig(memory_bytes=1_000_000)))
    alive = pricer.aged_product(date, knocked_in=False)
    ki = pricer.aged_product(date, knocked_in=True)
    older = pricer.aged_product(pd.Timestamp(one.dates[d + 1]), knocked_in=False)
    base = env_on_day(one, d)
    engine = create_pricing_engine(product, pde)
    variants = {
        "repeat": (alive, base, False),
        "spot": (alive, env_on_day(one, d, spot=float(one.spot[0, d]) * 1.001), True),
        "vol": (alive, env_on_day(one, d, vol=float(one.atm_vol[0, d]) + 0.001), True),
        "rate": (alive, env_on_day(one, d, rate=float(one.rate[0, d]) + 0.0001), True),
        "dividend object": (alive, env_on_day(one, d, div=SignedDividendYield(0.10)), True),
        "valuation date": (alive, env_on_day(one, d, when=date + pd.Timedelta(days=1)), True),
        "knocked in (_otc_ attribute only)": (ki, base, True),
        "product aged one more day": (older, base, True),
    }
    out: Dict[str, Any] = {
        "alive_vs_ki_differ_only_in_lifecycle_attrs": (repricing._canonical(alive, set()) == repricing._canonical(ki, set())
                                                       and structural_key(alive) != structural_key(ki)),
    }
    with arm("a_memo"):
        for label, (prod, env, must_miss) in variants.items():
            engine.price(alive, base)                                # prime the memo with the base state
            before = COUNTS["solves"]
            engine.calculate_greeks(prod, env)
            missed = COUNTS["solves"] > before
            out[label] = {"must_miss": must_miss, "missed": missed, "ok": missed == must_miss}
        engine.price(alive, base)
        clone = engine.create_bump_context(alive, base)
        before = COUNTS["solves"]
        clone.calculate_greeks(alive, base)
        out["bump clone"] = {"must_miss": True, "missed": COUNTS["solves"] > before,
                             "ok": COUNTS["solves"] > before}
    return out


def run_suite(arm_name: str, suite: str) -> Dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--suite", arm_name, *SUITES[suite]],
                          cwd=C.PROJECT_ROOT, capture_output=True, text=True)
    tail = [line for line in proc.stdout.splitlines() if line.strip()][-1:] or [""]
    return {"arm": arm_name, "suite": suite, "exit_code": proc.returncode, "passed": proc.returncode == 0,
            "summary": tail[0], "seconds": time.perf_counter() - started}


def suite_mode(arm_name: str, targets: List[str]) -> int:
    import pytest
    with arm(arm_name):
        return int(pytest.main([*targets, "-n0", "-q", "-p", "no:cacheprovider"]))


def callers() -> List[str]:
    proc = subprocess.run(["git", "grep", "-l", "-E", r"EngineType\.PDE|PDEEngine\(", "--", "quantark"],
                          cwd=C.PROJECT_ROOT, capture_output=True, text=True)
    return sorted(proc.stdout.split())


def write_matrix(record: Dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms = record["arms"]
    suites = {(s["arm"], s["suite"]): s for s in record["suites"]}

    def suite_cell(arm_name, suite):
        s = suites.get((arm_name, suite))
        return "–" if s is None else f"{'pass' if s['passed'] else 'FAIL'} ({s['summary']})"

    probes = record["memo_probes"]
    probe_ok = all(v["ok"] for k, v in probes.items() if isinstance(v, dict))
    rows = [
        ("solves per state", *(f"{arms[a]['solves_per_state']:.3f}" for a in ("baseline", "a_memo", "b_greeks"))),
        ("seconds per state", f"{arms['baseline']['seconds_per_state']:.3f} / {arms['baseline_again']['seconds_per_state']:.3f} (repeat)",
         f"{arms['a_memo']['seconds_per_state']:.3f}", f"{arms['b_greeks']['seconds_per_state']:.3f}"),
        ("study path bitwise vs baseline", "–", str(record["bitwise"]["a_memo"]["identical"]), str(record["bitwise"]["b_greeks"]["identical"])),
        ("conformance (oracle parity)", suite_cell("baseline", "conformance"), suite_cell("a_memo", "conformance"), suite_cell("b_greeks", "conformance")),
        ("replay goldens", suite_cell("baseline", "replay_goldens"), suite_cell("a_memo", "replay_goldens"), "– (replay untouched)"),
        ("memo miss probes", "–", f"{'all ok' if probe_ok else 'FAIL'}; alive/KI differ only in lifecycle attrs: {probes['alive_vs_ki_differ_only_in_lifecycle_attrs']}", "–"),
        ("callers affected", "–", f"every price+greeks pair on the snowball PDE family ({len(record['callers'])} files reach PDEEngine)", "RepricingPricer only"),
    ]
    lines = ["# Phase 0 decision matrix: one PDE solve per state", "",
             f"Path {record['path']} of the banked batch, `{MODEL}`/`{HEDGE}`, pinned PDE mesh; "
             f"{arms['baseline']['states']} states; arms back-to-back in one window.", "",
             "| | baseline (two calls) | (a) solve memo | (b) greeks-only provider |", "|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", "Callers reaching PDEEngine:", ""] + [f"- `{c}`" for c in record["callers"]]
    (OUT / "decision_matrix.md").write_text("\n".join(lines) + "\n")
    serializable = {k: v for k, v in record.items() if k != "arms"}
    serializable["arms"] = {a: {k: v for k, v in r.items() if k != "results"} for a, r in arms.items()}
    (OUT / "decision_matrix.json").write_text(json.dumps(serializable, indent=2, default=str))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=int, default=0)
    parser.add_argument("--skip-suites", action="store_true")
    parser.add_argument("--suite", choices=tuple(ARMS), default=None, help=argparse.SUPPRESS)
    parser.add_argument("targets", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.suite is not None:
        return suite_mode(args.suite, args.targets)
    _, one, product, cfg, pde, s0 = study_setup(args.path)
    arms = {}
    for name in ("baseline", "a_memo", "b_greeks"):
        arms[name] = run_arm(name, cfg, one)
        print(f"{name:14s} {arms[name]['states']} states, {arms[name]['solves']} solves, "
              f"{arms[name]['seconds_per_state']:.3f} s/state", flush=True)
    arms["baseline_again"] = run_arm("baseline", cfg, one)
    print(f"baseline again {arms['baseline_again']['seconds_per_state']:.3f} s/state", flush=True)
    record = {
        "path": args.path, "arms": arms, "callers": callers(),
        "bitwise": {name: bitwise(arms[name]["results"], arms["baseline"]["results"]) for name in ("a_memo", "b_greeks")},
        "baseline_repeat_bitwise": bitwise(arms["baseline_again"]["results"], arms["baseline"]["results"]),
        "memo_probes": memo_probes(one, product, pde, s0),
        "suites": [],
    }
    if not args.skip_suites:
        for arm_name, suite in (("baseline", "conformance"), ("a_memo", "conformance"), ("b_greeks", "conformance"),
                                ("baseline", "replay_goldens"), ("a_memo", "replay_goldens")):
            record["suites"].append(run_suite(arm_name, suite))
            print(f"suite {suite} under {arm_name}: {record['suites'][-1]['summary']}", flush=True)
    write_matrix(record)
    print((OUT / "decision_matrix.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run without the suites**

Run: `.venv/bin/python example/snowball_simulated_paths/phase0_single_solve.py --skip-suites 2>&1 | grep -v -i warn | tail -30`
Expected: four arm lines (`baseline`, `a_memo`, `b_greeks`, `baseline again`); `baseline` about 2.0 solves per state; the matrix printed. Roughly 8–12 minutes (path 0 knocks in and lives to maturity). If an arm raises, fix the demo (not library code) and re-run.

- [ ] **Step 3: Full run with the parity suites**

Run: `.venv/bin/python example/snowball_simulated_paths/phase0_single_solve.py 2>&1 | grep -v -i warn | tail -40`
Expected: five suite lines and the complete matrix at `output/snowball_simulated_paths/phase0/decision_matrix.md`. Check `baseline_repeat_bitwise.identical` in the JSON: if the baseline is not bitwise-identical to its own repeat, the bitwise row for the candidates cannot be read, so report that rather than a candidate verdict.

- [ ] **Step 4: Commit the script (not its output)**

```bash
cat > /tmp/phase0_msg.txt <<'EOF'
example(simulated-paths): Phase 0 demo -- one PDE solve per state, two candidates

A runtime-patched script, no library change: (a) a one-entry solve memo on
SnowballPDESolver keyed structurally WITH lifecycle attributes, (b) the
repricing provider taking its mark from calculate_greeks. Evidence per arm:
one study path bitwise against the two-call baseline, the conformance suite,
the replay goldens under (a), memo miss probes, seconds per state
back-to-back. Writes the decision matrix under output/.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git add example/snowball_simulated_paths/phase0_single_solve.py
git commit -F /tmp/phase0_msg.txt -- example/snowball_simulated_paths/phase0_single_solve.py
```

---

### Task 2: Decision checkpoint (user)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md` (section 3.2, append a "Decision" paragraph)

**Interfaces:**
- Consumes: `output/snowball_simulated_paths/phase0/decision_matrix.md`.
- Produces: the chosen branch: **A** (Tasks 3A, 4A), **B** (Task 3B), or **neither** (skip to Task 5; stage-2 budget doubles).

- [ ] **Step 1: Present the matrix to the user and stop**

Paste the matrix table into the conversation with a recommendation grounded in its rows: prefer (a) only if the study path is bitwise, conformance and replay goldens pass under (a), and every memo probe is `ok`; prefer (b) if (a) fails any of those but (b) is bitwise and passes conformance; recommend neither if both fail bitwise. Wait for the user's choice. Do not start Task 3A/3B/4A before it.

- [ ] **Step 2: Record the decision in the spec**

Append to section 3.2 of the spec (fill the bracketed words from the user's answer and the matrix; they are not optional):

```markdown
**Decision (2026-09-15, Phase 0).** The user chose candidate [a | b | neither]
from `output/snowball_simulated_paths/phase0/decision_matrix.md`: solves per
state [baseline x.xxx, chosen y.yyy], study path bitwise [true/false],
conformance [pass/fail], replay goldens [pass/fail/n.a.].
```

- [ ] **Step 3: Commit**

```bash
git add -f docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md
git commit -m "docs(spec): record the Phase 0 single-solve decision

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md
```

---

### Task 3A: (branch A) structural rendering moves to `quantark/util/fingerprint.py`

**Files:**
- Create: `quantark/util/fingerprint.py`
- Modify: `quantark/backtest/simulation/pricing/repricing.py:62-120` (remove `_canonical`, import the util)
- Test: `test/simulation/test_util_fingerprint.py`

**Interfaces:**
- Produces: `canonical_form(obj: Any, *, skip_prefixes: Tuple[str, ...] = ()) -> str`. `repricing.product_fingerprint(product) -> str` unchanged in value.

- [ ] **Step 1: Confirm the pinned product fingerprint on the unmodified tree**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'test'); from simulation.conftest import short_snowball; from quantark.backtest.simulation.pricing.repricing import product_fingerprint; print(product_fingerprint(short_snowball()))"`
Expected: `e0903a579ef46f0bbd0179a61d67ad10` (measured on `f283d459`, 2026-09-15). If it prints anything else, the fixture changed since this plan was written: use the printed value in Step 2 and say so in the commit message.

- [ ] **Step 2: Write the failing tests**

```python
"""``canonical_form``: the structural rendering behind product fingerprints and the PDE solve memo."""
from __future__ import annotations

import pandas as pd

from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer, product_fingerprint

from .conftest import pde_engine_config, short_snowball

#: Measured before the move (f283d459). The disk tier keys shards by this
#: value, so a change here orphans every banked cache.
PINNED_SHORT_SNOWBALL = "e0903a579ef46f0bbd0179a61d67ad10"


def test_the_product_fingerprint_is_unchanged_by_the_move():
    assert product_fingerprint(short_snowball()) == PINNED_SHORT_SNOWBALL


def test_lifecycle_attributes_are_kept_unless_skipped():
    from quantark.util.fingerprint import canonical_form

    pricer = RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=pd.Timestamp("2024-01-02"),
                             underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=1_000_000)))
    day = pd.Timestamp("2024-01-05")
    alive = pricer.aged_product(day, knocked_in=False)
    ki = pricer.aged_product(day, knocked_in=True)
    assert canonical_form(alive) != canonical_form(ki)
    assert "_otc_lifecycle_knocked_in" in canonical_form(ki)
    assert canonical_form(alive, skip_prefixes=("_otc_",)) == canonical_form(ki, skip_prefixes=("_otc_",))
```

- [ ] **Step 3: Run to verify the second test fails**

Run: `.venv/bin/python -m pytest test/simulation/test_util_fingerprint.py -n0 -q`
Expected: `test_the_product_fingerprint_is_unchanged_by_the_move` PASSES (pinned on the current tree); `test_lifecycle_attributes_are_kept_unless_skipped` FAILS with `ModuleNotFoundError: No module named 'quantark.util.fingerprint'`.

- [ ] **Step 4: Create the util (the body is `repricing._canonical` verbatim, with the skip set passed down)**

```python
"""Structural, address-free renderings of objects, for cache keys and fingerprints.

``repr`` is not enough for a key: ``SnowballOption.__repr__`` is a rounded
summary that omits the observation schedule, so two contracts that differ
only in their observation days would print alike.  ``canonical_form``
walks the object graph instead -- dataclass fields, ``__dict__`` entries,
sequences, arrays, enums, dates.

``skip_prefixes`` drops attributes by name prefix.  The simulation's state
cache skips lifecycle state (``_otc_``) because it keys the knock-in flag
separately; a solver memo must NOT skip it, because the PDE solver reads
``_otc_lifecycle_knocked_in``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
from typing import Any, Tuple

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError


def canonical_form(obj: Any, *, skip_prefixes: Tuple[str, ...] = ()) -> str:
    """The structural rendering of ``obj``; attributes starting with a prefix in ``skip_prefixes`` are left out."""
    return _canonical(obj, set(), 0, tuple(skip_prefixes))


def _canonical(obj: Any, seen: set, depth: int, skip: Tuple[str, ...]) -> str:
    if depth > 12:
        raise ValidationError("object nests too deeply to fingerprint")
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        return repr(obj)
    if isinstance(obj, float):
        return repr(float(obj))
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__name__}.{obj.name}"
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return f"{type(obj).__name__}:{obj.isoformat()}"
    if isinstance(obj, np.ndarray):
        return f"ndarray:{obj.dtype}:{obj.shape}:{np.ascontiguousarray(obj).tobytes().hex()}"
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return repr(obj.item())
    if isinstance(obj, (list, tuple)):
        inner = ",".join(_canonical(x, seen, depth + 1, skip) for x in obj)
        return f"{type(obj).__name__}[{inner}]"
    if isinstance(obj, (set, frozenset)):
        inner = ",".join(sorted(_canonical(x, seen, depth + 1, skip) for x in obj))
        return f"{type(obj).__name__}{{{inner}}}"
    if isinstance(obj, dict):
        inner = ",".join(
            f"{_canonical(k, seen, depth + 1, skip)}:{_canonical(v, seen, depth + 1, skip)}"
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))
        )
        return f"dict{{{inner}}}"
    if callable(obj) and not hasattr(obj, "__dict__"):
        return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"
    marker = id(obj)
    if marker in seen:
        return f"<cycle:{type(obj).__name__}>"
    seen.add(marker)
    if dataclasses.is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        items = sorted(vars(obj).items())
    else:
        return f"{type(obj).__name__}:{obj!r}"
    inner = ",".join(
        f"{name}={_canonical(value, seen, depth + 1, skip)}"
        for name, value in items
        if not name.startswith(skip) and not callable(value)
    )
    seen.discard(marker)
    return f"{type(obj).__name__}({inner})"
```

Check the verbatim claim before continuing: `diff <(sed -n '62,115p' quantark/backtest/simulation/pricing/repricing.py) quantark/util/fingerprint.py` must show only the signature, the `skip` threading, the error message and the docstring.

- [ ] **Step 5: Point `repricing.py` at the util**

In `quantark/backtest/simulation/pricing/repricing.py`: delete the whole `def _canonical(...)` function (lines 62–115) and the now-unused imports `dataclasses`, `datetime as _dt`, `enum` if nothing else in the file uses them (check with `grep -n "dataclasses\.\|_dt\.\|enum\." quantark/backtest/simulation/pricing/repricing.py`). Add `from quantark.util.fingerprint import canonical_form` and replace `product_fingerprint` with:

```python
def product_fingerprint(product: Any) -> str:
    """Identity of the contract as priced (terms only, no lifecycle state)."""
    return hashlib.blake2b(canonical_form(product, skip_prefixes=_STATE_PREFIXES).encode(),
                           digest_size=16).hexdigest()
```

Then `git grep -n "_canonical" -- quantark test example` must show no remaining reference to `repricing._canonical` except in `example/snowball_simulated_paths/phase0_single_solve.py`; update that script's `structural_key` to `return canonical_form(obj)` and `repricing._canonical(x, set())` calls to `canonical_form(x, skip_prefixes=("_otc_",))`, importing `canonical_form` from the util.

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest test/simulation/test_util_fingerprint.py test/simulation/test_repricing.py test/simulation/test_cache.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add quantark/util/fingerprint.py test/simulation/test_util_fingerprint.py
git commit -m "refactor(util): the structural object rendering moves to quantark.util.fingerprint

The state cache's product fingerprint is unchanged (pinned). The move lets
the PDE solver key a memo on it without importing the backtest layer, and
the skip set becomes a parameter: the solver must keep _otc_ lifecycle
attributes that the state cache drops.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- quantark/util/fingerprint.py quantark/backtest/simulation/pricing/repricing.py test/simulation/test_util_fingerprint.py example/snowball_simulated_paths/phase0_single_solve.py
```

---

### Task 4A: (branch A) one-entry solve memo on the snowball PDE family

**Files:**
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py` (`__init__` near line 360; `_solve` at 380; `_price_with_solution` line 694; `calculate_greeks` line 1495; new methods; `_bump_transient_attrs` override)
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py:286` and `ko_reset_snowball_pde_solver.py:340` (first line of `_solve`)
- Test: `test/simulation/test_solve_memo.py`

**Interfaces:**
- Consumes: `canonical_form` (Task 3A).
- Produces: `SnowballPDESolver._readout_solve(product, pricing_env) -> PDESolutionResult`; attribute `_solve_memo: Optional[Tuple[str, Tuple[Any, ...], PDESolutionResult]]`. Public behaviour: `price` then `calculate_greeks` on the same state runs one `_solve`.

- [ ] **Step 1: Write the failing tests**

```python
"""The snowball PDE solve memo: price then greeks on one state solves once, and nothing else ever hits."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.priceenv import PricingEnvironment

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")
DAY = START + pd.Timedelta(days=3)


def _env(spot=SPOT, vol=0.22, rate=RATE, div=None, when=DAY):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="CSI1000"), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=SignedDividendYield(0.05) if div is None else div,
        valuation_date=when.to_pydatetime(),
    )


def _aged(knocked_in=False, day=DAY):
    pricer = RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=START,
                             underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=1_000_000)))
    return pricer.aged_product(day, knocked_in=knocked_in)


@pytest.fixture
def solves(monkeypatch):
    count = {"n": 0}
    original = SnowballPDESolver._solve

    def counting(self, product, pricing_env):
        count["n"] += 1
        return original(self, product, pricing_env)

    monkeypatch.setattr(SnowballPDESolver, "_solve", counting)
    return count


def test_price_then_greeks_on_one_state_solves_once_and_changes_no_bit(solves):
    product, env = _aged(), _env()
    engine = create_pricing_engine(product, pde_engine_config())
    price = engine.price(product, env)
    greeks = engine.calculate_greeks(product, env)
    assert solves["n"] == 1
    reference = create_pricing_engine(product, pde_engine_config())
    ref_price = reference.price(product, env)
    reference._get_solver(product)._solve_memo = None          # today's behaviour: a second real solve
    ref_greeks = reference.calculate_greeks(product, env)
    assert price == ref_price and greeks == ref_greeks
    assert greeks["price"] == price


VARIANTS = {
    "spot": lambda: (_aged(), _env(spot=SPOT * 1.001)),
    "vol": lambda: (_aged(), _env(vol=0.221)),
    "rate": lambda: (_aged(), _env(rate=RATE + 1e-4)),
    "dividend object": lambda: (_aged(), _env(div=TermStructureDividendYield(times=[0.01, 0.05], yields=[0.05, 0.06]))),
    "valuation date": lambda: (_aged(), _env(when=DAY + pd.Timedelta(days=1))),
    "knocked in": lambda: (_aged(knocked_in=True), _env()),
    "aged one more day": lambda: (_aged(day=DAY + pd.Timedelta(days=1)), _env()),
}


@pytest.mark.parametrize("label", sorted(VARIANTS))
def test_each_input_changed_alone_is_a_new_solve(solves, label):
    product, env = _aged(), _env()
    engine = create_pricing_engine(product, pde_engine_config())
    engine.price(product, env)
    other_product, other_env = VARIANTS[label]()
    engine.calculate_greeks(other_product, other_env)
    assert solves["n"] == 2, label


def test_injected_session_state_is_compared_by_identity(solves):
    product, env = _aged(), _env()
    engine = create_pricing_engine(product, pde_engine_config())
    engine.price(product, env)
    # An empty matrix pack is a valid injection: misses fall through and build locally.
    engine._get_solver(product)._session_matrix_pack = {}
    engine.calculate_greeks(product, env)
    assert solves["n"] == 2


def test_a_bump_clone_starts_without_the_memo():
    product, env = _aged(), _env()
    engine = create_pricing_engine(product, pde_engine_config())
    engine.price(product, env)
    assert engine._get_solver(product)._solve_memo is not None
    clone = engine.create_bump_context(product, env)
    assert clone._get_solver(product)._solve_memo is None


def test_a_life_surface_solve_between_clears_the_memo(solves):
    product, env = _aged(), _env()
    engine = create_pricing_engine(product, pde_engine_config())
    solver = engine._get_solver(product)
    engine.price(product, env)
    solver.solve_life_surface(product, env, extra_times=())
    engine.calculate_greeks(product, env)
    assert solves["n"] == 3


def _first_statement(func) -> str:
    """The first statement of a method body, docstring skipped (with or without one)."""
    import ast
    import textwrap

    node = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.unparse(body[0])


@pytest.mark.parametrize("cls", [SnowballPDESolver, PhoenixPDESolver, KOResetSnowballPDESolver])
def test_every_solve_in_the_family_clears_the_memo_first(cls):
    assert _first_statement(cls.__dict__["_solve"]) == "self._solve_memo = None", cls.__name__
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/simulation/test_solve_memo.py -n0 -q`
Expected: FAIL — `AttributeError: ... has no attribute '_solve_memo'` and `assert 2 == 1` on the first test.

- [ ] **Step 3: Implement the memo in `SnowballPDESolver`**

Add the import near the other `quantark.util` imports:

```python
from quantark.util.fingerprint import canonical_form
```

In `__init__`, directly after `self._product_token_memo: Dict[...] = {}`:

```python
        # One-entry memo for price()/calculate_greeks(): the replay and the
        # simulation call both on the same state, and each used to solve.
        # Keyed structurally on everything _solve reads, lifecycle
        # attributes included, plus the injected session objects BY
        # IDENTITY; every _solve clears it on entry (_readout_solve).
        self._solve_memo: Optional[Tuple[str, Tuple[Any, ...], PDESolutionResult]] = None
```

Add as class attribute next to `_extra_time_nodes`:

```python
    #: Injected state a memoised result must not outlive: a hit needs the
    #: SAME objects (``is``) the result was solved with.
    _SOLVE_MEMO_STATE_ATTRS: Tuple[str, ...] = (
        "_session_grids", "_session_step_coefficients", "_session_matrix_pack",
        "_session_banded_pack", "_frozen_base_layout",
    )
```

Add methods after `_solve`:

```python
    def _solve_memo_key(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> str:
        """Everything a solve reads, rendered structurally (lifecycle attributes kept)."""
        return repr((
            type(self).__qualname__, canonical_form(product), canonical_form(pricing_env),
            canonical_form(self._params_cache_key()), self._resolve_cache_strategy(),
            tuple(self._extra_time_nodes),
        ))

    def _readout_solve(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> PDESolutionResult:
        """``_solve`` for ``price`` and ``calculate_greeks``: the same state twice in a row solves once."""
        key = self._solve_memo_key(product, pricing_env)
        refs = tuple(getattr(self, name, None) for name in self._SOLVE_MEMO_STATE_ATTRS)
        memo = self._solve_memo
        if memo is not None and memo[0] == key and all(a is b for a, b in zip(memo[1], refs)):
            return memo[2]
        result = self._solve(product, pricing_env)          # clears the memo on entry
        self._solve_memo = (key, refs, result)
        return result

    def _bump_transient_attrs(self) -> tuple:
        return super()._bump_transient_attrs() + ("_solve_memo",)
```

Make the first statement of `SnowballPDESolver._solve` (after its docstring):

```python
        self._solve_memo = None
```

Replace line 694 (`result = self._solve(product, pricing_env)` in `_price_with_solution`) and line 1495 (in `calculate_greeks`) with:

```python
        result = self._readout_solve(product, pricing_env)
```

Check that `Any`, `Optional`, `Tuple` are imported from `typing` at the top of the file; add any missing.

- [ ] **Step 4: Clear the memo in the two subclass `_solve` overrides**

In `quantark/asset/equity/engine/pde/phoenix_pde_solver.py` `_solve` (line 286), first statement after the docstring, and in `quantark/asset/equity/engine/pde/ko_reset_snowball_pde_solver.py` `_solve` (line 340), first statement:

```python
        self._solve_memo = None
```

- [ ] **Step 5: Run the memo tests**

Run: `.venv/bin/python -m pytest test/simulation/test_solve_memo.py -n0 -q`
Expected: all PASS.

- [ ] **Step 6: Run the parity and PDE suites**

Run: `.venv/bin/python -m pytest test/simulation test/test_replay_goldens.py test/test_snowball_pde.py test/test_snowball_pde_date_schedule.py test/test_snowball_pde_knocked_in_grid.py test/test_ko_reset_snowball_pde.py test/pde_grid -q`
Expected: all PASS, replay goldens byte-identical (the golden test compares frames).

- [ ] **Step 7: Commit**

```bash
git add test/simulation/test_solve_memo.py
git commit -m "perf(pde): price then greeks on one snowball state solves once

Exact repricing and the replay call price() and calculate_greeks() on the
same state, and each ran _solve. A one-entry memo on the snowball PDE family
serves the second call from the first solve. The key is structural and keeps
_otc_ lifecycle attributes (the solver reads _otc_lifecycle_knocked_in);
injected session objects compare by identity; every _solve clears the memo
first and bump clones start without it. Bitwise: Phase 0 matrix, replay
goldens unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- quantark/asset/equity/engine/pde/snowball_pde_solver.py quantark/asset/equity/engine/pde/phoenix_pde_solver.py quantark/asset/equity/engine/pde/ko_reset_snowball_pde_solver.py test/simulation/test_solve_memo.py
```

---

### Task 3B: (branch B) exact repricing on the flat-vol PDE engine makes one call per state

**Files:**
- Modify: `quantark/backtest/simulation/pricing/repricing.py` (`__init__` after `self._engine = ...`; `_price_env` lines 368–393; class docstring)
- Test: `test/simulation/test_repricing.py`

**Interfaces:**
- Produces: `RepricingPricer._greeks_carry_price: bool` (True iff `engine_config.pricing_engine_type == EngineType.PDE` and `engine_config.vol_model is None`). Values returned by `price_day` unchanged.

- [ ] **Step 1: Write the failing tests** (append to `test/simulation/test_repricing.py`)

```python
def test_exact_pde_repricing_solves_each_state_once(monkeypatch):
    from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

    count = {"n": 0}
    original = SnowballPDESolver._solve

    def counting(self, product, pricing_env):
        count["n"] += 1
        return original(self, product, pricing_env)

    monkeypatch.setattr(SnowballPDESolver, "_solve", counting)
    pricer = _pricer()
    pricer.price_day(_states(3, [SPOT * 0.9, SPOT, SPOT * 1.02]))
    assert count["n"] == 3 and pricer.stats()["engine_calls"] == 3


def test_only_the_flat_vol_pde_engine_takes_the_one_call_path():
    from quantark.util.enum.engine_enums import EngineType

    assert _pricer()._greeks_carry_price
    quad = RepricingPricer(short_snowball(), engine_config=pde_engine_config(pricing_engine_type=EngineType.QUADRATURE),
                           start_date=START, underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=1_000_000)))
    assert not quad._greeks_carry_price
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/simulation/test_repricing.py -n0 -q -k "solves_each_state_once or one_call_path"`
Expected: FAIL — `assert 6 == 3` and `AttributeError: 'RepricingPricer' object has no attribute '_greeks_carry_price'`.

- [ ] **Step 3: Implement**

Add import: `from quantark.util.enum.engine_enums import EngineType`.

In `__init__`, after `self._engine = create_pricing_engine(...)`:

```python
        # On the flat-vol PDE engine calculate_greeks returns the price
        # price() would, read off the same solve, so a separate price() call
        # only repeated the solve (Phase 0, spec 3.2).  Every other engine
        # keeps the replay's two calls.
        self._greeks_carry_price = (
            getattr(engine_config, "pricing_engine_type", None) == EngineType.PDE
            and getattr(engine_config, "vol_model", None) is None
        )
```

Replace the `try:` block of `_price_env` with:

```python
        try:
            if self._greeks_carry_price:
                greeks = self._engine.calculate_greeks(product, env)
                if "price" not in greeks:
                    raise ValidationError(
                        f"{type(self._engine).__name__}.calculate_greeks returned no price; "
                        "one-call repricing reads the price off the same solve"
                    )
                price = float(greeks["price"])
            else:
                price = float(self._engine.price(product, env))
                greeks = self._engine.calculate_greeks(product, env)
        except Exception as exc:  # fail closed with the state in the message
            raise ValidationError(
                f"pricing failed at {label}: spot={spot!r}, vol={vol!r}, knocked_in={knocked_in}: {exc}"
            ) from exc
```

Update the `_price_env` docstring to: `"""One state: on the flat-vol PDE engine one calculate_greeks call (price read off the same solve); otherwise price then calculate_greeks, as the replay calls them."""`

- [ ] **Step 4: Run the repricing and conformance suites**

Run: `.venv/bin/python -m pytest test/simulation/test_repricing.py test/simulation/test_conformance.py test/simulation/test_engine.py -q`
Expected: all PASS — in particular `test_exact_mode_equals_a_direct_engine_call_state_by_state` and every `test_every_path_matches_the_replay_engine_exactly` case (zero-tolerance oracle).

- [ ] **Step 5: Commit**

```bash
git commit -m "perf(simulation): exact repricing on the flat-vol PDE engine solves each state once

calculate_greeks already returns the price price() reads off the same solve;
the extra price() call repeated a full backward solve. The replay keeps its
two calls; the conformance oracle still matches at zero tolerance (Phase 0).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- quantark/backtest/simulation/pricing/repricing.py test/simulation/test_repricing.py
```

---

### Task 5: `01_build_paths.py --history-end`

**Files:**
- Modify: `example/snowball_simulated_paths/01_build_paths.py` (`load_real_history`, `write_paths`, `parse_args`, `main`; new `cut_history_frames`)
- Test: `test/test_snowball_simulated_paths_study.py`

**Interfaces:**
- Produces: `cut_history_frames(spot: pd.DataFrame, futures: pd.DataFrame, history_end: date) -> Tuple[pd.DataFrame, pd.DataFrame]`; `load_real_history(history_dir, *, rate, vol_tenor, history_end: Optional[date] = None) -> PathHistory`; `write_paths(out_dir, bootstrap, stress, *, history, history_end: Optional[date] = None) -> Dict`; CLI `--history-end YYYY-MM-DD` → `args.history_end: Optional[date]`.

- [ ] **Step 1: Write the failing test** (after `test_paths_are_written_with_a_manifest_and_read_back`)

```python
def test_history_end_cuts_the_frames_and_is_recorded(tmp_path):
    spot, vol, futures = synthetic_frames()
    end = pd.Timestamp(spot["date"].iloc[59])
    cut_spot, cut_futures = S01.cut_history_frames(spot, futures, end.date())
    assert len(cut_spot) == 60 and cut_spot["date"].max() == end and cut_futures["date"].max() == end
    history = S01.build_history(cut_spot, vol, cut_futures, rate=RATE)
    assert history.n_days == 60 and history.dates[-1] == end
    bootstrap, stress = S01.build_paths(history, n_paths=2, n_days=10, seed=1, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="levels")
    manifest = S01.write_paths(tmp_path / "cut", bootstrap, stress, history=history, history_end=end.date())
    assert manifest["history_end"] == str(end.date())
    assert "history_end" not in S01.write_paths(tmp_path / "plain", bootstrap, stress, history=history)
    with pytest.raises(C.Q.StudyDataError, match="no history on or before"):
        S01.cut_history_frames(spot, futures, date(2000, 1, 1))
    assert S01.parse_args(["--history-end", "2026-09-09"]).history_end == date(2026, 9, 9)
    assert S01.parse_args([]).history_end is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -n0 -q -k history_end`
Expected: FAIL with `AttributeError: module ... has no attribute 'cut_history_frames'`.

- [ ] **Step 3: Implement**

Add `from datetime import date, timedelta` (replacing `from datetime import timedelta`). Replace `load_real_history` and add `cut_history_frames`:

```python
def cut_history_frames(spot: pd.DataFrame, futures: pd.DataFrame, history_end: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The spot and futures frames through ``history_end`` inclusive.

    Cutting today's cache at 2026-09-09 reproduces the banked batch's
    history, bootstrap and stress fingerprints exactly (checked 2026-09-15),
    so a larger batch can start from the same state as a banked one.
    """
    end = pd.Timestamp(history_end)
    cut_spot = spot[pd.to_datetime(spot["date"]) <= end].reset_index(drop=True)
    cut_futures = futures[pd.to_datetime(futures["date"]) <= end].reset_index(drop=True)
    if cut_spot.empty:
        raise C.Q.StudyDataError(f"no history on or before {end.date()}")
    return cut_spot, cut_futures


def load_real_history(history_dir, *, rate: float, vol_tenor: float, history_end: Optional[date] = None) -> PathHistory:
    """The study's inputs through the q study's fail-closed loaders, optionally cut at ``history_end``."""
    frames = C.Q.load_history(history_dir)
    spot, futures = frames.spot, frames.futures
    if history_end is not None:
        spot, futures = cut_history_frames(spot, futures, history_end)
    surfaces = C.Q.surface_history(history_dir)
    vol = C.Q.atm_vol_channel([pd.Timestamp(d) for d in spot["date"]], surfaces, vol_tenor)
    return build_history(spot, vol, futures, rate=rate)
```

In `write_paths`, add the keyword `history_end: Optional[date] = None` and build the manifest dict first, then:

```python
    record = {
        "bootstrap_fingerprint": bootstrap.fingerprint(), "stress_fingerprint": stress.fingerprint(),
        "history_fingerprint": history.source_fingerprint,
        "history_first_day": str(history.dates[0].date()), "history_last_day": str(history.dates[-1].date()),
        "n_paths": bootstrap.n_paths, "n_days": bootstrap.n_days,
        "calendar_first_day": str(bootstrap.dates[0].date()), "calendar_last_day": str(bootstrap.dates[-1].date()),
        "bootstrap_meta": bootstrap.meta, "stress_meta": stress.meta,
    }
    if history_end is not None:
        record["history_end"] = str(history_end)
    manifest = jsonable(record)          # JSON-safe now, so it equals what load_paths reads back
```

In `parse_args` add:

```python
    parser.add_argument("--history-end", type=date.fromisoformat, default=None,
                        help="cut the history at this day inclusive (YYYY-MM-DD)")
```

In `main`, pass `history_end=args.history_end` to both `load_real_history(...)` and `write_paths(...)`, and append `f", history cut at {args.history_end}" if args.history_end else ""` to the printed line.

- [ ] **Step 4: Run the study tests**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q -k "history_end or build_history or manifest"`
Expected: PASS.

- [ ] **Step 5: Verify the real cut reproduces the banked batch**

Run:
```bash
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py --quick --history-end 2026-09-09 \
    --out-dir output/snowball_simulated_paths/history_cut_check 2>&1 | grep -v -i warn | tail -2
.venv/bin/python - <<'EOF'
import json
a = json.load(open("output/snowball_simulated_paths/paths/manifest.json"))
b = json.load(open("output/snowball_simulated_paths/history_cut_check/paths/manifest.json"))
for k in ("history_fingerprint", "bootstrap_fingerprint", "stress_fingerprint"):
    print(k, a[k] == b[k])
EOF
```
Expected: `history 2023-05-04..2026-09-09 (816 days)` and three `True` lines.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(example): the simulated-path study cuts its history at a stated day

--history-end reproduces the banked 2026-09-09 batch's history, bootstrap
and stress fingerprints from today's longer cache, so a 2,000-path batch can
start from the same state; the cut is recorded in the paths manifest.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/01_build_paths.py test/test_snowball_simulated_paths_study.py
```

---

### Task 6: `02_ensemble_fleet.py`: `per_date` provider, defaults, check oracle, `--spot-range`, per-run batching, day-0 mark

**Files:**
- Modify: `example/snowball_simulated_paths/02_ensemble_fleet.py` (`PROVIDERS`, `_pricing`, `cell_config`, `run_cell`, `parse_args`, `main`; new `PDE_PROVIDERS`, `batch_for`)
- Modify: `example/snowball_simulated_paths/_sim_common.py` (`CHECK_PATHS`, new `EXACT_CHECK_PATHS`)
- Test: `test/test_snowball_simulated_paths_study.py`

**Interfaces:**
- Produces: `PROVIDERS = ("per_date", "exact", "life_surface", "ladder")`; `PDE_PROVIDERS = ("per_date", "life_surface")`; `batch_for(n_paths: int, workers: int, batch_paths: Optional[int]) -> Optional[int]`; run records gain `day0_product_mtm: float`, `day0_book_mark_bp: float` (absent on a failed run); CLI `--spot-range LO HI`; defaults `--provider per_date`, `--check-paths 0`, `--exact-paths 40`.

- [ ] **Step 1: Write the failing tests** (append; add `from quantark.util.enum.engine_enums import EngineType` to the test imports)

```python
def test_per_date_is_exact_repricing_on_the_pde_engine(tiny_fleet):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    product = C.Q.build_product(fixture_terms(bootstrap.dates), float(bootstrap.spot[0, 0]), coupon.coupon)
    cfg = S02.cell_config(product, "term_flat_q", "front", provider="per_date", **CELL)
    assert cfg.pricing.provider == "repricing" and cfg.pricing.mode == "exact" and cfg.pricing.gate.sample_states == 0
    assert cfg.engine_config.pricing_engine_type == EngineType.PDE
    assert cfg.engine_config.pde_params.grid.points == C.SURFACE_POINTS
    assert (cfg.metadata["provider"], cfg.metadata["engine"]) == ("per_date", "pde")
    exact = S02.cell_config(product, "term_flat_q", "front", provider="exact", **CELL)
    assert exact.engine_config.pricing_engine_type == EngineType.QUADRATURE
    assert S02.config_fingerprint(cfg, bootstrap) != S02.config_fingerprint(exact, bootstrap)
    assert S02.oracle_tolerances(cfg) == {"pv_tolerance": 0.0, "delta_tolerance": 0.0, "contracts_tolerance": 0.0}
    args = S02.parse_args([])
    assert (args.provider, args.check_paths, args.exact_paths, args.oracle_paths) == ("per_date", 0, 40, 3)
    assert tuple(args.spot_range) == C.SURFACE_SPOT_RANGE


def test_each_run_is_batched_for_its_own_path_count():
    assert S02.batch_for(2000, 12, 170) == 167
    assert S02.batch_for(40, 12, 170) == 4
    assert S02.batch_for(5, 12, 170) == 1
    assert S02.batch_for(40, 6, 7) == 7
    assert S02.batch_for(40, 1, None) is None


def test_the_cli_runs_a_per_date_cell_with_an_exact_quad_check(tiny_fleet, tmp_path):
    import shutil
    out, _, _ = tiny_fleet
    shutil.copytree(out / "paths", tmp_path / "paths")
    rc = S02.main(["--out-dir", str(tmp_path), "--cells", f"{C.MODELS[0]}:front", "--exact-paths", "2",
                   "--oracle-paths", "2", "--spot-range", *map(str, FIXTURE_SPOT_RANGE), "--quad-grid", "101",
                   "--maturity-months", "1", "--lockout-months", "1"])
    assert rc == 0
    runs = C.read_json(tmp_path / "fleet_manifest.json")["runs"]
    assert set(runs) == {C.BASELINE_CELL, C.BASELINE_CELL + "__stress", C.BASELINE_CELL + "__exact_quad"}
    main = runs[C.BASELINE_CELL]
    assert main["provider"] == "per_date" and not main["failed"] and main["gate"]["max_pv_gap_bp"] == 0.0
    assert [r["path"] for r in main["oracle"]] == [0, 1] and all(r["passed"] for r in main["oracle"])
    check = runs[C.BASELINE_CELL + "__exact_quad"]
    assert check["provider"] == "exact" and check["n_paths"] == 2 and [r["path"] for r in check["oracle"]] == [0]
    assert runs[C.BASELINE_CELL + "__stress"]["oracle"] == []
    assert np.isfinite(main["day0_book_mark_bp"]) and np.isfinite(check["day0_book_mark_bp"])
```

Also edit the existing `test_the_cli_runs_every_cell_of_the_quick_grid`: add `"--exact-paths", "0",` after `"--check-paths", "0",` (the new default would otherwise add an exact check run on top of an exact bootstrap run, and the test counts four runs).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q -k "per_date or batched or exact_quad_check"`
Expected: FAIL — `ValidationError: provider must be one of ('life_surface', 'ladder', 'exact'), got 'per_date'`, `AttributeError: ... 'batch_for'`.

- [ ] **Step 3: Implement in `_sim_common.py`**

Replace `CHECK_PATHS = 200` with:

```python
#: The QUAD spot-ladder check is off by default: its 2026-09-09 failures were
#: the flat-node carry gap, structural, and Design B replaced the provider it
#: was checking.  Pass --check-paths to run it.
CHECK_PATHS = 0
#: Exact QUAD repricing on each cell's first paths, paired with the cell's
#: own provider on the same paths: the engine check (reported, not gated).
EXACT_CHECK_PATHS = 40
```

- [ ] **Step 4: Implement in `02_ensemble_fleet.py`**

Add `import math`. Update the module docstring's first paragraph to:

```python
Cells are {baseline flat q, term_flat_q, term_opt_tail} x {front, far}.  Each
cell runs the bootstrap batch and the stress set on ``--provider`` (default
``per_date``: exact repricing on the PDE engine, one solve from maturity back
to each state's date with that date's real dividend object), then the first
``--exact-paths`` bootstrap paths on exact QUAD repricing -- the engine check,
paired with the cell on the same paths -- and optionally ``--check-paths``
on the QUAD spot ladder.  Every bootstrap run is oracle-checked on
``--oracle-paths`` single paths against the replay engine, every check run
on the first of them.
```

Replace `PROVIDERS = ...` with:

```python
PROVIDERS = ("per_date", "exact", "life_surface", "ladder")
#: Providers priced by the PDE engine; the others price on QUAD.
PDE_PROVIDERS = ("per_date", "life_surface")
```

In `_pricing`, replace the `if provider == "exact":` branch with:

```python
    if provider in ("per_date", "exact"):
        return PricingProviderConfig(provider="repricing", cache=cache,
                                     gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0))
```

In `cell_config`, replace `engine = "pde" if provider == "life_surface" else "quad"` with:

```python
    engine = "pde" if provider in PDE_PROVIDERS else "quad"
```

Add after `oracle_tolerances`:

```python
def batch_for(n_paths: int, workers: int, batch_paths: Optional[int]) -> Optional[int]:
    """This run's batch size: never more than one batch per worker needs.

    One ``--batch-paths`` for 2,000 bootstrap paths would leave a 40-path
    check run as a single batch on one worker.  Batching is bit-inert and
    not part of the resume fingerprint, so sizing it per run changes no
    number.
    """
    if batch_paths is None:
        return None
    return max(1, min(int(batch_paths), math.ceil(int(n_paths) / max(1, int(workers)))))
```

In `run_cell`, directly before `single = replace(config, workers=1, batch_paths=None)`, add:

```python
    # The book's mark on day 0.  Cells start from the traded price (0), and
    # the coupon is solved on QUAD, so under the PDE this is the engine gap
    # at inception; it is recorded rather than buried in terminal P&L.
    day0_mtm = float(results.cube.product_mtm[0, 0])
```

and add to the success `run = {...}` dict:

```python
        "day0_product_mtm": day0_mtm, "day0_book_mark_bp": day0_mtm / results.notional * 1e4,
```

In `parse_args`, change and add:

```python
    parser.add_argument("--provider", choices=PROVIDERS, default="per_date",
                        help="the bootstrap and stress runs' provider; per_date = exact PDE, exact = exact QUAD")
    parser.add_argument("--check-paths", type=int, default=C.CHECK_PATHS, help="QUAD spot-ladder subset (0 = none)")
    parser.add_argument("--exact-paths", type=int, default=C.EXACT_CHECK_PATHS, help="exact QUAD check subset (0 = none)")
    parser.add_argument("--spot-range", type=float, nargs=2, default=list(C.SURFACE_SPOT_RANGE), metavar=("LO", "HI"),
                        help="PDE grid bounds as fractions of the initial spot")
    parser.add_argument("--quick", action="store_true", help="two cells, at most 8 check paths, 1 oracle path")
```

In `main`:
- the quick block becomes:

```python
    if args.quick:
        args.cells = args.cells or [f"{C.MODELS[0]}:front", "term_flat_q:front"]
        args.check_paths, args.exact_paths = min(args.check_paths, 8), min(args.exact_paths, 8)
        args.oracle_paths = min(args.oracle_paths, 1)
```

- `common` becomes (no `batch_paths` there any more):

```python
    common = dict(cost_bp=args.cost_bp, workers=args.workers, quad_grid=args.quad_grid, disk_dir=disk_dir,
                  spot_range=tuple(args.spot_range))
```

- the plan builds check runs with the first oracle path only:

```python
        if args.check_paths:
            plan.append((f"{cell}__ladder_quad", bootstrap.take(range(min(args.check_paths, bootstrap.n_paths))),
                         "ladder", oracle[:1]))
        if args.exact_paths:
            plan.append((f"{cell}__exact_quad", bootstrap.take(range(min(args.exact_paths, bootstrap.n_paths))),
                         "exact", oracle[:1]))
```

- the loop passes the per-run batch and prints the day-0 mark:

```python
        for name, batch, provider, checks in plan:
            config = cell_config(product, model, hedge, provider=provider,
                                 batch_paths=batch_for(batch.n_paths, args.workers, args.batch_paths), **common)
            run = run_cell(batch, config, args.out_dir / "cells" / name, resume=args.resume, oracle_paths=checks)
            runs[name] = run
            state = "skipped" if run["skipped"] else f"{run['seconds']:.0f}s"
            gate = run["gate"]
            oracle_state = ("–" if not run["oracle"] else "ok" if all(r["passed"] for r in run["oracle"]) else "FAIL")
            day0 = run.get("day0_book_mark_bp")
            print(f"  {name:32s} {state:>8s}  gate {gate['max_pv_gap_bp']:.2f} bp / {gate['max_delta_gap_hands']:.2f} hands "
                  f"({'ok' if gate['passed'] else 'FAIL, no results'})  oracle {oracle_state}  "
                  f"day0 {'–' if day0 is None else f'{day0:.2f} bp'}", flush=True)
```

A resumed-and-skipped run returns its stored `run.json`; runs banked before this task have no `day0_book_mark_bp`, which the print handles with `–`.

- [ ] **Step 5: Run the study tests**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(example): per_date is the simulated-path study's provider

Exact repricing on the pinned PDE engine: each state is one solve from
maturity back to its date with that date's real dividend object, for every
cell (Design B, no hybrid). The QUAD ladder check goes off by default; an
exact-QUAD check on each cell's first 40 paths takes its place, oracle-checked
on one path. --spot-range exposes the grid bounds; each run is batched for its
own path count (bit-inert); every run records its day-0 book mark.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/02_ensemble_fleet.py example/snowball_simulated_paths/_sim_common.py test/test_snowball_simulated_paths_study.py
```

---

### Task 7: `--paths-dir`

**Files:**
- Modify: `example/snowball_simulated_paths/01_build_paths.py` (`load_paths`)
- Modify: `example/snowball_simulated_paths/02_ensemble_fleet.py` (`parse_args`, `main`)
- Test: `test/test_snowball_simulated_paths_study.py`

**Interfaces:**
- Produces: `load_paths(out_dir, *, paths_dir: Optional[Path] = None) -> Tuple[MarketPath, MarketPath, Dict]`; CLI `--paths-dir DIR`.

- [ ] **Step 1: Write the failing test**

```python
def test_paths_dir_reads_a_batch_from_elsewhere_and_writes_nothing_there(tiny_fleet, tmp_path):
    out, _, _ = tiny_fleet
    before = sorted((p.name, p.stat().st_mtime_ns) for p in (out / "paths").iterdir())
    bootstrap, _, manifest = S01.load_paths(tmp_path / "unused", paths_dir=out / "paths")
    assert bootstrap.n_paths == 4 and manifest["n_paths"] == 4
    rc = S02.main(["--out-dir", str(tmp_path / "run"), "--paths-dir", str(out / "paths"), "--provider", "exact",
                   "--cells", f"{C.MODELS[0]}:front", "--exact-paths", "0", "--oracle-paths", "0",
                   "--quad-grid", "101", "--maturity-months", "1", "--lockout-months", "1"])
    assert rc == 0 and (tmp_path / "run" / "cells" / C.BASELINE_CELL / "run.json").exists()
    assert not (tmp_path / "run" / "paths").exists()
    assert sorted((p.name, p.stat().st_mtime_ns) for p in (out / "paths").iterdir()) == before
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q -k paths_dir`
Expected: FAIL with `TypeError: load_paths() got an unexpected keyword argument 'paths_dir'`.

- [ ] **Step 3: Implement**

In `01_build_paths.py`:

```python
def load_paths(out_dir, *, paths_dir=None) -> Tuple[MarketPath, MarketPath, Dict[str, Any]]:
    """The batch under ``paths_dir``, or ``<out_dir>/paths`` when it is not given."""
    out = Path(paths_dir) if paths_dir is not None else Path(out_dir) / "paths"
    for name in ("bootstrap.npz", "stress.npz", "manifest.json"):
        if not (out / name).exists():
            raise C.Q.StudyDataError(f"missing {out / name}; run 01_build_paths.py first")
    return (MarketPath.from_npz(out / "bootstrap.npz"), MarketPath.from_npz(out / "stress.npz"),
            C.read_json(out / "manifest.json"))
```

In `02_ensemble_fleet.py` `parse_args`:

```python
    parser.add_argument("--paths-dir", type=Path, default=None,
                        help="the batch to run (bootstrap.npz, stress.npz, manifest.json); default <out-dir>/paths")
```

and in `main`: `bootstrap, stress, paths_manifest = S01.load_paths(args.out_dir, paths_dir=args.paths_dir)`.

- [ ] **Step 4: Run the study tests**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(example): a simulated-path fleet reads its batch from --paths-dir

A new output directory can run the banked 2026-09-09 batch without copying it
and without touching the banked cells.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/01_build_paths.py example/snowball_simulated_paths/02_ensemble_fleet.py test/test_snowball_simulated_paths_study.py
```

---

### Task 8: `03_report.py`: provider labels, day-0 marks, provider-neutral caveat

**Files:**
- Modify: `example/snowball_simulated_paths/03_report.py` (`aggregate`, `_engine_section`, `_caveats_section`, `build_report`; new `_provider_of`, `_day0_section`)
- Test: `test/test_snowball_simulated_paths_study.py`

**Interfaces:**
- Produces: `agg["cells"][name]` gains `provider: str`, `day0_book_mark_bp: float`; `agg["engine_check"][i]` gains `pair: str` (`"<cell provider> minus <check provider>"`).

- [ ] **Step 1: Write the failing assertions** (append to `test_aggregate_reduces_every_cell_and_pairs_them`)

```python
    assert agg["cells"][C.BASELINE_CELL]["provider"] == "life_surface"
    assert np.isfinite(agg["cells"][C.BASELINE_CELL]["day0_book_mark_bp"])
    assert check["term_flat_q__front"]["pair"] == "life_surface minus ladder"
```

and to `test_tables_and_report_are_written`:

```python
    assert "Day-0 book marks" in html and "life_surface minus ladder" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q -k "aggregate_reduces or tables_and_report"`
Expected: FAIL with `KeyError: 'provider'`.

- [ ] **Step 3: Implement**

Add after `_portable`:

```python
def _provider_of(out_dir, name: str) -> str:
    """The provider a run was configured with, from its ``config.json``."""
    path = Path(out_dir) / "cells" / name / "config.json"
    if not path.exists():
        raise ValidationError(f"{path} is missing; every persisted run writes its config")
    return str(C.read_json(path)["metadata"]["provider"])
```

In `aggregate`, extend each `agg["cells"]` entry with:

```python
                         "provider": _provider_of(out_dir, name),
                         "day0_book_mark_bp": float(r.cube.product_mtm[0, 0]) / r.notional * 1e4,
```

and replace the engine-check loop body with:

```python
    for name, r in cells.items():
        for suffix in ("__ladder_quad", "__exact_quad"):
            cell = name[: -len(suffix)]
            if name.endswith(suffix) and cell in bootstrap:
                same_paths = bootstrap[cell].take(range(r.n_paths))
                agg["engine_check"].append({
                    "cell": cell, "check": suffix[2:], "n": r.n_paths,
                    "pair": f"{_provider_of(out_dir, cell)} minus {_provider_of(out_dir, name)}",
                    "measures": _paired(same_paths, r),
                })
```

In `write_tables`, add `"pair": e["pair"],` to each `check_rows` dict.

Replace `_engine_section`:

```python
def _engine_section(agg: Dict[str, Any]) -> str:
    parts = ["<h2>Engine check: each cell against QUAD on the same paths</h2>",
             "<p>A cell's first paths paired against the same paths repriced on QUAD: the cell's provider minus "
             "the check's, per measure. Reported, not gated: exact PDE and exact QUAD are both exact engines "
             "with different numerics, so no threshold is claimed.</p>"]
    headers = ["cell", "pair", "n", "measure", "mean difference", "median", "std"]
    rows = [[html.escape(e["cell"]), html.escape(e["pair"]), _fmt(e["n"]), html.escape(m), _fmt(d.get("mean"), 2),
             _fmt(d.get("median"), 2), _fmt(d.get("std"), 2)]
            for e in agg["engine_check"] for m, d in e["measures"].items()]
    parts.append(_table(headers, rows) if rows else "<p>No QUAD check run persisted.</p>")
    return "\n".join(parts)
```

Add:

```python
def _day0_section(agg: Dict[str, Any]) -> str:
    headers = ["cell", "provider", "day-0 book mark bp"]
    rows = [[html.escape(name), html.escape(entry["provider"]), _fmt(entry["day0_book_mark_bp"], 2)]
            for name, entry in agg["cells"].items()]
    return ("<h2>Day-0 book marks</h2><p>Every cell books the traded price (0) at inception and the fair coupon "
            "is solved on QUAD, so a cell's day-0 mark is its provider's price of the traded contract: the "
            "engine gap at inception, carried in every terminal P&amp;L of that cell.</p>" + _table(headers, rows))
```

In `build_report`, insert `_day0_section(agg),` after `_setup_section(agg),`.

In `_caveats_section`, replace the second `<li>` with:

```python
            "<li>Under the life surface and the spot ladder the term dividend models enter only through the scalar "
            "yield at the remaining maturity, flat at the bucket centre; under per_date and exact every state "
            "hands the engine that date's term dividend object.</li>"
```

- [ ] **Step 4: Run the study tests**

Run: `.venv/bin/python -m pytest test/test_snowball_simulated_paths_study.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(example): the simulated-path report names each run's provider and day-0 mark

The engine check pairs a cell's provider against QUAD on the same paths and
says which pair it is; a day-0 table shows each cell's mark of the traded
contract; the carry caveat now covers per_date and exact.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/03_report.py test/test_snowball_simulated_paths_study.py
```

---

### Task 9: README rewrite for Design B

**Files:**
- Modify: `example/snowball_simulated_paths/README.md`

**Interfaces:** documentation only.

- [ ] **Step 1: Replace the section "## What a cell measures under each provider"** (from that heading up to, not including, "## Bucket steps") with:

```markdown
## How a cell is priced

Every cell is priced by one provider, `per_date`: exact repricing on the PDE
engine.  Each state is one backward solve from maturity to that state's date,
on the tracker's aged contract, with that date's own term dividend object.
Nothing is collapsed to a scalar yield, so the carry models under test reach
the engine intact.  One provider for all six cells is deliberate: a flat-arm
provider different from the term-arm provider would make the provider choice
move with the treatment the study measures.

**Why not the whole-life surface.**  The surface solved one PDE over the
product's life per `(vol, q, rate)` bucket and read a column per day.  A term
curve has to be flattened to `q_T` to fit that key, and on 2026-09-14 that
flattening was 107.34 of the 108.73 bp worst gate gap on `term_flat_q`.  A key
that holds the curve does not rescue it: the surface's reuse comes from a flat
market being the same market on every day, while a term curve is
re-snapshotted each day relative to that day.  Counted on the 2,000 × 275
batch, the flat key shares 22.7 states per solve and a (day, vol, rate, carry)
key 1.05.  So the per-date solve costs about what exact repricing costs, and it
is exact.

**One solve per state.**  Exact repricing used to call `price()` and then
`calculate_greeks()`, and each ran the full solve.  Phase 0
(`phase0_single_solve.py`) compared removing the second solve with a solver
memo or in the provider; the decision and its evidence are in the design note
`docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md`.

The mesh is pinned (1601 points, 16 steps a day, step cap 8000, bounds
0.40–1.60 of the initial spot); the day-zero convergence evidence for it is
in "Earlier provider" below.

**Engine check.**  Each cell's first 40 bootstrap paths are also repriced on
exact QUAD and paired with the cell on the same paths.  It is reported, not
gated.  The fair coupon is solved on QUAD, so each cell's day-0 mark is the
PDE-vs-QUAD gap at inception (about 0.5 bp of notional measured on day 1),
shown in the report's day-0 table.
```

- [ ] **Step 2: Turn "Gates and checks" into "Checks" and move the surface material into a dated history section**

The README currently runs "## Bucket steps" → "## Gates and checks" → "## Running it". Make it "## Checks" → "## Earlier provider: the whole-life surface (2026-09-09 to 2026-09-15)" → "## Running it":

1. Rename "## Gates and checks" to "## Checks". Delete its first three bullets (the one beginning "The surface gate:", the one beginning "The oracle spot check:", the one beginning "The engine check:") and put these in their place:

```markdown
- `per_date` is exact: its gate report is zero by construction.
- The oracle spot check: 3 single paths per bootstrap run through the replay
  engine at zero tolerance, and the first of them on each exact-QUAD check.
- The engine check: exact QUAD on each cell's first 40 paths, paired with the
  cell (reported, not gated).
```

2. Cut the four remaining bullets of that section (beginning "A near-barrier readout defect", "The mesh, since pinned", "What the mesh costs", "Known cost:") and the whole "## Bucket steps" section (heading and its two paragraphs).

3. Directly after "## Checks", insert a new section consisting of this heading and paragraph, then a "### Bucket steps" sub-heading with the two cut "Bucket steps" paragraphs, then a "### Surface checks" sub-heading with the four cut bullets, all text unchanged:

```markdown
## Earlier provider: the whole-life surface (2026-09-09 to 2026-09-15)

The study's first fleet provider, kept in the library for flat-carry books.
It ran the flat arm within its 25 bp gate after the readout fix and the
pinned mesh below, and could not run the term arms (see "How a cell is
priced").  The measurements that shaped it stay here because the mesh they
pinned is the one `per_date` uses.
```

- [ ] **Step 3: Replace "## Running it"** (heading through the paragraph ending "the engine receives the term object on every call.") with:

```markdown
## Running it

```bash
# stage 1: the banked 40-path batch of 2026-09-09, all six cells, in its own directory
.venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --paths-dir output/snowball_simulated_paths/paths --workers 6 --batch-paths 7 --resume
.venv/bin/python example/snowball_simulated_paths/03_report.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --data-dir output/snowball_simulated_paths/per_date_40/data

# stage 2: 2,000 paths from the same history cut
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py \
    --history-end 2026-09-09 --out-dir output/snowball_simulated_paths/per_date_2000
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_2000 \
    --workers 12 --batch-paths 170 --disk-cache --resume \
    > output/snowball_simulated_paths/per_date_2000/fleet.log 2>&1 &
.venv/bin/python example/snowball_simulated_paths/03_report.py \
    --out-dir output/snowball_simulated_paths/per_date_2000
```

Paths, cells (`cells/<cell>[__stress|__exact_quad|__ladder_quad]/`, each with
`config.json` and `run.json`), `coupon.json` and `fleet_manifest.json` go to
the `--out-dir`; `--paths-dir` reads a batch from elsewhere without copying.
`--resume` skips a run whose `config.json` fingerprint matches, a recorded
failure included, and a run interrupted in its oracle reuses its results.
`--history-end` cuts the history at a day: at 2026-09-09 it reproduces the
banked batch's history, bootstrap and stress fingerprints from the longer
cache.  `--batch-paths` must be set for `--workers` to take effect; each run
is batched for its own path count.  `--provider exact|life_surface|ladder`
remain available.
```

- [ ] **Step 4: Mark the old results as pre-Design-B**

Directly under "## Results", replace the bold sentence "**The 2,000-path fleet has not been run yet; its numbers will replace this section.**" with:

```markdown
**Recorded before Design B, on exact QUAD, before the ageing fixes of
2026-09-15 (`8cb0ec67`, `f8987344`).  Stage 1 and stage 2 replace this
section.**
```

In "## Caveats", replace the bullet beginning "Under the surface and the ladder" with:

```markdown
- The fair coupon is solved on QUAD while cells price on the PDE, so each
  cell's day-0 mark is the engine gap at inception (report, day-0 table).
```

- [ ] **Step 5: Check links and paths**

Run: `grep -n "/Users/\|/private/" example/snowball_simulated_paths/README.md; ls docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md`
Expected: no grep output; the spec file exists.

- [ ] **Step 6: Commit**

```bash
git commit -m "docs(example): the simulated-path README follows Design B

How a cell is priced (per_date, one provider, why not the whole-life surface,
the 1.05 states-per-solve count), the surface material kept as a dated history
section because its mesh is the one per_date uses, stage 1/2 commands, and the
2026-09-09 results marked as pre-Design-B.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/README.md
```

---

### Task 10: Full regression

**Files:** none modified (unless a failure is traced to this plan's changes).

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q 2>&1 | tail -15`
Expected: `7830+ passed` with zero failures (the branch was clean at 7830 passed, 44 skipped on 2026-09-15 before this plan).

If anything fails, decide whether this plan caused it before touching it. Create a native worktree (the `EnterWorktree` tool — never `git worktree add`), run `git checkout --detach f283d459` inside it and verify `git rev-parse HEAD`, then run the failing test there against that source: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <test node id> -n0`. A test that also fails at `f283d459` is not this plan's; report it and leave it. Fix only failures that pass at `f283d459`.

- [ ] **Step 2: Restore churned sample data**

Run: `git checkout -- example/mo_volmodels/data/ && git status --short`
Expected: only `?? example/standalone_rho/` and the peer's `M docs/superpowers/plans/2026-09-15-intraday-pricing-design.md` (if still present).

---

### Task 11: Stage 1 — the banked 40 paths, six cells; go/no-go

**Files:** outputs only under `output/snowball_simulated_paths/per_date_40/` and `per_date_scaling/`.

**Interfaces:**
- Consumes: Tasks 5–8 CLI; the banked batch `output/snowball_simulated_paths/paths/`.
- Produces: measured seconds per state, worker scaling, bridge table; a go/no-go message to the user.

- [ ] **Step 1: Check the machine**

Run: `ps -axo pid,pgid,pcpu,etime,command | grep "[s]pawn_main\|[r]un_fleet" | head; sysctl -n hw.ncpu; uptime`
Expected: note how many cores other jobs hold. Stage 1 uses 6 workers; if more than 8 cores are busy, wait and re-check rather than start.

- [ ] **Step 2: Worker-scaling probe (back-to-back)**

Create the 6-path probe batch and run it at 1 worker, then at 6:

```bash
.venv/bin/python - <<'EOF'
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "example/snowball_simulated_paths")
import _sim_common as C
from quantark.backtest.simulation import MarketPath
src = C.DEFAULT_OUT_DIR / "paths"
dst = C.DEFAULT_OUT_DIR / "per_date_scaling" / "paths"
dst.mkdir(parents=True, exist_ok=True)
MarketPath.from_npz(src / "bootstrap.npz").take(range(6)).to_npz(dst / "bootstrap.npz")
shutil.copy(src / "stress.npz", dst / "stress.npz")
manifest = json.loads((src / "manifest.json").read_text()); manifest["n_paths"] = 6
(dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
EOF
for W in 1 6; do
  .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
      --out-dir output/snowball_simulated_paths/per_date_scaling/w$W \
      --paths-dir output/snowball_simulated_paths/per_date_scaling/paths \
      --cells term_flat_q:front --exact-paths 0 --oracle-paths 0 --workers $W --batch-paths 1 \
      2>&1 | grep -v -i warn | tail -3
done
.venv/bin/python - <<'EOF'
import json
import numpy as np
from quantark.backtest.simulation.results import FLOAT_COLUMNS, EnsembleResults
root = "output/snowball_simulated_paths/per_date_scaling"
w1 = json.load(open(f"{root}/w1/cells/term_flat_q__front/run.json"))
w6 = json.load(open(f"{root}/w6/cells/term_flat_q__front/run.json"))
print(f"states {w1['engine_calls']}; 1 worker {w1['seconds']:.0f}s ({w1['seconds']/w1['engine_calls']:.3f} s/state); "
      f"6 workers {w6['seconds']:.0f}s; speedup {w1['seconds']/w6['seconds']:.2f}x of 6")
a = EnsembleResults.from_dir(f"{root}/w1/cells/term_flat_q__front")
b = EnsembleResults.from_dir(f"{root}/w6/cells/term_flat_q__front")
print("bitwise across batching:", all(np.array_equal(getattr(a.cube, n), getattr(b.cube, n), equal_nan=True) for n in FLOAT_COLUMNS))
EOF
```

Expected: a speedup line and `bitwise across batching: True`. `--batch-paths 1` with 6 paths gives 6 batches. If the bitwise line is `False`, stop and report: batching was asserted bit-inert in plan 3.

- [ ] **Step 3: Launch stage 1**

```bash
mkdir -p output/snowball_simulated_paths/per_date_40
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --paths-dir output/snowball_simulated_paths/paths \
    --workers 6 --batch-paths 7 --resume \
    > output/snowball_simulated_paths/per_date_40/fleet.log 2>&1 &
echo $! > output/snowball_simulated_paths/per_date_40/fleet.pid
```

Expected: the first log line `fair coupon 37.8254% under term_flat_q ...; 6 cells x 40 paths, provider per_date, workers 6`. If the coupon differs from 37.8254%, stop the run (kill its process group: `kill -- -$(ps -o pgid= -p $(cat .../fleet.pid) | tr -d ' ')`) and report to the user before continuing.

- [ ] **Step 4: Wait for completion**

Monitor `output/snowball_simulated_paths/per_date_40/fleet.log` (one line per run; 18 runs) until `fleet_manifest.json` exists. Expect about 2 hours. If the process dies (no PID, no new line, no manifest), relaunch the Step 3 command unchanged (`--resume` skips banked runs).

- [ ] **Step 5: Bridge and summary**

```bash
.venv/bin/python example/snowball_simulated_paths/03_report.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --data-dir output/snowball_simulated_paths/per_date_40/data 2>&1 | tail -2
.venv/bin/python - <<'EOF'
import json, sys
sys.path.insert(0, "example/snowball_simulated_paths")
import _sim_common as C
from quantark.backtest.simulation.results import EnsembleResults
base, new = C.DEFAULT_OUT_DIR, C.DEFAULT_OUT_DIR / "per_date_40"
fleet = json.load(open(new / "fleet_manifest.json"))
print("coupon", fleet["coupon"]["coupon"])
total_states = total_seconds = 0
for name, run in fleet["runs"].items():
    ok = all(r["passed"] for r in run["oracle"]) if run["oracle"] else None
    print(f"{name:36s} failed={run['failed']} seconds={run['seconds']:.0f} calls={run.get('engine_calls')} "
          f"oracle={ok} day0={run.get('day0_book_mark_bp')}")
    if not name.endswith(("__stress", "__exact_quad", "__ladder_quad")):
        total_states += run["engine_calls"]; total_seconds += run["seconds"]
print(f"bootstrap runs: {total_states} states in {total_seconds:.0f}s wall at 6 workers")
for cell in ("flat_from_hedge__front", "term_flat_q__front"):
    comparison = EnsembleResults.from_dir(new / "cells" / cell).paired(EnsembleResults.from_dir(base / "cells" / cell))
    for measure in ("terminal_pnl_bp", "daily_pnl_std_bp", "variance_reduction_r2", "delta_churn"):
        d = comparison.describe(measure)
        print(f"bridge {cell} per_date(2026-09-15) - exactQUAD(2026-09-09) {measure}: mean {d['mean']:.3f} median {d['median']:.3f}")
EOF
```

Expected: every run `failed=False`, every oracle `True`, the coupon `0.378254...`, day-0 marks within a few bp, and the bridge lines.

- [ ] **Step 6: Go/no-go message to the user and stop**

Report, in one message: the coupon check; per-cell seconds and states; measured seconds per state (bootstrap runs) and the scaling speedup; the engine-check table (`output/snowball_simulated_paths/per_date_40/data/engine_check.csv`, terminal P&L and daily std rows); the bridge lines; any oracle failure. Give the stage-2 projection computed from the measurements:

`stage-2 wall hours ≈ (stage-1 bootstrap seconds_per_state × 50 × stage-1 bootstrap states per cell × 6) / (12 × speedup_per_worker) / 3600`, where `speedup_per_worker = speedup / 6` from Step 2, plus the six 40-path QUAD checks and the oracles at stage-1 wall times.

Wait for the user's go before Task 12.

---

### Task 12: Stage 2 — 2,000 paths, six cells

**Files:** outputs only under `output/snowball_simulated_paths/per_date_2000/`.

- [ ] **Step 1: Confirm the machine is free**

Run: `ps -axo pid,pgid,pcpu,etime,command | grep "[s]pawn_main\|[r]un_fleet" | head`
Expected: no other fleet's workers (the peer's `run_fleet_v3.py` gone). If present, wait; do not share 12 workers with another fleet.

- [ ] **Step 2: Build the batch**

Run: `.venv/bin/python example/snowball_simulated_paths/01_build_paths.py --history-end 2026-09-09 --out-dir output/snowball_simulated_paths/per_date_2000 2>&1 | grep -v -i warn | tail -2`
Expected: `history 2023-05-04..2026-09-09 (816 days), 2000 bootstrap paths x 275 days from 2026-09-10, 5 stress paths, ... history cut at 2026-09-09`. Check `paths/manifest.json` `history_fingerprint` equals the banked `77a5a0df5a7d2c6abfd9102d3dc650f775db358b46203e92d6f2b5626abdc707` and `stress_fingerprint` equals `7ca2c20777c398a957c28c297230a205982028b805dde60778ea6112d5073525`.

- [ ] **Step 3: Launch**

```bash
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_2000 \
    --workers 12 --batch-paths 170 --disk-cache --resume \
    > output/snowball_simulated_paths/per_date_2000/fleet.log 2>&1 &
echo $! > output/snowball_simulated_paths/per_date_2000/fleet.pid
```

Expected: first line `fair coupon 37.8254% ...; 6 cells x 2000 paths, provider per_date, workers 12`.

- [ ] **Step 4: Monitor to completion**

Check every 30–60 minutes: the PID is alive (`ps -p $(cat .../fleet.pid)`), CPU across workers is about 12 cores (`ps -axo pcpu,command | grep "[s]pawn_main" | awk '{s+=$1} END {print s}'`), and the log gains a line per finished run. If the process died, check `pmset -g log | grep -i sleep | tail` for sleep, then relaunch the Step 3 command unchanged. Done when `fleet_manifest.json` exists and the log has 18 run lines.

- [ ] **Step 5: Verify**

Run the Task 11 Step 5 summary script with `new = C.DEFAULT_OUT_DIR / "per_date_2000"` and without the bridge loop (the batch differs from the banked one).
Expected: every run `failed=False`, every oracle `True`.

---

### Task 13: Report, README results, commit

**Files:**
- Modify: `example/snowball_simulated_paths/data/*` (regenerated tables and report)
- Modify: `example/snowball_simulated_paths/README.md` ("## Results")

- [ ] **Step 1: Generate the report into the tracked data directory**

Run: `.venv/bin/python example/snowball_simulated_paths/03_report.py --out-dir output/snowball_simulated_paths/per_date_2000 2>&1 | tail -2`
Expected: `6 cells, N paired comparisons, 30 stress rows, historical located; report at .../data/simulated_paths_report.html`.

- [ ] **Step 2: Check for machine paths**

Run: `grep -rln "/Users/\|/private/" example/snowball_simulated_paths/data/ || echo clean`
Expected: `clean`.

- [ ] **Step 3: Rewrite "## Results"**

Replace everything from "## Results" up to "## Caveats" with a section built from `data/fleet_cells.json`, `data/fleet_paired.csv`, `data/stress_table.csv`, `data/engine_check.csv`, `data/historical_location.csv` and `fleet_manifest.json`, in this order, every number read from those files:
1. One paragraph: date run, batch (2,000 × 275 from the 2026-09-09 cut, fingerprints), coupon, provider `per_date` for all six cells, the Phase 0 decision, wall time and workers.
2. Run table: run, seconds, engine calls, oracle (one row per run, 18 rows).
3. Distributions table for the six cells: terminal P&L, daily P&L std, R², cost, roll-day jump, delta churn (mean, q05, q50, q95, ES, share > 0), with KO/KI/maturity frequencies.
4. Paired tables: each term model minus the baseline (front), and far minus front per model (mean, median, share > 0, t).
5. Stress table.
6. Engine check: per cell, `per_date minus exact` on 40 paths for terminal P&L and daily std; the day-0 marks.
7. Historical location paragraph.
8. The stage-1 bridge (from Task 11 Step 5): per_date on the banked 40 against the 2026-09-09 exact-QUAD cells.

Keep the two paragraphs of prose interpretation tied to the tables' numbers; no claim without a table row behind it.

- [ ] **Step 4: Commit by pathspec**

```bash
git checkout -- example/mo_volmodels/data/
git status --short example/snowball_simulated_paths/
git commit -m "example: simulated-path snowball study, 2,000 paths x 6 cells on the per-date PDE solve

Stage 2 of Design B: every cell exact per date on the pinned PDE mesh, the
term dividend object handed to the engine on every state, exact-QUAD checks
on 40 paths per cell, oracles passing. Tables and report regenerated; README
results replaced.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" -- example/snowball_simulated_paths/data example/snowball_simulated_paths/README.md
```

Expected: the commit lists only files under `example/snowball_simulated_paths/`.
