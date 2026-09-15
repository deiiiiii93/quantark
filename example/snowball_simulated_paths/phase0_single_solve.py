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
              It also renders a dataclass's instance attributes outside
              its declared fields: the tracker ``setattr``s that flag onto
              a dataclass product, and a fields-only rendering (the plan's
              first key) cannot tell an alive state from a knocked-in one.
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
import datetime as _dt
import enum
import importlib.util
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def planned_key(obj: Any) -> str:
    """The key the plan first specified: ``repricing._canonical`` with lifecycle attributes not skipped.

    Kept only for the probe that shows why it is unsafe: ``_canonical``
    renders a dataclass by its declared FIELDS, and the lifecycle tracker
    ``setattr``s ``_otc_lifecycle_knocked_in`` onto the dataclass product,
    so an alive and a knocked-in product render alike.
    """
    saved = repricing._STATE_PREFIXES
    repricing._STATE_PREFIXES = ()
    try:
        return repricing._canonical(obj, set())
    finally:
        repricing._STATE_PREFIXES = saved


def structural_key(obj: Any, seen: Optional[set] = None, depth: int = 0) -> str:
    """``repricing._canonical`` for a solver key: nothing skipped, and a dataclass's
    instance attributes outside its declared fields rendered after the fields."""
    seen = set() if seen is None else seen
    if depth > 12:
        raise ValidationError("object nests too deeply to key")
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
        return f"{type(obj).__name__}[{','.join(structural_key(x, seen, depth + 1) for x in obj)}]"
    if isinstance(obj, (set, frozenset)):
        return f"{type(obj).__name__}{{{','.join(sorted(structural_key(x, seen, depth + 1) for x in obj))}}}"
    if isinstance(obj, dict):
        inner = ",".join(f"{structural_key(k, seen, depth + 1)}:{structural_key(v, seen, depth + 1)}"
                         for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0])))
        return f"dict{{{inner}}}"
    if callable(obj) and not hasattr(obj, "__dict__"):
        return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"
    marker = id(obj)
    if marker in seen:
        return f"<cycle:{type(obj).__name__}>"
    seen.add(marker)
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
        extra = sorted(set(getattr(obj, "__dict__", {})) - set(names))
        items = [(name, getattr(obj, name)) for name in names + extra]
    elif hasattr(obj, "__dict__"):
        items = sorted(vars(obj).items())
    else:
        return f"{type(obj).__name__}:{obj!r}"
    inner = ",".join(f"{name}={structural_key(value, seen, depth + 1)}"
                     for name, value in items if not callable(value))
    seen.discard(marker)
    return f"{type(obj).__name__}({inner})"


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
        # False on 2026-09-15: the plan's first key rendered dataclass fields
        # only, so the knock-in flag (set with setattr) never reached it.
        "planned_fields_only_key_sees_knock_in": planned_key(alive) != planned_key(ki),
        "product_attrs_outside_dataclass_fields": sorted(set(vars(alive)) - {f.name for f in dataclasses.fields(alive)}),
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
        ("memo miss probes", "–", f"{'all ok' if probe_ok else 'FAIL'}; alive/KI differ only in lifecycle attrs: {probes['alive_vs_ki_differ_only_in_lifecycle_attrs']}; "
         f"fields-only key sees KI: {probes['planned_fields_only_key_sees_knock_in']} "
         f"(attrs outside dataclass fields: {', '.join(probes['product_attrs_outside_dataclass_fields'])})", "–"),
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
