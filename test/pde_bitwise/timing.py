"""The PDE timing gate (spec 2 §9.5): cold and warm price medians and peak memory per family.

Run the same file against two source trees back to back, in one window:

    PYTHONPATH=<base tree> python test/pde_bitwise/timing.py --out base.json --reps 5
    PYTHONPATH=<branch tree> python test/pde_bitwise/timing.py --out branch.json --reps 5
    python test/pde_bitwise/timing.py --compare base.json branch.json

cold = a new solver and ``price`` after clearing the module-level market memos;
warm = the same solver, a second ``price``. One warm-up, then ``--reps`` repetitions;
medians are written. Peak memory is ``ru_maxrss`` of a subprocess that prices one family.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_book as cb  # noqa: E402

FAMILIES = ("european", "american", "barrier", "snowball", "phoenix", "ko_reset", "lv_snowball",
            "statistics", "statistics_phoenix", "statistics_phoenix_memory",
            "statistics_phoenix_expiry", "statistics_ko_reset", "statistics_lv")
# memory and EXPIRY Phoenix statistics have no 0.8.0 baseline worth gating (spec 3B §7.3)
UNGATED = ("statistics_phoenix_memory", "statistics_phoenix_expiry")
# the four routes spec 3C moved onto the core: declared at each preset, price and calculate_greeks, no gate (§7.3)
MIGRATED = ("lv_vanilla", "lv_barrier_cont", "lv_barrier_daily", "dcn", "lv_dcn")
DECLARED = tuple(f"{route}|{accuracy}|{op}" for route in MIGRATED for accuracy in ("fast", "standard", "high")
                 for op in ("price", "greeks"))


def _migrated_case(name):
    """A declared case ``route|accuracy|op``: the same call on both trees (every one takes PDEParams(accuracy))."""
    from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn

    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
    from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine
    from quantark.asset.equity.engine.pde.local_vol_pde_solver import LocalVolPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType, OptionType

    route, accuracy, op = name.split("|")
    daily = BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=130.0,
                          barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.DISCRETE,
                          observation_dates=[k / 252.0 for k in range(1, 253)])
    table = {
        "lv_vanilla": (LocalVolPDESolver, cb.european(), cb.env_grid_vol()),
        "lv_barrier_cont": (LocalVolBarrierPDESolver, cb.barrier(ObservationType.CONTINUOUS), cb.env_grid_vol()),
        "lv_barrier_daily": (LocalVolBarrierPDESolver, daily, cb.env_grid_vol()),
        "dcn": (DCNPDEEngine, make_dcn(DCN_A), flat_env(**FLAT, spot=DCN_A["initial_price"])),
        "lv_dcn": (LocalVolDCNPDEEngine, make_dcn(DCN_A), cb.env_grid_vol_dcn()),
    }
    cls, product, env = table[route]
    call = ((lambda s: s.price(product, env)) if op == "price"
            else (lambda s: s.calculate_greeks(product, env)))
    return (lambda: cls(params=PDEParams(accuracy=accuracy))), call


def _case(family):
    """(solver factory, call(solver)) for a family's book-standard flat product."""
    from quantark.asset.equity.engine import pde
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.util.enum import ObservationType

    if family in DECLARED:
        return _migrated_case(family)
    flat = cb.env_flat()
    table = {
        "european": (pde.EuropeanPDESolver, cb.european(), flat),
        "american": (pde.AmericanPDESolver, cb.american_tenor(), flat),
        "barrier": (pde.BarrierPDESolver, cb.barrier(ObservationType.CONTINUOUS), flat),
        "snowball": (pde.SnowballPDESolver, cb.snowball(), flat),
        "phoenix": (pde.PhoenixPDESolver, cb.make_phoenix(cb.PHOENIX), flat),
        "ko_reset": (pde.KOResetSnowballPDESolver, cb.make_ko_reset(cb.KO_RESET), flat),
        "lv_snowball": (LocalVolSnowballPDESolver, cb.snowball(), cb.env_grid_vol()),
        "statistics": (pde.SnowballPDESolver, cb.snowball(), flat),
        "statistics_phoenix": (pde.PhoenixPDESolver,
                               cb.make_phoenix(dict(cb.PHOENIX, memory_coupon=False)), flat),
        "statistics_phoenix_memory": (pde.PhoenixPDESolver, cb.make_phoenix(cb.PHOENIX), flat),
        "statistics_phoenix_expiry": (pde.PhoenixPDESolver,
                                      cb.make_phoenix(dict(cb.PHOENIX, coupon_pay_type="expiry")),
                                      flat),
        "statistics_ko_reset": (pde.KOResetSnowballPDESolver, cb.make_ko_reset(cb.KO_RESET), flat),
        "statistics_lv": (LocalVolSnowballPDESolver, cb.snowball(), cb.env_grid_vol()),
    }
    cls, product, env = table[family]
    if family.startswith("statistics"):
        return (lambda: cls(params=PDEParams())), (lambda s: s.calculate_event_stats(product, env))
    return (lambda: cls(params=PDEParams())), (lambda s: s.price(product, env))


def _clear_memos():
    from quantark.asset.equity.engine.pde import base_pde_solver

    # both module memos exist on 0.6.0 and 0.7.0: a cold solve starts with neither
    for name in ("_ENV_STEP_COEFF_MEMO", "_ENV_DF_MEMO"):
        memo = getattr(base_pde_solver, name, None)
        if memo is not None:
            memo.clear()


def _once(family):
    make, call = _case(family)
    _clear_memos()
    t0 = time.perf_counter()
    solver = make()
    call(solver)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    call(solver)
    return cold, time.perf_counter() - t0


def _peak_rss(family) -> int:
    """ru_maxrss (bytes on macOS, KiB on Linux) of a fresh process pricing one family."""
    out = subprocess.run([sys.executable, __file__, "--memory-family", family],
                         capture_output=True, text=True, env=os.environ.copy(), check=True)
    return int(out.stdout.strip().splitlines()[-1])


def measure(reps: int, families=FAMILIES) -> dict:
    result = {}
    for family in families:
        _once(family)                                    # warm-up
        cold, warm = zip(*(_once(family) for _ in range(reps)))
        result[family] = {"cold": statistics.median(cold), "warm": statistics.median(warm),
                          "peak_rss": _peak_rss(family)}
    return result


def compare(base_path, branch_path, limit=0.05) -> int:
    base, branch = (json.loads(Path(p).read_text()) for p in (base_path, branch_path))
    worst = 0
    for family in base:
        b, n = base[family], branch[family]
        row = []
        for key in ("cold", "warm", "peak_rss"):
            ratio = n[key] / b[key]
            row.append(f"{key} {ratio:5.3f}")
            if ratio > 1.0 + limit and family not in UNGATED and family not in DECLARED:
                worst = 1
        print(f"{family:26s} " + "  ".join(row) + ("  (declared)" if family in DECLARED else ""))
    print("PASS" if worst == 0 else f"REGRESSION (> {limit:.0%})")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--memory-family")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--limit", type=float, default=0.05, help="the gate's allowed slowdown (3C: 1.05x)")
    # a subset keeps one family's allocations from disturbing the next (the 3B statistics gate)
    ap.add_argument("--families", nargs="+", choices=FAMILIES + DECLARED, default=FAMILIES + DECLARED)
    args = ap.parse_args(argv)
    if args.memory_family:
        _once(args.memory_family)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return 0
    if args.compare:
        return compare(*args.compare, limit=args.limit)
    Path(args.out).write_text(json.dumps(measure(args.reps, args.families), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
