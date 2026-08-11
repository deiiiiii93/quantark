"""Does the one-step-survival estimator help DELTA, not just gamma?

The mc2d-gamma-convergence session proved OSS is a decisive gamma fix: 7.4x /
34x / 101x variance at h = 1% / 0.3% / 0.1% on the real snowball fixture. But it
measured delta's advantage only in the two-observation Black-Scholes reduction
(125x), and that toy case is the most favourable possible geometry.

That gap lands squarely on the certification's critical path. Feeding the
measured gamma gains into the stage-16 allocation sizing collapses every gamma
row to the 16-batch floor and leaves DELTA binding -- heston/near_ki at 256,
heston_slv/low_feller at 256, heston_slv/ordinary_decayed at 64. So the number
that sets the certification's cost after an OSS port is OSS's delta advantage,
which nobody has measured on a daily-monitored snowball.

There is reason to doubt it is large. Gamma's win comes from killing the
J^2/h^3 indicator noise in a second difference; delta's plain-FD law is only
J^2/h, far milder. Meanwhile OSS pays a real price on smooth functionals,
because the survival weights carry their own variance: the session's own gate 2
measured `oss_pv_stderr_vs_engine = 1.0166`, i.e. OSS is 1.7% WORSE on PV. Delta
sits between PV and gamma in smoothness, so its sign is genuinely open.

This reuses their prototype verbatim -- same fixture, same pricer, same CRN
convention, same up/mid/down prices their gate 3 already computes -- and simply
records the first difference alongside the second. No engine edits; no edits to
their files either, since that session may still be live.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_oss_delta_gate.py
    ... --seeds 10 --paths 50000        # their gate-3 sizing (~6 min)
    ... --seeds 4 --paths 20000         # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
# Their demos live in the main checkout; import, never modify.
DEMOS = Path("/Users/fuxinyao/quant-ark/docs/mc2d-gamma-convergence/demos")
OUTPUT_DIR = ROOT / "output" / "oss_delta_gate"

if not DEMOS.is_dir():
    raise SystemExit(f"mc2d-gamma-convergence demos not found at {DEMOS}")
sys.path.insert(0, str(DEMOS))

from common import SPOT0, batch_seeds, engine_price  # noqa: E402
from demo_b_oss import OSSSnowballPricer  # noqa: E402

BUMPS = (0.01, 0.003, 0.001)


def measure(*, seeds: Sequence[int], paths: int, bump_rel: float) -> dict:
    """Plain-engine and OSS delta+gamma from the SAME three prices per seed."""
    h = bump_rel * SPOT0
    rows = {"engine": {"delta": [], "gamma": []}, "oss": {"delta": [], "gamma": []}}
    seconds = {"engine": 0.0, "oss": 0.0}

    for seed in seeds:
        started = time.perf_counter()
        up = engine_price(SPOT0 + h, seed, paths).price
        mid = engine_price(SPOT0, seed, paths).price
        down = engine_price(SPOT0 - h, seed, paths).price
        seconds["engine"] += time.perf_counter() - started
        rows["engine"]["delta"].append((up - down) / (2.0 * h))
        rows["engine"]["gamma"].append((up - 2.0 * mid + down) / (h * h))

        started = time.perf_counter()
        pricer = OSSSnowballPricer(paths, seed)
        up_o = pricer.oss_price(SPOT0 + h)
        mid_o = pricer.oss_price(SPOT0)
        down_o = pricer.oss_price(SPOT0 - h)
        seconds["oss"] += time.perf_counter() - started
        rows["oss"]["delta"].append((up_o - down_o) / (2.0 * h))
        rows["oss"]["gamma"].append((up_o - 2.0 * mid_o + down_o) / (h * h))

    result = {"bump": bump_rel, "paths": paths, "seeds": len(seeds), "seconds": seconds}
    for estimator, greeks in rows.items():
        result[estimator] = {}
        for greek, values in greeks.items():
            arr = np.asarray(values, dtype=float)
            result[estimator][greek] = {
                "mean": float(arr.mean()),
                "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size)),
            }
    for greek in ("delta", "gamma"):
        engine_se = result["engine"][greek]["stderr"]
        oss_se = result["oss"][greek]["stderr"]
        # Variance ratio > 1 means OSS wins; < 1 means OSS is worse.
        result[f"{greek}_variance_ratio"] = float(
            (engine_se / oss_se) ** 2 if oss_se > 0 else float("inf")
        )
    return result


def render(results: Sequence[dict]) -> str:
    lines = [
        "",
        "plain-engine FD vs OSS FD, equal paths and seeds, shared prices",
        "variance ratio > 1 means OSS wins; < 1 means OSS is WORSE",
        "",
        f"| {'bump':>6s} | {'delta engine':>14s} | {'delta OSS':>14s} | "
        f"{'d ratio':>8s} | {'gamma engine':>14s} | {'gamma OSS':>14s} | "
        f"{'g ratio':>9s} |",
        "|" + "|".join(["-" * 8, "-" * 16, "-" * 16, "-" * 10, "-" * 16, "-" * 16, "-" * 11]) + "|",
    ]
    for row in results:
        lines.append(
            f"| {row['bump']:>6.3f} "
            f"| {row['engine']['delta']['mean']:+7.4f}±{row['engine']['delta']['stderr']:.4f} "
            f"| {row['oss']['delta']['mean']:+7.4f}±{row['oss']['delta']['stderr']:.4f} "
            f"| {row['delta_variance_ratio']:>8.2f} "
            f"| {row['engine']['gamma']['mean']:+7.4f}±{row['engine']['gamma']['stderr']:.4f} "
            f"| {row['oss']['gamma']['mean']:+7.4f}±{row['oss']['gamma']['stderr']:.4f} "
            f"| {row['gamma_variance_ratio']:>9.2f} |"
        )
    lines += [
        "",
        "Reference: their gate 3 measured gamma ratios 7.4 / 34 / 101 at these bumps,",
        "and delta 125x in the 2-observation Black-Scholes reduction only.",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--bumps", type=float, nargs="+", default=BUMPS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "oss_delta_gate.json")
    args = parser.parse_args(argv)

    seeds = batch_seeds(999_331, args.seeds)
    payload = {
        "purpose": (
            "OSS delta variance advantage on the real snowball fixture; the "
            "quantity that sets certification cost once OSS collapses gamma"
        ),
        "fixture": "mc2d-gamma-convergence common.py (2y snowball, Feller-violated Heston)",
        "seeds": args.seeds,
        "paths_per_seed": args.paths,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for bump in args.bumps:
        row = measure(seeds=seeds, paths=args.paths, bump_rel=float(bump))
        payload["results"].append(row)
        print(
            f"  h={bump:<6.3f} delta ratio {row['delta_variance_ratio']:8.2f}   "
            f"gamma ratio {row['gamma_variance_ratio']:9.2f}",
            flush=True,
        )
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(render(payload["results"]))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
