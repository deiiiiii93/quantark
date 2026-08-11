"""Is OSS actually better than the estimator the certification already uses?

Every OSS number measured so far -- 3.67x on delta, 33x on gamma -- is against
`QESnowballMCEngine(method="pseudo")`, i.e. plain Monte Carlo with a central FD
bump. That is NOT what the certification's references run. Stage 16 builds its
Heston reference as

    QESnowballMCEngine(
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        rqmc_affine_spot_factor=True,        # <-- exact spot-factor integration
        rqmc_spot_bridge_strata=...,
        rqmc_spot_bridge_dimensions=...,
    )

and the comment on that flag states its purpose outright: "QE variance is
independent of the residual spot Brownian factor. Integrate that factor exactly
so barrier indicators do not dominate finite-bump delta/gamma uncertainty."

That is conditional smoothing of the spot factor -- the same mechanism OSS uses,
applied to the same noise. So the certification's baseline may already capture
much of what OSS offers, and a gain measured against plain MC cannot be applied
to standard deviations produced by the stronger estimator. Doing so is what my
28.4x allocation projection did.

This measures the three estimators head to head on one fixture at equal paths:

  1. plain pseudo FD               -- the prototype's baseline
  2. certification-style RQMC FD   -- affine spot factor + bridge, what stage 16 runs
  3. OSS FD                        -- the candidate

If (2) is already close to (3), the engine port buys little and should not be
built. If (3) still wins clearly over (2), the port is justified and the
allocation can be re-derived against the correct baseline.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_oss_vs_certification_stack.py
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
DEMOS = Path("/Users/fuxinyao/quant-ark/docs/mc2d-gamma-convergence/demos")
OUTPUT_DIR = ROOT / "output" / "oss_vs_certification"

if not DEMOS.is_dir():
    raise SystemExit(f"mc2d-gamma-convergence demos not found at {DEMOS}")
sys.path.insert(0, str(DEMOS))

from common import (  # noqa: E402
    HESTON,
    SPOT0,
    batch_seeds,
    build_env,
    build_product,
    engine_price,
)
from demo_b_oss import OSSSnowballPricer  # noqa: E402

from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (  # noqa: E402
    QESnowballMCEngine,
)
from quantark.asset.equity.param import MCParams  # noqa: E402
from quantark.util.enum import MonteCarloMethod  # noqa: E402

# Stage-16's per-case HESTON profile is strata=1, dimensions=1: the bridge8
# treatment (dimensions=8) went to the SLV cells, not these. So the Heston
# reference's smoothing is RQMC plus exact spot-factor integration, nothing else.
BRIDGE_STRATA = 1
BRIDGE_DIMENSIONS = 1


def certification_price(spot: float, seed: int, paths: int, batches: int = 1) -> float:
    """Price with stage-16's Heston reference configuration."""
    engine = QESnowballMCEngine(
        HESTON,
        params=MCParams(
            seed=seed,
            num_paths=paths,
            rqmc_min_batches=batches,
            rqmc_max_batches=batches,
            rqmc_target_std=1e-12,
            rqmc_paths_mode="per_batch",
        ),
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        martingale_correction=True,
        rqmc_affine_spot_factor=True,
        rqmc_spot_bridge_strata=BRIDGE_STRATA,
        rqmc_spot_bridge_dimensions=BRIDGE_DIMENSIONS,
    )
    return float(engine.price(build_product(), build_env(spot)))


def measure(*, seeds: Sequence[int], paths: int, bump: float) -> dict:
    h = bump * SPOT0
    rows: dict[str, dict[str, list]] = {
        name: {"delta": [], "gamma": []}
        for name in ("plain", "certification", "oss")
    }
    seconds = {name: 0.0 for name in rows}

    for seed in seeds:
        started = time.perf_counter()
        up = engine_price(SPOT0 + h, seed, paths).price
        mid = engine_price(SPOT0, seed, paths).price
        dn = engine_price(SPOT0 - h, seed, paths).price
        seconds["plain"] += time.perf_counter() - started
        rows["plain"]["delta"].append((up - dn) / (2.0 * h))
        rows["plain"]["gamma"].append((up - 2.0 * mid + dn) / (h * h))

        started = time.perf_counter()
        up = certification_price(SPOT0 + h, seed, paths)
        mid = certification_price(SPOT0, seed, paths)
        dn = certification_price(SPOT0 - h, seed, paths)
        seconds["certification"] += time.perf_counter() - started
        rows["certification"]["delta"].append((up - dn) / (2.0 * h))
        rows["certification"]["gamma"].append((up - 2.0 * mid + dn) / (h * h))

        started = time.perf_counter()
        pricer = OSSSnowballPricer(paths, seed)
        up = pricer.oss_price(SPOT0 + h)
        mid = pricer.oss_price(SPOT0)
        dn = pricer.oss_price(SPOT0 - h)
        seconds["oss"] += time.perf_counter() - started
        rows["oss"]["delta"].append((up - dn) / (2.0 * h))
        rows["oss"]["gamma"].append((up - 2.0 * mid + dn) / (h * h))

    out: dict = {"bump": bump, "paths": paths, "seeds": len(seeds), "seconds": seconds}
    for name, greeks in rows.items():
        out[name] = {}
        for greek, values in greeks.items():
            arr = np.asarray(values, dtype=float)
            out[name][greek] = {
                "mean": float(arr.mean()),
                "stderr": float(arr.std(ddof=1) / np.sqrt(arr.size)),
            }
    for greek in ("delta", "gamma"):
        base = out["certification"][greek]["stderr"]
        for name in ("plain", "oss"):
            other = out[name][greek]["stderr"]
            out[f"{greek}_{name}_variance_ratio_vs_certification"] = float(
                (base / other) ** 2 if other > 0 else float("inf")
            )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--paths", type=int, default=25_000)
    parser.add_argument("--bumps", type=float, nargs="+", default=(0.01, 0.003))
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "oss_vs_certification.json"
    )
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seeds = batch_seeds(999_331, args.seeds)
    payload = {
        "purpose": (
            "OSS vs the estimator stage 16 actually uses (RQMC + exact spot-factor "
            "integration + bridge), not vs plain pseudo MC"
        ),
        "bridge_strata": BRIDGE_STRATA,
        "bridge_dimensions": BRIDGE_DIMENSIONS,
        "results": [],
    }
    for bump in args.bumps:
        row = measure(seeds=seeds, paths=args.paths, bump=float(bump))
        payload["results"].append(row)
        print(
            f"h={bump:<6.3f}  "
            f"delta stderr: plain {row['plain']['delta']['stderr']:.5f}  "
            f"cert {row['certification']['delta']['stderr']:.5f}  "
            f"oss {row['oss']['delta']['stderr']:.5f}   |   "
            f"gamma stderr: plain {row['plain']['gamma']['stderr']:.5f}  "
            f"cert {row['certification']['gamma']['stderr']:.5f}  "
            f"oss {row['oss']['gamma']['stderr']:.5f}",
            flush=True,
        )
        print(
            f"          OSS vs CERT variance ratio: "
            f"delta {row['delta_oss_variance_ratio_vs_certification']:.2f}x  "
            f"gamma {row['gamma_oss_variance_ratio_vs_certification']:.2f}x   "
            f"(plain vs cert: delta "
            f"{row['delta_plain_variance_ratio_vs_certification']:.2f}x  gamma "
            f"{row['gamma_plain_variance_ratio_vs_certification']:.2f}x)",
            flush=True,
        )
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
