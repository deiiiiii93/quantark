"""Run the V1 treatment matrix for one cell and append JSONL rows.

Three rows per cell:
  baseline  production profile (bridge dimensions 1) -- the SD/cost anchor
  bridge8   the treatment candidate (8 residual bridge coordinates)
  unbias    bridge8 on an independent seed -- V1-G1 agreement evidence

The summary row reports the SD factor, the SE^2 x seconds efficiency factor
(the number that actually decides shipping), and the agreement sigma.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import measure_row  # noqa: E402

ROWS = (
    ("baseline", 1, 20260810),
    ("bridge8", 8, 20260810),
    ("unbias", 8, 20260811),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--variant", default="heston_slv")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "logs" / f"{args.cell}.jsonl"
    out.parent.mkdir(exist_ok=True)

    rows = {}
    for label, dimensions, seed in ROWS:
        record = measure_row(
            args.cell,
            args.variant,
            label,
            batches=args.batches,
            seed=seed,
            bridge_dimensions=dimensions,
            workers=args.workers,
        )
        rows[label] = record
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    base, treat, unbias = rows["baseline"], rows["bridge8"], rows["unbias"]
    sd_factor = base["batch_sd_contracts"] / max(treat["batch_sd_contracts"], 1e-12)
    efficiency_factor = (
        base["batch_sd_contracts"] ** 2 * base["seconds_per_batch"]
    ) / max(treat["batch_sd_contracts"] ** 2 * treat["seconds_per_batch"], 1e-12)
    combined_se = (
        base["delta_se_contracts"] ** 2 + unbias["delta_se_contracts"] ** 2
    ) ** 0.5
    agreement_contracts = (
        abs(base["delta_mean"] - unbias["delta_mean"]) * base["scale_factor"]
    )
    agreement_sigma = agreement_contracts / max(combined_se, 1e-12)
    summary = {
        "summary": {
            "cell": args.cell,
            "variant": args.variant,
            "batches_per_row": args.batches,
            "sd_factor": round(sd_factor, 2),
            "se2_sec_factor": round(efficiency_factor, 2),
            "agreement_contracts": round(agreement_contracts, 5),
            "unbias_sigma": round(agreement_sigma, 2),
            "v1_g1_pass": bool(agreement_sigma <= 2.0),
            "v1_g2_pass": bool(sd_factor >= 4.0),
        }
    }
    with out.open("a") as handle:
        handle.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
