"""Price the otc-price-adapter trade book under a chosen quantark tree.

Dumps every numeric field at full precision so two trees can be compared
exactly, and reports wall time. Run from the adapter repo root with
PYTHONPATH=<quantark-tree>:<adapter-repo>.

    python book_probe.py --model mc --out /path/out.json [--limit N] [--workers N]

NEVER run this as a heredoc: stdin parents cannot spawn workers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ADAPTER = Path("/Users/fuxinyao/otc-price-adapter")
sys.path.insert(0, str(ADAPTER))
sys.path.insert(0, str(ADAPTER / "tests"))


NUMERIC_FIELDS = [
    "pv", "pv_total", "pv_premium", "pv_interest", "pv_rebate",
    "pv_margin", "pv_minimum_return",
    "delta", "gamma", "vega", "theta", "rho", "rhoQ",
    "delta_total", "gamma_total", "vega_total", "theta_total",
    "rho_total", "rhoQ_total",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pde", choices=["pde", "mc", "quad"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None,
                    help="price N rows per distinct structure (diverse subset)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import quantark
    from freeze_v023_baseline import build_frame_and_settings, diverse_subset
    import otc_quantark_pricer_v023 as pricer_v023

    try:
        from quantark.montecarlo.gbm_kernels import gbm_backend
        gbm = gbm_backend()
    except ImportError:
        gbm = "absent (pre-change tree)"
    try:
        from quantark.montecarlo.qe_kernels import qe_backend
        qe = qe_backend()
    except ImportError:
        qe = "absent"

    frame, settings = build_frame_and_settings()
    settings = dataclasses.replace(
        settings, autocallable_model=args.model, workers=args.workers
    )
    if args.subset:
        frame = diverse_subset(frame, per_structure=args.subset)
    if args.limit:
        frame = frame.iloc[: args.limit]

    print(f"label={args.label}", flush=True)
    print(f"quantark from {quantark.__file__}", flush=True)
    print(f"gbm_backend={gbm}  qe_backend={qe}", flush=True)
    print(f"model={args.model} rows={len(frame)} workers={args.workers}", flush=True)

    rows = {}
    t0 = time.perf_counter()
    for trade_id, row in frame.iterrows():
        r0 = time.perf_counter()
        result = pricer_v023.price_row(row, settings)
        dt = time.perf_counter() - r0
        rec = {"pricing_status": str(result["pricing_status"]), "seconds": dt}
        for f in NUMERIC_FIELDS:
            v = result.get(f)
            try:
                rec[f] = float(v).hex()
            except (TypeError, ValueError):
                rec[f] = None
        rec["structure"] = str(row.get("结构类型", ""))
        rows[str(trade_id)] = rec
        print(f"  {trade_id} {rec['structure'][:24]:24} {dt:7.2f}s "
              f"{rec['pricing_status'][:30]}", flush=True)
    total = time.perf_counter() - t0

    payload = {
        "label": args.label,
        "model": args.model,
        "gbm_backend": gbm,
        "qe_backend": qe,
        "quantark_path": quantark.__file__,
        "total_seconds": total,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"TOTAL {total:.2f}s over {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
