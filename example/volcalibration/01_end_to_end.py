"""End-to-end tour of `quantark.volcalibration`, from quotes to a priced model.

Runs from a clean clone: no network, no private history.  The six committed
CFETS USD/CNY snapshots carry their own spot, so this exercises every stage of
the module for real -- normalization, SABR smoothing, static-arbitrage
admission, Dupire calibration, and the `CalibrationSet` handover -- in seconds.

    .venv/bin/python example/volcalibration/01_end_to_end.py

Six acts:

  1. Get quotes into the store          (the one job that stays in example/)
  2. Ask what a run would do            (--plan writes nothing)
  3. Run it                             (surfaces + calibration, atomic, locked)
  4. Ask again                          (idempotent: current, exit 0)
  5. Inspect one date                   (show: admission + per-expiry diagnostics)
  6. Consume it                         (CalibrationSet -> surface, model, env)

The equity-index workflow this module was built for is the same six acts
against example/mo_volmodels/mo_calibration.yaml; see the README.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantark.volcalibration import CalibrationSet, QuoteSnapshot  # noqa: E402
from quantark.volcalibration.cli import main as cli  # noqa: E402
from quantark.volcalibration.store import (  # noqa: E402
    StoreLayout,
    atomic_write_json,
)
from quantark.volcalibration.yaml_loader import load_run_config  # noqa: E402

CONFIG = HERE / "usdcny_calibration.yaml"
CFETS_DIR = PROJECT_ROOT / "example" / "fx_volmodels" / "data"
AS_OF = "2026-07-20"  # the newest committed snapshot


def banner(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n{n}. {title}\n{'=' * 72}")


def stage_snapshots(layout: StoreLayout) -> list[str]:
    """Vendor payloads -> canonical snapshots the library will read.

    D1 keeps vendor formats in example/: the library reads canonical
    ``QuoteSnapshot`` envelopes from ``<root>/snapshots/{YYYYMMDD}.json`` and
    never parses a vendor file.  This is that bridge for CFETS -- the FX
    analogue of example/mo_volmodels/export_snapshots.py.
    """
    tags = []
    for path in sorted(CFETS_DIR.glob("cfets_usdcny_snapshot_2*.json")):
        snapshot = QuoteSnapshot.from_legacy_fx(json.loads(path.read_text()))
        tag = snapshot.trade_date.strftime("%Y%m%d")
        atomic_write_json(layout.snapshot_path(tag), snapshot.to_payload())
        tags.append(tag)
        print(
            f"  {tag}  spot={snapshot.spot:.4f}  "
            f"tenors={len(snapshot.expiries)}  id={snapshot.sha256[:12]}"
        )
    if not tags:
        raise SystemExit(f"no CFETS snapshots found under {CFETS_DIR}")
    return tags


def main() -> int:
    config = load_run_config(CONFIG)
    layout = StoreLayout.from_config(config)

    banner(1, "Get quotes into the store")
    print(f"store: {layout.history_dir}")
    tags = stage_snapshots(layout)

    banner(2, "Ask what a run would do (--plan writes nothing)")
    code = cli(["run", str(CONFIG), "--as-of", AS_OF, "--backfill", "--plan"])
    print(f"exit={code}   (2 = work pending; 0 would mean nothing to do)")

    banner(3, "Run it")
    code = cli(["run", str(CONFIG), "--as-of", AS_OF, "--backfill"])
    print(f"exit={code}")

    banner(4, "Ask again -- a successful re-run is a no-op")
    code = cli(["status", str(CONFIG), "--as-of", AS_OF, "--json"])
    print(f"exit={code}   (0 = current)")

    banner(5, f"Inspect one date ({AS_OF})")
    code = cli(["show", str(CONFIG), "--date", AS_OF, "--json"])
    print(f"exit={code}")

    banner(6, "Consume it: CalibrationSet is the supported entry point")
    calibrations = CalibrationSet.open(layout.history_dir)
    dates = calibrations.dates()
    print(f"admitted dates: {[d.isoformat() for d in dates]}")

    when = dates[-1]
    artifact = calibrations.surface_for(when)
    print(
        f"\nsurface_for({when}): sha={artifact.sha256[:12]} "
        f"strikes={len(artifact.strikes)} maturities={len(artifact.maturities)}"
    )

    env, _surface, spot = calibrations.environment_for(when)
    print(f"environment_for({when}): spot={spot:.4f} "
          f"valuation_date={env.valuation_date.date()}")
    print("  the rate and dividend curves are built from the artifact's own")
    print("  parity pillars, so they reproduce the published forwards:")
    for pillar in artifact.per_expiry[:3]:
        T = float(pillar["T"])
        published = float(pillar["forward"])
        # For FX the 'dividend yield' slot carries the foreign rate, so this
        # is the covered-interest-parity forward.
        rebuilt = spot * math.exp(
            (env.rate_curve.get_rate(T) - env.div_yield.get_yield(T)) * T
        )
        print(
            f"    T={T:.4f}  published={published:.6f}  "
            f"from curves={rebuilt:.6f}  diff={abs(rebuilt - published):.2e}"
        )

    model = calibrations.model_for(when, "localvol")
    record = model.record
    print(
        f"\nmodel_for({when}, 'localvol'): "
        f"grid={record['grid_shape']} "
        f"local vol in [{record['lv_min']:.4f}, {record['lv_max']:.4f}] "
        f"cache_hit={record['cache_hit']}"
    )

    print(
        "\nA backtest consumes exactly this object:\n"
        "    AutocallableMarketDataSet(..., calibration_set=calibrations)\n"
        "which derives the surface history from it -- nothing in backtest\n"
        "reaches into a directory."
    )

    banner(0, "Where the bytes went")
    for label, path in (
        ("snapshots", layout.snapshots_dir),
        ("artifacts", layout.surface_dir),
        ("surface manifest", layout.surface_manifest),
        ("calibration manifest", layout.calibration_manifest),
        ("calibration cache", layout.calibration_cache),
        ("status", layout.status),
    ):
        print(f"  {label:<22} {path}")
    print(f"\n{len(tags)} date(s) calibrated. Re-run this script: it is a no-op.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
