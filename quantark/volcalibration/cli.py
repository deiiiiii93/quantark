"""One command line for daily vol-model calibration.

    python -m quantark.volcalibration run    <config.yaml> [--as-of DATE]
                                             [--backfill] [--from D] [--to D]
                                             [--max-dates N] [--variants ...]
                                             [--workers N] [--force] [--plan]
                                             [--json]
    python -m quantark.volcalibration status <config.yaml> [--as-of DATE] [--json]
    python -m quantark.volcalibration show   <config.yaml> --date D [--json]
    python -m quantark.volcalibration list   [--config <config.yaml>] [--json]

The agent contract (spec 6.3):

1.  **stdout is data, stderr is narrative.** Under ``--json`` stdout carries
    exactly one JSON object and every progress line goes to stderr.
2.  **Exit codes carry state.** ``0`` current; ``2`` non-current but
    fail-closed; ``1`` pipeline or config failure; ``75`` another process holds
    the lock.  A ``2`` means the system is working correctly and the answer is
    "not yet" -- an agent must not retry-loop on it.  A date can be
    legitimately uncalibratable forever: two MO surfaces are excluded in the
    manifest and will never become admissible.
3.  **Idempotent and resumable.** Re-running a successful config is a no-op
    reporting ``current``; ``--plan`` writes nothing.
4.  **Failures are machine-readable.** Every ``--json`` exit carries a
    ``reason`` code, never prose to parse.

``--as-of`` defaults to the system's local date.  A scheduler should pin it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from quantark.util.exceptions import QuantArkException
from quantark.volcalibration.config import RunConfig
from quantark.volcalibration.runner import (
    EXIT_CURRENT,
    EXIT_FAILED,
    EXIT_LOCKED,
    EXIT_NON_CURRENT,
    LockBusy,
    build_status,
    plan_surface_dates,
    run_pipeline,
    select_calibration_dates,
    status_exit_code,
)
from quantark.volcalibration.store import (
    StoreLayout,
    load_calibration_manifest,
    load_surface_manifest,
    read_json,
)
from quantark.volcalibration.yaml_loader import load_run_config

MODULE = "quantark.volcalibration"
CONFIG_SEARCH_DIRS = (Path("example/mo_volmodels"), Path("example/fx_volmodels"))


# ------------------------------------------------------------------- output


def _emit(payload: Mapping[str, Any], *, as_json: bool, human) -> None:
    if as_json:
        # allow_nan=False: NaN/Infinity are not JSON and strict parsers reject
        # them, which would break the "stdout is one parseable object"
        # guarantee for the agent reading it.  Absent numbers are null.
        json.dump(
            dict(payload),
            sys.stdout,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        sys.stdout.write("\n")
    else:
        human(payload)


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def _parse_date_tag(value: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}; expected YYYY-MM-DD or YYYYMMDD"
    )


# ------------------------------------------------------------------- parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantark.volcalibration",
        description="Build admitted IV surfaces and calibrate vol models from quotes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="build and calibrate pending dates")
    run.add_argument("config", help="path to a YAML run config")
    run.add_argument("--as-of", type=_parse_iso_date, default=None)
    run.add_argument(
        "--backfill",
        action="store_true",
        help="calibrate every stale admitted date, not just the latest",
    )
    run.add_argument("--from", dest="date_from", type=_parse_date_tag, default=None)
    run.add_argument("--to", dest="date_to", type=_parse_date_tag, default=None)
    run.add_argument("--max-dates", type=int, default=None)
    run.add_argument("--variants", nargs="+", default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument(
        "--force",
        action="store_true",
        help="rebuild surfaces a normal run would trust, including exclusions",
    )
    run.add_argument(
        "--plan", action="store_true", help="report what a run would do; write nothing"
    )
    run.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status", help="report pipeline freshness")
    status.add_argument("config", help="path to a YAML run config")
    status.add_argument("--as-of", type=_parse_iso_date, default=None)
    status.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show", help="report one date's verdict in detail")
    show.add_argument("config", help="path to a YAML run config")
    show.add_argument("--date", required=True, type=_parse_date_tag)
    show.add_argument("--variant", default=None)
    show.add_argument("--json", action="store_true")

    listing = subparsers.add_parser("list", help="list configs, or a config's dates")
    listing.add_argument("--config", default=None, help="path to a YAML run config")
    listing.add_argument("--json", action="store_true")
    return parser


# ------------------------------------------------------------------ verb: run


def _apply_overrides(args: argparse.Namespace, config: RunConfig) -> RunConfig:
    if getattr(args, "variants", None):
        config = replace(
            config,
            calibration=replace(config.calibration, variants=tuple(args.variants)),
        )
    if getattr(args, "workers", None) is not None:
        config = replace(config, workers=int(args.workers))
    return config


def _print_plan(payload: Mapping[str, Any]) -> None:
    surfaces = payload["surfaces_to_build"]
    calibrations = payload["calibrations_to_run"]
    print(f"{payload['name']}: plan only, nothing written")
    print(f"  surfaces to build   : {len(surfaces)} {surfaces[:5]}")
    print(f"  calibrations to run : {len(calibrations)} {calibrations[:5]}")


def _print_status(payload: Mapping[str, Any]) -> None:
    freshness = payload.get("freshness", {})
    print(
        f"{payload.get('pipeline')}: {payload.get('overall_status')} | "
        f"expected={payload.get('expected_trade_date')} "
        f"snapshot={freshness.get('snapshot_latest')} "
        f"surface={freshness.get('surface_latest')} "
        f"calibration={freshness.get('calibration_latest')}"
    )
    if payload.get("grandfathered_surface_dates"):
        print(
            f"  {payload['grandfathered_surface_dates']} grandfathered surface "
            "date(s): trusted as-is, never rebuilt without --force"
        )
    if payload.get("last_error"):
        error = payload["last_error"]
        print(f"error: {error.get('error_type')}: {error.get('message')}")


def _cmd_run(args: argparse.Namespace, config: RunConfig) -> int:
    config = _apply_overrides(args, config)
    as_of = args.as_of or date.today()

    if args.plan:
        return _cmd_plan(args, config)

    try:
        code, status = run_pipeline(
            config,
            as_of=as_of,
            backfill=bool(args.backfill),
            max_dates=args.max_dates,
            start_date=args.date_from,
            end_date=args.date_to,
            force=bool(args.force),
            log=_log,
        )
    except LockBusy as exc:
        _emit(
            {"module": MODULE, "reason": "locked", "detail": str(exc)},
            as_json=args.json,
            human=lambda p: print(f"locked: {p['detail']}"),
        )
        return EXIT_LOCKED
    _emit(status, as_json=args.json, human=_print_status)
    return code


def _cmd_plan(args: argparse.Namespace, config: RunConfig) -> int:
    """Resolve the config, scan state, report -- writing nothing."""
    layout = StoreLayout.from_config(config)
    _surface_payload, surface_records = load_surface_manifest(layout)
    calibration_payload, calibration_records = load_calibration_manifest(layout)
    surfaces = plan_surface_dates(
        layout, surface_records, config, force=bool(args.force)
    )
    # What would be calibrated, assuming every planned surface admits.  An
    # optimistic projection is the honest one here: the alternative is to
    # build the surfaces, which --plan promises not to do.
    #
    # Assign, do not setdefault: a date being rebuilt already has a record, and
    # keeping it would let its old sha match an old calibration record and hide
    # the calibration the rebuild will require -- or leave a --force'd excluded
    # date still projected as excluded.
    projected = dict(surface_records)
    for tag in surfaces:
        projected[tag] = {"status": "ok", "artifact_sha256": None}
    calibrations = select_calibration_dates(
        projected,
        calibration_records,
        config=config.calibration.manifest_payload(),
        backfill=bool(args.backfill),
        max_dates=args.max_dates,
        baseline_date=calibration_payload.get("baseline_date")
        or (min(calibration_records) if calibration_records else None),
        start_date=args.date_from,
        end_date=args.date_to,
        variants=tuple(config.calibration.variants),
    )
    payload = {
        "module": MODULE,
        "name": config.name,
        "plan": True,
        "config": config.echo(),
        "surfaces_to_build": surfaces,
        "calibrations_to_run": calibrations,
        "projected_surfaces_assumed_admissible": surfaces,
    }
    _emit(payload, as_json=args.json, human=_print_plan)
    return EXIT_CURRENT if not surfaces and not calibrations else EXIT_NON_CURRENT


# --------------------------------------------------------------- verb: status


def _cmd_status(args: argparse.Namespace, config: RunConfig) -> int:
    layout = StoreLayout.from_config(config)
    status = build_status(layout, config, as_of=args.as_of or date.today())
    _emit(status, as_json=args.json, human=_print_status)
    return status_exit_code(status)


# ----------------------------------------------------------------- verb: show


def _expiry_diagnostics(artifact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-expiry parity and fit quality, in the units a human reads.

    A diagnostic the artifact does not carry is reported ``null``, never NaN.
    Delta-quoted surfaces have no put-call-parity residual at all, and a NaN
    would both be invalid JSON and read as a computed value.
    """
    rows = []
    for pillar in artifact.get("per_expiry", []):
        forward = float(pillar["forward"])
        raw_rmse = pillar.get("parity_rmse_points")
        rmse_points = None if raw_rmse is None else float(raw_rmse)
        raw_mse = pillar.get("sabr_params", {}).get("mse")
        rows.append(
            {
                "T": float(pillar["T"]),
                "expiry_date": pillar.get("expiry_date"),
                "forward": forward,
                "discount_factor": float(pillar["df"]),
                "r": float(pillar["r"]),
                "q": float(pillar["q"]),
                "n_nodes": len(pillar.get("points", [])),
                "off_grid_node_count": pillar.get("off_grid_node_count"),
                "parity_pair_count": pillar.get("pair_count"),
                "parity_rmse_points": rmse_points,
                "parity_rmse_over_forward": (
                    None if rmse_points is None else rmse_points / forward
                ),
                # 1 vol point = 0.01 of implied volatility.
                "sabr_fit_rmse_vol_points": (
                    None if raw_mse is None else math.sqrt(float(raw_mse)) * 100.0
                ),
            }
        )
    return rows


def _cmd_show(args: argparse.Namespace, config: RunConfig) -> int:
    layout = StoreLayout.from_config(config)
    tag = args.date
    _surface_payload, surface_records = load_surface_manifest(layout)
    _calibration_payload, calibration_records = load_calibration_manifest(layout)
    record = surface_records.get(tag, {})
    artifact = read_json(layout.artifact_path(tag), default=None)

    if artifact is None:
        surface: Dict[str, Any] = {
            "status": record.get("status", "absent"),
            "reason": record.get("reason"),
            "detail": record.get("detail"),
            "artifact_path": str(layout.artifact_path(tag)),
        }
        if not record:
            surface["status"] = "absent"
            surface["detail"] = "no manifest record and no artifact on disk"
    else:
        surface = {
            "status": record.get("status", "ok"),
            "reason": record.get("reason"),
            "detail": record.get("detail"),
            "artifact_path": str(layout.artifact_path(tag)),
            "artifact_sha256": record.get("artifact_sha256"),
            "snapshot_sha256": record.get("snapshot_sha256"),
            "provenance": record.get("provenance"),
            "s0": artifact.get("s0"),
            "admission": artifact.get("admission"),
            "node_universe": artifact.get("node_universe"),
            "atm_pillars": artifact.get("atm_pillars"),
            "extrapolation_policy": artifact.get("extrapolation_policy"),
            "per_expiry": _expiry_diagnostics(artifact),
        }

    calibration = calibration_records.get(tag, {})
    payload = {
        "module": MODULE,
        "name": config.name,
        "date": tag,
        "surface": surface,
        "calibration": {
            "status": calibration.get("status", "absent"),
            "surface_sha": calibration.get("surface_sha"),
            "config": calibration.get("config"),
            "variants": calibration.get("variants", {}),
            "temporal_scheme": calibration.get("temporal_scheme"),
        },
    }
    _emit(payload, as_json=args.json, human=_print_show)

    if surface["status"] != "ok":
        return EXIT_NON_CURRENT
    if calibration.get("status") == "failed":
        return EXIT_FAILED
    if calibration.get("status") != "ok":
        return EXIT_NON_CURRENT
    return EXIT_CURRENT


def _print_show(payload: Mapping[str, Any]) -> None:
    surface = payload["surface"]
    print(f"{payload['date']}: surface {surface['status']}", end="")
    if surface.get("reason"):
        print(f" ({surface['reason']}: {surface.get('detail')})")
    else:
        print()
    for row in surface.get("per_expiry", []):
        parity = row["parity_rmse_over_forward"]
        sabr = row["sabr_fit_rmse_vol_points"]
        print(
            f"  T={row['T']:.4f} {row['expiry_date']} F={row['forward']:.2f} "
            f"r={row['r']:+.4f} q={row['q']:+.4f} nodes={row['n_nodes']} "
            f"parity_rmse/F={'n/a' if parity is None else f'{parity:.2e}'} "
            f"sabr_rmse={'n/a' if sabr is None else f'{sabr:.3f}vp'}"
        )
    calibration = payload["calibration"]
    print(f"  calibration: {calibration['status']}")
    for variant, item in sorted(calibration.get("variants", {}).items()):
        record = item.get("record", {})
        extra = ""
        if "feller_ratio" in record:
            extra = (
                f" feller={record['feller_ratio']:.3f} "
                f"rmse_iv={record.get('overall_rmse_iv', float('nan')):.5f} "
                f"bound_hits={record.get('bound_hits')}"
            )
        elif "leverage_min" in record:
            extra = (
                f" leverage=[{record['leverage_min']:.3f}, "
                f"{record['leverage_max']:.3f}]"
            )
        elif "lv_min" in record:
            extra = f" lv=[{record['lv_min']:.4f}, {record['lv_max']:.4f}]"
        print(f"    {variant:<12} {item.get('status')}{extra}")


# ----------------------------------------------------------------- verb: list


def _cmd_list(args: argparse.Namespace, config: Optional[RunConfig]) -> int:
    if config is None:
        configs = []
        for directory in CONFIG_SEARCH_DIRS:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.yaml")):
                try:
                    loaded = load_run_config(candidate)
                except QuantArkException as exc:
                    # Reported, not skipped: a directory holds YAML that is not
                    # a run config, and saying so beats an unexplained absence.
                    configs.append({"path": str(candidate), "error": str(exc)})
                else:
                    configs.append(
                        {
                            "path": str(candidate),
                            "name": loaded.name,
                            "symbol": loaded.underlying.symbol,
                            "convention": loaded.underlying.convention,
                        }
                    )
        _emit(
            {"module": MODULE, "configs": configs},
            as_json=args.json,
            human=_print_config_list,
        )
        return EXIT_CURRENT

    layout = StoreLayout.from_config(config)
    _surface_payload, surface_records = load_surface_manifest(layout)
    _calibration_payload, calibration_records = load_calibration_manifest(layout)
    rows = []
    for tag in sorted(surface_records):
        record = surface_records[tag]
        calibration = calibration_records.get(tag, {})
        rows.append(
            {
                "date": tag,
                "surface_status": record.get("status"),
                "reason": record.get("reason"),
                "provenance": record.get("provenance"),
                "n_expiries": record.get("n_expiries"),
                "calibrated_variants": sorted(
                    name
                    for name, item in calibration.get("variants", {}).items()
                    if item.get("status") == "ok"
                ),
            }
        )
    _emit(
        {"module": MODULE, "name": config.name, "dates": rows},
        as_json=args.json,
        human=_print_date_list,
    )
    return EXIT_CURRENT


def _print_config_list(payload: Mapping[str, Any]) -> None:
    print("Run configs:")
    for item in payload["configs"]:
        if "error" in item:
            print(f"  {item['path']}  (not a run config: {item['error']})")
        else:
            print(
                f"  {item['path']}  {item['name']} "
                f"[{item['symbol']}, {item['convention']}]"
            )
    if not payload["configs"]:
        print("  (none)")


def _print_date_list(payload: Mapping[str, Any]) -> None:
    print(f"{payload['name']}: {len(payload['dates'])} date(s)")
    for row in payload["dates"]:
        variants = ",".join(row["calibrated_variants"]) or "-"
        suffix = f" ({row['reason']})" if row.get("reason") else ""
        print(f"  {row['date']}  {row['surface_status']:<9}{suffix}  {variants}")


# ------------------------------------------------------------------- dispatch

_COMMANDS = {
    "run": _cmd_run,
    "status": _cmd_status,
    "show": _cmd_show,
    "list": _cmd_list,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns an exit code; raises SystemExit only on usage errors."""
    args = _build_parser().parse_args(argv)
    as_json = bool(getattr(args, "json", False))

    # Only `list` can arrive without one: the other three take it as a
    # required positional, so argparse has already refused a missing config.
    config: Optional[RunConfig] = None
    if args.config is not None:
        try:
            config = load_run_config(args.config)
        except QuantArkException as exc:
            print(f"error: {exc}", file=sys.stderr)
            _emit(
                {"module": MODULE, "reason": "invalid_config", "detail": str(exc)},
                as_json=as_json,
                human=lambda p: None,
            )
            return EXIT_FAILED

    try:
        return _COMMANDS[args.command](args, config)
    except LockBusy as exc:
        print(f"locked: {exc}", file=sys.stderr)
        _emit(
            {"module": MODULE, "reason": "locked", "detail": str(exc)},
            as_json=as_json,
            human=lambda p: None,
        )
        return EXIT_LOCKED
    except QuantArkException as exc:
        print(f"error: {exc}", file=sys.stderr)
        _emit(
            {"module": MODULE, "reason": "pipeline_failed", "detail": str(exc)},
            as_json=as_json,
            human=lambda p: None,
        )
        return EXIT_FAILED
