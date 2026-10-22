"""Stage 03-history — SABR-smoothed IV-surface history from frozen CFFEX settlements.

Turns every frozen official CFFEX end-of-day settlement CSV
(``data/history/settlement_csv/{YYYYMMDD}_1.csv``, produced by
``01_bulk_fetch_settlement_history.py``) into a SABR-smoothed implied-vol
surface artifact ``mo_iv_surface_{YYYYMMDD}.json`` plus a fail-closed
``surface_manifest.json``.  This is the vol-history data build for multi-year
daily backtests (e.g. the 3Y snowball study); it runs fully offline under
``.venv/bin/python`` — no akshare, no network.

Pipeline per trading date (reusing the suite's existing machinery):

1. ``01_fetch_mo_settlement_history.parse_cffex_csv`` — fail-closed GB18030
   CSV parse into a validated settlement snapshot.
2. ``10_calibration_diagnostics.build_calibration_nodes`` — settlement-based
   put-call parity (OLS of C−P on strike), the frozen parity quality gates
   (±10% implied rate, 1%-of-forward RMSE), the OTM liquidity filter
   (volume > 0 and open interest > 0 on both wings) and the normalized
   Black-IV inversion.  Its coverage gate is relaxed here (``min_expiries`` /
   ``min_nodes``) because this stage admits any date with >= 2 SABR-fittable
   expiries instead of the frozen cross-date Heston universe.
3. Stage-02 grid assembly (strikes shared by >= 2 expiries inside the overlap
   of the per-expiry quoted strike ranges, per-expiry linear interpolation)
   followed by ``_mo_common.sabr_smoothed_surface`` — one Hagan SABR slice per
   expiry, fit on the nodes inside that shared domain and evaluated on the
   rectangular grid, with total variance projected non-decreasing in maturity.
4. Static-arbitrage admission: the smoothed grid must pass the same
   calendar + butterfly validation the local-vol stage uses —
   ``quantark.volmodels.localvol.build_dupire_local_vol`` with its default
   ``validate_arbitrage=True``.  With exactly 2 surviving expiries the Dupire
   builder's >= 3-maturity requirement does not apply, so the identical two
   checks are evaluated in reduced form with the same ``quantark`` finite
   differences and tolerances.

Fail-closed: a date whose data or surface fails any check gets NO artifact
and a manifest entry ``{date, status: "excluded", reason, detail}``.  Gaps are
never filled: the manifest provenance records
``"gap_policy": "consumers carry forward previous admitted surface"``.

Determinism: identical input CSV → byte-identical artifact (``sort_keys``,
``allow_nan=False``, no timestamps inside the artifact body other than
``trade_date``; generation time lives only in the manifest).

Example::

    .venv/bin/python example/mo_volmodels/03_build_iv_surface_history.py \
        --max-dates 5 --workers 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc  # noqa: E402

from quantark.util.exceptions import NumericalError, ValidationError  # noqa: E402
from quantark.util.numerical import (  # noqa: E402
    fd1_nonuniform,
    fd2_nonuniform,
    is_positive,
    safe_log,
)
from quantark.util.numerical.constants import Tolerance  # noqa: E402


def _load_numbered(filename: str, module_name: str):
    """Load a sibling stage script whose filename starts with digits."""
    spec = importlib.util.spec_from_file_location(module_name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage01 = _load_numbered("01_fetch_mo_settlement_history.py", "mo_settlement_fetcher_history")
stage10 = _load_numbered("10_calibration_diagnostics.py", "mo_calibration_diagnostics_history")

SOURCE_CLASS = "official_cffex_eod_settlement"
PRICE_FIELD = "settlement"
ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
# Admission configuration (frozen for the whole history build).
SABR_BETA = 1.0
MIN_EXPIRIES = 2
MIN_STRIKES_PER_EXPIRY = 5  # stage-02 MIN_STRIKES convention
MIN_COMMON_STRIKES = 3  # butterfly check needs >= 3 grid strikes
GAP_POLICY = "consumers carry forward previous admitted surface"
EXTRAPOLATION_POLICY = "flat_total_variance"

DEFAULT_CSV_DIR = HERE / "data" / "history" / "settlement_csv"
DEFAULT_OUTPUT_DIR = HERE / "data" / "history" / "iv_surface"
DEFAULT_MANIFEST = HERE / "data" / "history" / "surface_manifest.json"
DEFAULT_SPOT_CSV = HERE / "data" / "history" / "csi1000_spot.csv"


class AdmissionError(Exception):
    """A date failed an admission check; it is excluded, never repaired."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def artifact_path(output_dir: Path, trade_date: str) -> Path:
    return output_dir / f"mo_iv_surface_{trade_date}.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temp file, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_spot_map(spot_csv: Path) -> dict[str, float]:
    """Load the CSI 1000 spot cache (ISO date -> close) from the market cache."""
    spots: dict[str, float] = {}
    with spot_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["spot"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                spots[str(row["date"])] = value
    if not spots:
        raise ValueError(f"spot cache {spot_csv} contains no usable rows")
    return spots


def _surface_base(snapshot: dict, s0: float) -> dict:
    """Parity + OTM IV nodes per expiry, assembled into the stage-02 grid schema."""
    try:
        nodes, universe = stage10.build_calibration_nodes(
            snapshot, min_expiries=1, min_nodes=1
        )
    except stage10.CoverageError as exc:
        raise AdmissionError("parity_gating_failed", str(exc)) from exc
    except ValueError as exc:
        raise AdmissionError("invalid_snapshot", str(exc)) from exc

    by_expiry: dict[str, list[dict]] = {}
    for node in nodes:
        by_expiry.setdefault(str(node["expiry_date"]), []).append(node)

    parity_by_expiry = {
        str(row["expiry_date"]): row for row in universe["per_expiry"]
    }
    per_expiry = []
    dropped = []
    for expiry_date, exp_nodes in sorted(
        by_expiry.items(), key=lambda item: float(item[1][0]["T"])
    ):
        if len(exp_nodes) < MIN_STRIKES_PER_EXPIRY:
            dropped.append(
                {
                    "expiry_date": expiry_date,
                    "reason": "fewer_than_min_strikes",
                    "node_count": len(exp_nodes),
                    "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
                }
            )
            continue
        exp_nodes = sorted(exp_nodes, key=lambda node: float(node["strike"]))
        head = exp_nodes[0]
        T = float(head["T"])
        forward = float(head["forward"])
        df = float(head["discount_factor"])
        r = float(-safe_log(df) / T)
        q = float(r - safe_log(forward / s0) / T)
        if not (math.isfinite(r) and math.isfinite(q)):
            raise AdmissionError(
                "non_finite_carry",
                f"expiry {expiry_date}: r={r}, q={q}",
            )
        parity_row = parity_by_expiry.get(expiry_date, {})
        per_expiry.append(
            {
                "expiry_date": expiry_date,
                "T": T,
                "r": r,
                "q": q,
                "forward": forward,
                "df": df,
                "pair_count": int(parity_row.get("pair_count", 0)),
                "parity_rmse_points": (
                    float(parity_row["parity_rmse_points"])
                    if "parity_rmse_points" in parity_row
                    else None
                ),
                "points": [
                    (float(node["strike"]), float(node["market_iv"]))
                    for node in exp_nodes
                ],
            }
        )

    if len(per_expiry) < MIN_EXPIRIES:
        raise AdmissionError(
            "insufficient_expiries",
            f"{len(per_expiry)} expiries with >= {MIN_STRIKES_PER_EXPIRY} nodes "
            f"(need >= {MIN_EXPIRIES}); dropped={dropped}",
        )
    for i in range(len(per_expiry) - 1):
        if per_expiry[i + 1]["T"] <= per_expiry[i]["T"]:
            raise AdmissionError(
                "duplicate_maturity",
                f"{per_expiry[i]['expiry_date']} and "
                f"{per_expiry[i + 1]['expiry_date']} share T={per_expiry[i]['T']}",
            )

    # Domain of the surface: the overlap of the per-expiry QUOTED strike
    # ranges.  Listed ladders differ by contract month, so a union-range grid
    # would force SABR wing evaluation where the market never quoted, and a
    # single expiry's deep-wing settlement ticks (tick-floor IV inflation)
    # would pull its SABR vol-of-vol and distort the whole slice.  Each SABR
    # fit therefore uses only the nodes inside this shared domain; off-grid
    # wing nodes are counted for audit, not silently dropped.  Dropping an
    # expiry can widen the overlap, so iterate to the fixed point.  Every
    # trim-dropped expiry is persisted in node_universe.excluded_expiries.
    dropped_trimmed: list[dict] = []
    for _ in range(len(per_expiry)):
        grid_lo = max(min(k for k, _ in pe["points"]) for pe in per_expiry)
        grid_hi = min(max(k for k, _ in pe["points"]) for pe in per_expiry)
        kept = []
        dropped_trim = []
        for pe in per_expiry:
            in_grid = [
                (k, v) for k, v in pe["points"] if grid_lo - 1e-9 <= k <= grid_hi + 1e-9
            ]
            if len(in_grid) >= MIN_STRIKES_PER_EXPIRY:
                pe["off_grid_node_count"] = len(pe["points"]) - len(in_grid)
                pe["points"] = in_grid
                kept.append(pe)
            else:
                dropped_trim.append(pe["expiry_date"])
                dropped_trimmed.append(
                    {
                        "expiry_date": pe["expiry_date"],
                        "reason": "fewer_than_min_strikes_inside_quoted_range_overlap",
                        "node_count": len(pe["points"]),
                        "in_domain_node_count": len(in_grid),
                        "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
                        "quoted_range_overlap": [grid_lo, grid_hi],
                    }
                )
        if not dropped_trim:
            break
        per_expiry = kept
        if len(per_expiry) < MIN_EXPIRIES:
            raise AdmissionError(
                "insufficient_expiries",
                f"< {MIN_EXPIRIES} expiries with >= {MIN_STRIKES_PER_EXPIRY} "
                f"nodes inside the quoted-range overlap; "
                f"dropped={dropped_trim}",
            )

    # Rectangular grid: strikes present (near-exactly) in >= 2 expiries inside
    # the domain, each expiry row linearly interpolated from its own smile —
    # the stage-02 rule.
    all_strikes = sorted({k for pe in per_expiry for k, _ in pe["points"]})

    def _count(k: float) -> int:
        return sum(
            any(abs(k - kk) < 1e-6 for kk, _ in pe["points"]) for pe in per_expiry
        )

    strikes = [
        k for k in all_strikes if _count(k) >= 2 and grid_lo <= k <= grid_hi
    ]
    if len(strikes) < MIN_COMMON_STRIKES:
        raise AdmissionError(
            "insufficient_common_strikes",
            f"{len(strikes)} shared strikes inside quoted-range overlap "
            f"[{grid_lo}, {grid_hi}] (< {MIN_COMMON_STRIKES})",
        )
    maturities = [pe["T"] for pe in per_expiry]
    grid = np.empty((len(maturities), len(strikes)))
    for i, pe in enumerate(per_expiry):
        ks = np.array([k for k, _ in pe["points"]])
        vs = np.array([v for _, v in pe["points"]])
        grid[i] = np.interp(strikes, ks, vs)  # flat past this expiry's wings

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "trade_date": snapshot["trade_date"],
        "source_class": SOURCE_CLASS,
        "price_field": PRICE_FIELD,
        "source_url": snapshot.get("source_url"),
        "source_sha256": snapshot["source_sha256"],
        "s0": float(s0),
        "strikes": strikes,
        "maturities": maturities,
        "iv_grid": grid.tolist(),
        "per_expiry": per_expiry,
        "node_universe": {
            "node_count": int(universe["node_count"]),
            "expiry_count": int(universe["expiry_count"]),
            "filtered_quote_counts": universe["filtered_quote_counts"],
            "excluded_expiries": universe["excluded_expiries"] + dropped + dropped_trimmed,
        },
    }


def _validate_static_arbitrage(surface: dict) -> str:
    """Run the suite's LV-input arbitrage validation on the smoothed grid.

    With >= 3 maturities this is exactly ``build_dupire_local_vol``'s default
    ``validate_arbitrage=True`` path (calendar dw/dT|_y >= 0 plus butterfly
    denominator > 0).  With exactly 2 maturities the Dupire builder refuses
    (it needs >= 3), so the same two checks are evaluated in reduced form with
    the same quantark finite differences and ``Tolerance`` thresholds; the
    maturity direction then uses the two-point one-sided stencil.  Returns the
    validation method label for the artifact's admission record.
    """
    from quantark.param import GridVolSurface
    from quantark.param.div import TermStructureDividendYield
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.volmodels.localvol import build_dupire_local_vol

    s0 = float(surface["s0"])
    pe = surface["per_expiry"]
    ts = [float(p["T"]) for p in pe]
    rate_curve = LinearRateCurve([(float(p["T"]), float(p["r"])) for p in pe])
    div = TermStructureDividendYield(
        times=ts, yields=[float(p["q"]) for p in pe]
    )
    surf = GridVolSurface(
        surface["strikes"], surface["maturities"], np.array(surface["iv_grid"])
    )
    if len(ts) >= 3:
        build_dupire_local_vol(
            surf, spot=s0, rate_curve=rate_curve, div_yield=div.get_yield
        )
        return "build_dupire_local_vol(validate_arbitrage=True)"

    # Reduced-form replica of the Dupire checks for the 2-maturity edge case.
    K = np.asarray(surf.strikes, dtype=float)
    T = np.asarray(surf.maturities, dtype=float)
    iv = np.asarray(surf.iv_grid, dtype=float)
    ln_k = np.log(K)
    r_zero = np.array([rate_curve.get_rate(t) for t in T])
    q_zero = np.array([div.get_yield(t) for t in T])
    fwd = s0 * np.exp((r_zero - q_zero) * T)
    w = iv**2 * T[:, None]
    y = ln_k[None, :] - np.log(fwd)[:, None]
    if np.any(w <= 1e-12):
        raise NumericalError("degenerate total implied variance in 2-maturity grid")
    w_y = fd1_nonuniform(w, ln_k)
    w_yy = fd2_nonuniform(w, ln_k)
    dw_dT = (w[1] - w[0]) / (T[1] - T[0])  # only stencil available with 2 rows
    dlnF_dT = (math.log(fwd[1]) - math.log(fwd[0])) / (T[1] - T[0])
    dw_dT_y = dw_dT + dlnF_dT * w_y
    inv_w = 1.0 / w
    denom = (
        1.0
        - y * inv_w * w_y
        + 0.25 * (-0.25 - inv_w + (y * y) * inv_w * inv_w) * (w_y * w_y)
        + 0.5 * w_yy
    )
    nT, nK = iv.shape
    is_edge = np.zeros((nT, nK), dtype=bool)
    is_edge[0, :] = is_edge[-1, :] = True
    is_edge[:, 0] = is_edge[:, -1] = True
    cal_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, Tolerance.PRECISION)
    if np.any(dw_dT_y < -cal_thresh):
        raise NumericalError(
            "calendar arbitrage: dw/dT|_y < 0 (moneyness) in 2-maturity grid"
        )
    bf_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, 0.0)
    if np.any(denom < -bf_thresh):
        raise NumericalError(
            "butterfly arbitrage: Dupire denominator < 0 in 2-maturity grid"
        )
    # Dupire's post-check, same semantics as build_dupire_local_vol with
    # vol_floor=None: non-finite or non-positive local variance rejects.
    with np.errstate(divide="ignore", invalid="ignore"):
        lv2 = dw_dT_y / denom
    bad = ~np.isfinite(lv2) | (lv2 <= 0)
    if np.any(bad):
        idx = np.argwhere(bad)
        raise NumericalError(
            "Dupire produced non-positive/instable local variance at nodes "
            f"{idx.tolist()} in 2-maturity grid; the input surface is "
            "inadmissible — fix the input, do not floor"
        )
    return "reduced_form_dupire_checks_2_maturities"


def build_surface_artifact(trade_date: str, *, csv_dir: Path, s0: float | None) -> dict:
    """Build one date's SABR-smoothed surface artifact or raise AdmissionError."""
    source_path = csv_dir / f"{trade_date}_1.csv"
    if s0 is None:
        raise AdmissionError("missing_spot", f"no CSI 1000 spot for {trade_date}")
    if not is_positive(s0):
        raise AdmissionError("invalid_spot", f"non-positive spot {s0} for {trade_date}")
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise AdmissionError("missing_csv", str(exc)) from exc
    try:
        snapshot = stage01.parse_cffex_csv(payload, trade_date)
    except ValueError as exc:
        raise AdmissionError("parse_failed", str(exc)) from exc

    raw_surface = _surface_base(snapshot, float(s0))
    try:
        smoothed = mc.sabr_smoothed_surface(raw_surface, beta=SABR_BETA)
    except (ValueError, NumericalError, ValidationError) as exc:
        raise AdmissionError(
            "sabr_smoothing_failed", f"{type(exc).__name__}: {exc}"
        ) from exc

    # ATM pillars: SABR-smoothed vol at each expiry's parity forward.
    from quantark.param.vol.sabr.hagan import sabr_implied_vol_black

    atm_pillars = []
    for pe in smoothed["per_expiry"]:
        params = pe["sabr_params"]
        atm_vol = float(
            sabr_implied_vol_black(
                float(pe["forward"]),
                [float(pe["forward"])],
                [float(pe["T"])],
                params["alpha"],
                params["beta"],
                params["rho"],
                params["nu"],
                shift=params["shift"],
            )[0]
        )
        if not (math.isfinite(atm_vol) and atm_vol > 0.0):
            raise AdmissionError(
                "invalid_atm_pillar",
                f"expiry {pe['expiry_date']}: SABR ATM vol {atm_vol}",
            )
        atm_pillars.append(
            {
                "T": float(pe["T"]),
                "expiry_date": pe["expiry_date"],
                "atm_vol": atm_vol,
            }
        )

    try:
        validation_method = _validate_static_arbitrage(smoothed)
    except (NumericalError, ValidationError) as exc:
        raise AdmissionError(
            "static_arbitrage", f"{type(exc).__name__}: {exc}"
        ) from exc

    max_listed_t = max(float(t) for t in smoothed["maturities"])
    smoothed["atm_pillars"] = atm_pillars
    smoothed["extrapolation_policy"] = {
        "beyond_last_listed_expiry": EXTRAPOLATION_POLICY,
        "max_listed_T": max_listed_t,
    }
    smoothed["admission"] = {
        "min_expiries": MIN_EXPIRIES,
        "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
        "min_common_strikes": MIN_COMMON_STRIKES,
        "sabr_beta": SABR_BETA,
        "parity_quality_gate": {
            "maximum_absolute_annualized_implied_rate": stage10.MAX_ABS_PARITY_IMPLIED_RATE,
            "maximum_rmse_divided_by_forward": stage10.MAX_PARITY_RMSE_FORWARD_RATIO,
        },
        "static_arbitrage_validation": validation_method,
        "strike_grid": "shared_by_>=2_expiries_within_quoted_range_overlap",
        "sabr_fit_domain": "nodes_inside_strike_grid_only",
    }
    return smoothed


def serialize_artifact(artifact: dict) -> bytes:
    """Deterministic artifact bytes: sorted keys, no NaN, trailing newline."""
    return (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _remove_stale_artifact(output_dir: Path, trade_date: str) -> None:
    """Delete a previously-built artifact when its date is re-excluded.

    Only reachable under ``--force`` (existing artifacts are skipped
    otherwise); directory-globbing consumers must never pick up a stale
    surface for a date that no longer passes admission.
    """
    try:
        artifact_path(output_dir, trade_date).unlink()
    except FileNotFoundError:
        pass


def _build_one(task: tuple) -> dict:
    """Worker: build + atomically write one date's artifact; return its manifest entry."""
    trade_date, csv_dir, output_dir, s0 = task
    entry = {
        "date": trade_date,
        "status": "excluded",
        "reason": None,
        "detail": None,
        "n_expiries": 0,
        "artifact_sha256": None,
    }
    try:
        artifact = build_surface_artifact(
            trade_date, csv_dir=Path(csv_dir), s0=s0
        )
        data = serialize_artifact(artifact)
    except AdmissionError as exc:
        _remove_stale_artifact(Path(output_dir), trade_date)
        entry["reason"] = exc.reason
        entry["detail"] = exc.detail
        return entry
    except Exception as exc:  # fail closed on anything unexpected
        _remove_stale_artifact(Path(output_dir), trade_date)
        entry["reason"] = "unexpected_error"
        entry["detail"] = f"{type(exc).__name__}: {exc}"
        return entry
    _atomic_write_bytes(artifact_path(Path(output_dir), trade_date), data)
    entry.update(
        status="ok",
        n_expiries=len(artifact["maturities"]),
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )
    return entry


def load_manifest_records(manifest_path: Path) -> dict[str, dict]:
    """Load existing surface-manifest records keyed by date; tolerate absence."""
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"surface manifest {manifest_path} is corrupt ({exc}); "
            "delete it and rerun to rebuild records from existing artifacts"
        ) from exc
    return {str(record["date"]): record for record in payload.get("records", [])}


def save_manifest(
    manifest_path: Path, records: dict[str, dict], *, window: dict
) -> None:
    """Rewrite the surface manifest without dropping downstream policy metadata."""
    previous: dict = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"surface manifest {manifest_path} is corrupt ({exc}); "
                "refusing to replace its provenance metadata"
            ) from exc
        if isinstance(loaded, dict):
            previous = loaded
    payload = dict(previous)
    payload.update({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": SOURCE_CLASS,
        "price_field": PRICE_FIELD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "gap_policy": GAP_POLICY,
        "config": {
            "sabr_beta": SABR_BETA,
            "min_expiries": MIN_EXPIRIES,
            "min_strikes_per_expiry": MIN_STRIKES_PER_EXPIRY,
            "min_common_strikes": MIN_COMMON_STRIKES,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        },
        "records": [records[tag] for tag in sorted(records)],
    })
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(manifest_path, serialized.encode("utf-8"))


def _record_from_existing_artifact(output_dir: Path, trade_date: str) -> dict:
    """Rebuild an ok manifest record from an orphaned artifact (manifest lost)."""
    path = artifact_path(output_dir, trade_date)
    data = path.read_bytes()
    artifact = json.loads(data.decode("utf-8"))
    return {
        "date": trade_date,
        "status": "ok",
        "reason": None,
        "detail": "rebuilt from artifact",
        "n_expiries": len(artifact.get("maturities", [])),
        "artifact_sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--spot-csv", type=Path, default=DEFAULT_SPOT_CSV,
                        help="CSI 1000 spot cache supplying s0 per trade date")
    parser.add_argument("--start", help="first trade date, YYYYMMDD (inclusive)")
    parser.add_argument("--end", help="last trade date, YYYYMMDD (inclusive)")
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel date builds via multiprocessing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild dates whose artifact already exists")
    args = parser.parse_args()

    if args.max_dates is not None and args.max_dates <= 0:
        raise SystemExit("--max-dates must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    for bound in (args.start, args.end):
        if bound is not None:
            stage01._parse_trade_date(bound)

    dates = []
    for path in sorted(args.csv_dir.glob("*_1.csv")):
        tag = path.name[: -len("_1.csv")]
        try:
            stage01._parse_trade_date(tag)
        except ValueError:
            continue
        if args.start and tag < args.start:
            continue
        if args.end and tag > args.end:
            continue
        dates.append(tag)
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    if not dates:
        raise SystemExit(f"no settlement CSVs selected in {args.csv_dir}")

    spots = load_spot_map(args.spot_csv)
    records = load_manifest_records(args.manifest)
    pending: list[tuple] = []
    for trade_date in dates:
        iso = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
        if artifact_path(args.output_dir, trade_date).is_file() and not args.force:
            if trade_date not in records:
                records[trade_date] = _record_from_existing_artifact(
                    args.output_dir, trade_date
                )
            print(f"{trade_date}: skip (artifact exists)")
            continue
        pending.append((trade_date, str(args.csv_dir), str(args.output_dir),
                        spots.get(iso)))

    t0 = time.perf_counter()
    results: list[dict] = []
    if pending and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for entry in pool.map(_build_one, pending):
                results.append(entry)
                print(
                    f"{entry['date']}: {entry['status']}"
                    + (f" ({entry['reason']})" if entry["reason"] else "")
                )
    else:
        for task in pending:
            started = time.perf_counter()
            entry = _build_one(task)
            results.append(entry)
            elapsed = time.perf_counter() - started
            print(
                f"{entry['date']}: {entry['status']}"
                + (f" ({entry['reason']})" if entry["reason"] else "")
                + f"  [{elapsed:.2f}s]"
            )
    for entry in results:
        records[entry["date"]] = entry

    # ``--start``/``--end`` select the incremental work for this invocation;
    # they must not shrink the manifest's provenance window to only that
    # incremental slice.  Derive the persisted window from the complete
    # record set after merging the new results.
    record_dates = sorted(records)
    window = {
        "start": record_dates[0] if record_dates else dates[0],
        "end": record_dates[-1] if record_dates else dates[-1],
    }
    save_manifest(args.manifest, records, window=window)
    n_ok = sum(1 for entry in results if entry["status"] == "ok")
    n_excluded = len(results) - n_ok
    elapsed = time.perf_counter() - t0
    print(
        f"built={n_ok} excluded={n_excluded} skipped={len(dates) - len(pending)} "
        f"in {elapsed:.1f}s -> {args.output_dir} (+ {args.manifest})"
    )


if __name__ == "__main__":
    main()
