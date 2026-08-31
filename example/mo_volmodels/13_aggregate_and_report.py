"""Stage 13 - Aggregate the vol-model backtest fleet and write the lecture report.

Consumes the per-run artifacts written by stage 12
(``12_snowball_volmodel_backtest.py``) and produces:

    aggregate.json          per-run metrics + per-variant distributions +
                            paired (same-inception) comparisons vs flat BSM
    per_run_metrics.csv     one row per inception x variant
    variant_summary.csv     one row per variant
    paired_vs_flat_bsm.csv  one row per inception x variant (flat BSM excluded)
    volmodel_backtest_lecture.html   the explanatory report

Why paired comparisons: the five variants of one inception share an identical
contract and an identical market path, so their PnL difference is entirely
attributable to the pricing/hedging model.  Comparing pooled distributions
instead would be dominated by which inceptions happened to knock out.

Run:
    .venv/bin/python example/mo_volmodels/13_aggregate_and_report.py \
        --run-dir output/volmodel_backtest
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import statistics
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quantark.util.numerical import is_zero

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_ROOT / "output/volmodel_backtest"

BASELINE_VARIANT = "flat_bsm"
VARIANT_ORDER = ("flat_bsm", "ts_bsm", "localvol", "heston", "heston_slv")
VARIANT_LABELS = {
    "flat_bsm": "Flat BSM",
    "ts_bsm": "TS BSM",
    "localvol": "Local Vol",
    "heston": "Heston",
    "heston_slv": "Heston-SLV",
}
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Small numeric helpers (empty-safe; never fabricate a value)
# ---------------------------------------------------------------------------

def _finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = _finite(values)
    return float(statistics.fmean(vals)) if vals else None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else None


def _stdev(values: Sequence[float]) -> Optional[float]:
    vals = _finite(values)
    return float(statistics.stdev(vals)) if len(vals) > 1 else None


def _rms(values: Sequence[float]) -> Optional[float]:
    vals = _finite(values)
    if not vals:
        return None
    return float(math.sqrt(statistics.fmean([v * v for v in vals])))


def _load_certificate_transfer():
    """Import the sibling transfer module (the stages are not a package)."""
    path = Path(__file__).resolve().parent / "certificate_transfer.py"
    spec = importlib.util.spec_from_file_location("mo_certificate_transfer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_certificate_transfer"] = module
    spec.loader.exec_module(module)
    return module


certificate_transfer = _load_certificate_transfer()

_STAGE12 = None


def stage12():
    """Import the fleet runner lazily -- it pulls the whole engine stack in.

    Stage 13 needs exactly one thing from it: the scope declaration, so a
    manifest written BEFORE that declaration existed still reports its
    declared-out-of-scope cells as declared rather than as failures.
    """
    global _STAGE12
    if _STAGE12 is None:
        path = Path(__file__).resolve().parent / "12_snowball_volmodel_backtest.py"
        spec = importlib.util.spec_from_file_location("mo_stage12_for_s13", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["mo_stage12_for_s13"] = module
        spec.loader.exec_module(module)
        _STAGE12 = module
    return _STAGE12


# A failure is only reclassified as a declared exclusion when its error is the
# one the declaration is ABOUT.  A declared cell that broke some other way is
# still a failure -- the declaration covers a known grid limit, not the cell.
GRID_SCOPE_ERROR_MARKER = "exceeds 2x target eps_crit"

# The variants that actually run an ADI 2-D solver, and so are the only ones
# the Greek certificate is about.  flat_bsm / ts_bsm / localvol never touch it.
ADI_CERTIFIED_VARIANTS = ("heston", "heston_slv")


# --- Feller regime -------------------------------------------------------
#
# The study calibrates with ``enforce_feller=True``, which makes the record's
# own ``feller_satisfied`` flag True BY CONSTRUCTION -- 257 of 257 fits in
# output/mo_daily_calibration/calibration_manifest.json.  Screening that flag
# can therefore only ever report "clean".  Three of those same 257 fits carry
# ratios of 7.9e3, 1.7e5 and 2.3e5 with sigma pinned at its 0.001 lower bound:
# sigma-collapse, where Heston has degenerated into a deterministic-variance
# model.  Spec section 7A.10(3) requires those dates be flagged, never
# averaged into a ``heston`` result -- so the screen reads the RATIO.
#
# The cut points mirror 11_pde_convergence_gate.py and are measured, not
# chosen: 0.5 separates a real Feller violation from an enforced fit sitting
# on the boundary at ratio ~= 1.0; 10 is the sigma-collapse marker.  Stage 11
# is an implementation input to the Stage 16 certification hash, so it is not
# refactored to export them here; the two copies are kept honest by
# test_stage13_agrees_with_stage11_on_the_feller_cut_points.
FELLER_VIOLATED_BELOW = 0.5
FELLER_DEGENERATE_ABOVE = 10.0
FELLER_BUCKETS = ("violated", "boundary", "degenerate", "unknown")


def feller_bucket(ratio: Optional[float]) -> str:
    """Bucket a Heston Feller ratio (2*kappa*theta/sigma**2) into a regime.

    ``None`` or non-finite -- an uncomputable ratio, or a variant such as
    ``flat_bsm`` / ``localvol`` that never carries one -- buckets as
    "unknown", never as "boundary": a record whose regime cannot be
    determined must not read as the common, passing case.
    """
    if ratio is None or not math.isfinite(float(ratio)):
        return "unknown"
    ratio = float(ratio)
    if ratio < FELLER_VIOLATED_BELOW:
        return "violated"
    if ratio > FELLER_DEGENERATE_ABOVE:
        return "degenerate"
    return "boundary"


def record_feller_ratio(record: Dict[str, Any]) -> Optional[float]:
    """The Feller ratio a calibration record implies, or None.

    ``heston`` records carry ``feller_ratio`` outright.  ``heston_slv``
    records do not: they carry the Heston fit NESTED, as the five raw
    parameters under ``heston``.  SLV inherits that fit, so it inherits its
    regime -- reading only the top level would leave sigma-collapse invisible
    for one of the two certified 2-D variants.  The calibrator's own ratio
    wins where it exists and is usable; the nested parameters are the
    fallback, never an override.  One malformed record must not abort the
    aggregation of a fleet that cost ~143 CPU-hours, so an unusable value
    ranks as "unknown" rather than raising.
    """
    stated = record.get("feller_ratio")
    if stated is not None:
        try:
            value = float(stated)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return value
    nested = record.get("heston")
    if not isinstance(nested, dict):
        return None
    try:
        kappa = float(nested["kappa"])
        theta = float(nested["theta"])
        sigma = float(nested["sigma"])
    except (KeyError, TypeError, ValueError):
        return None
    # sigma == 0 sits outside the calibration's own lower bound (0.001 under
    # the mo_frozen preset), so it cannot come from a real fit.  If one ever
    # appears, "unknown" is the honest answer: the regime could not be ranked.
    if not (sigma > 0.0) or not math.isfinite(kappa * theta):
        return None
    return 2.0 * kappa * theta / (sigma * sigma)


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = _finite(values)
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "median": _median(vals),
        "stdev": _stdev(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def run_dir_for(root: Path, inception: str, variant: str) -> Path:
    return Path(root) / "runs" / inception / variant


def load_run_frames(run_dir: Path) -> Dict[str, Any]:
    """Load one run's artifacts. Missing optional frames come back empty."""
    def read(name: str) -> pd.DataFrame:
        path = run_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, index_col=0)

    calibration: List[Dict[str, Any]] = []
    cal_path = run_dir / "calibration_records.json"
    if cal_path.exists():
        loaded = json.loads(cal_path.read_text())
        if isinstance(loaded, list):
            calibration = loaded

    summary: Dict[str, Any] = {}
    sum_path = run_dir / "run_summary.json"
    if sum_path.exists():
        summary = json.loads(sum_path.read_text())

    return {
        "states": read("states.csv"),
        "greeks": read("greeks.csv"),
        "trades": read("trades.csv"),
        "rebalances": read("rebalances.csv"),
        "calibration": calibration,
        "summary": summary,
    }


def _col(frame: pd.DataFrame, name: str) -> List[float]:
    if frame.empty or name not in frame.columns:
        return []
    return [float(v) for v in frame[name].tolist()]


def calibration_quality(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise per-day model fit: RMSE, bound hits, Feller regime, leverage.

    Reported honestly - the repo's own diagnostics show Heston is weakly
    identified on public CFFEX settlement data, so bound hits and the Feller
    regime are surfaced rather than hidden.

    The Feller screen ranks the RATIO, not the record's ``feller_satisfied``
    flag: these fits run with ``enforce_feller=True``, which makes that flag
    true by construction, so screening it can only ever report "clean" while
    sigma-collapse dates pass straight through.  See the cut points above.
    """
    if not records:
        return {"n_records": 0}
    rmse = [r.get("overall_rmse_iv") for r in records if r.get("overall_rmse_iv") is not None]
    bound_hits = [r for r in records if r.get("bound_hits")]
    ratios = [record_feller_ratio(r) for r in records]
    buckets = Counter(feller_bucket(x) for x in ratios)
    # Fractions are over the records that actually carry a ratio; a localvol
    # run that carries none must report None, not a reassuring zero.
    n_ranked = sum(buckets[b] for b in ("violated", "boundary", "degenerate"))
    breaches = sum(1 for r in records if r.get("feller_satisfied") is False)
    lev_min = [r.get("leverage_min") for r in records if r.get("leverage_min") is not None]
    lev_max = [r.get("leverage_max") for r in records if r.get("leverage_max") is not None]
    neg_mass = [
        r.get("max_negative_mass") for r in records
        if r.get("max_negative_mass") is not None
    ]
    return {
        "n_records": len(records),
        "n_unique_surfaces": len({r.get("surface_sha") for r in records if r.get("surface_sha")}),
        "rmse_iv": _distribution(rmse),
        "n_bound_hits": len(bound_hits),
        "bound_hit_fraction": len(bound_hits) / len(records) if records else None,
        "feller_ratio": _distribution(ratios),
        "feller_buckets": {name: buckets[name] for name in FELLER_BUCKETS},
        "n_feller_violated": buckets["violated"],
        "feller_violated_fraction": (
            buckets["violated"] / n_ranked if n_ranked else None
        ),
        "n_sigma_collapse": buckets["degenerate"],
        "sigma_collapse_fraction": (
            buckets["degenerate"] / n_ranked if n_ranked else None
        ),
        # Under enforce_feller=True this must be 0.  If it is not, the
        # enforcement did not take on that date, and that is its own finding.
        "n_enforcement_breaches": breaches,
        "leverage_min": min(_finite(lev_min)) if _finite(lev_min) else None,
        "leverage_max": max(_finite(lev_max)) if _finite(lev_max) else None,
        "max_negative_mass": max(_finite(neg_mass)) if _finite(neg_mass) else None,
    }


def certificate_span_audit(
    *, variant: str, frames: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Coverage of this run's visited states by the banked Greek certificate.

    Returns None for a variant that never runs an ADI 2-D solver -- the
    certificate is not about it, and an empty audit would read as "checked
    and clean".  Report-only: see certificate_transfer for why an aggregate
    mean-bias admission cannot be decomposed into per-date permissions.
    """
    if variant not in ADI_CERTIFIED_VARIANTS:
        return None
    records = frames.get("calibration") or []
    maturity = _parse_day((frames.get("summary") or {}).get("maturity_date"))
    states = []
    for record in records:
        day = _parse_day(record.get("date") or record.get("surface_date"))
        if day is None or maturity is None:
            # Cannot place the state on the maturity axis; state_in_span
            # fails closed on a non-finite remaining maturity.
            remaining = float("nan")
        else:
            remaining = (maturity - day).days / 365.25
        params = record if "kappa" in record else (record.get("heston") or {})
        states.append((str(record.get("date") or record.get("surface_date")), params, remaining))
    return certificate_transfer.audit(states)


def _parse_day(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value)[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def _residual(
    total: Optional[float],
    plus: Sequence[Optional[float]],
    minus: Sequence[Optional[float]] = (),
) -> Optional[float]:
    """``total - (sum(plus) - sum(minus))``, or None if any term is missing.

    Returning None rather than 0.0 for an incomplete decomposition matters:
    a missing term must never be reported as a satisfied identity.
    """
    terms = [total, *plus, *minus]
    if any(t is None or not math.isfinite(float(t)) for t in terms):
        return None
    return float(total) - (
        sum(float(p) for p in plus) - sum(float(m) for m in minus)
    )


def metrics_for_run(
    *, inception: str, variant: str, notional: float, frames: Dict[str, Any]
) -> Dict[str, Any]:
    """Per-run metrics: PnL decomposition, hedge quality, greek path, costs."""
    states = frames["states"]
    greeks = frames["greeks"]
    trades = frames["trades"]
    summary = frames["summary"]

    total_pnl = _col(states, "total_pnl")
    product_pnl = _col(states, "product_pnl")
    hedge_pnl = _col(states, "hedge_pnl")
    costs = _col(states, "transaction_costs")
    product_mtm = _col(states, "product_mtm")
    cashflows = _col(states, "cashflows")

    # Hedge quality: residual position delta AFTER the day's rebalance,
    # expressed in cash per 1% spot move so it is comparable across spot levels.
    residual_delta_cash = _col(greeks, "post_hedge_delta_cash_1pct")
    pre_hedge_delta_cash = _col(greeks, "pre_hedge_delta_cash_1pct")
    gamma_cash = _col(greeks, "gamma_cash_1pct")
    position_delta = _col(greeks, "product_position_delta")

    traded_contracts = [abs(v) for v in _col(trades, "quantity")]
    traded_notional = [abs(v) for v in _col(trades, "notional")]

    final_pnl = total_pnl[-1] if total_pnl else None
    final_costs = costs[-1] if costs else None

    # The total admits TWO independent decompositions.  They are different cuts
    # of the same number, not additive parts of one list -- mixing terms across
    # them double-counts, because every Cut-B term that lands after day 1 is
    # already inside Cut A's hedging half.
    #
    # Cut A -- BY TIME:      total = inception + hedging
    #
    # The contract is booked at initial_product_price=0 and every variant
    # prices the SAME contract -- one whose coupon was solved so that flat BSM
    # values it at zero (Gate G4).  So day 1 carries each model's disagreement
    # with that solve, marked instantly: ~0 for flat_bsm by construction, and
    # whatever the model thinks for everyone else.  That is a one-off valuation
    # opinion, not hedging skill, and blending it into the total masks the
    # thing this study is asking about -- the two components can carry opposite
    # signs for the same variant.  ``total_pnl`` is cumulative, so the hedging
    # component is simply what accrued after day 1.
    inception_pnl = total_pnl[0] if total_pnl else None
    hedging_pnl = (
        final_pnl - inception_pnl
        if final_pnl is not None and inception_pnl is not None
        else None
    )
    inception_mark = product_pnl[0] if product_pnl else None

    # Cut B -- BY COMPONENT: total = open mark + cashflows + hedge - costs
    #
    # Read straight off the final row of the cumulative ledger, whose per-row
    # identities sanity_check_run already enforces.  The terms answer "where
    # did the money come from", where Cut A answers "when was it booked".
    #
    # ``open_mark`` is the contract still marked to model at the end.  It goes
    # to EXACTLY zero once the trade terminates, because settlement moves the
    # value from product_mtm into cashflows without touching their sum -- so
    # the KO coupon is not income arriving, it is the mark already carried
    # turning into cash.  A non-zero open_mark flags a run whose PnL is still
    # an opinion (censored at data end) rather than a realized outcome.
    open_mark = product_mtm[-1] if product_mtm else None
    realized_cashflows = cashflows[-1] if cashflows else None
    final_hedge_pnl = hedge_pnl[-1] if hedge_pnl else None

    def pct_notional(value: Optional[float]) -> Optional[float]:
        if value is None or not notional:
            return None
        return 100.0 * value / float(notional)

    # Each cut is an identity, so its residual is a self-check on the emitted
    # numbers rather than on the ledger sanity_check_run already validated: if
    # a term were ever dropped or mis-signed here, this is what would show it.
    time_residual = _residual(final_pnl, [inception_pnl, hedging_pnl])
    component_residual = _residual(
        final_pnl, [open_mark, realized_cashflows, final_hedge_pnl], minus=[final_costs]
    )

    return {
        "inception": inception,
        "variant": variant,
        "n_days": int(len(states)),
        "notional": float(notional),
        "lifecycle": summary.get("lifecycle", {}),
        "coupon": summary.get("coupon"),
        "vol_model_solver": summary.get("vol_model_solver"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        # --- PnL ---
        "total_pnl": final_pnl,
        "total_pnl_pct_notional": pct_notional(final_pnl),
        # --- Cut A, by time: total = inception + hedging (exactly) ---
        "pnl_inception": inception_pnl,
        "pnl_inception_pct_notional": pct_notional(inception_pnl),
        "pnl_inception_mark": inception_mark,
        "pnl_inception_mark_pct_notional": pct_notional(inception_mark),
        "pnl_hedging": hedging_pnl,
        "pnl_hedging_pct_notional": pct_notional(hedging_pnl),
        # --- Cut B, by component: total = mark + cashflows + hedge - costs ---
        "pnl_open_mark": open_mark,
        "pnl_open_mark_pct_notional": pct_notional(open_mark),
        "pnl_cashflows": realized_cashflows,
        "pnl_cashflows_pct_notional": pct_notional(realized_cashflows),
        "pnl_hedge_pct_notional": pct_notional(final_hedge_pnl),
        "pnl_decomposition_residual": {
            "time": time_residual,
            "component": component_residual,
        },
        "product_pnl": product_pnl[-1] if product_pnl else None,
        "hedge_pnl": final_hedge_pnl,
        "transaction_costs": final_costs,
        "cost_drag_pct_notional": pct_notional(final_costs),
        "pnl_path_stdev": _stdev(total_pnl),
        "pnl_max_drawdown": _max_drawdown(total_pnl),
        # --- hedge quality ---
        "residual_delta_cash_rms": _rms(residual_delta_cash),
        "residual_delta_cash_rms_pct_notional": pct_notional(_rms(residual_delta_cash)),
        "residual_delta_cash_max_abs": max((abs(v) for v in _finite(residual_delta_cash)), default=None),
        "pre_hedge_delta_cash_rms": _rms(pre_hedge_delta_cash),
        # --- greek path ---
        "position_delta": _distribution(position_delta),
        "gamma_cash_1pct": _distribution(gamma_cash),
        # --- turnover / costs ---
        "n_trades": int(len(trades)),
        "traded_contracts_total": sum(traded_contracts) if traded_contracts else 0.0,
        "traded_notional_total": sum(traded_notional) if traded_notional else 0.0,
        # --- model fit ---
        "calibration": calibration_quality(frames["calibration"]),
        "certificate_span": certificate_span_audit(variant=variant, frames=frames),
    }


def _max_drawdown(series: Sequence[float]) -> Optional[float]:
    vals = _finite(series)
    if not vals:
        return None
    peak = vals[0]
    worst = 0.0
    for v in vals:
        peak = max(peak, v)
        worst = min(worst, v - peak)
    return float(worst)


# ---------------------------------------------------------------------------
# Output completeness (Gate: design doc section 7)
# ---------------------------------------------------------------------------

# Each category names the artifact that must exist, be non-empty, and carry
# the listed columns.  Categories are checked per run so a partially-written
# run is reported rather than quietly averaged into the results.
REQUIRED_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "market_data": {"file": "states.csv", "columns": ["spot", "futures_price", "basis_yield"]},
    "implied_vol_and_q": {"file": "states.csv", "columns": ["volatility", "implied_q", "pricing_q"]},
    "position_info": {
        "file": "states.csv",
        "columns": ["futures_contracts", "alive", "knocked_in", "knocked_out"],
    },
    "pnl_path": {
        "file": "states.csv",
        "columns": ["total_pnl", "product_pnl", "hedge_pnl", "transaction_costs"],
    },
    "greeks_path": {
        "file": "greeks.csv",
        "columns": ["price", "delta", "gamma", "post_hedge_delta_cash_1pct"],
    },
    "trading_records": {"file": "trades.csv", "columns": ["quantity", "price", "transaction_cost"],
                        "may_be_empty": True},
}

# Categories only meaningful for the calibrated variants.
#
# The surface diagnostic is named per variant, not shared: localvol records the
# Dupire surface it prices on (lv_min/lv_max), while heston_slv records the
# LEVERAGE surface multiplying its Heston backbone (leverage_min/leverage_max).
# Demanding lv_min/lv_max from an SLV record asks for a field the variant never
# writes, which reads as "incomplete run" when the run is fine.
CALIBRATED_CATEGORIES = {
    "calibration_records": {"variants": {"localvol", "heston", "heston_slv"}},
    "lv_surface_records": {
        "variants": {"localvol"},
        "keys": ["lv_min", "lv_max"],
    },
    "leverage_surface_records": {
        "variants": {"heston_slv"},
        "keys": ["leverage_min", "leverage_max"],
    },
}


def verify_run_completeness(
    run_dir: Path,
    *,
    variant: str,
    expected_days: Optional[int] = None,
    window_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Check every daily output category the design promises for one run.

    ``trades.csv`` may legitimately be empty (a run can end before any
    rebalance clears the rounding threshold), so it is checked for schema
    rather than row count.  Everything else must have one row per replay day.

    ``expected_days`` is the number of days actually REPLAYED, not the number
    of days in the window: a run that knocks out stops there, so its states.csv
    is legitimately shorter than the window.  Every run in this study knocks
    out, so comparing against the window length would report every single one
    as incomplete.  ``window_days`` is kept as the upper bound -- a replay can
    never produce more rows than the window holds.
    """
    issues: List[str] = []
    present: Dict[str, bool] = {}
    frames: Dict[str, pd.DataFrame] = {}

    for category, rule in REQUIRED_CATEGORIES.items():
        path = run_dir / rule["file"]
        if not path.exists():
            issues.append(f"{category}: missing {rule['file']}")
            present[category] = False
            continue
        if rule["file"] not in frames:
            frames[rule["file"]] = pd.read_csv(path, index_col=0)
        frame = frames[rule["file"]]
        if frame.empty and not rule.get("may_be_empty"):
            issues.append(f"{category}: {rule['file']} has no rows")
            present[category] = False
            continue
        missing_cols = [c for c in rule["columns"] if c not in frame.columns]
        if missing_cols:
            issues.append(f"{category}: {rule['file']} missing columns {missing_cols}")
        present[category] = not missing_cols

    n_days = len(frames.get("states.csv", pd.DataFrame()))
    if expected_days is not None and n_days != expected_days:
        issues.append(f"states.csv has {n_days} rows, run summary claims {expected_days}")
    if window_days is not None and n_days > window_days:
        issues.append(
            f"states.csv has {n_days} rows, more than the {window_days} days in "
            "the replay window"
        )
    greeks = frames.get("greeks.csv")
    if greeks is not None and not greeks.empty and n_days and len(greeks) != n_days:
        issues.append(f"greeks.csv has {len(greeks)} rows but states.csv has {n_days}")

    cal_path = run_dir / "calibration_records.json"
    records: List[Dict[str, Any]] = []
    if cal_path.exists():
        loaded = json.loads(cal_path.read_text())
        records = loaded if isinstance(loaded, list) else []

    for category, rule in CALIBRATED_CATEGORIES.items():
        if variant not in rule["variants"]:
            present[category] = True  # not applicable to this variant
            continue
        if not records:
            issues.append(f"{category}: no calibration records for variant {variant}")
            present[category] = False
            continue
        keys = rule.get("keys")
        if keys and not any(all(k in r for k in keys) for r in records):
            issues.append(f"{category}: no record carries {keys}")
            present[category] = False
            continue
        present[category] = True

    return {
        "run_dir": str(run_dir),
        "variant": variant,
        "n_days": n_days,
        "categories": present,
        "issues": issues,
        "complete": not issues,
    }


def verify_fleet_completeness(
    run_dir: Path, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the completeness check over every run in the manifest."""
    checks = []
    sanity = []
    for run in manifest.get("runs", []):
        d = run_dir_for(run_dir, run["inception"], run["variant"])
        metrics = run.get("metrics") or {}
        replayed = metrics.get("days_replayed", metrics.get("num_days"))
        checks.append(
            verify_run_completeness(
                d,
                variant=run["variant"],
                expected_days=replayed,
                window_days=run.get("n_days"),
            )
        )
        report = sanity_check_run(d)
        report["inception"] = run["inception"]
        report["variant"] = run["variant"]
        sanity.append(report)
    incomplete = [c for c in checks if not c["complete"]]
    insane = [s for s in sanity if not s["sane"]]
    return {
        "n_runs_checked": len(checks),
        "n_complete": len(checks) - len(incomplete),
        "n_incomplete": len(incomplete),
        "all_complete": not incomplete,
        "incomplete": incomplete,
        "n_sane": len(sanity) - len(insane),
        "all_sane": not insane,
        "sanity_failures": insane,
    }


# ---------------------------------------------------------------------------
# Run sanity invariants (Gate G3)
# ---------------------------------------------------------------------------

def sanity_check_run(run_dir: Path, *, rel_tol: float = 1e-9) -> Dict[str, Any]:
    """Check the accounting identities a correct replay must satisfy.

    Completeness (``verify_run_completeness``) asks whether the outputs are
    THERE; this asks whether they are CONSISTENT.  Every check below is an
    identity the engine constructs by definition, so any violation is a real
    defect rather than a tolerance question:

      * PnL decomposition   total = product + hedge - costs
      * portfolio identity  value = product_mtm + hedge_mtm + cash
      * cash identity       cash = cashflows - costs
      * cost reconciliation cumulative costs == sum of per-trade costs
      * position tracking   futures_contracts == cumulative traded quantity
      * lifecycle monotone  a dead trade never comes back to life
      * event corroboration a knocked-out run has a KO action row
      * hedge effectiveness post-hedge |delta| <= pre-hedge |delta|
    """
    issues: List[str] = []
    checks: Dict[str, Any] = {}

    states_path = run_dir / "states.csv"
    if not states_path.exists():
        return {"run_dir": str(run_dir), "issues": ["states.csv missing"], "sane": False}
    states = pd.read_csv(states_path, index_col=0, parse_dates=True)
    greeks_path = run_dir / "greeks.csv"
    greeks = (
        pd.read_csv(greeks_path, index_col=0, parse_dates=True)
        if greeks_path.exists() else pd.DataFrame()
    )
    trades_path = run_dir / "trades.csv"
    trades = (
        pd.read_csv(trades_path, index_col=0, parse_dates=True)
        if trades_path.exists() else pd.DataFrame()
    )
    actions_path = run_dir / "actions.csv"
    actions = (
        pd.read_csv(actions_path, index_col=0, parse_dates=True)
        if actions_path.exists() else pd.DataFrame()
    )

    def has(frame: pd.DataFrame, *cols: str) -> bool:
        return not frame.empty and all(c in frame.columns for c in cols)

    def worst_abs(series) -> float:
        return float(pd.Series(series).abs().max()) if len(series) else 0.0

    # --- dates ---
    if not states.index.is_monotonic_increasing:
        issues.append("states.csv dates are not increasing")
    if states.index.duplicated().any():
        issues.append("states.csv has duplicate dates")

    scale = max(1.0, float(pd.Series(states.get("portfolio_value", [1.0])).abs().max()))

    # --- PnL decomposition ---
    if has(states, "total_pnl", "product_pnl", "hedge_pnl", "transaction_costs"):
        resid = (
            states["total_pnl"]
            - (states["product_pnl"] + states["hedge_pnl"] - states["transaction_costs"])
        )
        checks["pnl_identity_max_abs"] = worst_abs(resid)
        if worst_abs(resid) > rel_tol * scale:
            issues.append(
                f"PnL identity violated by up to {worst_abs(resid):,.6f} "
                "(total != product + hedge - costs)"
            )

    # --- portfolio value / cash identities ---
    if has(states, "portfolio_value", "product_mtm", "hedge_mtm", "cash"):
        resid = states["portfolio_value"] - (
            states["product_mtm"] + states["hedge_mtm"] + states["cash"]
        )
        checks["portfolio_identity_max_abs"] = worst_abs(resid)
        if worst_abs(resid) > rel_tol * scale:
            issues.append(
                f"portfolio identity violated by up to {worst_abs(resid):,.6f}"
            )
    if has(states, "cash", "cashflows", "transaction_costs"):
        resid = states["cash"] - (states["cashflows"] - states["transaction_costs"])
        checks["cash_identity_max_abs"] = worst_abs(resid)
        if worst_abs(resid) > rel_tol * scale:
            issues.append(f"cash identity violated by up to {worst_abs(resid):,.6f}")

    # --- transaction costs ---
    if has(states, "transaction_costs"):
        costs = states["transaction_costs"]
        if (costs.diff().dropna() < -rel_tol * scale).any():
            issues.append("cumulative transaction costs decrease on some day")
        if has(trades, "transaction_cost"):
            booked = float(trades["transaction_cost"].sum())
            final = float(costs.iloc[-1])
            checks["cost_reconciliation_gap"] = abs(final - booked)
            if abs(final - booked) > max(1e-6, rel_tol * scale):
                issues.append(
                    f"cost reconciliation: states {final:,.4f} vs trades {booked:,.4f}"
                )

    # --- futures position tracking ---
    if has(states, "futures_contracts") and has(trades, "quantity"):
        traded = float(trades["quantity"].sum())
        final_pos = float(states["futures_contracts"].iloc[-1])
        checks["position_tracking_gap"] = abs(final_pos - traded)
        if abs(final_pos - traded) > 1e-6:
            issues.append(
                f"position tracking: final {final_pos:,.4f} contracts vs "
                f"{traded:,.4f} traded"
            )

    # --- lifecycle monotonicity ---
    for flag in ("knocked_out", "knocked_in", "matured"):
        if flag in states.columns:
            series = states[flag].astype(bool).astype(int)
            if (series.diff().dropna() < 0).any():
                issues.append(f"{flag} flag turns back off after being set")
    if "alive" in states.columns:
        alive = states["alive"].astype(bool).astype(int)
        if (alive.diff().dropna() > 0).any():
            issues.append("a dead trade came back to life")

    # --- event corroboration ---
    if "knocked_out" in states.columns and bool(states["knocked_out"].iloc[-1]):
        if actions.empty or "action_type" not in actions.columns:
            issues.append("run knocked out but has no action log")
        elif not (actions["action_type"].astype(str).str.upper() == "KO").any():
            issues.append("run knocked out but no KO action row was recorded")

    # --- hedge effectiveness ---
    if has(greeks, "pre_hedge_delta_cash_1pct", "post_hedge_delta_cash_1pct"):
        pre = greeks["pre_hedge_delta_cash_1pct"].abs()
        post = greeks["post_hedge_delta_cash_1pct"].abs()
        worse = int((post > pre + 1e-6).sum())
        checks["days_hedge_increased_delta"] = worse
        checks["post_hedge_delta_rms"] = float((post ** 2).mean() ** 0.5)
        checks["pre_hedge_delta_rms"] = float((pre ** 2).mean() ** 0.5)
        if worse:
            issues.append(
                f"hedging increased |delta| on {worse} day(s) - rebalance sign error?"
            )

    # --- no NaNs where they would be silent ---
    for col in ("total_pnl", "spot", "futures_price", "portfolio_value"):
        if col in states.columns and states[col].isna().any():
            issues.append(f"states.csv column {col!r} contains NaN")

    return {
        "run_dir": str(run_dir),
        "n_days": int(len(states)),
        "checks": checks,
        "issues": issues,
        "sane": not issues,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def variant_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pooled distribution of the headline metrics for one variant."""
    lifecycles = [r.get("lifecycle") or {} for r in rows]
    return {
        "n_runs": len(rows),
        "pnl_pct_notional": _distribution([r["total_pnl_pct_notional"] for r in rows]),
        "pnl_inception_pct_notional": _distribution(
            [r["pnl_inception_pct_notional"] for r in rows]
        ),
        "pnl_hedging_pct_notional": _distribution(
            [r["pnl_hedging_pct_notional"] for r in rows]
        ),
        # Cut B -- by component.  cost_drag is the cut's fourth term (it enters
        # with a minus sign), so it is not repeated under another name.
        "pnl_open_mark_pct_notional": _distribution(
            [r["pnl_open_mark_pct_notional"] for r in rows]
        ),
        "pnl_cashflows_pct_notional": _distribution(
            [r["pnl_cashflows_pct_notional"] for r in rows]
        ),
        "pnl_hedge_pct_notional": _distribution(
            [r["pnl_hedge_pct_notional"] for r in rows]
        ),
        "cost_drag_pct_notional": _distribution(
            [r["cost_drag_pct_notional"] for r in rows]
        ),
        "residual_delta_cash_rms_pct_notional": _distribution(
            [r["residual_delta_cash_rms_pct_notional"] for r in rows]
        ),
        "pnl_max_drawdown": _distribution([r["pnl_max_drawdown"] for r in rows]),
        "n_trades": _distribution([r["n_trades"] for r in rows]),
        "gamma_cash_mean": _distribution(
            [(r["gamma_cash_1pct"] or {}).get("mean") for r in rows]
        ),
        "lifecycle_counts": {
            "knocked_out": sum(1 for lc in lifecycles if lc.get("knocked_out")),
            "knocked_in": sum(1 for lc in lifecycles if lc.get("knocked_in")),
            "matured": sum(1 for lc in lifecycles if lc.get("matured")),
            "censored_at_data_end": sum(
                1 for lc in lifecycles if lc.get("censored_at_data_end")
            ),
        },
        "calibration": _pooled_calibration([r.get("calibration") or {} for r in rows]),
        "total_elapsed_seconds": sum(
            _finite([r.get("elapsed_seconds") for r in rows])
        ),
    }


def _pooled_calibration(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    with_records = [e for e in entries if e.get("n_records")]
    if not with_records:
        return {"n_records": 0}
    return {
        "n_records": sum(int(e["n_records"]) for e in with_records),
        "rmse_iv_mean": _mean(
            [(e.get("rmse_iv") or {}).get("mean") for e in with_records]
        ),
        "rmse_iv_max": max(
            _finite([(e.get("rmse_iv") or {}).get("max") for e in with_records]),
            default=None,
        ),
        "bound_hit_fraction": _mean(
            [e.get("bound_hit_fraction") for e in with_records]
        ),
        "feller_violated_fraction": _mean(
            [e.get("feller_violated_fraction") for e in with_records]
        ),
        "sigma_collapse_fraction": _mean(
            [e.get("sigma_collapse_fraction") for e in with_records]
        ),
        "n_sigma_collapse": sum(
            int(e.get("n_sigma_collapse") or 0) for e in with_records
        ),
        "feller_ratio_max": max(
            _finite([(e.get("feller_ratio") or {}).get("max") for e in with_records]),
            default=None,
        ),
        "n_enforcement_breaches": sum(
            int(e.get("n_enforcement_breaches") or 0) for e in with_records
        ),
    }


def paired_comparisons(
    per_run: Sequence[Dict[str, Any]], baseline: str = BASELINE_VARIANT
) -> List[Dict[str, Any]]:
    """Same-inception differences vs the baseline variant.

    Each inception's five variants share one contract and one market path, so
    a paired difference isolates the model.  Inceptions where the baseline run
    is missing are skipped (and counted by the caller), never imputed.
    """
    by_key = {(r["inception"], r["variant"]): r for r in per_run}
    inceptions = sorted({r["inception"] for r in per_run})
    out: List[Dict[str, Any]] = []
    for inception in inceptions:
        base = by_key.get((inception, baseline))
        if base is None:
            continue
        for variant in sorted({r["variant"] for r in per_run}):
            if variant == baseline:
                continue
            row = by_key.get((inception, variant))
            if row is None:
                continue
            out.append(
                {
                    "inception": inception,
                    "variant": variant,
                    "baseline": baseline,
                    "d_pnl_pct_notional": _diff(
                        row["total_pnl_pct_notional"], base["total_pnl_pct_notional"]
                    ),
                    # The two halves of that total, paired separately.  They can
                    # carry OPPOSITE signs for the same variant -- a model that
                    # marks the contract up on day 1 and then hedges it worse --
                    # so the split is what answers "does this model hedge
                    # better", which the blended figure cannot.
                    "d_pnl_inception_pct_notional": _diff(
                        row["pnl_inception_pct_notional"],
                        base["pnl_inception_pct_notional"],
                    ),
                    "d_pnl_hedging_pct_notional": _diff(
                        row["pnl_hedging_pct_notional"],
                        base["pnl_hedging_pct_notional"],
                    ),
                    # Cut B, paired.  The contract's own cashflows are set by
                    # the REALIZED index path, so within one inception they are
                    # identical across variants and this difference collapses to
                    # zero -- as does the open mark once every arm terminates.
                    # That is not an assumption: these columns are carried so
                    # the collapse is visible, leaving the paired edge as
                    # (hedge - cost) whenever it holds, and flagging the
                    # censored runs where it does not.
                    "d_pnl_open_mark_pct_notional": _diff(
                        row["pnl_open_mark_pct_notional"],
                        base["pnl_open_mark_pct_notional"],
                    ),
                    "d_pnl_cashflows_pct_notional": _diff(
                        row["pnl_cashflows_pct_notional"],
                        base["pnl_cashflows_pct_notional"],
                    ),
                    "d_pnl_hedge_pct_notional": _diff(
                        row["pnl_hedge_pct_notional"], base["pnl_hedge_pct_notional"]
                    ),
                    "d_cost_drag_pct_notional": _diff(
                        row["cost_drag_pct_notional"], base["cost_drag_pct_notional"]
                    ),
                    "d_residual_delta_rms_pct_notional": _diff(
                        row["residual_delta_cash_rms_pct_notional"],
                        base["residual_delta_cash_rms_pct_notional"],
                    ),
                    "d_n_trades": _diff(row["n_trades"], base["n_trades"]),
                    "same_lifecycle": (
                        (row.get("lifecycle") or {}).get("knocked_out")
                        == (base.get("lifecycle") or {}).get("knocked_out")
                    ),
                }
            )
    return out


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return None
    return float(a) - float(b)


def paired_summary(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-variant summary of the paired edge vs baseline, with a win rate."""
    out: Dict[str, Any] = {}
    for variant in sorted({p["variant"] for p in pairs}):
        rows = [p for p in pairs if p["variant"] == variant]
        deltas = _finite([p["d_pnl_pct_notional"] for p in rows])
        hedge_deltas = _finite([p["d_residual_delta_rms_pct_notional"] for p in rows])
        hedging_pnl = _finite([p["d_pnl_hedging_pct_notional"] for p in rows])
        out[variant] = {
            "n_pairs": len(rows),
            "d_pnl_pct_notional": _distribution(deltas),
            "d_pnl_inception_pct_notional": _distribution(
                _finite([p["d_pnl_inception_pct_notional"] for p in rows])
            ),
            "d_pnl_hedging_pct_notional": _distribution(hedging_pnl),
            # Win rate on the hedging half alone: the sign test for "does this
            # model hedge better", free of the day-1 mark.
            "pnl_hedging_win_rate": (
                sum(1 for d in hedging_pnl if d > 0.0) / len(hedging_pnl)
                if hedging_pnl else None
            ),
            # Cut B, paired.  ``contract_terms_max_abs`` is the largest paired
            # difference in the two contract-side terms; when it is zero the
            # paired edge provably reduces to hedge minus cost.
            "d_pnl_open_mark_pct_notional": _distribution(
                _finite([p["d_pnl_open_mark_pct_notional"] for p in rows])
            ),
            "d_pnl_cashflows_pct_notional": _distribution(
                _finite([p["d_pnl_cashflows_pct_notional"] for p in rows])
            ),
            "d_pnl_hedge_pct_notional": _distribution(
                _finite([p["d_pnl_hedge_pct_notional"] for p in rows])
            ),
            "contract_terms_max_abs": max(
                (
                    abs(v)
                    for v in _finite(
                        [p["d_pnl_open_mark_pct_notional"] for p in rows]
                        + [p["d_pnl_cashflows_pct_notional"] for p in rows]
                    )
                ),
                default=None,
            ),
            "d_residual_delta_rms_pct_notional": _distribution(hedge_deltas),
            "d_cost_drag_pct_notional": _distribution(
                [p["d_cost_drag_pct_notional"] for p in rows]
            ),
            # Fraction of inceptions where the variant beat the baseline.
            "pnl_win_rate": (
                sum(1 for d in deltas if d > 0.0) / len(deltas) if deltas else None
            ),
            "hedge_win_rate": (
                sum(1 for d in hedge_deltas if d < 0.0) / len(hedge_deltas)
                if hedge_deltas else None
            ),
        }
    return out


def pooled_certificate_span(per_run: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fleet-level coverage: which dates left the certified regime span.

    ``covered`` is None when no ADI-certified variant ran -- nothing measured
    is not the same as nothing wrong.
    """
    audits = [
        (r["variant"], r["certificate_span"])
        for r in per_run
        if r.get("certificate_span")
    ]
    if not audits:
        return {"n_states": 0, "n_out_of_span": 0, "covered": None, "variants": []}
    rows = [row for _, a in audits for row in a["out_of_span"]]
    return {
        "n_states": sum(a["n_states"] for _, a in audits),
        "n_out_of_span": len(rows),
        "dates_out_of_span": sorted({str(row["label"]) for row in rows}),
        "variants": sorted({v for v, _ in audits}),
        "buckets": {
            name: sum(a["buckets"].get(name, 0) for _, a in audits)
            for name in certificate_transfer.BUCKETS
        },
        "feller_ratio_max": max(
            _finite([(a["feller_ratio"] or {}).get("max") for _, a in audits]),
            default=None,
        ),
        "certificate": audits[0][1]["certificate"],
        "reasons": sorted({str(row["reason"]) for row in rows}),
        "covered": not rows,
    }


def resolve_scope(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Split declared-out-of-scope cells from genuine failures.

    A fleet run after the declaration lands records its own
    ``scope_exclusions`` and never attempts those cells.  A manifest written
    BEFORE it instead carries them as failures -- reporting those as breakage
    would be wrong, and quietly dropping them would be worse.  So failures are
    matched against the declaration on BOTH the cell and the error class: a
    declared cell that failed some other way stays a failure.
    """
    declared = manifest.get("scope_exclusions")
    records: List[Dict[str, Any]] = list(declared or [])
    try:
        declaration = stage12().SCOPE_EXCLUSIONS
    except Exception:  # pragma: no cover - stage 12 unavailable in isolation
        declaration = ()
    keys = {
        (exclusion.inception, variant): exclusion
        for exclusion in declaration
        for variant in exclusion.variants
    }
    already = {(str(r.get("inception")), str(r.get("variant"))) for r in records}

    failures: List[Dict[str, Any]] = []
    reclassified = 0
    for failure in manifest.get("failures", []):
        key = (str(failure.get("inception")), str(failure.get("variant")))
        exclusion = keys.get(key)
        matches = exclusion is not None and GRID_SCOPE_ERROR_MARKER in str(
            failure.get("error", "")
        )
        if not matches or key in already:
            failures.append(failure)
            continue
        records.append(
            {
                "inception": exclusion.inception,
                "variant": failure.get("variant"),
                "reason": exclusion.reason,
                "achieved_spacing": exclusion.achieved,
                "target_eps_crit": exclusion.target,
                "reclassified_from_failure": True,
            }
        )
        already.add(key)
        reclassified += 1

    counts = dict(manifest.get("counts", {}))
    expected = counts.get("runs_expected")
    counts["runs_out_of_scope"] = len(records)
    if expected is not None:
        counts["runs_in_scope"] = int(expected) - len(records)
    counts["runs_failed"] = len(failures)
    return {
        "scope_exclusions": records,
        "failures": failures,
        "counts": counts,
        "n_reclassified_from_failure": reclassified,
    }


def variant_coverage(
    per_run: Sequence[Dict[str, Any]],
    *,
    scope_exclusions: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Which inceptions each variant actually covers, and whether that differs.

    Declared scope exclusions make coverage UNEVEN, and uneven coverage is the
    one thing that silently invalidates a pooled mean: variants would then be
    averaged over different inceptions, so a difference between their means
    could be entirely the sample rather than the model.  Paired comparisons are
    immune (they only ever use inceptions both arms ran), which is why the
    report leans on them -- but the pooled table has to say so.
    """
    by_variant: Dict[str, set] = {}
    for row in per_run:
        by_variant.setdefault(row["variant"], set()).add(row["inception"])
    all_inceptions = set().union(*by_variant.values()) if by_variant else set()
    excluded: Dict[str, List[str]] = {}
    for record in scope_exclusions:
        excluded.setdefault(str(record.get("variant")), []).append(
            str(record.get("inception"))
        )
    full = {v: sorted(all_inceptions - set(covered)) for v, covered in by_variant.items()}
    return {
        "n_inceptions_any_variant": len(all_inceptions),
        "variants": {
            variant: {
                "n_covered": len(covered),
                "missing_inceptions": full[variant],
                "excluded_inceptions": sorted(excluded.get(variant, [])),
            }
            for variant, covered in sorted(by_variant.items())
        },
        # True only when every variant ran on the same inceptions, which is the
        # precondition for comparing pooled means directly.
        "uniform": all(not missing for missing in full.values()),
    }


def pnl_decomposition_audit(per_run: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fleet-level check that both cuts of the PnL still add up.

    Both are identities, so the only interesting number is the worst residual.
    ``n_unchecked`` counts runs where a term was missing entirely -- those are
    reported, never silently treated as satisfying the identity.
    """
    residuals = [r.get("pnl_decomposition_residual") or {} for r in per_run]

    def worst(key: str) -> Dict[str, Any]:
        values = [d.get(key) for d in residuals]
        checked = _finite([v for v in values if v is not None])
        return {
            "n_checked": len(checked),
            "n_unchecked": len(values) - len(checked),
            "max_abs_residual": max((abs(v) for v in checked), default=None),
        }

    # Terminated trades settle their mark to exactly zero, so a non-zero open
    # mark is the marker of a censored run whose PnL is still an opinion.
    open_marks = [r.get("pnl_open_mark") for r in per_run]
    return {
        "time_cut": worst("time"),
        "component_cut": worst("component"),
        "n_runs_with_open_mark": sum(
            1 for v in open_marks if v is not None and not is_zero(float(v))
        ),
        "n_runs": len(per_run),
    }


def aggregate(run_dir: Path) -> Dict[str, Any]:
    """Read the fleet manifest and build every comparison table."""
    manifest_path = Path(run_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no run manifest at {manifest_path}; run stage 12 first "
            "(12_snowball_volmodel_backtest.py)"
        )
    manifest = json.loads(manifest_path.read_text())
    notional = float(manifest.get("config", {}).get("notional", 0.0))

    per_run: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for run in manifest.get("runs", []):
        inception, variant = run["inception"], run["variant"]
        d = run_dir_for(run_dir, inception, variant)
        if not (d / "states.csv").exists():
            missing.append({"inception": inception, "variant": variant})
            continue
        frames = load_run_frames(d)
        per_run.append(
            metrics_for_run(
                inception=inception, variant=variant, notional=notional, frames=frames
            )
        )

    variants = sorted(
        {r["variant"] for r in per_run},
        key=lambda v: VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99,
    )
    summaries = {
        v: variant_summary([r for r in per_run if r["variant"] == v]) for v in variants
    }
    pairs = paired_comparisons(per_run)
    scope = resolve_scope(manifest)

    return {
        "schema_version": SCHEMA_VERSION,
        "study": "snowball_volmodel_backtest_aggregate",
        "run_dir": str(run_dir),
        "manifest_counts": scope["counts"],
        "config": manifest.get("config", {}),
        "term_sheet": manifest.get("term_sheet", {}),
        "hedge_costs": manifest.get("hedge_costs", {}),
        "gate_g2": manifest.get("gate_g2", {}),
        "adi_greek_certification": manifest.get("adi_greek_certification", {}),
        "certificate_span": pooled_certificate_span(per_run),
        "inceptions": manifest.get("inceptions", []),
        "variants": variants,
        "per_run": per_run,
        "variant_summary": summaries,
        "paired_vs_baseline": pairs,
        "paired_summary": paired_summary(pairs),
        "pnl_decomposition": pnl_decomposition_audit(per_run),
        "scope_exclusions": scope["scope_exclusions"],
        "coverage": variant_coverage(
            per_run, scope_exclusions=scope["scope_exclusions"]
        ),
        "missing_runs": missing,
        "completeness": verify_fleet_completeness(Path(run_dir), manifest),
        "failures": scope["failures"],
    }


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------

def write_tables(agg: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    rows = []
    for r in agg["per_run"]:
        lc = r.get("lifecycle") or {}
        rows.append(
            {
                "inception": r["inception"],
                "variant": r["variant"],
                "n_days": r["n_days"],
                "coupon": r.get("coupon"),
                "solver": r.get("vol_model_solver"),
                "total_pnl": r["total_pnl"],
                "pnl_pct_notional": r["total_pnl_pct_notional"],
                # Cut A (by time): pnl_pct_notional == inception + hedging.
                "pnl_inception_pct_notional": r["pnl_inception_pct_notional"],
                "pnl_hedging_pct_notional": r["pnl_hedging_pct_notional"],
                # Cut B (by component): pnl_pct_notional == open_mark
                #   + cashflows + hedge - cost_drag.  A DIFFERENT cut of the
                #   same total, not more parts of Cut A -- both cashflows and
                #   most of the hedge sit inside pnl_hedging above.
                "pnl_open_mark_pct_notional": r["pnl_open_mark_pct_notional"],
                "pnl_cashflows_pct_notional": r["pnl_cashflows_pct_notional"],
                "pnl_hedge_pct_notional": r["pnl_hedge_pct_notional"],
                "product_pnl": r["product_pnl"],
                "hedge_pnl": r["hedge_pnl"],
                "transaction_costs": r["transaction_costs"],
                "cost_drag_pct_notional": r["cost_drag_pct_notional"],
                "pnl_max_drawdown": r["pnl_max_drawdown"],
                "residual_delta_rms_pct_notional": r[
                    "residual_delta_cash_rms_pct_notional"
                ],
                "gamma_cash_mean": (r.get("gamma_cash_1pct") or {}).get("mean"),
                "n_trades": r["n_trades"],
                "knocked_out": lc.get("knocked_out"),
                "knocked_in": lc.get("knocked_in"),
                "matured": lc.get("matured"),
                "censored": lc.get("censored_at_data_end"),
                "calib_rmse_iv_mean": (
                    (r.get("calibration") or {}).get("rmse_iv") or {}
                ).get("mean"),
                "calib_bound_hit_fraction": (r.get("calibration") or {}).get(
                    "bound_hit_fraction"
                ),
                "elapsed_seconds": r.get("elapsed_seconds"),
            }
        )
    per_run = pd.DataFrame(rows).sort_values(["inception", "variant"])
    paths["per_run"] = out_dir / "per_run_metrics.csv"
    per_run.to_csv(paths["per_run"], index=False)

    srows = []
    for variant, s in agg["variant_summary"].items():
        srows.append(
            {
                "variant": variant,
                "n_runs": s["n_runs"],
                "pnl_pct_mean": s["pnl_pct_notional"]["mean"],
                # Cut A: the two halves of pnl_pct_mean, by time.
                "pnl_inception_pct_mean": s["pnl_inception_pct_notional"]["mean"],
                "pnl_hedging_pct_mean": s["pnl_hedging_pct_notional"]["mean"],
                # Cut B: the four terms of pnl_pct_mean, by component
                # (cost_drag_pct_mean below is the fourth, subtracted).
                "pnl_open_mark_pct_mean": s["pnl_open_mark_pct_notional"]["mean"],
                "pnl_cashflows_pct_mean": s["pnl_cashflows_pct_notional"]["mean"],
                "pnl_hedge_pct_mean": s["pnl_hedge_pct_notional"]["mean"],
                "pnl_pct_median": s["pnl_pct_notional"]["median"],
                "pnl_pct_stdev": s["pnl_pct_notional"]["stdev"],
                "pnl_pct_min": s["pnl_pct_notional"]["min"],
                "pnl_pct_max": s["pnl_pct_notional"]["max"],
                "cost_drag_pct_mean": s["cost_drag_pct_notional"]["mean"],
                "residual_delta_rms_pct_mean": s[
                    "residual_delta_cash_rms_pct_notional"
                ]["mean"],
                "n_trades_mean": s["n_trades"]["mean"],
                "knocked_out": s["lifecycle_counts"]["knocked_out"],
                "knocked_in": s["lifecycle_counts"]["knocked_in"],
                "matured": s["lifecycle_counts"]["matured"],
                "censored": s["lifecycle_counts"]["censored_at_data_end"],
                "calib_rmse_iv_mean": s["calibration"].get("rmse_iv_mean"),
                "calib_bound_hit_fraction": s["calibration"].get("bound_hit_fraction"),
                "hours": (s["total_elapsed_seconds"] or 0.0) / 3600.0,
            }
        )
    paths["variant_summary"] = out_dir / "variant_summary.csv"
    pd.DataFrame(srows).to_csv(paths["variant_summary"], index=False)

    paths["paired"] = out_dir / "paired_vs_flat_bsm.csv"
    pd.DataFrame(agg["paired_vs_baseline"]).to_csv(paths["paired"], index=False)
    return paths


# ---------------------------------------------------------------------------
# Lecture-style HTML report
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "&mdash;"
    return f"{value:,.{digits}f}{suffix}"


def _num_cells(values: Sequence[Any], digits: int = 3, suffix: str = "") -> str:
    return "".join(f'<td class="num">{_fmt(v, digits, suffix)}</td>' for v in values)


def _scope_section(agg: Dict[str, Any]) -> str:
    """Name every declared-out-of-scope cell, with the measurement behind it."""
    records = agg.get("scope_exclusions", []) or []
    if not records:
        return ""
    by_inception: Dict[str, Dict[str, Any]] = {}
    for record in records:
        entry = by_inception.setdefault(
            str(record.get("inception")),
            {
                "variants": [],
                "achieved": record.get("achieved_spacing"),
                "target": record.get("target_eps_crit"),
            },
        )
        entry["variants"].append(
            VARIANT_LABELS.get(str(record.get("variant")), str(record.get("variant")))
        )
    rows = "".join(
        f"<tr><td>{inception}</td><td>{', '.join(sorted(e['variants']))}</td>"
        f'<td class="num">{_fmt(e["achieved"], 5)}</td>'
        f'<td class="num">{_fmt(e["target"], 5)}</td>'
        f'<td class="num">{_fmt((e["achieved"] or 0.0) / (e["target"] or 1.0), 2, "&times;")}</td></tr>'
        for inception, e in sorted(by_inception.items())
    )
    reasons = sorted({str(r.get("reason")) for r in records if r.get("reason")})
    # Say which actually happened.  On a fleet that predates the declaration
    # these cells were attempted and failed closed; a later fleet skips them
    # outright.  Both are the same decision, but only one of them is "never
    # priced", and claiming the stronger one would be false.
    n_reclassified = sum(
        1 for r in records if r.get("reclassified_from_failure")
    )
    if n_reclassified == len(records):
        provenance = (
            f"{len(records)} cell(s) are <b>excluded by declaration</b>. This fleet "
            "predates the declaration, so it attempted them and the grid check "
            "<em>failed closed</em> &mdash; no price was produced, and none was "
            "estimated. A fleet run against the current declaration skips them "
            "outright."
        )
    elif n_reclassified:
        provenance = (
            f"{len(records)} cell(s) are <b>excluded by declaration</b>; "
            f"{len(records) - n_reclassified} were skipped outright and "
            f"{n_reclassified} were attempted by an earlier fleet and failed closed."
        )
    else:
        provenance = (
            f"{len(records)} cell(s) were <b>never priced</b>, by declaration rather "
            "than by failure."
        )
    return f"""
<h3>2.1 &nbsp; Cells declared out of scope</h3>
<p>{provenance} The 2-D ADI route builds its spatial grid at a resolution the 1-D profile's
target does not admit on these dates, Gate G5 does not sweep the 2-D arm, and Gate G2 certified
no finer configuration &mdash; so pricing them would have meant running on settings no gate
covers. That trades a visible gap for an invisible one, and the visible gap is the better
trade.</p>
<div class="tablewrap"><table><thead><tr><th>inception</th><th>variants</th>
<th>achieved spacing</th><th>target eps_crit</th><th>ratio</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="small">{"; ".join(reasons)}.</p>
"""


def _scope_caveat_bullet(agg: Dict[str, Any]) -> str:
    """The declared gap, restated where a reader looks for what is missing."""
    records = agg.get("scope_exclusions", []) or []
    if not records:
        return ""
    inceptions = sorted({str(r.get("inception")) for r in records})
    variants = sorted({str(r.get("variant")) for r in records})
    labels = ", ".join(VARIANT_LABELS.get(v, v) for v in variants)
    return (
        f"<li><b>A declared 2-D gap.</b> {labels} were never run on "
        f"{len(inceptions)} consecutive inceptions ({inceptions[0]} to "
        f"{inceptions[-1]}) &mdash; see &sect;2.1. This study therefore says "
        "nothing about how those models would have hedged that particular "
        "stretch of market, and because the stretch is contiguous rather than "
        "scattered, that is a gap in a specific regime, not a thinner sample "
        "of the same one.</li>\n"
    )


def _coverage_caveat(agg: Dict[str, Any]) -> str:
    """Warn when pooled means span variants that ran on different inceptions."""
    coverage = agg.get("coverage", {}) or {}
    if not coverage or coverage.get("uniform", True):
        return ""
    uneven = {
        variant: entry
        for variant, entry in (coverage.get("variants", {}) or {}).items()
        if entry.get("missing_inceptions")
    }
    if not uneven:
        return ""
    detail = "; ".join(
        f"<b>{VARIANT_LABELS.get(v, v)}</b> covers {e['n_covered']} of "
        f"{coverage.get('n_inceptions_any_variant', '?')} "
        f"(absent: {', '.join(e['missing_inceptions'])})"
        for v, e in sorted(uneven.items())
    )
    return (
        '<div class="callout key"><b>Coverage is uneven &mdash; do not compare the pooled '
        "means across variants.</b> " + detail + ". Those inceptions are consecutive, not a "
        "random subsample, so a pooled mean that includes them is averaging a different "
        "market period from one that does not. Every model comparison in &sect;4 is "
        "<b>paired</b>, which uses only inceptions both arms ran and is therefore immune; "
        "the pooled table above is a per-variant description, not a comparison.</div>"
    )


def _decomposition_note(decomp: Dict[str, Any]) -> str:
    """State both identities' worst residual, and how many marks are still open."""
    if not decomp:
        return ""
    time_cut = decomp.get("time_cut", {}) or {}
    comp_cut = decomp.get("component_cut", {}) or {}

    def worst(cut: Dict[str, Any]) -> str:
        value = cut.get("max_abs_residual")
        if value is None:
            return "not checked"
        return f"{value:.2e} (n={cut.get('n_checked', 0)})"

    unchecked = int(time_cut.get("n_unchecked", 0)) + int(
        comp_cut.get("n_unchecked", 0)
    )
    open_marks = int(decomp.get("n_runs_with_open_mark", 0))
    n_runs = int(decomp.get("n_runs", 0))
    bits = [
        f"worst residual &mdash; by time: <b>{worst(time_cut)}</b>, "
        f"by component: <b>{worst(comp_cut)}</b>, both in currency units"
    ]
    if unchecked:
        bits.append(
            f"<b>{unchecked}</b> run(s) had a term missing and could not be checked"
        )
    if open_marks:
        bits.append(
            f"<b>{open_marks}</b> of {n_runs} runs still carry an open mark "
            "(censored before termination), so their total is a mark-to-model "
            "figure rather than a realized one"
        )
    else:
        bits.append(
            f"all {n_runs} runs settled to a zero open mark, so every total "
            "shown is realized cash, not an opinion"
        )
    return (
        '<p class="small">Both rows are identities, checked per run: '
        + "; ".join(bits)
        + ".</p>"
    )


def _paired_collapse_note(agg: Dict[str, Any]) -> str:
    """Report whether the paired edge provably reduces to hedge minus cost."""
    paired = agg.get("paired_summary", {}) or {}
    values = [
        p.get("contract_terms_max_abs")
        for p in paired.values()
        if p.get("contract_terms_max_abs") is not None
    ]
    if not values:
        return ""
    worst = max(float(v) for v in values)
    if is_zero(worst):
        verdict = (
            "Across every pair those two differences are <b>exactly zero</b>, so the "
            "paired edge in the table above is <b>entirely hedge PnL minus trading "
            "cost</b> &mdash; arithmetic, not an assumption. That is the strongest "
            "statement this study can make that it is measuring hedging and nothing "
            "else."
        )
    else:
        verdict = (
            f"The largest such difference here is <b>{worst:.2e}</b> percent of "
            "notional, which is not zero &mdash; some pair did not share a contract "
            "outcome (a censored run leaves an open mark that each model values "
            "differently), so that much of the paired edge is a valuation "
            "disagreement rather than hedging."
        )
    return (
        '<div class="callout key"><b>What the pairing removes.</b> Under Cut B the '
        "two contract-side terms &mdash; the coupon actually paid and any mark still "
        "open &mdash; are set by the <em>realized index path</em>, which every variant "
        "of one inception shares. " + verdict + "</div>"
    )


def build_report(agg: Dict[str, Any]) -> str:
    cfg = agg.get("config", {})
    term = agg.get("term_sheet", {})
    costs = agg.get("hedge_costs", {})
    gate = agg.get("gate_g2", {})
    greek_gate = agg.get("adi_greek_certification", {})
    counts = agg.get("manifest_counts", {})
    variants = agg["variants"]
    summaries = agg["variant_summary"]
    paired = agg["paired_summary"]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    incomplete_banner = ""
    # Judge completion against what was IN SCOPE.  Declared exclusions are not
    # a partial result -- they are a stated boundary, and flagging them red
    # would train the reader to ignore the banner that marks real breakage.
    n_expected = counts.get("runs_in_scope", counts.get("runs_expected"))
    n_done = counts.get("runs_completed")
    missing_variants = [v for v in VARIANT_ORDER if v not in variants]
    certification_exclusions = greek_gate.get("excluded_variants", {}) or {}
    if missing_variants or (n_expected and n_done and n_done < n_expected):
        bits = []
        if missing_variants:
            bits.append(
                "variants not in this run: <b>"
                + ", ".join(VARIANT_LABELS.get(v, v) for v in missing_variants)
                + "</b>"
            )
        if certification_exclusions:
            bits.append(
                "fail-closed ADI Greek exclusions: <b>"
                + ", ".join(
                    f"{VARIANT_LABELS.get(v, v)} ({reason})"
                    for v, reason in certification_exclusions.items()
                )
                + "</b>"
            )
        if n_expected and n_done and n_done < n_expected:
            bits.append(f"<b>{n_done}/{n_expected}</b> runs completed")
        incomplete_banner = (
            '<div style="background:#b5432f;color:#fff;padding:.7rem 1rem;'
            'border-radius:8px;margin:1rem 0;font-weight:600">'
            "&#9888; PARTIAL RESULT &mdash; " + "; ".join(bits) + ". Conclusions below "
            "cover only what actually ran.</div>"
        )

    # --- table: per-variant PnL distribution -------------------------------
    dist_rows = ""
    for v in variants:
        s = summaries[v]
        p = s["pnl_pct_notional"]
        dist_rows += (
            f"<tr><td><b>{VARIANT_LABELS.get(v, v)}</b></td>"
            f'<td class="num">{s["n_runs"]}</td>'
            + _num_cells([p["mean"], p["median"], p["stdev"], p["min"], p["max"]], 3)
            + _num_cells([s["cost_drag_pct_notional"]["mean"]], 4)
            + _num_cells([s["residual_delta_cash_rms_pct_notional"]["mean"]], 4)
            + _num_cells([s["n_trades"]["mean"]], 1)
            + "</tr>"
        )

    # --- table: the two decompositions of the same total --------------------
    decomp_rows = ""
    for v in variants:
        s = summaries[v]
        decomp_rows += (
            f"<tr><td><b>{VARIANT_LABELS.get(v, v)}</b></td>"
            f'<td class="num">{s["n_runs"]}</td>'
            # Cut A
            + _num_cells(
                [
                    s["pnl_inception_pct_notional"]["mean"],
                    s["pnl_hedging_pct_notional"]["mean"],
                ],
                4,
            )
            # Cut B
            + _num_cells(
                [
                    s["pnl_open_mark_pct_notional"]["mean"],
                    s["pnl_cashflows_pct_notional"]["mean"],
                    s["pnl_hedge_pct_notional"]["mean"],
                    -(s["cost_drag_pct_notional"]["mean"] or 0.0),
                ],
                4,
            )
            + _num_cells([s["pnl_pct_notional"]["mean"]], 4)
            + "</tr>"
        )
    decomp = agg.get("pnl_decomposition", {}) or {}
    decomp_note = _decomposition_note(decomp)

    # --- table: paired edge vs flat BSM ------------------------------------
    paired_rows = ""
    for v in variants:
        if v == BASELINE_VARIANT or v not in paired:
            continue
        p = paired[v]
        d = p["d_pnl_pct_notional"]
        inc = p["d_pnl_inception_pct_notional"]
        hed = p["d_pnl_hedging_pct_notional"]
        h = p["d_residual_delta_rms_pct_notional"]
        win = p["pnl_win_rate"]
        hedge_pnl_win = p.get("pnl_hedging_win_rate")
        hwin = p["hedge_win_rate"]
        paired_rows += (
            f"<tr><td><b>{VARIANT_LABELS.get(v, v)}</b></td>"
            f'<td class="num">{p["n_pairs"]}</td>'
            + _num_cells([d["mean"], d["median"], d["stdev"]], 3)
            + f'<td class="num">{_fmt(100.0 * win, 1, "%") if win is not None else "&mdash;"}</td>'
            + _num_cells([inc["mean"], hed["mean"]], 3)
            + f'<td class="num">'
            + (
                _fmt(100.0 * hedge_pnl_win, 1, "%")
                if hedge_pnl_win is not None else "&mdash;"
            )
            + "</td>"
            + _num_cells([h["mean"]], 4)
            + f'<td class="num">{_fmt(100.0 * hwin, 1, "%") if hwin is not None else "&mdash;"}</td>'
            + "</tr>"
        )
    if not paired_rows:
        paired_rows = (
            '<tr><td colspan="11">No paired comparisons &mdash; the baseline '
            f"variant ({BASELINE_VARIANT}) is not in this run.</td></tr>"
        )

    # --- table: lifecycle outcomes ----------------------------------------
    life_rows = ""
    for v in variants:
        lc = summaries[v]["lifecycle_counts"]
        life_rows += (
            f"<tr><td><b>{VARIANT_LABELS.get(v, v)}</b></td>"
            f'<td class="num">{lc["knocked_out"]}</td>'
            f'<td class="num">{lc["knocked_in"]}</td>'
            f'<td class="num">{lc["matured"]}</td>'
            f'<td class="num">{lc["censored_at_data_end"]}</td></tr>'
        )

    # --- table: calibration quality ---------------------------------------
    # enforce_feller=True is the premise the sigma-collapse screen rests on:
    # it is what makes a record's own `feller_satisfied` flag uninformative
    # and the ratio authoritative.  A record that reports the constraint
    # unsatisfied means the enforcement did not take on that date, so the
    # premise failed and the banner says so rather than leaving it in JSON.
    breached = [
        (v, int(summaries[v]["calibration"].get("n_enforcement_breaches") or 0))
        for v in variants
        if int(summaries[v]["calibration"].get("n_enforcement_breaches") or 0)
    ]
    breach_banner = ""
    if breached:
        detail = "; ".join(
            f"{VARIANT_LABELS.get(v, v)}: {n} record(s)" for v, n in breached
        )
        breach_banner = (
            '<div style="background:#b5432f;color:#fff;padding:.7rem 1rem;'
            'border-radius:8px;margin:1rem 0;font-weight:600">'
            "&#9888; FELLER ENFORCEMENT BREACH &mdash; " + detail + ". These fits run "
            "with <code>enforce_feller=True</code>, so this should be zero; the "
            "constraint did not take on those dates and the regime columns below "
            "rest on a premise that failed.</div>"
        )

    # The ADI Greek certificate admits an AGGREGATE mean signed bias over
    # seven regime archetypes.  When the fleet visits states past the extremes
    # of that design, the aggregate result is being read outside the span it
    # was measured on.  Say so; do NOT gate -- see certificate_transfer.
    span = agg.get("certificate_span") or {}
    span_banner = ""
    if span.get("covered") is False:
        dates = span.get("dates_out_of_span") or []
        shown = ", ".join(dates[:6]) + (" and more" if len(dates) > 6 else "")
        cert = span.get("certificate") or {}
        envelope = cert.get("ratio_envelope") or [None, None]
        span_banner = (
            '<div style="background:#8a6d1f;color:#fff;padding:.7rem 1rem;'
            'border-radius:8px;margin:1rem 0;font-weight:600">'
            "&#9888; OUTSIDE THE CERTIFIED REGIME SPAN &mdash; "
            f"{span['n_out_of_span']} of {span['n_states']} day-cells, on "
            f"{len(dates)} date(s): {shown}. The banked ADI Greek certificate "
            "admits a <em>mean</em> signed delta bias across seven regime "
            "archetypes spanning Feller ratios "
            f"{_fmt(envelope[0], 3)}&ndash;{_fmt(envelope[1], 1)}; these states "
            "sit past those extremes, so the admitted bias is being read "
            "outside the design it was measured on. Nothing was gated &mdash; "
            "pricing is unaffected and these dates remain in the hedge path. "
            "Read the Heston / Heston-SLV result with this attached.</div>"
        )

    calib_rows = ""
    for v in variants:
        c = summaries[v]["calibration"]
        if not c.get("n_records"):
            continue
        calib_rows += (
            f"<tr><td><b>{VARIANT_LABELS.get(v, v)}</b></td>"
            f'<td class="num">{c["n_records"]:,}</td>'
            + _num_cells([c.get("rmse_iv_mean"), c.get("rmse_iv_max")], 5)
            + _num_cells(
                [
                    None if c.get("bound_hit_fraction") is None
                    else 100.0 * c["bound_hit_fraction"],
                    None if c.get("feller_violated_fraction") is None
                    else 100.0 * c["feller_violated_fraction"],
                    None if c.get("sigma_collapse_fraction") is None
                    else 100.0 * c["sigma_collapse_fraction"],
                ],
                1,
                "%",
            )
            + _num_cells([c.get("feller_ratio_max")], 4)
            + "</tr>"
        )
    calib_section = (
        f"""<h3>5.1 &nbsp; Per-day model fit</h3>
{breach_banner}
{span_banner}
<table><thead><tr><th>variant</th><th>records</th><th>mean RMSE (IV)</th>
<th>max RMSE (IV)</th><th>bound hits</th><th>Feller violated</th>
<th>&sigma;-collapse</th><th>max Feller ratio</th></tr></thead>
<tbody>{calib_rows}</tbody></table>
<div class="callout"><b>Read the bound hits honestly.</b> The repo's own diagnostics
(<code>example/mo_volmodels/MODEL_DIAGNOSTICS.md</code>) already showed Heston to be
<em>weakly identified</em> on public CFFEX settlement data &mdash; &kappa; and &sigma; hit their
frozen bounds on roughly half the sampled dates. A low IV RMSE therefore does <em>not</em>
mean the parameters are well determined; it means several very different parameter sets
fit the observed smile about equally well. Any Heston/SLV edge reported above should be
read with that caveat attached.</div>
<div class="callout"><b>Why &sigma;-collapse is reported separately.</b> These fits run with
<code>enforce_feller=True</code>, so each record&rsquo;s own <em>Feller satisfied</em> flag is true
<em>by construction</em> &mdash; 257 of 257 in the daily calibration pool. Screening that flag
can only ever say &ldquo;clean&rdquo;. The columns above therefore rank the Feller
<em>ratio</em> 2&kappa;&theta;/&sigma;&sup2; on the same measured cut points Gate&nbsp;G2 uses
(&lt;&nbsp;0.5 violated, &gt;&nbsp;10 &sigma;-collapse). Three of those 257 fits satisfy Feller
by driving &sigma; to its 0.001 lower bound, reaching ratios of 7.9e3 to 2.3e5 &mdash; Heston
degenerated into a <em>deterministic-variance</em> model. A non-zero &sigma;-collapse column
means that variant&rsquo;s result contains such dates, and they should be read out, not
averaged in.</div>"""
        if calib_rows
        else '<p class="lede">No calibrated variants in this run.</p>'
    )

    # --- inception table ---------------------------------------------------
    inception_rows = ""
    for entry in agg.get("inceptions", []):
        sol = entry.get("coupon_solution", {})
        inception_rows += (
            f"<tr><td>{entry['inception']}</td>"
            f'<td class="num">{_fmt(entry.get("initial_spot"), 2)}</td>'
            f'<td class="num">{_fmt(100.0 * entry["coupon"], 3, "%")}</td>'
            f'<td class="num">{_fmt(entry.get("atm_vol_at_inception"), 4)}</td>'
            f'<td class="num">{_fmt(entry.get("futures_implied_q"), 4)}</td>'
            f'<td class="num">{sol.get("iterations", "&mdash;")}</td>'
            f'<td class="num">{_fmt(abs(sol.get("pv", 0.0)), 2)}</td></tr>'
        )

    verdict = _verdict_paragraph(agg)
    outcome_caveat = _outcome_concentration_caveat(agg)
    paired_collapse = _paired_collapse_note(agg)
    scope_section = _scope_section(agg)
    coverage_caveat = _coverage_caveat(agg)
    scope_caveat = _scope_caveat_bullet(agg)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Does Vol-Model Sophistication Pay in Snowball Hedging?</title>
<style>
:root {{ --ink:#1a2230; --muted:#5a6577; --line:#e2e6ee; --accent:#1E3A5F; --accent2:#b5432f;
        --bg:#ffffff; --card:#f7f9fc; --code:#0d1b2a; }}
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;
       color:var(--ink); background:var(--bg); margin:0; line-height:1.62; }}
.wrap {{ max-width:960px; margin:0 auto; padding:2.5rem 1.4rem 5rem; }}
h1 {{ font-size:2rem; line-height:1.2; margin:.2rem 0 .4rem; }}
h2 {{ font-size:1.4rem; margin:2.6rem 0 .6rem; padding-top:1rem; border-top:2px solid var(--line); color:var(--accent); }}
h3 {{ font-size:1.08rem; margin:1.5rem 0 .3rem; color:var(--accent2); }}
.lede {{ color:var(--muted); font-size:1.05rem; }}
.meta {{ font-size:.85rem; color:var(--muted); background:var(--card); border:1px solid var(--line);
         border-radius:8px; padding:.7rem 1rem; margin:1rem 0 0; }}
p, li {{ font-size:.98rem; }}
code {{ font-family:"SF Mono",Menlo,Consolas,monospace; background:#eef1f6; padding:.05rem .3rem; border-radius:4px; font-size:.9em; }}
.eq {{ background:var(--code); color:#e6edf3; border-radius:8px; padding:.8rem 1.1rem; margin:.8rem 0;
       font-family:"SF Mono",Menlo,Consolas,monospace; font-size:.92rem; overflow-x:auto; }}
.eq .c {{ color:#7d8aa0; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.9rem; }}
th,td {{ border:1px solid var(--line); padding:.42rem .6rem; text-align:left; }}
th {{ background:var(--accent); color:#fff; font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:nth-child(even) td {{ background:var(--card); }}
.callout {{ border-left:4px solid var(--accent2); background:#fbf2ef; padding:.8rem 1.1rem; border-radius:0 8px 8px 0; margin:1.1rem 0; }}
.callout.key {{ border-color:var(--accent); background:#eef3f9; }}
.callout b {{ color:var(--accent2); }}
.callout.key b {{ color:var(--accent); }}
.toc {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1rem 1.3rem; margin:1.4rem 0; }}
.toc ol {{ margin:.3rem 0; padding-left:1.2rem; }}
.toc a {{ color:var(--accent); text-decoration:none; }}
footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line); font-size:.82rem; color:var(--muted); }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e6edf3; --muted:#9aa7b8; --line:#2b3648; --accent:#8fb4e0; --accent2:#e8927c;
           --bg:#0d131c; --card:#141d2b; --code:#060b12; }}
  code {{ background:#1c2634; }} .callout {{ background:#1b1512; }} .callout.key {{ background:#111a26; }}
}}
</style></head><body><div class="wrap">
{incomplete_banner}
<h1>Does vol-model sophistication pay in snowball hedging?</h1>
<p class="lede">A historical backtest of a <b>short 3-year CSI&nbsp;1000 snowball</b>
(中证1000, <code>000852.SH</code>), delta-hedged daily with IM index futures, priced and
hedged five different ways &mdash; flat Black-Scholes, a term-structure of ATM vol, Dupire
local volatility, Heston, and Heston-SLV &mdash; over overlapping monthly inceptions on
real CFFEX settlement data.</p>

<div class="meta">
{counts.get("inceptions", "?")} monthly inceptions &middot; {len(variants)} variants &middot;
{counts.get("runs_completed", "?")}/{counts.get("runs_expected", "?")} runs completed &middot;
notional {_fmt(cfg.get("notional"), 0)} CNY &middot; flat rate {_fmt(cfg.get("rate"), 4)} &middot;
generated {generated}
</div>

<div class="toc"><b>Contents</b>
<ol>
<li><a href="#s1">What is being tested, and why it is a fair test</a></li>
<li><a href="#s2">Data provenance and the honest gaps</a></li>
<li><a href="#s3">Results: PnL distribution across inceptions</a></li>
<li><a href="#s4">The paired edge over flat BSM</a></li>
<li><a href="#s5">Calibration quality &mdash; can we even trust the models?</a></li>
<li><a href="#s6">Caveats and what this study does not show</a></li>
</ol></div>

<h2 id="s1">1 &nbsp; What is being tested, and why it is a fair test</h2>
<p>A snowball is a short volatility, short skew, path-dependent position: the seller collects a
coupon for as long as the index stays between a knock-in floor and a knock-out ceiling, and takes
equity-like downside if the floor is breached and the trade never knocks out. Its delta is unstable
&mdash; it flips sign near the KO barrier and grows sharply near the KI barrier &mdash; so the
<em>model that produces the delta</em> matters in a way it never does for a vanilla option.</p>

<h3>1.1 &nbsp; The term sheet</h3>
<div class="tablewrap"><table><tbody>
<tr><th>Underlying</th><td>{term.get("underlying", "000852.SH")} (CSI 1000)</td></tr>
<tr><th>Position</th><td>Seller, quantity {term.get("product_quantity", -1.0)}, notional {_fmt(cfg.get("notional"), 0)} CNY</td></tr>
<tr><th>Tenor</th><td>{term.get("tenor_months", 36)} months</td></tr>
<tr><th>KO barrier</th><td>{_fmt(100.0 * float(term.get("ko_pct", 1.03)), 0, "%")} of initial spot, monthly, {term.get("lockout_months", 3)}-month lockout</td></tr>
<tr><th>KI barrier</th><td>{_fmt(100.0 * float(term.get("ki_pct", 0.75)), 0, "%")} of initial spot, observed every trading day</td></tr>
<tr><th>Coupon</th><td>{term.get("coupon", "solved per inception under flat BSM")}</td></tr>
<tr><th>Hedge</th><td>IM futures, daily rebalance at close, contract rounding, roll 5 days before expiry</td></tr>
<tr><th>Costs</th><td>{costs.get("model", "?")} &mdash; {_fmt(1e4 * float(costs.get("proportional_rate") or 0.0), 2, "bp")} commission, {_fmt(costs.get("spread_bps"), 1, "bp")} spread</td></tr>
</tbody></table></div>

<h3>1.2 &nbsp; The one design choice that makes this a controlled experiment</h3>
<p>The <b>fair coupon is solved once per inception under flat BSM</b>, and all five variants then
trade <em>that identical contract</em>. If each model priced its own fair coupon, the variants would
be trading different products and any PnL difference would confound model quality with term
generosity. Fixing the terms means every difference below is attributable to the model's
<em>delta</em>, not to what it charged.</p>
<div class="callout key"><b>Why the coupon solve is cheap.</b> Snowball PV is
<em>exactly affine</em> in the coupon: neither the KO trigger nor the KI trigger depends on it,
so the coupon leg and the rebate leg both scale linearly. False position therefore lands on the
fair coupon in a single step &mdash; three pricing calls instead of the forty a bisection needs.
The solver still prices the answer and checks it against tolerance, so the affine structure is
exploited but never trusted.</div>

<h3>1.3 &nbsp; Solved terms per inception</h3>
<div class="tablewrap"><table><thead><tr><th>inception</th><th>initial spot</th><th>fair coupon</th>
<th>ATM vol (3Y)</th><th>futures-implied q</th><th>solver iters</th><th>|PV| residual</th></tr></thead>
<tbody>{inception_rows}</tbody></table></div>

<h2 id="s2">2 &nbsp; Data provenance and the honest gaps</h2>
<p>Every surface is built from <b>official CFFEX end-of-day MO settlement files</b>, one per trading
date: put-call parity recovers the discount factor and forward per expiry, out-of-the-money quotes
are inverted to Black implied vols, and a SABR fit smooths the smile. Dates that fail the arbitrage
checks are <em>excluded and logged</em>, never patched &mdash; an excluded date reuses the previous
admitted surface under the manifest's carry-forward policy.</p>
<div class="callout"><b>The gap that matters most.</b> MO options list out to roughly one year,
but the trade runs three. Beyond the last listed expiry the surface is extrapolated
<b>flat in total variance</b>. So for the first two years of every trade, the long end of the
surface every model calibrates to is an <em>assumption</em>, not a quote. This affects all five
variants, but it bites the term-structure and stochastic-vol variants hardest, since they are
precisely the ones trying to exploit term-structure shape.</div>

<h3>2.1 &nbsp; Which engine priced which variant</h3>
<p>The Heston and Heston-SLV 2D-ADI PDE routes were put through a convergence gate against a
high-quality RQMC reference before being allowed near this backtest. The gate's verdict, which
this run reads from disk rather than assuming:</p>
<div class="eq">{json.dumps(gate.get("routes", {}), sort_keys=True)}
&nbsp;&nbsp;<span class="c">// evidence sha256: {str(gate.get("evidence_sha256"))[:16]}...</span></div>
<p>The 2-D variants also require the independent Stage-16 delta/gamma decision:</p>
<div class="eq">excluded = {json.dumps(certification_exclusions, sort_keys=True)}
&nbsp;&nbsp;<span class="c">// Greek evidence sha256:
{str(greek_gate.get("evidence_sha256"))[:16]}...</span></div>
<p>A Heston or Heston-SLV run appears only when the PV gate and Greek gate both admit PDE.
An unresolved 2-D estimator is excluded and shown above; it is never replaced with a noisy
daily Monte Carlo hedge. The tables below therefore report only variants with complete numerical
admission evidence.</p>
{scope_section}

<h2 id="s3">3 &nbsp; Results: PnL distribution across inceptions</h2>
<p>All figures are <b>percent of notional</b>, from the seller's perspective, net of hedging costs.
Residual delta is the RMS of the position's leftover delta <em>after</em> each day's rebalance,
expressed as cash per 1% spot move &mdash; the direct measure of hedge quality.</p>
<div class="tablewrap"><table><thead><tr><th>variant</th><th>runs</th><th>mean PnL</th>
<th>median</th><th>stdev</th><th>min</th><th>max</th><th>cost drag</th>
<th>residual &delta; RMS</th><th>trades</th></tr></thead>
<tbody>{dist_rows}</tbody></table></div>
{coverage_caveat}

<h3>3.1 &nbsp; Where the PnL comes from &mdash; two cuts of the same total</h3>
<p>The total admits two decompositions. They are <b>different cuts of one number</b>, not two
halves of a longer list: each row below sums to the same total on the right, along a different
seam. <b>Cut A</b> asks <em>when</em> the money was booked; <b>Cut B</b> asks <em>where it came
from</em>.</p>
<div class="tablewrap"><table><thead>
<tr><th rowspan="2">variant</th><th rowspan="2">runs</th>
<th colspan="2">Cut A &mdash; by time</th><th colspan="4">Cut B &mdash; by component</th>
<th rowspan="2">= total</th></tr>
<tr><th>inception</th><th>hedging</th><th>open mark</th><th>cashflows</th><th>hedge</th>
<th>costs</th></tr></thead>
<tbody>{decomp_rows}</tbody></table></div>
<p class="small">Every column is a <b>signed contribution</b>, so each cut adds straight across to
the total on the right &mdash; costs appear negative because they are money spent.</p>
{decomp_note}
<div class="callout key"><b>Do not add across the cuts.</b> Every Cut-B term that lands after day 1
&mdash; the coupon, the hedge, all but the first day's costs &mdash; is <em>already inside</em> Cut
A's hedging column. Summing an inception PnL, a hedging PnL, a coupon and a cost double-counts most
of the trade.</div>
<div class="callout key"><b>The knock-out coupon is not income.</b> It is the largest single number
in the ledger, and it is <em>not</em> a source of profit: settlement moves value out of
<code>product_mtm</code> and into <code>cashflows</code> without changing their sum, so the PnL
does not jump on the coupon date. The engine had accrued that value every day beforehand. This is
why <b>open mark</b> reads zero for every trade that terminated &mdash; the mark, having started at
the day-1 valuation and ended at nothing, contributes <em>exactly zero</em> over a completed life.
Cut A's split of it therefore measures <em>timing</em>, not profit: it shows when a model booked
value it would later have to give back. For a terminated trade the economics reduce to three terms:
<b>coupon paid, hedge earned, costs spent</b>.</div>

<h3>3.2 &nbsp; Lifecycle outcomes</h3>
<div class="tablewrap"><table><thead><tr><th>variant</th><th>knocked out</th><th>knocked in</th>
<th>matured</th><th>censored at data end</th></tr></thead>
<tbody>{life_rows}</tbody></table></div>
<p>Lifecycle counts are near-identical across variants by construction &mdash; KO and KI are
triggered by the <em>realized index path</em>, not by anybody's model. Differences between variants
therefore live entirely in the hedge, which is exactly what this study wants to isolate.</p>
{outcome_caveat}

<h2 id="s4">4 &nbsp; The paired edge over flat BSM</h2>
<p>Because all five variants of one inception share a contract and a market path, the honest
comparison is <b>paired</b>: for each inception, subtract the flat-BSM result. Pooled averages
would instead be dominated by which inceptions happened to knock out early.</p>
<div class="tablewrap"><table><thead><tr><th>variant</th><th>pairs</th><th>mean &Delta;PnL</th>
<th>median &Delta;PnL</th><th>stdev</th><th>PnL win rate</th>
<th>&Delta; inception</th><th>&Delta; hedging</th><th>hedging win rate</th>
<th>mean &Delta;residual &delta;</th><th>hedge win rate</th></tr></thead>
<tbody>{paired_rows}</tbody></table></div>
<div class="callout key"><b>Inception vs hedging &mdash; read these before the total.</b>
&Delta;PnL splits along Cut A (&sect;3.1) into two parts that mean different things. Every variant prices the
SAME contract, whose coupon was solved so that flat BSM values it at zero (Gate G4), and the
contract is booked at zero &mdash; so <em>day 1</em> marks each model's disagreement with that
solve instantly. That is a one-off valuation opinion, not hedging skill. <b>&Delta; inception</b>
is that day-1 mark; <b>&Delta; hedging</b> is everything accrued afterwards, over hundreds of
daily rebalances, and it is the column that answers &ldquo;does this model hedge better?&rdquo;
The two can carry <em>opposite signs</em> for the same variant &mdash; a model that marks the
contract up on day 1 and then gives it back while hedging &mdash; in which case the blended total
understates both effects.</div>
<div class="callout key"><b>How to read the two win rates.</b> <em>PnL win rate</em> is the fraction
of inceptions where the variant made more money than flat BSM. <em>Hedge win rate</em> is the
fraction where it left <em>less</em> residual delta. They can disagree, and when they do it is
informative: a model can hedge more tightly and still lose money if the tighter hedge costs more
in turnover than the risk it removes.</div>
{paired_collapse}
{verdict}

<h2 id="s5">5 &nbsp; Calibration quality &mdash; can we even trust the models?</h2>
{calib_section}

<h2 id="s6">6 &nbsp; Caveats and what this study does not show</h2>
<ul>
<li><b>Long-end extrapolation.</b> Two of every three years of each trade sit beyond the last
listed MO expiry, on a flat-total-variance extrapolation. A study on a market with 3-year listed
options could reach a different conclusion.</li>
<li><b>Censoring.</b> {counts.get("censored_at_data_end", "?")} of {counts.get("runs_completed", "?")}
runs reached the end of the data before knocking out or maturing. Their PnL is a mark-to-market
snapshot, not a realized outcome, and they are over-weighted toward the expensive early life of
the trade.</li>
<li><b>One underlying, one term sheet, one regime.</b> Every inception is the same product on the
same index over an overlapping window, so the runs are <em>not</em> independent observations. The
spread across inceptions understates true sampling uncertainty.</li>
{scope_caveat}<li><b>Heston identification.</b> See &sect;5: parameters that fit the smile are not
necessarily determined by it.</li>
<li><b>Monte Carlo noise in the SV deltas.</b> Heston and Heston-SLV deltas come from bumped MC
reprices. Randomized-QMC with common random numbers keeps bump noise far below raw price noise,
but it is not zero, and it does not shrink the way a PDE delta's discretization error does.</li>
</ul>

<footer>
Generated {generated} &middot; source
<code>example/mo_volmodels/12_snowball_volmodel_backtest.py</code> (fleet) and
<code>example/mo_volmodels/13_aggregate_and_report.py</code> (this report) &middot;
engines from <code>quantark.volmodels</code> / <code>quantark.asset.equity.engine</code> &middot;
market data: official CFFEX MO settlement files via AKShare.
</footer>
</div></body></html>"""


def _outcome_concentration_caveat(agg: Dict[str, Any]) -> str:
    """Warn when the sample contains only one kind of terminal outcome.

    A snowball's model risk is largest for a trade that is knocked IN at
    maturity and settles into equity downside.  If the realized window never
    produced that path, the study simply cannot speak to it, and the report
    has to say so rather than let the reader assume otherwise.
    """
    per_run = agg.get("per_run") or []
    baseline = [r for r in per_run if r["variant"] == BASELINE_VARIANT] or per_run
    if not baseline:
        return ""
    lifecycles = [r.get("lifecycle") or {} for r in baseline]
    n = len(lifecycles)
    ko = sum(1 for lc in lifecycles if lc.get("knocked_out"))
    matured = sum(1 for lc in lifecycles if lc.get("matured"))
    ki = sum(1 for lc in lifecycles if lc.get("knocked_in"))
    ki_at_maturity = sum(
        1 for lc in lifecycles if lc.get("knocked_in") and lc.get("matured")
    )
    if ko < n or matured > 0:
        return ""  # a mixed sample needs no special warning
    return (
        '<div class="callout"><b>Every trade in this sample knocked out.</b> '
        f"All {n} inceptions terminated at a knock-out; none reached maturity, and "
        f"{ki_at_maturity} settled knocked-in at maturity "
        f"({ki} knocked in at some point and then recovered into a knock-out). "
        "That is a property of the realized 2023&ndash;2026 CSI&nbsp;1000 path, not a "
        "modelling choice &mdash; but it bounds the conclusion hard: a snowball's model "
        "risk is largest precisely for the trade that is <em>knocked in at maturity</em> "
        "and settles into equity downside, and that path does not occur here. Any "
        "'no measurable edge' finding below should be read as <em>no edge on "
        "knock-out paths</em>, not as a general statement about snowball model risk."
        "</div>"
    )


def _verdict_paragraph(agg: Dict[str, Any]) -> str:
    """State the finding in plain words, including 'no measurable edge'."""
    paired = agg.get("paired_summary", {})
    if not paired:
        return ""
    lines = []
    for variant, p in paired.items():
        dist = p.get("d_pnl_pct_notional") or {}
        mean, stdev, n = dist.get("mean"), dist.get("stdev"), dist.get("n") or 0
        label = VARIANT_LABELS.get(variant, variant)
        if mean is None or n == 0:
            continue
        if stdev is not None and n > 1:
            se = stdev / math.sqrt(n)
            # se == 0 means every pair gave the identical edge; that is the
            # most consistent case, not an unknown one, so a non-zero mean
            # counts as significant rather than falling through.
            significant = abs(mean) > 2.0 * se if se > 0.0 else mean != 0.0
            verdict = (
                f"<b>{label}</b>: mean edge {mean:+.3f}% of notional over {n} paired "
                f"inceptions (standard error {se:.3f}%) &mdash; "
                + (
                    "outside two standard errors, so a real effect on this sample."
                    if significant
                    else "well inside two standard errors, i.e. <em>not</em> "
                    "distinguishable from zero on this sample."
                )
            )
        else:
            verdict = f"<b>{label}</b>: mean edge {mean:+.3f}% of notional over {n} pair(s)."
        lines.append(f"<li>{verdict}</li>")
    if not lines:
        return ""
    return (
        "<h3>4.1 &nbsp; The finding, stated plainly</h3><ul>"
        + "".join(lines)
        + "</ul><p>A standard error computed across overlapping inceptions of the same "
        "product understates the true uncertainty (the runs share market history), so "
        "treat the two-standard-error test above as a <em>generous</em> bar: an edge "
        "that fails it is certainly not established.</p>"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--out-dir", default=None, help="defaults to <run-dir>/aggregate"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "aggregate"

    agg = aggregate(run_dir)
    _atomic_write_json(out_dir / "aggregate.json", agg)
    paths = write_tables(agg, out_dir)
    report_path = out_dir / "volmodel_backtest_lecture.html"
    report_path.write_text(build_report(agg), encoding="utf-8")

    print(f"[aggregate] {len(agg['per_run'])} runs, {len(agg['variants'])} variants")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"  report: {report_path}")
    comp = agg["completeness"]
    print(
        f"  completeness: {comp['n_complete']}/{comp['n_runs_checked']} runs "
        "have every required daily output category"
    )
    for check in comp["incomplete"][:10]:
        print(f"    INCOMPLETE {check['run_dir']}: {'; '.join(check['issues'][:3])}")
    print(
        f"  sanity: {comp['n_sane']}/{comp['n_runs_checked']} runs satisfy every "
        "accounting identity"
    )
    for check in comp["sanity_failures"][:10]:
        print(f"    UNSOUND {check['run_dir']}: {'; '.join(check['issues'][:3])}")
    if agg["missing_runs"]:
        print(f"  WARNING: {len(agg['missing_runs'])} runs had no states.csv")
    if agg["failures"]:
        print(f"  WARNING: {len(agg['failures'])} runs failed in the fleet")
    ok = comp["all_complete"] and comp["all_sane"] and not agg["failures"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
