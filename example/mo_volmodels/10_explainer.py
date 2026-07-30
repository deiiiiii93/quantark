"""Stage 10 — render the MO calibration evidence as one self-contained HTML file.

This is the decision-oriented companion to the longer stage-06 lecture.  It
reads the frozen MO snapshot, the stage-02 through stage-05 artifacts, and the
independent official-CFFEX settlement-date diagnostics.  It validates the two
source cohorts separately and renders the same evidence-led document system
used by the CFETS USD/CNY explainer.  Optional futures, barrier, Snowball, and
hedging artifacts enrich the report when present.

Run:
    .venv/bin/python example/mo_volmodels/10_explainer.py --tag latest
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REQUIRED_STEMS = {
    "snapshot": "mo_snapshot",
    "surface": "mo_iv_surface",
    "localvol": "mo_reprice_localvol",
    "heston": "mo_calib_heston",
    "slv": "mo_reprice_slv",
    "diagnostics": "mo_calibration_diagnostics",
}
OPTIONAL_STEMS = {
    "futures": "im_futures",
    "barrier": "mo_barrier",
    "snowball": "mo_snowball",
    "hedging": "mo_hedging",
}
PARAMETERS = ("v0", "kappa", "theta", "sigma", "rho")
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FIT_GATE_VOL_POINTS = 0.10
OFFICIAL_SETTLEMENT_SOURCE = "official_cffex_eod_settlement"
SYNTHETIC_DIAGNOSTIC_SOURCE = "synthetic_test_fixture"
MARKET_EVIDENCE = (
    {
        "fact": "CSI 1000 index-option contract",
        "value": "European, cash-settled; RMB 100 per index point",
        "scope": "Current month, next two months and three quarterly months; strikes cover ±10% around the prior close.",
        "url": "https://www.cffex.com.cn/zz1000gzqq/",
    },
    {
        "fact": "2025 CFFEX index-option market",
        "value": "97.34 million lots; +23.85% year on year",
        "scope": "Exchange aggregate across the index-option complex, not MO-only volume or executable depth at each strike.",
        "url": "https://www.cffex.com.cn/sj/yearlymarketReportEng/2025/2025YearlyMarketReport.pdf",
    },
    {
        "fact": "Index-option trading rules",
        "value": "Continuous auction, European exercise, holiday-adjusted third-Friday expiry",
        "scope": "The listed-market structure is established; model calibratability remains an empirical surface question.",
        "url": "https://www.cffex.com.cn/ssxz/20221214/43100.html",
    },
    {
        "fact": "Official historical daily statistics",
        "value": "CFFEX publishes dated close and settlement cross sections",
        "scope": "The cross-date cohort uses settlement only. It is genuine EOD evidence, not executable bid/ask history.",
        "url": "https://www.cffex.com.cn/lssjxz/",
    },
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = _number(value)
    return "—" if number is None else f"{number:,.{digits}f}{suffix}"


def _fmt_param(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:.6f}" if abs(number) < 0.01 else f"{number:.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return payload


def _require_number(
    payload: Mapping[str, Any],
    key: str,
    context: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = _number(payload.get(key))
    if value is None:
        raise ValueError(f"{context} requires finite {key}")
    if positive and value <= 0.0:
        raise ValueError(f"{context} requires positive {key}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} requires non-negative {key}")
    return value


def _same_vector(left: Sequence[float], right: Sequence[float], *, atol: float = 1e-10) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= atol for a, b in zip(left, right))


def _smoothing_fingerprint(payload: Mapping[str, Any], context: str) -> str:
    smoothing = payload.get("target_smoothing")
    if not isinstance(smoothing, Mapping):
        raise ValueError(f"{context} requires target_smoothing metadata")
    if not isinstance(smoothing.get("method"), str):
        raise ValueError(f"{context} target_smoothing requires method")
    for key in ("raw_grid_rmse_iv", "raw_points_rmse_iv"):
        _require_number(smoothing, key, f"{context} target_smoothing", nonnegative=True)
    return json.dumps(smoothing, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_snapshot(snapshot: Mapping[str, Any]) -> tuple[float, str]:
    underlying = snapshot.get("underlying")
    if not isinstance(underlying, Mapping) or underlying.get("code") != "000852.SH":
        raise ValueError("snapshot underlying must be 000852.SH")
    spot = _require_number(underlying, "spot", "snapshot underlying", positive=True)
    fetched_at = snapshot.get("fetched_at")
    if not isinstance(fetched_at, str) or len(fetched_at) < 10:
        raise ValueError("snapshot requires fetched_at")
    expiries = snapshot.get("expiries")
    if not isinstance(expiries, list) or not expiries:
        raise ValueError("snapshot requires non-empty expiries")
    for expiry_index, expiry in enumerate(expiries):
        if not isinstance(expiry, Mapping):
            raise ValueError(f"snapshot expiry {expiry_index} must be an object")
        _require_number(expiry, "T_years", f"snapshot expiry {expiry_index}", positive=True)
        quotes = expiry.get("quotes")
        if not isinstance(quotes, list) or not quotes:
            raise ValueError(f"snapshot expiry {expiry_index} requires quotes")
        for quote_index, quote in enumerate(quotes):
            context = f"snapshot expiry {expiry_index} quote {quote_index}"
            if not isinstance(quote, Mapping) or quote.get("type") not in ("C", "P"):
                raise ValueError(f"{context} requires call/put type")
            _require_number(quote, "strike", context, positive=True)
            _require_number(quote, "last", context, positive=True)
    return spot, fetched_at


def _validate_surface(surface: Mapping[str, Any], spot: float, fetched_at: str) -> list[float]:
    surface_spot = _require_number(surface, "s0", "surface", positive=True)
    if not math.isclose(surface_spot, spot, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"snapshot/surface spot drift: {spot} != {surface_spot}")
    if surface.get("fetched_at") != fetched_at:
        raise ValueError("snapshot/surface fetched_at drift")
    raw_maturities = surface.get("maturities")
    strikes = surface.get("strikes")
    grid = surface.get("iv_grid")
    per_expiry = surface.get("per_expiry")
    if not isinstance(raw_maturities, list) or not raw_maturities:
        raise ValueError("surface requires maturities")
    maturities = [_number(value) for value in raw_maturities]
    if any(value is None or value <= 0.0 for value in maturities):
        raise ValueError("surface maturities must be finite and positive")
    maturity_values = [float(value) for value in maturities if value is not None]
    if maturity_values != sorted(maturity_values) or len(set(maturity_values)) != len(maturity_values):
        raise ValueError("surface maturities must be unique and increasing")
    if not isinstance(strikes, list) or not strikes or any(_number(value) is None or float(value) <= 0.0 for value in strikes):
        raise ValueError("surface strikes must be finite and positive")
    if not isinstance(grid, list) or len(grid) != len(maturity_values):
        raise ValueError("surface iv_grid maturity dimension mismatch")
    for row in grid:
        if not isinstance(row, list) or len(row) != len(strikes):
            raise ValueError("surface iv_grid strike dimension mismatch")
        if any(_number(value) is None or float(value) <= 0.0 for value in row):
            raise ValueError("surface iv_grid must contain finite positive IVs")
    if not isinstance(per_expiry, list) or len(per_expiry) != len(maturity_values):
        raise ValueError("surface per_expiry dimension mismatch")
    expiry_maturities: list[float] = []
    for index, expiry in enumerate(per_expiry):
        if not isinstance(expiry, Mapping):
            raise ValueError(f"surface expiry {index} must be an object")
        maturity = _require_number(expiry, "T", f"surface expiry {index}", positive=True)
        _require_number(expiry, "forward", f"surface expiry {index}", positive=True)
        _require_number(expiry, "r", f"surface expiry {index}")
        _require_number(expiry, "q", f"surface expiry {index}")
        points = expiry.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"surface expiry {index} requires raw OTM points")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or _number(point[0]) is None
                or _number(point[1]) is None
                or float(point[0]) <= 0.0
                or float(point[1]) <= 0.0
            ):
                raise ValueError(f"surface expiry {index} has invalid OTM point")
        expiry_maturities.append(maturity)
    if not _same_vector(maturity_values, expiry_maturities):
        raise ValueError("surface maturity/per_expiry drift")
    return maturity_values


def _validate_model(name: str, payload: Mapping[str, Any], maturities: Sequence[float]) -> None:
    _require_number(payload, "overall_rmse_iv", name, nonnegative=True)
    _require_number(payload, "raw_overall_rmse_iv", name, nonnegative=True)
    rows = payload.get("per_expiry")
    if not isinstance(rows, list):
        raise ValueError(f"{name} requires per_expiry")
    row_maturities: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} per_expiry {index} must be an object")
        row_maturities.append(_require_number(row, "T", f"{name} per_expiry {index}", positive=True))
        _require_number(row, "rmse_iv", f"{name} per_expiry {index}", nonnegative=True)
        _require_number(row, "raw_rmse_iv", f"{name} per_expiry {index}", nonnegative=True)
    if not _same_vector(list(maturities), row_maturities):
        raise ValueError(f"{name} maturity vector drift")
    _smoothing_fingerprint(payload, name)
    if name == "localvol":
        low = _require_number(payload, "lv_min", name, positive=True)
        high = _require_number(payload, "lv_max", name, positive=True)
        if low > high:
            raise ValueError("localvol lv_min exceeds lv_max")
    if name == "slv":
        low = _require_number(payload, "leverage_min", name, positive=True)
        high = _require_number(payload, "leverage_max", name, positive=True)
        if low > high:
            raise ValueError("slv leverage_min exceeds leverage_max")


def _validate_heston(heston: Mapping[str, Any], spot: float, surface_path: Path) -> None:
    if not isinstance(heston.get("success"), bool):
        raise ValueError("heston requires boolean success")
    params = heston.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("heston requires params")
    values: dict[str, float] = {}
    for name in PARAMETERS:
        values[name] = _require_number(params, name, "heston params")
    if min(values["v0"], values["kappa"], values["theta"], values["sigma"]) <= 0.0:
        raise ValueError("heston variance parameters must be positive")
    if not -1.0 <= values["rho"] <= 1.0:
        raise ValueError("heston rho must be in [-1, 1]")
    stored_feller = _require_number(heston, "feller", "heston", nonnegative=True)
    calculated = 2.0 * values["kappa"] * values["theta"] / values["sigma"] ** 2
    if not math.isclose(stored_feller, calculated, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError("heston stored Feller ratio does not match params")

    spec = heston.get("calibration_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("heston requires persisted calibration_spec")
    if spec.get("parameter_order") != list(PARAMETERS):
        raise ValueError("heston calibration_spec parameter_order drift")
    spec_spot = _require_number(spec, "s0", "heston calibration_spec", positive=True)
    if not math.isclose(spec_spot, spot, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("heston calibration_spec spot drift")
    provenance = spec.get("surface_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("kind") != "file_sha256"
        or provenance.get("filename") != surface_path.name
    ):
        raise ValueError("heston calibration_spec requires surface file provenance")
    expected_hash = hashlib.sha256(surface_path.read_bytes()).hexdigest()
    if provenance.get("sha256") != expected_hash:
        raise ValueError("heston input surface SHA-256 drift")
    node_count = int(_require_number(spec, "node_count", "heston calibration_spec", positive=True))
    bounds = spec.get("bounds")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(not isinstance(side, list) or len(side) != len(PARAMETERS) for side in bounds)
    ):
        raise ValueError("heston calibration_spec requires 2x5 bounds")
    lower = [_number(value) for value in bounds[0]]
    upper = [_number(value) for value in bounds[1]]
    if any(value is None for value in (*lower, *upper)):
        raise ValueError("heston calibration bounds must be finite")
    for index, name in enumerate(PARAMETERS):
        lo, hi = float(lower[index]), float(upper[index])
        if lo >= hi or not lo <= values[name] <= hi:
            raise ValueError(f"heston parameter {name} lies outside persisted bounds")

    node_rows = heston.get("node_rows")
    if not isinstance(node_rows, list) or len(node_rows) != node_count:
        raise ValueError("heston node_rows do not match calibration_spec node_count")
    bound_hits = heston.get("bound_hits")
    if not isinstance(bound_hits, Mapping):
        raise ValueError("heston requires bound_hits")
    for name in PARAMETERS:
        row = bound_hits.get(name)
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("lower"), bool)
            or not isinstance(row.get("upper"), bool)
        ):
            raise ValueError(f"heston bound_hits requires booleans for {name}")

    jacobian = heston.get("jacobian")
    if not isinstance(jacobian, Mapping):
        raise ValueError("heston requires Jacobian/SVD evidence")
    if jacobian.get("shape") != [node_count, len(PARAMETERS)]:
        raise ValueError("heston Jacobian shape does not match node count")
    if jacobian.get("parameter_order") != list(PARAMETERS):
        raise ValueError("heston Jacobian parameter_order drift")
    if jacobian.get("excludes_feller_penalty") is not True:
        raise ValueError("heston Jacobian must exclude the Feller penalty")
    base = jacobian.get("base_parameters")
    if not isinstance(base, Mapping) or any(
        not math.isclose(
            _require_number(base, name, "heston Jacobian base_parameters"),
            values[name],
            rel_tol=1e-9,
            abs_tol=1e-11,
        )
        for name in PARAMETERS
    ):
        raise ValueError("heston Jacobian base parameters drift from fitted params")
    _validate_svd(jacobian, "heston Jacobian")

    bootstrap = heston.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("heston requires bootstrap evidence")
    requested = int(
        _require_number(bootstrap, "requested_replicates", "heston bootstrap", positive=True)
    )
    successful = int(
        _require_number(bootstrap, "successful_replicates", "heston bootstrap", nonnegative=True)
    )
    failed = int(
        _require_number(bootstrap, "failed_replicates", "heston bootstrap", nonnegative=True)
    )
    if successful + failed != requested:
        raise ValueError("heston bootstrap replicate counts do not reconcile")
    if successful == 0:
        raise ValueError("heston bootstrap has no successful replicate")
    if bootstrap.get("is_statistical_confidence_interval") is not False:
        raise ValueError("heston bootstrap must not be labelled a confidence interval")
    replicates = bootstrap.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != requested:
        raise ValueError("heston bootstrap replicate rows do not reconcile")
    quantiles = bootstrap.get("parameter_quantiles")
    rates = bootstrap.get("bound_hit_rates")
    if not isinstance(quantiles, Mapping) or not isinstance(rates, Mapping):
        raise ValueError("successful heston bootstrap requires quantiles and bound-hit rates")
    for name in PARAMETERS:
        distribution = quantiles.get(name)
        hit_rate = rates.get(name)
        if not isinstance(distribution, Mapping) or not isinstance(hit_rate, Mapping):
            raise ValueError(f"heston bootstrap missing parameter summary for {name}")
        q05 = _require_number(distribution, "q05", f"heston bootstrap {name}")
        q50 = _require_number(distribution, "q50", f"heston bootstrap {name}")
        q95 = _require_number(distribution, "q95", f"heston bootstrap {name}")
        if not q05 <= q50 <= q95:
            raise ValueError(f"heston bootstrap quantiles are not ordered for {name}")
        either = _require_number(hit_rate, "either", f"heston bootstrap {name}")
        if not 0.0 <= either <= 1.0:
            raise ValueError(f"heston bootstrap bound-hit rate is invalid for {name}")


def _validate_svd(jacobian: Mapping[str, Any], context: str) -> None:
    svd = jacobian.get("svd")
    if not isinstance(svd, Mapping):
        raise ValueError(f"{context} requires SVD evidence")
    fixed = svd.get("fixed_economic")
    if not isinstance(fixed, Mapping):
        raise ValueError(f"{context} requires fixed_economic SVD")
    singular = fixed.get("singular_values")
    if (
        not isinstance(singular, list)
        or len(singular) != len(PARAMETERS)
        or any(_number(value) is None or float(value) < 0.0 for value in singular)
    ):
        raise ValueError(f"{context} fixed_economic singular values are invalid")
    for key in ("numerical_rank", "policy_effective_rank"):
        value = int(_require_number(fixed, key, f"{context} fixed_economic", nonnegative=True))
        if not 0 <= value <= len(PARAMETERS):
            raise ValueError(f"{context} {key} is invalid")
    condition = fixed.get("condition_number")
    if condition is not None and (_number(condition) is None or float(condition) < 1.0):
        raise ValueError(f"{context} condition number is invalid")


def _validate_diagnostics(payload: Mapping[str, Any], tag: str) -> None:
    source = payload.get("source_class")
    expected = SYNTHETIC_DIAGNOSTIC_SOURCE if tag == "sample" else OFFICIAL_SETTLEMENT_SOURCE
    if source != expected:
        raise ValueError(f"diagnostics source_class must be {expected!r} for tag {tag!r}")
    if payload.get("price_field") != "settlement":
        raise ValueError("diagnostics price_field must be settlement")
    included = payload.get("included")
    if not isinstance(included, list) or not included:
        raise ValueError("diagnostics requires at least one included date")
    comparability_gate = payload.get("strict_comparability_gate")
    if not isinstance(comparability_gate, Mapping):
        raise ValueError("diagnostics requires strict_comparability_gate")
    if comparability_gate.get("required_source_class") != source:
        raise ValueError("diagnostics comparability source_class drift")
    if comparability_gate.get("required_price_field") != "settlement":
        raise ValueError("diagnostics comparability price_field drift")
    if (
        comparability_gate.get("unique_trade_dates") is not True
        or comparability_gate.get("unique_source_sha256") is not True
    ):
        raise ValueError("diagnostics comparability identity gate is not strict")
    required_config = comparability_gate.get("required_config")
    if not isinstance(required_config, Mapping):
        raise ValueError("diagnostics comparability gate requires calibration config")
    required_config_fingerprint = json.dumps(
        required_config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    dates: set[str] = set()
    hashes: set[str] = set()
    for index, row in enumerate(included):
        context = f"diagnostics included date {index}"
        if not isinstance(row, Mapping) or row.get("source_class") != source:
            raise ValueError(f"{context} source_class drift")
        trade_date = row.get("trade_date")
        digest = row.get("source_sha256")
        if not isinstance(trade_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", trade_date):
            raise ValueError(f"{context} requires ISO trade_date")
        if trade_date in dates:
            raise ValueError("diagnostics contains duplicate trade dates")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{context} requires source_sha256")
        if digest in hashes:
            raise ValueError("diagnostics contains duplicate source hashes")
        dates.add(trade_date)
        hashes.add(digest)
        config = row.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"{context} requires calibration config")
        fingerprint = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if fingerprint != required_config_fingerprint:
            raise ValueError("diagnostics calibration config drift from required config")
        universe = row.get("node_universe")
        best = row.get("best")
        if not isinstance(universe, Mapping) or not isinstance(best, Mapping):
            raise ValueError(f"{context} requires node_universe and best fit")
        _require_number(universe, "node_count", context, positive=True)
        _require_number(universe, "expiry_count", context, positive=True)
        if best.get("success") is not True:
            raise ValueError(f"{context} best fit was not successful")
        params = best.get("params")
        if not isinstance(params, Mapping):
            raise ValueError(f"{context} requires fitted params")
        for name in PARAMETERS:
            _require_number(params, name, f"{context} params")
        _require_number(best, "weighted_rmse_iv", context, nonnegative=True)
        _require_number(best, "feller_ratio", context, nonnegative=True)
        jacobian = row.get("jacobian")
        if not isinstance(jacobian, Mapping):
            raise ValueError(f"{context} requires Jacobian evidence")
        _validate_svd(jacobian, context)
        if source == OFFICIAL_SETTLEMENT_SOURCE:
            weighting = jacobian.get("row_weighting")
            if (
                not isinstance(weighting, Mapping)
                or weighting.get("cross_date_svd_uses_weighted_rows") is not True
                or weighting.get("policy") != "sqrt_equal_total_weight_per_expiry"
            ):
                raise ValueError(f"{context} Jacobian weighting drift")
    stability = payload.get("stability")
    if not isinstance(stability, Mapping):
        raise ValueError("diagnostics requires stability summary")
    if source == OFFICIAL_SETTLEMENT_SOURCE and (
        not isinstance(stability.get("parity_quality"), Mapping)
        or not isinstance(stability.get("static_arbitrage"), Mapping)
    ):
        raise ValueError("official diagnostics require parity and static-arbitrage summaries")
    if not isinstance(payload.get("verdicts"), list):
        raise ValueError("diagnostics requires verdicts")


def _validate_optional(
    name: str,
    payload: Mapping[str, Any],
    spot: float,
    smoothing_fingerprint: str,
    snapshot_date: str,
) -> None:
    if name == "futures":
        if payload.get("valuation_date") != snapshot_date:
            raise ValueError("futures valuation_date does not match snapshot date")
        quotes = payload.get("quotes")
        if not isinstance(quotes, list):
            raise ValueError("futures artifact requires quotes")
        for index, quote in enumerate(quotes):
            if not isinstance(quote, Mapping) or not quote.get("expiry_date"):
                raise ValueError(f"futures quote {index} requires expiry_date")
            _require_number(quote, "maturity", f"futures quote {index}", positive=True)
            _require_number(quote, "close", f"futures quote {index}", positive=True)
        return
    spec = payload.get("spec")
    models = payload.get("models")
    if not isinstance(spec, Mapping) or not isinstance(models, Mapping) or not models:
        raise ValueError(f"{name} artifact requires spec and models")
    spec_spot = _require_number(spec, "s0", f"{name} spec", positive=True)
    if not math.isclose(spec_spot, spot, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} spot does not match surface")
    smoothing = spec.get("iv_smoothing")
    if not isinstance(smoothing, Mapping):
        raise ValueError(f"{name} spec requires iv_smoothing metadata")
    fingerprint = json.dumps(smoothing, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if fingerprint != smoothing_fingerprint:
        raise ValueError(f"{name} smoothing fingerprint drift")
    expected = {
        "barrier": ("BSM (flat ATM)", "Local Vol", "Heston", "SLV"),
        "snowball": ("BSM (flat ATM)", "Local Vol", "Heston QE", "SLV", "SLV QE"),
        "hedging": ("BSM (flat vol)", "Local Vol", "Heston", "SLV"),
    }[name]
    missing = [model_name for model_name in expected if model_name not in models]
    if missing:
        raise ValueError(f"{name} artifact missing model rows: {missing}")
    required_fields = {
        "barrier": ("mc", "pde"),
        "snowball": ("mc_pct_initial", "ko_probability"),
        "hedging": ("initial_delta", "total_rebalance_units", "total_residual_pnl"),
    }[name]
    for model_name in expected:
        model = models[model_name]
        if not isinstance(model, Mapping):
            raise ValueError(f"{name} model {model_name} must be an object")
        for field in required_fields:
            _require_number(model, field, f"{name} model {model_name}")


def load_artifacts(data_dir: str | Path, tag: str) -> dict[str, dict[str, Any]]:
    """Load and cross-check the core MO evidence before any HTML is written."""
    if not SAFE_TAG.fullmatch(tag):
        raise ValueError("tag must contain only letters, numbers, '.', '_' or '-'")
    directory = Path(data_dir)
    required_paths = {
        name: directory / f"{stem}_{tag}.json"
        for name, stem in REQUIRED_STEMS.items()
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"required MO explainer artifacts are missing:\n  - {formatted}")
    artifacts = {name: _read_json(path) for name, path in required_paths.items()}
    spot, fetched_at = _validate_snapshot(artifacts["snapshot"])
    maturities = _validate_surface(artifacts["surface"], spot, fetched_at)
    for name in ("localvol", "heston", "slv"):
        _validate_model(name, artifacts[name], maturities)
    _validate_heston(artifacts["heston"], spot, required_paths["surface"])
    _validate_diagnostics(artifacts["diagnostics"], tag)
    fingerprints = {_smoothing_fingerprint(artifacts[name], name) for name in ("localvol", "heston", "slv")}
    if len(fingerprints) != 1:
        raise ValueError("localvol/heston/slv smoothing fingerprint drift")
    smoothing_fingerprint = next(iter(fingerprints))
    snapshot_date = fetched_at[:10]
    for name, stem in OPTIONAL_STEMS.items():
        path = directory / f"{stem}_{tag}.json"
        if path.is_file():
            payload = _read_json(path)
            _validate_optional(name, payload, spot, smoothing_fingerprint, snapshot_date)
            artifacts[name] = payload
    return artifacts


def _flat_atm_baseline(per_expiry: Sequence[Mapping[str, Any]]) -> float:
    errors: list[float] = []
    for expiry in per_expiry:
        forward = float(expiry["forward"])
        points = [(float(point[0]), float(point[1])) for point in expiry["points"]]
        atm = min(points, key=lambda point: abs(point[0] - forward))[1]
        errors.extend(iv - atm for _, iv in points)
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else math.nan


def _grid_cell_counts(surface: Mapping[str, Any]) -> tuple[int, int, int]:
    grid_strikes = [float(value) for value in surface["strikes"]]
    direct = interior = flat = 0
    for expiry in surface["per_expiry"]:
        observed = [float(point[0]) for point in expiry["points"]]
        low, high = min(observed), max(observed)
        for strike in grid_strikes:
            if any(abs(strike - value) <= 1e-9 for value in observed):
                direct += 1
            elif low <= strike <= high:
                interior += 1
            else:
                flat += 1
    return direct, interior, flat


def _artifact_table(tag: str, artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    rows = []
    for name, stem in {**REQUIRED_STEMS, **OPTIONAL_STEMS}.items():
        included = name in artifacts
        role = "required" if name in REQUIRED_STEMS else "optional"
        rows.append(
            "<tr>"
            f"<td><code>{_escape(stem)}_{_escape(tag)}.json</code></td>"
            f"<td><span class='chip {'pass' if included else 'neutral'}'>{'LOADED' if included else 'ABSENT'}</span></td>"
            f"<td>{role}</td><td>{_escape(name)}</td>"
            "</tr>"
        )
    return "".join(rows)


CSS = r"""
:root{--paper:#F7F6F2;--panel:#EFEDE6;--panel2:#E7E4DB;--ink:#20242C;--ink2:#4A5160;--faint:#858B98;--line:#D8D5CC;--grid:#DCDAD2;--cinnabar:#BE3A2B;--cinnabar-soft:#BE3A2B22;--jade:#2F7D6D;--jade-soft:#2F7D6D1E;--amber:#A66B1F;--amber-soft:#A66B1F1F;--slate:#5B6B84;--slate-soft:#5B6B841B;--code-bg:#ECEAE2}
@media(prefers-color-scheme:dark){:root{--paper:#151A21;--panel:#1C222B;--panel2:#232A35;--ink:#E2E4E8;--ink2:#AAB1BE;--faint:#79808E;--line:#2E3541;--grid:#252C37;--cinnabar:#E0604C;--cinnabar-soft:#E0604C26;--jade:#4FA98F;--jade-soft:#4FA98F22;--amber:#D9A03F;--amber-soft:#D9A03F22;--slate:#8FA2BE;--slate-soft:#8FA2BE20;--code-bg:#1A202A}}
:root[data-theme='dark']{--paper:#151A21;--panel:#1C222B;--panel2:#232A35;--ink:#E2E4E8;--ink2:#AAB1BE;--faint:#79808E;--line:#2E3541;--grid:#252C37;--cinnabar:#E0604C;--cinnabar-soft:#E0604C26;--jade:#4FA98F;--jade-soft:#4FA98F22;--amber:#D9A03F;--amber-soft:#D9A03F22;--slate:#8FA2BE;--slate-soft:#8FA2BE20;--code-bg:#1A202A}
:root[data-theme='light']{--paper:#F7F6F2;--panel:#EFEDE6;--panel2:#E7E4DB;--ink:#20242C;--ink2:#4A5160;--faint:#858B98;--line:#D8D5CC;--grid:#DCDAD2;--cinnabar:#BE3A2B;--cinnabar-soft:#BE3A2B22;--jade:#2F7D6D;--jade-soft:#2F7D6D1E;--amber:#A66B1F;--amber-soft:#A66B1F1F;--slate:#5B6B84;--slate-soft:#5B6B841B;--code-bg:#ECEAE2}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Charter,"Bitstream Charter",Cambria,Georgia,"Noto Serif SC",serif;font-size:17px;line-height:1.62}
main,.hero,.footer{max-width:72ch;margin:0 auto;padding-left:20px;padding-right:20px}main{padding-bottom:6rem}h1,h2,h3,h4{font-family:Palatino,"Palatino Linotype","URW Palladio L",Georgia,"Noto Serif SC",serif;line-height:1.18;text-wrap:balance}h1{font-size:clamp(2.2rem,6vw,3.5rem);font-weight:600;margin:.35rem 0 .8rem;letter-spacing:-.025em}h2{font-size:1.75rem;margin:0 0 1rem}h3{font-size:1.2rem;margin:2rem 0 .55rem}p{margin:.85rem 0}section{margin-top:4.7rem;scroll-margin-top:4rem}a{color:var(--cinnabar);text-underline-offset:2px}.mono,code,.num{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}code{font-size:.84em;background:var(--code-bg);padding:.08em .35em;border-radius:3px}
.topnav{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--paper) 94%,transparent);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}.topnav-inner{max-width:1060px;margin:0 auto;padding:.48rem 20px;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem}.brand{letter-spacing:.12em;text-transform:uppercase;color:var(--ink2);font-weight:700}.topnav nav{display:flex;gap:.78rem;flex:1;flex-wrap:wrap}.topnav a{color:var(--ink2);text-decoration:none;text-transform:uppercase;letter-spacing:.05em}.topnav a:hover{color:var(--cinnabar)}
.hero{padding-top:3.5rem}.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--cinnabar);margin:0 0 .3rem}.kicker{color:var(--ink2);font-size:1.08rem;max-width:62ch}.hero-meta{display:flex;gap:1rem 1.5rem;flex-wrap:wrap;margin-top:1.3rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.73rem;color:var(--faint)}.hero-meta b{color:var(--ink)}.evidence-banner{margin:1.5rem 0;padding:.8rem 1rem;border:1px solid var(--amber);border-left:5px solid var(--amber);background:var(--amber-soft);border-radius:0 7px 7px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.75rem;line-height:1.55}.evidence-banner b{display:block;color:var(--amber);letter-spacing:.12em;text-transform:uppercase}
.toc{columns:2;gap:2.4rem;margin:2rem 0 0}.toc a{display:block;color:var(--ink);text-decoration:none;padding:.2rem 0;border-bottom:1px dotted var(--line);break-inside:avoid}.toc .no{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;color:var(--cinnabar);margin-right:.55rem}.wide{width:min(1000px,calc(100vw - 32px));margin-left:50%;transform:translateX(-50%)}figure{margin:2rem 0}.fig-frame{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:1rem 1.1rem 1.15rem;overflow-x:auto;box-shadow:0 18px 50px color-mix(in srgb,var(--ink) 5%,transparent)}figcaption{font-size:.82rem;color:var(--ink2);margin-top:.65rem;line-height:1.48;max-width:90ch}.figno{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.69rem;letter-spacing:.1em;text-transform:uppercase;color:var(--cinnabar);margin-right:.5em}canvas{display:block;width:100%;height:auto}
.controls{display:flex;gap:.8rem 1.3rem;flex-wrap:wrap;align-items:end;margin-bottom:.8rem}.ctl{display:flex;flex-direction:column;gap:.18rem}.ctl label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink2)}button,select{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:.4rem .72rem}button{cursor:pointer}button:hover,button:focus-visible,select:focus-visible{border-color:var(--cinnabar);outline:2px solid transparent}button[aria-pressed='true']{background:var(--ink);color:var(--paper);border-color:var(--ink)}.chiprow{display:flex;gap:.35rem;flex-wrap:wrap}.readout{display:flex;gap:.7rem 1.3rem;flex-wrap:wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;margin-top:.65rem}.readout .k{color:var(--faint)}.readout .v{font-weight:700}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.75rem;margin:1.2rem 0}.metric-card{border:1px solid var(--line);border-radius:7px;background:var(--panel);padding:.8rem .9rem}.metric-card .label{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}.metric-card .value{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.15rem;font-weight:700;margin:.16rem 0}.metric-card .sub{font-size:.78rem;color:var(--ink2)}.tbl-wrap{overflow-x:auto;margin:1.2rem 0}table{border-collapse:collapse;width:100%;font-size:.84rem}th,td{padding:.42rem .62rem;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}thead th{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);border-bottom:2px solid var(--ink2)}td.n,th.n{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}.fallback-evidence{margin-top:1rem}.fallback-evidence summary{font-weight:700}
.chip{display:inline-block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.64rem;letter-spacing:.05em;padding:.12em .55em;border-radius:99px;font-weight:700}.chip.pass{background:var(--jade-soft);color:var(--jade)}.chip.warn{background:var(--amber-soft);color:var(--amber)}.chip.fail{background:var(--cinnabar-soft);color:var(--cinnabar)}.chip.neutral{background:var(--slate-soft);color:var(--slate)}.note{border:1px solid var(--line);border-left:4px solid var(--jade);background:var(--jade-soft);border-radius:0 6px 6px 0;padding:.82rem 1.05rem;margin:1.35rem 0;font-size:.92rem}.note.warn{border-left-color:var(--amber);background:var(--amber-soft)}.note.risk{border-left-color:var(--cinnabar);background:var(--cinnabar-soft)}.note .t{display:block;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;font-weight:700;color:var(--jade);margin-bottom:.2rem}.note.warn .t{color:var(--amber)}.note.risk .t{color:var(--cinnabar)}
.eq{margin:1.2rem 0;padding:1rem 1.15rem;background:var(--panel);border-left:3px solid var(--slate);border-radius:0 6px 6px 0;overflow-x:auto;font-family:Palatino,"Palatino Linotype",Georgia,serif;font-size:1rem;line-height:1.9}.eq .lbl{float:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.66rem;color:var(--faint);margin-left:1rem}details{border:1px solid var(--line);border-radius:6px;margin:1rem 0;background:var(--panel)}details summary{cursor:pointer;padding:.7rem 1rem;font-weight:700;font-size:.9rem}details summary::marker{color:var(--cinnabar)}details .body{padding:0 1.05rem 1rem;font-size:.9rem}.pipe{display:flex;flex-wrap:wrap;gap:.45rem;align-items:stretch;margin:1.2rem 0}.pipe .box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.55rem .7rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.68rem;line-height:1.45;flex:1 1 145px}.pipe .box b{display:block;color:var(--cinnabar)}.pipe .arrow{align-self:center;color:var(--faint)}.footer{padding-top:2rem;padding-bottom:4rem;border-top:1px solid var(--line);color:var(--faint);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem}
@media(max-width:700px){.toc{columns:1}.topnav nav{display:none}.wide{width:calc(100vw - 20px)}body{font-size:16px}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}@media print{.topnav,.controls{display:none!important}.wide{width:100%;margin-left:0;transform:none}.fig-frame{box-shadow:none}details{break-inside:avoid}details>.body{display:block}.footer{padding-bottom:1rem}}
"""


JS = r"""
(function(){
"use strict";
var DATA=window.__MO_REPORT_DATA__,activeError='raw',activeStability='rmse';
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim();}
function finite(v){return typeof v==='number'&&Number.isFinite(v);}
function fmt(v,d){return finite(v)?v.toFixed(d===undefined?3:d):'—';}
function setup(canvas,height){var dpr=window.devicePixelRatio||1,W=canvas.clientWidth||880,H=height;canvas.width=W*dpr;canvas.height=H*dpr;var g=canvas.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);return{g:g,W:W,H:H};}
function selectedSmile(){var sel=document.getElementById('expirySelect');return DATA.smiles[Number(sel.value)||0];}
function drawSmile(){
  var canvas=document.getElementById('smileCanvas'),c=setup(canvas,350),g=c.g,W=c.W,H=c.H,s=selectedSmile();
  var pad={l:60,r:18,t:20,b:48},nodes=s.nodes.slice().sort(function(a,b){return a.strike-b.strike;});
  var xs=nodes.map(function(n){return n.strike/s.forward;}),ys=nodes.map(function(n){return n.iv*100;});
  var xmin=Math.min.apply(null,xs),xmax=Math.max.apply(null,xs),ymin=Math.min.apply(null,ys),ymax=Math.max.apply(null,ys),xp=Math.max((xmax-xmin)*.08,.01),yp=Math.max((ymax-ymin)*.18,.15);xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
  function X(x){return pad.l+(W-pad.l-pad.r)*(x-xmin)/(xmax-xmin);}function Y(y){return pad.t+(H-pad.t-pad.b)*(ymax-y)/(ymax-ymin);}
  g.font='10px ui-monospace,Menlo,monospace';g.lineWidth=1;g.strokeStyle=css('--grid');g.fillStyle=css('--faint');
  for(var i=0;i<=4;i++){var y=ymin+(ymax-ymin)*i/4;g.beginPath();g.moveTo(pad.l,Y(y));g.lineTo(W-pad.r,Y(y));g.stroke();g.textAlign='right';g.fillText(y.toFixed(1)+'%',pad.l-7,Y(y)+3);}
  if(xmin<1&&xmax>1){g.save();g.setLineDash([4,4]);g.strokeStyle=css('--amber');g.beginPath();g.moveTo(X(1),pad.t);g.lineTo(X(1),H-pad.b);g.stroke();g.restore();}
  g.strokeStyle=css('--slate');g.lineWidth=1.8;g.beginPath();nodes.forEach(function(n,i){var x=X(n.strike/s.forward),y=Y(n.iv*100);if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();
  nodes.forEach(function(n){g.fillStyle=css('--slate');g.beginPath();g.arc(X(n.strike/s.forward),Y(n.iv*100),3.5,0,Math.PI*2);g.fill();});
  g.fillStyle=css('--faint');g.textAlign='center';g.fillText('strike / parity-implied forward',pad.l+(W-pad.l-pad.r)/2,H-4);
  document.getElementById('smileReadout').innerHTML='<span><span class="k">expiry </span><span class="v">'+s.expiry+'</span></span><span><span class="k">T </span><span class="v">'+fmt(s.T,3)+'y</span></span><span><span class="k">forward </span><span class="v">'+fmt(s.forward,1)+'</span></span><span><span class="k">OTM nodes </span><span class="v">'+nodes.length+'</span></span>';
}
function drawErrors(){
  var canvas=document.getElementById('rmseCanvas'),c=setup(canvas,340),g=c.g,W=c.W,H=c.H,rows=DATA.errors,pad={l:62,r:20,t:22,b:54};
  var names=['heston','localvol','slv'],colors={'heston':'--cinnabar','localvol':'--jade','slv':'--slate'},values=[];
  rows.forEach(function(r){names.forEach(function(n){var v=r[n][activeError];if(finite(v))values.push(v);});});
  var ymax=Math.max.apply(null,values)*1.15,ymin=0;function X(i){return rows.length===1?(pad.l+W-pad.r)/2:pad.l+(W-pad.l-pad.r)*i/(rows.length-1);}function Y(y){return pad.t+(H-pad.t-pad.b)*(ymax-y)/(ymax-ymin);}
  g.font='10px ui-monospace,Menlo,monospace';g.strokeStyle=css('--grid');g.fillStyle=css('--faint');g.lineWidth=1;
  for(var i=0;i<=4;i++){var y=ymax*i/4;g.beginPath();g.moveTo(pad.l,Y(y));g.lineTo(W-pad.r,Y(y));g.stroke();g.textAlign='right';g.fillText(y.toFixed(1),pad.l-8,Y(y)+3);}
  names.forEach(function(name){g.strokeStyle=css(colors[name]);g.lineWidth=2;g.beginPath();rows.forEach(function(r,i){var y=Y(r[name][activeError]);if(i===0)g.moveTo(X(i),y);else g.lineTo(X(i),y);});g.stroke();rows.forEach(function(r,i){g.fillStyle=css(colors[name]);g.beginPath();g.arc(X(i),Y(r[name][activeError]),3.7,0,Math.PI*2);g.fill();});});
  rows.forEach(function(r,i){g.fillStyle=css('--ink2');g.textAlign=i===0?'left':(i===rows.length-1?'right':'center');var label=W<650?r.expiry.slice(5):r.expiry;g.fillText(label,X(i),H-18);});
  g.fillStyle=css('--faint');g.textAlign='left';g.fillText('RMSE (vol points)',pad.l,pad.t-7);
  document.getElementById('rmseReadout').innerHTML='<span><span class="k">target </span><span class="v">'+(activeError==='raw'?'raw OTM quotes':'SABR-prepared surface')+'</span></span><span><span class="k">red </span><span class="v">Heston</span></span><span><span class="k">jade </span><span class="v">Local Vol</span></span><span><span class="k">slate </span><span class="v">SLV</span></span>';
}
function drawStability(){
  var canvas=document.getElementById('stabilityCanvas'),c=setup(canvas,360),g=c.g,W=c.W,H=c.H,rows=DATA.stability,pad={l:72,r:22,t:24,b:58};
  var labels={rmse:'weighted RMSE (vol pts)',condition:'fixed-scale condition number',v0:'v0',kappa:'kappa',theta:'theta',sigma:'sigma',rho:'rho'};
  function raw(r){return activeStability==='rmse'?r.rmse:(activeStability==='condition'?r.condition:r.params[activeStability]);}
  var finiteRows=rows.filter(function(r){return finite(raw(r))&&!(activeStability==='condition'&&raw(r)<=0);});
  if(!finiteRows.length){g.fillStyle=css('--faint');g.font='12px ui-monospace,Menlo,monospace';g.fillText('No finite values',pad.l,pad.t+20);return;}
  var values=finiteRows.map(function(r){var v=raw(r);return activeStability==='condition'?Math.log10(v):v;});
  var ymin=Math.min.apply(null,values),ymax=Math.max.apply(null,values),spread=ymax-ymin,puff=Math.max(spread*.16,Math.max(Math.abs(ymin),Math.abs(ymax),1)*.035);ymin-=puff;ymax+=puff;
  function X(i){return rows.length===1?(pad.l+W-pad.r)/2:pad.l+(W-pad.l-pad.r)*i/(rows.length-1);}function Y(y){return pad.t+(H-pad.t-pad.b)*(ymax-y)/(ymax-ymin);}
  g.font='10px ui-monospace,Menlo,monospace';g.strokeStyle=css('--grid');g.fillStyle=css('--faint');g.lineWidth=1;
  for(var i=0;i<=4;i++){var y=ymin+(ymax-ymin)*i/4,label=activeStability==='condition'?Math.pow(10,y).toPrecision(3):y.toFixed(activeStability==='rmse'?2:3);g.beginPath();g.moveTo(pad.l,Y(y));g.lineTo(W-pad.r,Y(y));g.stroke();g.textAlign='right';g.fillText(label,pad.l-8,Y(y)+3);}
  g.strokeStyle=css('--cinnabar');g.lineWidth=2;g.beginPath();rows.forEach(function(r,i){var value=raw(r);if(!finite(value)||(activeStability==='condition'&&value<=0))return;var y=activeStability==='condition'?Math.log10(value):value;if(i===0)g.moveTo(X(i),Y(y));else g.lineTo(X(i),Y(y));});g.stroke();
  rows.forEach(function(r,i){var value=raw(r);if(!finite(value)||(activeStability==='condition'&&value<=0))return;var y=activeStability==='condition'?Math.log10(value):value;g.fillStyle=r.rank<5?css('--amber'):css('--cinnabar');g.beginPath();g.arc(X(i),Y(y),r.rank<5?5:3.8,0,Math.PI*2);g.fill();g.fillStyle=css('--ink2');g.textAlign=i===0?'left':(i===rows.length-1?'right':'center');g.fillText(W<650?r.date.slice(5):r.date,X(i),H-20);});
  g.fillStyle=css('--faint');g.textAlign='left';g.fillText(labels[activeStability],pad.l,pad.t-8);
  var actual=finiteRows.map(raw),lo=Math.min.apply(null,actual),hi=Math.max.apply(null,actual),minRank=Math.min.apply(null,rows.map(function(r){return r.rank;}));
  document.getElementById('stabilityReadout').innerHTML='<span><span class="k">metric </span><span class="v">'+labels[activeStability]+'</span></span><span><span class="k">range </span><span class="v">'+fmt(lo,activeStability==='condition'?2:4)+' – '+fmt(hi,activeStability==='condition'?2:4)+'</span></span><span><span class="k">minimum effective rank </span><span class="v">'+minRank+'/5</span></span><span><span class="k">amber point </span><span class="v">rank deficient</span></span>';
}
function boot(){
  var expiry=document.getElementById('expirySelect');DATA.smiles.forEach(function(s,i){var o=document.createElement('option');o.value=String(i);o.textContent=s.expiry+' · T='+s.T.toFixed(3);expiry.appendChild(o);});expiry.addEventListener('change',drawSmile);
  document.querySelectorAll('[data-error-target]').forEach(function(button){button.addEventListener('click',function(){document.querySelectorAll('[data-error-target]').forEach(function(b){b.setAttribute('aria-pressed','false');});button.setAttribute('aria-pressed','true');activeError=button.getAttribute('data-error-target');drawErrors();});});
  var stability=document.getElementById('stabilitySelect');stability.addEventListener('change',function(){activeStability=stability.value;drawStability();});
  document.getElementById('themeToggle').addEventListener('click',function(){var root=document.documentElement,current=root.getAttribute('data-theme');if(!current){current=window.matchMedia&&window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}root.setAttribute('data-theme',current==='dark'?'light':'dark');drawSmile();drawErrors();drawStability();});
  var timer;window.addEventListener('resize',function(){clearTimeout(timer);timer=setTimeout(function(){drawSmile();drawErrors();drawStability();},120);});drawSmile();drawErrors();drawStability();
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();
"""


def render_html(artifacts: Mapping[str, Mapping[str, Any]], tag: str) -> str:
    snapshot = artifacts["snapshot"]
    surface = artifacts["surface"]
    localvol = artifacts["localvol"]
    heston = artifacts["heston"]
    slv = artifacts["slv"]
    diagnostics = artifacts["diagnostics"]
    per_expiry = surface["per_expiry"]
    spot = float(surface["s0"])
    fetched_at = str(surface["fetched_at"])
    snapshot_date = fetched_at[:10]

    raw_quotes = [
        quote
        for expiry in snapshot["expiries"]
        for quote in expiry["quotes"]
    ]
    raw_quote_count = len(raw_quotes)
    outside_band = sum(
        1
        for quote in raw_quotes
        if _number(quote.get("bid")) is not None
        and _number(quote.get("ask")) is not None
        and float(quote["bid"]) > 0.0
        and float(quote["ask"]) > 0.0
        and not float(quote["bid"]) <= float(quote["last"]) <= float(quote["ask"])
    )
    oi_alias = bool(raw_quotes) and all(
        _number(quote.get("volume")) == _number(quote.get("oi"))
        for quote in raw_quotes
    )
    oi_total = sum(int(_number(quote.get("oi")) or 0) for quote in raw_quotes)
    expiry_oi = [
        (
            float(expiry["T_years"]),
            sum(int(_number(quote.get("oi")) or 0) for quote in expiry["quotes"]),
        )
        for expiry in snapshot["expiries"]
    ]
    front_oi = min(expiry_oi, key=lambda item: item[0])[1]
    long_oi = max(expiry_oi, key=lambda item: item[0])[1]
    front_oi_share = 100.0 * front_oi / oi_total if oi_total else math.nan
    long_oi_share = 100.0 * long_oi / oi_total if oi_total else math.nan
    node_count = sum(len(expiry["points"]) for expiry in per_expiry)
    direct_cells, interior_cells, flat_cells = _grid_cell_counts(surface)

    smoothing = heston["target_smoothing"]
    smoothing_points = float(smoothing["raw_points_rmse_iv"]) * 100.0
    smoothing_grid = float(smoothing["raw_grid_rmse_iv"]) * 100.0
    heston_prepared = float(heston["overall_rmse_iv"]) * 100.0
    heston_raw = float(heston["raw_overall_rmse_iv"]) * 100.0
    localvol_prepared = float(localvol["overall_rmse_iv"]) * 100.0
    localvol_raw = float(localvol["raw_overall_rmse_iv"]) * 100.0
    slv_prepared = float(slv["overall_rmse_iv"]) * 100.0
    slv_raw = float(slv["raw_overall_rmse_iv"]) * 100.0
    baseline_raw = _flat_atm_baseline(per_expiry) * 100.0
    heston_improvement = 100.0 * (1.0 - heston_raw / baseline_raw) if baseline_raw else math.nan
    fit_gate_pass = heston_raw <= FIT_GATE_VOL_POINTS
    gate_multiple = heston_raw / FIT_GATE_VOL_POINTS

    params = heston["params"]
    feller = float(heston["feller"])
    feller_reading = (
        "barely above one in the saved fit"
        if 1.0 <= feller <= 1.05
        else ("satisfied in the saved fit" if feller > 1.05 else "violated in the saved fit")
    )
    calibration_spec = heston["calibration_spec"]
    lower_bounds = dict(zip(PARAMETERS, calibration_spec["bounds"][0]))
    upper_bounds = dict(zip(PARAMETERS, calibration_spec["bounds"][1]))
    local_bound_hits = [
        name
        for name in PARAMETERS
        if heston["bound_hits"][name]["lower"] or heston["bound_hits"][name]["upper"]
    ]
    local_svd = heston["jacobian"]["svd"]["fixed_economic"]
    local_condition = _number(local_svd.get("condition_number"))
    local_rank = int(local_svd["policy_effective_rank"])
    weakest = local_svd["right_singular_vectors"][-1]
    weakest_components = sorted(
        weakest["components"].items(), key=lambda item: abs(float(item[1])), reverse=True
    )
    weakest_label = " + ".join(name for name, _value in weakest_components[:2])
    bootstrap = heston["bootstrap"]
    bootstrap_success = int(bootstrap["successful_replicates"])
    bootstrap_requested = int(bootstrap["requested_replicates"])
    bootstrap_success_pct = 100.0 * bootstrap_success / bootstrap_requested

    cross_dates = diagnostics["included"]
    cross_date_count = len(cross_dates)
    cross_stability = diagnostics["stability"]
    cross_identification = cross_stability.get("identification", {})
    cross_condition_range = cross_identification.get("condition_number_range")
    cross_min_rank = cross_identification.get("minimum_policy_effective_rank")
    cross_rmse_range = cross_stability.get("weighted_rmse_iv", {})
    cross_rmse_min = 100.0 * float(cross_rmse_range.get("min", math.nan))
    cross_rmse_max = 100.0 * float(cross_rmse_range.get("max", math.nan))
    cross_exclusions = diagnostics.get("exclusions", [])
    cross_condition_min = (
        float(cross_condition_range[0]) if isinstance(cross_condition_range, list) else math.nan
    )
    cross_condition_max = (
        float(cross_condition_range[1]) if isinstance(cross_condition_range, list) else math.nan
    )
    rho_stability = cross_stability["parameters"]["rho"]
    cross_rho_min = float(rho_stability["min"])
    cross_rho_max = float(rho_stability["max"])
    cross_rho_cv = _number(rho_stability.get("cv_abs_mean"))
    cross_feller = cross_stability["feller_ratio"]
    frequent_cross_bounds = [
        name
        for name, frequency in cross_stability["bound_hit_frequency"].items()
        if float(frequency) >= 0.5
    ]
    parity_quality = cross_stability.get("parity_quality", {})
    parity_failed_pillars = parity_quality.get("failed_pillars", [])
    parity_max_rmse_pct_forward = (
        100.0 * float(parity_quality.get("maximum_rmse_forward_ratio", 0.0))
    )
    parity_rate_sensitivity = _number(
        parity_quality.get("near_atm_sensitivity", {}).get(
            "maximum_absolute_implied_rate_difference"
        )
    )
    static_arbitrage = cross_stability.get("static_arbitrage", {})
    static_affected_expiries = int(static_arbitrage.get("affected_expiry_count", 0))
    static_convex_violations = int(static_arbitrage.get("convex_slope_violations", 0))
    rate_values = [float(expiry["r"]) * 100.0 for expiry in per_expiry]
    carry_values = [float(expiry["q"]) * 100.0 for expiry in per_expiry]

    model_by_t = {
        name: {round(float(row["T"]), 12): row for row in payload["per_expiry"]}
        for name, payload in (("heston", heston), ("localvol", localvol), ("slv", slv))
    }
    smiles: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    node_rows: list[str] = []
    error_rows: list[str] = []
    expiry_rows: list[str] = []
    future_gaps: list[float] = []
    futures = artifacts.get("futures", {})
    futures_by_expiry = {
        str(row.get("expiry_date")): row
        for row in futures.get("quotes", [])
        if isinstance(row, Mapping) and row.get("expiry_date")
    } if isinstance(futures, Mapping) else {}

    for expiry in per_expiry:
        maturity = float(expiry["T"])
        key = round(maturity, 12)
        expiry_date = str(expiry["expiry_date"])
        forward = float(expiry["forward"])
        points = [
            {"strike": float(point[0]), "iv": float(point[1])}
            for point in expiry["points"]
        ]
        smiles.append(
            {
                "expiry": expiry_date,
                "T": maturity,
                "forward": forward,
                "nodes": points,
            }
        )
        model_row = {
            name: {
                "prepared": float(model_by_t[name][key]["rmse_iv"]) * 100.0,
                "raw": float(model_by_t[name][key]["raw_rmse_iv"]) * 100.0,
            }
            for name in ("heston", "localvol", "slv")
        }
        errors.append({"expiry": expiry_date, "T": maturity, **model_row})
        error_rows.append(
            "<tr>"
            f"<td>{_escape(expiry_date)}</td><td class='n'>{maturity:.3f}</td><td class='n'>{len(points)}</td>"
            f"<td class='n'>{model_row['heston']['prepared']:.3f}</td><td class='n'>{model_row['heston']['raw']:.3f}</td>"
            f"<td class='n'>{model_row['localvol']['raw']:.3f}</td><td class='n'>{model_row['slv']['raw']:.3f}</td>"
            "</tr>"
        )
        for point in points:
            node_rows.append(
                "<tr>"
                f"<td>{_escape(expiry_date)}</td><td class='n'>{maturity:.3f}</td>"
                f"<td class='n'>{point['strike']:.1f}</td><td class='n'>{point['strike']/forward:.4f}</td>"
                f"<td class='n'>{point['iv']*100.0:.3f}%</td>"
                "</tr>"
            )
        future = futures_by_expiry.get(expiry_date)
        close = _number(future.get("close")) if isinstance(future, Mapping) else None
        gap = None if close is None else 100.0 * (forward - close) / close
        if gap is not None:
            future_gaps.append(abs(gap))
        expiry_rows.append(
            "<tr>"
            f"<td>{_escape(expiry_date)}</td><td class='n'>{maturity:.3f}</td><td class='n'>{len(points)}</td>"
            f"<td class='n'>{forward:.1f}</td><td class='n'>{_fmt(close,1)}</td><td class='n'>{_fmt(gap,3,'%')}</td>"
            f"<td class='n'>{float(expiry['r'])*100.0:+.2f}%</td><td class='n'>{float(expiry['q'])*100.0:+.2f}%</td>"
            "</tr>"
        )

    param_row_parts: list[str] = []
    for name, description in (
        ("v0", "initial variance; short-dated volatility level"),
        ("kappa", "mean-reversion speed; term-structure control"),
        ("theta", "long-run variance"),
        ("sigma", "vol-of-vol; smile curvature"),
        ("rho", "spot/variance correlation; skew"),
    ):
        hit = heston["bound_hits"][name]
        boundary = (
            '<span class="chip warn">LOWER BOUND</span>'
            if hit["lower"]
            else (
                '<span class="chip warn">UPPER BOUND</span>'
                if hit["upper"]
                else '<span class="chip neutral">INTERIOR</span>'
            )
        )
        param_row_parts.append(
            f"<tr><td><code>{name}</code></td><td class='n'>{_fmt_param(params[name])}</td>"
            f"<td class='n'>[{_fmt_param(lower_bounds[name])}, {_fmt_param(upper_bounds[name])}]</td>"
            f"<td>{boundary}</td><td>{description}</td></tr>"
        )
    param_rows = "".join(param_row_parts)

    bootstrap_rows: list[str] = []
    for name in PARAMETERS:
        distribution = bootstrap["parameter_quantiles"][name]
        hit_rate = bootstrap["bound_hit_rates"][name]["either"]
        bootstrap_rows.append(
            "<tr>"
            f"<td><code>{name}</code></td><td class='n'>{_fmt_param(params[name])}</td>"
            f"<td class='n'>{_fmt_param(distribution['q05'])}</td>"
            f"<td class='n'>{_fmt_param(distribution['q50'])}</td>"
            f"<td class='n'>{_fmt_param(distribution['q95'])}</td>"
            f"<td class='n'>{100.0 * float(hit_rate):.1f}%</td>"
            "</tr>"
        )

    stability_rows: list[dict[str, Any]] = []
    stability_table_rows: list[str] = []
    for row in cross_dates:
        fixed = row["jacobian"]["svd"]["fixed_economic"]
        best = row["best"]
        universe = row["node_universe"]
        hit_names = [
            name
            for name in PARAMETERS
            if best["bound_hits"][name]["lower"] or best["bound_hits"][name]["upper"]
        ]
        chart_row = {
            "date": row["trade_date"],
            "params": {name: float(best["params"][name]) for name in PARAMETERS},
            "rmse": 100.0 * float(best["weighted_rmse_iv"]),
            "condition": _number(fixed.get("condition_number")),
            "rank": int(fixed["policy_effective_rank"]),
            "nodes": int(universe["node_count"]),
            "expiries": int(universe["expiry_count"]),
        }
        stability_rows.append(chart_row)
        stability_table_rows.append(
            "<tr>"
            f"<td>{_escape(row['trade_date'])}</td>"
            f"<td class='n'>{chart_row['nodes']}</td><td class='n'>{chart_row['expiries']}</td>"
            f"<td class='n'>{chart_row['rmse']:.3f}</td>"
            f"<td class='n'>{_fmt(chart_row['condition'], 2)}</td><td class='n'>{chart_row['rank']}/5</td>"
            f"<td>{_escape(', '.join(hit_names) if hit_names else 'none')}</td>"
            "</tr>"
        )
    exclusion_items = "".join(
        f"<li><code>{_escape(row.get('tag', row.get('trade_date', 'candidate')))}</code>: "
        f"{_escape(row.get('reason', 'unspecified exclusion'))}</li>"
        for row in cross_exclusions
        if isinstance(row, Mapping)
    ) or "<li>None.</li>"

    if future_gaps:
        fetched_for_demo = futures.get("fetched_for_demo")
        fetched_suffix = (
            f"; the futures artifact was fetched on {_escape(fetched_for_demo)}"
            if isinstance(fetched_for_demo, str)
            else ""
        )
        futures_diagnostic = (
            f"The parity forwards are within {max(future_gaps):.3f}% of asynchronous "
            f"same-date IM daily closes. The option chain is an intraday snapshot at "
            f"{_escape(fetched_at)}, while the futures values are end-of-day closes{fetched_suffix}. "
            "This is a scale diagnostic, not synchronized executable-price validation. "
        )
    else:
        futures_diagnostic = (
            "No optional IM futures artifact is present, so this report makes no external "
            "forward-level comparison. "
        )

    market_evidence_rows = "".join(
        "<tr>"
        f"<td><a href='{_escape(item['url'])}'>{_escape(item['fact'])}</a></td>"
        f"<td>{_escape(item['value'])}</td><td>{_escape(item['scope'])}</td>"
        "</tr>"
        for item in MARKET_EVIDENCE
    )

    optional_blocks: list[str] = []
    barrier = artifacts.get("barrier")
    if isinstance(barrier, Mapping):
        rows = "".join(
            "<tr>"
            f"<td>{_escape(name)}</td><td class='n'>{_fmt(model.get('mc'),3)}</td>"
            f"<td class='n'>{_fmt(model.get('pde'),3)}</td><td class='n'>{_fmt(model.get('gap_pct'),2,'%')}</td>"
            "</tr>"
            for name, model in barrier["models"].items()
            if isinstance(model, Mapping)
        )
        optional_blocks.append(
            "<h3>Barrier scenario</h3><p>The saved 0.45-year up-and-out scenario is inside the vanilla horizon. "
            "It demonstrates model-dynamics dispersion, not a new calibration observation.</p>"
            f"<div class='tbl-wrap'><table><thead><tr><th>Model</th><th class='n'>MC</th><th class='n'>PDE</th><th class='n'>Gap / MC</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    snowball = artifacts.get("snowball")
    if isinstance(snowball, Mapping):
        rows = "".join(
            "<tr>"
            f"<td>{_escape(name)}</td><td class='n'>{_fmt(model.get('mc_pct_initial'),3,'%')}</td>"
            f"<td class='n'>{_fmt(model.get('pde_pct_initial'),3,'%')}</td><td class='n'>{_fmt(model.get('ko_probability')*100.0 if _number(model.get('ko_probability')) is not None else None,1,'%')}</td>"
            "</tr>"
            for name, model in snowball["models"].items()
            if isinstance(model, Mapping)
        )
        optional_blocks.append(
            "<h3>Snowball extrapolation scenario</h3><p>The Snowball maturity is 2.0 years while the frozen vanilla surface ends at 0.95 years. "
            "The results are therefore a dynamics-and-extrapolation scenario, not an in-domain validation.</p>"
            f"<div class='tbl-wrap'><table><thead><tr><th>Model</th><th class='n'>MC / initial</th><th class='n'>PDE / initial</th><th class='n'>KO probability</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    hedging = artifacts.get("hedging")
    if isinstance(hedging, Mapping):
        rows = "".join(
            "<tr>"
            f"<td>{_escape(name)}</td><td class='n'>{_fmt(model.get('initial_delta'),4)}</td>"
            f"<td class='n'>{_fmt(model.get('total_rebalance_units'),3)}</td><td class='n'>{_fmt(model.get('total_residual_pnl'),3)}</td>"
            "</tr>"
            for name, model in hedging["models"].items()
            if isinstance(model, Mapping)
        )
        optional_blocks.append(
            "<h3>Fixed-surface hedging scenario</h3><p>The saved deterministic down-up path omits funding, costs and recalibration. "
            "It shows hedge sensitivity to model choice; it is not a realized or holdout backtest.</p>"
            f"<div class='tbl-wrap'><table><thead><tr><th>Model</th><th class='n'>Initial delta</th><th class='n'>Turnover units</th><th class='n'>Residual P&amp;L</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    optional_html = "".join(optional_blocks) or (
        "<div class='note warn'><span class='t'>Optional scenarios absent</span>Barrier, Snowball and hedging artifacts were not present. "
        "The core calibration verdict remains complete without them.</div>"
    )

    fixture_banner = "" if tag != "sample" else (
        "<div class='note risk'><span class='t'>Synthetic test fixture</span>This report uses the deterministic sample snapshot. "
        "It validates the pipeline and document contract; it is not live MO market evidence.</div>"
    )
    report_data = json.dumps(
        {"smiles": smiles, "errors": errors, "stability": stability_rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>MO Heston and SLV calibration — evidence-led explainer</title>
<style>{CSS}</style>
</head>
<body>
<div class="topnav"><div class="topnav-inner"><span class="brand">CFFEX · MO · {_escape(snapshot_date)}</span><nav>
<a href="#s1">Verdict</a><a href="#s2">Quotes</a><a href="#s3">Surface</a><a href="#s4">Heston</a><a href="#s5">Dynamics</a><a href="#s6">Dates</a><a href="#s7">Limits</a><a href="#s8">Appendix</a>
</nav><button id="themeToggle" type="button" aria-label="Toggle light and dark theme">Theme</button></div></div>

<header class="hero">
<p class="eyebrow">Technical explainer · CFFEX MO + QuantArk</p>
<h1>Can MO options support a well-calibrated Heston or SLV model?</h1>
<p class="kicker">The exchange lists a real, active strike surface, but the expanded evidence still answers <b>no</b>. The intraday fit misses raw quotes and lands on parameter bounds; the independent official-settlement panel shows unstable fit quality, bound pressure, and at least one locally rank-deficient date.</p>
<div class="evidence-banner"><b>Two cohorts · never pooled</b>The primary fit is one AKShare/Sina bid/ask-midpoint replay. Cross-date evidence comes from independently dated official CFFEX end-of-day settlements. Settlements have source hashes and genuine dates, but no executable bid/ask; their parity-normalized results diagnose stability and public-data quality, not intraday execution.</div>
<div class="hero-meta"><span>Underlying <b>CSI 1000 · 000852.SH</b></span><span>Snapshot <b>{_escape(fetched_at)}</b></span><span>Spot <b>{spot:,.2f}</b></span><span>Expiries <b>{len(per_expiry)}</b></span><span>Artifact tag <b>{_escape(tag)}</b></span></div>
{fixture_banner}
<div class="toc">
<a href="#s1"><span class="no">§1</span>Decision: what the artifacts prove</a><a href="#s2"><span class="no">§2</span>Listed strikes and snapshot marks</a>
<a href="#s3"><span class="no">§3</span>From parity to a prepared surface</a><a href="#s4"><span class="no">§4</span>Heston fit, Jacobian, and bootstrap</a>
<a href="#s5"><span class="no">§5</span>Local vol, SLV, and numerics</a><a href="#s6"><span class="no">§6</span>Official-settlement cross-date fit</a>
<a href="#s7"><span class="no">§7</span>Exotics, comparison, limitations</a><a href="#s8"><span class="no">§8</span>Artifacts and reproduction</a>
</div></header>

<main>
<section id="s1"><p class="eyebrow">§1 · Decision</p><h2>A listed market can still be calibration-poor</h2>
<p>The intraday study contains <b class="num">{node_count}</b> usable OTM observations across <b class="num">{len(per_expiry)}</b> expiries, and the independent settlement panel admits <b class="num">{cross_date_count}</b> genuine dates after strict source, expiry, liquidity, parity-quality, and coverage gates. More data changed the epistemic status from “not tested” to <b>tested and still not robust</b>: raw-fit error remains material, κ/σ hit local bounds, and the cross-date minimum effective Jacobian rank is {_escape(cross_min_rank)}/5.</p>
<div class="metric-grid">
<div class="metric-card"><span class="label">Research verdict</span><span class="value"><span class="chip fail">NOT ROBUST</span></span><span class="sub">evidence rejects production identification</span></div>
<div class="metric-card"><span class="label">Study-defined fit threshold</span><span class="value"><span class="chip {'pass' if fit_gate_pass else 'fail'}">{'PASS' if fit_gate_pass else 'FAIL'}</span></span><span class="sub">{heston_raw:.4f} vs {FIT_GATE_VOL_POINTS:.2f} vol pts · {gate_multiple:.1f}× threshold</span></div>
<div class="metric-card"><span class="label">Local fitted-bound hits</span><span class="value"><span class="chip {'warn' if local_bound_hits else 'pass'}">{len(local_bound_hits)} / 5</span></span><span class="sub">{', '.join(local_bound_hits) if local_bound_hits else 'none'} · persisted calibration box</span></div>
<div class="metric-card"><span class="label">Cross-date evidence</span><span class="value"><span class="chip {'pass' if cross_date_count >= 5 else 'warn'}">{cross_date_count} DATES</span></span><span class="sub">official EOD settlements · {len(cross_exclusions)} excluded candidates</span></div>
<div class="metric-card"><span class="label">Raw OTM geometry</span><span class="value">{node_count} nodes</span><span class="sub">{len(per_expiry)} expiries · {len(surface['strikes'])} common strikes</span></div>
<div class="metric-card"><span class="label">Cross-date fit range</span><span class="value">{cross_rmse_min:.3f}–{cross_rmse_max:.3f}</span><span class="sub">weighted RMSE · vol points</span></div>
</div>
<div class="note risk"><span class="t">Decision boundary</span>Optimizer success, full numerical rank on one date, and a near-one Feller ratio are not substitutes for stable identification. Bounds, objective, start policy, Jacobian scale, bootstrap seed, failures, and cross-date exclusions are now persisted. The evidence is stronger; the promotion verdict is not.</div></section>

<section id="s2"><p class="eyebrow">§2 · Quote geometry</p><h2>A broad listed chain, but not an audit-grade calibration feed</h2>
<p>CFFEX defines MO as a European, cash-settled CSI 1000 option with six listed contract months. The frozen adapter snapshot contains paired calls and puts; stage 02 uses the stored bid/ask midpoint when both sides exist, otherwise last, then keeps the OTM side around the parity-implied forward.</p>
<div class="metric-grid">
<div class="metric-card"><span class="label">Snapshot records</span><span class="value">{raw_quote_count}</span><span class="sub">call and put records before OTM selection</span></div>
<div class="metric-card"><span class="label">Usable OTM observations</span><span class="value">{node_count}</span><span class="sub">the actual raw calibration language</span></div>
<div class="metric-card"><span class="label">Last outside stored band</span><span class="value">{outside_band}</span><span class="sub">midpoint avoids stale-last contamination</span></div>
<div class="metric-card"><span class="label">Reported OI sum</span><span class="value">{oi_total:,}</span><span class="sub">not traded volume; front concentration {front_oi_share:.1f}%</span></div>
<div class="metric-card"><span class="label">Longest-expiry OI share</span><span class="value">{long_oi_share:.2f}%</span><span class="sub">coverage does not imply balanced depth</span></div>
<div class="metric-card"><span class="label">Volume/OI alias check</span><span class="value"><span class="chip {'warn' if oi_alias else 'neutral'}">{'IDENTICAL' if oi_alias else 'NOT IDENTICAL'}</span></span><span class="sub">adapter liquidity proxy, disclosed explicitly</span></div>
</div>
<figure class="wide"><div class="fig-frame"><div class="controls"><div class="ctl"><label for="expirySelect">Expiry</label><select id="expirySelect"></select></div><div class="ctl"><label>Chart language</label><div class="chiprow"><span class="chip neutral">slate = raw OTM IV</span><span class="chip warn">amber = parity forward</span></div></div></div><canvas id="smileCanvas" height="350" aria-label="MO raw OTM implied-volatility smile by expiry"></canvas><div class="readout" id="smileReadout" aria-live="polite"></div></div><figcaption><span class="figno">Fig 1 · Raw smile explorer</span>The chart deliberately isolates the stage-02 OTM observations. Pointwise Heston model rows are persisted for Jacobian and error audit, while the fit summaries below keep prepared-target and raw-quote errors separate.</figcaption></figure>
<details class="fallback-evidence"><summary>Fallback evidence table — all raw OTM surface points</summary><div class="body"><div class="tbl-wrap"><table><thead><tr><th>Expiry</th><th class="n">T</th><th class="n">Strike</th><th class="n">K/F</th><th class="n">Raw IV</th></tr></thead><tbody>{''.join(node_rows)}</tbody></table></div></div></details>
<h3>Expiry-level market reconstruction</h3><div class="tbl-wrap"><table><thead><tr><th>Expiry</th><th class="n">T</th><th class="n">OTM nodes</th><th class="n">Parity F</th><th class="n">IM daily close</th><th class="n">F vs IM</th><th class="n">r</th><th class="n">q / carry</th></tr></thead><tbody>{''.join(expiry_rows)}</tbody></table></div>
<div class="note warn"><span class="t">Carry is diagnostic</span>{futures_diagnostic}The separate slope-derived rates span {min(rate_values):+.2f}% to {max(rate_values):+.2f}% and carry spans {min(carry_values):+.2f}% to {max(carry_values):+.2f}%. Treat <code>q</code> as implied futures basis/carry, not a dividend forecast.</div></section>

<section id="s3"><p class="eyebrow">§3 · Surface preparation</p><h2>Observed strikes and model-filled cells stay separate</h2>
<div class="pipe wide"><div class="box"><b>Adapter snapshot</b>bid/ask midpoint, otherwise last</div><div class="arrow">→</div><div class="box"><b>Put-call parity</b>forward, discount factor, implied carry</div><div class="arrow">→</div><div class="box"><b>OTM filter</b>puts below F, calls at/above F</div><div class="arrow">→</div><div class="box"><b>Rectangular grid</b>linear interior + flat wings</div><div class="arrow">→</div><div class="box"><b>SABR + calendar</b>prepared target for LV/Heston/SLV</div></div>
<div class="metric-grid">
<div class="metric-card"><span class="label">Direct grid cells</span><span class="value">{direct_cells}</span><span class="sub">observed strikes on the common grid</span></div>
<div class="metric-card"><span class="label">Interior interpolation</span><span class="value">{interior_cells}</span><span class="sub">model-filled, not new observations</span></div>
<div class="metric-card"><span class="label">Flat-wing extrapolation</span><span class="value">{flat_cells}</span><span class="sub">outside an expiry's observed strike range</span></div>
<div class="metric-card"><span class="label">Smoothing vs raw points</span><span class="value">{smoothing_points:.4f} vol pts</span><span class="sub">SABR target minus {node_count} OTM observations</span></div>
<div class="metric-card"><span class="label">Smoothing vs raw grid</span><span class="value">{smoothing_grid:.4f} vol pts</span><span class="sub">includes rectangular interpolation effects</span></div>
<div class="metric-card"><span class="label">Calendar adjustments</span><span class="value">{_escape(smoothing.get('calendar_adjusted_nodes','—'))}</span><span class="sub">total-variance projection count</span></div>
</div>
<div class="note warn"><span class="t">The fit target is already modeled</span>Heston is calibrated to the SABR-prepared target, not directly to untouched raw quotes. Both prepared and raw errors are therefore shown everywhere; the raw error is compared with a study-defined 0.10-vol-point (10-vol-bp) threshold chosen solely to keep the MO and CFETS reports on one scale. It is not an exchange or industry acceptance standard.</div></section>

<section id="s4"><p class="eyebrow">§4 · Heston identification</p><h2>Full local rank does not rescue a bound-pinned fit</h2>
<div class="eq"><span class="lbl">risk-neutral equity Heston</span>dS/S = (r−q)dt + √v dW<sub>S</sub><br>dv = κ(θ−v)dt + σ√v dW<sub>v</sub>, &nbsp; d⟨W<sub>S</sub>,W<sub>v</sub>⟩ = ρdt</div>
<div class="metric-grid">
<div class="metric-card"><span class="label">Prepared-target RMSE</span><span class="value">{heston_prepared:.4f} vol pts</span><span class="sub">after SABR and calendar preparation</span></div>
<div class="metric-card"><span class="label">Raw-quote RMSE</span><span class="value">{heston_raw:.4f} vol pts</span><span class="sub">{heston_raw*100.0:.1f} vol bp</span></div>
<div class="metric-card"><span class="label">Flat-ATM raw baseline</span><span class="value">{baseline_raw:.4f} vol pts</span><span class="sub">Heston improves only {heston_improvement:.1f}%</span></div>
<div class="metric-card"><span class="label">Feller ratio</span><span class="value">{feller:.4f}</span><span class="sub">{feller_reading}</span></div>
<div class="metric-card"><span class="label">Fixed-scale Jacobian</span><span class="value">{_fmt(local_condition,2)}</span><span class="sub">condition number · effective rank {local_rank}/5</span></div>
<div class="metric-card"><span class="label">Weakest local direction</span><span class="value">{_escape(weakest_label)}</span><span class="sub">largest absolute components of the final right-singular vector</span></div>
<div class="metric-card"><span class="label">Fitted-bound hits</span><span class="value">{len(local_bound_hits)} / 5</span><span class="sub">{_escape(', '.join(local_bound_hits) if local_bound_hits else 'none')}</span></div>
<div class="metric-card"><span class="label">Multiplier bootstrap</span><span class="value">{bootstrap_success} / {bootstrap_requested}</span><span class="sub">{bootstrap_success_pct:.1f}% successful conditional reweights</span></div>
</div>
<div class="tbl-wrap"><table><thead><tr><th>Parameter</th><th class="n">Fit</th><th class="n">Persisted box</th><th>Fit location</th><th>Primary role</th></tr></thead><tbody>{param_rows}</tbody></table></div>
<div class="eq"><span class="lbl">quote-only local sensitivity</span>J<sub>ij</sub> = ∂σ<sup>Heston</sup><sub>i</sub>/∂θ<sub>j</sub>, &nbsp; J<sub>scaled</sub> = J diag(0.01, 1, 0.01, 0.1, 0.1)<br>J<sub>scaled</sub> = UΣVᵀ, &nbsp; cond = s<sub>max</sub>/s<sub>min</sub></div>
<p>The finite-difference Jacobian is bound-aware and excludes the soft Feller residual, so the penalty cannot masquerade as quote information. The fixed economic scale makes units explicit. On this date it is numerically full rank, but κ and σ are active at their upper bounds; a box-constrained optimum can be full rank and still fail to reveal where the unconstrained market optimum lies.</p>
<h3>Seeded maturity-stratified multiplier bootstrap</h3>
<div class="tbl-wrap"><table><thead><tr><th>Parameter</th><th class="n">Fit</th><th class="n">q05</th><th class="n">q50</th><th class="n">q95</th><th class="n">Any-bound rate</th></tr></thead><tbody>{''.join(bootstrap_rows)}</tbody></table></div>
<div class="note warn"><span class="t">Conditional influence evidence, not a confidence interval</span>Each replicate applies seeded Exp(1) weights normalized within maturity, then repeats the same start, bounds, objective and soft-Feller policy. The quantiles show sensitivity to prepared-target node influence. They are not quote-time sampling intervals, do not repair the single intraday snapshot, and cannot remove the fact that the target was SABR prepared.</div></section>

<section id="s5"><p class="eyebrow">§5 · Dynamics</p><h2>Local volatility and SLV amplify the preparation dependency</h2>
<p>Dupire differentiates the prepared surface. SLV then calibrates leverage to that Dupire target while inheriting the saved Heston process. These are valid numerical demonstrations, but their vanilla repricing errors do not support model promotion.</p>
<div class="metric-grid">
<div class="metric-card"><span class="label">Local-vol prepared / raw</span><span class="value">{localvol_prepared:.3f} / {localvol_raw:.3f}</span><span class="sub">vol points · PDE repricing</span></div>
<div class="metric-card"><span class="label">SLV prepared / raw</span><span class="value">{slv_prepared:.3f} / {slv_raw:.3f}</span><span class="sub">vol points · PDE repricing</span></div>
<div class="metric-card"><span class="label">Local-vol range</span><span class="value">{float(localvol['lv_min'])*100.0:.1f}%–{float(localvol['lv_max'])*100.0:.1f}%</span><span class="sub">derived from the prepared target</span></div>
<div class="metric-card"><span class="label">SLV leverage range</span><span class="value">{float(slv['leverage_min']):.3f}–{float(slv['leverage_max']):.3f}</span><span class="sub">L=1 would be pure Heston scaling</span></div>
<div class="metric-card"><span class="label">Shortest-expiry LV raw</span><span class="value">{errors[0]['localvol']['raw']:.3f} vol pts</span><span class="sub">short-end numerics dominate headline error</span></div>
<div class="metric-card"><span class="label">Shortest-expiry SLV raw</span><span class="value">{errors[0]['slv']['raw']:.3f} vol pts</span><span class="sub">not a production vanilla fit</span></div>
</div>
<div class="note warn"><span class="t">Correct interpretation</span>Local vol and SLV are downstream demonstrations against a repaired surface. They cannot upgrade interpolated cells into liquidity or turn a bound-sensitive Heston parameter vector into identified dynamics. The persisted Jacobian and bootstrap improve auditability; they do not change the information content of the input quotes.</div></section>

<section id="s6"><p class="eyebrow">§6 · Cross-date robustness</p><h2>Genuine dates now expose instability instead of filling the gap</h2>
<p>The panel is deliberately independent of the intraday snapshot: official CFFEX settlement cross sections are source-hashed, holiday-adjusted, filtered to liquid OTM observations, normalized expiry by expiry through put-call parity, and fitted without interpolation, extrapolation or smile smoothing. A date is admitted only under the same persisted configuration and minimum coverage gates. This gives genuine time evidence while preserving the critical limitation that settlement is not executable bid/ask. It is a caller-selected seven-candidate April–July 2026 panel, not a systematic or representative market-history sample.</p>
<div class="metric-grid">
<div class="metric-card"><span class="label">Admitted / excluded dates</span><span class="value">{cross_date_count} / {len(cross_exclusions)}</span><span class="sub">strict source, hash, parity, liquidity and coverage gates</span></div>
<div class="metric-card"><span class="label">Weighted fit range</span><span class="value">{cross_rmse_min:.3f}–{cross_rmse_max:.3f}</span><span class="sub">vol points · equal total weight per expiry</span></div>
<div class="metric-card"><span class="label">Fixed-scale condition range</span><span class="value">{cross_condition_min:.2f}–{cross_condition_max:.2f}</span><span class="sub">objective-weighted quote Jacobian</span></div>
<div class="metric-card"><span class="label">Minimum effective rank</span><span class="value"><span class="chip {'fail' if cross_min_rank is None or int(cross_min_rank) < 5 else 'pass'}">{_escape(cross_min_rank)}/5</span></span><span class="sub">policy cutoff 10⁻³ of largest singular value</span></div>
<div class="metric-card"><span class="label">ρ range / absolute-mean CV</span><span class="value">{cross_rho_min:.3f}–{cross_rho_max:.3f}</span><span class="sub">CV {_fmt(cross_rho_cv,3)} · material skew instability</span></div>
<div class="metric-card"><span class="label">Feller ratio range</span><span class="value">{float(cross_feller['min']):.3f}–{float(cross_feller['max']):.3f}</span><span class="sub">not a fit-quality or identification test</span></div>
<div class="metric-card"><span class="label">Rejected parity pillars</span><span class="value">{len(parity_failed_pillars)}</span><span class="sub">broad ±10% annual-rate and 1%-of-forward RMSE gate</span></div>
<div class="metric-card"><span class="label">Raw convexity violations</span><span class="value">{static_convex_violations}</span><span class="sub">across {static_affected_expiries} expiry cross sections · no repair</span></div>
</div>
<figure class="wide"><div class="fig-frame"><div class="controls"><div class="ctl"><label for="stabilitySelect">Cross-date metric</label><select id="stabilitySelect"><option value="rmse">weighted RMSE</option><option value="condition">Jacobian condition</option><option value="v0">v0</option><option value="kappa">kappa</option><option value="theta">theta</option><option value="sigma">sigma</option><option value="rho">rho</option></select></div><div class="ctl"><label>Chart language</label><div class="chiprow"><span class="chip fail">red = full effective rank</span><span class="chip warn">amber = rank deficient</span></div></div></div><canvas id="stabilityCanvas" height="360" aria-label="MO Heston cross-date fit and parameter stability"></canvas><div class="readout" id="stabilityReadout" aria-live="polite"></div></div><figcaption><span class="figno">Fig 2 · Official-settlement stability explorer</span>Each point is a separate official CFFEX settlement date under one frozen raw-node configuration. Condition numbers use the fixed economic parameter scale and the square-root objective weights. Rolling listed strikes are not interpolated into a fictional common grid.</figcaption></figure>
<details class="fallback-evidence"><summary>Fallback evidence table — every admitted settlement date</summary><div class="body"><div class="tbl-wrap"><table><thead><tr><th>Date</th><th class="n">Nodes</th><th class="n">Expiries</th><th class="n">Weighted RMSE</th><th class="n">Condition</th><th class="n">Rank</th><th>Bound hits</th></tr></thead><tbody>{''.join(stability_table_rows)}</tbody></table></div><p>RMSE is in volatility points. Condition and rank use the fixed-economic-scale, objective-weighted Jacobian.</p></div></details>
<details><summary>Rejected candidates and fail-closed reasons</summary><div class="body"><ul>{exclusion_items}</ul></div></details>
<div class="note warn"><span class="t">Settlement quality is part of the result</span>The primary full-wing parity regression reaches a maximum residual of {parity_max_rmse_pct_forward:.3f}% of forward, and its implied annual rate differs from a diagnostic near-ATM regression by as much as {_fmt(100.0 * parity_rate_sensitivity if parity_rate_sensitivity is not None else None,1)} percentage points. Raw normalized call rows contain {static_convex_violations} irregular-grid convex-slope violations across {static_affected_expiries} expiry cross sections. Nothing is smoothed or repaired. This prevents a clean attribution of all parameter movement to Heston alone.</div>
<div class="note risk"><span class="t">What the panel says</span>Fit quality moves materially, ρ is unstable, and at least one admitted date is locally rank deficient. Parameters also encounter the frozen box across dates{': ' + _escape(', '.join(frequent_cross_bounds)) if frequent_cross_bounds else ''}. Every RMSE is therefore conditional on that economically constrained box, not a measurement of unrestricted Heston capacity. Because the date panel is short and caller-selected, and parity pillars and raw settlements can themselves be noisy, this is evidence that the available public EOD data cannot establish a robust Heston calibration—not proof that every institutional MO feed must fail.</div>
<h3>Intraday cross-expiry context</h3>
<figure class="wide"><div class="fig-frame"><div class="controls"><div class="ctl"><label>Error target</label><div class="chiprow"><button type="button" data-error-target="raw" aria-pressed="true">raw quotes</button><button type="button" data-error-target="prepared" aria-pressed="false">prepared surface</button></div></div></div><canvas id="rmseCanvas" height="340" aria-label="MO per-expiry Heston local-vol and SLV RMSE"></canvas><div class="readout" id="rmseReadout" aria-live="polite"></div></div><figcaption><span class="figno">Fig 3 · Intraday tenor-error explorer</span>These lines remain tied to the one bid/ask-midpoint snapshot. They complement, but are never pooled with, the official-settlement panel.</figcaption></figure>
<details class="fallback-evidence"><summary>Fallback evidence table — every intraday expiry and model error</summary><div class="body"><div class="tbl-wrap"><table><thead><tr><th>Expiry</th><th class="n">T</th><th class="n">Nodes</th><th class="n">Heston prepared</th><th class="n">Heston raw</th><th class="n">LV raw</th><th class="n">SLV raw</th></tr></thead><tbody>{''.join(error_rows)}</tbody></table></div></div></details></section>

<section id="s7"><p class="eyebrow">§7 · Honesty section</p><h2>Market development, model dispersion, and limits</h2>
<div class="tbl-wrap"><table><thead><tr><th>Dimension</th><th>MO index options</th><th>CFETS USD/CNY comparison</th></tr></thead><tbody>
<tr><td>Quote geometry</td><td>listed strike ladder across six contract months</td><td>standard five-delta benchmark across a broad tenor ladder</td></tr>
<tr><td>Data evidence here</td><td>one AKShare/Sina midpoint snapshot plus {cross_date_count} admitted official-settlement dates</td><td>complete CFETS public-composite snapshots in the separate study</td></tr>
<tr><td>Study-defined 10-vol-bp comparison</td><td><span class="chip fail">FAIL</span> {heston_raw:.4f} vol pts</td><td><span class="chip pass">CORE PASS</span> in the separate CFETS report</td></tr>
<tr><td>Identification evidence</td><td>local SVD + {bootstrap_requested} bootstrap reweights + objective-weighted cross-date SVD</td><td>local SVD + cross-date diagnostics in the separate study</td></tr>
<tr><td>Dynamics conclusion</td><td><span class="chip fail">NOT ROBUST</span> bound pressure and cross-date instability</td><td><span class="chip warn">CONDITIONAL</span> fit quality exceeded stability evidence</td></tr>
</tbody></table></div>
<h3>Official market-development context</h3><div class="tbl-wrap"><table><thead><tr><th>Evidence</th><th>Published fact</th><th>Correct scope</th></tr></thead><tbody>{market_evidence_rows}</tbody></table></div>
<div class="note warn"><span class="t">No category error</span>Large exchange-wide index-option turnover and a well-defined MO contract prove that the listed market exists and is economically relevant. They do not prove clean depth at every strike/expiry or identify a five-parameter stochastic-volatility process.</div>
{optional_html}
<h3>Ranked limitations</h3><ol>
<li><b>Cohort mismatch by design:</b> the intraday midpoint fit and official EOD settlement panel are independent evidence classes. They are never pooled, and cross-date settlements are not executable marks.</li>
<li><b>Adapter provenance:</b> the primary AKShare/Sina snapshot has no exchange-native quote timestamps, trades or source hash.</li>
<li><b>Liquidity-field ambiguity:</b> <code>volume</code> is copied from OI and <code>market_open</code> is an OI-presence flag.</li>
<li><b>Prepared-target dependence:</b> {interior_cells + flat_cells} of {direct_cells + interior_cells + flat_cells} rectangular cells are interpolated or flat-wing filled before SABR preparation.</li>
<li><b>Identification scope:</b> Jacobian/SVD is local; multiplier quantiles are conditional node-influence evidence, not sampling confidence intervals; the local stage still uses one deterministic start and a soft Feller penalty.</li>
<li><b>Panel selection:</b> the seven candidate dates were explicitly requested and span only April–July 2026. Six admitted dates are preliminary evidence, not a representative regime history.</li>
<li><b>Settlement normalization:</b> cross-date forwards and discount factors come from raw settlement put-call parity. Pillar quality and exclusions are persisted, but normalization noise cannot be separated cleanly from Heston instability without synchronized rates, forwards and bid/ask; raw convex-slope violations are reported without repair.</li>
<li><b>Bound-conditioned capacity:</b> cross-date RMSE is measured inside one frozen economic parameter box. A documented bound-stress study is required before interpreting it as unrestricted Heston fit capacity.</li>
<li><b>Environment date:</b> the shared stage code fixes valuation date 2026-07-06; this tag matches, but future replay must not silently drift.</li>
<li><b>Exotic scope:</b> barrier, Snowball and hedging scenarios show model dispersion; they are not independent calibration observations or realized backtests.</li>
</ol>
<h3>Artifact flow</h3><div class="pipe wide"><div class="box"><b>01A snapshot</b>AKShare/Sina chain</div><div class="arrow">→</div><div class="box"><b>02–05 models</b>surface · LV · Heston · SLV</div><div class="arrow">→</div><div class="box"><b>01B settlements</b>official dated CFFEX CSVs</div><div class="arrow">→</div><div class="box"><b>10A diagnostics</b>raw cross-date Heston fits</div><div class="arrow">→</div><div class="box"><b>10B explainer</b>two cohorts, one decision</div></div></section>

<section id="s8"><p class="eyebrow">§8 · Appendix</p><h2>Artifact identity and reproduction contract</h2>
<p>This document has no runtime stylesheet, font, image, library, data or network dependency. CSS, JavaScript, tables and chart coordinates are inline. Six core JSON artifacts are mandatory; optional scenario artifacts are validated if present.</p>
<div class="tbl-wrap"><table><thead><tr><th>Artifact</th><th>Status</th><th>Contract</th><th>Role</th></tr></thead><tbody>{_artifact_table(tag, artifacts)}</tbody></table></div>
<details><summary>Official source registry</summary><div class="body"><ul>
{''.join(f'<li><a href="{_escape(item["url"])}">{_escape(item["fact"])}</a></li>' for item in MARKET_EVIDENCE)}
<li>Snapshot provenance carried by the suite: AKShare functions backed by Sina option and index endpoints.</li>
</ul><p>These links document product and aggregate-market context. The HTML does not access them at runtime.</p></div></details>
<details><summary>Exact command and fail-closed behavior</summary><div class="body"><pre><code>.venv/bin/python example/mo_volmodels/10_explainer.py --tag {_escape(tag)}</code></pre><p>The command fails before writing if a core artifact is missing or malformed; snapshot and surface disagree on spot or timestamp; model maturities drift; Heston Feller does not recompute from its parameters; persisted calibration/Jacobian/bootstrap evidence is inconsistent; cross-date source hashes or configurations drift; or model smoothing fingerprints disagree. Optional artifacts are ignored only when absent—if present and inconsistent, they fail validation.</p></div></details>
<details><summary>What would be needed for production promotion</summary><div class="body"><ul><li>Exchange-native quote/trade history with timestamps, sizes, spreads and source identity.</li><li>A longer comparable regime panel plus next-day holdout errors.</li><li>Raw-node objective with spread-aware weights and transparent exclusions.</li><li>Multi-start and free-vs-Feller/bound stress alongside the existing Jacobian/SVD/bootstrap evidence.</li><li>Synchronized rates and forwards plus an arbitrage-controlled surface with explicit interpolation/extrapolation limits.</li><li>Independent exotic and hedge validation over the intended maturity domain.</li></ul></div></details>
<p>Related local report: <a href="../../fx_volmodels/data/fx_calibration_explainer_latest.html">CFETS USD/CNY calibration explainer</a>.</p></section>
</main>
<div class="footer">QuantArk · CFFEX MO volatility-model study · artifact tag {_escape(tag)} · listed-market scale separated from calibration evidence</div>
<script>window.__MO_REPORT_DATA__={report_data};</script>
<script>{JS}</script>
</body></html>
"""


def generate(data_dir: str | Path, tag: str, output: str | Path | None = None) -> Path:
    artifacts = load_artifacts(data_dir, tag)
    document = render_html(artifacts, tag)
    destination = Path(output) if output is not None else Path(data_dir) / f"mo_calibration_explainer_{tag}.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = generate(args.data_dir, args.tag, args.out)
    print(output)


if __name__ == "__main__":
    main()
