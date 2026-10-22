"""Stage 04 — calibrate Heston and report parameter-identification evidence.

Run::

    .venv/bin/python example/mo_volmodels/04_heston_calibration.py \
        [--tag latest|sample] [--bootstrap-reps 16]

Heston is an arbitrage-free stochastic-volatility model: variance follows a CIR process
``dv = kappa(theta - v)dt + sigma sqrt(v) dW``, correlated (rho) with spot.  The main
fit retains the suite's original SABR-prepared OTM-IV target and soft-Feller policy.
Jacobian/SVD and multiplier-bootstrap evidence are additional diagnostics; they do not
replace that calibration or turn a prepared surface into independent market quotes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _heston_diagnostics as hd  # noqa: E402
import _mo_common as mc  # noqa: E402

from quantark.volmodels.black_scholes import bs_call_price, implied_vol_call  # noqa: E402
from quantark.volmodels.heston import (  # noqa: E402
    HestonParams,
    MarketOption,
    calibrate_heston,
    heston_call_prices_vectorized,
)


PARAMETER_NAMES = hd.PARAMETER_NAMES
HESTON_BOUNDS = (
    (1e-6, 1e-3, 1e-4, 1e-3, -0.95),
    (0.5, 3.0, 0.5, 0.7, 0.0),
)
REGULARIZE_FELLER = 0.05
SOLVER_TOLERANCES = {"xtol": 1e-6, "ftol": 1e-6, "gtol": 1e-6}
DEFAULT_BOOTSTRAP_REPS = 16
DEFAULT_BOOTSTRAP_SEED = 20260721


def _params_dict(params: HestonParams) -> dict[str, float]:
    return {name: float(getattr(params, name)) for name in PARAMETER_NAMES}


def _params_vector(params: HestonParams) -> np.ndarray:
    return np.array([getattr(params, name) for name in PARAMETER_NAMES], dtype=float)


def _params_from_vector(values: Sequence[float]) -> HestonParams:
    return HestonParams(**dict(zip(PARAMETER_NAMES, map(float, values))))


def _atm_iv_shortest(per_expiry: Sequence[dict]) -> float:
    """ATM implied vol at the shortest expiry — anchors the Heston level guess."""
    pe0 = min(per_expiry, key=lambda p: p["T"])
    ks = np.array([k for k, _ in pe0["points"]])
    vs = np.array([v for _, v in pe0["points"]])
    return float(vs[np.argmin(np.abs(ks - pe0["forward"]))])


def _initial_guess(per_expiry: Sequence[dict]) -> HestonParams:
    atm = _atm_iv_shortest(per_expiry)
    v_level = float(np.clip(atm**2, 0.01, 0.2))
    return HestonParams(v0=v_level, kappa=2.0, theta=v_level, sigma=0.6, rho=-0.5)


def _calibration_nodes(s0: float, per_expiry: Sequence[dict]) -> list[dict]:
    """Flatten the exact prepared target while retaining its raw-quote counterpart."""
    nodes: list[dict] = []
    for expiry in per_expiry:
        raw_by_k = {float(k): float(v) for k, v in expiry.get("raw_points", [])}
        for strike, target_iv in expiry["points"]:
            strike = float(strike)
            target_iv = float(target_iv)
            nodes.append(
                {
                    "expiry_date": expiry.get("expiry_date"),
                    "T": float(expiry["T"]),
                    "K": strike,
                    "r": float(expiry["r"]),
                    "q": float(expiry["q"]),
                    "target_iv": target_iv,
                    "raw_iv": raw_by_k.get(strike),
                    # Preserve the historical Stage-04 path: the target IV is first
                    # converted to a call-equivalent price, then the native calibrator
                    # inverts it because target="iv".
                    "target_price": float(
                        bs_call_price(
                            s0,
                            strike,
                            float(expiry["T"]),
                            target_iv,
                            float(expiry["r"]),
                            float(expiry["q"]),
                        )
                    ),
                }
            )
    if not nodes:
        raise ValueError("Heston calibration requires at least one prepared OTM node")
    return nodes


def _curve_functions(per_expiry: Sequence[dict]):
    maturities = np.array([p["T"] for p in per_expiry], dtype=float)
    rates = np.array([p["r"] for p in per_expiry], dtype=float)
    carries = np.array([p["q"] for p in per_expiry], dtype=float)

    def rate_at(t: float) -> float:
        return float(np.interp(t, maturities, rates))

    def carry_at(t: float) -> float:
        return float(np.interp(t, maturities, carries))

    return rate_at, carry_at


def _options(nodes: Sequence[dict], weights: Sequence[float] | None = None) -> list[MarketOption]:
    if weights is None:
        weights_array = np.ones(len(nodes), dtype=float)
    else:
        weights_array = np.asarray(weights, dtype=float)
        if weights_array.shape != (len(nodes),) or not np.all(np.isfinite(weights_array)):
            raise ValueError("calibration weights must be a finite vector matching nodes")
        if np.any(weights_array < 0.0):
            raise ValueError("calibration weights must be non-negative")
    return [
        MarketOption(
            K=float(node["K"]),
            T=float(node["T"]),
            price=float(node["target_price"]),
            weight=float(weight),
        )
        for node, weight in zip(nodes, weights_array)
    ]


def _run_calibration(
    s0: float,
    nodes: Sequence[dict],
    rate_at,
    carry_at,
    initial: HestonParams,
    *,
    max_nfev: int,
    weights: Sequence[float] | None = None,
):
    return calibrate_heston(
        s0=s0,
        options=_options(nodes, weights),
        r=rate_at,
        carry=carry_at,
        initial=initial,
        bounds=HESTON_BOUNDS,
        target="iv",
        method="lewis",
        regularize_feller=REGULARIZE_FELLER,
        enforce_feller=False,
        max_nfev=max_nfev,
        **SOLVER_TOLERANCES,
    )


def _model_ivs(s0: float, nodes: Sequence[dict], params: HestonParams) -> np.ndarray:
    """Lewis model IVs in node order, grouped by maturity like the calibrator."""
    maturities = np.array([node["T"] for node in nodes], dtype=float)
    model = np.empty(len(nodes), dtype=float)
    for maturity in np.unique(maturities):
        indices = np.flatnonzero(maturities == maturity)
        strikes = np.array([nodes[index]["K"] for index in indices], dtype=float)
        rate = float(nodes[int(indices[0])]["r"])
        carry = float(nodes[int(indices[0])]["q"])
        # A maturity stratum must use one parity pillar.  Drift would make the
        # strike-vectorized objective inconsistent with the saved surface.
        if any(
            not math.isclose(
                float(nodes[int(index)]["r"]), rate, rel_tol=0.0, abs_tol=1e-14
            )
            or not math.isclose(
                float(nodes[int(index)]["q"]), carry, rel_tol=0.0, abs_tol=1e-14
            )
            for index in indices
        ):
            raise ValueError(f"rate/carry drift within maturity {maturity}")
        prices = heston_call_prices_vectorized(
            s0, strikes, float(maturity), params, rate, carry
        )
        values = [
            implied_vol_call(
                s0, float(strike), float(maturity), float(price), rate, carry
            )
            for strike, price in zip(strikes, prices)
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite Heston implied vol at maturity {maturity}")
        model[indices] = values
    return model


def _bound_hits(params: HestonParams) -> dict[str, dict[str, bool]]:
    values = _params_vector(params)
    lower = np.asarray(HESTON_BOUNDS[0], dtype=float)
    upper = np.asarray(HESTON_BOUNDS[1], dtype=float)
    tolerance = np.maximum(1e-10, 1e-7 * (upper - lower))
    return {
        name: {
            "lower": bool(abs(value - lo) <= tol),
            "upper": bool(abs(hi - value) <= tol),
        }
        for name, value, lo, hi, tol in zip(
            PARAMETER_NAMES, values, lower, upper, tolerance
        )
    }


def _feller_ratio(params: HestonParams) -> float:
    return float(2.0 * params.kappa * params.theta / (params.sigma**2))


def _bootstrap_evidence(
    s0: float,
    nodes: Sequence[dict],
    rate_at,
    carry_at,
    initial: HestonParams,
    *,
    reps: int,
    seed: int,
    max_nfev: int,
) -> dict:
    maturities = np.array([node["T"] for node in nodes], dtype=float)
    target = np.array([node["target_iv"] for node in nodes], dtype=float)
    children = np.random.SeedSequence(seed).spawn(reps)
    replicates: list[dict] = []

    for index, child in enumerate(children):
        replicate_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        weights = hd.stratified_exponential_weights(maturities, seed=replicate_seed)
        weight_sums = {
            f"{maturity:.12g}": float(np.sum(weights[maturities == maturity]))
            for maturity in np.unique(maturities)
        }
        try:
            result = _run_calibration(
                s0,
                nodes,
                rate_at,
                carry_at,
                initial,
                max_nfev=max_nfev,
                weights=weights,
            )
            if result.success is not True:
                replicates.append(
                    {
                        "index": index,
                        "seed": replicate_seed,
                        "success": False,
                        "failure_type": "optimizer_failure",
                        "message": str(result.message),
                        "nfev": int(result.nfev),
                        "weight_sum_by_maturity": weight_sums,
                    }
                )
                continue
            params = result.params
            model = _model_ivs(s0, nodes, params)
            errors = model - target
            ratio = _feller_ratio(params)
            replicates.append(
                {
                    "index": index,
                    "seed": replicate_seed,
                    "success": True,
                    "message": str(result.message),
                    "optimizer": str(result.optimizer),
                    "nfev": int(result.nfev),
                    "params": _params_dict(params),
                    "cost": float(result.cost),
                    "data_cost": float(result.data_cost),
                    "feller_penalty_cost": float(result.feller_penalty_cost),
                    "feller_ratio": ratio,
                    "feller_satisfied": bool(ratio >= 1.0),
                    "feller_margin": float(result.feller_margin),
                    "full_sample_rmse_iv": float(np.sqrt(np.mean(errors**2))),
                    "bootstrap_weighted_rmse_iv": float(
                        np.sqrt(np.average(errors**2, weights=weights))
                    ),
                    "bound_hits": _bound_hits(params),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "weight_sum_by_maturity": weight_sums,
                }
            )
        except Exception as exc:  # each failure is evidence and remains in the artifact
            replicates.append(
                {
                    "index": index,
                    "seed": replicate_seed,
                    "success": False,
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                    "weight_sum_by_maturity": weight_sums,
                }
            )

    summary = hd.summarize_bootstrap_replicates(replicates, requested=reps)
    return {
        "method": "maturity_stratified_exponential_multiplier_weights",
        "interpretation": (
            "Conditional prepared-target node-influence evidence. Empirical quantiles "
            "are not statistical confidence intervals and do not represent a quote-time "
            "sampling distribution."
        ),
        "is_statistical_confidence_interval": False,
        "target": "same_SABR_prepared_OTM_IV_nodes_as_main_fit",
        "normalization": (
            "independent exponential weights, normalized within each maturity so each "
            "stratum retains its original total weight"
        ),
        "seed": int(seed),
        "seed_policy": "numpy_SeedSequence_spawn_then_PCG64",
        "configured_initial": _params_dict(initial),
        "bounds": [list(HESTON_BOUNDS[0]), list(HESTON_BOUNDS[1])],
        "regularize_feller": REGULARIZE_FELLER,
        "enforce_feller": False,
        "max_nfev": int(max_nfev),
        **summary,
    }


def build_calibration_report(
    raw_surface: dict,
    *,
    tag: str,
    surface_provenance: dict | None = None,
    iv_smoothing: str = "sabr",
    sabr_beta: float = 1.0,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    max_nfev: int = 200,
) -> tuple[dict, list[tuple[str, list[float], list[float]]]]:
    """Build the Stage-04 JSON payload and plot rows without writing files."""
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    if bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be non-negative")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")

    surface = mc.prepare_model_surface(
        raw_surface, iv_smoothing=iv_smoothing, sabr_beta=sabr_beta
    )
    s0 = float(surface["s0"])
    per_expiry = surface["per_expiry"]
    smoothing = surface.get("target_smoothing", {"method": "none"})
    nodes = _calibration_nodes(s0, per_expiry)
    rate_at, carry_at = _curve_functions(per_expiry)
    initial = _initial_guess(per_expiry)

    result = _run_calibration(
        s0,
        nodes,
        rate_at,
        carry_at,
        initial,
        max_nfev=max_nfev,
    )
    if result.success is not True:
        raise RuntimeError(
            "main Heston calibration failed: "
            f"{result.message} (nfev={result.nfev}, cost={result.cost:.6g})"
        )
    params = result.params
    model = _model_ivs(s0, nodes, params)
    target = np.array([node["target_iv"] for node in nodes], dtype=float)
    raw_mask = np.array([node["raw_iv"] is not None for node in nodes], dtype=bool)
    raw_values = np.array(
        [float(node["raw_iv"]) if node["raw_iv"] is not None else np.nan for node in nodes]
    )
    prepared_errors = model - target
    raw_errors = model[raw_mask] - raw_values[raw_mask]

    node_rows = []
    for node, model_iv, error in zip(nodes, model, prepared_errors):
        raw_iv = node["raw_iv"]
        node_rows.append(
            {
                "expiry_date": node["expiry_date"],
                "T": node["T"],
                "K": node["K"],
                "r": node["r"],
                "q": node["q"],
                "target_iv": node["target_iv"],
                "raw_iv": raw_iv,
                "model_iv": float(model_iv),
                "error_iv": float(error),
                "raw_error_iv": None if raw_iv is None else float(model_iv - raw_iv),
            }
        )

    per_expiry_report: list[dict] = []
    smile_rows: list[tuple[str, list[float], list[float]]] = []
    for expiry in per_expiry:
        maturity = float(expiry["T"])
        indices = np.array(
            [index for index, node in enumerate(nodes) if node["T"] == maturity], dtype=int
        )
        expiry_errors = prepared_errors[indices]
        row = {
            "T": maturity,
            "rmse_iv": float(np.sqrt(np.mean(expiry_errors**2))),
        }
        expiry_raw = [nodes[int(index)]["raw_iv"] for index in indices]
        if all(value is not None for value in expiry_raw):
            raw_array = np.array(expiry_raw, dtype=float)
            row["raw_rmse_iv"] = float(
                np.sqrt(np.mean((model[indices] - raw_array) ** 2))
            )
        per_expiry_report.append(row)
        strikes = [float(nodes[int(index)]["K"]) for index in indices]
        smile_rows.append((f"target T={maturity:.2f}", strikes, target[indices].tolist()))
        smile_rows.append((f"Heston T={maturity:.2f}", strikes, model[indices].tolist()))

    fitted_vector = _params_vector(params)
    jacobian = hd.finite_difference_model_jacobian(
        lambda values: _model_ivs(s0, nodes, _params_from_vector(values)),
        fitted_vector,
        HESTON_BOUNDS[0],
        HESTON_BOUNDS[1],
    )
    bootstrap = _bootstrap_evidence(
        s0,
        nodes,
        rate_at,
        carry_at,
        initial,
        reps=bootstrap_reps,
        seed=bootstrap_seed,
        max_nfev=max_nfev,
    )
    feller = _feller_ratio(params)
    calibration_spec = {
        "schema_version": 1,
        "tag": tag,
        "s0": s0,
        "surface_provenance": surface_provenance or {"kind": "in_memory_payload"},
        "parameter_order": list(PARAMETER_NAMES),
        "calibration_target": (
            "SABR_calendar_prepared_OTM_implied_vols"
            if smoothing.get("method") == "sabr_calendar_projected"
            else "raw_OTM_implied_vols"
        ),
        "target_smoothing_method": smoothing.get("method", "none"),
        "node_count": len(nodes),
        "maturity_count": len(per_expiry),
        "initial": _params_dict(initial),
        "start_policy": "single_deterministic_shortest_expiry_ATM_variance_anchor",
        "bounds": [list(HESTON_BOUNDS[0]), list(HESTON_BOUNDS[1])],
        "target": "iv",
        "method": "lewis",
        "weight_mode": "equal_per_prepared_OTM_node",
        "rate_and_carry": "piecewise_linear_interpolation_of_parity_pillars",
        "regularize_feller": REGULARIZE_FELLER,
        "enforce_feller": False,
        "solver": {"max_nfev": int(max_nfev), **SOLVER_TOLERANCES},
        "jacobian": {
            "relative_step": jacobian["relative_step"],
            "policy_rcond": hd.POLICY_RCOND,
            "fixed_economic_scales": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, hd.FIXED_ECONOMIC_SCALES)
            },
            "excludes_feller_penalty": True,
        },
        "bootstrap": {
            "reps": int(bootstrap_reps),
            "seed": int(bootstrap_seed),
            "method": bootstrap["method"],
            "same_start_and_constraints_as_main_fit": True,
        },
    }

    output = {
        # Original downstream fields are intentionally retained.
        "params": _params_dict(params),
        "feller": feller,
        "cost": float(result.cost),
        "success": bool(result.success),
        "overall_rmse_iv": float(np.sqrt(np.mean(prepared_errors**2))),
        "per_expiry": per_expiry_report,
        "target_smoothing": smoothing,
        # New provenance and identification evidence.
        "calibration_spec": calibration_spec,
        "optimizer": {
            "name": str(result.optimizer),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "data_cost": float(result.data_cost),
            "feller_penalty_cost": float(result.feller_penalty_cost),
            "feller_margin": float(result.feller_margin),
        },
        "bound_hits": _bound_hits(params),
        "node_rows": node_rows,
        "jacobian": jacobian,
        "bootstrap": bootstrap,
    }
    if raw_errors.size:
        output["raw_overall_rmse_iv"] = float(np.sqrt(np.mean(raw_errors**2)))
    return output, smile_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument(
        "--iv-smoothing",
        choices=["sabr", "none"],
        default="sabr",
        help="model target preparation: SABR + calendar projection (default) or raw",
    )
    parser.add_argument(
        "--sabr-beta",
        type=float,
        default=1.0,
        help="SABR beta used by --iv-smoothing sabr; beta=1 is lognormal",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--max-nfev", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    if args.max_nfev <= 0:
        parser.error("--max-nfev must be positive")

    surface_path = HERE / f"data/mo_iv_surface_{args.tag}.json"
    surface_bytes = surface_path.read_bytes()
    raw_surface = json.loads(surface_bytes)
    output, smile_rows = build_calibration_report(
        raw_surface,
        tag=args.tag,
        surface_provenance={
            "kind": "file_sha256",
            "filename": surface_path.name,
            "sha256": hashlib.sha256(surface_bytes).hexdigest(),
        },
        iv_smoothing=args.iv_smoothing,
        sabr_beta=args.sabr_beta,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        max_nfev=args.max_nfev,
    )
    smoothing = output["target_smoothing"]
    if smoothing.get("method") == "sabr_calendar_projected":
        print(
            "SABR-smoothed Heston target: "
            f"raw-grid RMSE={smoothing['raw_grid_rmse_iv'] * 100:.3f} vol-pts, "
            f"calendar-adjusted nodes={smoothing['calendar_adjusted_nodes']}"
        )
    params = output["params"]
    print(
        f"v0={params['v0']:.4f} kappa={params['kappa']:.3f} "
        f"theta={params['theta']:.4f} sigma={params['sigma']:.3f} "
        f"rho={params['rho']:.3f}"
    )
    print(
        f"Feller 2*kappa*theta/sigma^2 = {output['feller']:.2f} "
        f"({'satisfied' if output['feller'] >= 1 else 'VIOLATED (v can hit 0)'}) "
        f"cost={output['cost']:.3e} success={output['success']}"
    )
    for row in output["per_expiry"]:
        print(f"  T={row['T']:.3f}  Heston RMSE={row['rmse_iv'] * 100:.3f} vol-pts")

    output_path = HERE / f"data/mo_calib_heston_{args.tag}.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False))
    mc.plot_smiles(
        smile_rows,
        HERE / f"data/plots/04_heston_fit_{args.tag}.png",
        title="Heston fit vs calibration target — MO (000852.SH)",
    )
    print(f"overall Heston RMSE = {output['overall_rmse_iv'] * 100:.3f} vol-pts")
    if "raw_overall_rmse_iv" in output:
        print(
            "overall Heston vs raw quotes = "
            f"{output['raw_overall_rmse_iv'] * 100:.3f} vol-pts"
        )
    fixed = output["jacobian"]["svd"]["fixed_economic"]
    condition = fixed["condition_number"]
    print(
        "fixed-economic Jacobian: "
        f"rank={fixed['policy_effective_rank']}/5 "
        f"condition={'undefined' if condition is None else f'{condition:.3g}'}"
    )
    bootstrap = output["bootstrap"]
    print(
        f"bootstrap: {bootstrap['successful_replicates']}/"
        f"{bootstrap['requested_replicates']} successful ({bootstrap['status']})"
    )


if __name__ == "__main__":
    main()
