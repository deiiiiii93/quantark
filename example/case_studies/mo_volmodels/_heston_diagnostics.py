"""Reusable Heston calibration-identification diagnostics for the MO examples.

The finite-difference routine is deliberately independent of QuantArk's optimizer: it
accepts any callable that maps a five-parameter vector to the model values at the
calibration nodes.  This keeps the reported Jacobian limited to market-data residuals;
the soft Feller penalty is not allowed to masquerade as quote information.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np


PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
FIT_RELATIVE_SCALE_FLOORS = np.array([1e-4, 0.1, 1e-4, 0.01, 0.1], dtype=float)
FIXED_ECONOMIC_SCALES = np.array([0.01, 1.0, 0.01, 0.1, 0.1], dtype=float)
POLICY_RCOND = 1e-3


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional vector")
    return array


def heston_scale_policies(
    parameters: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, np.ndarray]:
    """Return explicit unit policies for interpretable and cross-date SVDs."""
    x = _finite_vector(parameters, name="parameters")
    lo = _finite_vector(lower, name="lower")
    hi = _finite_vector(upper, name="upper")
    if x.size != len(PARAMETER_NAMES) or lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError("Heston scale policies require five matching parameter vectors")
    span = hi - lo
    if np.any(span <= 0.0):
        raise ValueError("diagnostic bounds must have strictly positive width")
    return {
        "raw": np.ones_like(x),
        "fit_relative": np.maximum(np.abs(x), FIT_RELATIVE_SCALE_FLOORS),
        # This vector is fixed across dates.  It is the like-for-like policy for a
        # condition-number history; fit-relative scales vary with each calibration.
        "fixed_economic": FIXED_ECONOMIC_SCALES.copy(),
        "bound_span": span,
    }


def _canonical_right_vectors(vectors: np.ndarray) -> np.ndarray:
    """Remove the arbitrary SVD sign so JSON and tests remain deterministic."""
    out = np.asarray(vectors, dtype=float).copy()
    for row in out:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return out


def _svd_summary(
    matrix: np.ndarray,
    parameter_names: Sequence[str],
    *,
    policy_rcond: float,
) -> dict:
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular_values[0])
    machine_tolerance = float(np.finfo(float).eps * max(matrix.shape) * largest)
    numerical_rank = int(np.sum(singular_values > machine_tolerance))
    policy_tolerance = float(policy_rcond * largest)
    effective_rank = int(np.sum(singular_values > policy_tolerance))
    smallest = float(singular_values[-1])
    condition = None if smallest <= machine_tolerance else float(largest / smallest)
    relative = (
        singular_values / largest if largest > 0.0 else np.zeros_like(singular_values)
    )
    canonical_vh = _canonical_right_vectors(vh)
    directions = []
    for index, (value, relative_value, vector) in enumerate(
        zip(singular_values, relative, canonical_vh)
    ):
        directions.append(
            {
                "index": int(index),
                "singular_value": float(value),
                "relative_singular_value": float(relative_value),
                "components": {
                    name: float(component)
                    for name, component in zip(parameter_names, vector)
                },
            }
        )
    return {
        "singular_values": [float(value) for value in singular_values],
        "relative_singular_values": [float(value) for value in relative],
        "condition_number": condition,
        "numerical_rank": numerical_rank,
        "machine_rank_tolerance": machine_tolerance,
        "policy_effective_rank": effective_rank,
        "policy_relative_tolerance": float(policy_rcond),
        "policy_absolute_tolerance": policy_tolerance,
        "right_singular_vectors": directions,
    }


def finite_difference_model_jacobian(
    model_values: Callable[[np.ndarray], Sequence[float]],
    parameters: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    parameter_names: Sequence[str] = PARAMETER_NAMES,
    relative_step: float = 1e-4,
    policy_rcond: float = POLICY_RCOND,
    scale_policies: Mapping[str, Sequence[float]] | None = None,
) -> dict:
    """Return a bound-aware model-value Jacobian and four scaled SVD views.

    Central differences are used in the interior.  At a box boundary, the routine uses
    the second-order one-sided formula, not a first-order shortcut.  All evaluations must
    return the same finite vector; an invalid perturbation fails the diagnostic instead of
    silently filling a column with zeros or NaNs.
    """
    if not math.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    if not math.isfinite(policy_rcond) or not 0.0 < policy_rcond < 1.0:
        raise ValueError("policy_rcond must be finite and lie in (0, 1)")

    x = _finite_vector(parameters, name="parameters")
    lo = _finite_vector(lower, name="lower")
    hi = _finite_vector(upper, name="upper")
    names = tuple(parameter_names)
    if len(names) != x.size or lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError("parameters, names and bounds must have matching dimensions")
    span = hi - lo
    if np.any(span <= 0.0):
        raise ValueError("diagnostic bounds must have strictly positive width")
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("parameters must lie within the diagnostic bounds")

    def evaluate(values: np.ndarray, label: str) -> np.ndarray:
        result = _finite_vector(model_values(values), name=f"model values at {label}")
        if result.shape != base.shape:
            raise ValueError(
                f"model values at {label} have shape {result.shape}, expected {base.shape}"
            )
        return result

    base = _finite_vector(model_values(x.copy()), name="model values at base parameters")
    steps = np.maximum(np.abs(x) * relative_step, span * 1e-6)
    # Retain room for the two evaluations required by a second-order one-sided stencil.
    steps = np.minimum(steps, 0.25 * span)
    jacobian = np.empty((base.size, x.size), dtype=float)
    schemes: dict[str, str] = {}

    bound_tolerance = np.maximum(1e-10, 1e-7 * span)
    active_bounds: dict[str, dict[str, bool]] = {}
    for index, (name, step) in enumerate(zip(names, steps)):
        active_bounds[name] = {
            "lower": bool(abs(x[index] - lo[index]) <= bound_tolerance[index]),
            "upper": bool(abs(hi[index] - x[index]) <= bound_tolerance[index]),
        }
        can_central = x[index] - step >= lo[index] and x[index] + step <= hi[index]
        can_forward = x[index] + 2.0 * step <= hi[index]
        can_backward = x[index] - 2.0 * step >= lo[index]
        if can_central:
            down_x = x.copy()
            up_x = x.copy()
            down_x[index] -= step
            up_x[index] += step
            down = evaluate(down_x, f"{name}-step")
            up = evaluate(up_x, f"{name}+step")
            jacobian[:, index] = (up - down) / (2.0 * step)
            schemes[name] = "central_second_order"
        elif can_forward:
            one_x = x.copy()
            two_x = x.copy()
            one_x[index] += step
            two_x[index] += 2.0 * step
            one = evaluate(one_x, f"{name}+step")
            two = evaluate(two_x, f"{name}+2step")
            jacobian[:, index] = (-3.0 * base + 4.0 * one - two) / (2.0 * step)
            schemes[name] = "forward_second_order"
        elif can_backward:
            one_x = x.copy()
            two_x = x.copy()
            one_x[index] -= step
            two_x[index] -= 2.0 * step
            one = evaluate(one_x, f"{name}-step")
            two = evaluate(two_x, f"{name}-2step")
            jacobian[:, index] = (3.0 * base - 4.0 * one + two) / (2.0 * step)
            schemes[name] = "backward_second_order"
        else:
            raise ValueError(f"cannot construct a second-order stencil for {name}")

    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference Jacobian contains non-finite values")

    policies = (
        heston_scale_policies(x, lo, hi)
        if scale_policies is None
        else {
            str(name): _finite_vector(values, name=f"scale policy {name}")
            for name, values in scale_policies.items()
        }
    )
    if not policies:
        raise ValueError("at least one scale policy is required")
    scales_out: dict[str, dict] = {}
    svd_out: dict[str, dict] = {}
    for policy_name, scales in policies.items():
        if scales.shape != x.shape or np.any(scales <= 0.0):
            raise ValueError(f"scale policy {policy_name} must be positive and match parameters")
        scales_out[policy_name] = {
            name: float(scale) for name, scale in zip(names, scales)
        }
        svd_out[policy_name] = _svd_summary(
            jacobian * scales[None, :], names, policy_rcond=policy_rcond
        )

    return {
        "method": "bound_aware_second_order_finite_difference_of_model_implied_vols",
        "excludes_feller_penalty": True,
        "shape": [int(value) for value in jacobian.shape],
        "parameter_order": list(names),
        "base_parameters": {name: float(value) for name, value in zip(names, x)},
        "relative_step": float(relative_step),
        "steps": {name: float(value) for name, value in zip(names, steps)},
        "difference_schemes": schemes,
        "active_bound_tolerance": {
            name: float(value) for name, value in zip(names, bound_tolerance)
        },
        "active_bounds": active_bounds,
        "matrix": jacobian.tolist(),
        "scales": scales_out,
        "svd": svd_out,
    }


def stratified_exponential_weights(
    strata: Sequence[float],
    *,
    seed: int,
) -> np.ndarray:
    """Draw deterministic exponential multiplier weights, normalized by stratum."""
    labels = _finite_vector(strata, name="strata")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    weights = np.empty(labels.size, dtype=float)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        draws = rng.exponential(scale=1.0, size=indices.size)
        total = float(np.sum(draws))
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("exponential multiplier draw has invalid stratum total")
        weights[indices] = draws * (indices.size / total)
    return weights


def _distribution(values: Sequence[float]) -> dict | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return {
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size >= 2 else None,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_bootstrap_replicates(
    replicates: Sequence[Mapping],
    *,
    requested: int,
    parameter_names: Sequence[str] = PARAMETER_NAMES,
) -> dict:
    """Summarize successful replicates without hiding any failed solve."""
    if not isinstance(requested, int) or requested <= 0:
        raise ValueError("requested bootstrap replicates must be positive")
    rows = [dict(row) for row in replicates]
    if len(rows) != requested:
        raise ValueError("replicate list length must equal requested count")
    successful = [row for row in rows if row.get("success") is True]
    failures = [row for row in rows if row.get("success") is not True]
    names = tuple(parameter_names)

    parameter_summary = None
    covariance = None
    correlation = None
    bound_hit_rates = None
    feller_pass_fraction = None
    full_rmse = None
    weighted_rmse = None
    if successful:
        samples = np.asarray(
            [[float(row["params"][name]) for name in names] for row in successful],
            dtype=float,
        )
        parameter_summary = {
            name: _distribution(samples[:, index])
            for index, name in enumerate(names)
        }
        if samples.shape[0] >= 2:
            covariance_array = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
            covariance = covariance_array.tolist()
            standard_deviation = np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
            correlation_array: list[list[float | None]] = []
            scale = np.maximum(1.0, np.max(np.abs(samples), axis=0))
            defined = standard_deviation > 1e-10 * scale
            for i in range(len(names)):
                correlation_row: list[float | None] = []
                for j in range(len(names)):
                    if not (defined[i] and defined[j]):
                        correlation_row.append(None)
                    else:
                        value = covariance_array[i, j] / (
                            standard_deviation[i] * standard_deviation[j]
                        )
                        correlation_row.append(float(np.clip(value, -1.0, 1.0)))
                correlation_array.append(correlation_row)
            correlation = correlation_array
        bound_hit_rates = {}
        for name in names:
            lower_hits = [bool(row["bound_hits"][name]["lower"]) for row in successful]
            upper_hits = [bool(row["bound_hits"][name]["upper"]) for row in successful]
            bound_hit_rates[name] = {
                "lower": float(np.mean(lower_hits)),
                "upper": float(np.mean(upper_hits)),
                "either": float(np.mean(np.logical_or(lower_hits, upper_hits))),
            }
        feller_pass_fraction = float(
            np.mean([bool(row["feller_satisfied"]) for row in successful])
        )
        full_rmse = _distribution(
            [float(row["full_sample_rmse_iv"]) for row in successful]
        )
        weighted_rmse = _distribution(
            [float(row["bootstrap_weighted_rmse_iv"]) for row in successful]
        )

    return {
        "status": (
            "failed" if not successful else ("complete" if not failures else "partial")
        ),
        "requested_replicates": int(requested),
        "successful_replicates": len(successful),
        "failed_replicates": len(failures),
        "success_fraction": float(len(successful) / requested),
        "parameter_order": list(names),
        "parameter_quantiles": parameter_summary,
        "sample_covariance": covariance,
        "sample_correlation": correlation,
        "bound_hit_rates": bound_hit_rates,
        "feller_pass_fraction": feller_pass_fraction,
        "rmse_distribution": {
            "full_sample_iv": full_rmse,
            "bootstrap_weighted_iv": weighted_rmse,
        },
        "replicates": rows,
        "failures": failures,
    }
