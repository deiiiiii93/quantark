"""Shared loader + measurement core for the MC reference-convergence demos.

Loads the stage-16 harness by path (semantic fidelity: cases, products,
environments, engines, and the paired-RQMC reference estimator are the
harness's own), runs one treatment row, and reports the decision-matrix
numbers: batch SD in contracts, seconds/batch, and peak RSS.

Run with the worktree shadowed:
    PYTHONPATH=<repo-root> <venv>/bin/python docs/mc-reference-convergence/<demo>.py
"""

from __future__ import annotations

import importlib.util
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "cert16", REPO / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
)
assert _spec is not None and _spec.loader is not None
cert = importlib.util.module_from_spec(_spec)
sys.modules["cert16"] = cert
_spec.loader.exec_module(cert)

# Recovered schema-11 anchor; the scale relation was re-verified bitwise on
# 2026-08-10 against the banked heston/sigma_collapse gap.
HEDGE_INCEPTION_SPOT = 4532.52


def peak_rss_gib() -> float:
    """Peak RSS of this process and its finished children, in GiB.

    macOS reports ``ru_maxrss`` in bytes (Linux uses kilobytes). The paired
    reference runs its batches in worker processes, so children dominate.
    """
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return max(own, children) / divisor


def case_scale(case):
    """The certification's economic scale for one case."""
    return cert.EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=HEDGE_INCEPTION_SPOT,
        study_notional=cert.STUDY_NOTIONAL,
        hedge_multiplier=cert.HEDGE_MULTIPLIER,
    )


def case_fixture(cell: str, variant: str):
    """Harness-faithful (case, product, env, leverage) for one cell."""
    case = next(c for c in cert.certification_cases(quick=False) if c.name == cell)
    product = cert.make_snowball(case, dense_ki=True)
    env = cert.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = (
        cert.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    )
    return case, product, env, leverage


def batch_deltas(
    cell: str,
    variant: str,
    *,
    batches: int,
    seed: int,
    bridge_dimensions: int,
    workers: int = 2,
) -> tuple[np.ndarray, float]:
    """Per-batch deltas from the harness estimator, plus elapsed seconds."""
    case, product, env, leverage = case_fixture(cell, variant)
    substeps = cert.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][cell]["target"]
    paths = (
        cert.PRODUCTION_SLV_PATHS_PER_BATCH
        if variant == "heston_slv"
        else cert.PRODUCTION_HESTON_PATHS_PER_BATCH
    )
    kwargs = dict(
        paths_per_batch=paths,
        batches=batches,
        seed=seed,
        substeps=substeps,
        bump=cert.SPOT_BUMP,
        rqmc_batch_workers=workers,
    )
    if variant == "heston_slv":
        kwargs["slv_spot_bridge_dimensions"] = bridge_dimensions
    else:
        kwargs["heston_spot_bridge_dimensions"] = bridge_dimensions
        kwargs["heston_spot_bridge_strata"] = 1 if bridge_dimensions == 1 else 4

    started = time.perf_counter()
    result = cert.paired_mc_reference(variant, case, product, env, leverage, **kwargs)
    elapsed = time.perf_counter() - started
    return np.asarray(result.batch_delta, dtype=float), elapsed


def measure_row(
    cell: str,
    variant: str,
    label: str,
    *,
    batches: int,
    seed: int,
    bridge_dimensions: int,
    workers: int = 2,
) -> dict:
    """One decision-matrix row: precision, cost, and memory of a treatment."""
    deltas, elapsed = batch_deltas(
        cell,
        variant,
        batches=batches,
        seed=seed,
        bridge_dimensions=bridge_dimensions,
        workers=workers,
    )
    case, _, _, _ = case_fixture(cell, variant)
    scale = case_scale(case)
    # delta_contracts is linear through the origin, so one factor converts any
    # per-unit delta quantity (SD, SE, or a difference of means) to contracts.
    scale_factor = abs(scale.delta_contracts(1.0))
    sd_contracts = scale_factor * float(np.std(deltas, ddof=1))
    return {
        "label": label,
        "cell": cell,
        "variant": variant,
        "batches": int(deltas.size),
        "bridge_dimensions": int(bridge_dimensions),
        "seed": int(seed),
        "delta_mean": float(deltas.mean()),
        "delta_se": float(deltas.std(ddof=1) / np.sqrt(deltas.size)),
        "delta_se_contracts": scale_factor
        * float(deltas.std(ddof=1) / np.sqrt(deltas.size)),
        "batch_sd_contracts": sd_contracts,
        "scale_factor": scale_factor,
        "seconds_per_batch": round(elapsed / max(deltas.size, 1), 3),
        "elapsed_seconds": round(elapsed, 1),
        "peak_rss_gib": round(peak_rss_gib(), 2),
    }
