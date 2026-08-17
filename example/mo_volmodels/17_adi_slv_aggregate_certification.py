"""Aggregate-only Heston-SLV ADI Greek certification amendment.

Schema 11 already certifies every deterministic anchor and all fourteen
individual Heston/Heston-SLV regime cells.  This module never recomputes those
cells.  It links the exact schema-11 artifact and adds only reference-estimator
evidence for the unresolved seven-cell mean signed Delta-bias gate.

Two details are deliberate:

* target/fine uncertainty is certified through the two signed endpoints
  ``D + S`` and ``D - S`` (``D = PDE - fine`` and ``S = target - fine``).
  Simultaneous confidence intervals for those endpoints cover every residual
  reference bias in ``[-abs(E[S]), +abs(E[S])]`` while preserving the observed
  covariance between the fine estimate and its paired substep diagnostic;
* every added control variate is an unbiased difference of expectations on
  mutually independent, fixed-size scramble families.  Development pilots and
  production seeds are disjoint, and production has no optional stopping.

The development-only control pilot is intentionally non-admissive::

    .venv/bin/python example/mo_volmodels/17_adi_slv_aggregate_certification.py \
      --development-pilot --output-dir output/adi_slv_aggregate_pilot

The allocation is frozen from the complete development family and its
projection hashes before either held-out seed is opened.  Run the production
amendment with the exact schema-11 parent::

    .venv/bin/python example/mo_volmodels/17_adi_slv_aggregate_certification.py \
      --parent-evidence output/adi_greek_certification_schema13/adi_greek_certification.json \
      --parent-decision output/adi_greek_certification_schema13/adi_greek_certification_decision.json \
      --resume --output-dir output/adi_greek_certification
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import scipy
from scipy.stats import chi2, t as student_t

from quantark.validation import (
    CellPrecision,
    EquivalenceStatus,
    neyman_allocation,
    precision_stop,
    projected_aggregate_halfwidth,
)


SCHEMA_VERSION = 12
CERTIFICATION_MODE = "aggregate_only_amendment"
STUDY = "adi_2d_snowball_greek_certification"

STAGE16_PATH = Path(__file__).resolve().parent / "16_adi_greek_certification.py"
_STAGE16 = None

# Immutable stage-16 parent: the 2026-08-17 strided re-publish
# (output/p18_strided, commit 258fd7e), a schema-13 full recertification whose
# aggregate uses the declared strided pooling.  It replaces the lost schema-13
# incremental amendment b5463093/3d4cb66b as this amendment's pinned parent.
# Both serialized bytes and embedded canonical identities are checked.  The
# parent remains the authority for anchors, individual cells, Heston
# admission, and the production PDE implementation.  A full-recertification
# parent has no cell_provenance / auxiliary_controls / aggregate_cohorts
# projections; its per-cell reuse identities are pinned instead.
PARENT_SCHEMA_VERSION = 13
PARENT_SOURCE_COMMIT = "258fd7ec39416335a00e7fad70822c15c8c1294a"
PARENT_EVIDENCE_FILE_SHA256 = (
    "9041c3a299bb518d683768ff4c852c7556078786e2b6d0115ed6027a36867ba7"
)
PARENT_DECISION_FILE_SHA256 = (
    "63e959ab274e7816a1f99760b72a2020c87fd63c634fcf888f95f73ec80e89ab"
)
PARENT_EVIDENCE_SHA256 = (
    "5c2bd5796e42b6c241f534325738227c38077f978dc4de2b2f345b571e907d07"
)
PARENT_DECISION_SHA256 = (
    "55e618eaccf1f4eb97c2620f5b5ee652ac590cc4e8c599a2acf84ed01edc4ef3"
)
PARENT_IMPLEMENTATION_SHA256 = (
    "ce027ee25157c30ffd54aedd86032d21ed20a288087f3c82a02cd9b2691c1fba"
)
PARENT_RUN_CONFIGURATION_SHA256 = (
    "ceb69449baa967069b04fa777c067825d1191512aa5c3870fc7c30722e5c235c"
)
PARENT_PRODUCTION_PDE_SHA256 = (
    "f87b5086a3f1d1538a2e346016a15e193cbc6e9de06b949968295a503e75c2ff"
)
PARENT_ANCHORS_SHA256 = (
    "4e7e9cdc72f5410f7c42713f802dcf2291d870e0bc65840d5a438a0da814e74a"
)
PARENT_CELLS_SHA256 = (
    "06a48c772b20bbcdedb677700958cc71a200ec0d6316cf13c50cbc8b1b52ff30"
)
PARENT_CELL_IDENTITIES_SHA256 = (
    "fe10eecd7509678df2d5112f9dfdf1b33c4bee7f0769bac022727a79cbabbb43"
)
PARENT_DECISIONS_SHA256 = (
    "bc8aa961e3aafb86cf9996bcfe0618e60099678d6ad39ff9a3b33ac08cbd155b"
)
PARENT_BASE_HESTON_DECISION_SHA256 = (
    "d7a1209ffa82ef0a788486e35f00216332aeffd290e370b95a8c79a2c1cea04b"
)
PARENT_SAMPLING_BY_VARIANT_SHA256 = (
    "5b603d6c1589e384e87462996a056fdf9e9df40a59b3f58a7ec26309420bdac8"
)

DEVELOPMENT_SEED = 20260806
DEVELOPMENT_PATHS_PER_BATCH = 8192
DEVELOPMENT_BATCHES = 16
DEVELOPMENT_WORKERS = 4
DEVELOPMENT_PRIMARY_BATCH_GRID = (512, 1024, 2048, 4096)
DEVELOPMENT_MIDDLE_BATCH_GRID = (32, 64, 128, 256, 512, 1024)
DEVELOPMENT_HESTON_WEIGHT_GRID = tuple(
    round(index / 10.0, 1) for index in range(11)
)
DEVELOPMENT_NEW_FAMILY_GUARD_SIGMAS = 1.0
DEVELOPMENT_VARIANCE_UPPER_CONFIDENCE = 0.95
CONTROL_CASES = (
    "ordinary_full",
    "ordinary_decayed",
    "near_ko",
    "low_feller",
    "sigma_collapse",
    "near_expiry",
)
PRIMARY_REFRESH_CASES = (
    "ordinary_full",
    "ordinary_decayed",
    "near_ko",
    "sigma_collapse",
    "near_expiry",
)
# The two carried schema-11 replacement cases must be aligned into this common
# cohort. New primary/middle families retain all of their own raw scrambles;
# the independent-cohort Welch gate does not force them to have this size.
AGGREGATE_OUTER_BATCHES = 128
AGGREGATE_COHORT_NAMES = (
    "primary_refresh",
    "middle_control",
    "parent_heston_high",
    "schema11_replacements",
)
# The frozen-SLV coefficient inherits the independently selected schema-10
# development design.  Before either held-out seed is opened, the single
# smooth-case Heston coefficient is selected from the declared development
# grid using only the pilot's matched F/H rows and the known carried-high batch
# count. For each middle allocation it minimizes the worse D+S/D-S variance of
# ``F-b*H_low+b*H_high`` under an independent matched-high proxy. The allocation
# objective is then the smallest total path count whose projected simultaneous
# endpoint interval, plus one new-family standard error of guard, is wholly
# inside the economic bound; ties use the smaller worst endpoint SE.
# Low-Feller deliberately omits the Heston layer: its 7->14 ladder has no
# matching parent Heston expectation, and this amendment never reruns a
# completed Heston case.
FROZEN_SMOOTH_HESTON_WEIGHT = 0.7
AGGREGATE_CONTROL_WEIGHTS = {
    case_name: {
        "frozen_slv": 0.95,
        "heston": (
            FROZEN_SMOOTH_HESTON_WEIGHT
            if case_name in PRIMARY_REFRESH_CASES
            else 0.0
        ),
    }
    for case_name in CONTROL_CASES
}

# Frozen before either held-out seed was opened.  The exact development design
# commit, non-admissive pilot bytes, and allocation projection are retained as
# provenance; production does not read them or adapt its fixed sample sizes.
FROZEN_ALLOCATION_DESIGN_COMMIT = (
    "b5a5243d0335081e18c9c92dfebbb5f1f450f859"
)
FROZEN_DEVELOPMENT_IMPLEMENTATION_SHA256 = (
    "e9d21aa02b9e86a49cdc674ce97a2dd886c06f672344a28a7a8187e68ce3846f"
)
FROZEN_ALLOCATION_PROJECTION_SHA256 = (
    "3e007060710eaba934180c69ffe6579822bfe84a13bca9f8c81751c21bf65bc6"
)
FROZEN_ALLOCATION_PROJECTION_FILE_SHA256 = (
    "3e1327c76b88ce53eb2695f786117fa911579878990690a899a3c6d7b1f18c7e"
)
FROZEN_DEVELOPMENT_PILOTS = (
    (
        ("ordinary_full",),
        "9640f0cf4a3eb7f20a7ac2954b36f6458187b4b956a3abfc8eee09794d4f16e8",
        "c1beb0f4b4383d9941fe09ea1626274a486a76aa2da39da268ad08d99a0a90ff",
    ),
    (
        ("ordinary_decayed", "near_ko", "sigma_collapse", "near_expiry"),
        "14c6747ef4668c96a82e2f599417fc22926f331397ef9091fe3ce8329527e164",
        "fbdfa01158dcc7b0380aad23211676ea8e059cfb1585cf18a7d824248efa3208",
    ),
    (
        ("low_feller",),
        "5199937088502e85299f412f17f8db950171d195bc525ce678bc3114cb9e70ec",
        "4b158c959463db71857da9bccc3d1d891752eed8b8ebbd06808693a644f82a97",
    ),
)
FROZEN_ALLOCATION_PROJECTED_INTERVAL = (
    -0.09606341420341855,
    -0.041846226147214255,
)
FROZEN_ALLOCATION_GUARDED_INTERVAL = (
    -0.09927214226746098,
    -0.038637498083171816,
)
FROZEN_ALLOCATION_TOTAL_UNIQUE_PATHS = 67_108_864

PRODUCTION_ALLOCATION_FROZEN = True
PRODUCTION_PRIMARY_SEED = 20260811
PRODUCTION_MIDDLE_SEED = 20260812
PRODUCTION_PRIMARY_PATHS_PER_BATCH = 1024
PRODUCTION_PRIMARY_BATCHES = 4096
PRODUCTION_MIDDLE_PATHS_PER_BATCH = 8192
PRODUCTION_MIDDLE_BATCHES = 256
PRODUCTION_PRIMARY_BATCH_WORKERS = 4
PRODUCTION_MIDDLE_BATCH_WORKERS = 2
PRODUCTION_PRIMARY_CELL_WORKERS = 3
PRODUCTION_MIDDLE_CELL_WORKERS = 2

IMPLEMENTATION_INPUTS = (
    "example/mo_volmodels/12_snowball_volmodel_backtest.py",
    "example/mo_volmodels/16_adi_greek_certification.py",
    "example/mo_volmodels/17_adi_slv_aggregate_certification.py",
    "quantark/validation/greek_certification.py",
    "quantark/asset/equity/engine/mc/snowball_mc_engine.py",
    "quantark/asset/equity/engine/mc/snowball_vol_mc_engines.py",
    "quantark/montecarlo/qmc_rqmc_driver.py",
    "quantark/montecarlo/qmc_qe_coupling.py",
)


def stage16():
    """Load the schema-11 implementation without relying on package imports."""
    global _STAGE16
    if _STAGE16 is None:
        spec = importlib.util.spec_from_file_location(
            "mo_adi_greek_certification_16_for_17", STAGE16_PATH
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load schema-11 module at {STAGE16_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _STAGE16 = module
    return _STAGE16


def _canonical_sha256(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_environment() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_INPUTS:
        path = root / relative
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot hash aggregate certification input {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def frozen_allocation_manifest() -> dict:
    """Return the immutable development-to-production allocation provenance."""
    return {
        "design_commit": FROZEN_ALLOCATION_DESIGN_COMMIT,
        "development_implementation_sha256": (
            FROZEN_DEVELOPMENT_IMPLEMENTATION_SHA256
        ),
        "projection_sha256": FROZEN_ALLOCATION_PROJECTION_SHA256,
        "projection_file_sha256": FROZEN_ALLOCATION_PROJECTION_FILE_SHA256,
        "development_pilots": [
            {
                "cases": list(cases),
                "evidence_sha256": evidence_hash,
                "file_sha256": file_hash,
            }
            for cases, evidence_hash, file_hash in FROZEN_DEVELOPMENT_PILOTS
        ],
        "recommendation": {
            "primary_batches": PRODUCTION_PRIMARY_BATCHES,
            "middle_batches": PRODUCTION_MIDDLE_BATCHES,
            "smooth_heston_weight": FROZEN_SMOOTH_HESTON_WEIGHT,
            "total_unique_paths": FROZEN_ALLOCATION_TOTAL_UNIQUE_PATHS,
            "projected_interval": list(FROZEN_ALLOCATION_PROJECTED_INTERVAL),
            "guarded_interval": list(FROZEN_ALLOCATION_GUARDED_INTERVAL),
        },
    }


MIN_ADAPTIVE_BATCHES = 32
DEFAULT_PRECISION_TARGET_CONTRACTS = 0.02
DEFAULT_ADAPTIVE_BUDGET_HOURS = 12.0


def freeze_adaptive_allocation(pilot: dict, *, budget_hours: float) -> dict:
    """Turn pilot precision statistics into a frozen, hash-pinned allocation.

    The pilot supplies per-cell batch SD and per-batch cost; the allocation is
    the cost-weighted Neyman optimum for that budget. It is frozen and hashed
    BEFORE the main run so the recorded decision cannot drift with the data
    that follows it.

    Note that Neyman weighting answers a shared-budget question. When every
    cell gets its own worker stream (7 cells on 14 cores) the streams do not
    compete and each simply fills its stream instead; the weighting matters in
    the memory-constrained and serial configurations.
    """
    cells = [
        CellPrecision(
            name=name,
            n_batches=int(stats["batches"]),
            batch_sd=float(stats["batch_sd"]),
            seconds_per_batch=float(stats["seconds_per_batch"]),
        )
        for name, stats in sorted(pilot.items())
    ]
    allocation = neyman_allocation(
        cells,
        budget_seconds=float(budget_hours) * 3600.0,
        min_batches=MIN_ADAPTIVE_BATCHES,
    )
    frozen = {
        "pilot": {name: dict(stats) for name, stats in sorted(pilot.items())},
        "allocation": allocation,
        "budget_hours": float(budget_hours),
        "pilot_halfwidth": projected_aggregate_halfwidth(cells),
    }
    frozen["allocation_sha256"] = _canonical_sha256(frozen)
    return frozen


def adaptive_batches_by_case(frozen: dict, *, default_batches: int) -> dict:
    """Per-case batch counts from a frozen allocation, defaulting elsewhere.

    Cells absent from the pilot (already-treated or low-variance cells) keep
    the frozen production count, so an adaptive run never *reduces* evidence
    below what the fixed allocation would have banked.
    """
    allocation = frozen["allocation"]
    return {
        case_name: max(int(allocation.get(case_name, 0)), int(default_batches))
        for case_name in CONTROL_CASES
    }


def monitor_cells_from_banked(banked: dict) -> list:
    """Precision-only view of the banked batches, estimate-blind by type.

    ``CellPrecision`` has no field an estimate can travel through, so the
    stopping path structurally cannot read one (spec gate S-G1) no matter what
    else the banked records carry.
    """
    cells = []
    for name, record in sorted(banked.items()):
        deltas = np.asarray(record["batch_deltas"], dtype=float)
        cells.append(
            CellPrecision(
                name=name,
                n_batches=int(deltas.size),
                batch_sd=float(np.std(deltas, ddof=1)),
                seconds_per_batch=float(record["seconds_per_batch"]),
            )
        )
    return cells


def production_run_configuration(
    *,
    implementation_hash: str,
    runtime: dict,
    adaptive: bool = False,
    precision_target: float = DEFAULT_PRECISION_TARGET_CONTRACTS,
    budget_hours: float = DEFAULT_ADAPTIVE_BUDGET_HOURS,
    pilot_batches: int = MIN_ADAPTIVE_BATCHES,
) -> dict:
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "certification_mode": CERTIFICATION_MODE,
        "implementation_sha256": implementation_hash,
        "runtime_environment": runtime,
        "parent_certificate": parent_certificate_manifest(),
        "production_pde_compatibility_sha256": PARENT_PRODUCTION_PDE_SHA256,
        "allocation_freeze": frozen_allocation_manifest(),
        "schema11_replacement_batches": AGGREGATE_OUTER_BATCHES,
        "primary_refresh": {
            "cases": list(PRIMARY_REFRESH_CASES),
            "seed": PRODUCTION_PRIMARY_SEED,
            "paths_per_batch": PRODUCTION_PRIMARY_PATHS_PER_BATCH,
            "batches": PRODUCTION_PRIMARY_BATCHES,
            "batch_workers": PRODUCTION_PRIMARY_BATCH_WORKERS,
            "cell_workers": PRODUCTION_PRIMARY_CELL_WORKERS,
        },
        "middle_control": {
            "cases": list(CONTROL_CASES),
            "seed": PRODUCTION_MIDDLE_SEED,
            "paths_per_batch": PRODUCTION_MIDDLE_PATHS_PER_BATCH,
            "batches": PRODUCTION_MIDDLE_BATCHES,
            "batch_workers": PRODUCTION_MIDDLE_BATCH_WORKERS,
            "cell_workers": PRODUCTION_MIDDLE_CELL_WORKERS,
        },
        "weights_by_case": AGGREGATE_CONTROL_WEIGHTS,
        "allocation_selection_policy": {
            "development_seed": DEVELOPMENT_SEED,
            "primary_batch_grid": list(DEVELOPMENT_PRIMARY_BATCH_GRID),
            "middle_batch_grid": list(DEVELOPMENT_MIDDLE_BATCH_GRID),
            "smooth_heston_weight_grid": list(DEVELOPMENT_HESTON_WEIGHT_GRID),
            "new_family_guard_sigmas": DEVELOPMENT_NEW_FAMILY_GUARD_SIGMAS,
            "variance_upper_confidence": (
                DEVELOPMENT_VARIANCE_UPPER_CONFIDENCE
            ),
            "objective": (
                "pilot_only_weight_then_least_paths_then_worst_endpoint_se"
            ),
            "projection_center": "schema11_all_available_rows_D_plus_minus_S",
        },
        "near_ki_source": "schema11_published_multilevel_reference",
        "heston_high_source": "schema11_parent_cells_no_rerun",
        "endpoint_method": "paired_D_plus_minus_S_with_independent_cohort_Welch_t",
        "independent_source_cohorts": list(AGGREGATE_COHORT_NAMES),
        "component_confidence": stage16().STOCHASTIC_COMPONENT_CONFIDENCE,
        "economic_bound_contracts": stage16().DELTA_BIAS_BOUND_CONTRACTS,
        # Precision stopping is not optional stopping: the rule reads achieved
        # standard errors and elapsed time, never the estimate, so the terminal
        # fixed-confidence verdict needs no sequential-testing correction. Both
        # facts are recorded explicitly rather than left to be inferred from
        # the absence of a stopping record.
        "no_optional_stopping": True,
        "stopping_rule": "precision_blind" if adaptive else "fixed_allocation",
        "stopping_rule_reads_estimate": False,
    }
    if adaptive:
        configuration["adaptive_run"] = {
            "precision_target": float(precision_target),
            "budget_hours": float(budget_hours),
            "pilot_batches": int(pilot_batches),
            "frozen_allocation_fallback": {
                "primary_batches": PRODUCTION_PRIMARY_BATCHES,
                "middle_batches": PRODUCTION_MIDDLE_BATCHES,
            },
            # The allocation frozen from the pilot sets each cell's batch count
            # up front; batches are then banked in one call per cell/level, so
            # the run has no mid-flight cohort boundary at which a monitor
            # could act. `precision_stop` is exported and tested for the
            # incremental case, and wiring it requires cohort support in
            # build_primary_reference -- deferred deliberately rather than
            # simulated with a hook that could never fire.
            "stopping_granularity": "allocation_frozen_from_pilot",
            "mid_run_monitor": False,
        }
    return configuration


def _checkpoint_path(output_dir: Path, name: str) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not name or any(character not in allowed for character in name):
        raise ValueError(f"unsafe aggregate checkpoint name: {name!r}")
    return output_dir / "checkpoints" / f"{name}.json"


def _write_checkpoint(
    output_dir: Path,
    name: str,
    *,
    run_configuration_sha256: str,
    kind: str,
    evidence: dict,
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_configuration_sha256": run_configuration_sha256,
        "kind": str(kind),
        "evidence_sha256": _canonical_sha256(evidence),
        "evidence": evidence,
    }
    stage16()._atomic_write(
        _checkpoint_path(output_dir, name),
        json.dumps(record, indent=2, sort_keys=True) + "\n",
    )


def _load_checkpoint(
    output_dir: Path,
    name: str,
    *,
    run_configuration_sha256: str,
    kind: str,
) -> Optional[dict]:
    path = _checkpoint_path(output_dir, name)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"aggregate checkpoint {path} is unreadable") from exc
    evidence = record.get("evidence")
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("run_configuration_sha256") != run_configuration_sha256
        or record.get("kind") != kind
        or not isinstance(evidence, dict)
        or record.get("evidence_sha256") != _canonical_sha256(evidence)
    ):
        raise ValueError(f"aggregate checkpoint {path} provenance mismatch")
    return evidence


def parent_certificate_manifest() -> dict:
    return {
        "schema_version": PARENT_SCHEMA_VERSION,
        "source_commit": PARENT_SOURCE_COMMIT,
        "evidence_file_sha256": PARENT_EVIDENCE_FILE_SHA256,
        "decision_file_sha256": PARENT_DECISION_FILE_SHA256,
        "evidence_sha256": PARENT_EVIDENCE_SHA256,
        "decision_sha256": PARENT_DECISION_SHA256,
        "implementation_sha256": PARENT_IMPLEMENTATION_SHA256,
        "run_configuration_sha256": PARENT_RUN_CONFIGURATION_SHA256,
        "production_pde_compatibility_sha256": PARENT_PRODUCTION_PDE_SHA256,
        "anchors_sha256": PARENT_ANCHORS_SHA256,
        "cells_sha256": PARENT_CELLS_SHA256,
        "cell_identities_sha256": PARENT_CELL_IDENTITIES_SHA256,
        "decisions_sha256": PARENT_DECISIONS_SHA256,
        "base_heston_decision_sha256": PARENT_BASE_HESTON_DECISION_SHA256,
        "sampling_by_variant_sha256": PARENT_SAMPLING_BY_VARIANT_SHA256,
    }


def _require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def load_and_validate_parent_certificate(
    evidence_path: Path,
    decision_path: Path,
) -> tuple[dict, dict, dict]:
    """Load the exact stage-16 parent artifact and re-audit its identity.

    The pinned parent is the 2026-08-17 strided re-publish (schema-13 full
    recertification, output/p18_strided): Heston admitted at a 41% margin,
    Heston-SLV honestly unresolved -- the gate this amendment exists to
    resolve with control variates instead of ~150 h of extra paths.
    """
    try:
        evidence_bytes = Path(evidence_path).read_bytes()
        decision_bytes = Path(decision_path).read_bytes()
    except OSError as exc:
        raise ValueError("schema-11 parent certificate is not readable") from exc
    if hashlib.sha256(evidence_bytes).hexdigest() != PARENT_EVIDENCE_FILE_SHA256:
        raise ValueError("schema-11 parent evidence file hash mismatch")
    if hashlib.sha256(decision_bytes).hexdigest() != PARENT_DECISION_FILE_SHA256:
        raise ValueError("schema-11 parent decision file hash mismatch")
    try:
        evidence = json.loads(evidence_bytes)
        decision = json.loads(decision_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("schema-11 parent certificate is not valid JSON") from exc

    if (
        evidence.get("schema_version") != PARENT_SCHEMA_VERSION
        or evidence.get("study") != STUDY
        or evidence.get("certification_mode") != "full_recertification"
        or evidence.get("quick") is not False
        or evidence.get("profile") != "production"
    ):
        raise ValueError("stage-16 parent metadata mismatch")
    # A full-recertification parent embeds no production-PDE compatibility
    # digest (that key is amendment-only); the live projection is still pinned
    # below, which is the check that actually protects the carried cells.
    if (
        evidence.get("evidence_sha256") != PARENT_EVIDENCE_SHA256
        or stage16()._projected_evidence_sha256(evidence) != PARENT_EVIDENCE_SHA256
        or evidence.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256
        or evidence.get("run_configuration_sha256")
        != PARENT_RUN_CONFIGURATION_SHA256
        or _canonical_sha256(evidence.get("run_configuration"))
        != PARENT_RUN_CONFIGURATION_SHA256
        or evidence.get("numerical_implementation_sha256")
        != evidence.get("run_configuration", {}).get(
            "numerical_implementation_sha256"
        )
    ):
        raise ValueError("stage-16 parent canonical provenance mismatch")
    if (
        decision.get("decision_sha256") != PARENT_DECISION_SHA256
        or decision.get("evidence_sha256") != PARENT_EVIDENCE_SHA256
        or decision.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256
        or decision.get("run_configuration_sha256")
        != PARENT_RUN_CONFIGURATION_SHA256
    ):
        raise ValueError("schema-11 parent decision provenance mismatch")
    unsigned_decision = dict(decision)
    unsigned_decision.pop("decision_sha256", None)
    if _canonical_sha256(unsigned_decision) != PARENT_DECISION_SHA256:
        raise ValueError("schema-11 parent decision self-hash mismatch")
    if evidence.get("runtime_environment") != runtime_environment():
        raise ValueError("schema-11 parent numerical runtime mismatch")
    if stage16().production_pde_compatibility_sha256() != PARENT_PRODUCTION_PDE_SHA256:
        raise ValueError("live production PDE differs from the schema-11 parent")

    projections = {
        "anchors": PARENT_ANCHORS_SHA256,
        "cells": PARENT_CELLS_SHA256,
        "cell_identities": PARENT_CELL_IDENTITIES_SHA256,
        "decisions": PARENT_DECISIONS_SHA256,
    }
    for key, expected_hash in projections.items():
        if _canonical_sha256(evidence.get(key)) != expected_hash:
            raise ValueError(f"stage-16 parent {key} projection mismatch")
    if evidence.get("decisions") != decision.get("decisions"):
        raise ValueError("schema-11 parent evidence/decision routing mismatch")
    if _canonical_sha256(evidence.get("sampling_by_variant")) != (
        PARENT_SAMPLING_BY_VARIANT_SHA256
    ):
        raise ValueError("schema-11 parent sampling projection mismatch")
    parent_heston = json.loads(json.dumps(evidence["decisions"].get("heston", {})))
    parent_heston.pop("certification_source", None)
    parent_heston.pop("parent_evidence_sha256", None)
    if _canonical_sha256(parent_heston) != PARENT_BASE_HESTON_DECISION_SHA256:
        raise ValueError("schema-11 parent Heston decision mismatch")
    if evidence["decisions"]["heston"].get("route") != "pde":
        raise ValueError("schema-11 parent does not admit Heston")
    if evidence["decisions"]["heston_slv"].get("route") != (
        "excluded_greek_unresolved"
    ):
        raise ValueError("schema-11 parent SLV route is not the expected unresolved gate")
    cells = evidence.get("cells", [])
    if len(cells) != 14 or any(cell.get("status") != "PASS" for cell in cells):
        raise ValueError("schema-11 parent does not contain fourteen PASS cells")
    return evidence, decision, parent_certificate_manifest()


@dataclass(frozen=True)
class CohortSummary:
    estimate: float
    standard_error: float
    half_width: float
    degrees_of_freedom: int
    confidence: float
    cohort_means: tuple[float, ...]
    cohort_standard_errors: tuple[float, ...]
    cohort_sizes: tuple[int, ...]

    def as_dict(self) -> dict:
        payload = asdict(self)
        for key in ("cohort_means", "cohort_standard_errors", "cohort_sizes"):
            payload[key] = list(payload[key])
        return payload


def summarize_independent_cohorts(
    cohorts: Sequence[np.ndarray],
    *,
    confidence: float,
) -> CohortSummary:
    """Summarize a sum of independent cohort means with floored Welch df."""
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arrays = tuple(np.asarray(cohort, dtype=float) for cohort in cohorts)
    if not arrays:
        raise ValueError("at least one independent cohort is required")
    if any(
        array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array))
        for array in arrays
    ):
        raise ValueError("cohorts must be finite one-dimensional samples of size >= 2")
    means = tuple(float(np.mean(array)) for array in arrays)
    standard_errors = tuple(
        float(np.std(array, ddof=1) / math.sqrt(array.size)) for array in arrays
    )
    variances = tuple(value * value for value in standard_errors)
    total_variance = float(sum(variances))
    if total_variance == 0.0:
        degrees_of_freedom = int(sum(array.size - 1 for array in arrays))
        half_width = 0.0
    else:
        denominator = sum(
            variance * variance / (array.size - 1)
            for variance, array in zip(variances, arrays)
        )
        raw_df = total_variance * total_variance / denominator
        degrees_of_freedom = max(1, int(math.floor(raw_df)))
        half_width = float(
            student_t.ppf(0.5 + 0.5 * confidence, degrees_of_freedom)
            * math.sqrt(total_variance)
        )
    return CohortSummary(
        estimate=float(sum(means)),
        standard_error=math.sqrt(total_variance),
        half_width=half_width,
        degrees_of_freedom=degrees_of_freedom,
        confidence=float(confidence),
        cohort_means=means,
        cohort_standard_errors=standard_errors,
        cohort_sizes=tuple(int(array.size) for array in arrays),
    )


def certify_joint_bias_endpoints(
    delta_cohorts: Sequence[np.ndarray],
    substep_cohorts: Sequence[np.ndarray],
    *,
    economic_bound: float,
    pde_discretization_envelope: float,
    component_confidence: float,
    label: str,
) -> dict:
    """Certify ``D +/- S`` endpoints with simultaneous component coverage.

    If the remaining fine-reference bias is bounded by ``abs(E[S])``, the true
    signed PDE bias lies between ``E[D-S]`` and ``E[D+S]``.  The two endpoint
    intervals use Bonferroni component confidence and keep the paired
    covariance in every cohort before independent cohort variances are added.
    """
    if len(delta_cohorts) != len(substep_cohorts) or not delta_cohorts:
        raise ValueError("delta and substep cohort structures must match")
    if not np.isfinite(economic_bound) or economic_bound <= 0.0:
        raise ValueError("economic_bound must be finite and positive")
    if (
        not np.isfinite(pde_discretization_envelope)
        or pde_discretization_envelope < 0.0
    ):
        raise ValueError("PDE envelope must be finite and non-negative")
    endpoint_summaries = {}
    for name, sign in (("delta_plus_substep", 1.0), ("delta_minus_substep", -1.0)):
        rows = []
        for delta, substep in zip(delta_cohorts, substep_cohorts):
            delta_array = np.asarray(delta, dtype=float)
            substep_array = np.asarray(substep, dtype=float)
            if delta_array.shape != substep_array.shape:
                raise ValueError("paired delta/substep rows must have matching shapes")
            rows.append(delta_array + sign * substep_array)
        endpoint_summaries[name] = summarize_independent_cohorts(
            rows, confidence=component_confidence
        )
    lower = min(
        summary.estimate - summary.half_width
        for summary in endpoint_summaries.values()
    ) - float(pde_discretization_envelope)
    upper = max(
        summary.estimate + summary.half_width
        for summary in endpoint_summaries.values()
    ) + float(pde_discretization_envelope)
    if lower > economic_bound or upper < -economic_bound:
        status = EquivalenceStatus.FAIL.value
        reason = "simultaneous signed-bias interval is wholly outside the bound"
    elif lower >= -economic_bound and upper <= economic_bound:
        status = EquivalenceStatus.PASS.value
        reason = "simultaneous signed-bias interval is wholly inside the bound"
    else:
        status = EquivalenceStatus.INCONCLUSIVE.value
        reason = "simultaneous signed-bias interval overlaps the bound"
    return {
        "label": str(label),
        "status": status,
        "reason": reason,
        "economic_bound": float(economic_bound),
        "interval": [float(lower), float(upper)],
        "pde_discretization_envelope": float(pde_discretization_envelope),
        "component_confidence": float(component_confidence),
        "simultaneous_coverage_lower_bound": float(
            1.0 - 2.0 * (1.0 - component_confidence)
        ),
        "endpoint_method": "paired_D_plus_minus_S_with_independent_cohort_Welch_t",
        "endpoints": {
            name: summary.as_dict() for name, summary in endpoint_summaries.items()
        },
    }


def _delta_contract_factor(cell: dict) -> float:
    scale = cell.get("economic_scale", {})
    values = (
        float(scale.get("study_notional", float("nan"))),
        float(scale.get("hedge_inception_spot", float("nan"))),
        float(scale.get("hedge_multiplier", float("nan"))),
    )
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("cell has an invalid economic Delta scale")
    return values[0] / values[1] / values[2]


def _serialized_delta_rows(payload: dict, *, control: bool = False) -> np.ndarray:
    key = "control_batch_estimates" if control else "batch_estimates"
    rows = np.asarray(payload.get(key), dtype=float)
    batches = int(payload.get("batches_used", 0))
    if rows.shape != (batches, 5) or batches < 2 or not np.all(np.isfinite(rows)):
        raise ValueError(f"invalid serialized paired-RQMC {key}")
    return rows[:, 3]


def _group_delta_rows(
    payload: dict,
    *,
    output_batches: int,
    control: bool = False,
) -> np.ndarray:
    rows = _serialized_delta_rows(payload, control=control)
    if output_batches < 2 or rows.size % output_batches != 0:
        raise ValueError("paired-RQMC rows are not divisible by outer batches")
    # STRIDED grouping: output row j averages scrambles {j, j+m, j+2m, ...}
    # (m = output_batches), not the consecutive block {gj .. gj+g-1}. Scramble
    # j of every case therefore lands in output row j, so same-scramble CRN
    # coupling across cases stays inside one output row and the output rows
    # remain mutually independent. Consecutive grouping pushes that coupling
    # across output-row boundaries, where the empirical standard error cannot
    # see it -- measured on the banked stage-16 fleet in
    # docs/adi2d-greek-perf/probes/probe_crn_strided_alignment.py, where the
    # over-cell coupling is real (corr -0.43 on heston low_feller x near_ki).
    # Identical to the parent stage-16 aggregate's declared alignment.
    return rows.reshape(rows.size // output_batches, output_batches).mean(axis=0)


def controlled_case_economic_rows(
    *,
    pde_delta: float,
    economic_factor: float,
    primary_reference: dict,
    middle_reference: dict,
    heston_high_reference: Optional[dict],
    frozen_weight: float,
    heston_weight: float,
    output_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompose unbiased aggregate-only Delta and substep rows for one case."""
    components = controlled_case_economic_components(
        pde_delta=pde_delta,
        economic_factor=economic_factor,
        primary_reference=primary_reference,
        middle_reference=middle_reference,
        heston_high_reference=heston_high_reference,
        frozen_weight=frozen_weight,
        heston_weight=heston_weight,
        output_batches=output_batches,
    )
    delta = sum(
        (component["delta"] for component in components.values()),
        np.zeros(output_batches, dtype=float),
    )
    substep = sum(
        (component["substep"] for component in components.values()),
        np.zeros(output_batches, dtype=float),
    )
    return delta, substep


def controlled_case_economic_components(
    *,
    pde_delta: float,
    economic_factor: float,
    primary_reference: dict,
    middle_reference: dict,
    heston_high_reference: Optional[dict],
    frozen_weight: float,
    heston_weight: float,
    output_batches: int | dict[str, int],
) -> dict[str, dict[str, np.ndarray]]:
    """Split one unbiased case estimate along independent seed families."""
    if not np.isfinite(pde_delta) or not np.isfinite(economic_factor):
        raise ValueError("case PDE Delta and economic factor must be finite")
    if economic_factor <= 0.0:
        raise ValueError("case economic factor must be positive")
    if not np.isfinite(frozen_weight) or not np.isfinite(heston_weight):
        raise ValueError("control weights must be finite")
    if heston_weight != 0.0 and heston_high_reference is None:
        raise ValueError("non-zero Heston control requires a high expectation")

    if isinstance(output_batches, dict):
        expected_names = {"primary", "middle", "heston_high"}
        if set(output_batches) != expected_names:
            raise ValueError("component output-batch map is incomplete")
        batches_by_component = {
            name: int(value) for name, value in output_batches.items()
        }
    else:
        batches_by_component = {
            name: int(output_batches)
            for name in ("primary", "middle", "heston_high")
        }
    if any(value < 2 for value in batches_by_component.values()):
        raise ValueError("component output batches must be at least two")

    primary = {}
    middle = {}
    heston_high_rows = {}
    for level in ("target", "fine"):
        primary_payload = primary_reference.get(level, {})
        middle_payload = middle_reference.get(level, {})
        state_dependent = _group_delta_rows(
            primary_payload, output_batches=batches_by_component["primary"]
        )
        frozen_low = _group_delta_rows(
            primary_payload,
            output_batches=batches_by_component["primary"],
            control=True,
        )
        frozen_high = _group_delta_rows(
            middle_payload,
            output_batches=batches_by_component["middle"],
        )
        heston_low = _group_delta_rows(
            middle_payload,
            output_batches=batches_by_component["middle"],
            control=True,
        )
        primary[level] = state_dependent - frozen_weight * frozen_low
        middle[level] = frozen_weight * frozen_high - heston_weight * heston_low
        if heston_weight != 0.0:
            heston_high_rows[level] = heston_weight * _group_delta_rows(
                heston_high_reference.get(level, {}),
                output_batches=batches_by_component["heston_high"],
            )
        else:
            heston_high_rows[level] = np.zeros(
                batches_by_component["heston_high"], dtype=float
            )
    components = {
        "primary": {
            "delta": (float(pde_delta) - primary["fine"]) * economic_factor,
            "substep": (primary["target"] - primary["fine"]) * economic_factor,
        },
        "middle": {
            "delta": -middle["fine"] * economic_factor,
            "substep": (middle["target"] - middle["fine"]) * economic_factor,
        },
        "heston_high": {
            "delta": -heston_high_rows["fine"] * economic_factor,
            "substep": (
                heston_high_rows["target"] - heston_high_rows["fine"]
            )
            * economic_factor,
        },
    }
    if any(
        not np.all(np.isfinite(values))
        for component in components.values()
        for values in component.values()
    ):
        raise ValueError("controlled case recomposition returned non-finite rows")
    return components


def aggregate_reference_cohorts(
    parent: dict,
    *,
    primary_by_case: dict,
    middle_by_case: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Recompose four independent 128-row source families for the fleet."""
    base = stage16()
    cells_by_key = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell
        for cell in parent.get("cells", [])
    }
    expected = {
        (variant, case.name)
        for variant in ("heston", "heston_slv")
        for case in base.certification_cases(quick=False)
    }
    if set(cells_by_key) != expected:
        raise ValueError("schema-11 parent regime matrix is incomplete")
    if set(primary_by_case) != set(PRIMARY_REFRESH_CASES):
        raise ValueError("aggregate primary refresh case set mismatch")
    if set(middle_by_case) != set(CONTROL_CASES):
        raise ValueError("aggregate middle-control case set mismatch")

    primary_batches = {
        int(primary_by_case[name]["target"]["batches_used"])
        for name in PRIMARY_REFRESH_CASES
    }
    middle_batches = {
        int(middle_by_case[name]["target"]["batches_used"])
        for name in CONTROL_CASES
    }
    heston_batches = {
        int(cells_by_key[("heston", name)]["reference"]["target"]["batches_used"])
        for name in PRIMARY_REFRESH_CASES
    }
    if len(primary_batches) != 1 or len(middle_batches) != 1 or len(
        heston_batches
    ) != 1:
        raise ValueError("aggregate source families do not have common batch counts")
    cohort_sizes = {
        "primary_refresh": primary_batches.pop(),
        "middle_control": middle_batches.pop(),
        "parent_heston_high": heston_batches.pop(),
        "schema11_replacements": AGGREGATE_OUTER_BATCHES,
    }
    delta_cohorts = {
        name: np.zeros(cohort_sizes[name], dtype=float)
        for name in AGGREGATE_COHORT_NAMES
    }
    substep_cohorts = {
        name: np.zeros(cohort_sizes[name], dtype=float)
        for name in AGGREGATE_COHORT_NAMES
    }
    component_hashes = {}
    for case_name in CONTROL_CASES:
        slv_cell = cells_by_key[("heston_slv", case_name)]
        uses_refresh = case_name in PRIMARY_REFRESH_CASES
        primary_reference = (
            primary_by_case[case_name] if uses_refresh else slv_cell["reference"]
        )
        weights = AGGREGATE_CONTROL_WEIGHTS[case_name]
        heston_high = (
            cells_by_key[("heston", case_name)]["reference"]
            if weights["heston"] != 0.0
            else None
        )
        components = controlled_case_economic_components(
            pde_delta=float(slv_cell["certifications"]["delta"]["pde"]),
            economic_factor=_delta_contract_factor(slv_cell),
            primary_reference=primary_reference,
            middle_reference=middle_by_case[case_name],
            heston_high_reference=heston_high,
            frozen_weight=float(weights["frozen_slv"]),
            heston_weight=float(weights["heston"]),
            output_batches={
                "primary": (
                    cohort_sizes["primary_refresh"]
                    if uses_refresh
                    else cohort_sizes["schema11_replacements"]
                ),
                "middle": cohort_sizes["middle_control"],
                "heston_high": cohort_sizes["parent_heston_high"],
            },
        )
        primary_cohort = "primary_refresh" if uses_refresh else "schema11_replacements"
        mapping = {
            "primary": primary_cohort,
            "middle": "middle_control",
            "heston_high": "parent_heston_high",
        }
        for component_name, cohort_name in mapping.items():
            delta_cohorts[cohort_name] += components[component_name]["delta"] / 7.0
            substep_cohorts[cohort_name] += (
                components[component_name]["substep"] / 7.0
            )
        component_hashes[case_name] = {
            "primary_reference_sha256": _canonical_sha256(primary_reference),
            "middle_reference_sha256": _canonical_sha256(
                middle_by_case[case_name]
            ),
            "heston_high_source_sha256": (
                None if heston_high is None else _canonical_sha256(heston_high)
            ),
        }

    near_ki = cells_by_key[("heston_slv", "near_ki")]
    near_target = _group_delta_rows(
        near_ki["reference"]["target"], output_batches=AGGREGATE_OUTER_BATCHES
    )
    near_fine = _group_delta_rows(
        near_ki["reference"]["fine"], output_batches=AGGREGATE_OUTER_BATCHES
    )
    near_factor = _delta_contract_factor(near_ki)
    delta_cohorts["schema11_replacements"] += (
        float(near_ki["certifications"]["delta"]["pde"]) - near_fine
    ) * near_factor / 7.0
    substep_cohorts["schema11_replacements"] += (
        near_target - near_fine
    ) * near_factor / 7.0
    component_hashes["near_ki"] = {
        "source": "schema11_published_multilevel_reference",
        "reference_sha256": _canonical_sha256(near_ki["reference"]),
    }
    return delta_cohorts, substep_cohorts, component_hashes


def aggregate_pde_refinement(parent: dict) -> tuple[dict, float]:
    rows = [
        cell
        for cell in parent.get("cells", [])
        if cell.get("variant") == "heston_slv"
    ]
    if len(rows) != 7:
        raise ValueError("schema-11 parent must contain seven SLV cells")
    axes = {
        axis: float(
            np.mean(
                [
                    row["certifications"]["delta"][
                        "pde_signed_refinement_contracts"
                    ][axis]
                    for row in rows
                ]
            )
        )
        for axis in ("n_x", "n_v", "n_t")
    }
    return axes, float(sum(abs(value) for value in axes.values()))


def make_aggregate_decisions(
    parent: dict,
    delta_cohorts: dict[str, np.ndarray],
    substep_cohorts: dict[str, np.ndarray],
) -> dict:
    """Carry Heston and route SLV from four independent source cohorts."""
    if tuple(delta_cohorts) != AGGREGATE_COHORT_NAMES or tuple(
        substep_cohorts
    ) != AGGREGATE_COHORT_NAMES:
        raise ValueError("aggregate decision cohort ordering or membership mismatch")
    cohort_sizes = {}
    for name in AGGREGATE_COHORT_NAMES:
        delta = np.asarray(delta_cohorts[name], dtype=float)
        substep = np.asarray(substep_cohorts[name], dtype=float)
        if (
            delta.ndim != 1
            or delta.size < 2
            or substep.shape != delta.shape
            or not np.all(np.isfinite(delta))
            or not np.all(np.isfinite(substep))
        ):
            raise ValueError(
                f"aggregate decision rows do not match frozen cohort {name}"
            )
        cohort_sizes[name] = int(delta.size)
    cells = parent.get("cells", [])
    if len(cells) != 14 or any(cell.get("status") != "PASS" for cell in cells):
        raise ValueError("aggregate amendment requires fourteen parent PASS cells")
    parent_decisions = parent.get("decisions", {})
    heston = json.loads(json.dumps(parent_decisions.get("heston", {})))
    if heston.get("route") != "pde" or heston.get("evidence_complete") is not True:
        raise ValueError("aggregate amendment requires the admitted parent Heston route")
    heston["certification_source"] = "schema11_parent"
    heston["parent_evidence_sha256"] = PARENT_EVIDENCE_SHA256

    axes, pde_envelope = aggregate_pde_refinement(parent)
    bias = certify_joint_bias_endpoints(
        [delta_cohorts[name] for name in AGGREGATE_COHORT_NAMES],
        [substep_cohorts[name] for name in AGGREGATE_COHORT_NAMES],
        economic_bound=stage16().DELTA_BIAS_BOUND_CONTRACTS,
        pde_discretization_envelope=pde_envelope,
        component_confidence=stage16().STOCHASTIC_COMPONENT_CONFIDENCE,
        label="heston_slv/mean_signed_delta_bias",
    )
    bias["aggregate_pde_signed_refinement_contracts"] = axes
    bias["aggregate_cohort_sizes"] = cohort_sizes
    bias["aggregate_independent_cohorts"] = [
        {
            "name": name,
            "batches": cohort_sizes[name],
            "seed_family": {
                "primary_refresh": PRODUCTION_PRIMARY_SEED,
                "middle_control": PRODUCTION_MIDDLE_SEED,
                "parent_heston_high": 20260807,
                "schema11_replacements": 20260809,
            }[name],
        }
        for name in AGGREGATE_COHORT_NAMES
    ]
    bias["aggregate_cases"] = [
        case.name for case in stage16().certification_cases(quick=False)
    ]
    admissible = bias["status"] == EquivalenceStatus.PASS.value
    slv = {
        "route": "pde" if admissible else "excluded_greek_unresolved",
        "cell_status": EquivalenceStatus.PASS.value,
        "anchor_status": EquivalenceStatus.PASS.value,
        "evidence_complete": True,
        "missing_anchors": [],
        "missing_cases": [],
        "sampling_complete": True,
        "aggregate_common_scrambles": min(cohort_sizes.values()),
        "aggregate_cohort_sizes": cohort_sizes,
        "aggregate_method": "independent_source_cohorts_joint_D_plus_minus_S",
        "delta_bias": bias,
        "reason": (
            "all certification gates pass"
            if admissible
            else f"mean signed delta bias is {bias['status']}"
        ),
        "certification_source": "schema12_aggregate_only_amendment",
        "parent_evidence_sha256": PARENT_EVIDENCE_SHA256,
    }
    return {"heston": heston, "heston_slv": slv}


def build_control_only_reference(
    case,
    *,
    paths_per_batch: int,
    batches: int,
    seed: int,
    batch_workers: int,
    levels: Sequence[str] = ("target", "fine"),
    purpose: str = "aggregate_frozen_slv_heston_control_pilot",
) -> dict:
    """Evaluate only frozen-SLV and matched Heston controls at target/fine."""
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels) or any(
        level not in {"target", "fine"} for level in levels
    ):
        raise ValueError("control-only levels must be a unique target/fine subset")
    base = stage16()
    product = base.make_snowball(case, dense_ki=True)
    env = base.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = base.make_leverage_surface(case.maturity)
    substeps = base.PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE[case.name]
    bridge = base.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
    probe = base.make_mc_engine(
        "heston_slv",
        case,
        leverage,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=1,
        slv_spot_strata=1,
        slv_spot_antithetic=False,
        slv_spot_bridge_strata=bridge["strata"],
        slv_spot_bridge_dimensions=bridge["dimensions"],
    )
    _, contractual_dt, _, _ = probe._build_time_grid(
        product, env, float(case.maturity)
    )
    target_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / substeps["target"],
        substeps["target"],
    )
    fine_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / substeps["fine"],
        substeps["fine"],
    )
    target_provider, fine_provider = base.coupled_qe_providers(
        seed=seed,
        paths_per_batch=paths_per_batch,
        target_dt=target_dt,
        fine_dt=fine_dt,
    )
    references = {}
    providers = {"target": target_provider, "fine": fine_provider}
    for level in levels:
        provider = providers[level]
        references[level] = base.paired_mc_reference(
            "heston_slv",
            case,
            product,
            env,
            leverage,
            paths_per_batch=paths_per_batch,
            batches=batches,
            seed=seed,
            substeps=substeps[level],
            bump=base.SPOT_BUMP,
            qe_draw_provider=provider,
            slv_spot_strata=1,
            slv_spot_antithetic=False,
            slv_spot_bridge_strata=bridge["strata"],
            slv_spot_bridge_dimensions=bridge["dimensions"],
            slv_conditional_control_only=True,
            rqmc_batch_workers=batch_workers,
        )
    payload = {
        "variant": "heston_slv",
        "case": case.as_dict(),
        "purpose": str(purpose),
        "seed": int(seed),
        "paths_per_batch": int(paths_per_batch),
        "batches": int(batches),
        "batch_workers": int(batch_workers),
        "target_substeps_per_interval": int(substeps["target"]),
        "fine_substeps_per_interval": int(substeps["fine"]),
        "slv_spot_bridge_profile": dict(bridge),
    }
    payload.update(
        {
            level: references[level].as_dict(include_batches=True)
            for level in levels
        }
    )
    return payload


def build_primary_reference(
    case,
    *,
    paths_per_batch: int,
    batches: int,
    seed: int,
    batch_workers: int,
    levels: Sequence[str] = ("target", "fine"),
) -> dict:
    """Evaluate only the state-dependent target/fine SLV reference pair."""
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels) or any(
        level not in {"target", "fine"} for level in levels
    ):
        raise ValueError("primary levels must be a unique target/fine subset")
    base = stage16()
    product = base.make_snowball(case, dense_ki=True)
    env = base.make_environment(
        case.spot, math.sqrt(max(case.params.v0, case.params.theta))
    )
    leverage = base.make_leverage_surface(case.maturity)
    substeps = base.PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE[case.name]
    bridge = base.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
    probe = base.make_mc_engine(
        "heston_slv",
        case,
        leverage,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=1,
        slv_spot_strata=base.SLV_SPOT_STRATA,
        slv_spot_antithetic=base.SLV_SPOT_ANTITHETIC,
        slv_spot_bridge_strata=bridge["strata"],
        slv_spot_bridge_dimensions=bridge["dimensions"],
    )
    _, contractual_dt, _, _ = probe._build_time_grid(
        product, env, float(case.maturity)
    )
    target_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / substeps["target"],
        substeps["target"],
    )
    fine_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / substeps["fine"],
        substeps["fine"],
    )
    target_provider, fine_provider = base.coupled_qe_providers(
        seed=seed,
        paths_per_batch=paths_per_batch,
        target_dt=target_dt,
        fine_dt=fine_dt,
    )
    references = {}
    providers = {"target": target_provider, "fine": fine_provider}
    for level in levels:
        provider = providers[level]
        references[level] = base.paired_mc_reference(
            "heston_slv",
            case,
            product,
            env,
            leverage,
            paths_per_batch=paths_per_batch,
            batches=batches,
            seed=seed,
            substeps=substeps[level],
            bump=base.SPOT_BUMP,
            qe_draw_provider=provider,
            slv_spot_strata=base.SLV_SPOT_STRATA,
            slv_spot_antithetic=base.SLV_SPOT_ANTITHETIC,
            slv_spot_bridge_strata=bridge["strata"],
            slv_spot_bridge_dimensions=bridge["dimensions"],
            rqmc_batch_workers=batch_workers,
        )
    payload = {
        "variant": "heston_slv",
        "case": case.as_dict(),
        "purpose": "aggregate_only_primary_reference_refresh",
        "seed": int(seed),
        "paths_per_batch": int(paths_per_batch),
        "batches": int(batches),
        "batch_workers": int(batch_workers),
        "target_substeps_per_interval": int(substeps["target"]),
        "fine_substeps_per_interval": int(substeps["fine"]),
        "slv_spot_strata": int(base.SLV_SPOT_STRATA),
        "slv_spot_antithetic": bool(base.SLV_SPOT_ANTITHETIC),
        "slv_spot_bridge_profile": dict(bridge),
    }
    payload.update(
        {
            level: references[level].as_dict(include_batches=True)
            for level in levels
        }
    )
    return payload


def validate_added_reference(
    evidence: dict,
    *,
    case,
    purpose: str,
    seed: int,
    paths_per_batch: int,
    batches: int,
    batch_workers: int,
    primary: bool,
) -> None:
    base = stage16()
    substeps = base.PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE[case.name]
    bridge = base.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
    expected = {
        "variant": "heston_slv",
        "case": case.as_dict(),
        "purpose": purpose,
        "seed": int(seed),
        "paths_per_batch": int(paths_per_batch),
        "batches": int(batches),
        "batch_workers": int(batch_workers),
        "target_substeps_per_interval": int(substeps["target"]),
        "fine_substeps_per_interval": int(substeps["fine"]),
        "slv_spot_bridge_profile": dict(bridge),
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"{case.name}: added reference {key} mismatch")
    if primary:
        if (
            int(evidence.get("slv_spot_strata", 0)) != base.SLV_SPOT_STRATA
            or bool(evidence.get("slv_spot_antithetic", False))
            != base.SLV_SPOT_ANTITHETIC
        ):
            raise ValueError(f"{case.name}: primary SLV stratification mismatch")
    elif "slv_spot_strata" in evidence or "slv_spot_antithetic" in evidence:
        raise ValueError(f"{case.name}: control-only evidence has primary metadata")
    for level in ("target", "fine"):
        payload = evidence.get(level, {})
        result = base.paired_result_from_serialized(
            payload,
            randomization_label=f"schema12/{case.name}/{purpose}/{level}",
        )
        controls = np.asarray(payload.get("control_batch_estimates"), dtype=float)
        if (
            result.paths_per_batch != paths_per_batch
            or result.batches_used != batches
            or controls.shape != (batches, 5)
            or not np.all(np.isfinite(controls))
        ):
            raise ValueError(f"{case.name}: invalid added {level} RQMC rows")
    if evidence["target"].get("randomization_key") == evidence["fine"].get(
        "randomization_key"
    ):
        raise ValueError(f"{case.name}: target/fine roles are not distinct")


def merge_added_reference_levels(
    target: dict,
    fine: dict,
) -> dict:
    """Join two hash-locked level checkpoints into one reference record."""
    if set(target) - {"target"} != set(fine) - {"fine"}:
        raise ValueError("target/fine checkpoint metadata keys do not match")
    target_metadata = {
        key: value for key, value in target.items() if key != "target"
    }
    fine_metadata = {key: value for key, value in fine.items() if key != "fine"}
    if (
        target_metadata != fine_metadata
        or "target" not in target
        or "fine" not in fine
    ):
        raise ValueError("target/fine checkpoint metadata do not match")
    return {**target_metadata, "target": target["target"], "fine": fine["fine"]}


def summarize_control_reference(reference: dict) -> dict:
    """Return Delta variance diagnostics without exposing an admissive verdict."""
    factor = (
        stage16().STUDY_NOTIONAL
        / stage16().DEFAULT_HEDGE_INCEPTION_SPOT
        / stage16().HEDGE_MULTIPLIER
    )
    rows = {}
    for level in ("target", "fine"):
        payload = reference[level]
        frozen = _serialized_delta_rows(payload) * factor
        heston = _serialized_delta_rows(payload, control=True) * factor
        rows[level] = (frozen, heston)

    def diagnostics(frozen: np.ndarray, heston: np.ndarray) -> dict:
        n = frozen.size
        return {
            "batches": int(n),
            "frozen_mean_contracts": float(np.mean(frozen)),
            "frozen_standard_error_contracts": float(
                np.std(frozen, ddof=1) / math.sqrt(n)
            ),
            "heston_mean_contracts": float(np.mean(heston)),
            "heston_standard_error_contracts": float(
                np.std(heston, ddof=1) / math.sqrt(n)
            ),
            "correlation": float(np.corrcoef(frozen, heston)[0, 1]),
            "unit_difference_standard_error_contracts": float(
                np.std(frozen - heston, ddof=1) / math.sqrt(n)
            ),
        }

    target_frozen, target_heston = rows["target"]
    fine_frozen, fine_heston = rows["fine"]
    return {
        "admissible": False,
        "purpose": "allocation_only_development_measurement",
        "target": diagnostics(target_frozen, target_heston),
        "fine": diagnostics(fine_frozen, fine_heston),
        "substep": diagnostics(
            target_frozen - fine_frozen,
            target_heston - fine_heston,
        ),
    }


def run_development_pilot(args: argparse.Namespace) -> int:
    base = stage16()
    implementation_hash = implementation_sha256()
    case_by_name = {
        case.name: case for case in base.certification_cases(quick=False)
    }
    requested = tuple(args.cases or CONTROL_CASES)
    unknown = sorted(set(requested) - set(CONTROL_CASES))
    if unknown:
        raise ValueError(f"unsupported development control cases: {unknown}")
    started = time.perf_counter()
    results = []
    for case_name in requested:
        print(f"[adi-greeks aggregate] development control {case_name}", flush=True)
        reference = build_control_only_reference(
            case_by_name[case_name],
            paths_per_batch=DEVELOPMENT_PATHS_PER_BATCH,
            batches=DEVELOPMENT_BATCHES,
            seed=DEVELOPMENT_SEED,
            batch_workers=DEVELOPMENT_WORKERS,
        )
        results.append(
            {
                "case": case_name,
                "reference": reference,
                "diagnostics": summarize_control_reference(reference),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "certification_mode": "development_control_pilot",
        "admissible": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": DEVELOPMENT_SEED,
        "paths_per_batch": DEVELOPMENT_PATHS_PER_BATCH,
        "batches": DEVELOPMENT_BATCHES,
        "batch_workers": DEVELOPMENT_WORKERS,
        "cases": list(requested),
        "runtime_environment": runtime_environment(),
        "implementation_sha256": implementation_hash,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    payload["evidence_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "elapsed_seconds"}
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "adi_slv_aggregate_control_pilot.json"
    stage16()._atomic_write(
        output,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    print(f"[adi-greeks aggregate] wrote {output}", flush=True)
    return 0


def load_development_pilots(paths: Sequence[Path]) -> tuple[dict, list[dict]]:
    """Load a complete, non-admissive development family for allocation only."""
    case_by_name = {
        case.name: case for case in stage16().certification_cases(quick=False)
    }
    references = {}
    manifests = []
    for path in paths:
        try:
            raw = Path(path).read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"development pilot is not readable: {path}") from exc
        expected_hash = _canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key not in {"created_at", "elapsed_seconds", "evidence_sha256"}
            }
        )
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("study") != STUDY
            or payload.get("certification_mode") != "development_control_pilot"
            or payload.get("admissible") is not False
            or payload.get("seed") != DEVELOPMENT_SEED
            or payload.get("paths_per_batch") != DEVELOPMENT_PATHS_PER_BATCH
            or payload.get("batches") != DEVELOPMENT_BATCHES
            or payload.get("batch_workers") != DEVELOPMENT_WORKERS
            or payload.get("runtime_environment") != runtime_environment()
            or payload.get("evidence_sha256") != expected_hash
        ):
            raise ValueError(f"development pilot provenance mismatch: {path}")
        implementation_hash = _require_sha256(
            payload.get("implementation_sha256"),
            f"{path}: implementation_sha256",
        )
        declared_cases = payload.get("cases", [])
        results = payload.get("results", [])
        if declared_cases != [row.get("case") for row in results]:
            raise ValueError(f"development pilot case ordering mismatch: {path}")
        for row in results:
            case_name = row.get("case")
            if case_name not in CONTROL_CASES or case_name in references:
                raise ValueError(f"duplicate or unsupported pilot case: {case_name}")
            reference = row.get("reference", {})
            validate_added_reference(
                reference,
                case=case_by_name[case_name],
                purpose="aggregate_frozen_slv_heston_control_pilot",
                seed=DEVELOPMENT_SEED,
                paths_per_batch=DEVELOPMENT_PATHS_PER_BATCH,
                batches=DEVELOPMENT_BATCHES,
                batch_workers=DEVELOPMENT_WORKERS,
                primary=False,
            )
            if row.get("diagnostics") != summarize_control_reference(reference):
                raise ValueError(f"development pilot diagnostics mismatch: {case_name}")
            references[case_name] = reference
        manifests.append(
            {
                "path": str(Path(path)),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "evidence_sha256": expected_hash,
                "implementation_sha256": implementation_hash,
                "cases": list(declared_cases),
            }
        )
    if set(references) != set(CONTROL_CASES):
        missing = sorted(set(CONTROL_CASES) - set(references))
        raise ValueError(f"development pilot family is incomplete: {missing}")
    return references, manifests


def _endpoint_rows(delta: np.ndarray, substep: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "delta_plus_substep": np.asarray(delta, dtype=float)
        + np.asarray(substep, dtype=float),
        "delta_minus_substep": np.asarray(delta, dtype=float)
        - np.asarray(substep, dtype=float),
    }


def allocation_projection_sources(
    parent: dict,
    pilot_by_case: dict,
    *,
    smooth_heston_weight: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Build pre-held-out endpoint rows used only to size the production run."""
    if smooth_heston_weight not in DEVELOPMENT_HESTON_WEIGHT_GRID:
        raise ValueError("smooth Heston weight is outside the declared grid")
    cells = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell
        for cell in parent.get("cells", [])
    }
    heston_batches = int(
        cells[("heston", PRIMARY_REFRESH_CASES[0])]["reference"]["target"][
            "batches_used"
        ]
    )
    sources = {
        name: {
            "delta": np.zeros(
                DEVELOPMENT_BATCHES
                if name == "middle_control"
                else heston_batches
                if name == "parent_heston_high"
                else AGGREGATE_OUTER_BATCHES,
                dtype=float,
            ),
            "substep": np.zeros(
                DEVELOPMENT_BATCHES
                if name == "middle_control"
                else heston_batches
                if name == "parent_heston_high"
                else AGGREGATE_OUTER_BATCHES,
                dtype=float,
            ),
        }
        for name in AGGREGATE_COHORT_NAMES
    }
    baseline_delta = np.zeros(AGGREGATE_OUTER_BATCHES, dtype=float)
    baseline_substep = np.zeros_like(baseline_delta)
    for case_name in CONTROL_CASES:
        slv_cell = cells[("heston_slv", case_name)]
        factor = _delta_contract_factor(slv_cell) / 7.0
        smooth = case_name in PRIMARY_REFRESH_CASES
        heston_weight = smooth_heston_weight if smooth else 0.0
        parent_target = _group_delta_rows(
            slv_cell["reference"]["target"],
            output_batches=AGGREGATE_OUTER_BATCHES,
        )
        parent_fine = _group_delta_rows(
            slv_cell["reference"]["fine"],
            output_batches=AGGREGATE_OUTER_BATCHES,
        )
        baseline_delta += (
            float(slv_cell["certifications"]["delta"]["pde"]) - parent_fine
        ) * factor
        baseline_substep += (parent_target - parent_fine) * factor

        parent_control_target = _group_delta_rows(
            slv_cell["reference"]["target"],
            output_batches=AGGREGATE_OUTER_BATCHES,
            control=True,
        )
        parent_control_fine = _group_delta_rows(
            slv_cell["reference"]["fine"],
            output_batches=AGGREGATE_OUTER_BATCHES,
            control=True,
        )
        primary_target = parent_target - 0.95 * parent_control_target
        primary_fine = parent_fine - 0.95 * parent_control_fine
        primary_name = "primary_refresh" if smooth else "schema11_replacements"
        sources[primary_name]["delta"] += (
            float(slv_cell["certifications"]["delta"]["pde"]) - primary_fine
        ) * factor
        sources[primary_name]["substep"] += (
            primary_target - primary_fine
        ) * factor

        pilot = pilot_by_case[case_name]
        middle_target = 0.95 * _serialized_delta_rows(pilot["target"])
        middle_fine = 0.95 * _serialized_delta_rows(pilot["fine"])
        if heston_weight:
            middle_target -= heston_weight * _serialized_delta_rows(
                pilot["target"], control=True
            )
            middle_fine -= heston_weight * _serialized_delta_rows(
                pilot["fine"], control=True
            )
            high = cells[("heston", case_name)]["reference"]
            high_target = heston_weight * _group_delta_rows(
                high["target"], output_batches=heston_batches
            )
            high_fine = heston_weight * _group_delta_rows(
                high["fine"], output_batches=heston_batches
            )
            sources["parent_heston_high"]["delta"] -= high_fine * factor
            sources["parent_heston_high"]["substep"] += (
                high_target - high_fine
            ) * factor
        sources["middle_control"]["delta"] -= middle_fine * factor
        sources["middle_control"]["substep"] += (
            middle_target - middle_fine
        ) * factor

    near_ki = cells[("heston_slv", "near_ki")]
    near_factor = _delta_contract_factor(near_ki) / 7.0
    near_target = _group_delta_rows(
        near_ki["reference"]["target"], output_batches=AGGREGATE_OUTER_BATCHES
    )
    near_fine = _group_delta_rows(
        near_ki["reference"]["fine"], output_batches=AGGREGATE_OUTER_BATCHES
    )
    near_delta = (
        float(near_ki["certifications"]["delta"]["pde"]) - near_fine
    ) * near_factor
    near_substep = (near_target - near_fine) * near_factor
    baseline_delta += near_delta
    baseline_substep += near_substep
    sources["schema11_replacements"]["delta"] += near_delta
    sources["schema11_replacements"]["substep"] += near_substep
    baseline = {
        name: float(np.mean(rows))
        for name, rows in _endpoint_rows(
            baseline_delta, baseline_substep
        ).items()
    }
    endpoints = {
        source: _endpoint_rows(values["delta"], values["substep"])
        for source, values in sources.items()
    }
    return endpoints, baseline


def project_allocation_candidate(
    source_endpoints: dict[str, dict[str, np.ndarray]],
    baseline: dict[str, float],
    *,
    primary_batches: int,
    middle_batches: int,
    smooth_heston_weight: float,
    pde_envelope: float,
) -> dict:
    """Project one fixed candidate without consuming a production seed."""
    if primary_batches not in DEVELOPMENT_PRIMARY_BATCH_GRID or (
        middle_batches not in DEVELOPMENT_MIDDLE_BATCH_GRID
    ):
        raise ValueError("allocation candidate is outside the declared grid")
    endpoint_results = {}
    for endpoint_name in ("delta_plus_substep", "delta_minus_substep"):
        standard_errors = {}
        sample_sizes = {}
        variance_multipliers = {}
        for source_name in AGGREGATE_COHORT_NAMES:
            rows = np.asarray(
                source_endpoints[source_name][endpoint_name], dtype=float
            )
            denominator = {
                "primary_refresh": primary_batches,
                "middle_control": middle_batches,
                "parent_heston_high": rows.size,
                "schema11_replacements": AGGREGATE_OUTER_BATCHES,
            }[source_name]
            sample_sizes[source_name] = int(denominator)
            multiplier = (
                math.sqrt(
                    (rows.size - 1)
                    / chi2.ppf(
                        1.0 - DEVELOPMENT_VARIANCE_UPPER_CONFIDENCE,
                        rows.size - 1,
                    )
                )
                if source_name in {"primary_refresh", "middle_control"}
                else 1.0
            )
            variance_multipliers[source_name] = float(multiplier)
            standard_errors[source_name] = float(
                np.std(rows, ddof=1) / math.sqrt(denominator) * multiplier
            )
        variances = [value * value for value in standard_errors.values()]
        total_variance = float(sum(variances))
        standard_error = math.sqrt(total_variance)
        if total_variance == 0.0:
            degrees_of_freedom = sum(
                value - 1 for value in sample_sizes.values()
            )
            half_width = 0.0
        else:
            denominator = sum(
                variance * variance / (sample_sizes[name] - 1)
                for name, variance in zip(AGGREGATE_COHORT_NAMES, variances)
            )
            degrees_of_freedom = max(
                1,
                int(math.floor(total_variance * total_variance / denominator)),
            )
            half_width = float(
                student_t.ppf(
                    0.5 + 0.5 * stage16().STOCHASTIC_COMPONENT_CONFIDENCE,
                    degrees_of_freedom,
                )
                * standard_error
            )
        new_family_se = math.sqrt(
            standard_errors["primary_refresh"] ** 2
            + standard_errors["middle_control"] ** 2
        )
        endpoint_results[endpoint_name] = {
            "baseline_center": float(baseline[endpoint_name]),
            "standard_error": standard_error,
            "half_width": half_width,
            "degrees_of_freedom": degrees_of_freedom,
            "source_standard_errors": standard_errors,
            "source_sample_sizes": sample_sizes,
            "development_variance_multipliers": variance_multipliers,
            "new_family_center_guard": (
                DEVELOPMENT_NEW_FAMILY_GUARD_SIGMAS * new_family_se
            ),
        }
    lower = min(
        row["baseline_center"] - row["half_width"]
        for row in endpoint_results.values()
    ) - pde_envelope
    upper = max(
        row["baseline_center"] + row["half_width"]
        for row in endpoint_results.values()
    ) + pde_envelope
    guarded_lower = min(
        row["baseline_center"]
        - row["half_width"]
        - row["new_family_center_guard"]
        for row in endpoint_results.values()
    ) - pde_envelope
    guarded_upper = max(
        row["baseline_center"]
        + row["half_width"]
        + row["new_family_center_guard"]
        for row in endpoint_results.values()
    ) + pde_envelope
    bound = stage16().DELTA_BIAS_BOUND_CONTRACTS
    total_unique_paths = int(
        2
        * PRODUCTION_PRIMARY_PATHS_PER_BATCH
        * primary_batches
        * len(PRIMARY_REFRESH_CASES)
        + 2
        * PRODUCTION_MIDDLE_PATHS_PER_BATCH
        * middle_batches
        * len(CONTROL_CASES)
    )
    return {
        "primary_batches": int(primary_batches),
        "middle_batches": int(middle_batches),
        "smooth_heston_weight": float(smooth_heston_weight),
        "total_unique_paths": total_unique_paths,
        "endpoint_results": endpoint_results,
        "projected_interval": [float(lower), float(upper)],
        "guarded_interval": [float(guarded_lower), float(guarded_upper)],
        "projected_pass": bool(lower >= -bound and upper <= bound),
        "guarded_pass": bool(guarded_lower >= -bound and guarded_upper <= bound),
        "worst_endpoint_standard_error": max(
            row["standard_error"] for row in endpoint_results.values()
        ),
    }


def select_production_allocation(parent: dict, pilot_by_case: dict) -> dict:
    """Apply the frozen least-path development selection rule."""
    _, pde_envelope = aggregate_pde_refinement(parent)
    sources_by_weight = {}
    baseline = None
    for weight in DEVELOPMENT_HESTON_WEIGHT_GRID:
        sources, candidate_baseline = allocation_projection_sources(
            parent,
            pilot_by_case,
            smooth_heston_weight=weight,
        )
        sources_by_weight[weight] = sources
        if baseline is None:
            baseline = candidate_baseline
        elif baseline != candidate_baseline:
            raise AssertionError("allocation baseline depends on a control weight")

    zero_sources = sources_by_weight[0.0]
    selected_weight_by_middle_batches = {}
    weight_scores_by_middle_batches = {}
    for middle_batches in DEVELOPMENT_MIDDLE_BATCH_GRID:
        scores = {}
        for weight, sources in sources_by_weight.items():
            endpoint_scores = []
            for endpoint_name in (
                "delta_plus_substep",
                "delta_minus_substep",
            ):
                middle_rows = np.asarray(
                    sources["middle_control"][endpoint_name], dtype=float
                )
                zero_rows = np.asarray(
                    zero_sources["middle_control"][endpoint_name], dtype=float
                )
                matched_high_proxy = -(middle_rows - zero_rows)
                high_batches = int(
                    np.asarray(
                        sources["parent_heston_high"][endpoint_name],
                        dtype=float,
                    ).size
                )
                variance = float(
                    np.var(middle_rows, ddof=1) / middle_batches
                    + np.var(matched_high_proxy, ddof=1) / high_batches
                )
                endpoint_scores.append(math.sqrt(max(variance, 0.0)))
            scores[weight] = max(endpoint_scores)
        selected_weight = min(scores, key=lambda weight: (scores[weight], weight))
        selected_weight_by_middle_batches[middle_batches] = selected_weight
        weight_scores_by_middle_batches[middle_batches] = scores

    candidates = []
    for middle_batches in DEVELOPMENT_MIDDLE_BATCH_GRID:
        weight = selected_weight_by_middle_batches[middle_batches]
        sources = sources_by_weight[weight]
        for primary_batches in DEVELOPMENT_PRIMARY_BATCH_GRID:
            candidate = project_allocation_candidate(
                sources,
                baseline,
                primary_batches=primary_batches,
                middle_batches=middle_batches,
                smooth_heston_weight=weight,
                pde_envelope=pde_envelope,
            )
            candidate["development_weight_score"] = (
                weight_scores_by_middle_batches[middle_batches][weight]
            )
            candidates.append(candidate)
    passing = [row for row in candidates if row["guarded_pass"]]
    recommended = (
        None
        if not passing
        else min(
            passing,
            key=lambda row: (
                row["total_unique_paths"],
                row["worst_endpoint_standard_error"],
                row["primary_batches"],
                row["middle_batches"],
            ),
        )
    )
    return {
        "selection_rule": {
            "primary_batch_grid": list(DEVELOPMENT_PRIMARY_BATCH_GRID),
            "middle_batch_grid": list(DEVELOPMENT_MIDDLE_BATCH_GRID),
            "smooth_heston_weight_grid": list(DEVELOPMENT_HESTON_WEIGHT_GRID),
            "new_family_guard_sigmas": DEVELOPMENT_NEW_FAMILY_GUARD_SIGMAS,
            "variance_upper_confidence": (
                DEVELOPMENT_VARIANCE_UPPER_CONFIDENCE
            ),
            "objective": (
                "pilot_only_weight_then_least_paths_then_worst_endpoint_se"
            ),
            "selected_weight_by_middle_batches": {
                str(key): value
                for key, value in selected_weight_by_middle_batches.items()
            },
        },
        "recommended": recommended,
        "candidates": candidates,
    }


def run_allocation_projection(args: argparse.Namespace) -> int:
    parent, _, parent_manifest = load_and_validate_parent_certificate(
        args.parent_evidence, args.parent_decision
    )
    pilot_by_case, pilot_manifests = load_development_pilots(args.pilot_evidence)
    selection = select_production_allocation(parent, pilot_by_case)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "certification_mode": "allocation_projection",
        "admissible": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_environment": runtime_environment(),
        "implementation_sha256": implementation_sha256(),
        "parent_certificate": parent_manifest,
        "development_pilots": pilot_manifests,
        **selection,
    }
    payload["projection_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "adi_slv_aggregate_allocation_projection.json"
    stage16()._atomic_write(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    recommended = payload["recommended"]
    lines = [
        "# ADI Heston-SLV Aggregate Allocation Projection",
        "",
        "This development-seed projection is allocation-only and non-admissive.",
        "",
    ]
    if recommended is None:
        lines.append("**Recommendation:** no declared candidate satisfies the guard.")
    else:
        lines.extend(
            [
                "**Recommendation:** freeze the least-path guarded candidate.",
                "",
                f"- primary batches: `{recommended['primary_batches']}`",
                f"- middle batches: `{recommended['middle_batches']}`",
                f"- smooth Heston weight: `{recommended['smooth_heston_weight']:.1f}`",
                f"- added unique paths: `{recommended['total_unique_paths']:,}`",
                "- projected interval: "
                f"`[{recommended['projected_interval'][0]:+.6f}, "
                f"{recommended['projected_interval'][1]:+.6f}]`",
                "- one-SE guarded interval: "
                f"`[{recommended['guarded_interval'][0]:+.6f}, "
                f"{recommended['guarded_interval'][1]:+.6f}]`",
            ]
        )
    lines.extend(
        [
            "",
            f"Projection SHA-256: `{payload['projection_sha256']}`",
            "",
        ]
    )
    stage16()._atomic_write(
        args.output_dir / "adi_slv_aggregate_allocation_projection.md",
        "\n".join(lines),
    )
    print(f"[adi-greeks aggregate] wrote {output}", flush=True)
    return 0


def evidence_projection(value):
    """Remove wall-clock metadata before hashing reproducible evidence."""
    if isinstance(value, dict):
        return {
            key: evidence_projection(item)
            for key, item in value.items()
            if key not in {"created_at", "elapsed_seconds", "seconds"}
        }
    if isinstance(value, list):
        return [evidence_projection(item) for item in value]
    return value


def _projected_evidence_sha256(payload: dict) -> str:
    unsigned = json.loads(json.dumps(payload))
    unsigned.pop("evidence_sha256", None)
    return _canonical_sha256(evidence_projection(unsigned))


def _added_work_counts(primary_by_case: dict, middle_by_case: dict) -> dict:
    rows = list(primary_by_case.values()) + list(middle_by_case.values())
    return {
        "new_primary_cases": len(primary_by_case),
        "new_middle_control_cases": len(middle_by_case),
        "pde_solves": 0,
        "heston_cell_reruns": 0,
        "near_ki_reruns": 0,
        "total_unique_paths": int(
            sum(
                int(row[level]["total_unique_paths"])
                for row in rows
                for level in ("target", "fine")
            )
        ),
        "total_path_valuations": int(
            sum(
                int(row[level]["total_path_valuations"])
                for row in rows
                for level in ("target", "fine")
            )
        ),
    }


def run_production_amendment(args: argparse.Namespace) -> int:
    if not PRODUCTION_ALLOCATION_FROZEN:
        raise ValueError("schema-12 production allocation is not frozen")
    parent, parent_decision, manifest = load_and_validate_parent_certificate(
        args.parent_evidence,
        args.parent_decision,
    )
    implementation_hash = implementation_sha256()
    runtime = runtime_environment()
    run_configuration = production_run_configuration(
        implementation_hash=implementation_hash,
        runtime=runtime,
        adaptive=bool(getattr(args, "adaptive", False)),
        precision_target=float(
            getattr(args, "precision_target", DEFAULT_PRECISION_TARGET_CONTRACTS)
        ),
        budget_hours=float(
            getattr(args, "budget_hours", DEFAULT_ADAPTIVE_BUDGET_HOURS)
        ),
        pilot_batches=int(getattr(args, "pilot_batches", MIN_ADAPTIVE_BATCHES)),
    )
    run_configuration_hash = _canonical_sha256(run_configuration)
    case_by_name = {
        case.name: case for case in stage16().certification_cases(quick=False)
    }
    started = time.perf_counter()
    print(
        "[adi-greeks aggregate] schema-12 aggregate-only amendment: "
        "PDE=0, Heston cells=0, Near-KI=0 reruns",
        flush=True,
    )
    print(
        f"[adi-greeks aggregate] implementation={implementation_hash[:16]} "
        f"configuration={run_configuration_hash[:16]} "
        f"parent={manifest['evidence_sha256'][:16]}",
        flush=True,
    )

    def checkpointed(name: str, kind: str, compute):
        evidence = None
        if args.resume:
            evidence = _load_checkpoint(
                args.output_dir,
                name,
                run_configuration_sha256=run_configuration_hash,
                kind=kind,
            )
            if evidence is not None:
                print(f"[adi-greeks aggregate] resumed {name}", flush=True)
        if evidence is None:
            evidence = compute()
            _write_checkpoint(
                args.output_dir,
                name,
                run_configuration_sha256=run_configuration_hash,
                kind=kind,
                evidence=evidence,
            )
            print(f"[adi-greeks aggregate] checkpointed {name}", flush=True)
        return evidence

    def primary(case_name: str) -> dict:
        parts = {}
        for level in ("target", "fine"):
            print(
                f"[adi-greeks aggregate] primary refresh {case_name}/{level}",
                flush=True,
            )
            parts[level] = checkpointed(
                f"primary__{case_name}__{level}",
                "primary_reference_level",
                lambda level=level: build_primary_reference(
                    case_by_name[case_name],
                    paths_per_batch=PRODUCTION_PRIMARY_PATHS_PER_BATCH,
                    batches=PRODUCTION_PRIMARY_BATCHES,
                    seed=PRODUCTION_PRIMARY_SEED,
                    batch_workers=PRODUCTION_PRIMARY_BATCH_WORKERS,
                    levels=(level,),
                ),
            )
        return merge_added_reference_levels(parts["target"], parts["fine"])

    def middle(case_name: str) -> dict:
        parts = {}
        for level in ("target", "fine"):
            print(
                f"[adi-greeks aggregate] middle control {case_name}/{level}",
                flush=True,
            )
            parts[level] = checkpointed(
                f"middle__{case_name}__{level}",
                "middle_control_level",
                lambda level=level: build_control_only_reference(
                    case_by_name[case_name],
                    paths_per_batch=PRODUCTION_MIDDLE_PATHS_PER_BATCH,
                    batches=PRODUCTION_MIDDLE_BATCHES,
                    seed=PRODUCTION_MIDDLE_SEED,
                    batch_workers=PRODUCTION_MIDDLE_BATCH_WORKERS,
                    levels=(level,),
                    purpose=(
                        "aggregate_only_frozen_slv_heston_middle_control"
                    ),
                ),
            )
        return merge_added_reference_levels(parts["target"], parts["fine"])

    def run_parallel(
        case_names: Sequence[str],
        *,
        workers: int,
        prefix: str,
        compute,
    ) -> dict:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"adi-aggregate-{prefix}",
        ) as pool:
            values = list(pool.map(compute, case_names))
        return dict(zip(case_names, values))

    primary_by_case = run_parallel(
        PRIMARY_REFRESH_CASES,
        workers=PRODUCTION_PRIMARY_CELL_WORKERS,
        prefix="primary",
        compute=primary,
    )
    smooth_middle_cases = tuple(
        case_name for case_name in CONTROL_CASES if case_name != "low_feller"
    )
    middle_by_case = run_parallel(
        smooth_middle_cases,
        workers=PRODUCTION_MIDDLE_CELL_WORKERS,
        prefix="middle",
        compute=middle,
    )
    # The 21,168-dimensional Low-Feller 7->14 pair owns the memory envelope.
    # It runs alone, while all other cases can safely use the declared workers.
    middle_by_case["low_feller"] = middle("low_feller")
    middle_by_case = {
        case_name: middle_by_case[case_name] for case_name in CONTROL_CASES
    }

    delta_cohorts, substep_cohorts, component_hashes = (
        aggregate_reference_cohorts(
            parent,
            primary_by_case=primary_by_case,
            middle_by_case=middle_by_case,
        )
    )
    decisions = make_aggregate_decisions(
        parent,
        delta_cohorts,
        substep_cohorts,
    )
    pde_axes, pde_envelope = aggregate_pde_refinement(parent)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "certification_mode": CERTIFICATION_MODE,
        "profile": "production aggregate-only amendment",
        "created_at": datetime.now().astimezone().isoformat(),
        "quick": False,
        "parent_certificate": manifest,
        "production_pde_compatibility_sha256": (
            stage16().production_pde_compatibility_sha256()
        ),
        "runtime_environment": runtime,
        "implementation_sha256": implementation_hash,
        "run_configuration_sha256": run_configuration_hash,
        "run_configuration": run_configuration,
        "elapsed_seconds": time.perf_counter() - started,
        "production_engine_controls": stage16().PRODUCTION_ENGINE_CONTROLS,
        "reference_seeds": {
            "schema11_parent": parent["reference_seeds"],
            "aggregate_primary_refresh": PRODUCTION_PRIMARY_SEED,
            "aggregate_middle_control": PRODUCTION_MIDDLE_SEED,
        },
        "sampling_by_variant": parent["sampling_by_variant"],
        "aggregate_sampling": {
            "schema11_replacement_batches": AGGREGATE_OUTER_BATCHES,
            "primary_refresh": run_configuration["primary_refresh"],
            "middle_control": run_configuration["middle_control"],
        },
        "policy": {
            "economic_bound_contracts": stage16().DELTA_BIAS_BOUND_CONTRACTS,
            "component_confidence": stage16().STOCHASTIC_COMPONENT_CONFIDENCE,
            "endpoint_method": run_configuration["endpoint_method"],
            "weights_by_case": AGGREGATE_CONTROL_WEIGHTS,
            "no_optional_stopping": True,
            "individual_cell_authority": "schema11_parent",
            "production_engine_controls": stage16().PRODUCTION_ENGINE_CONTROLS,
        },
        "anchors": parent["anchors"],
        "cells": parent["cells"],
        # The full-recertification parent states per-cell reuse identities
        # rather than an amendment's cell_provenance chain; they are carried
        # under their own name and pinned by validate_payload.
        "cell_identities": parent["cell_identities"],
        "aggregate_reference": {
            "primary_by_case": primary_by_case,
            "middle_by_case": middle_by_case,
            "component_hashes": component_hashes,
            "weights_by_case": AGGREGATE_CONTROL_WEIGHTS,
            "near_ki_source": "schema11_published_multilevel_reference",
            "heston_high_source": "schema11_parent_cells_no_rerun",
        },
        "aggregate_cohorts": {
            "order": list(AGGREGATE_COHORT_NAMES),
            "delta_contracts": {
                name: delta_cohorts[name].tolist()
                for name in AGGREGATE_COHORT_NAMES
            },
            "substep_contracts": {
                name: substep_cohorts[name].tolist()
                for name in AGGREGATE_COHORT_NAMES
            },
        },
        "aggregate_pde_signed_refinement_contracts": pde_axes,
        "aggregate_pde_discretization_envelope": pde_envelope,
        "added_work": _added_work_counts(primary_by_case, middle_by_case),
        "decisions": decisions,
        "parent_decision_sha256": parent_decision["decision_sha256"],
    }
    publish_payload(payload, args.output_dir)
    print(f"[adi-greeks aggregate] wrote {args.output_dir}", flush=True)
    return 0


def validate_payload(payload: dict) -> None:
    """Recompute every schema-12 aggregate statistic and fail closed on drift."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("aggregate certification schema version mismatch")
    if payload.get("study") != STUDY:
        raise ValueError("aggregate certification study tag mismatch")
    if payload.get("certification_mode") != CERTIFICATION_MODE:
        raise ValueError("only the frozen aggregate-only amendment is admissible")
    if not PRODUCTION_ALLOCATION_FROZEN:
        raise ValueError("schema-12 production allocation is not frozen yet")
    if payload.get("quick") is not False or payload.get("profile") != (
        "production aggregate-only amendment"
    ):
        raise ValueError("schema-12 aggregate amendment must be production evidence")
    if payload.get("parent_certificate") != parent_certificate_manifest():
        raise ValueError("schema-12 parent certificate manifest mismatch")
    if payload.get("parent_decision_sha256") != PARENT_DECISION_SHA256:
        raise ValueError("schema-12 parent decision link mismatch")
    if payload.get("production_pde_compatibility_sha256") != (
        PARENT_PRODUCTION_PDE_SHA256
    ) or stage16().production_pde_compatibility_sha256() != (
        PARENT_PRODUCTION_PDE_SHA256
    ):
        raise ValueError("schema-12 production PDE compatibility mismatch")
    runtime = runtime_environment()
    implementation_hash = implementation_sha256()
    if payload.get("runtime_environment") != runtime:
        raise ValueError("schema-12 numerical runtime mismatch")
    if payload.get("implementation_sha256") != implementation_hash:
        raise ValueError("schema-12 evidence does not match the live implementation")
    expected_run_configuration = production_run_configuration(
        implementation_hash=implementation_hash,
        runtime=runtime,
    )
    if payload.get("run_configuration") != expected_run_configuration or (
        payload.get("run_configuration_sha256")
        != _canonical_sha256(expected_run_configuration)
    ):
        raise ValueError("schema-12 run configuration mismatch")
    if payload.get("production_engine_controls") != (
        stage16().PRODUCTION_ENGINE_CONTROLS
    ) or payload.get("policy", {}).get("production_engine_controls") != (
        stage16().PRODUCTION_ENGINE_CONTROLS
    ):
        raise ValueError("schema-12 production engine controls mismatch")
    expected_reference_seeds = {
        # The stage-16 strided re-publish parent records exactly these three
        # seed families; the lost amendment parent's extra families
        # (parent_schema9, heston_near_ki_control) do not exist in a
        # full-recertification payload.
        "schema11_parent": {
            "heston": 20260808,
            "heston_slv_mid_control": 20260810,
            "heston_slv_primary": 20260809,
        },
        "aggregate_primary_refresh": PRODUCTION_PRIMARY_SEED,
        "aggregate_middle_control": PRODUCTION_MIDDLE_SEED,
    }
    if payload.get("reference_seeds") != expected_reference_seeds:
        raise ValueError("schema-12 reference seed families mismatch")
    if len({20260806, *range(20260807, 20260813)}) != 7:
        raise AssertionError("declared development/production seed families overlap")

    projections = {
        "anchors": PARENT_ANCHORS_SHA256,
        "cells": PARENT_CELLS_SHA256,
        "cell_identities": PARENT_CELL_IDENTITIES_SHA256,
    }
    for key, expected_hash in projections.items():
        if _canonical_sha256(payload.get(key)) != expected_hash:
            raise ValueError(f"schema-12 carried parent {key} mismatch")
    if _canonical_sha256(payload.get("sampling_by_variant")) != (
        PARENT_SAMPLING_BY_VARIANT_SHA256
    ):
        raise ValueError("schema-12 carried parent sampling profile mismatch")
    expected_aggregate_sampling = {
        "schema11_replacement_batches": AGGREGATE_OUTER_BATCHES,
        "primary_refresh": expected_run_configuration["primary_refresh"],
        "middle_control": expected_run_configuration["middle_control"],
    }
    if payload.get("aggregate_sampling") != expected_aggregate_sampling:
        raise ValueError("schema-12 aggregate sampling profile mismatch")
    expected_policy = {
        "economic_bound_contracts": stage16().DELTA_BIAS_BOUND_CONTRACTS,
        "component_confidence": stage16().STOCHASTIC_COMPONENT_CONFIDENCE,
        "endpoint_method": expected_run_configuration["endpoint_method"],
        "weights_by_case": AGGREGATE_CONTROL_WEIGHTS,
        "no_optional_stopping": True,
        "individual_cell_authority": "schema11_parent",
        "production_engine_controls": stage16().PRODUCTION_ENGINE_CONTROLS,
    }
    if payload.get("policy") != expected_policy:
        raise ValueError("schema-12 aggregate policy mismatch")
    cells = payload.get("cells", [])
    if len(cells) != 14 or any(cell.get("status") != "PASS" for cell in cells):
        raise ValueError("schema-12 carried cells are not fourteen PASS rows")

    aggregate_reference = payload.get("aggregate_reference", {})
    primary_by_case = aggregate_reference.get("primary_by_case", {})
    middle_by_case = aggregate_reference.get("middle_by_case", {})
    if set(primary_by_case) != set(PRIMARY_REFRESH_CASES):
        raise ValueError("schema-12 primary refresh membership mismatch")
    if set(middle_by_case) != set(CONTROL_CASES):
        raise ValueError("schema-12 middle control membership mismatch")
    case_by_name = {
        case.name: case for case in stage16().certification_cases(quick=False)
    }
    for case_name in PRIMARY_REFRESH_CASES:
        validate_added_reference(
            primary_by_case[case_name],
            case=case_by_name[case_name],
            purpose="aggregate_only_primary_reference_refresh",
            seed=PRODUCTION_PRIMARY_SEED,
            paths_per_batch=PRODUCTION_PRIMARY_PATHS_PER_BATCH,
            batches=PRODUCTION_PRIMARY_BATCHES,
            batch_workers=PRODUCTION_PRIMARY_BATCH_WORKERS,
            primary=True,
        )
    for case_name in CONTROL_CASES:
        validate_added_reference(
            middle_by_case[case_name],
            case=case_by_name[case_name],
            purpose="aggregate_only_frozen_slv_heston_middle_control",
            seed=PRODUCTION_MIDDLE_SEED,
            paths_per_batch=PRODUCTION_MIDDLE_PATHS_PER_BATCH,
            batches=PRODUCTION_MIDDLE_BATCHES,
            batch_workers=PRODUCTION_MIDDLE_BATCH_WORKERS,
            primary=False,
        )
    if (
        aggregate_reference.get("weights_by_case") != AGGREGATE_CONTROL_WEIGHTS
        or aggregate_reference.get("near_ki_source")
        != "schema11_published_multilevel_reference"
        or aggregate_reference.get("heston_high_source")
        != "schema11_parent_cells_no_rerun"
    ):
        raise ValueError("schema-12 aggregate estimator policy mismatch")

    delta_cohorts, substep_cohorts, component_hashes = (
        aggregate_reference_cohorts(
            {"cells": cells},
            primary_by_case=primary_by_case,
            middle_by_case=middle_by_case,
        )
    )
    if aggregate_reference.get("component_hashes") != component_hashes:
        raise ValueError("schema-12 aggregate component hashes mismatch")
    serialized_cohorts = payload.get("aggregate_cohorts", {})
    if serialized_cohorts.get("order") != list(AGGREGATE_COHORT_NAMES):
        raise ValueError("schema-12 independent cohort ordering mismatch")
    for field, expected in (
        ("delta_contracts", delta_cohorts),
        ("substep_contracts", substep_cohorts),
    ):
        observed = serialized_cohorts.get(field, {})
        if set(observed) != set(AGGREGATE_COHORT_NAMES) or any(
            not np.array_equal(
                np.asarray(observed[name], dtype=float), expected[name]
            )
            for name in AGGREGATE_COHORT_NAMES
        ):
            raise ValueError(f"schema-12 {field} independent cohorts do not recompose")
    pde_axes, pde_envelope = aggregate_pde_refinement({"cells": cells})
    if payload.get("aggregate_pde_signed_refinement_contracts") != pde_axes or (
        payload.get("aggregate_pde_discretization_envelope") != pde_envelope
    ):
        raise ValueError("schema-12 aggregate PDE envelope mismatch")
    if payload.get("added_work") != _added_work_counts(
        primary_by_case, middle_by_case
    ):
        raise ValueError("schema-12 added work accounting mismatch")
    if payload["added_work"] != {
        **payload["added_work"],
        "pde_solves": 0,
        "heston_cell_reruns": 0,
        "near_ki_reruns": 0,
    }:
        raise ValueError("schema-12 illegally reran carried numerical evidence")
    # make_aggregate_decisions needs the immutable parent Heston row, not the
    # schema-12 row whose provenance labels are already updated.
    expected_heston_parent = json.loads(json.dumps(payload["decisions"]["heston"]))
    expected_heston_parent.pop("certification_source", None)
    expected_heston_parent.pop("parent_evidence_sha256", None)
    if _canonical_sha256(expected_heston_parent) != (
        PARENT_BASE_HESTON_DECISION_SHA256
    ):
        raise ValueError("schema-12 carried Heston decision mismatch")
    expected_parent_view = {
        "cells": cells,
        "decisions": {"heston": expected_heston_parent},
    }
    expected_decisions = make_aggregate_decisions(
        expected_parent_view,
        delta_cohorts,
        substep_cohorts,
    )
    if payload.get("decisions") != expected_decisions:
        raise ValueError("schema-12 decisions do not match aggregate raw evidence")
    for variant, decision in expected_decisions.items():
        if decision.get("route") not in {"pde", "excluded_greek_unresolved"}:
            raise ValueError(f"{variant}: invalid schema-12 route")
        if decision.get("route") == "pde" and decision.get("evidence_complete") is not True:
            raise ValueError(f"{variant}: incomplete schema-12 evidence cannot admit PDE")


def build_decision_payload(payload: dict) -> dict:
    validate_payload(payload)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "certification_mode": CERTIFICATION_MODE,
        "profile": payload["profile"],
        "quick": False,
        "evidence_sha256": _require_sha256(
            payload.get("evidence_sha256"), "evidence_sha256"
        ),
        "implementation_sha256": payload["implementation_sha256"],
        "run_configuration_sha256": payload["run_configuration_sha256"],
        "run_configuration": payload["run_configuration"],
        "runtime_environment": payload["runtime_environment"],
        "production_engine_controls": payload["production_engine_controls"],
        "parent_certificate": payload["parent_certificate"],
        "aggregate_reference_sha256": _canonical_sha256(
            payload["aggregate_reference"]
        ),
        "aggregate_cohorts_sha256": _canonical_sha256(
            payload["aggregate_cohorts"]
        ),
        "added_work": payload["added_work"],
        "decisions": payload["decisions"],
    }
    decision["decision_sha256"] = _canonical_sha256(decision)
    return decision


def render_markdown(payload: dict) -> str:
    slv = payload["decisions"]["heston_slv"]
    bias = slv["delta_bias"]
    interval = bias["interval"]
    plus = bias["endpoints"]["delta_plus_substep"]
    minus = bias["endpoints"]["delta_minus_substep"]
    work = payload["added_work"]
    lines = [
        "# ADI 2-D Greek Certification — Schema 12 Aggregate Amendment",
        "",
        f"**Heston route:** `{payload['decisions']['heston']['route']}` (carried)",
        f"**Heston-SLV route:** `{slv['route']}`",
        f"**Aggregate verdict:** `{bias['status']}`",
        "",
        "## Aggregate signed Delta bias",
        "",
        f"- simultaneous interval: `[{interval[0]:+.6f}, {interval[1]:+.6f}]` contracts",
        f"- economic bound: `±{bias['economic_bound']:.6f}` contracts",
        f"- D + S: `{plus['estimate']:+.6f} ± {plus['half_width']:.6f}`",
        f"- D - S: `{minus['estimate']:+.6f} ± {minus['half_width']:.6f}`",
        f"- PDE envelope: `{bias['pde_discretization_envelope']:.6f}` contracts",
        "- independent source cohort sizes: `"
        + ", ".join(
            f"{name}={size}"
            for name, size in bias["aggregate_cohort_sizes"].items()
        )
        + "`",
        "",
        "The two endpoint intervals retain the paired fine/substep covariance "
        "inside each seed family and add independent-family variances with a "
        "floored Welch degree of freedom. Each uses 97.5% Student-t coverage, "
        "giving at least 95% simultaneous coverage before the deterministic "
        "PDE envelope is added.",
        "",
        "## Scope and work",
        "",
        "- Parent schema-11 anchors and all 14 individual PASS cells are carried byte-for-byte.",
        "- No PDE solve, Heston cell, Near-KI cell, or Low-Feller cell is rerun.",
        f"- New primary reference cases: `{work['new_primary_cases']}`.",
        f"- New control-only cases: `{work['new_middle_control_cases']}`.",
        f"- Added unique paths: `{work['total_unique_paths']:,}`.",
        f"- Added conditional path valuations: `{work['total_path_valuations']:,}`.",
        "",
        "## Provenance",
        "",
        f"- Parent evidence SHA-256: `{PARENT_EVIDENCE_SHA256}`",
        f"- Evidence SHA-256: `{payload.get('evidence_sha256', 'unpublished')}`",
        f"- Implementation SHA-256: `{payload['implementation_sha256']}`",
        f"- Run-configuration SHA-256: `{payload['run_configuration_sha256']}`",
        f"- Elapsed seconds: `{float(payload.get('elapsed_seconds', 0.0)):.3f}`",
        "",
    ]
    return "\n".join(lines)


def publish_payload(payload: dict, output_dir: Path) -> None:
    payload.pop("evidence_sha256", None)
    validate_payload(payload)
    payload["evidence_sha256"] = _projected_evidence_sha256(payload)
    validate_payload(payload)
    decision = build_decision_payload(payload)
    stage16()._atomic_write(
        output_dir / "adi_greek_certification.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    stage16()._atomic_write(
        output_dir / "adi_greek_certification.md",
        render_markdown(payload),
    )
    stage16()._atomic_write(
        output_dir / "adi_greek_certification_decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-pilot", action="store_true")
    parser.add_argument("--project-allocation", action="store_true")
    parser.add_argument(
        "--pilot-evidence",
        nargs="+",
        type=Path,
        help="complete development pilot JSON family for --project-allocation",
    )
    parser.add_argument("--cases", nargs="+", choices=CONTROL_CASES)
    parser.add_argument(
        "--parent-evidence",
        type=Path,
        help="exact schema-11 evidence JSON carried by the aggregate amendment",
    )
    parser.add_argument(
        "--parent-decision",
        type=Path,
        help="exact schema-11 decision JSON paired with --parent-evidence",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only hash-matched schema-12 per-case checkpoints",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "size each cell from a pilot-frozen Neyman allocation instead of "
            "the frozen 4096/256 counts; the frozen counts stay the floor"
        ),
    )
    parser.add_argument(
        "--precision-target",
        type=float,
        default=DEFAULT_PRECISION_TARGET_CONTRACTS,
        help="aggregate statistical half-width target in contracts",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=DEFAULT_ADAPTIVE_BUDGET_HOURS,
        help="wall-clock budget the allocation must fit inside",
    )
    parser.add_argument(
        "--pilot-batches",
        type=int,
        default=MIN_ADAPTIVE_BATCHES,
        help="batches per cell in the allocation-sizing pilot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/adi_slv_aggregate_certification"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.development_pilot:
        if args.project_allocation or args.pilot_evidence:
            raise ValueError("development pilot and allocation projection are exclusive")
        if args.parent_evidence is not None or args.parent_decision is not None:
            raise ValueError("development pilot does not consume a parent certificate")
        if args.resume:
            raise ValueError("development pilot is intentionally not resumable")
        return run_development_pilot(args)
    if args.project_allocation:
        if args.cases or args.resume:
            raise ValueError("allocation projection does not run cases or resume")
        if not args.pilot_evidence:
            raise ValueError("allocation projection requires --pilot-evidence")
        if args.parent_evidence is None or args.parent_decision is None:
            raise ValueError("allocation projection requires the exact parent certificate")
        return run_allocation_projection(args)
    if args.pilot_evidence:
        raise ValueError("--pilot-evidence is allocation-projection only")
    if args.cases:
        raise ValueError("case subsets are development-pilot only")
    if args.parent_evidence is None or args.parent_decision is None:
        raise ValueError(
            "schema-12 production requires both --parent-evidence and "
            "--parent-decision"
        )
    return run_production_amendment(args)


if __name__ == "__main__":
    raise SystemExit(main())
