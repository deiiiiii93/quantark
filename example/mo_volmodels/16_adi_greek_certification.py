"""Stage 16 - certify 2-D ADI Snowball delta/gamma with independent evidence.

This is a self-contained synthetic certification harness.  It deliberately
does not consume the calibration-history working tree: deterministic reduction
anchors, a regime matrix, separate ``n_x``/``n_v``/``n_t`` ladders, a spot-bump
ladder, and paired QE-M RQMC references can therefore be reproduced from a
clean checkout.

The reference forms each delta/gamma estimate *inside* an outer scrambled-
Sobol batch at ``S-h``, ``S`` and ``S+h``.  Student-t uncertainty, the QE-M
substep envelope, and the deterministic PDE refinement envelope all enter a
tri-state PASS/FAIL/INCONCLUSIVE equivalence verdict.  An unresolved Greek is
never converted into an MC route: the decision is
``excluded_greek_unresolved`` until a production-sized run proves equivalence.

Production incremental amendment::

    .venv/bin/python example/mo_volmodels/16_adi_greek_certification.py \
      --amend-parent-evidence output/adi_greek_certification_schema9/adi_greek_certification.json \
      --amend-parent-decision output/adi_greek_certification_schema9/adi_greek_certification_decision.json

A fresh 14-cell production recertification requires the explicit
``--full-recertification`` flag.

Fast plumbing run (explicitly non-admissive)::

    .venv/bin/python example/mo_volmodels/16_adi_greek_certification.py --quick
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy
from scipy.stats import t as student_t

from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.equity.engine.mc import (
    HestonSLVQESnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde import (
    HestonPDESolver,
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.param import BumpConfig, MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.montecarlo import (
    concatenate_paired_results,
    CoupledQESubstepDrawProvider,
    PairedRQMCGreeksResult,
    run_paired_rqmc_greeks,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.validation import (
    EconomicGreekScale,
    EquivalenceStatus,
    SequentialAdmissionPolicy,
    SequentialAdmissionStatus,
    sequential_admission,
    certify_equivalence,
    certify_signed_bias_from_batches,
    certify_signed_bias_from_independent_cohorts,
    summarize_independent_cohort_means,
)

# Imported from the module rather than re-exported through
# `quantark/validation/__init__.py` on purpose: that __init__ is inside
# IMPLEMENTATION_INPUTS with no exemptions, so widening its `__all__` would move
# the numerical digest and invalidate banked cells for a change that cannot
# affect a number.
from quantark.validation.cell_identity import (
    cell_identity_sha256,
    source_projection_sha256,
)
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface


SCHEMA_VERSION = 13
CERTIFICATION_MODE_FULL = "full_recertification"
CERTIFICATION_MODE_AMENDMENT = "incremental_amendment"

# Schema 11 is a cryptographically bound amendment to the completed schema-9
# certificate.  The parent already proved every deterministic anchor and all
# seven Heston cells; rerunning those cells would add compute cost without new
# information.  Only the two unresolved SLV cells are replaced.  Near-KI also
# needs one reference-only Heston 8->16 control because its new three-level
# estimator cannot consume the parent's 4->8 Heston cell.
PARENT_SCHEMA_VERSION = 9
PARENT_SOURCE_COMMIT = "426c8dc4864f9684a73e3602b62f99e9a5df1f5a"
PARENT_SEED = 20260807
PARENT_EVIDENCE_FILE_SHA256 = (
    "a0ed3bbdaaaf8e6c60d4fd97b3e17d39b0434107ddd365ea67b1680bcf319457"
)
PARENT_DECISION_FILE_SHA256 = (
    "3fca3ae1af40a733487eac2533408dd1bd5749e1bee1a080680464c470f0cdd1"
)
PARENT_EVIDENCE_SHA256 = (
    "3daf611f2a0b0d16fce94f66c80acf99beec5fe5ad092fb69b211a52537847fa"
)
PARENT_DECISION_SHA256 = (
    "fa23738f02b1ddc1bb83870ac23c4b9f0d4bda22e91911188e0e3af13eb5a3a9"
)
PARENT_IMPLEMENTATION_SHA256 = (
    "240aa9f0824a7b101b0db50f9aa9166f86112cc4ccb3294818e7e8de557cba88"
)
PARENT_RUN_CONFIGURATION_SHA256 = (
    "c6c79eb88a5eb34a1daeec83f3ef6ef159b2406d877198c22f1ed9a0610a06b4"
)
PARENT_PRODUCTION_PDE_SHA256 = (
    "328f9c2daa8e84d055e13ead53a99fc72690030732602c736bf19529a9526b62"
)
PARENT_ANCHORS_SHA256 = (
    "4662f6813dc4fc97318b5bd7b92da6f9310ceb91aa5e3b5d4ee2104c4ed604b7"
)
PARENT_HESTON_DECISION_SHA256 = (
    "c0287e36c336906746bf5756e41b1afc365f363193342aff05174b588b9696ba"
)
PARENT_CARRIED_CELL_SHA256 = {
    "heston/ordinary_full": "d7f7fb42d86296f5c89ef3e5a6525b61af9ec416cff65c4ccf63743281c4506f",
    "heston/ordinary_decayed": "0876d3225dd77d6ac08598eda32d8dd0dc75aebaf4a7235cd8504b5643fb7016",
    "heston/near_ko": "87968cceaf13016b23caf077d7ce6017a505281c6fcfd16987a65e0d86a8f7e5",
    "heston/near_ki": "51001163bf96a7b861839ca328f9f516b15e7f601ceb3452cfcac3ceff33ce70",
    "heston/low_feller": "bb120c7745fdaf12c5a75f4c4725bed2966b1c3d2509f0ad139e6587ecb30a3f",
    "heston/sigma_collapse": "e702c7641322307b699423b13ef082c6234a48eab68a1f71ca4f309ccd0c0896",
    "heston/near_expiry": "cba106f047f46390b62ae22dbf0410ba0bddb357d64cf44f5b7630c990649bfb",
    "heston_slv/ordinary_full": "4d6f72141fb85894931ed12fb7e2868db2062a691a4b61e23104cb329e18cc23",
    "heston_slv/ordinary_decayed": "7d7c09af410ae01b55d773ab0c3f583faceddcb3b13aefa06b8524990320fcb0",
    "heston_slv/near_ko": "7d6ff806f23fbe1aa6742d385c2af932601e2fbe65803ae3b4cd75079bcb06b3",
    "heston_slv/sigma_collapse": "a7c670f4268bc065482dbc9851670ec6d278c22312fa5091ac23ed515fe0a744",
    "heston_slv/near_expiry": "e46194c1f2146c9e687132f9ca1d6e8107d2744cdd74b980de0cd903961a8681",
}
AMENDMENT_REPLACEMENT_CASES = frozenset({"near_ki", "low_feller"})
AMENDMENT_CARRIED_SLV_CASES = frozenset(
    {
        "ordinary_full",
        "ordinary_decayed",
        "near_ko",
        "sigma_collapse",
        "near_expiry",
    }
)
AMENDMENT_AGGREGATE_BATCHES = 128
# Schema 9's untouched 20260807 family certified Heston and isolated the two
# remaining SLV cells. The 20260806 development family then selected the
# multilevel estimator, bridge allocation, and QE-M refinement levels. Schema
# 10 keeps all three estimator levels on mutually independent held-out scramble
# families so that neither tuning nor cross-level covariance can narrow the
# reported intervals accidentally.
HESTON_REFERENCE_SEED = 20260808
SLV_PRIMARY_SEED = 20260809
SLV_MID_CONTROL_SEED = 20260810
# Compatibility alias for developer scripts that ask for the primary Heston
# scramble family explicitly. Production metadata pins the full seed map.
SEED = HESTON_REFERENCE_SEED
CONFIDENCE = 0.95
# Two stochastic components enter each verdict: the fine QE-M confidence
# interval and the target-to-fine substep-bias upper bound. Bonferroni at
# 97.5% per component preserves at least 95% simultaneous coverage.
STOCHASTIC_COMPONENT_CONFIDENCE = 1.0 - (1.0 - CONFIDENCE) / 2.0
SPOT_BUMP = 0.01
FULL_BUMP_LADDER = (0.02, 0.01, 0.005, 0.0025)
QUICK_BUMP_LADDER = (0.02, 0.01, 0.005)
STUDY_NOTIONAL = 50_000_000.0
HEDGE_MULTIPLIER = 200.0
# Strictest inception-level quantum in the frozen 27-inception MO cohort
# (§5.3: observed range 4,532.52 to 6,733.97).  The numerical cases are
# homogeneously normalized around 100; this actual index level must therefore
# remain separate from CaseSpec.spot.
DEFAULT_HEDGE_INCEPTION_SPOT = 4_532.52
DELTA_CELL_BOUND_CONTRACTS = 0.5
DELTA_BIAS_BOUND_CONTRACTS = 0.1
GAMMA_CELL_BOUND_CONTRACTS = 0.5
# Long-maturity equal-cost pilots allocate the SLV conditional samples across
# the terminal and midpoint Brownian-bridge factors. Four terminal strata plus
# their antithetic shifts, crossed with eight midpoint strata, outperformed 64
# terminal-only samples for fine delta/gamma and the paired substep envelope.
SLV_SPOT_STRATA = 4
SLV_SPOT_ANTITHETIC = True
SLV_SPOT_BRIDGE_STRATA = 8
# Heston already integrates the terminal Brownian-bridge factor exactly. The
# near-KI cell additionally uses four inner points across eight leading
# residual bridge coordinates. Two design seeds selected eight dimensions over
# four and sixteen at identical runtime; smooth cells retain the exact
# terminal-factor estimator without multiplying their long path matrices.
HESTON_SPOT_BRIDGE_STRATA = 4
HESTON_SPOT_BRIDGE_DIMENSIONS = 8
HESTON_SPOT_BRIDGE_PROFILE_BY_CASE = {
    "ordinary_full": {"strata": 1, "dimensions": 1},
    "ordinary_decayed": {"strata": 1, "dimensions": 1},
    "near_ko": {"strata": 1, "dimensions": 1},
    "near_ki": {
        "strata": HESTON_SPOT_BRIDGE_STRATA,
        "dimensions": HESTON_SPOT_BRIDGE_DIMENSIONS,
    },
    "low_feller": {"strata": 1, "dimensions": 1},
    "sigma_collapse": {"strata": 1, "dimensions": 1},
    "near_expiry": {"strata": 1, "dimensions": 1},
}
PRODUCTION_HESTON_PATHS_PER_BATCH = 8192
PRODUCTION_HESTON_BATCHES = 1024
PRODUCTION_HESTON_BATCHES_BY_CASE = {
    "ordinary_full": PRODUCTION_HESTON_BATCHES,
    "ordinary_decayed": PRODUCTION_HESTON_BATCHES,
    "near_ko": PRODUCTION_HESTON_BATCHES,
    # Near KI is a discontinuity-dominated gamma regime.  The common-scramble
    # held-out pilot passed every deterministic refinement gate but left the
    # paired target-to-fine QE-M bias interval inconclusive.  Keep the first
    # 1,024 scramble ids common with every other regime for the signed-bias gate,
    # then extend this cell alone until its reference interval is decisive.
    "near_ki": 2048,
    "low_feller": PRODUCTION_HESTON_BATCHES,
    "sigma_collapse": PRODUCTION_HESTON_BATCHES,
    "near_expiry": PRODUCTION_HESTON_BATCHES,
}
PRODUCTION_SLV_PATHS_PER_BATCH = 1024
PRODUCTION_SLV_BATCHES = 128
PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE = {
    "ordinary_full": PRODUCTION_SLV_BATCHES,
    "ordinary_decayed": PRODUCTION_SLV_BATCHES,
    "near_ko": PRODUCTION_SLV_BATCHES,
    "near_ki": 256,
    # Independent 20260805/20260806 development measurements selected the
    # direct estimator. At the final 7->14 ladder, 512 batches project margins
    # of 0.290 delta and 0.154 gamma contracts. Heston/frozen middle controls
    # reduced some level variance but increased the gamma substep envelope, so
    # this cell buys independent direct evidence instead of a costly control.
    "low_feller": 512,
    "sigma_collapse": PRODUCTION_SLV_BATCHES,
    "near_expiry": PRODUCTION_SLV_BATCHES,
}
# The Near-KI discontinuity alone pays for the higher-resolution middle level.
# It estimates frozen-SLV minus matched Heston on one shared Sobol family; the
# already-certified Heston cell supplies the final independent expectation.
SLV_MULTILEVEL_CASES = frozenset({"near_ki"})


def stopping_excluded_variant_cases() -> tuple[str, ...]:
    """Cells whose batch count is declared and may never be stopped early.

    BOTH cells of a multilevel case, not just the SLV one. The telescoping
    estimator weights a Heston control (``heston_high_reference``) against the
    SLV levels, so the Heston cell's batch count is as much part of the contract
    as the SLV cell's own. One definition, because the exclusion is read in three
    places -- the policy builder, the payload validator, and the run declaration
    -- and the first version of it disagreed between them.
    """
    return tuple(
        sorted(
            f"{variant}/{case_name}"
            for case_name in SLV_MULTILEVEL_CASES
            for variant in ("heston", "heston_slv")
        )
    )


def stopping_excluded(variant: str, case_name: str) -> bool:
    """Whether this cell must spend its declared allocation."""
    return f"{variant}/{case_name}" in stopping_excluded_variant_cases()


PRODUCTION_SLV_BATCHES_BY_CASE = {
    case_name: (
        PRODUCTION_SLV_BATCHES if case_name in SLV_MULTILEVEL_CASES else primary_batches
    )
    for case_name, primary_batches in (PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE.items())
}
SLV_MID_CONTROL_PATHS_PER_BATCH = 8192
SLV_MID_CONTROL_BATCHES = 128
SLV_FROZEN_CONTROL_WEIGHT = 0.95
SLV_HESTON_CONTROL_WEIGHT = 0.85
# The three cells that carry 89.7% of the aggregate MC variance were measured
# on 2026-08-10 (docs/mc-reference-convergence): extending the eight-dimension
# residual-bridge profile to them is unbiased (0.28-0.65 sigma against an
# independent seed) and free (per-batch cost flat or slightly lower), worth
# 2.14x / 2.62x / 1.49x in SE^2*seconds. near_ko and near_expiry contribute
# 4.3% of the variance between them and stay on the single-factor profile.
SLV_SPOT_BRIDGE_PROFILE_BY_CASE = {
    "ordinary_full": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8},
    "ordinary_decayed": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8},
    "near_ko": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 1},
    "near_ki": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8},
    "low_feller": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8},
    "sigma_collapse": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8},
    "near_expiry": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 1},
}

# Schema 13 records, per cell, the estimator treatment its reference was built
# with, so a later reader can tell a re-run apart from a re-treatment. The
# control entry names the aggregate control structure in use; cross-fitted
# weights (spec WS-V2) are deferred to post-regeneration post-processing, so
# every cell currently declares the frozen-constant structure.
SCHEMA13_CONTROL_BY_VARIANT_CASE: dict[str, dict[str, str]] = {}


def reference_treatment_descriptor(variant: str, case_name: str) -> dict:
    """The exact estimator treatment this cell's MC reference was built with."""
    if variant == "heston_slv":
        profile = SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
    elif variant == "heston":
        profile = HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
    else:
        raise ValueError(f"unsupported variant: {variant}")
    control = SCHEMA13_CONTROL_BY_VARIANT_CASE.get(variant, {}).get(
        case_name, "none"
    )
    return {
        "bridge_strata": int(profile["strata"]),
        "bridge_dimensions": int(profile["dimensions"]),
        "control": control,
        "control_weights": None,
    }
# The unresolved SLV cells use finer bias ladders selected on development data.
# Low Feller stops at the finest valid 7->14 pair: 21,168 dimensions. The next
# integer refinement, 8->16, would exceed SciPy's 21,201 Sobol-dimension limit
# (3 streams * 504 contractual intervals * 16 substeps).
PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE = {
    "ordinary_full": {"target": 4, "fine": 8},
    "ordinary_decayed": {"target": 4, "fine": 8},
    "near_ko": {"target": 4, "fine": 8},
    "near_ki": {"target": 8, "fine": 16},
    "low_feller": {"target": 7, "fine": 14},
    "sigma_collapse": {"target": 4, "fine": 8},
    "near_expiry": {"target": 4, "fine": 8},
}
# Heston must match Near-KI because it is the independent high control there.
# Its low-Feller 4->8 cell is already certified and is not a control for SLV,
# so forcing the 21,168-dimensional SLV ladder onto it would add memory risk
# without changing the SLV estimator.
PRODUCTION_HESTON_QE_SUBSTEPS_BY_CASE = {
    case_name: (
        {"target": 8, "fine": 16}
        if case_name == "near_ki"
        else {"target": 4, "fine": 8}
    )
    for case_name in PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE
}
PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE = {
    "heston": PRODUCTION_HESTON_QE_SUBSTEPS_BY_CASE,
    "heston_slv": PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE,
}
PRODUCTION_RQMC_BATCH_WORKERS = 4
# Uniform by measurement, not by taste.  Batch workers only schedule the RQMC
# reduction, which is seed-keyed and ordered, so batch estimates are bitwise
# identical across worker counts: probe P2 ran the real paired reference at
# 1/4/8 workers and recorded max_abs_batch_diff 0.0 with 2.04x wall at four,
# flat to eight (roughly half the loop is serial or GIL-bound, so more workers
# is not the lever).  Peak RSS at four workers is 7.85 GB on the heaviest cell,
# the three-year ordinary_full horizon, leaving two concurrent cells near
# 15.7 GB against a ~32 GB budget.  The earlier 2-3 worker downgrades on the
# long-dated cells bought nothing measurable and cost wall-clock.
PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE = {
    variant: {
        case_name: PRODUCTION_RQMC_BATCH_WORKERS
        for case_name in PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE
    }
    for variant in ("heston", "heston_slv")
}
# Smooth cells can run two-at-a-time on the 14-core certification host. The
# high-dimensional Near-KI SLV middle level stays serialized below because a
# concurrent copy would exceed the host's 48 GiB memory envelope.
PRODUCTION_CELL_WORKERS = 2
PDE_REFINEMENT_RATIO_LIMIT = 1.25
PDE_REFINEMENT_NEGLIGIBLE_BOUND_FRACTION = 0.10
MIN_PRODUCTION_RQMC_BATCHES = 16
PRODUCTION_ENGINE_CONTROLS = {
    "grid_style": "concentrated",
    "v0_boundary": "degenerate_pde",
    "variance_grid_mode": "auto",
    "v_drift_scheme": "auto",
    "barrier_greek_steps_per_tick": 16,
    "greek_min_n_x": 300,
    "greek_min_n_v": 135,
    "greek_min_steps_per_year": 1600,
    "barrier_greek_min_n_x": 600,
}
REQUIRED_ANCHOR_NAMES = frozenset(
    {
        "heston_vanilla_analytical",
        "snowball_constant_variance_reduction",
        "slv_unit_leverage_reduction",
        "snowball_deterministic_variance_reduction",
    }
)

# A resumable production run fingerprints every implementation input that can
# change its numerical evidence.  Checkpoints from a different source state or
# run configuration are rejected rather than silently mixed into one report.
IMPLEMENTATION_INPUTS = (
    "example/mo_volmodels/11_pde_convergence_gate.py",
    "example/mo_volmodels/12_snowball_volmodel_backtest.py",
    "example/mo_volmodels/16_adi_greek_certification.py",
    "quantark/validation/__init__.py",
    "quantark/validation/greek_certification.py",
    # The rest of the certification package, which the file-by-file list had
    # silently outgrown: adaptive_allocation freezes the adaptive batch
    # allocation and sequential_admission decides early stops, so both govern
    # what evidence a run banks.
    "quantark/validation/adaptive_allocation.py",
    "quantark/validation/sequential_admission.py",
    "quantark/volmodels/adi_core.py",
    "quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py",
    "quantark/asset/equity/engine/pde/pde_execution_adapters.py",
    "quantark/backtest/replay/engine_factory.py",
    "quantark/asset/equity/engine/mc/snowball_mc_engine.py",
    "quantark/asset/equity/engine/mc/snowball_vol_mc_engines.py",
    "quantark/montecarlo/qmc_rqmc_driver.py",
    "quantark/montecarlo/conditional_snowball.py",
    "quantark/montecarlo/qmc_qe_coupling.py",
    # The SLV reference's QE variance step lives here, not in the engine file:
    # it was extracted so one definition could carry the optional Numba backend.
    # Both backends are bitwise equal, so this does not change any value -- but
    # the arithmetic that produces SLV reference values must stay inside the
    # fail-closed digest, or editing it would invalidate no checkpoint.
    "quantark/montecarlo/qe_kernels.py",
    # Identity machinery: it decides which banked cells may be reused, so the
    # full-source digest must cover it. It is excluded from the NUMERICAL
    # projection below, because it cannot change a number -- and a behaviour
    # change in it is self-revealing anyway, since it would change the projected
    # bytes of every file it projects.
    "quantark/validation/cell_identity.py",
)

# Inputs hashed for provenance but outside the numerical projection.
NUMERICAL_EXEMPT_INPUTS = frozenset({"quantark/validation/cell_identity.py"})

# Symbols in THIS file that cannot change any cell's numbers, and are therefore
# removed before the numerical digest is taken. Everything not listed here is
# numerical by default, so adding a function invalidates banked cells until it is
# audited and listed -- the list cannot rot silently, and `project_source`
# refuses a name that no longer exists.
#
# Two kinds qualify:
#
# (1) RUNS AFTER THE NUMBERS EXIST -- validation, reporting, publishing, hashing,
#     checkpoint I/O, and the amendment/parent-certificate paths. These read
#     evidence; they cannot produce it.
#
# (2) REACHES A CELL ONLY THROUGH THE DECLARED PLAN -- argument parsing, the case
#     table, the sampling and policy builders, `main` itself. Their entire effect
#     on a cell is the plan they declare, and `cell_plan_projection` hashes that
#     plan directly, so hashing their bytes as well is double counting. That
#     double counting is exactly what made the old fleet-wide guard invalidate
#     fourteen cells whenever one cell's plan moved.
#
# Category (2) is sound ONLY while the plan projection stays complete. Anything
# added here that could influence a cell by a route the plan does not record is a
# hole, so scrutinise `cell_plan_projection` before extending this list.
NON_NUMERICAL_SYMBOLS = (
    # (0) PROVENANCE BOOKKEEPING -- constants that describe the digest rather than
    #     feed the arithmetic. They must sit outside the projection they govern,
    #     or exempting is self-defeating: adding a validator helper and listing it
    #     here would change this tuple and invalidate every banked cell, charging
    #     a 36-hour re-run for a validator tweak. Nothing is lost, because each
    #     one's effect stays visible in the projection anyway -- exempting a
    #     pre-existing function still removes its body, a newly added exempt
    #     function was never present, and changing the input list changes which
    #     files' bytes are hashed. Only lists that DESCRIBE the digest belong
    #     here; an allocation table or a bound the arithmetic reads does not.
    "IMPLEMENTATION_INPUTS",
    "NUMERICAL_EXEMPT_INPUTS",
    "NON_NUMERICAL_SYMBOLS",
    # (1) after the fact
    "validate_payload",
    "_admissible_batch_count",
    # `make_decisions` reads banked cell evidence and renders routing verdicts;
    # it can decide, but it cannot produce or alter a single banked number.
    # Reclassified 2026-08-17 so the aggregate-alignment change (truncation ->
    # strided pooling) re-publishes from banked checkpoints instead of charging
    # a 51 h fleet re-run for publish-time decision code.
    "make_decisions",
    "render_markdown",
    "publish_payload",
    "build_decision_payload",
    "evidence_projection",
    "_projected_evidence_sha256",
    "_cell_evidence_sha256",
    "_canonical_sha256",
    "_json_default",
    "_atomic_write",
    "_checkpoint_path",
    "_write_checkpoint",
    "_load_checkpoint",
    "implementation_sha256",
    "numerical_implementation_sha256",
    "production_pde_compatibility_sha256",
    "cell_plan_projection",
    "cell_identity",
    "runtime_environment",
    # amendment and parent-certificate handling: a separate, later workflow
    "_validate_amendment_payload",
    "load_and_validate_parent_certificate",
    "parent_certificate_manifest",
    "make_amendment_decisions",
    "amendment_run_configuration",
    "amendment_cell_provenance",
    "amendment_sampling_by_variant",
    "amendment_reference_seeds",
    "amendment_qe_substeps_by_variant_case",
    "run_incremental_amendment",
    # (2) mediated entirely by the declared plan
    "main",
    "parse_args",
    "certification_cases",
    "build_sequential_policy",
    "stopping_excluded",
    "stopping_excluded_variant_cases",
    "_sampling_batches_for_case",
    "_sampling_primary_batches_for_case",
    "_normalized_batches_by_case",
    "_normalized_primary_batches_by_case",
)

# This narrower projection binds carried schema-9 results to the live
# production PDE dependency surface.  Certification-harness and Monte Carlo
# estimator changes intentionally do not invalidate previously completed PDE
# cells; a change anywhere in this projection does.
PRODUCTION_PDE_INPUT_ROOTS = (
    "quantark/asset/equity/engine/base_engine.py",
    "quantark/asset/equity/engine/pde",
    "quantark/asset/equity/param",
    "quantark/asset/equity/product/option/snowball_option.py",
    "quantark/asset/equity/product/option/snowball_helpers.py",
    "quantark/backtest/replay",
    "quantark/priceenv",
    "quantark/util/enum",
    "quantark/util/numerical",
    "quantark/volmodels",
)


def runtime_environment() -> dict:
    """Return the numerical runtime identity included in checkpoint hashes."""
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


ORDINARY = HestonParams(
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma=0.30,
    rho=-0.55,
)
LOW_FELLER = HestonParams(
    v0=0.04,
    kappa=0.60,
    theta=0.04,
    sigma=0.50,
    rho=-0.70,
)
SIGMA_COLLAPSE = HestonParams(
    v0=0.14027,
    kappa=3.0,
    theta=0.00306,
    sigma=0.00311,
    rho=-0.50,
)


@dataclass(frozen=True)
class GridPoint:
    n_x: int
    n_v: int
    n_t: int

    def as_dict(self) -> dict:
        return {"n_x": self.n_x, "n_v": self.n_v, "n_t": self.n_t}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    params: HestonParams
    maturity: float
    spot: float
    tags: tuple[str, ...]

    def as_dict(self) -> dict:
        p = self.params
        feller = (
            float("inf")
            if p.sigma == 0.0
            else 2.0 * p.kappa * p.theta / (p.sigma * p.sigma)
        )
        return {
            "name": self.name,
            "maturity": self.maturity,
            "spot": self.spot,
            "tags": list(self.tags),
            "feller_ratio": feller,
            "heston": {
                "v0": p.v0,
                "kappa": p.kappa,
                "theta": p.theta,
                "sigma": p.sigma,
                "rho": p.rho,
            },
        }


def certification_cases(*, quick: bool) -> list[CaseSpec]:
    """The minimum regime matrix; quick keeps one case per failure mechanism."""
    cases = [
        CaseSpec("ordinary_full", ORDINARY, 3.0, 100.0, ("full", "ordinary")),
        CaseSpec("ordinary_decayed", ORDINARY, 1.50, 92.0, ("decayed", "ordinary")),
        CaseSpec("near_ko", ORDINARY, 1.0, 102.7, ("near_barrier", "near_ko")),
        CaseSpec("near_ki", ORDINARY, 1.0, 75.5, ("near_barrier", "near_ki")),
        CaseSpec("low_feller", LOW_FELLER, 2.0, 100.0, ("low_feller",)),
        CaseSpec(
            "sigma_collapse",
            SIGMA_COLLAPSE,
            3.0,
            100.0,
            ("sigma_collapse", "convection_dominated"),
        ),
        CaseSpec("near_expiry", ORDINARY, 0.25, 100.0, ("near_expiry",)),
    ]
    if quick:
        keep = {"ordinary_decayed", "near_ko", "sigma_collapse"}
        return [case for case in cases if case.name in keep]
    return cases


def grid_ladders(
    maturity: float,
    *,
    quick: bool,
    dense_ki_stencil: bool = False,
) -> dict:
    """Separate single-axis ladders around one production candidate grid."""
    if quick:
        # The declarative S-grid's barrier-spacing guard needs at least ~80
        # nodes in the broad sigma-collapse domain; quick reduces V/time, not
        # below that correctness floor on the S axis.
        coarse = GridPoint(160, 16, max(12, int(math.ceil(24 * maturity))))
        target = GridPoint(180, 24, max(20, int(math.ceil(40 * maturity))))
        fine = GridPoint(220, 32, max(28, int(math.ceil(64 * maturity))))
    else:
        coarse = GridPoint(200, 90, max(120, int(math.ceil(800 * maturity))))
        target = GridPoint(300, 135, max(180, int(math.ceil(1600 * maturity))))
        fine = GridPoint(450, 180, max(240, int(math.ceil(2400 * maturity))))
        if dense_ki_stencil:
            # The production Greek policy uses sixteen ADI steps per 252-clock
            # KI tick. Certify that policy against a 32-step-per-tick solve.
            schedule_ticks = max(1, int(round(252.0 * maturity)))
            coarse = replace(coarse, n_t=8 * schedule_ticks)
            target = replace(target, n_t=16 * schedule_ticks)
            fine = replace(fine, n_t=32 * schedule_ticks)
            coarse = replace(coarse, n_x=450)
            target = replace(target, n_x=600)
            fine = replace(fine, n_x=750)
    return {
        "target": target,
        "n_x": [
            GridPoint(coarse.n_x, target.n_v, target.n_t),
            target,
            GridPoint(fine.n_x, target.n_v, target.n_t),
        ],
        "n_v": [
            GridPoint(target.n_x, coarse.n_v, target.n_t),
            target,
            GridPoint(target.n_x, fine.n_v, target.n_t),
        ],
        "n_t": [
            GridPoint(target.n_x, target.n_v, coarse.n_t),
            target,
            GridPoint(target.n_x, target.n_v, fine.n_t),
        ],
    }


def make_environment(spot: float, representative_vol: float) -> PricingEnvironment:
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.02),
        valuation_date=datetime(2026, 8, 3),
        spot_quote=SpotQuote(float(spot), asset_name="synthetic_index"),
        vol_surface=FlatVolSurface(float(representative_vol)),
        div_yield=ContinuousDividendYield(0.01),
    )


def bumped_environment(env: PricingEnvironment, spot: float) -> PricingEnvironment:
    quote = env.spot_quote
    return replace(
        env,
        spot_quote=SpotQuote(
            float(spot),
            timestamp=None if quote is None else quote.timestamp,
            asset_name=None if quote is None else quote.asset_name,
        ),
    )


def make_snowball(case: CaseSpec, *, dense_ki: bool = True) -> SnowballOption:
    maturity = float(case.maturity)
    if dense_ki:
        n_ki = max(1, int(math.ceil(252.0 * maturity)))
        # Build KO and KI times from the SAME integer clock.  Near-equal but
        # non-identical floats create zero-length MC intervals after cumulative
        # summation when the substep ladder is active.
        ki_times = np.arange(1, n_ki + 1, dtype=float) * maturity / n_ki
        ki_times[-1] = maturity
        first_ko = max(1, int(round(0.25 * n_ki / maturity)))
        ko_step = max(1, int(round((1.0 / 12.0) * n_ki / maturity)))
        ko_indices = list(range(first_ko, n_ki + 1, ko_step))
        if not ko_indices or ko_indices[-1] != n_ki:
            ko_indices.append(n_ki)
        ko_times = ki_times[np.asarray(ko_indices, dtype=int) - 1]
        ki_type = ObservationType.DISCRETE
        ki_continuous = False
    else:
        ko_times = np.arange(0.25, maturity + 1e-12, 1.0 / 12.0)
        if ko_times.size == 0 or ko_times[-1] < maturity - 1e-10:
            ko_times = np.append(ko_times, maturity)
        elif abs(float(ko_times[-1]) - maturity) <= 1e-10:
            ko_times[-1] = maturity
        ki_times = None
        ki_type = ObservationType.CONTINUOUS
        ki_continuous = True
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=maturity,
        contract_multiplier=1.0,
        is_reverse=False,
        payoff_config=PayoffConfig(
            include_principal=False,
            rebate_rate=0.15,
        ),
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[float(x) for x in ko_times],
            ki_barrier=75.0,
            ki_observation_type=ki_type,
            ki_observation_dates=(
                None if ki_times is None else [float(x) for x in ki_times]
            ),
            ki_continuous=ki_continuous,
        ),
    )


def make_leverage_surface(maturity: float, *, unit: bool = False) -> LeverageSurface:
    strikes = np.array([40.0, 60.0, 75.0, 90.0, 100.0, 103.0, 115.0, 140.0, 180.0])
    times = np.array([0.0, max(0.05, 0.5 * maturity), maturity])
    if unit:
        grid = np.ones((times.size, strikes.size))
    else:
        smile = np.array([1.14, 1.10, 1.06, 1.02, 1.00, 0.99, 0.97, 0.96, 0.95])
        grid = np.vstack([smile, 0.5 * (smile + 1.0), np.ones_like(smile)])
    return LeverageSurface(times, strikes, grid)


def central_bump_greeks(engine, product, env, bump: float) -> dict:
    """One frozen-domain central finite-bump price/delta/gamma exposure."""
    spot = float(env.spot)
    engine.params.bump_config = BumpConfig(
        spot_bump=float(bump), gamma_spot_bump=float(bump)
    )
    native = engine.calculate_greeks(product, env)
    values = {
        "price": float(native["price"]),
        "delta": float(native["delta"]),
        "gamma": float(native["gamma"]),
        "bump": float(bump),
        "spot_down": spot * (1.0 - bump),
        "spot_base": spot,
        "spot_up": spot * (1.0 + bump),
        "finite_bump_exposure": True,
    }
    policy_fn = getattr(engine, "greek_time_grid_policy", None)
    if policy_fn is not None:
        values["time_grid_policy"] = policy_fn(product, env)
    if not all(np.isfinite(values[key]) for key in ("price", "delta", "gamma")):
        raise ValueError("PDE central bump returned a non-finite result")
    return values


def make_pde_engine(
    variant: str,
    case: CaseSpec,
    grid: GridPoint,
    leverage: Optional[LeverageSurface],
):
    # Certification ladders must solve the declared grid exactly. Production
    # constructs the Stage-11 medium PV engine with PRODUCTION_ENGINE_CONTROLS,
    # whose Greek policy resolves that engine to this ladder's target. Leaving
    # those floors active on the coarse row would collapse two ladder points
    # onto one solve and falsely report a non-convergent axis.
    certification_controls = dict(PRODUCTION_ENGINE_CONTROLS)
    certification_controls.update(
        barrier_greek_steps_per_tick=0,
        greek_min_n_x=0,
        greek_min_n_v=0,
        greek_min_steps_per_year=0,
        barrier_greek_min_n_x=0,
    )
    kwargs = dict(
        n_x=grid.n_x,
        n_v=grid.n_v,
        n_t=grid.n_t,
        params=PDEParams(cache_enabled=False),
        **certification_controls,
    )
    if variant == "heston":
        return HestonSnowballPDESolver(case.params, **kwargs)
    if variant == "heston_slv":
        return HestonSLVSnowballPDESolver(
            case.params,
            leverage_surface=leverage,
            **kwargs,
        )
    raise ValueError(f"unsupported variant: {variant}")


def make_mc_params(paths_per_batch: int, batches: int, seed: int) -> MCParams:
    return MCParams(
        num_paths=int(paths_per_batch),
        time_steps=252,
        seed=int(seed),
        rqmc_min_batches=int(batches),
        rqmc_max_batches=int(batches),
        rqmc_target_std=1e-12,
        rqmc_paths_mode="per_batch",
    )


def make_mc_engine(
    variant: str,
    case: CaseSpec,
    leverage: Optional[LeverageSurface],
    *,
    paths_per_batch: int,
    batches: int,
    seed: int,
    substeps: int,
    qe_draw_provider=None,
    heston_spot_bridge_strata: int = HESTON_SPOT_BRIDGE_STRATA,
    heston_spot_bridge_dimensions: int = HESTON_SPOT_BRIDGE_DIMENSIONS,
    slv_spot_strata: int = SLV_SPOT_STRATA,
    slv_spot_antithetic: bool = SLV_SPOT_ANTITHETIC,
    slv_spot_bridge_strata: int = SLV_SPOT_BRIDGE_STRATA,
    slv_spot_bridge_dimensions: int = 1,
    slv_conditional_control_only: bool = False,
):
    common = dict(
        params=make_mc_params(paths_per_batch, batches, seed),
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        substeps_per_interval=int(substeps),
        rqmc_qe_draw_provider=qe_draw_provider,
    )
    if variant == "heston":
        return QESnowballMCEngine(
            case.params,
            martingale_correction=True,
            # QE variance is independent of the residual spot Brownian
            # factor. Integrate that factor exactly so barrier indicators do
            # not dominate finite-bump delta/gamma uncertainty.
            rqmc_affine_spot_factor=True,
            rqmc_spot_bridge_strata=int(heston_spot_bridge_strata),
            rqmc_spot_bridge_dimensions=int(heston_spot_bridge_dimensions),
            **common,
        )
    if variant == "heston_slv":
        # The SLV engine's construction seam resolves a local-vol artifact even
        # when a precomputed leverage surface makes it inactive in the path
        # loop. Supply an explicit constant surface so this self-contained
        # harness does not pretend its FlatVolSurface is a Dupire input grid.
        representative_vol = math.sqrt(max(case.params.v0, case.params.theta))
        local_vol = LocalVolSurface(
            strike_grid=np.array([40.0, 180.0]),
            time_grid=np.array([0.0, float(case.maturity)]),
            lv_grid=np.full((2, 2), representative_vol),
        )
        return HestonSLVQESnowballMCEngine(
            case.params,
            leverage_surface=leverage,
            local_vol_surface=local_vol,
            martingale_correction=True,
            rqmc_heston_conditional_control=True,
            rqmc_frozen_leverage_conditional_control=True,
            rqmc_spot_strata=int(slv_spot_strata),
            rqmc_spot_antithetic=bool(slv_spot_antithetic),
            rqmc_spot_bridge_strata=int(slv_spot_bridge_strata),
            rqmc_spot_bridge_dimensions=int(slv_spot_bridge_dimensions),
            rqmc_conditional_control_only=bool(slv_conditional_control_only),
            **common,
        )
    raise ValueError(f"unsupported variant: {variant}")


def paired_mc_reference(
    variant: str,
    case: CaseSpec,
    product: SnowballOption,
    env: PricingEnvironment,
    leverage: Optional[LeverageSurface],
    *,
    paths_per_batch: int,
    batches: int,
    seed: int,
    substeps: int,
    bump: float,
    qe_draw_provider=None,
    heston_spot_bridge_strata: int = HESTON_SPOT_BRIDGE_STRATA,
    heston_spot_bridge_dimensions: int = HESTON_SPOT_BRIDGE_DIMENSIONS,
    slv_spot_strata: int = SLV_SPOT_STRATA,
    slv_spot_antithetic: bool = SLV_SPOT_ANTITHETIC,
    slv_spot_bridge_strata: int = SLV_SPOT_BRIDGE_STRATA,
    slv_spot_bridge_dimensions: int = 1,
    slv_conditional_control_only: bool = False,
    rqmc_batch_workers: int = 1,
    first_batch: int = 0,
) -> PairedRQMCGreeksResult:
    spot = float(env.spot)
    specs = []
    # The engine spec bounds which batch IDS exist, while ``batches`` counts how
    # many this call runs.  A chunk starting at ``first_batch`` therefore needs a
    # spec that reaches ``first_batch + batches``, not merely the chunk size.
    spec_batches = int(first_batch) + int(batches)
    for shifted_spot in (spot * (1.0 - bump), spot, spot * (1.0 + bump)):
        shifted_env = bumped_environment(env, shifted_spot)
        engine = make_mc_engine(
            variant,
            case,
            leverage,
            paths_per_batch=paths_per_batch,
            batches=spec_batches,
            seed=seed,
            substeps=substeps,
            qe_draw_provider=qe_draw_provider,
            heston_spot_bridge_strata=heston_spot_bridge_strata,
            heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
            slv_spot_strata=slv_spot_strata,
            slv_spot_antithetic=slv_spot_antithetic,
            slv_spot_bridge_strata=slv_spot_bridge_strata,
            slv_spot_bridge_dimensions=slv_spot_bridge_dimensions,
            slv_conditional_control_only=slv_conditional_control_only,
        )
        spec = engine.build_rqmc_session_spec(product, shifted_env)
        if spec is None:
            raise RuntimeError("RQMC reference did not produce a run spec")
        specs.append(spec)
    return run_paired_rqmc_greeks(
        specs[0],
        specs[1],
        specs[2],
        spot=spot,
        relative_bump=bump,
        batches=batches,
        batch_workers=rqmc_batch_workers,
        first_batch=first_batch,
    )


def combine_two_level_control(
    controlled: PairedRQMCGreeksResult,
    low_control: PairedRQMCGreeksResult,
    high_control: PairedRQMCGreeksResult,
    *,
    high_batches_per_low: int = 1,
) -> PairedRQMCGreeksResult:
    """Return ``controlled-low control+high control`` by scramble.

    The low component must be the exact conditional-control term already
    present in ``controlled``. Replacing it with an independent high-resolution
    estimate of the same control expectation preserves unbiasedness. When the
    high run has multiple batches per low scramble, disjoint high batches are
    averaged into one row before the components are combined. The resulting
    rows remain independent and retain one auditable Student-t sample.
    """
    if isinstance(high_batches_per_low, bool) or int(high_batches_per_low) < 1:
        raise ValueError("high_batches_per_low must be a positive integer")
    high_batches_per_low = int(high_batches_per_low)
    low_shape = np.asarray(controlled.batch_estimates).shape
    if low_shape != np.asarray(low_control.batch_estimates).shape:
        raise ValueError("controlled and low-control batch shapes do not match")
    if len(low_shape) != 2 or low_shape[1] != 5:
        raise ValueError("two-level Heston control batches must have five columns")
    expected_high_shape = (low_shape[0] * high_batches_per_low, 5)
    if np.asarray(high_control.batch_estimates).shape != expected_high_shape:
        raise ValueError(
            "high-control batch shape does not match grouped low scrambles: "
            f"expected {expected_high_shape}, got "
            f"{np.asarray(high_control.batch_estimates).shape}"
        )
    if controlled.batches_used != low_control.batches_used:
        raise ValueError("controlled and low-control batch counts do not match")
    if high_control.batches_used != expected_high_shape[0]:
        raise ValueError("high-control batch count does not match its rows")
    results = (controlled, low_control, high_control)
    if (
        len({result.spot for result in results}) != 1
        or len({result.relative_bump for result in results}) != 1
    ):
        raise ValueError("two-level Heston control bump semantics do not match")

    estimates = (
        np.asarray(controlled.batch_estimates, dtype=float)
        - np.asarray(low_control.batch_estimates, dtype=float)
        + np.asarray(high_control.batch_estimates, dtype=float)
        .reshape(controlled.batches_used, high_batches_per_low, 5)
        .mean(axis=1)
    )
    batches = int(controlled.batches_used)
    means = np.mean(estimates, axis=0)
    covariance = np.asarray(np.cov(estimates, rowvar=False, ddof=1), dtype=float)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0) / batches)
    return PairedRQMCGreeksResult(
        price=float(means[1]),
        price_std_error=float(standard_errors[1]),
        delta=float(means[3]),
        delta_std_error=float(standard_errors[3]),
        gamma=float(means[4]),
        gamma_std_error=float(standard_errors[4]),
        spot=float(controlled.spot),
        relative_bump=float(controlled.relative_bump),
        absolute_bump=float(controlled.absolute_bump),
        paths_per_batch=int(controlled.paths_per_batch),
        batches_used=batches,
        total_unique_paths=sum(result.total_unique_paths for result in results),
        total_path_valuations=sum(result.total_path_valuations for result in results),
        randomization_key=(
            "two_level_control("
            f"{controlled.randomization_key};"
            f"{low_control.randomization_key};"
            f"{high_control.randomization_key};"
            f"high_batches_per_low={high_batches_per_low})"
        ),
        batch_estimates=estimates,
        covariance=covariance,
    )


def paired_result_from_serialized(
    payload: dict,
    *,
    randomization_label: str,
) -> PairedRQMCGreeksResult:
    """Rebuild and verify a paired result from checkpointed raw batches."""
    estimates = np.asarray(payload.get("batch_estimates"), dtype=float)
    batches = int(payload.get("batches_used", 0))
    if (
        estimates.shape != (batches, 5)
        or batches < 2
        or not np.all(np.isfinite(estimates))
    ):
        raise ValueError("serialized paired result has invalid batch estimates")
    covariance = np.asarray(np.cov(estimates, rowvar=False, ddof=1), dtype=float)
    means = np.mean(estimates, axis=0)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0) / batches)
    expected = np.asarray(
        [
            payload.get("price"),
            payload.get("price_std_error"),
            payload.get("delta"),
            payload.get("delta_std_error"),
            payload.get("gamma"),
            payload.get("gamma_std_error"),
        ],
        dtype=float,
    )
    actual = np.asarray(
        [
            means[1],
            standard_errors[1],
            means[3],
            standard_errors[3],
            means[4],
            standard_errors[4],
        ],
        dtype=float,
    )
    serialized_covariance = np.asarray(payload.get("covariance"), dtype=float)
    if (
        expected.shape != (6,)
        or not np.all(np.isfinite(expected))
        or not np.allclose(actual, expected, rtol=1e-12, atol=1e-14)
        or serialized_covariance.shape != (5, 5)
        or not np.allclose(covariance, serialized_covariance, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("serialized paired result does not match its raw batches")
    return PairedRQMCGreeksResult(
        price=float(means[1]),
        price_std_error=float(standard_errors[1]),
        delta=float(means[3]),
        delta_std_error=float(standard_errors[3]),
        gamma=float(means[4]),
        gamma_std_error=float(standard_errors[4]),
        spot=float(payload["spot"]),
        relative_bump=float(payload["relative_bump"]),
        absolute_bump=float(payload["absolute_bump"]),
        paths_per_batch=int(payload["paths_per_batch"]),
        batches_used=batches,
        total_unique_paths=int(payload["total_unique_paths"]),
        total_path_valuations=int(payload["total_path_valuations"]),
        randomization_key=(
            f"{randomization_label}({payload.get('randomization_key')})"
        ),
        batch_estimates=estimates,
        covariance=covariance,
        control_batch_estimates=None,
    )


def combine_grouped_rqmc_components(
    components: Sequence[tuple[float, PairedRQMCGreeksResult]],
    *,
    output_batches: int,
    estimator_label: str,
) -> PairedRQMCGreeksResult:
    """Combine independent/coupled multilevel terms into outer batch rows.

    Each component's rows are partitioned into the same number of disjoint
    outer groups before the signed means are added. Components with matching
    batch counts and scramble families therefore retain their pathwise
    covariance, while independently seeded levels remain independent. No row
    is reused, and the final Student-t sample has ``output_batches - 1``
    degrees of freedom.
    """
    if not components:
        raise ValueError("grouped RQMC estimator requires at least one component")
    if isinstance(output_batches, bool) or int(output_batches) < 2:
        raise ValueError("output_batches must be an integer >= 2")
    output_batches = int(output_batches)
    results = tuple(result for _, result in components)
    if (
        len({result.spot for result in results}) != 1
        or len({result.relative_bump for result in results}) != 1
    ):
        raise ValueError("grouped RQMC components use inconsistent bump semantics")

    estimates = np.zeros((output_batches, 5), dtype=float)
    key_parts = []
    total_unique_paths = 0
    total_path_valuations = 0
    effective_paths_per_batch = 0
    for coefficient, result in components:
        coefficient = float(coefficient)
        rows = np.asarray(result.batch_estimates, dtype=float)
        if not np.isfinite(coefficient):
            raise ValueError("grouped RQMC coefficient must be finite")
        if (
            rows.shape != (int(result.batches_used), 5)
            or not np.all(np.isfinite(rows))
            or result.batches_used % output_batches != 0
        ):
            raise ValueError(
                "grouped RQMC component rows must be finite and divisible by "
                "output_batches"
            )
        rows_per_group = int(result.batches_used) // output_batches
        estimates += coefficient * rows.reshape(output_batches, rows_per_group, 5).mean(
            axis=1
        )
        key_parts.append(
            f"{coefficient:+.17g}*group{rows_per_group}({result.randomization_key})"
        )
        total_unique_paths += int(result.total_unique_paths)
        total_path_valuations += int(result.total_path_valuations)
        effective_paths_per_batch += int(result.total_unique_paths) // output_batches

    covariance = np.asarray(np.cov(estimates, rowvar=False, ddof=1), dtype=float)
    means = np.mean(estimates, axis=0)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0) / output_batches)
    return PairedRQMCGreeksResult(
        price=float(means[1]),
        price_std_error=float(standard_errors[1]),
        delta=float(means[3]),
        delta_std_error=float(standard_errors[3]),
        gamma=float(means[4]),
        gamma_std_error=float(standard_errors[4]),
        spot=float(results[0].spot),
        relative_bump=float(results[0].relative_bump),
        absolute_bump=float(results[0].absolute_bump),
        paths_per_batch=effective_paths_per_batch,
        batches_used=output_batches,
        total_unique_paths=total_unique_paths,
        total_path_valuations=total_path_valuations,
        randomization_key=f"{estimator_label}({';'.join(key_parts)})",
        batch_estimates=estimates,
        covariance=covariance,
        control_batch_estimates=None,
    )


def extract_embedded_conditional_control(
    controlled: PairedRQMCGreeksResult,
) -> PairedRQMCGreeksResult:
    """Materialize the control rows already evaluated by ``controlled``.

    The returned result carries zero incremental work counts. It can therefore
    enter :func:`combine_two_level_control` without double-counting paths while
    retaining one independently auditable row per outer scramble.
    """
    if controlled.control_batch_estimates is None:
        raise ValueError("paired RQMC result has no embedded conditional control")
    estimates = np.asarray(controlled.control_batch_estimates, dtype=float)
    expected_shape = (int(controlled.batches_used), 5)
    if estimates.shape != expected_shape or not np.all(np.isfinite(estimates)):
        raise ValueError(
            "embedded conditional-control batches must be finite with shape "
            f"{expected_shape}, got {estimates.shape}"
        )
    covariance = np.asarray(np.cov(estimates, rowvar=False, ddof=1), dtype=float)
    means = np.mean(estimates, axis=0)
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0.0) / controlled.batches_used
    )
    return replace(
        controlled,
        price=float(means[1]),
        price_std_error=float(standard_errors[1]),
        delta=float(means[3]),
        delta_std_error=float(standard_errors[3]),
        gamma=float(means[4]),
        gamma_std_error=float(standard_errors[4]),
        total_unique_paths=0,
        total_path_valuations=0,
        randomization_key=(
            f"embedded_conditional_control({controlled.randomization_key})"
        ),
        batch_estimates=estimates,
        covariance=covariance,
        control_batch_estimates=None,
    )


def coupled_qe_providers(
    *,
    seed: int,
    paths_per_batch: int,
    target_dt: np.ndarray,
    fine_dt: np.ndarray,
    reuse_count: int = 1,
) -> tuple[CoupledQESubstepDrawProvider, CoupledQESubstepDrawProvider]:
    target = CoupledQESubstepDrawProvider(
        seed=int(seed),
        n_paths=int(paths_per_batch),
        target_dt=target_dt,
        fine_dt=fine_dt,
        role="target",
        reuse_count=int(reuse_count),
    )
    return target, replace(target, role="fine")


def build_slv_multilevel_reference(
    case: CaseSpec,
    *,
    product: SnowballOption,
    env: PricingEnvironment,
    leverage: LeverageSurface,
    primary_target: PairedRQMCGreeksResult,
    primary_fine: PairedRQMCGreeksResult,
    target_dt: np.ndarray,
    fine_dt: np.ndarray,
    target_substeps: int,
    fine_substeps: int,
    heston_high_cell: dict,
    primary_seed: int,
    mid_seed: int,
    slv_spot_bridge_strata: int,
    slv_spot_bridge_dimensions: int,
    rqmc_batch_workers: int,
) -> tuple[PairedRQMCGreeksResult, PairedRQMCGreeksResult, dict]:
    """Build the held-out three-level SLV reference for hard regimes.

    The estimator is, by outer scramble,

    ``Y - a F_low + a F_high - b H_low + b H_high``.

    ``F`` is the frozen-leverage exact conditional control and ``H`` is the
    exact-affine Heston reference. Each subtraction/addition pair estimates the
    same expectation on independent held-out seed families, so fixed weights
    selected on the development family preserve unbiasedness.
    """
    if case.name not in SLV_MULTILEVEL_CASES:
        raise ValueError(f"{case.name}: no production SLV multilevel profile")
    if (
        heston_high_cell.get("variant") != "heston"
        or heston_high_cell.get("case", {}).get("name") != case.name
    ):
        raise ValueError("SLV multilevel control requires the matching Heston cell")
    high_reference = heston_high_cell.get("reference", {})
    heston_high_seed = int(high_reference.get("seed", -1))
    if len({int(primary_seed), int(mid_seed), heston_high_seed}) != 3:
        raise ValueError("SLV multilevel seed families must be mutually independent")
    if int(high_reference.get("target_substeps_per_interval", 0)) != int(
        target_substeps
    ) or int(high_reference.get("fine_substeps_per_interval", 0)) != int(fine_substeps):
        raise ValueError("SLV and Heston control substep profiles do not match")
    if (
        primary_target.batches_used != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE[case.name]
        or primary_fine.batches_used != primary_target.batches_used
        or primary_target.batches_used % PRODUCTION_SLV_BATCHES != 0
    ):
        raise ValueError("SLV primary batches do not match the multilevel profile")

    frozen_low = {
        "target": extract_embedded_conditional_control(primary_target),
        "fine": extract_embedded_conditional_control(primary_fine),
    }
    mid_target_provider, mid_fine_provider = coupled_qe_providers(
        seed=mid_seed,
        paths_per_batch=SLV_MID_CONTROL_PATHS_PER_BATCH,
        target_dt=target_dt,
        fine_dt=fine_dt,
    )
    frozen_high = {}
    heston_low = {}
    for level, substeps, provider in (
        ("target", target_substeps, mid_target_provider),
        ("fine", fine_substeps, mid_fine_provider),
    ):
        frozen_high[level] = paired_mc_reference(
            "heston_slv",
            case,
            product,
            env,
            leverage,
            paths_per_batch=SLV_MID_CONTROL_PATHS_PER_BATCH,
            batches=SLV_MID_CONTROL_BATCHES,
            seed=mid_seed,
            substeps=substeps,
            bump=SPOT_BUMP,
            qe_draw_provider=provider,
            slv_spot_strata=1,
            slv_spot_antithetic=False,
            slv_spot_bridge_strata=slv_spot_bridge_strata,
            slv_spot_bridge_dimensions=slv_spot_bridge_dimensions,
            slv_conditional_control_only=True,
            rqmc_batch_workers=rqmc_batch_workers,
        )
        heston_low[level] = replace(
            extract_embedded_conditional_control(frozen_high[level]),
            # Both estimates were evaluated from the same generated QE and
            # residual-bridge paths; one randomization key makes that coupling
            # explicit while the frozen run's doubled valuation count records
            # the bundled Heston payoff work.
            randomization_key=frozen_high[level].randomization_key,
        )

    heston_high = {
        level: paired_result_from_serialized(
            high_reference[level],
            randomization_label=f"heston-high/{case.name}/{level}",
        )
        for level in ("target", "fine")
    }
    combined = {}
    for level, primary in (
        ("target", primary_target),
        ("fine", primary_fine),
    ):
        combined[level] = combine_grouped_rqmc_components(
            (
                (1.0, primary),
                (-SLV_FROZEN_CONTROL_WEIGHT, frozen_low[level]),
                (SLV_FROZEN_CONTROL_WEIGHT, frozen_high[level]),
                (-SLV_HESTON_CONTROL_WEIGHT, heston_low[level]),
                (SLV_HESTON_CONTROL_WEIGHT, heston_high[level]),
            ),
            output_batches=PRODUCTION_SLV_BATCHES,
            estimator_label=f"slv-three-level/{case.name}/{level}",
        )

    metadata = {
        "name": "three_level_frozen_slv_heston_control",
        "identity": "Y-a*F_low+a*F_high-b*H_low+b*H_high",
        "weights": {
            "frozen_slv": SLV_FROZEN_CONTROL_WEIGHT,
            "heston": SLV_HESTON_CONTROL_WEIGHT,
        },
        "primary_seed": int(primary_seed),
        "mid_control_seed": int(mid_seed),
        "heston_high_seed": heston_high_seed,
        "outer_batches": PRODUCTION_SLV_BATCHES,
        "primary_paths_per_batch": int(primary_target.paths_per_batch),
        "primary_batches": int(primary_target.batches_used),
        "mid_paths_per_batch": SLV_MID_CONTROL_PATHS_PER_BATCH,
        "mid_batches": SLV_MID_CONTROL_BATCHES,
        "heston_high_cell_sha256": hashlib.sha256(
            json.dumps(
                evidence_projection(heston_high_cell),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "components": {
            level: {
                "primary_low": primary.as_dict(include_batches=True),
                "frozen_high": frozen_high[level].as_dict(include_batches=True),
                "heston_low": heston_low[level].as_dict(include_batches=True),
                "heston_high_reference": {
                    "variant": "heston",
                    "case": case.name,
                    "batches": int(heston_high[level].batches_used),
                    "randomization_key": heston_high[level].randomization_key,
                },
            }
            for level, primary in (
                ("target", primary_target),
                ("fine", primary_fine),
            )
        },
    }
    return combined["target"], combined["fine"], metadata


def _economic_value(scale: EconomicGreekScale, greek: str, value: float) -> float:
    if greek == "delta":
        return scale.delta_contracts(value)
    if greek == "gamma":
        return scale.gamma_hedge_contract_change(value)
    raise ValueError(greek)


def _axis_ladder_rows(
    variant: str,
    case: CaseSpec,
    product: SnowballOption,
    env: PricingEnvironment,
    leverage: Optional[LeverageSurface],
    ladders: dict,
    bumps: Sequence[float],
) -> tuple[dict, dict]:
    target = ladders["target"]
    cache: dict[tuple[int, int, int, float], dict] = {}

    def solve(grid: GridPoint, bump: float) -> dict:
        key = (grid.n_x, grid.n_v, grid.n_t, float(bump))
        if key not in cache:
            engine = make_pde_engine(variant, case, grid, leverage)
            started = time.perf_counter()
            values = central_bump_greeks(engine, product, env, bump)
            values["seconds"] = time.perf_counter() - started
            values["grid"] = grid.as_dict()
            cache[key] = values
        return cache[key]

    axes = {}
    for axis in ("n_x", "n_v", "n_t"):
        axes[axis] = [solve(grid, SPOT_BUMP) for grid in ladders[axis]]
    bump_rows = [solve(target, bump) for bump in bumps]
    return {"axes": axes, "bump": bump_rows}, cache


def _pde_envelope_components(ladders: dict, target: dict) -> dict[str, dict]:
    components = {"delta": {}, "gamma": {}}
    for greek in components:
        for axis in ("n_x", "n_v", "n_t"):
            fine = ladders["axes"][axis][-1]
            components[greek][axis] = abs(float(fine[greek]) - float(target[greek]))
        finer_bumps = [row for row in ladders["bump"] if float(row["bump"]) < SPOT_BUMP]
        bump_ref = max(finer_bumps, key=lambda row: row["bump"])
        components[greek]["spot_bump_semantic_diagnostic"] = abs(
            float(bump_ref[greek]) - float(target[greek])
        )
        # The 1% bump is the declared hedge exposure, not a small-h
        # approximation to a classical derivative. Smaller bumps diagnose
        # that semantic choice but are not PDE discretization uncertainty.
        components[greek]["total"] = float(
            sum(components[greek][axis] for axis in ("n_x", "n_v", "n_t"))
        )
    return components


def _pde_signed_refinement_components(ladders: dict, target: dict) -> dict[str, dict]:
    components = {"delta": {}, "gamma": {}}
    for greek in components:
        for axis in ("n_x", "n_v", "n_t"):
            components[greek][axis] = float(ladders["axes"][axis][-1][greek]) - float(
                target[greek]
            )
    return components


def _pde_refinement_diagnostics(ladders: dict, scale: EconomicGreekScale) -> dict:
    """Require each PDE axis to contract, apart from immaterial grid jitter."""
    diagnostics = {}
    bounds = {
        "delta": DELTA_CELL_BOUND_CONTRACTS,
        "gamma": GAMMA_CELL_BOUND_CONTRACTS,
    }
    for greek, bound in bounds.items():
        axes = {}
        for axis in ("n_x", "n_v", "n_t"):
            rows = ladders["axes"][axis]
            if len(rows) != 3:
                raise ValueError(f"{axis} refinement ladder must have three points")
            values = [_economic_value(scale, greek, float(row[greek])) for row in rows]
            coarse_to_target = values[1] - values[0]
            target_to_fine = values[2] - values[1]
            tolerance = max(
                PDE_REFINEMENT_RATIO_LIMIT * abs(coarse_to_target),
                PDE_REFINEMENT_NEGLIGIBLE_BOUND_FRACTION * bound,
            )
            axes[axis] = {
                "coarse_to_target_contracts": coarse_to_target,
                "target_to_fine_contracts": target_to_fine,
                "absolute_refinement_ratio": (
                    None
                    if coarse_to_target == 0.0 and target_to_fine != 0.0
                    else (
                        0.0
                        if coarse_to_target == 0.0
                        else abs(target_to_fine / coarse_to_target)
                    )
                ),
                "tolerance_contracts": tolerance,
                "status": (
                    EquivalenceStatus.PASS.value
                    if abs(target_to_fine) <= tolerance
                    else EquivalenceStatus.FAIL.value
                ),
            }
        diagnostics[greek] = {
            "axes": axes,
            "status": _combined_status(item["status"] for item in axes.values()),
        }
    diagnostics["status"] = _combined_status(
        diagnostics[greek]["status"] for greek in ("delta", "gamma")
    )
    return diagnostics


def build_heston_high_control_evidence(
    case: CaseSpec,
    *,
    paths_per_batch: int,
    batches: int,
    seed: int,
    target_substeps: int,
    fine_substeps: int,
    heston_spot_bridge_strata: int,
    heston_spot_bridge_dimensions: int,
    rqmc_batch_workers: int,
) -> dict:
    """Compute reference-only Heston evidence for an SLV control variate.

    This deliberately does not evaluate a PDE grid or issue a Heston routing
    verdict.  The schema-9 Heston certificate remains the production authority;
    these rows exist only to provide an independent, resolution-matched high
    expectation to the schema-11 Near-KI SLV estimator.
    """
    product = make_snowball(case, dense_ki=True)
    env = make_environment(
        case.spot,
        math.sqrt(max(case.params.v0, case.params.theta)),
    )
    probe_engine = make_mc_engine(
        "heston",
        case,
        None,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=1,
        heston_spot_bridge_strata=heston_spot_bridge_strata,
        heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
        slv_spot_strata=1,
        slv_spot_antithetic=False,
        slv_spot_bridge_strata=1,
        slv_spot_bridge_dimensions=1,
    )
    _, contractual_dt, _, _ = probe_engine._build_time_grid(
        product, env, float(case.maturity)
    )
    target_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / target_substeps,
        target_substeps,
    )
    fine_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / fine_substeps,
        fine_substeps,
    )
    target_provider, fine_provider = coupled_qe_providers(
        seed=seed,
        paths_per_batch=paths_per_batch,
        target_dt=target_dt,
        fine_dt=fine_dt,
        reuse_count=1,
    )
    references = {}
    for level, substeps, provider in (
        ("target", target_substeps, target_provider),
        ("fine", fine_substeps, fine_provider),
    ):
        references[level] = paired_mc_reference(
            "heston",
            case,
            product,
            env,
            None,
            paths_per_batch=paths_per_batch,
            batches=batches,
            seed=seed,
            substeps=substeps,
            bump=SPOT_BUMP,
            qe_draw_provider=provider,
            heston_spot_bridge_strata=heston_spot_bridge_strata,
            heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
            slv_spot_strata=1,
            slv_spot_antithetic=False,
            slv_spot_bridge_strata=1,
            slv_spot_bridge_dimensions=1,
            rqmc_batch_workers=rqmc_batch_workers,
        )
    return {
        "variant": "heston",
        "case": case.as_dict(),
        "purpose": "reference_only_slv_high_control",
        "reference": {
            "scheme": (
                "QE-M paired randomized Sobol with exact affine spot-factor "
                "conditioning"
            ),
            "seed": int(seed),
            "primary": "fine",
            "target_substeps_per_interval": int(target_substeps),
            "fine_substeps_per_interval": int(fine_substeps),
            "heston_spot_bridge_strata": int(heston_spot_bridge_strata),
            "heston_spot_bridge_dimensions": int(heston_spot_bridge_dimensions),
            "batch_workers": int(rqmc_batch_workers),
            "target": references["target"].as_dict(include_batches=True),
            "fine": references["fine"].as_dict(include_batches=True),
        },
    }


def gate_driven_reference_levels(
    *,
    run_level,
    policy: SequentialAdmissionPolicy,
    scale: EconomicGreekScale,
    pde_target: dict,
    raw_pde_envelopes: dict,
    bounds: dict,
    chunk_batches: int,
    max_batches: int,
) -> tuple:
    """Price a cell in chunks, stopping once every greek's gate is decided.

    The batch stream is prefix-invariant (verified bitwise), so a chunk extends
    the run rather than perturbing it: batch k is the same batch however the
    run was segmented, and the accumulated mean stays the mean of a fixed point
    set.  ``concatenate_paired_results`` recomputes the statistics from the
    joined rows, so the banked evidence is identical to a single call at the
    stopping count.

    Stopping is judged against the anytime-valid width, never the fixed-sample
    standard error: the stop time is data-dependent, and a Student-t interval is
    valid at one pre-chosen batch count only.

    Returns ``(target, fine, record)``; ``record`` carries the per-greek
    decisions and the declared policy for the evidence.
    """
    target_chunks: list = []
    fine_chunks: list = []
    banked = 0
    decisions: dict = {}
    while banked < int(max_batches):
        count = min(int(chunk_batches), int(max_batches) - banked)
        target_chunks.append(run_level("target", banked, count))
        fine_chunks.append(run_level("fine", banked, count))
        banked += count
        target = concatenate_paired_results(target_chunks)
        fine = concatenate_paired_results(fine_chunks)

        decisions = {}
        for greek, cell_bound in bounds.items():
            attribute = "batch_delta" if greek == "delta" else "batch_gamma"
            fine_series = np.asarray(
                [
                    _economic_value(scale, greek, float(value))
                    for value in getattr(fine, attribute)
                ],
                dtype=float,
            )
            target_series = np.asarray(
                [
                    _economic_value(scale, greek, float(value))
                    for value in getattr(target, attribute)
                ],
                dtype=float,
            )
            substep_series = target_series - fine_series
            decisions[greek] = sequential_admission(
                policy=policy,
                batches_used=banked,
                reference_gap=_economic_value(
                    scale, greek, float(pde_target[greek])
                )
                - float(np.mean(fine_series)),
                greek_batch_standard_deviation=float(np.std(fine_series, ddof=1)),
                pde_discretization_envelope=abs(
                    _economic_value(
                        scale, greek, raw_pde_envelopes[greek]["total"]
                    )
                ),
                economic_bound=float(cell_bound),
                substep_bias_mean=float(np.mean(substep_series)),
                substep_batch_standard_deviation=float(
                    np.std(substep_series, ddof=1)
                ),
            )
        if all(
            decision.status
            in (
                SequentialAdmissionStatus.ADMIT,
                SequentialAdmissionStatus.REJECT,
            )
            for decision in decisions.values()
        ):
            break

    record = {
        "stopping_rule": "anytime_valid_sequential",
        "batches_banked": int(banked),
        "chunk_batches": int(chunk_batches),
        "max_batches": int(max_batches),
        "policy": policy.declaration(),
        "policy_sha256": policy.sha256(),
        "decisions": {
            greek: decision.as_dict() for greek, decision in decisions.items()
        },
    }
    return target, fine, record


def _admissible_batch_count(
    cell: dict,
    run_configuration: dict,
    *,
    variant: str,
    case_name: str,
    declared_batches: int,
) -> int:
    """How many batches this cell is permitted to have banked.

    A fixed-allocation cell must bank exactly its declared count: that equality
    is what proves nobody quietly ran fewer batches than the configuration says.
    Gate-driven stopping breaks the equality on purpose, so the invariant becomes
    strictly more to check rather than less --

      * the shortfall must be EXPLAINED, by a recorded decision on every greek;
        a cell that banked less without deciding has simply run short,
      * the cap in the recorded policy must equal the declared allocation, which
        is what makes gate-driven stopping unable to overspend,
      * the recorded policy must be the one the run declared, so a cell cannot
        arrive stopped under some other, looser rule,
      * and a cell may never bank MORE than declared.

    Returns the count the level arrays must have.
    """
    declared = int(declared_batches)
    record = cell.get("sequential_stopping")
    policy_config = run_configuration.get("sequential_stopping", {}) or {}
    label = f"{variant}/{case_name}"

    if record is None:
        if policy_config.get("enabled") and not stopping_excluded(variant, case_name):
            raise ValueError(
                f"{label}: run declares sequential stopping but the cell "
                "records no stopping decision"
            )
        return declared

    if not policy_config.get("enabled"):
        raise ValueError(
            f"{label}: cell stopped sequentially but the run does not declare it"
        )

    banked = int(record.get("batches_banked", -1))
    if not 0 < banked <= declared:
        raise ValueError(
            f"{label}: banked {banked} batches outside (0, {declared}]"
        )

    declaration = record.get("policy", {}) or {}
    if int(declaration.get("max_batches", -1)) != declared:
        raise ValueError(
            f"{label}: stopping policy cap {declaration.get('max_batches')} "
            f"does not equal the declared allocation {declared}"
        )
    for key in ("margin_fraction", "family_alpha"):
        if float(declaration.get(key, float("nan"))) != float(policy_config[key]):
            raise ValueError(f"{label}: stopping policy {key} is not the declared one")
    if int(record.get("chunk_batches", -1)) != int(policy_config["chunk_batches"]):
        raise ValueError(f"{label}: stopping chunk size is not the declared one")

    decisions = record.get("decisions", {}) or {}
    if set(decisions) != {"delta", "gamma"}:
        raise ValueError(f"{label}: stopping record does not cover both greeks")
    if banked < declared:
        undecided = sorted(
            greek
            for greek, decision in decisions.items()
            if decision.get("status")
            not in (
                SequentialAdmissionStatus.ADMIT.value,
                SequentialAdmissionStatus.REJECT.value,
            )
        )
        if undecided:
            raise ValueError(
                f"{label}: banked {banked} of {declared} batches with "
                f"{', '.join(undecided)} undecided"
            )
    for greek, decision in sorted(decisions.items()):
        if int(decision.get("batches_used", -1)) != banked:
            raise ValueError(
                f"{label}/{greek}: decision batch count does not match the "
                "banked evidence"
            )
    return banked


def build_sequential_policy(
    args, variant: str, case_name: str, *, cap: int
) -> Optional[SequentialAdmissionPolicy]:
    """The declared stopping policy for one cell, or None to spend the cap.

    Returns None -- the fixed-allocation path -- unless ``--sequential`` is set,
    and always for BOTH cells of a multilevel case, whose telescoping level
    weights require exactly their declared batch counts. Those cells are
    declared, not sized.

    The Heston cell is excluded for the same reason as the SLV cell it feeds: it
    enters that estimator as ``heston_high_reference``, so its batch count is
    part of the same contract. Excluding only the consumer is not enough, and the
    35.5h fleet showed why -- it stopped ``heston/near_ki`` at 1664 of 2048 and
    fed the truncated control into the SLV estimator at weight 0.85. That
    truncation is SELECTED, not merely short: the cell stopped because its own
    delta and gamma gate closed, so the stop time is a function of the estimates
    that become the control, and ``E[H_high(tau)] != E[H_high]``. The
    anytime-valid sequence keeps a cell's OWN interval honest under optional
    stopping; it carries no such guarantee into a downstream estimator that
    consumes the stopped mean at a fixed weight.

    The cap is the cell's frozen allocation, so a gate-driven run can never cost
    more than the fixed run it replaces. Every parameter is fixed here, before
    any batch is priced, and the digest travels with the evidence.
    """
    if not getattr(args, "sequential", False):
        return None
    if stopping_excluded(variant, case_name):
        return None
    floor = min(int(AMENDMENT_AGGREGATE_BATCHES), int(cap))
    return SequentialAdmissionPolicy(
        family_alpha=float(args.sequential_family_alpha),
        # Declared from the regime matrix, never counted from what ran: a
        # data-dependent K is not a Bonferroni correction.
        tests=2 * len(certification_cases(quick=False)) * 2,
        min_batches=min(int(MIN_PRODUCTION_RQMC_BATCHES), int(cap)),
        aggregate_floor_batches=floor,
        planned_batches=min(max(floor, 256), int(cap)),
        max_batches=int(cap),
        margin_fraction=float(args.sequential_margin),
    )


def certify_case(
    variant: str,
    case: CaseSpec,
    *,
    quick: bool,
    paths_per_batch: int,
    batches: int,
    seed: int,
    hedge_inception_spot: float,
    sequential_policy: Optional[SequentialAdmissionPolicy] = None,
    sequential_chunk_batches: int = 128,
    heston_spot_bridge_strata: int = HESTON_SPOT_BRIDGE_STRATA,
    heston_spot_bridge_dimensions: int = HESTON_SPOT_BRIDGE_DIMENSIONS,
    slv_spot_strata: int = SLV_SPOT_STRATA,
    slv_spot_antithetic: bool = SLV_SPOT_ANTITHETIC,
    slv_spot_bridge_strata: int = SLV_SPOT_BRIDGE_STRATA,
    slv_spot_bridge_dimensions: int = 1,
    slv_mid_control_seed: int = SLV_MID_CONTROL_SEED,
    heston_high_cell: Optional[dict] = None,
    rqmc_batch_workers: int = 1,
) -> dict:
    product = make_snowball(case, dense_ki=True)
    env = make_environment(case.spot, math.sqrt(max(case.params.v0, case.params.theta)))
    leverage = make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    ladders = grid_ladders(
        case.maturity,
        quick=quick,
        dense_ki_stencil=(case.name == "near_ki"),
    )
    bumps = QUICK_BUMP_LADDER if quick else FULL_BUMP_LADDER
    ladder_evidence, cache = _axis_ladder_rows(
        variant, case, product, env, leverage, ladders, bumps
    )
    target_key = (
        ladders["target"].n_x,
        ladders["target"].n_v,
        ladders["target"].n_t,
        SPOT_BUMP,
    )
    pde_target = cache[target_key]
    raw_pde_envelopes = _pde_envelope_components(ladder_evidence, pde_target)
    raw_pde_signed = _pde_signed_refinement_components(ladder_evidence, pde_target)

    if quick:
        target_substeps, fine_substeps = (1, 2)
    else:
        substep_profile = PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][case.name]
        target_substeps = int(substep_profile["target"])
        fine_substeps = int(substep_profile["fine"])
    # Both substep levels are projections of the same finest scrambled-Sobol
    # point set. This makes target-minus-fine a genuine CRN refinement
    # experiment instead of the difference of two noisy high-dimensional runs.
    probe_engine = make_mc_engine(
        variant,
        case,
        leverage,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=1,
        heston_spot_bridge_strata=heston_spot_bridge_strata,
        heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
        slv_spot_strata=slv_spot_strata,
        slv_spot_antithetic=slv_spot_antithetic,
        slv_spot_bridge_strata=slv_spot_bridge_strata,
        slv_spot_bridge_dimensions=slv_spot_bridge_dimensions,
    )
    _, contractual_dt, _, _ = probe_engine._build_time_grid(
        product, env, float(case.maturity)
    )
    target_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / target_substeps,
        target_substeps,
    )
    fine_dt = np.repeat(
        np.asarray(contractual_dt, dtype=float) / fine_substeps,
        fine_substeps,
    )
    target_provider, fine_provider = coupled_qe_providers(
        seed=seed,
        paths_per_batch=paths_per_batch,
        target_dt=target_dt,
        fine_dt=fine_dt,
        # Vol-model RQMC specs declare homogeneous spot scaling, so the paired
        # driver generates only the base path set and rescales it for S-h/S+h.
        # Caching three copies would retain unused high-dimensional arrays for
        # every batch and can exhaust memory in the long-maturity SLV cells.
        reuse_count=1,
    )
    scale = EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=hedge_inception_spot,
        study_notional=STUDY_NOTIONAL,
        hedge_multiplier=HEDGE_MULTIPLIER,
    )

    def run_level(level: str, first_batch: int, count: int):
        return paired_mc_reference(
            variant,
            case,
            product,
            env,
            leverage,
            paths_per_batch=paths_per_batch,
            batches=count,
            seed=seed,
            substeps=(target_substeps if level == "target" else fine_substeps),
            bump=SPOT_BUMP,
            qe_draw_provider=(
                target_provider if level == "target" else fine_provider
            ),
            heston_spot_bridge_strata=heston_spot_bridge_strata,
            heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
            slv_spot_strata=slv_spot_strata,
            slv_spot_antithetic=slv_spot_antithetic,
            slv_spot_bridge_strata=slv_spot_bridge_strata,
            slv_spot_bridge_dimensions=slv_spot_bridge_dimensions,
            rqmc_batch_workers=rqmc_batch_workers,
            first_batch=first_batch,
        )

    if sequential_policy is None:
        # Fixed allocation: the declared count, judged once. Retained for the
        # multilevel SLV cell, whose telescoping weights require exactly the
        # declared batch count, and for any run that opts out.
        reference = run_level("target", 0, batches)
        reference_fine = run_level("fine", 0, batches)
        sequential_record = None
    else:
        reference, reference_fine, sequential_record = gate_driven_reference_levels(
            run_level=run_level,
            policy=sequential_policy,
            scale=scale,
            pde_target=pde_target,
            raw_pde_envelopes=raw_pde_envelopes,
            bounds={
                "delta": DELTA_CELL_BOUND_CONTRACTS,
                "gamma": GAMMA_CELL_BOUND_CONTRACTS,
            },
            chunk_batches=sequential_chunk_batches,
            max_batches=batches,
        )
    estimator_metadata = {
        "name": "primary_conditional_rqmc",
        "primary_seed": int(seed),
        "outer_batches": int(reference.batches_used),
    }
    if not quick and variant == "heston_slv" and case.name in SLV_MULTILEVEL_CASES:
        if heston_high_cell is None:
            raise ValueError(
                f"heston_slv/{case.name}: matching Heston evidence is required"
            )
        reference, reference_fine, estimator_metadata = build_slv_multilevel_reference(
            case,
            product=product,
            env=env,
            leverage=leverage,
            primary_target=reference,
            primary_fine=reference_fine,
            target_dt=target_dt,
            fine_dt=fine_dt,
            target_substeps=target_substeps,
            fine_substeps=fine_substeps,
            heston_high_cell=heston_high_cell,
            primary_seed=seed,
            mid_seed=slv_mid_control_seed,
            slv_spot_bridge_strata=slv_spot_bridge_strata,
            slv_spot_bridge_dimensions=slv_spot_bridge_dimensions,
            rqmc_batch_workers=rqmc_batch_workers,
        )
    reference_batches = int(reference.batches_used)
    if reference_fine.batches_used != reference_batches:
        raise ValueError("target and fine reference batch counts do not match")
    pde_refinement = _pde_refinement_diagnostics(ladder_evidence, scale)
    certifications = {}
    batch_differences = {}
    for greek, cell_bound in (
        ("delta", DELTA_CELL_BOUND_CONTRACTS),
        ("gamma", GAMMA_CELL_BOUND_CONTRACTS),
    ):
        pde_value = float(pde_target[greek])
        # The fine level is the oracle estimate. The paired coarser level
        # estimator exists only to quantify remaining time-discretization
        # bias; using it as the point reference would double-count that bias.
        ref_value = float(getattr(reference_fine, greek))
        ref_se = float(getattr(reference_fine, f"{greek}_std_error"))
        difference = _economic_value(scale, greek, pde_value - ref_value)
        se_economic = abs(_economic_value(scale, greek, ref_se))
        pde_components = {
            key: abs(_economic_value(scale, greek, value))
            for key, value in raw_pde_envelopes[greek].items()
        }
        pde_signed_components = {
            axis: _economic_value(scale, greek, value)
            for axis, value in raw_pde_signed[greek].items()
        }
        target_batches = (
            reference.batch_delta if greek == "delta" else reference.batch_gamma
        )
        fine_batches = (
            reference_fine.batch_delta
            if greek == "delta"
            else reference_fine.batch_gamma
        )
        substep_batches = np.asarray(
            [
                _economic_value(scale, greek, float(target - fine))
                for target, fine in zip(target_batches, fine_batches)
            ],
            dtype=float,
        )
        substep_mean = float(np.mean(substep_batches))
        substep_se = float(np.std(substep_batches, ddof=1) / np.sqrt(reference_batches))
        component_critical = float(
            student_t.ppf(
                0.5 + 0.5 * STOCHASTIC_COMPONENT_CONFIDENCE,
                reference_batches - 1,
            )
        )
        substep_half_width = component_critical * substep_se
        ref_bias = abs(substep_mean) + substep_half_width
        verdict = certify_equivalence(
            difference,
            se_economic,
            cell_bound,
            reference_degrees_of_freedom=reference_batches - 1,
            pde_discretization_envelope=pde_components["total"],
            reference_bias_envelope=ref_bias,
            confidence=STOCHASTIC_COMPONENT_CONFIDENCE,
            label=f"{variant}/{case.name}/{greek}",
        )
        batch_differences[greek] = [
            _economic_value(scale, greek, pde_value - float(value))
            for value in fine_batches
        ]
        certifications[greek] = {
            "pde": pde_value,
            "reference": ref_value,
            "reference_standard_error": ref_se,
            "difference_economic_contracts": difference,
            "pde_envelope_contracts": pde_components,
            "pde_signed_refinement_contracts": pde_signed_components,
            "reference_substep_envelope_contracts": ref_bias,
            "reference_substep_components_contracts": {
                "paired_target_minus_fine_mean": substep_mean,
                "paired_standard_error": substep_se,
                "paired_half_width": substep_half_width,
                "absolute_bias_upper_bound": ref_bias,
                "component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
            },
            "reference_substep_batch_contracts": substep_batches.tolist(),
            "verdict": verdict.as_dict(),
            "economic_definition": (
                "futures contracts of delta"
                if greek == "delta"
                else "change in futures hedge contracts for a 1% spot move"
            ),
        }

    diagnostic_engine = make_pde_engine(variant, case, ladders["target"], leverage)
    core = diagnostic_engine._make_core(product, env, case.maturity)
    diagnostics = core.variance_operator_diagnostics()
    pv_grid = GridPoint(200, 60, max(120, int(math.ceil(400 * case.maturity))))
    production_grid_policy = None
    if not quick:
        production_kwargs = dict(
            n_x=pv_grid.n_x,
            n_v=pv_grid.n_v,
            n_t=pv_grid.n_t,
            params=PDEParams(cache_enabled=False),
            **PRODUCTION_ENGINE_CONTROLS,
        )
        if variant == "heston":
            production_engine = HestonSnowballPDESolver(
                case.params, **production_kwargs
            )
        else:
            production_engine = HestonSLVSnowballPDESolver(
                case.params,
                leverage_surface=leverage,
                **production_kwargs,
            )
        production_grid_policy = production_engine.greek_time_grid_policy(product, env)
        resolved_candidate = GridPoint(
            int(production_grid_policy["resolved_n_x"]),
            int(production_grid_policy["resolved_n_v"]),
            int(production_grid_policy["resolved_n_t"]),
        )
        if resolved_candidate != ladders["target"]:
            raise ValueError(
                "production Greek policy does not resolve to the certified target "
                f"grid: {resolved_candidate.as_dict()} != "
                f"{ladders['target'].as_dict()}"
            )
    row = {
        "variant": variant,
        "case": case.as_dict(),
        "target_grid": ladders["target"].as_dict(),
        "production_pv_grid": pv_grid.as_dict(),
        "production_greek_grid_policy": production_grid_policy,
        "economic_scale": scale.as_dict(),
        "finite_bump_semantics": (
            "barrier-adjacent values are h-width hedge exposures, not a claim "
            "that a pointwise classical derivative exists"
        ),
        "pde_ladders": ladder_evidence,
        "pde_refinement": pde_refinement,
        "variance_operator": diagnostics,
        "reference": {
            "scheme": (
                "QE-M paired randomized Sobol with exact affine spot-factor "
                "conditioning"
                if variant == "heston"
                else "QE-M paired randomized Sobol with antithetic terminal and "
                "midpoint Brownian-bridge stratification plus path-frozen "
                "leverage conditional control"
            ),
            "seed": int(seed),
            "estimator": estimator_metadata,
            "primary": "fine",
            "target_substeps_per_interval": target_substeps,
            "fine_substeps_per_interval": fine_substeps,
            "heston_spot_bridge_strata": (
                int(heston_spot_bridge_strata) if variant == "heston" else None
            ),
            "heston_spot_bridge_dimensions": (
                int(heston_spot_bridge_dimensions) if variant == "heston" else None
            ),
            "slv_spot_strata": (
                int(slv_spot_strata) if variant == "heston_slv" else None
            ),
            "slv_spot_antithetic": (
                bool(slv_spot_antithetic) if variant == "heston_slv" else None
            ),
            "slv_spot_bridge_strata": (
                int(slv_spot_bridge_strata) if variant == "heston_slv" else None
            ),
            "slv_spot_bridge_dimensions": (
                int(slv_spot_bridge_dimensions) if variant == "heston_slv" else None
            ),
            "slv_effective_spot_samples": (
                int(slv_spot_strata)
                * (2 if slv_spot_antithetic else 1)
                * int(slv_spot_bridge_strata)
                if variant == "heston_slv"
                else None
            ),
            "batch_workers": int(rqmc_batch_workers),
            "target": reference.as_dict(include_batches=True),
            "fine": reference_fine.as_dict(include_batches=True),
        },
        "certifications": certifications,
        "batch_difference_contracts": batch_differences,
        # Present only when the cell stopped on its gate rather than spending a
        # declared allocation. It records the policy the stop was judged under,
        # because a data-dependent stop is interpretable only against the
        # anytime-valid width that licensed it.
        "sequential_stopping": sequential_record,
    }
    row["status"] = (
        EquivalenceStatus.PASS.value
        if diagnostics["monotone"]
        and pde_refinement["status"] == EquivalenceStatus.PASS.value
        and all(
            item["verdict"]["status"] == EquivalenceStatus.PASS.value
            for item in certifications.values()
        )
        else (
            EquivalenceStatus.FAIL.value
            if (not diagnostics["monotone"])
            or pde_refinement["status"] == EquivalenceStatus.FAIL.value
            or any(
                item["verdict"]["status"] == EquivalenceStatus.FAIL.value
                for item in certifications.values()
            )
            else EquivalenceStatus.INCONCLUSIVE.value
        )
    )
    return row


def rescore_serialized_cell(cell: dict, hedge_inception_spot: float) -> dict:
    """Recompute economic verdicts from serialized target/fine evidence."""
    scale = EconomicGreekScale(
        model_spot=float(cell["case"]["spot"]),
        hedge_inception_spot=float(hedge_inception_spot),
        study_notional=STUDY_NOTIONAL,
        hedge_multiplier=HEDGE_MULTIPLIER,
    )
    target = cell["reference"]["target"]
    fine = cell["reference"]["fine"]
    batches = int(target["batches_used"])
    target_batch_estimates = np.asarray(target["batch_estimates"], dtype=float)
    fine_batch_estimates = np.asarray(fine["batch_estimates"], dtype=float)
    if target_batch_estimates.shape != (batches, 5):
        raise ValueError(
            "saved target paired-RQMC batch estimates must have shape "
            f"({batches}, 5), got {target_batch_estimates.shape}"
        )
    if fine_batch_estimates.shape != (batches, 5):
        raise ValueError(
            "saved fine paired-RQMC batch estimates must have shape "
            f"({batches}, 5), got {fine_batch_estimates.shape}"
        )
    old_certifications = cell.get("certifications", {})
    pde_target = {
        greek: float(old_certifications[greek]["pde"]) for greek in ("delta", "gamma")
    }
    raw_pde_envelopes = _pde_envelope_components(cell["pde_ladders"], pde_target)
    raw_pde_signed = _pde_signed_refinement_components(cell["pde_ladders"], pde_target)
    pde_refinement = _pde_refinement_diagnostics(cell["pde_ladders"], scale)

    certifications = {}
    batch_differences = {}
    batch_column = {"delta": 3, "gamma": 4}
    for greek, cell_bound in (
        ("delta", DELTA_CELL_BOUND_CONTRACTS),
        ("gamma", GAMMA_CELL_BOUND_CONTRACTS),
    ):
        pde_value = pde_target[greek]
        ref_value = float(fine[greek])
        ref_se = float(fine[f"{greek}_std_error"])
        difference = _economic_value(scale, greek, pde_value - ref_value)
        se_economic = abs(_economic_value(scale, greek, ref_se))
        pde_components = {
            key: abs(_economic_value(scale, greek, value))
            for key, value in raw_pde_envelopes[greek].items()
        }
        pde_signed_components = {
            axis: _economic_value(scale, greek, value)
            for axis, value in raw_pde_signed[greek].items()
        }
        target_batches = target_batch_estimates[:, batch_column[greek]]
        fine_batches = fine_batch_estimates[:, batch_column[greek]]
        substep_batches = np.asarray(
            [
                _economic_value(scale, greek, float(target_value - fine_value))
                for target_value, fine_value in zip(target_batches, fine_batches)
            ],
            dtype=float,
        )
        substep_mean = float(np.mean(substep_batches))
        substep_se = float(np.std(substep_batches, ddof=1) / np.sqrt(batches))
        component_critical = float(
            student_t.ppf(
                0.5 + 0.5 * STOCHASTIC_COMPONENT_CONFIDENCE,
                batches - 1,
            )
        )
        substep_half_width = component_critical * substep_se
        ref_bias = abs(substep_mean) + substep_half_width
        verdict = certify_equivalence(
            difference,
            se_economic,
            cell_bound,
            reference_degrees_of_freedom=batches - 1,
            pde_discretization_envelope=pde_components["total"],
            reference_bias_envelope=ref_bias,
            confidence=STOCHASTIC_COMPONENT_CONFIDENCE,
            label=f"{cell['variant']}/{cell['case']['name']}/{greek}",
        )
        batch_differences[greek] = [
            _economic_value(scale, greek, pde_value - float(value))
            for value in fine_batches
        ]
        certifications[greek] = {
            "pde": pde_value,
            "reference": ref_value,
            "reference_standard_error": ref_se,
            "difference_economic_contracts": difference,
            "pde_envelope_contracts": pde_components,
            "pde_signed_refinement_contracts": pde_signed_components,
            "reference_substep_envelope_contracts": ref_bias,
            "reference_substep_components_contracts": {
                "paired_target_minus_fine_mean": substep_mean,
                "paired_standard_error": substep_se,
                "paired_half_width": substep_half_width,
                "absolute_bias_upper_bound": ref_bias,
                "component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
            },
            "reference_substep_batch_contracts": substep_batches.tolist(),
            "verdict": verdict.as_dict(),
            "economic_definition": (
                "futures contracts of delta"
                if greek == "delta"
                else "change in futures hedge contracts for a 1% spot move"
            ),
        }

    cell["economic_scale"] = scale.as_dict()
    cell["certifications"] = certifications
    cell["batch_difference_contracts"] = batch_differences
    cell["pde_refinement"] = pde_refinement
    cell["status"] = (
        EquivalenceStatus.PASS.value
        if cell["variance_operator"]["monotone"]
        and pde_refinement["status"] == EquivalenceStatus.PASS.value
        and all(
            item["verdict"]["status"] == EquivalenceStatus.PASS.value
            for item in certifications.values()
        )
        else (
            EquivalenceStatus.FAIL.value
            if (not cell["variance_operator"]["monotone"])
            or pde_refinement["status"] == EquivalenceStatus.FAIL.value
            or any(
                item["verdict"]["status"] == EquivalenceStatus.FAIL.value
                for item in certifications.values()
            )
            else EquivalenceStatus.INCONCLUSIVE.value
        )
    )
    return cell


def rescore_payload(payload: dict, hedge_inception_spot: float) -> dict:
    """Recompute economics from complete same-schema numerical evidence."""
    if payload.get("study") != "adi_2d_snowball_greek_certification":
        raise ValueError("certification study tag mismatch")
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            "same-schema evidence with saved target and fine batches is required "
            "for rescoring"
        )
    source_hash = payload.get("evidence_sha256")
    source_run_configuration_hash = payload.get("run_configuration_sha256")
    rescored = copy.deepcopy(payload)
    rescored.pop("evidence_sha256", None)
    rescored["schema_version"] = SCHEMA_VERSION
    rescored["created_at"] = datetime.now().astimezone().isoformat()
    rescored["python"] = platform.python_version()
    rescored["cells"] = [
        rescore_serialized_cell(cell, hedge_inception_spot)
        for cell in rescored.get("cells", [])
    ]
    variants = list(rescored.get("decisions", {}))
    if not variants:
        variants = list(dict.fromkeys(cell["variant"] for cell in rescored["cells"]))
    rescored["decisions"] = make_decisions(
        rescored["cells"],
        rescored.get("anchors", []),
        quick=bool(rescored.get("quick")),
        variants=variants,
        sampling_by_variant=rescored.get("sampling_by_variant"),
        heston_spot_bridge_profile_by_case=rescored.get("policy", {}).get(
            "heston_spot_bridge_profile_by_case"
        ),
        slv_spot_strata=int(rescored.get("policy", {}).get("slv_spot_strata", 0)),
        slv_spot_antithetic=bool(
            rescored.get("policy", {}).get("slv_spot_antithetic", False)
        ),
        slv_spot_bridge_strata=int(
            rescored.get("policy", {}).get("slv_spot_bridge_strata", 0)
        ),
        slv_spot_bridge_profile_by_case=rescored.get("policy", {}).get(
            "slv_spot_bridge_profile_by_case"
        ),
    )
    policy = dict(rescored.get("policy", {}))
    policy.update(
        {
            "hedge_inception_spot": float(hedge_inception_spot),
            "hedge_inception_spot_policy": (
                "strictest minimum from the frozen 27-inception MO cohort"
            ),
        }
    )
    rescored["policy"] = policy
    run_configuration = dict(rescored.get("run_configuration", {}))
    run_configuration["hedge_inception_spot"] = float(hedge_inception_spot)
    run_configuration[
        "rescore_source_run_configuration_sha256"
    ] = source_run_configuration_hash
    rescored["run_configuration"] = run_configuration
    rescored["run_configuration_sha256"] = _canonical_sha256(run_configuration)
    rescored["rescore_provenance"] = {
        "source_evidence_sha256": source_hash,
        "numerical_evidence_reused": True,
        "reason": ("separate normalized model spot from actual hedge inception spot"),
    }
    return rescored


def deterministic_anchors(*, quick: bool) -> list[dict]:
    """Reduction identities that do not rely on Monte Carlo as an oracle."""
    rows: list[dict] = []
    vanilla_grid = (100, 40, 50) if quick else (240, 100, 140)
    vanilla = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    env = make_environment(100.0, 0.20)
    analytic = HestonAnalyticalEngine(ORDINARY).calculate_greeks(vanilla, env)
    pde = HestonPDESolver(
        ORDINARY,
        n_x=vanilla_grid[0],
        n_v=vanilla_grid[1],
        n_t=vanilla_grid[2],
    ).calculate_greeks(vanilla, env)
    vanilla_bounds = {"price": 0.30, "delta": 0.015, "gamma": 0.002}
    vanilla_checks = {
        greek: certify_equivalence(
            float(pde[greek]) - float(analytic[greek]),
            0.0,
            bound,
            label=f"heston_vanilla/{greek}",
        ).as_dict()
        for greek, bound in vanilla_bounds.items()
    }
    rows.append(
        {
            "name": "heston_vanilla_analytical",
            "reference": "Heston semi-analytical",
            "pde": {key: float(pde[key]) for key in vanilla_bounds},
            "reference_values": {key: float(analytic[key]) for key in vanilla_bounds},
            "checks": vanilla_checks,
            "status": _combined_status(
                check["status"] for check in vanilla_checks.values()
            ),
        }
    )

    anchor_case = CaseSpec(
        "constant_variance_anchor",
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.0, rho=0.0),
        1.0,
        100.0,
        ("deterministic", "constant_variance"),
    )
    product = make_snowball(anchor_case, dense_ki=False)
    grid = grid_ladders(1.0, quick=quick)["target"]
    heston = central_bump_greeks(
        make_pde_engine("heston", anchor_case, grid, None),
        product,
        env,
        SPOT_BUMP,
    )
    bsm = central_bump_greeks(
        SnowballPDESolver(PDEParams(cache_enabled=False)),
        product,
        env,
        SPOT_BUMP,
    )
    quad = central_bump_greeks(
        SnowballQuadEngine(QuadParams(grid_points=301 if quick else 1001)),
        product,
        env,
        SPOT_BUMP,
    )
    checks = {}
    for reference_name, reference_values in (("bsm_pde", bsm), ("quad", quad)):
        checks[reference_name] = {
            greek: certify_equivalence(
                float(heston[greek]) - float(reference_values[greek]),
                0.0,
                {"price": 0.35, "delta": 0.02, "gamma": 0.003}[greek],
                label=f"constant_variance/{reference_name}/{greek}",
            ).as_dict()
            for greek in ("price", "delta", "gamma")
        }
    rows.append(
        {
            "name": "snowball_constant_variance_reduction",
            "reference": "1-D BSM PDE and QUAD",
            "pde": heston,
            "reference_values": {"bsm_pde": bsm, "quad": quad},
            "checks": checks,
            "status": _combined_status(
                check["status"] for group in checks.values() for check in group.values()
            ),
        }
    )

    unit_case = CaseSpec(
        "unit_leverage_anchor",
        ORDINARY,
        1.0,
        100.0,
        ("stochastic_variance", "unit_leverage"),
    )
    unit_product = make_snowball(unit_case, dense_ki=False)
    unit_heston = central_bump_greeks(
        make_pde_engine("heston", unit_case, grid, None),
        unit_product,
        env,
        SPOT_BUMP,
    )
    unit_slv = central_bump_greeks(
        make_pde_engine(
            "heston_slv",
            unit_case,
            grid,
            make_leverage_surface(1.0, unit=True),
        ),
        unit_product,
        env,
        SPOT_BUMP,
    )
    unit_checks = {
        greek: certify_equivalence(
            float(unit_slv[greek]) - float(unit_heston[greek]),
            0.0,
            {"price": 1e-8, "delta": 1e-8, "gamma": 1e-9}[greek],
            label=f"unit_leverage/{greek}",
        ).as_dict()
        for greek in ("price", "delta", "gamma")
    }
    rows.append(
        {
            "name": "slv_unit_leverage_reduction",
            "reference": "Heston 2-D ADI",
            "pde": unit_slv,
            "reference_values": unit_heston,
            "checks": unit_checks,
            "status": _combined_status(
                check["status"] for check in unit_checks.values()
            ),
        }
    )

    deterministic_case = CaseSpec(
        "deterministic_variance_anchor",
        HestonParams(v0=0.09, kappa=1.5, theta=0.0225, sigma=0.0, rho=0.0),
        1.0,
        100.0,
        ("deterministic", "time_dependent_variance"),
    )
    deterministic_product = make_snowball(deterministic_case, dense_ki=False)
    det_env = make_environment(100.0, 0.25)
    time_grid = np.linspace(0.0, 1.0, 65 if quick else 257)
    variance = deterministic_case.params.theta + (
        deterministic_case.params.v0 - deterministic_case.params.theta
    ) * np.exp(-deterministic_case.params.kappa * time_grid)
    local_vol = LocalVolSurface(
        strike_grid=np.array([40.0, 180.0]),
        time_grid=time_grid,
        lv_grid=np.repeat(np.sqrt(variance)[:, None], 2, axis=1),
    )
    heston_det = central_bump_greeks(
        make_pde_engine("heston", deterministic_case, grid, None),
        deterministic_product,
        det_env,
        SPOT_BUMP,
    )
    lv_det = central_bump_greeks(
        LocalVolSnowballPDESolver(
            PDEParams(cache_enabled=False),
            local_vol_surface=local_vol,
        ),
        deterministic_product,
        det_env,
        SPOT_BUMP,
    )
    det_checks = {
        greek: certify_equivalence(
            float(heston_det[greek]) - float(lv_det[greek]),
            0.0,
            {"price": 0.50, "delta": 0.03, "gamma": 0.005}[greek],
            label=f"deterministic_variance/{greek}",
        ).as_dict()
        for greek in ("price", "delta", "gamma")
    }
    rows.append(
        {
            "name": "snowball_deterministic_variance_reduction",
            "reference": "1-D time-dependent local-vol PDE",
            "pde": heston_det,
            "reference_values": lv_det,
            "checks": det_checks,
            "status": _combined_status(
                check["status"] for check in det_checks.values()
            ),
        }
    )
    return rows


def _combined_status(statuses: Iterable[str]) -> str:
    values = list(statuses)
    if any(value == EquivalenceStatus.FAIL.value for value in values):
        return EquivalenceStatus.FAIL.value
    if values and all(value == EquivalenceStatus.PASS.value for value in values):
        return EquivalenceStatus.PASS.value
    return EquivalenceStatus.INCONCLUSIVE.value


def _sampling_batches_for_case(sampling: dict, variant: str, case_name: str) -> int:
    """Resolve the declared scramble count for one regime cell."""
    batches_by_case = sampling.get("batches_by_case")
    if isinstance(batches_by_case, dict) and case_name in batches_by_case:
        return int(batches_by_case[case_name])
    return int(sampling.get("batches", 0))


def _sampling_primary_batches_for_case(
    sampling: dict, variant: str, case_name: str
) -> int:
    """Resolve work-level batches before any multilevel outer grouping."""
    primary_by_case = sampling.get("primary_batches_by_case")
    if isinstance(primary_by_case, dict) and case_name in primary_by_case:
        return int(primary_by_case[case_name])
    return _sampling_batches_for_case(sampling, variant, case_name)


def _normalized_batches_by_case(sampling: dict) -> dict[str, int]:
    batches_by_case = sampling.get("batches_by_case")
    if not isinstance(batches_by_case, dict):
        return {}
    try:
        return {str(name): int(value) for name, value in batches_by_case.items()}
    except (TypeError, ValueError):
        return {}


def _normalized_primary_batches_by_case(sampling: dict) -> dict[str, int]:
    primary = sampling.get("primary_batches_by_case")
    if not isinstance(primary, dict):
        return {}
    try:
        return {str(name): int(value) for name, value in primary.items()}
    except (TypeError, ValueError):
        return {}


def make_decisions(
    rows: Sequence[dict],
    anchors: Sequence[dict],
    *,
    quick: bool,
    variants: Sequence[str],
    sampling_by_variant: Optional[dict] = None,
    heston_spot_bridge_profile_by_case: Optional[dict] = None,
    slv_spot_strata: int = SLV_SPOT_STRATA,
    slv_spot_antithetic: bool = SLV_SPOT_ANTITHETIC,
    slv_spot_bridge_strata: int = SLV_SPOT_BRIDGE_STRATA,
    slv_spot_bridge_profile_by_case: Optional[dict] = None,
) -> dict:
    anchor_status = _combined_status(anchor["status"] for anchor in anchors)
    anchor_names = {anchor.get("name") for anchor in anchors}
    missing_anchors = sorted(REQUIRED_ANCHOR_NAMES - anchor_names)
    required_cases = {case.name for case in certification_cases(quick=False)}
    decisions = {}
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        present_cases = {row.get("case", {}).get("name") for row in variant_rows}
        missing_cases = sorted(required_cases - present_cases)
        batch_counts_by_case = {
            str(row.get("case", {}).get("name")): len(
                row.get("batch_difference_contracts", {}).get("delta", [])
            )
            for row in variant_rows
        }
        common_batches = min(batch_counts_by_case.values(), default=0)
        sampling_complete = bool(variant_rows) and (
            common_batches >= MIN_PRODUCTION_RQMC_BATCHES
        )
        if sampling_by_variant is not None:
            sampling = sampling_by_variant.get(variant, {})
            minimum_paths = (
                PRODUCTION_HESTON_PATHS_PER_BATCH
                if variant == "heston"
                else PRODUCTION_SLV_PATHS_PER_BATCH
            )
            sampling_complete = (
                sampling_complete
                and int(sampling.get("paths_per_batch", 0)) >= minimum_paths
            )
            if variant == "heston":
                declared_profile = heston_spot_bridge_profile_by_case or {}
                sampling_complete = sampling_complete and (
                    int(sampling.get("batches", 0)) == PRODUCTION_HESTON_BATCHES
                    and _normalized_batches_by_case(sampling)
                    == PRODUCTION_HESTON_BATCHES_BY_CASE
                    and all(
                        actual_batches
                        == _sampling_batches_for_case(sampling, variant, case_name)
                        for case_name, actual_batches in (batch_counts_by_case.items())
                    )
                    and declared_profile == HESTON_SPOT_BRIDGE_PROFILE_BY_CASE
                    and all(
                        {
                            "strata": int(
                                row.get("reference", {}).get(
                                    "heston_spot_bridge_strata", 0
                                )
                            ),
                            "dimensions": int(
                                row.get("reference", {}).get(
                                    "heston_spot_bridge_dimensions", 0
                                )
                            ),
                        }
                        == HESTON_SPOT_BRIDGE_PROFILE_BY_CASE.get(
                            row.get("case", {}).get("name")
                        )
                        for row in variant_rows
                    )
                )
            else:
                declared_slv_profile = slv_spot_bridge_profile_by_case or {}
                sampling_complete = sampling_complete and (
                    int(sampling.get("batches", 0)) == PRODUCTION_SLV_BATCHES
                    and _normalized_batches_by_case(sampling)
                    == PRODUCTION_SLV_BATCHES_BY_CASE
                    and _normalized_primary_batches_by_case(sampling)
                    == PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE
                    and all(
                        actual_batches
                        == _sampling_batches_for_case(sampling, variant, case_name)
                        for case_name, actual_batches in (batch_counts_by_case.items())
                    )
                    and int(slv_spot_strata) == SLV_SPOT_STRATA
                    and bool(slv_spot_antithetic) == SLV_SPOT_ANTITHETIC
                    and int(slv_spot_bridge_strata) == SLV_SPOT_BRIDGE_STRATA
                    and declared_slv_profile == SLV_SPOT_BRIDGE_PROFILE_BY_CASE
                    and all(
                        {
                            "strata": int(
                                row.get("reference", {}).get(
                                    "slv_spot_bridge_strata", 0
                                )
                            ),
                            "dimensions": int(
                                row.get("reference", {}).get(
                                    "slv_spot_bridge_dimensions", 0
                                )
                            ),
                        }
                        == SLV_SPOT_BRIDGE_PROFILE_BY_CASE.get(
                            row.get("case", {}).get("name")
                        )
                        for row in variant_rows
                    )
                    and all(
                        (
                            row.get("reference", {}).get("estimator", {}).get("name")
                            == (
                                "three_level_frozen_slv_heston_control"
                                if not quick
                                and row.get("case", {}).get("name")
                                in SLV_MULTILEVEL_CASES
                                else "primary_conditional_rqmc"
                            )
                        )
                        for row in variant_rows
                    )
                )
            sampling_complete = sampling_complete and all(
                {
                    "target": int(
                        row.get("reference", {}).get("target_substeps_per_interval", 0)
                    ),
                    "fine": int(
                        row.get("reference", {}).get("fine_substeps_per_interval", 0)
                    ),
                }
                == PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant].get(
                    row.get("case", {}).get("name")
                )
                for row in variant_rows
            )
        evidence_complete = (
            not missing_anchors and not missing_cases and sampling_complete
        )
        cell_status = _combined_status(row["status"] for row in variant_rows)
        def strided_aggregate_rows(values) -> np.ndarray:
            # Strided pooling: output row j averages an over-allocated cell's
            # interleaved scrambles {j, j+m, j+2m, ...} (m = common_batches), so
            # scramble j of EVERY cell lands in output row j. Same-scramble
            # cross-case CRN covariance therefore stays inside one output row,
            # the output rows remain mutually independent, and -- unlike the
            # truncation this replaces -- every banked scramble reaches the
            # aggregate gate. Non-divisible counts fail closed rather than
            # silently dropping rows; a run whose stopping produced ragged
            # counts has no declared alignment and may not publish one.
            # Measured comparison against truncation and consecutive pooling:
            # docs/adi2d-greek-perf/probes/probe_crn_strided_alignment.py.
            banked = np.asarray(values, dtype=float)
            if banked.size % common_batches:
                raise ValueError(
                    f"{variant}: {banked.size} banked scrambles do not divide "
                    f"into {common_batches} common aggregate rows; strided "
                    "pooling needs a whole number of interleaved groups"
                )
            return banked.reshape(
                banked.size // common_batches, common_batches
            ).mean(axis=0)

        delta_batches = (
            np.asarray(
                [
                    strided_aggregate_rows(
                        row["batch_difference_contracts"]["delta"]
                    )
                    for row in variant_rows
                ],
                dtype=float,
            )
            if common_batches > 0
            else np.asarray([], dtype=float)
        )
        if delta_batches.ndim == 2 and delta_batches.shape[0] > 0:
            # Mean across cases *within* each output row preserves cross-case
            # covariance before the outer standard error is measured.
            bias_batches = np.mean(delta_batches, axis=0)
            pde_signed_axes = {
                axis: float(
                    np.mean(
                        [
                            row["certifications"]["delta"][
                                "pde_signed_refinement_contracts"
                            ][axis]
                            for row in variant_rows
                        ]
                    )
                )
                for axis in ("n_x", "n_v", "n_t")
            }
            pde_bias_envelope = float(
                sum(abs(value) for value in pde_signed_axes.values())
            )
            substep_matrix = np.asarray(
                [
                    strided_aggregate_rows(
                        row["certifications"]["delta"][
                            "reference_substep_batch_contracts"
                        ]
                    )
                    for row in variant_rows
                ],
                dtype=float,
            )
            aggregate_substep_batches = np.mean(substep_matrix, axis=0)
            aggregate_substep_mean = float(np.mean(aggregate_substep_batches))
            aggregate_substep_se = float(
                np.std(aggregate_substep_batches, ddof=1)
                / np.sqrt(aggregate_substep_batches.size)
            )
            aggregate_substep_half_width = float(
                student_t.ppf(
                    0.5 + 0.5 * STOCHASTIC_COMPONENT_CONFIDENCE,
                    aggregate_substep_batches.size - 1,
                )
                * aggregate_substep_se
            )
            reference_bias_envelope = (
                abs(aggregate_substep_mean) + aggregate_substep_half_width
            )
            bias = certify_signed_bias_from_batches(
                bias_batches,
                DELTA_BIAS_BOUND_CONTRACTS,
                pde_discretization_envelope=pde_bias_envelope,
                reference_bias_envelope=reference_bias_envelope,
                confidence=STOCHASTIC_COMPONENT_CONFIDENCE,
                label=f"{variant}/mean_signed_delta_bias",
            ).as_dict()
            bias["aggregate_pde_signed_refinement_contracts"] = pde_signed_axes
            bias["aggregate_reference_substep"] = {
                "mean": aggregate_substep_mean,
                "standard_error": aggregate_substep_se,
                "half_width": aggregate_substep_half_width,
                "absolute_bias_upper_bound": reference_bias_envelope,
                "component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
            }
            bias["aggregate_common_scrambles"] = common_batches
            # The alignment is part of the declared estimator, so it is stated
            # in the artifact and checked by `validate_payload` -- the published
            # truncated gate of 2026-08-17 showed an undeclared choice here can
            # decide admission by itself.
            bias["aggregate_alignment"] = "strided_pooled"
            bias["aggregate_batch_counts_by_case"] = batch_counts_by_case
        else:
            bias = certify_equivalence(
                float("nan"),
                float("nan"),
                DELTA_BIAS_BOUND_CONTRACTS,
                label=f"{variant}/mean_signed_delta_bias",
            ).as_dict()

        admissible = (
            not quick
            and evidence_complete
            and anchor_status == EquivalenceStatus.PASS.value
            and cell_status == EquivalenceStatus.PASS.value
            and bias["status"] == EquivalenceStatus.PASS.value
        )
        route = "pde" if admissible else "excluded_greek_unresolved"
        reasons = []
        if quick:
            reasons.append("quick profile is plumbing-only and cannot admit production")
        if missing_anchors:
            reasons.append(f"missing deterministic anchors: {missing_anchors}")
        if missing_cases:
            reasons.append(f"missing regime cells: {missing_cases}")
        if not sampling_complete:
            if variant == "heston":
                reasons.append(
                    "production sampling requires at least "
                    f"{PRODUCTION_HESTON_PATHS_PER_BATCH} paths/batch × "
                    f"{PRODUCTION_HESTON_BATCHES} common scrambles, "
                    f"near_ki={PRODUCTION_HESTON_BATCHES_BY_CASE['near_ki']}, "
                    "and the declared case-specific bridge profile"
                )
            else:
                reasons.append(
                    "production sampling requires at least "
                    f"{PRODUCTION_SLV_PATHS_PER_BATCH} paths/batch × "
                    f"{PRODUCTION_SLV_BATCHES} common scrambles, output map="
                    f"{PRODUCTION_SLV_BATCHES_BY_CASE}, primary map="
                    f"{PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE}, and terminal "
                    f"strata={SLV_SPOT_STRATA}, antithetic="
                    f"{SLV_SPOT_ANTITHETIC}, midpoint strata="
                    f"{SLV_SPOT_BRIDGE_STRATA}"
                )
        if anchor_status != EquivalenceStatus.PASS.value:
            reasons.append(f"deterministic anchors are {anchor_status}")
        if cell_status != EquivalenceStatus.PASS.value:
            reasons.append(f"one or more regime cells are {cell_status}")
        if bias["status"] != EquivalenceStatus.PASS.value:
            reasons.append(f"mean signed delta bias is {bias['status']}")
        decisions[variant] = {
            "route": route,
            "cell_status": cell_status,
            "anchor_status": anchor_status,
            "evidence_complete": evidence_complete,
            "missing_anchors": missing_anchors,
            "missing_cases": missing_cases,
            "sampling_complete": sampling_complete,
            "aggregate_common_scrambles": common_batches,
            "delta_bias": bias,
            "reason": "; ".join(reasons) if reasons else "all certification gates pass",
        }
    return decisions


def make_amendment_decisions(
    cells: Sequence[dict],
    anchors: Sequence[dict],
    parent_decisions: dict,
) -> tuple[dict, dict]:
    """Route a schema-11 certificate using two independent SLV cohorts.

    Five carried cells share the schema-9 scramble family and the two
    replacements share the held-out schema-11 family.  We preserve covariance
    within each family and add variances, never rows, across the independent
    families.
    """
    cells_by_key = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell for cell in cells
    }
    required_cases = {case.name for case in certification_cases(quick=False)}
    expected_keys = {
        (variant, case_name)
        for variant in ("heston", "heston_slv")
        for case_name in required_cases
    }
    if len(cells) != len(expected_keys) or set(cells_by_key) != expected_keys:
        raise ValueError("amendment regime matrix is incomplete or duplicated")
    if {anchor.get("name") for anchor in anchors} != REQUIRED_ANCHOR_NAMES or any(
        anchor.get("status") != EquivalenceStatus.PASS.value for anchor in anchors
    ):
        raise ValueError("amendment requires all parent anchors to PASS")

    heston_decision = copy.deepcopy(parent_decisions.get("heston", {}))
    if (
        heston_decision.get("route") != "pde"
        or heston_decision.get("evidence_complete") is not True
    ):
        raise ValueError("parent Heston decision is not admissible")
    heston_decision["certification_source"] = "schema9_parent"
    heston_decision["parent_evidence_sha256"] = PARENT_EVIDENCE_SHA256

    slv_rows = [
        cells_by_key[("heston_slv", case.name)]
        for case in certification_cases(quick=False)
    ]
    carried_rows = [
        row for row in slv_rows if row["case"]["name"] in AMENDMENT_CARRIED_SLV_CASES
    ]
    replacement_rows = [
        row for row in slv_rows if row["case"]["name"] in AMENDMENT_REPLACEMENT_CASES
    ]
    if len(carried_rows) != len(AMENDMENT_CARRIED_SLV_CASES) or len(
        replacement_rows
    ) != len(AMENDMENT_REPLACEMENT_CASES):
        raise ValueError("amendment SLV cohort membership mismatch")
    if any(row.get("status") != EquivalenceStatus.PASS.value for row in slv_rows):
        cell_status = _combined_status(row.get("status", "") for row in slv_rows)
    else:
        cell_status = EquivalenceStatus.PASS.value

    def cohort_contribution(
        rows: Sequence[dict], key_path: tuple[str, ...]
    ) -> np.ndarray:
        arrays = []
        for row in rows:
            value = row
            for key in key_path:
                value = value[key]
            values = np.asarray(value[:AMENDMENT_AGGREGATE_BATCHES], dtype=float)
            if values.shape != (AMENDMENT_AGGREGATE_BATCHES,) or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"invalid amendment cohort rows for {row['case']['name']}"
                )
            arrays.append(values)
        # Each cohort contributes its subset's sum divided by the full seven-
        # case denominator. Summing cohort means therefore equals the fleet
        # mean while retaining all within-family covariance.
        return np.sum(np.asarray(arrays), axis=0) / len(slv_rows)

    delta_cohorts = [
        cohort_contribution(
            carried_rows,
            ("batch_difference_contracts", "delta"),
        ),
        cohort_contribution(
            replacement_rows,
            ("batch_difference_contracts", "delta"),
        ),
    ]
    substep_cohorts = [
        cohort_contribution(
            carried_rows,
            (
                "certifications",
                "delta",
                "reference_substep_batch_contracts",
            ),
        ),
        cohort_contribution(
            replacement_rows,
            (
                "certifications",
                "delta",
                "reference_substep_batch_contracts",
            ),
        ),
    ]
    pde_signed_axes = {
        axis: float(
            np.mean(
                [
                    row["certifications"]["delta"]["pde_signed_refinement_contracts"][
                        axis
                    ]
                    for row in slv_rows
                ]
            )
        )
        for axis in ("n_x", "n_v", "n_t")
    }
    pde_bias_envelope = float(sum(abs(value) for value in pde_signed_axes.values()))
    substep_summary = summarize_independent_cohort_means(
        substep_cohorts,
        confidence=STOCHASTIC_COMPONENT_CONFIDENCE,
    )
    reference_bias_envelope = abs(substep_summary.estimate) + substep_summary.half_width
    bias = certify_signed_bias_from_independent_cohorts(
        delta_cohorts,
        DELTA_BIAS_BOUND_CONTRACTS,
        pde_discretization_envelope=pde_bias_envelope,
        reference_bias_envelope=reference_bias_envelope,
        confidence=STOCHASTIC_COMPONENT_CONFIDENCE,
        label="heston_slv/mean_signed_delta_bias",
    ).as_dict()
    bias["aggregate_pde_signed_refinement_contracts"] = pde_signed_axes
    bias["aggregate_reference_substep"] = {
        **substep_summary.as_dict(),
        "absolute_bias_upper_bound": reference_bias_envelope,
        "component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
    }
    bias["aggregate_independent_cohorts"] = [
        {
            "name": "schema9_parent",
            "seed": PARENT_SEED,
            "cases": sorted(AMENDMENT_CARRIED_SLV_CASES),
            "batches": AMENDMENT_AGGREGATE_BATCHES,
            "contribution_denominator": len(slv_rows),
        },
        {
            "name": "schema11_replacements",
            "seed": SLV_PRIMARY_SEED,
            "cases": sorted(AMENDMENT_REPLACEMENT_CASES),
            "batches": AMENDMENT_AGGREGATE_BATCHES,
            "contribution_denominator": len(slv_rows),
        },
    ]

    evidence_complete = cell_status == EquivalenceStatus.PASS.value and all(
        len(row["batch_difference_contracts"]["delta"]) >= AMENDMENT_AGGREGATE_BATCHES
        for row in slv_rows
    )
    admissible = evidence_complete and bias["status"] == EquivalenceStatus.PASS.value
    reasons = []
    if cell_status != EquivalenceStatus.PASS.value:
        reasons.append(f"one or more regime cells are {cell_status}")
    if not evidence_complete:
        reasons.append("amendment cohort evidence is incomplete")
    if bias["status"] != EquivalenceStatus.PASS.value:
        reasons.append(f"mean signed delta bias is {bias['status']}")
    slv_decision = {
        "route": "pde" if admissible else "excluded_greek_unresolved",
        "cell_status": cell_status,
        "anchor_status": EquivalenceStatus.PASS.value,
        "evidence_complete": evidence_complete,
        "missing_anchors": [],
        "missing_cases": [],
        "sampling_complete": evidence_complete,
        "aggregate_common_scrambles": AMENDMENT_AGGREGATE_BATCHES,
        "aggregate_method": "sum_of_independent_cohort_means",
        "delta_bias": bias,
        "reason": "; ".join(reasons) if reasons else "all certification gates pass",
        "certification_source": "schema11_amendment",
        "parent_evidence_sha256": PARENT_EVIDENCE_SHA256,
    }
    cohort_metadata = {
        "method": "sum_of_independent_cohort_means",
        "common_batches_per_cohort": AMENDMENT_AGGREGATE_BATCHES,
        "cohorts": bias["aggregate_independent_cohorts"],
    }
    return {"heston": heston_decision, "heston_slv": slv_decision}, cohort_metadata


def render_markdown(payload: dict) -> str:
    def interval_text(verdict: dict) -> str:
        interval = verdict.get("interval", [float("nan"), float("nan")])
        return (
            f"{float(verdict.get('estimate_difference', float('nan'))):+.4f} "
            f"[{float(interval[0]):+.4f}, {float(interval[1]):+.4f}]"
        )

    sampling = payload.get("sampling_by_variant")
    if sampling:
        sampling_parts = []
        for variant, values in sampling.items():
            part = (
                f"{variant}={values['paths_per_batch']} paths/batch × "
                f"{values['batches']} common scrambles"
            )
            batches_by_case = values.get("batches_by_case")
            if isinstance(batches_by_case, dict):
                extensions = {
                    name: int(batches)
                    for name, batches in batches_by_case.items()
                    if int(batches) != int(values["batches"])
                }
                if extensions:
                    detail = ", ".join(
                        f"{name}={batches}" for name, batches in extensions.items()
                    )
                    part += f" ({detail})"
            primary_batches_by_case = values.get("primary_batches_by_case")
            if isinstance(primary_batches_by_case, dict):
                primary_extensions = {
                    name: int(batches)
                    for name, batches in primary_batches_by_case.items()
                    if int(batches) != int(values["batches"])
                }
                if primary_extensions:
                    detail = ", ".join(
                        f"{name}={batches}"
                        for name, batches in primary_extensions.items()
                    )
                    part += f"; primary work ({detail})"
            sampling_parts.append(part)
        sampling_text = "; ".join(sampling_parts)
    else:
        sampling_text = (
            f"{payload['paths_per_batch']} paths/batch × "
            f"{payload['batches']} scrambles"
        )
    profile = payload.get(
        "profile", "quick (non-production)" if payload["quick"] else "production"
    )
    runtime = payload.get("runtime_environment", {})
    controls = payload.get("policy", {}).get("production_engine_controls", {})
    provenance_by_key = {
        (row.get("variant"), row.get("case")): row.get("source", "unknown")
        for row in payload.get("cell_provenance", [])
    }
    lines = [
        "# ADI 2-D Snowball Greek certification",
        "",
        f"- Profile: `{profile}`",
        f"- Certification mode: `{payload.get('certification_mode', 'unknown')}`",
        f"- Held-out seeds: `{payload.get('reference_seeds', payload.get('seed'))}`",
        f"- RQMC: {sampling_text}",
        (
            "- Numerical runtime: "
            f"Python {runtime.get('python_version', 'unknown')}; "
            f"NumPy {runtime.get('numpy_version', 'unknown')}; "
            f"SciPy {runtime.get('scipy_version', 'unknown')}; "
            f"{runtime.get('platform', 'unknown')}"
        ),
        (
            "- Certified engine controls: `"
            + "`, `".join(f"{key}={value}" for key, value in controls.items())
            + "`"
            if controls
            else "- Certified engine controls: unavailable"
        ),
        f"- Hedge inception spot: `{payload['policy']['hedge_inception_spot']}` "
        "(actual index level; numerical cases are normalized)",
        "- Verdict: fine QE-M 97.5% Student-t CI + paired case-specific "
        "target→fine substep "
        "97.5% bias upper bound + separate `n_x`/`n_v`/`n_t` PDE envelopes",
        (
            "- QE-M target→fine substeps by variant/case: `"
            f"{payload['policy'].get('qe_substeps_by_variant_case')}`"
        ),
        (
            "- Memory-safe RQMC workers by variant/case: `"
            f"{payload['policy'].get('rqmc_batch_workers_by_variant_case')}`"
        ),
        (
            "- Heston oracle: exact terminal spot-factor integration; "
            "case-specific residual Brownian-bridge profile="
            f"{payload['policy'].get('heston_spot_bridge_profile_by_case')}"
        ),
        (
            "- SLV oracle: unbiased path-frozen-leverage/Heston multilevel "
            "conditional control in the declared cases; "
            f"terminal strata={payload['policy'].get('slv_spot_strata')}, "
            f"antithetic={payload['policy'].get('slv_spot_antithetic')}, "
            f"midpoint strata={payload['policy'].get('slv_spot_bridge_strata')}; "
            "case-specific bridge profile="
            f"{payload['policy'].get('slv_spot_bridge_profile_by_case')}; "
            "no PDE control variate; policy="
            f"`{payload['policy'].get('slv_multilevel_policy')}`"
        ),
        "- Simultaneous stochastic coverage: at least 95% by Bonferroni; the "
        "spot-bump ladder is a semantic diagnostic and is not counted as PDE error",
        "- Delta bound: 0.5 hedge contracts per cell; mean signed bias bound: 0.1 contracts",
        "- Gamma bound/unit: 0.5 change in futures hedge contracts under a 1% spot move",
    ]
    if payload.get("certification_mode") == CERTIFICATION_MODE_AMENDMENT:
        lines.extend(
            [
                "- Parent schema-9 evidence: "
                f"`{payload['parent_certificate']['evidence_sha256']}`",
                "- Amendment scope: carry all 7 Heston and 5 passed SLV "
                "cells; replace only SLV Near-KI and Low-Feller; Heston "
                "Near-KI 8→16 is reference-only and does not recertify Heston",
                "- Fleet bias estimator: sum of two independent cohort means; "
                "within-cohort covariance retained and cross-cohort covariance fixed at zero",
            ]
        )
    lines.extend(
        [
            "",
            "## Routing decision",
            "",
            "| Variant | Route | Cells | Mean signed delta bias: estimate [interval] | Status | Reason |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for variant, decision in payload["decisions"].items():
        lines.append(
            f"| {variant} | `{decision['route']}` | {decision['cell_status']} | "
            f"{interval_text(decision['delta_bias'])} | "
            f"{decision['delta_bias']['status']} | {decision['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic anchors",
            "",
            "| Anchor | Reference | Status |",
            "|---|---|---:|",
        ]
    )
    for anchor in payload["anchors"]:
        lines.append(
            f"| {anchor['name']} | {anchor['reference']} | {anchor['status']} |"
        )
    lines.extend(
        [
            "",
            "## Regime cells",
            "",
            "All differences and intervals below are in their economic hedge units.",
            "",
            "| Variant | Case | Source | Tags | target grid | v grid | monotone V | fallback rows | QE/reference | Axis refinement | Delta: estimate [interval] | Delta | Gamma: estimate [interval] | Gamma | Cell |",
            "|---|---|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cells"]:
        diagnostics = row["variance_operator"]
        delta = row["certifications"]["delta"]["verdict"]
        gamma = row["certifications"]["gamma"]["verdict"]
        grid = row["target_grid"]
        reference = row["reference"]
        estimator_name = reference.get("estimator", {}).get("name", "unknown")
        reference_summary = (
            f"{reference.get('target_substeps_per_interval')}→"
            f"{reference.get('fine_substeps_per_interval')}; "
            f"{reference.get('fine', {}).get('batches_used')} rows; "
            f"{estimator_name}"
        )
        lines.append(
            f"| {row['variant']} | {row['case']['name']} | "
            f"{provenance_by_key.get((row['variant'], row['case']['name']), 'current run')} | "
            f"{', '.join(row['case']['tags'])} | "
            f"{grid['n_x']}×{grid['n_v']}×{grid['n_t']} | "
            f"{diagnostics['variance_grid_mode']} | "
            f"{diagnostics['monotone']} | "
            f"{diagnostics['fallback_nodes']} | "
            f"{reference_summary} | "
            f"{row['pde_refinement']['status']} | "
            f"{interval_text(delta)} | {delta['status']} | "
            f"{interval_text(gamma)} | {gamma['status']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Cell interval decomposition",
            "",
            "The substep column is the signed case-specific target→fine mean "
            "± its 97.5% half-width; "
            "the verdict uses `abs(mean) + half-width`.",
            "",
            "| Variant | Case | Greek | PDE | fine QE-M | fine SE | PDE−ref contracts | PDE envelope | substep mean ± half-width | total radius | interval | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cells"]:
        for greek in ("delta", "gamma"):
            certification = row["certifications"][greek]
            component = certification["reference_substep_components_contracts"]
            verdict = certification["verdict"]
            lines.append(
                f"| {row['variant']} | {row['case']['name']} | {greek} | "
                f"{certification['pde']:+.8f} | "
                f"{certification['reference']:+.8f} | "
                f"{certification['reference_standard_error']:.3g} | "
                f"{certification['difference_economic_contracts']:+.4f} | "
                f"{certification['pde_envelope_contracts']['total']:.4f} | "
                f"{component['paired_target_minus_fine_mean']:+.4f} ± "
                f"{component['paired_half_width']:.4f} | "
                f"{verdict['total_uncertainty']:.4f} | "
                f"{interval_text(verdict)} | {verdict['status']} |"
            )
    lines.extend(
        [
            "",
            "## Separate PDE-axis refinement",
            "",
            "Each entry is coarse→target / target→fine in hedge contracts.",
            "",
            "| Variant | Case | Greek | n_x | n_v | n_t | Status |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cells"]:
        for greek in ("delta", "gamma"):
            refinement = row["pde_refinement"][greek]
            axis_text = {}
            for axis in ("n_x", "n_v", "n_t"):
                values = refinement["axes"][axis]
                axis_text[axis] = (
                    f"{values['coarse_to_target_contracts']:+.4f} / "
                    f"{values['target_to_fine_contracts']:+.4f} "
                    f"({values['status']})"
                )
            lines.append(
                f"| {row['variant']} | {row['case']['name']} | {greek} | "
                f"{axis_text['n_x']} | {axis_text['n_v']} | "
                f"{axis_text['n_t']} | {refinement['status']} |"
            )
    lines.extend(
        [
            "",
            "Barrier-adjacent Greeks are explicitly certified as finite-bump hedge "
            "exposures. The report does not assert existence of a classical pointwise "
            "derivative at a discontinuous monitoring boundary.",
            "",
            f"Evidence SHA-256: `{payload.get('evidence_sha256', 'unpublished')}`",
            f"Implementation SHA-256: `{payload.get('implementation_sha256', 'unknown')}`",
            f"Run-configuration SHA-256: `{payload.get('run_configuration_sha256', 'unknown')}`",
            f"Elapsed seconds: `{float(payload.get('elapsed_seconds', 0.0)):.3f}`",
            "",
        ]
    )
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def evidence_projection(value):
    """Remove run-clock metadata before hashing reproducible evidence."""
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
    unsigned = copy.deepcopy(payload)
    unsigned.pop("evidence_sha256", None)
    canonical = json.dumps(
        evidence_projection(unsigned),
        sort_keys=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cell_evidence_sha256(cell: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            evidence_projection(cell),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def parent_certificate_manifest() -> dict:
    """Return the immutable schema-9 parent identity expected by schema 11."""
    return {
        "schema_version": PARENT_SCHEMA_VERSION,
        "source_commit": PARENT_SOURCE_COMMIT,
        "seed": PARENT_SEED,
        "evidence_file_sha256": PARENT_EVIDENCE_FILE_SHA256,
        "decision_file_sha256": PARENT_DECISION_FILE_SHA256,
        "evidence_sha256": PARENT_EVIDENCE_SHA256,
        "decision_sha256": PARENT_DECISION_SHA256,
        "implementation_sha256": PARENT_IMPLEMENTATION_SHA256,
        "run_configuration_sha256": PARENT_RUN_CONFIGURATION_SHA256,
        "production_pde_compatibility_sha256": PARENT_PRODUCTION_PDE_SHA256,
        "anchors_sha256": PARENT_ANCHORS_SHA256,
        "carried_cell_sha256": PARENT_CARRIED_CELL_SHA256,
    }


def load_and_validate_parent_certificate(
    evidence_path: Path,
    decision_path: Path,
) -> tuple[dict, dict, dict]:
    """Load the exact schema-9 parent and fail closed on any drift.

    Both whole-file hashes and embedded canonical hashes are checked.  The
    carried rows are additionally re-audited for matrix completeness and PASS
    status, while the live production-PDE projection must match the parent.
    """
    try:
        evidence_bytes = Path(evidence_path).read_bytes()
        decision_bytes = Path(decision_path).read_bytes()
    except OSError as exc:
        raise ValueError("schema-9 parent certificate is not readable") from exc
    if hashlib.sha256(evidence_bytes).hexdigest() != PARENT_EVIDENCE_FILE_SHA256:
        raise ValueError("schema-9 parent evidence file hash mismatch")
    if hashlib.sha256(decision_bytes).hexdigest() != PARENT_DECISION_FILE_SHA256:
        raise ValueError("schema-9 parent decision file hash mismatch")
    try:
        evidence = json.loads(evidence_bytes)
        decision = json.loads(decision_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError("schema-9 parent certificate is not valid JSON") from exc

    if (
        evidence.get("schema_version") != PARENT_SCHEMA_VERSION
        or evidence.get("study") != "adi_2d_snowball_greek_certification"
        or evidence.get("quick") is not False
        or evidence.get("profile") != "production"
        or int(evidence.get("seed", -1)) != PARENT_SEED
    ):
        raise ValueError("schema-9 parent certificate metadata mismatch")
    if evidence.get("evidence_sha256") != PARENT_EVIDENCE_SHA256 or (
        _projected_evidence_sha256(evidence) != PARENT_EVIDENCE_SHA256
    ):
        raise ValueError("schema-9 parent canonical evidence hash mismatch")
    if evidence.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256:
        raise ValueError("schema-9 parent implementation hash mismatch")
    if (
        evidence.get("run_configuration_sha256") != PARENT_RUN_CONFIGURATION_SHA256
        or _canonical_sha256(evidence.get("run_configuration", {}))
        != PARENT_RUN_CONFIGURATION_SHA256
    ):
        raise ValueError("schema-9 parent run-configuration hash mismatch")

    unsigned_decision = dict(decision)
    embedded_decision_hash = unsigned_decision.pop("decision_sha256", None)
    if (
        embedded_decision_hash != PARENT_DECISION_SHA256
        or _canonical_sha256(unsigned_decision) != PARENT_DECISION_SHA256
        or decision.get("evidence_sha256") != PARENT_EVIDENCE_SHA256
        or decision.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256
        or decision.get("run_configuration_sha256") != PARENT_RUN_CONFIGURATION_SHA256
    ):
        raise ValueError("schema-9 parent decision provenance mismatch")
    if evidence.get("runtime_environment") != runtime_environment():
        raise ValueError("schema-9 parent numerical runtime mismatch")
    if (
        evidence.get("policy", {}).get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
        or evidence.get("run_configuration", {}).get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
    ):
        raise ValueError("schema-9 parent production controls mismatch")
    live_pde_hash = production_pde_compatibility_sha256()
    if live_pde_hash != PARENT_PRODUCTION_PDE_SHA256:
        raise ValueError(
            "live production PDE dependencies differ from the schema-9 parent"
        )

    expected_cases = {
        case.name: case.as_dict() for case in certification_cases(quick=False)
    }
    anchors = evidence.get("anchors", [])
    if (
        {anchor.get("name") for anchor in anchors} != REQUIRED_ANCHOR_NAMES
        or any(
            anchor.get("status") != EquivalenceStatus.PASS.value for anchor in anchors
        )
        or _canonical_sha256(anchors) != PARENT_ANCHORS_SHA256
    ):
        raise ValueError("schema-9 parent deterministic anchors are incomplete")
    cells = evidence.get("cells", [])
    cells_by_key = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell for cell in cells
    }
    expected_keys = {
        (variant, case_name)
        for variant in ("heston", "heston_slv")
        for case_name in expected_cases
    }
    if len(cells) != len(expected_keys) or set(cells_by_key) != expected_keys:
        raise ValueError("schema-9 parent regime matrix is incomplete or duplicated")
    for (variant, case_name), cell in cells_by_key.items():
        if cell.get("case") != expected_cases[case_name]:
            raise ValueError(f"schema-9 parent case drift: {variant}/{case_name}")
        expected_status = (
            EquivalenceStatus.INCONCLUSIVE.value
            if variant == "heston_slv" and case_name in AMENDMENT_REPLACEMENT_CASES
            else EquivalenceStatus.PASS.value
        )
        if cell.get("status") != expected_status:
            raise ValueError(f"schema-9 parent status mismatch: {variant}/{case_name}")
        carried_key = f"{variant}/{case_name}"
        if carried_key in PARENT_CARRIED_CELL_SHA256 and (
            _cell_evidence_sha256(cell) != PARENT_CARRIED_CELL_SHA256[carried_key]
        ):
            raise ValueError(f"schema-9 carried cell hash mismatch: {carried_key}")
    parent_decisions = evidence.get("decisions", {})
    if (
        parent_decisions != decision.get("decisions")
        or parent_decisions.get("heston", {}).get("route") != "pde"
        or parent_decisions.get("heston", {}).get("evidence_complete") is not True
        or parent_decisions.get("heston_slv", {}).get("route")
        != "excluded_greek_unresolved"
    ):
        raise ValueError("schema-9 parent routing decision mismatch")
    if evidence.get("sampling_by_variant") != {
        "heston": {
            "paths_per_batch": PRODUCTION_HESTON_PATHS_PER_BATCH,
            "batches": PRODUCTION_HESTON_BATCHES,
            "batches_by_case": PRODUCTION_HESTON_BATCHES_BY_CASE,
        },
        "heston_slv": {
            "paths_per_batch": PRODUCTION_SLV_PATHS_PER_BATCH,
            "batches": PRODUCTION_SLV_BATCHES,
        },
    }:
        raise ValueError("schema-9 parent sampling profile mismatch")
    return evidence, decision, parent_certificate_manifest()


def amendment_reference_seeds() -> dict:
    return {
        "parent_schema9": PARENT_SEED,
        "heston_near_ki_control": HESTON_REFERENCE_SEED,
        "heston_slv_primary": SLV_PRIMARY_SEED,
        "heston_slv_mid_control": SLV_MID_CONTROL_SEED,
    }


def amendment_sampling_by_variant() -> dict:
    """Describe the carried and replacement sampling as one fleet view."""
    slv_output = {
        case.name: (
            PRODUCTION_SLV_BATCHES_BY_CASE[case.name]
            if case.name in AMENDMENT_REPLACEMENT_CASES
            else PRODUCTION_SLV_BATCHES
        )
        for case in certification_cases(quick=False)
    }


def amendment_qe_substeps_by_variant_case() -> dict:
    return {
        "heston": {
            case.name: {"target": 4, "fine": 8}
            for case in certification_cases(quick=False)
        },
        "heston_slv": copy.deepcopy(PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE),
    }
    slv_primary = {
        case.name: (
            PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE[case.name]
            if case.name in AMENDMENT_REPLACEMENT_CASES
            else PRODUCTION_SLV_BATCHES
        )
        for case in certification_cases(quick=False)
    }
    return {
        "heston": {
            "source": "schema9_parent",
            "paths_per_batch": PRODUCTION_HESTON_PATHS_PER_BATCH,
            "batches": PRODUCTION_HESTON_BATCHES,
            "batches_by_case": PRODUCTION_HESTON_BATCHES_BY_CASE,
        },
        "heston_slv": {
            "source": "mixed_schema9_schema11",
            "paths_per_batch": PRODUCTION_SLV_PATHS_PER_BATCH,
            "batches": PRODUCTION_SLV_BATCHES,
            "batches_by_case": slv_output,
            "primary_batches_by_case": slv_primary,
        },
    }


def amendment_run_configuration(
    *,
    implementation_hash: str,
    runtime: dict,
    hedge_inception_spot: float,
) -> dict:
    cases = certification_cases(quick=False)
    return {
        "schema_version": SCHEMA_VERSION,
        "certification_mode": CERTIFICATION_MODE_AMENDMENT,
        "implementation_sha256": implementation_hash,
        "runtime_environment": runtime,
        "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        "production_pde_compatibility_sha256": (production_pde_compatibility_sha256()),
        "quick": False,
        "skip_anchors": False,
        "parent_certificate": parent_certificate_manifest(),
        "reference_seeds": amendment_reference_seeds(),
        "hedge_inception_spot": float(hedge_inception_spot),
        "variants": ["heston", "heston_slv"],
        "cases": [case.as_dict() for case in cases],
        "carried_cases": {
            "heston": [case.name for case in cases],
            "heston_slv": sorted(AMENDMENT_CARRIED_SLV_CASES),
        },
        "replacement_cases": {
            "heston_slv": sorted(AMENDMENT_REPLACEMENT_CASES),
        },
        "auxiliary_controls": {
            "heston_near_ki_8_16": {
                "purpose": "reference_only_slv_high_control",
                "paths_per_batch": PRODUCTION_HESTON_PATHS_PER_BATCH,
                "batches": PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"],
                "seed": HESTON_REFERENCE_SEED,
                "target_substeps_per_interval": 8,
                "fine_substeps_per_interval": 16,
                "heston_spot_bridge_profile": (
                    HESTON_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]
                ),
                "batch_workers": (
                    PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston"]["near_ki"]
                ),
            }
        },
        "sampling_by_variant": amendment_sampling_by_variant(),
        "spot_bump": SPOT_BUMP,
        "full_bump_ladder": list(FULL_BUMP_LADDER),
        "stochastic_component_confidence": (STOCHASTIC_COMPONENT_CONFIDENCE),
        "heston_spot_bridge_profile_by_case": (HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
        "slv_spot_bridge_profile_by_case": SLV_SPOT_BRIDGE_PROFILE_BY_CASE,
        "qe_substeps_by_variant_case": amendment_qe_substeps_by_variant_case(),
        "slv_multilevel_policy": {
            "cases": sorted(SLV_MULTILEVEL_CASES),
            "mid_paths_per_batch": SLV_MID_CONTROL_PATHS_PER_BATCH,
            "mid_batches": SLV_MID_CONTROL_BATCHES,
            "frozen_control_weight": SLV_FROZEN_CONTROL_WEIGHT,
            "heston_control_weight": SLV_HESTON_CONTROL_WEIGHT,
        },
        "slv_spot_strata": SLV_SPOT_STRATA,
        "slv_spot_antithetic": SLV_SPOT_ANTITHETIC,
        "slv_spot_bridge_strata": SLV_SPOT_BRIDGE_STRATA,
        "rqmc_batch_workers_by_variant_case": {
            "heston": {
                "near_ki": PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston"][
                    "near_ki"
                ]
            },
            "heston_slv": {
                case_name: PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston_slv"][
                    case_name
                ]
                for case_name in sorted(AMENDMENT_REPLACEMENT_CASES)
            },
        },
        "aggregate_cohort_policy": {
            "method": "sum_of_independent_cohort_means",
            "common_batches_per_cohort": AMENDMENT_AGGREGATE_BATCHES,
            "preserve_within_cohort_covariance": True,
            "cross_cohort_covariance": 0.0,
        },
    }


def amendment_cell_provenance(cells: Sequence[dict]) -> list[dict]:
    provenance = []
    for cell in cells:
        variant = str(cell["variant"])
        case_name = str(cell["case"]["name"])
        key = f"{variant}/{case_name}"
        source = (
            "schema11_replacement"
            if variant == "heston_slv" and case_name in AMENDMENT_REPLACEMENT_CASES
            else "schema9_parent"
        )
        row = {
            "variant": variant,
            "case": case_name,
            "source": source,
            "cell_evidence_sha256": _cell_evidence_sha256(cell),
        }
        if source == "schema9_parent":
            row["parent_evidence_sha256"] = PARENT_EVIDENCE_SHA256
            row["expected_parent_cell_sha256"] = PARENT_CARRIED_CELL_SHA256[key]
        provenance.append(row)
    return provenance


def _validate_amendment_payload(payload: dict) -> None:
    if payload.get("quick") is not False or payload.get("profile") != (
        "production incremental amendment"
    ):
        raise ValueError("schema-11 amendment must be production evidence")
    if payload.get("parent_certificate") != parent_certificate_manifest():
        raise ValueError("schema-11 parent manifest mismatch")
    if payload.get("reference_seeds") != amendment_reference_seeds():
        raise ValueError("schema-11 amendment seed profile mismatch")
    if payload.get("sampling_by_variant") != amendment_sampling_by_variant():
        raise ValueError("schema-11 amendment sampling profile mismatch")
    if (
        payload.get("production_pde_compatibility_sha256")
        != (PARENT_PRODUCTION_PDE_SHA256)
        or production_pde_compatibility_sha256() != PARENT_PRODUCTION_PDE_SHA256
    ):
        raise ValueError("schema-11 amendment production PDE compatibility mismatch")

    hedge_spot = float(
        payload.get("policy", {}).get("hedge_inception_spot", float("nan"))
    )
    if not np.isfinite(hedge_spot) or hedge_spot <= 0.0:
        raise ValueError("schema-11 amendment lacks a valid hedge inception spot")
    runtime = runtime_environment()
    implementation_hash = implementation_sha256()
    expected_run_configuration = amendment_run_configuration(
        implementation_hash=implementation_hash,
        runtime=runtime,
        hedge_inception_spot=hedge_spot,
    )
    if payload.get("runtime_environment") != runtime:
        raise ValueError("schema-11 amendment numerical runtime mismatch")
    if payload.get("implementation_sha256") != implementation_hash:
        raise ValueError("schema-11 amendment does not match the live implementation")
    if payload.get("run_configuration") != expected_run_configuration or (
        payload.get("run_configuration_sha256")
        != _canonical_sha256(expected_run_configuration)
    ):
        raise ValueError("schema-11 amendment run configuration mismatch")
    if payload.get("policy", {}).get("production_engine_controls") != (
        PRODUCTION_ENGINE_CONTROLS
    ):
        raise ValueError("schema-11 amendment production controls mismatch")

    anchors = payload.get("anchors", [])
    if _canonical_sha256(anchors) != PARENT_ANCHORS_SHA256:
        raise ValueError("schema-11 amendment parent anchors mismatch")
    cells = payload.get("cells", [])
    expected_cases = {
        case.name: case.as_dict() for case in certification_cases(quick=False)
    }
    cells_by_key = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell for cell in cells
    }
    expected_keys = {
        (variant, case_name)
        for variant in ("heston", "heston_slv")
        for case_name in expected_cases
    }
    if len(cells) != len(expected_keys) or set(cells_by_key) != expected_keys:
        raise ValueError("schema-11 amendment regime matrix mismatch")
    for (variant, case_name), cell in cells_by_key.items():
        if cell.get("case") != expected_cases[case_name]:
            raise ValueError(f"schema-11 amendment case drift: {variant}/{case_name}")
        key = f"{variant}/{case_name}"
        carried = not (
            variant == "heston_slv" and case_name in AMENDMENT_REPLACEMENT_CASES
        )
        if carried:
            if (
                cell.get("status") != EquivalenceStatus.PASS.value
                or _cell_evidence_sha256(cell) != PARENT_CARRIED_CELL_SHA256[key]
            ):
                raise ValueError(f"schema-11 carried cell mismatch: {key}")
            continue
        if cell.get("status") not in {status.value for status in EquivalenceStatus}:
            raise ValueError(f"schema-11 replacement status invalid: {key}")
        expected_target = grid_ladders(
            float(cell["case"]["maturity"]),
            quick=False,
            dense_ki_stencil=(case_name == "near_ki"),
        )["target"].as_dict()
        grid_policy = cell.get("production_greek_grid_policy", {})
        if (
            cell.get("target_grid") != expected_target
            or {
                "n_x": int(grid_policy.get("resolved_n_x", 0)),
                "n_v": int(grid_policy.get("resolved_n_v", 0)),
                "n_t": int(grid_policy.get("resolved_n_t", 0)),
            }
            != expected_target
        ):
            raise ValueError(f"schema-11 replacement PDE grid mismatch: {key}")
        reference = cell.get("reference", {})
        expected_profile = PRODUCTION_SLV_QE_SUBSTEPS_BY_CASE[case_name]
        if (
            int(reference.get("seed", -1)) != SLV_PRIMARY_SEED
            or reference.get("primary") != "fine"
            or int(reference.get("target_substeps_per_interval", 0))
            != expected_profile["target"]
            or int(reference.get("fine_substeps_per_interval", 0))
            != expected_profile["fine"]
            or int(reference.get("slv_spot_strata", 0)) != SLV_SPOT_STRATA
            or bool(reference.get("slv_spot_antithetic", False)) != SLV_SPOT_ANTITHETIC
            or int(reference.get("batch_workers", 0))
            != PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston_slv"][case_name]
            or {
                "strata": int(reference.get("slv_spot_bridge_strata", 0)),
                "dimensions": int(reference.get("slv_spot_bridge_dimensions", 0)),
            }
            != SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
        ):
            raise ValueError(f"schema-11 replacement profile mismatch: {key}")
        expected_batches = PRODUCTION_SLV_BATCHES_BY_CASE[case_name]
        for level in ("target", "fine"):
            result = paired_result_from_serialized(
                reference.get(level, {}),
                randomization_label=f"validation/{key}/{level}",
            )
            if result.batches_used != expected_batches:
                raise ValueError(f"schema-11 replacement batches mismatch: {key}")
        rescored = rescore_serialized_cell(copy.deepcopy(cell), hedge_spot)
        if (
            rescored["economic_scale"] != cell.get("economic_scale")
            or rescored["certifications"] != cell.get("certifications")
            or rescored["pde_refinement"] != cell.get("pde_refinement")
            or rescored["batch_difference_contracts"]
            != cell.get("batch_difference_contracts")
            or rescored["status"] != cell.get("status")
        ):
            raise ValueError(f"schema-11 replacement verdict mismatch: {key}")

    if payload.get("cell_provenance") != amendment_cell_provenance(cells):
        raise ValueError("schema-11 cell provenance mismatch")
    auxiliary = payload.get("auxiliary_controls", {}).get("heston_near_ki_8_16")
    if not isinstance(auxiliary, dict):
        raise ValueError("schema-11 Heston Near-KI auxiliary control is missing")
    auxiliary_reference = auxiliary.get("reference", {})
    if (
        auxiliary.get("variant") != "heston"
        or auxiliary.get("case") != expected_cases["near_ki"]
        or auxiliary.get("purpose") != "reference_only_slv_high_control"
        or int(auxiliary_reference.get("seed", -1)) != HESTON_REFERENCE_SEED
        or int(auxiliary_reference.get("target_substeps_per_interval", 0)) != 8
        or int(auxiliary_reference.get("fine_substeps_per_interval", 0)) != 16
        or int(auxiliary_reference.get("heston_spot_bridge_strata", 0))
        != HESTON_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]["strata"]
        or int(auxiliary_reference.get("heston_spot_bridge_dimensions", 0))
        != HESTON_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]["dimensions"]
        or int(auxiliary_reference.get("batch_workers", 0))
        != PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston"]["near_ki"]
    ):
        raise ValueError("schema-11 Heston auxiliary profile mismatch")
    for level in ("target", "fine"):
        result = paired_result_from_serialized(
            auxiliary_reference.get(level, {}),
            randomization_label=f"validation/heston-aux/{level}",
        )
        if (
            result.paths_per_batch != PRODUCTION_HESTON_PATHS_PER_BATCH
            or result.batches_used != PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"]
        ):
            raise ValueError("schema-11 Heston auxiliary sampling mismatch")
    near_ki = cells_by_key[("heston_slv", "near_ki")]
    estimator = near_ki.get("reference", {}).get("estimator", {})
    if (
        estimator.get("name") != "three_level_frozen_slv_heston_control"
        or estimator.get("identity") != "Y-a*F_low+a*F_high-b*H_low+b*H_high"
        or estimator.get("weights")
        != {
            "frozen_slv": SLV_FROZEN_CONTROL_WEIGHT,
            "heston": SLV_HESTON_CONTROL_WEIGHT,
        }
        or int(estimator.get("primary_seed", -1)) != SLV_PRIMARY_SEED
        or int(estimator.get("mid_control_seed", -1)) != SLV_MID_CONTROL_SEED
        or int(estimator.get("heston_high_seed", -1)) != HESTON_REFERENCE_SEED
        or estimator.get("heston_high_cell_sha256") != _cell_evidence_sha256(auxiliary)
    ):
        raise ValueError("schema-11 Near-KI multilevel provenance mismatch")
    if (
        int(estimator.get("outer_batches", 0)) != PRODUCTION_SLV_BATCHES
        or int(estimator.get("primary_paths_per_batch", 0))
        != PRODUCTION_SLV_PATHS_PER_BATCH
        or int(estimator.get("primary_batches", 0))
        != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["near_ki"]
        or int(estimator.get("mid_paths_per_batch", 0))
        != SLV_MID_CONTROL_PATHS_PER_BATCH
        or int(estimator.get("mid_batches", 0)) != SLV_MID_CONTROL_BATCHES
    ):
        raise ValueError("schema-11 Near-KI multilevel sampling mismatch")
    for level in ("target", "fine"):
        components = estimator.get("components", {}).get(level, {})
        primary_payload = components.get("primary_low", {})
        frozen_high_payload = components.get("frozen_high", {})
        heston_low_payload = components.get("heston_low", {})
        primary = paired_result_from_serialized(
            primary_payload,
            randomization_label=f"validation/near-ki/primary/{level}",
        )
        frozen_high = paired_result_from_serialized(
            frozen_high_payload,
            randomization_label=f"validation/near-ki/frozen-high/{level}",
        )
        heston_low = paired_result_from_serialized(
            heston_low_payload,
            randomization_label=f"validation/near-ki/heston-low/{level}",
        )
        heston_high = paired_result_from_serialized(
            auxiliary_reference[level],
            randomization_label=f"heston-high/near_ki/{level}",
        )
        primary_controls = np.asarray(
            primary_payload.get("control_batch_estimates"), dtype=float
        )
        frozen_high_controls = np.asarray(
            frozen_high_payload.get("control_batch_estimates"), dtype=float
        )
        if (
            primary.batches_used != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["near_ki"]
            or primary_controls.shape != (primary.batches_used, 5)
            or frozen_high.batches_used != SLV_MID_CONTROL_BATCHES
            or frozen_high_controls.shape != (SLV_MID_CONTROL_BATCHES, 5)
            or heston_low.batches_used != SLV_MID_CONTROL_BATCHES
            or not np.all(np.isfinite(primary_controls))
            or not np.all(np.isfinite(frozen_high_controls))
            or not np.array_equal(
                frozen_high_controls,
                np.asarray(heston_low.batch_estimates),
            )
            or frozen_high_payload.get("randomization_key")
            != heston_low_payload.get("randomization_key")
        ):
            raise ValueError("schema-11 Near-KI component coupling mismatch")
        primary = replace(
            primary,
            control_batch_estimates=primary_controls,
        )
        frozen_high = replace(
            frozen_high,
            control_batch_estimates=frozen_high_controls,
        )
        frozen_low = extract_embedded_conditional_control(primary)
        linked_high = components.get("heston_high_reference", {})
        if linked_high != {
            "variant": "heston",
            "case": "near_ki",
            "batches": PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"],
            "randomization_key": (
                f"heston-high/near_ki/{level}("
                f"{auxiliary_reference[level]['randomization_key']})"
            ),
        }:
            raise ValueError("schema-11 Near-KI Heston high link mismatch")
        recombined = combine_grouped_rqmc_components(
            (
                (1.0, primary),
                (-SLV_FROZEN_CONTROL_WEIGHT, frozen_low),
                (SLV_FROZEN_CONTROL_WEIGHT, frozen_high),
                (-SLV_HESTON_CONTROL_WEIGHT, heston_low),
                (SLV_HESTON_CONTROL_WEIGHT, heston_high),
            ),
            output_batches=PRODUCTION_SLV_BATCHES,
            estimator_label=f"validation/near-ki/{level}",
        )
        published = near_ki["reference"][level]
        if (
            not np.allclose(
                recombined.batch_estimates,
                np.asarray(published.get("batch_estimates"), dtype=float),
                rtol=1e-12,
                atol=1e-14,
            )
            or recombined.total_unique_paths
            != int(published.get("total_unique_paths", -1))
            or recombined.total_path_valuations
            != int(published.get("total_path_valuations", -1))
        ):
            raise ValueError("schema-11 Near-KI recombination mismatch")
    low_feller_estimator = (
        cells_by_key[("heston_slv", "low_feller")]
        .get("reference", {})
        .get("estimator", {})
    )
    if low_feller_estimator.get("name") != "primary_conditional_rqmc":
        raise ValueError("schema-11 Low-Feller estimator mismatch")
    low_reference = cells_by_key[("heston_slv", "low_feller")]["reference"]
    for level in ("target", "fine"):
        controls = np.asarray(
            low_reference[level].get("control_batch_estimates"), dtype=float
        )
        if controls.shape != (
            PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["low_feller"],
            5,
        ) or not np.all(np.isfinite(controls)):
            raise ValueError("schema-11 Low-Feller control rows mismatch")

    decisions = payload.get("decisions", {})
    parent_heston = copy.deepcopy(decisions.get("heston", {}))
    parent_heston.pop("certification_source", None)
    parent_heston.pop("parent_evidence_sha256", None)
    if _canonical_sha256(parent_heston) != PARENT_HESTON_DECISION_SHA256:
        raise ValueError("schema-11 carried Heston decision mismatch")
    expected_decisions, expected_cohorts = make_amendment_decisions(
        cells,
        anchors,
        {"heston": parent_heston},
    )
    if decisions != expected_decisions:
        raise ValueError("schema-11 amendment decisions do not match raw evidence")
    if payload.get("aggregate_cohorts") != expected_cohorts:
        raise ValueError("schema-11 amendment aggregate cohort metadata mismatch")


def validate_payload(payload: dict) -> None:
    """Fail closed before a malformed certification artifact is published."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certification schema version mismatch")
    if payload.get("study") != "adi_2d_snowball_greek_certification":
        raise ValueError("certification study tag mismatch")
    certification_mode = payload.get("certification_mode")
    if certification_mode == CERTIFICATION_MODE_AMENDMENT:
        _validate_amendment_payload(payload)
        return
    if certification_mode != CERTIFICATION_MODE_FULL:
        raise ValueError("certification mode must be explicit")
    if int(payload.get("batches", 0)) < 2:
        raise ValueError("certification requires at least two RQMC batches")
    allowed_statuses = {status.value for status in EquivalenceStatus}
    hedge_inception_spot = float(
        payload.get("policy", {}).get("hedge_inception_spot", float("nan"))
    )
    if not np.isfinite(hedge_inception_spot) or hedge_inception_spot <= 0.0:
        raise ValueError("certification lacks a valid hedge inception spot")
    for cell in payload.get("cells", []):
        variant = str(cell.get("variant"))
        if cell.get("status") not in allowed_statuses:
            raise ValueError("certification cell has an invalid status")
        diagnostics = cell.get("variance_operator", {})
        if diagnostics.get("monotone") is not True:
            # A non-monotone row may be useful diagnostic evidence, but it can
            # never be serialized as an apparently certifiable production cell.
            if cell.get("status") != EquivalenceStatus.FAIL.value:
                raise ValueError("non-monotone operator cell must be FAIL")
        randomization_key = (
            cell.get("reference", {}).get("target", {}).get("randomization_key")
        )
        if not randomization_key:
            raise ValueError("certification cell lacks a paired-RQMC randomization key")
        reference = cell.get("reference", {})
        case_name = str(cell.get("case", {}).get("name"))
        expected_profile = (
            {"target": 1, "fine": 2}
            if payload.get("quick")
            else PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE.get(variant, {}).get(
                case_name, {}
            )
        )
        expected_substeps = (
            int(expected_profile.get("target", 0)),
            int(expected_profile.get("fine", 0)),
        )
        actual_substeps = (
            int(reference.get("target_substeps_per_interval", 0)),
            int(reference.get("fine_substeps_per_interval", 0)),
        )
        if actual_substeps != expected_substeps or reference.get("primary") != "fine":
            raise ValueError(
                "certification cell uses an invalid QE-M refinement contract"
            )
        refinement = cell.get("pde_refinement", {})
        if refinement.get("status") not in allowed_statuses:
            raise ValueError(
                "certification cell lacks valid PDE-axis refinement evidence"
            )
        economic_scale = cell.get("economic_scale", {})
        if not np.isclose(
            float(economic_scale.get("hedge_inception_spot", float("nan"))),
            hedge_inception_spot,
        ):
            raise ValueError("certification cell uses an inconsistent hedge spot")
        for greek in ("delta", "gamma"):
            verdict = cell.get("certifications", {}).get(greek, {}).get("verdict", {})
            if verdict.get("status") not in allowed_statuses:
                raise ValueError(f"certification cell has invalid {greek} verdict")
            for key in (
                "estimate_difference",
                "economic_bound",
                "reference_standard_error",
                "total_uncertainty",
            ):
                if not np.isfinite(float(verdict.get(key, float("nan")))):
                    raise ValueError(
                        f"certification {greek} verdict has non-finite {key}"
                    )
    if not payload.get("decisions"):
        raise ValueError("certification payload has no decisions")
    for variant, decision in payload["decisions"].items():
        route = decision.get("route")
        if route not in {"pde", "excluded_greek_unresolved"}:
            raise ValueError(f"{variant}: invalid certification route {route!r}")
        if payload.get("quick") and route == "pde":
            raise ValueError(f"{variant}: quick evidence cannot admit PDE")
        if route == "pde" and decision.get("evidence_complete") is not True:
            raise ValueError(f"{variant}: incomplete evidence cannot admit PDE")
        if route == "pde":
            expected_common = (
                PRODUCTION_HESTON_BATCHES
                if variant == "heston"
                else PRODUCTION_SLV_BATCHES
            )
            if int(decision.get("aggregate_common_scrambles", 0)) != (expected_common):
                raise ValueError(f"{variant}: invalid aggregate common-scramble count")
        delta_bias = decision.get("delta_bias") or {}
        if "aggregate_common_scrambles" in delta_bias and (
            delta_bias.get("aggregate_alignment") != "strided_pooled"
        ):
            raise ValueError(
                f"{variant}: aggregate delta-bias alignment must be the "
                "declared strided pooling"
            )

    for key in ("implementation_sha256", "run_configuration_sha256"):
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"certification lacks a valid {key}")
    run_configuration = payload.get("run_configuration")
    if not isinstance(run_configuration, dict) or (
        _canonical_sha256(run_configuration) != payload["run_configuration_sha256"]
    ):
        raise ValueError("certification run-configuration hash mismatch")
    if (
        run_configuration.get("implementation_sha256")
        != payload["implementation_sha256"]
    ):
        raise ValueError("certification implementation hash mismatch")
    live_runtime = runtime_environment()
    if (
        payload.get("runtime_environment") != live_runtime
        or run_configuration.get("runtime_environment") != live_runtime
    ):
        raise ValueError("certification numerical runtime mismatch")
    if payload["implementation_sha256"] != implementation_sha256():
        raise ValueError("certification does not match the live implementation")
    if payload.get("numerical_implementation_sha256") != (
        numerical_implementation_sha256()
    ):
        raise ValueError("certification does not match the live numerical projection")
    if run_configuration.get("numerical_implementation_sha256") != payload.get(
        "numerical_implementation_sha256"
    ):
        raise ValueError("numerical projection disagrees with the run configuration")
    if (
        run_configuration.get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
        or payload.get("policy", {}).get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
    ):
        raise ValueError("certification uses stale production engine controls")
    if run_configuration.get("heston_spot_bridge_profile_by_case") != payload.get(
        "policy", {}
    ).get("heston_spot_bridge_profile_by_case"):
        raise ValueError("certification Heston bridge profile metadata mismatch")
    if run_configuration.get("slv_spot_bridge_profile_by_case") != payload.get(
        "policy", {}
    ).get("slv_spot_bridge_profile_by_case"):
        raise ValueError("certification SLV bridge profile metadata mismatch")
    if run_configuration.get("qe_substeps_by_variant_case") != payload.get(
        "policy", {}
    ).get("qe_substeps_by_variant_case") or run_configuration.get(
        "slv_multilevel_policy"
    ) != payload.get(
        "policy", {}
    ).get(
        "slv_multilevel_policy"
    ):
        raise ValueError("certification reference-estimator policy mismatch")
    if run_configuration.get("reference_seeds") != payload.get("reference_seeds"):
        raise ValueError("certification reference seed metadata mismatch")
    if (
        run_configuration.get("rqmc_batch_workers")
        != payload.get("policy", {}).get("rqmc_batch_workers")
        or run_configuration.get("rqmc_batch_workers_by_variant_case")
        != payload.get("policy", {}).get("rqmc_batch_workers_by_variant_case")
        or run_configuration.get("cell_workers")
        != payload.get("policy", {}).get("cell_workers")
    ):
        raise ValueError("certification worker-profile metadata mismatch")
    sampling_by_variant = payload.get("sampling_by_variant")
    if not isinstance(sampling_by_variant, dict) or not sampling_by_variant:
        raise ValueError("certification lacks per-variant RQMC sampling metadata")
    for variant in payload["decisions"]:
        sampling = sampling_by_variant.get(variant, {})
        if int(sampling.get("paths_per_batch", 0)) <= 0:
            raise ValueError(f"{variant}: invalid paths_per_batch metadata")
        if int(sampling.get("batches", 0)) < 2:
            raise ValueError(f"{variant}: invalid batches metadata")
        if payload["decisions"][variant].get("route") == "pde":
            minimum_paths = (
                PRODUCTION_HESTON_PATHS_PER_BATCH
                if variant == "heston"
                else PRODUCTION_SLV_PATHS_PER_BATCH
            )
            if int(sampling["paths_per_batch"]) < minimum_paths:
                raise ValueError(
                    f"{variant}: insufficient production sampling for PDE admission"
                )
            if variant == "heston_slv":
                if (
                    int(sampling["batches"]) != PRODUCTION_SLV_BATCHES
                    or _normalized_batches_by_case(sampling)
                    != PRODUCTION_SLV_BATCHES_BY_CASE
                    or _normalized_primary_batches_by_case(sampling)
                    != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE
                ):
                    raise ValueError(
                        f"{variant}: insufficient production sampling for PDE admission"
                    )
                policy = payload.get("policy", {})
                if (
                    int(policy.get("slv_spot_strata", 0)) != SLV_SPOT_STRATA
                    or bool(policy.get("slv_spot_antithetic", False))
                    != SLV_SPOT_ANTITHETIC
                    or int(policy.get("slv_spot_bridge_strata", 0))
                    != SLV_SPOT_BRIDGE_STRATA
                    or policy.get("slv_spot_bridge_profile_by_case")
                    != SLV_SPOT_BRIDGE_PROFILE_BY_CASE
                    or policy.get("qe_substeps_by_variant_case")
                    != PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE
                    or policy.get("slv_multilevel_policy")
                    != {
                        "cases": sorted(SLV_MULTILEVEL_CASES),
                        "mid_paths_per_batch": (SLV_MID_CONTROL_PATHS_PER_BATCH),
                        "mid_batches": SLV_MID_CONTROL_BATCHES,
                        "frozen_control_weight": SLV_FROZEN_CONTROL_WEIGHT,
                        "heston_control_weight": SLV_HESTON_CONTROL_WEIGHT,
                    }
                ):
                    raise ValueError("heston_slv: stale conditional sampling profile")
            else:
                if (
                    int(sampling["batches"]) != PRODUCTION_HESTON_BATCHES
                    or _normalized_batches_by_case(sampling)
                    != PRODUCTION_HESTON_BATCHES_BY_CASE
                ):
                    raise ValueError("heston: stale case-specific batch profile")
                if (
                    payload.get("policy", {}).get("heston_spot_bridge_profile_by_case")
                    != HESTON_SPOT_BRIDGE_PROFILE_BY_CASE
                ):
                    raise ValueError("heston: stale conditional sampling profile")
    if any(
        decision.get("route") == "pde" for decision in payload["decisions"].values()
    ) and payload.get("reference_seeds") != {
        "heston": HESTON_REFERENCE_SEED,
        "heston_slv_primary": SLV_PRIMARY_SEED,
        "heston_slv_mid_control": SLV_MID_CONTROL_SEED,
    }:
        raise ValueError(
            "full PDE recertification requires the declared held-out seed families"
        )
    cells_by_key = {
        (cell.get("variant"), cell.get("case", {}).get("name")): cell
        for cell in payload.get("cells", [])
    }
    for cell in payload.get("cells", []):
        variant = cell["variant"]
        case_name = str(cell.get("case", {}).get("name"))
        reference = cell.get("reference", {})
        expected_seed = (
            payload["reference_seeds"]["heston"]
            if variant == "heston"
            else payload["reference_seeds"]["heston_slv_primary"]
        )
        if int(reference.get("seed", -1)) != int(expected_seed):
            raise ValueError(f"{variant}/{case_name}: reference seed mismatch")
        expected_batch_workers = int(
            run_configuration.get("rqmc_batch_workers_by_variant_case", {})
            .get(variant, {})
            .get(case_name, 0)
        )
        if (
            expected_batch_workers < 1
            or int(reference.get("batch_workers", 0)) != expected_batch_workers
        ):
            raise ValueError(f"{variant}/{case_name}: worker profile mismatch")
        expected_batches = _sampling_batches_for_case(
            sampling_by_variant[variant], variant, case_name
        )
        # Every per-cell evidence array is sized against the ADMISSIBLE count,
        # never the declared one: a gate-driven cell banks fewer batches on
        # purpose, and each array it banked is that much shorter. Bound once per
        # cell rather than per level, because the first version of this fix bound
        # it inside the level loop and the two arrays validated after that loop
        # kept reading `expected_batches` -- one of them only reachable for
        # heston_slv, so a single-variant replay could not see it. One binding
        # means a fourth array cannot quietly drift back to the declared count.
        allowed_batches = _admissible_batch_count(
            cell,
            run_configuration,
            variant=variant,
            case_name=case_name,
            declared_batches=expected_batches,
        )
        if variant == "heston":
            declared_profile = run_configuration.get(
                "heston_spot_bridge_profile_by_case", {}
            ).get(case_name, {})
            actual_profile = {
                "strata": int(
                    cell.get("reference", {}).get("heston_spot_bridge_strata", 0)
                ),
                "dimensions": int(
                    cell.get("reference", {}).get("heston_spot_bridge_dimensions", 0)
                ),
            }
            if actual_profile != declared_profile:
                raise ValueError(
                    f"heston/{case_name}: conditional sampling profile mismatch"
                )
        else:
            declared_profile = run_configuration.get(
                "slv_spot_bridge_profile_by_case", {}
            ).get(case_name, {})
            actual_profile = {
                "strata": int(reference.get("slv_spot_bridge_strata", 0)),
                "dimensions": int(reference.get("slv_spot_bridge_dimensions", 0)),
            }
            if actual_profile != declared_profile:
                raise ValueError(
                    f"heston_slv/{case_name}: conditional sampling profile mismatch"
                )
        multilevel = (
            not payload.get("quick")
            and variant == "heston_slv"
            and case_name in SLV_MULTILEVEL_CASES
        )
        expected_estimator = (
            "three_level_frozen_slv_heston_control"
            if multilevel
            else "primary_conditional_rqmc"
        )
        if reference.get("estimator", {}).get("name") != expected_estimator:
            raise ValueError(f"{variant}/{case_name}: estimator profile mismatch")
        for level in ("target", "fine"):
            level_evidence = reference.get(level, {})
            level_result = paired_result_from_serialized(
                level_evidence,
                randomization_label=f"validation/{variant}/{case_name}/{level}",
            )
            actual_batches = int(level_result.batches_used)
            if actual_batches != allowed_batches:
                raise ValueError(
                    f"{variant}: {level} batches do not match sampling metadata"
                )
            estimates = np.asarray(level_evidence.get("batch_estimates"), dtype=float)
            covariance = np.asarray(level_evidence.get("covariance"), dtype=float)
            if estimates.shape != (allowed_batches, 5) or not np.all(
                np.isfinite(estimates)
            ):
                raise ValueError(f"{variant}: invalid {level} paired batch evidence")
            if covariance.shape != (5, 5) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"{variant}: invalid {level} covariance evidence")
            control_estimates = level_evidence.get("control_batch_estimates")
            if variant == "heston_slv" and not multilevel:
                control_estimates = np.asarray(control_estimates, dtype=float)
                if control_estimates.shape != (allowed_batches, 5) or not np.all(
                    np.isfinite(control_estimates)
                ):
                    raise ValueError(
                        f"{variant}: invalid {level} conditional-control evidence"
                    )
            elif control_estimates is not None:
                raise ValueError(
                    f"{variant}: unexpected {level} conditional-control evidence"
                )
        if multilevel:
            estimator = reference["estimator"]
            if (
                estimator.get("identity") != "Y-a*F_low+a*F_high-b*H_low+b*H_high"
                or estimator.get("weights")
                != {
                    "frozen_slv": SLV_FROZEN_CONTROL_WEIGHT,
                    "heston": SLV_HESTON_CONTROL_WEIGHT,
                }
                or int(estimator.get("primary_seed", -1)) != SLV_PRIMARY_SEED
                or int(estimator.get("mid_control_seed", -1)) != SLV_MID_CONTROL_SEED
                or int(estimator.get("heston_high_seed", -1)) != HESTON_REFERENCE_SEED
                or int(estimator.get("outer_batches", 0)) != PRODUCTION_SLV_BATCHES
                or int(estimator.get("primary_paths_per_batch", 0))
                != PRODUCTION_SLV_PATHS_PER_BATCH
                or int(estimator.get("primary_batches", 0))
                != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE[case_name]
                or int(estimator.get("mid_paths_per_batch", 0))
                != SLV_MID_CONTROL_PATHS_PER_BATCH
                or int(estimator.get("mid_batches", 0)) != SLV_MID_CONTROL_BATCHES
            ):
                raise ValueError(
                    f"heston_slv/{case_name}: stale multilevel estimator metadata"
                )
            heston_cell = cells_by_key.get(("heston", case_name))
            if heston_cell is None:
                raise ValueError(f"heston_slv/{case_name}: missing Heston control cell")
            heston_cell_hash = hashlib.sha256(
                json.dumps(
                    evidence_projection(heston_cell),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if estimator.get("heston_high_cell_sha256") != heston_cell_hash:
                raise ValueError(
                    f"heston_slv/{case_name}: Heston control provenance mismatch"
                )
            components = estimator.get("components", {})
            primary_batches = PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE[case_name]
            heston_batches = _sampling_batches_for_case(
                sampling_by_variant["heston"], "heston", case_name
            )
            for level in ("target", "fine"):
                level_components = components.get(level, {})
                primary = level_components.get("primary_low", {})
                frozen_high = level_components.get("frozen_high", {})
                heston_low = level_components.get("heston_low", {})
                primary_rows = np.asarray(primary.get("batch_estimates"), dtype=float)
                primary_controls = np.asarray(
                    primary.get("control_batch_estimates"), dtype=float
                )
                frozen_rows = np.asarray(
                    frozen_high.get("batch_estimates"), dtype=float
                )
                bundled_heston_rows = np.asarray(
                    frozen_high.get("control_batch_estimates"), dtype=float
                )
                heston_rows = np.asarray(heston_low.get("batch_estimates"), dtype=float)
                if (
                    primary_rows.shape != (primary_batches, 5)
                    or primary_controls.shape != (primary_batches, 5)
                    or frozen_rows.shape != (SLV_MID_CONTROL_BATCHES, 5)
                    or bundled_heston_rows.shape != (SLV_MID_CONTROL_BATCHES, 5)
                    or heston_rows.shape != (SLV_MID_CONTROL_BATCHES, 5)
                    or not all(
                        np.all(np.isfinite(rows))
                        for rows in (
                            primary_rows,
                            primary_controls,
                            frozen_rows,
                            bundled_heston_rows,
                            heston_rows,
                        )
                    )
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: invalid {level} multilevel rows"
                    )
                expected_middle_valuations = (
                    3
                    * SLV_MID_CONTROL_PATHS_PER_BATCH
                    * SLV_MID_CONTROL_BATCHES
                    * 2
                    * SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]["strata"]
                )
                if (
                    not np.array_equal(bundled_heston_rows, heston_rows)
                    or int(frozen_high.get("total_unique_paths", -1))
                    != (SLV_MID_CONTROL_PATHS_PER_BATCH * SLV_MID_CONTROL_BATCHES)
                    or int(frozen_high.get("total_path_valuations", -1))
                    != expected_middle_valuations
                    or int(heston_low.get("total_unique_paths", -1)) != 0
                    or int(heston_low.get("total_path_valuations", -1)) != 0
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: invalid bundled Heston control provenance"
                    )
                if frozen_high.get("randomization_key") != heston_low.get(
                    "randomization_key"
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: middle controls are not coupled"
                    )
                high_link = level_components.get("heston_high_reference", {})
                if (
                    high_link.get("variant") != "heston"
                    or high_link.get("case") != case_name
                    or int(high_link.get("batches", 0)) != heston_batches
                    or high_link.get("randomization_key")
                    != (
                        f"heston-high/{case_name}/{level}("
                        f"{heston_cell['reference'][level]['randomization_key']})"
                    )
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: invalid Heston high-control link"
                    )
                for component_name, component in (
                    ("primary_low", primary),
                    ("frozen_high", frozen_high),
                    ("heston_low", heston_low),
                ):
                    paired_result_from_serialized(
                        component,
                        randomization_label=(
                            f"validation/{case_name}/{level}/{component_name}"
                        ),
                    )
                heston_high_rows = np.asarray(
                    heston_cell["reference"][level]["batch_estimates"],
                    dtype=float,
                )
                if (
                    heston_high_rows.shape != (heston_batches, 5)
                    or primary_batches % PRODUCTION_SLV_BATCHES != 0
                    or heston_batches % PRODUCTION_SLV_BATCHES != 0
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: multilevel rows cannot be grouped"
                    )

                def grouped(rows: np.ndarray) -> np.ndarray:
                    return rows.reshape(
                        PRODUCTION_SLV_BATCHES,
                        rows.shape[0] // PRODUCTION_SLV_BATCHES,
                        5,
                    ).mean(axis=1)

                recomposed_rows = (
                    grouped(primary_rows)
                    - SLV_FROZEN_CONTROL_WEIGHT * grouped(primary_controls)
                    + SLV_FROZEN_CONTROL_WEIGHT * grouped(frozen_rows)
                    - SLV_HESTON_CONTROL_WEIGHT * grouped(heston_rows)
                    + SLV_HESTON_CONTROL_WEIGHT * grouped(heston_high_rows)
                )
                published_rows = np.asarray(
                    reference[level]["batch_estimates"], dtype=float
                )
                if not np.allclose(
                    published_rows,
                    recomposed_rows,
                    rtol=1e-12,
                    atol=1e-14,
                ):
                    raise ValueError(
                        f"heston_slv/{case_name}: {level} multilevel recomposition mismatch"
                    )
        for greek in ("delta", "gamma"):
            batches = np.asarray(
                cell["certifications"][greek].get("reference_substep_batch_contracts"),
                dtype=float,
            )
            if batches.shape != (allowed_batches,) or not np.all(np.isfinite(batches)):
                raise ValueError(f"{variant}: invalid {greek} substep batch evidence")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def implementation_sha256() -> str:
    """Hash every in-repo numerical input used by the certification run."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_INPUTS:
        path = root / relative
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot hash certification input {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def numerical_implementation_sha256() -> str:
    """Hash only the inputs that can change a cell's numbers.

    Same file list as `implementation_sha256`, but this file is projected through
    `NON_NUMERICAL_SYMBOLS` first. `implementation_sha256` is kept alongside and
    still recorded in the payload, because it remains the honest answer to "what
    was the whole source state" -- it is simply the wrong question to ask of a
    banked cell.
    """
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().relative_to(root).as_posix()
    return source_projection_sha256(
        [
            (
                root / relative,
                NON_NUMERICAL_SYMBOLS if relative == here else (),
            )
            for relative in IMPLEMENTATION_INPUTS
            if relative not in NUMERICAL_EXEMPT_INPUTS
        ],
        root=root,
    )


def cell_plan_projection(
    variant: str, case_name: str, run_configuration: dict
) -> dict:
    """The declared plan for ONE cell.

    Everything here can change this cell's numbers; nothing outside it can. The
    exclusions are as load-bearing as the inclusions:

    * other cells' plans and the fleet's case selection -- a cell does not know
      what else ran, so `--cases near_ko` must reuse a cell banked by a full run;
    * `cell_workers` -- pure scheduling across cells;
    * output paths, timestamps, resume flags.

    `rqmc_batch_workers` IS included even though batch-worker count is proven
    bitwise-invariant, because that invariance is a property of the current
    driver rather than of the estimator, and it was briefly false: the threaded
    branch ignored the batch-range offset until `bbd452e`. Recording it costs a
    re-run only when someone changes worker counts, which is rare.
    """
    sampling = (run_configuration.get("sampling_by_variant", {}) or {}).get(variant, {})
    cases = {
        entry.get("name"): entry for entry in run_configuration.get("cases", []) or []
    }
    if case_name not in cases:
        raise ValueError(f"{variant}/{case_name}: case is not in the run configuration")
    stopping = run_configuration.get("sequential_stopping", {}) or {}
    excluded = set(stopping.get("excluded_variant_cases", []) or [])
    label = f"{variant}/{case_name}"
    bridge_key = (
        "heston_spot_bridge_profile_by_case"
        if variant == "heston"
        else "slv_spot_bridge_profile_by_case"
    )
    return {
        "schema_version": run_configuration.get("schema_version"),
        "certification_mode": run_configuration.get("certification_mode"),
        "quick": run_configuration.get("quick"),
        "skip_anchors": run_configuration.get("skip_anchors"),
        "variant": variant,
        "case": cases[case_name],
        "reference_seeds": run_configuration.get("reference_seeds"),
        "paths_per_batch": sampling.get("paths_per_batch"),
        "batches_by_case": (sampling.get("batches_by_case", {}) or {}).get(case_name),
        "primary_batches_by_case": (
            sampling.get("primary_batches_by_case", {}) or {}
        ).get(case_name),
        "spot_bump": run_configuration.get("spot_bump"),
        "full_bump_ladder": run_configuration.get("full_bump_ladder"),
        "stochastic_component_confidence": run_configuration.get(
            "stochastic_component_confidence"
        ),
        "hedge_inception_spot": run_configuration.get("hedge_inception_spot"),
        "production_engine_controls": run_configuration.get(
            "production_engine_controls"
        ),
        "spot_bridge_profile": (run_configuration.get(bridge_key, {}) or {}).get(
            case_name
        ),
        "qe_substeps": (
            run_configuration.get("qe_substeps_by_variant_case", {}) or {}
        ).get(variant, {}).get(case_name),
        "rqmc_batch_workers": (
            run_configuration.get("rqmc_batch_workers_by_variant_case", {}) or {}
        ).get(variant, {}).get(case_name),
        "slv_spot_strata": run_configuration.get("slv_spot_strata"),
        "slv_spot_antithetic": run_configuration.get("slv_spot_antithetic"),
        "slv_spot_bridge_strata": run_configuration.get("slv_spot_bridge_strata"),
        "slv_multilevel_policy": run_configuration.get("slv_multilevel_policy"),
        # The cell's OWN stopping rule, not the fleet's: whether it may stop, and
        # under exactly which declared parameters.
        "stopping": (
            {"enabled": False}
            if not stopping.get("enabled") or label in excluded
            else {
                "enabled": True,
                "chunk_batches": stopping.get("chunk_batches"),
                "margin_fraction": stopping.get("margin_fraction"),
                "family_alpha": stopping.get("family_alpha"),
            }
        ),
    }


def cell_identity(
    variant: str,
    case_name: str,
    run_configuration: dict,
    *,
    numerical_sha256: str,
    consumed: Optional[dict] = None,
) -> str:
    """The identity a banked cell must match to be reusable."""
    return cell_identity_sha256(
        numerical_sha256=numerical_sha256,
        plan=cell_plan_projection(variant, case_name, run_configuration),
        runtime=run_configuration.get("runtime_environment", {}) or {},
        consumed=dict(consumed or {}),
    )


def production_pde_compatibility_sha256() -> str:
    """Hash the live production-PDE dependency projection.

    The path set is deterministic and intentionally excludes the certification
    harness and reference estimators.  This lets a schema-11 amendment carry
    completed PDE evidence only when the production numerical implementation
    itself is byte-for-byte compatible with its schema-9 parent.
    """
    root = Path(__file__).resolve().parents[2]
    paths: set[Path] = set()
    for relative in PRODUCTION_PDE_INPUT_ROOTS:
        path = root / relative
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(path.rglob("*.py"))
        else:
            raise ValueError(f"cannot hash production PDE input {path}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot hash production PDE input {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_sha256(value: dict) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_path(output_dir: Path, name: str) -> Path:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    if not name or any(character not in allowed for character in name):
        raise ValueError(f"unsafe checkpoint name: {name!r}")
    return output_dir / "checkpoints" / f"{name}.json"


def _write_checkpoint(
    output_dir: Path,
    name: str,
    *,
    run_configuration_sha256: str,
    kind: str,
    evidence,
    identity_sha256: Optional[str] = None,
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification_checkpoint",
        # Recorded for provenance, no longer used to gate reuse: a fleet-wide
        # digest cannot say whether ONE cell may be reused.
        "run_configuration_sha256": run_configuration_sha256,
        "identity_sha256": identity_sha256,
        "kind": kind,
        "evidence": evidence,
    }
    _atomic_write(
        _checkpoint_path(output_dir, name),
        json.dumps(record, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


def _load_checkpoint(
    output_dir: Path,
    name: str,
    *,
    run_configuration_sha256: str,
    kind: str,
    identity_sha256: Optional[str] = None,
):
    """Reuse a banked artifact only when its identity matches.

    Cells are gated on `identity_sha256` -- the numerical projection, this cell's
    own declared plan, the runtime, and the identities of the cells it consumes.
    Artifacts with no per-cell identity (the shared deterministic anchors) keep
    the fleet-wide configuration hash, because they are genuinely fleet-wide.

    A checkpoint written before identities existed is refused rather than
    accepted or silently re-stamped: it cannot state what it depended on.
    """
    path = _checkpoint_path(output_dir, name)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid certification checkpoint {path}") from exc
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("study") != "adi_2d_snowball_greek_certification_checkpoint"
        or record.get("kind") != kind
    ):
        raise ValueError(f"checkpoint metadata mismatch: {path}")
    if identity_sha256 is None:
        if record.get("run_configuration_sha256") != run_configuration_sha256:
            raise ValueError(
                f"checkpoint configuration mismatch: {path}; use a new output "
                "directory or rerun without --resume"
            )
        return record.get("evidence")
    banked_identity = record.get("identity_sha256")
    if banked_identity is None:
        raise ValueError(
            f"checkpoint predates cell identities: {path}; it cannot state what "
            "it depended on, so it cannot be reused"
        )
    if banked_identity != identity_sha256:
        raise ValueError(
            f"cell identity mismatch: {path}; banked {banked_identity[:16]} "
            f"but this run requires {identity_sha256[:16]} -- this cell's "
            "arithmetic, declared plan, runtime, or a cell it consumes has changed"
        )
    return record.get("evidence")


def build_decision_payload(payload: dict) -> dict:
    """Build a self-hashed, standalone routing decision."""
    decision = {
        "schema_version": SCHEMA_VERSION,
        "certification_mode": payload["certification_mode"],
        "evidence_sha256": payload["evidence_sha256"],
        "implementation_sha256": payload["implementation_sha256"],
        "run_configuration_sha256": payload["run_configuration_sha256"],
        "run_configuration": payload["run_configuration"],
        "runtime_environment": payload["runtime_environment"],
        "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        "quick": payload["quick"],
        "decisions": payload["decisions"],
    }
    if payload["certification_mode"] == CERTIFICATION_MODE_AMENDMENT:
        decision.update(
            {
                "parent_certificate": payload["parent_certificate"],
                "production_pde_compatibility_sha256": payload[
                    "production_pde_compatibility_sha256"
                ],
                "cell_provenance": payload["cell_provenance"],
                "aggregate_cohorts": payload["aggregate_cohorts"],
                "auxiliary_control_sha256": _cell_evidence_sha256(
                    payload["auxiliary_controls"]["heston_near_ki_8_16"]
                ),
            }
        )
    decision["decision_sha256"] = _canonical_sha256(decision)
    return decision


def publish_payload(payload: dict, output_dir: Path) -> None:
    """Validate, hash, and atomically publish all certification artifacts."""
    payload.pop("evidence_sha256", None)
    validate_payload(payload)
    canonical = json.dumps(
        evidence_projection(payload),
        sort_keys=True,
        default=_json_default,
    )
    payload["evidence_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    json_text = (
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    report = render_markdown(payload)
    decision = build_decision_payload(payload)
    _atomic_write(output_dir / "adi_greek_certification.json", json_text)
    _atomic_write(output_dir / "adi_greek_certification.md", report)
    _atomic_write(
        output_dir / "adi_greek_certification_decision.json",
        json.dumps(decision, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


def run_incremental_amendment(args: argparse.Namespace) -> int:
    """Execute only the unresolved schema-9 boundary cells and one control."""
    parent, parent_decision, manifest = load_and_validate_parent_certificate(
        args.amend_parent_evidence,
        args.amend_parent_decision,
    )
    implementation_hash = implementation_sha256()
    runtime = runtime_environment()
    run_configuration = amendment_run_configuration(
        implementation_hash=implementation_hash,
        runtime=runtime,
        hedge_inception_spot=args.hedge_inception_spot,
    )
    run_configuration_hash = _canonical_sha256(run_configuration)
    print(
        "[adi-greeks] schema-11 incremental amendment: "
        "carry Heston 7/7 + SLV 5/7; compute Heston Near-KI control and "
        "replace SLV Near-KI/Low-Feller",
        flush=True,
    )
    print(
        f"[adi-greeks] implementation={implementation_hash[:16]} "
        f"configuration={run_configuration_hash[:16]} "
        f"parent={manifest['evidence_sha256'][:16]}",
        flush=True,
    )
    started = time.perf_counter()
    case_by_name = {case.name: case for case in certification_cases(quick=False)}

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
                print(f"[adi-greeks] resumed {name}", flush=True)
        if evidence is None:
            evidence = compute()
            _write_checkpoint(
                args.output_dir,
                name,
                run_configuration_sha256=run_configuration_hash,
                kind=kind,
                evidence=evidence,
            )
            print(f"[adi-greeks] checkpointed {name}", flush=True)
        return evidence

    def compute_heston_control() -> dict:
        print("[adi-greeks] control-only heston/near_ki 8->16", flush=True)
        profile = HESTON_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]
        return build_heston_high_control_evidence(
            case_by_name["near_ki"],
            paths_per_batch=PRODUCTION_HESTON_PATHS_PER_BATCH,
            batches=PRODUCTION_HESTON_BATCHES_BY_CASE["near_ki"],
            seed=HESTON_REFERENCE_SEED,
            target_substeps=8,
            fine_substeps=16,
            heston_spot_bridge_strata=profile["strata"],
            heston_spot_bridge_dimensions=profile["dimensions"],
            rqmc_batch_workers=(
                PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston"]["near_ki"]
            ),
        )

    def compute_low_feller() -> dict:
        print("[adi-greeks] replacement heston_slv/low_feller", flush=True)
        profile = SLV_SPOT_BRIDGE_PROFILE_BY_CASE["low_feller"]
        return certify_case(
            "heston_slv",
            case_by_name["low_feller"],
            quick=False,
            paths_per_batch=PRODUCTION_SLV_PATHS_PER_BATCH,
            batches=PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["low_feller"],
            seed=SLV_PRIMARY_SEED,
            hedge_inception_spot=args.hedge_inception_spot,
            slv_spot_strata=SLV_SPOT_STRATA,
            slv_spot_antithetic=SLV_SPOT_ANTITHETIC,
            slv_spot_bridge_strata=profile["strata"],
            slv_spot_bridge_dimensions=profile["dimensions"],
            slv_mid_control_seed=SLV_MID_CONTROL_SEED,
            rqmc_batch_workers=(
                PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston_slv"][
                    "low_feller"
                ]
            ),
        )

    # These two computations have independent seed families and fit the host's
    # memory envelope concurrently. Near-KI SLV waits for the Heston control.
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="adi-amendment",
    ) as pool:
        control_future = pool.submit(
            checkpointed,
            "control__heston__near_ki__8_16",
            "auxiliary_control",
            compute_heston_control,
        )
        low_feller_future = pool.submit(
            checkpointed,
            "replacement__heston_slv__low_feller",
            "cell",
            compute_low_feller,
        )
        heston_control = control_future.result()
        low_feller = low_feller_future.result()

    def compute_near_ki() -> dict:
        print("[adi-greeks] replacement heston_slv/near_ki", flush=True)
        profile = SLV_SPOT_BRIDGE_PROFILE_BY_CASE["near_ki"]
        return certify_case(
            "heston_slv",
            case_by_name["near_ki"],
            quick=False,
            paths_per_batch=PRODUCTION_SLV_PATHS_PER_BATCH,
            batches=PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE["near_ki"],
            seed=SLV_PRIMARY_SEED,
            hedge_inception_spot=args.hedge_inception_spot,
            slv_spot_strata=SLV_SPOT_STRATA,
            slv_spot_antithetic=SLV_SPOT_ANTITHETIC,
            slv_spot_bridge_strata=profile["strata"],
            slv_spot_bridge_dimensions=profile["dimensions"],
            slv_mid_control_seed=SLV_MID_CONTROL_SEED,
            heston_high_cell=heston_control,
            rqmc_batch_workers=(
                PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston_slv"]["near_ki"]
            ),
        )

    near_ki = checkpointed(
        "replacement__heston_slv__near_ki",
        "cell",
        compute_near_ki,
    )
    parent_cells = {
        (cell["variant"], cell["case"]["name"]): cell for cell in parent["cells"]
    }
    replacements = {"near_ki": near_ki, "low_feller": low_feller}
    cases = certification_cases(quick=False)
    cells = []
    for variant in ("heston", "heston_slv"):
        for case in cases:
            if variant == "heston_slv" and case.name in replacements:
                cells.append(replacements[case.name])
            else:
                cells.append(parent_cells[(variant, case.name)])
    decisions, aggregate_cohorts = make_amendment_decisions(
        cells,
        parent["anchors"],
        parent_decision["decisions"],
    )
    sampling = amendment_sampling_by_variant()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
        "certification_mode": CERTIFICATION_MODE_AMENDMENT,
        "created_at": datetime.now().astimezone().isoformat(),
        "quick": False,
        "profile": "production incremental amendment",
        "parent_certificate": manifest,
        "production_pde_compatibility_sha256": (production_pde_compatibility_sha256()),
        "reference_seeds": amendment_reference_seeds(),
        "paths_per_batch": PRODUCTION_SLV_PATHS_PER_BATCH,
        "batches": PRODUCTION_SLV_BATCHES,
        "sampling_by_variant": sampling,
        "confidence": CONFIDENCE,
        "python": runtime["python_version"],
        "runtime_environment": runtime,
        "implementation_sha256": implementation_hash,
        "run_configuration_sha256": run_configuration_hash,
        "run_configuration": run_configuration,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "delta_cell_bound_contracts": DELTA_CELL_BOUND_CONTRACTS,
            "delta_bias_bound_contracts": DELTA_BIAS_BOUND_CONTRACTS,
            "gamma_cell_bound_contracts_for_1pct_move": (GAMMA_CELL_BOUND_CONTRACTS),
            "hedge_inception_spot": float(args.hedge_inception_spot),
            "hedge_inception_spot_policy": (
                "strictest minimum from the frozen 27-inception MO cohort"
            ),
            "unresolved_route": "excluded_greek_unresolved",
            "heston_spot_bridge_profile_by_case": (HESTON_SPOT_BRIDGE_PROFILE_BY_CASE),
            "slv_spot_bridge_profile_by_case": (SLV_SPOT_BRIDGE_PROFILE_BY_CASE),
            "qe_substeps_by_variant_case": (amendment_qe_substeps_by_variant_case()),
            "slv_multilevel_policy": {
                "cases": sorted(SLV_MULTILEVEL_CASES),
                "mid_paths_per_batch": SLV_MID_CONTROL_PATHS_PER_BATCH,
                "mid_batches": SLV_MID_CONTROL_BATCHES,
                "frozen_control_weight": SLV_FROZEN_CONTROL_WEIGHT,
                "heston_control_weight": SLV_HESTON_CONTROL_WEIGHT,
            },
            "slv_spot_strata": SLV_SPOT_STRATA,
            "slv_spot_antithetic": SLV_SPOT_ANTITHETIC,
            "slv_spot_bridge_strata": SLV_SPOT_BRIDGE_STRATA,
            "rqmc_batch_workers_by_variant_case": run_configuration[
                "rqmc_batch_workers_by_variant_case"
            ],
            "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        },
        "anchors": parent["anchors"],
        "cells": cells,
        "cell_provenance": amendment_cell_provenance(cells),
        "auxiliary_controls": {"heston_near_ki_8_16": heston_control},
        "aggregate_cohorts": aggregate_cohorts,
        "decisions": decisions,
    }
    publish_payload(payload, args.output_dir)
    print(f"[adi-greeks] wrote {args.output_dir}", flush=True)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="non-production smoke profile"
    )
    parser.add_argument(
        "--full-recertification",
        action="store_true",
        help=(
            "explicitly authorize a fresh production run of all 14 cells; "
            "without this flag production mode requires a schema-9 amendment"
        ),
    )
    parser.add_argument(
        "--amend-parent-evidence",
        type=Path,
        help="exact schema-9 evidence JSON to carry into a schema-11 amendment",
    )
    parser.add_argument(
        "--amend-parent-decision",
        type=Path,
        help="exact schema-9 routing decision paired with --amend-parent-evidence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/adi_greek_certification"),
    )
    parser.add_argument(
        "--rescore-evidence",
        type=Path,
        help=(
            "reuse saved raw PDE/RQMC evidence and recompute only economic "
            "scaling, intervals, decisions, and hashes"
        ),
    )
    parser.add_argument("--paths-per-batch", type=int)
    parser.add_argument("--batches", type=int)
    parser.add_argument("--heston-paths-per-batch", type=int)
    parser.add_argument("--heston-batches", type=int)
    parser.add_argument("--heston-slv-paths-per-batch", type=int)
    parser.add_argument("--heston-slv-batches", type=int)
    parser.add_argument(
        "--heston-spot-bridge-strata",
        type=int,
        help=(
            "developer override applied to every Heston case; production uses "
            "the case-specific bridge profile"
        ),
    )
    parser.add_argument(
        "--heston-spot-bridge-dimensions",
        type=int,
        help=(
            "developer override for the number of leading residual Heston "
            "bridge coordinates; production uses the case-specific profile"
        ),
    )
    parser.add_argument(
        "--slv-spot-strata",
        type=int,
        help=(
            "randomized equal-probability terminal spot-factor strata "
            f"(production default: {SLV_SPOT_STRATA})"
        ),
    )
    parser.add_argument(
        "--slv-spot-antithetic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pair every randomized SLV spot stratum with its antithetic shift",
    )
    parser.add_argument(
        "--slv-spot-bridge-strata",
        type=int,
        help=(
            "randomized strata for the second spot Brownian-bridge coordinate "
            f"(production default: {SLV_SPOT_BRIDGE_STRATA})"
        ),
    )
    parser.add_argument(
        "--slv-spot-bridge-dimensions",
        type=int,
        help=(
            "developer override for leading residual SLV bridge coordinates; "
            "production uses the case-specific profile"
        ),
    )
    parser.add_argument(
        "--rqmc-batch-workers",
        type=int,
        help=(
            "parallel workers for independent RQMC scrambles "
            f"(production default: {PRODUCTION_RQMC_BATCH_WORKERS})"
        ),
    )
    parser.add_argument(
        "--cell-workers",
        type=int,
        help=(
            "parallel independent regime cells; memory-heavy SLV multilevel "
            "cells are always serialized "
            f"(production default: {PRODUCTION_CELL_WORKERS})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "developer base-seed override; derives independent Heston, SLV "
            "primary, and SLV middle-level families as seed/seed+1/seed+2"
        ),
    )
    parser.add_argument("--heston-seed", type=int)
    parser.add_argument("--heston-slv-seed", type=int)
    parser.add_argument("--slv-mid-control-seed", type=int)
    parser.add_argument(
        "--hedge-inception-spot",
        type=float,
        default=DEFAULT_HEDGE_INCEPTION_SPOT,
        help=(
            "actual index inception level used for the futures-contract delta "
            "quantum; independent of the normalized PDE spot"
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("heston", "heston_slv"),
        default=("heston", "heston_slv"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help="optional subset of case names",
    )
    parser.add_argument(
        "--skip-anchors",
        action="store_true",
        help="developer smoke only; forces unresolved routing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse hash-matched deterministic-anchor and per-cell checkpoints; "
            "mismatched source/configuration evidence fails closed"
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "stop each cell once its certification gate is decided, instead of "
            "spending the declared allocation; judged against anytime-valid "
            "widths and capped at that allocation, so it can never cost more"
        ),
    )
    parser.add_argument(
        "--sequential-chunk-batches",
        type=int,
        default=128,
        help=(
            "batches priced between gate evaluations; stops land on multiples "
            "of this, so it bounds the overshoot past the true stop (default 128)"
        ),
    )
    parser.add_argument(
        "--sequential-margin",
        type=float,
        default=0.0,
        help=(
            "fraction of the economic bound held back before admitting, so a "
            "certificate does not read as passing by a hair (default 0.0)"
        ),
    )
    parser.add_argument(
        "--sequential-family-alpha",
        type=float,
        default=0.05,
        help="family-wise error budget spread across every cell-greek test",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not np.isfinite(args.hedge_inception_spot) or args.hedge_inception_spot <= 0.0:
        raise ValueError("hedge-inception-spot must be finite and positive")
    amendment_requested = (
        args.amend_parent_evidence is not None or args.amend_parent_decision is not None
    )
    if amendment_requested:
        if args.amend_parent_evidence is None or args.amend_parent_decision is None:
            raise ValueError(
                "schema-11 amendment requires both parent evidence and decision"
            )
        changed = []
        if args.quick:
            changed.append("quick")
        if args.full_recertification:
            changed.append("full_recertification")
        if args.skip_anchors:
            changed.append("skip_anchors")
        optional_overrides = {
            "rescore_evidence": args.rescore_evidence,
            "paths_per_batch": args.paths_per_batch,
            "batches": args.batches,
            "heston_paths_per_batch": args.heston_paths_per_batch,
            "heston_batches": args.heston_batches,
            "heston_slv_paths_per_batch": args.heston_slv_paths_per_batch,
            "heston_slv_batches": args.heston_slv_batches,
            "heston_spot_bridge_strata": args.heston_spot_bridge_strata,
            "heston_spot_bridge_dimensions": args.heston_spot_bridge_dimensions,
            "slv_spot_strata": args.slv_spot_strata,
            "slv_spot_antithetic": args.slv_spot_antithetic,
            "slv_spot_bridge_strata": args.slv_spot_bridge_strata,
            "slv_spot_bridge_dimensions": args.slv_spot_bridge_dimensions,
            "rqmc_batch_workers": args.rqmc_batch_workers,
            "cell_workers": args.cell_workers,
            "seed": args.seed,
            "heston_seed": args.heston_seed,
            "heston_slv_seed": args.heston_slv_seed,
            "slv_mid_control_seed": args.slv_mid_control_seed,
            "cases": args.cases,
        }
        changed.extend(
            name for name, value in optional_overrides.items() if value is not None
        )
        changed.sort()
        if tuple(args.variants) != ("heston", "heston_slv"):
            changed.append("variants")
        if not np.isclose(
            args.hedge_inception_spot,
            DEFAULT_HEDGE_INCEPTION_SPOT,
        ):
            changed.append("hedge_inception_spot")
        if changed:
            raise ValueError(
                "schema-11 amendment has a frozen production profile; "
                f"unsupported overrides: {changed}"
            )
        return run_incremental_amendment(args)
    if (
        not args.quick
        and args.rescore_evidence is None
        and not args.full_recertification
    ):
        raise ValueError(
            "production certification is incremental by default: provide the "
            "schema-9 --amend-parent-evidence/--amend-parent-decision pair; "
            "use --full-recertification only to explicitly rerun all 14 cells"
        )
    explicit_seed_controls = (
        args.heston_seed,
        args.heston_slv_seed,
        args.slv_mid_control_seed,
    )
    if args.seed is not None and any(
        value is not None for value in explicit_seed_controls
    ):
        raise ValueError(
            "--seed cannot be combined with variant-specific seed controls"
        )
    if args.seed is not None:
        reference_seeds = {
            "heston": int(args.seed),
            "heston_slv_primary": int(args.seed) + 1,
            "heston_slv_mid_control": int(args.seed) + 2,
        }
    else:
        reference_seeds = {
            "heston": int(
                HESTON_REFERENCE_SEED if args.heston_seed is None else args.heston_seed
            ),
            "heston_slv_primary": int(
                SLV_PRIMARY_SEED
                if args.heston_slv_seed is None
                else args.heston_slv_seed
            ),
            "heston_slv_mid_control": int(
                SLV_MID_CONTROL_SEED
                if args.slv_mid_control_seed is None
                else args.slv_mid_control_seed
            ),
        }
    if len(set(reference_seeds.values())) != len(reference_seeds):
        raise ValueError("reference seed families must be mutually independent")
    if args.rescore_evidence is not None:
        sampling_controls = (
            args.paths_per_batch,
            args.batches,
            args.heston_paths_per_batch,
            args.heston_batches,
            args.heston_slv_paths_per_batch,
            args.heston_slv_batches,
            args.heston_spot_bridge_strata,
            args.heston_spot_bridge_dimensions,
            args.slv_spot_strata,
            args.slv_spot_antithetic,
            args.slv_spot_bridge_strata,
            args.slv_spot_bridge_dimensions,
            args.rqmc_batch_workers,
            args.cell_workers,
            args.seed,
            *explicit_seed_controls,
        )
        if args.quick or any(value is not None for value in sampling_controls):
            raise ValueError(
                "--rescore-evidence cannot be combined with quick/path/batch controls"
            )
        if args.resume:
            raise ValueError("--rescore-evidence cannot be combined with --resume")
        try:
            source = json.loads(args.rescore_evidence.read_text())
        except OSError as exc:
            raise ValueError(
                f"cannot read rescore evidence {args.rescore_evidence}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError("rescore evidence is not valid JSON") from exc
        if source.get("certification_mode") == CERTIFICATION_MODE_AMENDMENT:
            raise ValueError(
                "schema-11 amendment evidence has a frozen economic scale and "
                "cannot be rescored as a monolithic certificate"
            )
        payload = rescore_payload(source, args.hedge_inception_spot)
        publish_payload(payload, args.output_dir)
        print(f"[adi-greeks] rescored {args.rescore_evidence}", flush=True)
        print(f"[adi-greeks] wrote {args.output_dir}", flush=True)
        return 0

    heston_default_paths = 256 if args.quick else PRODUCTION_HESTON_PATHS_PER_BATCH
    heston_default_batches = 4 if args.quick else PRODUCTION_HESTON_BATCHES
    slv_default_paths = 256 if args.quick else PRODUCTION_SLV_PATHS_PER_BATCH
    slv_default_batches = 4 if args.quick else PRODUCTION_SLV_BATCHES
    common_paths = args.paths_per_batch
    common_batches = args.batches
    default_paths = heston_default_paths if common_paths is None else common_paths
    default_batches = (
        heston_default_batches if common_batches is None else common_batches
    )
    sampling_by_variant = {
        "heston": {
            "paths_per_batch": int(
                (heston_default_paths if common_paths is None else common_paths)
                if args.heston_paths_per_batch is None
                else args.heston_paths_per_batch
            ),
            "batches": int(
                (heston_default_batches if common_batches is None else common_batches)
                if args.heston_batches is None
                else args.heston_batches
            ),
        },
        "heston_slv": {
            "paths_per_batch": int(
                (slv_default_paths if common_paths is None else common_paths)
                if args.heston_slv_paths_per_batch is None
                else args.heston_slv_paths_per_batch
            ),
            "batches": int(
                (slv_default_batches if common_batches is None else common_batches)
                if args.heston_slv_batches is None
                else args.heston_slv_batches
            ),
        },
    }
    heston_batches_overridden = (
        common_batches is not None or args.heston_batches is not None
    )
    if not args.quick and not heston_batches_overridden:
        heston_batches_by_case = dict(PRODUCTION_HESTON_BATCHES_BY_CASE)
    else:
        heston_batches_by_case = {
            case_name: int(sampling_by_variant["heston"]["batches"])
            for case_name in PRODUCTION_HESTON_BATCHES_BY_CASE
        }
    sampling_by_variant["heston"]["batches_by_case"] = heston_batches_by_case
    slv_batches_overridden = (
        common_batches is not None or args.heston_slv_batches is not None
    )
    if not args.quick and not slv_batches_overridden:
        slv_primary_batches_by_case = dict(PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE)
        slv_outer_batches_by_case = dict(PRODUCTION_SLV_BATCHES_BY_CASE)
    else:
        slv_primary_batches_by_case = {
            case_name: int(sampling_by_variant["heston_slv"]["batches"])
            for case_name in PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE
        }
        slv_outer_batches_by_case = dict(slv_primary_batches_by_case)
    sampling_by_variant["heston_slv"]["batches_by_case"] = slv_outer_batches_by_case
    sampling_by_variant["heston_slv"][
        "primary_batches_by_case"
    ] = slv_primary_batches_by_case
    slv_spot_strata = int(
        (1 if args.quick else SLV_SPOT_STRATA)
        if args.slv_spot_strata is None
        else args.slv_spot_strata
    )
    slv_spot_antithetic = bool(
        (False if args.quick else SLV_SPOT_ANTITHETIC)
        if args.slv_spot_antithetic is None
        else args.slv_spot_antithetic
    )
    slv_spot_bridge_strata = int(
        (1 if args.quick else SLV_SPOT_BRIDGE_STRATA)
        if args.slv_spot_bridge_strata is None
        else args.slv_spot_bridge_strata
    )
    rqmc_batch_workers = int(
        (1 if args.quick else PRODUCTION_RQMC_BATCH_WORKERS)
        if args.rqmc_batch_workers is None
        else args.rqmc_batch_workers
    )
    cell_workers = int(
        (1 if args.quick else PRODUCTION_CELL_WORKERS)
        if args.cell_workers is None
        else args.cell_workers
    )
    for variant, sampling in sampling_by_variant.items():
        if sampling["paths_per_batch"] <= 0 or sampling["batches"] < 2:
            raise ValueError(
                f"{variant}: paths-per-batch must be positive and batches must be >= 2"
            )
        paths_per_batch = sampling["paths_per_batch"]
        if paths_per_batch & (paths_per_batch - 1):
            raise ValueError(f"{variant}: Sobol paths-per-batch must be a power of two")
        if variant == "heston" and any(
            int(batches) < 2 for batches in sampling["batches_by_case"].values()
        ):
            raise ValueError("heston: every case-specific batch count must be >= 2")
        if variant == "heston_slv" and any(
            int(batches) < 2 for batches in sampling["primary_batches_by_case"].values()
        ):
            raise ValueError(
                "heston_slv: every case-specific primary batch count must be >= 2"
            )
    for name, value in (
        ("heston-spot-bridge-strata", args.heston_spot_bridge_strata),
        ("heston-spot-bridge-dimensions", args.heston_spot_bridge_dimensions),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if slv_spot_strata < 1:
        raise ValueError("slv-spot-strata must be a positive integer")
    if slv_spot_bridge_strata < 1:
        raise ValueError("slv-spot-bridge-strata must be a positive integer")
    if (
        args.slv_spot_bridge_dimensions is not None
        and args.slv_spot_bridge_dimensions < 1
    ):
        raise ValueError("slv-spot-bridge-dimensions must be a positive integer")
    if rqmc_batch_workers < 1:
        raise ValueError("rqmc-batch-workers must be a positive integer")
    if cell_workers < 1:
        raise ValueError("cell-workers must be a positive integer")
    cases = certification_cases(quick=args.quick)
    if args.cases:
        wanted = set(args.cases)
        unknown = wanted - {case.name for case in certification_cases(quick=False)}
        if unknown:
            raise ValueError(f"unknown certification cases: {sorted(unknown)}")
        cases = [
            case for case in certification_cases(quick=False) if case.name in wanted
        ]
    if not cases:
        raise ValueError("no certification cases selected")

    if (
        args.heston_spot_bridge_strata is None
        and args.heston_spot_bridge_dimensions is None
    ):
        heston_spot_bridge_profile_by_case = {
            case.name: (
                {"strata": 1, "dimensions": 1}
                if args.quick
                else dict(HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case.name])
            )
            for case in cases
        }
    else:
        override_strata = int(
            HESTON_SPOT_BRIDGE_STRATA
            if args.heston_spot_bridge_strata is None
            else args.heston_spot_bridge_strata
        )
        override_dimensions = int(
            (1 if override_strata == 1 else HESTON_SPOT_BRIDGE_DIMENSIONS)
            if args.heston_spot_bridge_dimensions is None
            else args.heston_spot_bridge_dimensions
        )
        if override_dimensions > 1 and override_strata == 1:
            raise ValueError("heston bridge dimensions > 1 require bridge strata > 1")
        heston_spot_bridge_profile_by_case = {
            case.name: {
                "strata": override_strata,
                "dimensions": override_dimensions,
            }
            for case in cases
        }

    if args.slv_spot_bridge_strata is None and args.slv_spot_bridge_dimensions is None:
        slv_spot_bridge_profile_by_case = {
            case.name: (
                {"strata": 1, "dimensions": 1}
                if args.quick
                else dict(SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case.name])
            )
            for case in cases
        }
    else:
        slv_override_dimensions = int(
            1
            if args.slv_spot_bridge_dimensions is None
            else args.slv_spot_bridge_dimensions
        )
        if slv_override_dimensions > 1 and slv_spot_bridge_strata == 1:
            raise ValueError("SLV bridge dimensions > 1 require bridge strata > 1")
        slv_spot_bridge_profile_by_case = {
            case.name: {
                "strata": slv_spot_bridge_strata,
                "dimensions": slv_override_dimensions,
            }
            for case in cases
        }

    implementation_hash = implementation_sha256()
    # The narrower digest that gates per-cell reuse. Both are recorded: the
    # whole-file hash still answers "what was the source state", while this one
    # answers "may this cell be reused", which is the question a checkpoint asks.
    numerical_hash = numerical_implementation_sha256()
    runtime = runtime_environment()
    selected_variants = tuple(
        variant for variant in ("heston", "heston_slv") if variant in args.variants
    )
    if (
        not args.quick
        and "heston_slv" in selected_variants
        and "heston" not in selected_variants
    ):
        raise ValueError(
            "production Heston-SLV certification requires matching Heston cells"
        )
    selected_sampling = {
        variant: sampling_by_variant[variant] for variant in selected_variants
    }
    if args.quick or args.rqmc_batch_workers is not None:
        rqmc_batch_workers_by_variant_case = {
            variant: {case.name: rqmc_batch_workers for case in cases}
            for variant in selected_variants
        }
    else:
        rqmc_batch_workers_by_variant_case = {
            variant: {
                case.name: PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE[variant][
                    case.name
                ]
                for case in cases
            }
            for variant in selected_variants
        }
    run_configuration = {
        "schema_version": SCHEMA_VERSION,
        "certification_mode": CERTIFICATION_MODE_FULL,
        "implementation_sha256": implementation_hash,
        "numerical_implementation_sha256": numerical_hash,
        "runtime_environment": runtime,
        "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        "quick": bool(args.quick),
        "skip_anchors": bool(args.skip_anchors),
        "reference_seeds": reference_seeds,
        "hedge_inception_spot": float(args.hedge_inception_spot),
        "variants": list(selected_variants),
        "cases": [case.as_dict() for case in cases],
        "sampling_by_variant": selected_sampling,
        "spot_bump": SPOT_BUMP,
        "full_bump_ladder": list(FULL_BUMP_LADDER),
        "stochastic_component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
        "heston_spot_bridge_profile_by_case": (heston_spot_bridge_profile_by_case),
        "slv_spot_bridge_profile_by_case": (slv_spot_bridge_profile_by_case),
        "qe_substeps_by_variant_case": (PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE),
        "slv_multilevel_policy": {
            "cases": sorted(SLV_MULTILEVEL_CASES),
            "mid_paths_per_batch": SLV_MID_CONTROL_PATHS_PER_BATCH,
            "mid_batches": SLV_MID_CONTROL_BATCHES,
            "frozen_control_weight": SLV_FROZEN_CONTROL_WEIGHT,
            "heston_control_weight": SLV_HESTON_CONTROL_WEIGHT,
        },
        "slv_spot_strata": slv_spot_strata,
        "slv_spot_antithetic": slv_spot_antithetic,
        "slv_spot_bridge_strata": slv_spot_bridge_strata,
        "rqmc_batch_workers": rqmc_batch_workers,
        "rqmc_batch_workers_by_variant_case": (rqmc_batch_workers_by_variant_case),
        "cell_workers": cell_workers,
        # Whether cells stopped on their gate or spent their allocation changes
        # what the evidence means, so it belongs in the identity: a resume must
        # not mix fixed-allocation cells with gate-driven ones under one hash.
        "sequential_stopping": (
            {
                "enabled": True,
                "chunk_batches": int(args.sequential_chunk_batches),
                "margin_fraction": float(args.sequential_margin),
                "family_alpha": float(args.sequential_family_alpha),
                "excluded_variant_cases": list(stopping_excluded_variant_cases()),
            }
            if getattr(args, "sequential", False)
            else {"enabled": False}
        ),
    }
    run_configuration_hash = _canonical_sha256(run_configuration)
    print(
        f"[adi-greeks] implementation={implementation_hash[:16]} "
        f"configuration={run_configuration_hash[:16]}",
        flush=True,
    )

    started = time.perf_counter()
    anchors = None
    if args.resume:
        anchors = _load_checkpoint(
            args.output_dir,
            "anchors",
            run_configuration_sha256=run_configuration_hash,
            kind="anchors",
        )
        if anchors is not None:
            print("[adi-greeks] resumed deterministic anchors", flush=True)
    if anchors is None:
        anchors = [] if args.skip_anchors else deterministic_anchors(quick=args.quick)
        _write_checkpoint(
            args.output_dir,
            "anchors",
            run_configuration_sha256=run_configuration_hash,
            kind="anchors",
            evidence=anchors,
        )
    cells_by_variant_case = {}
    heston_cells_by_case = {}

    def cell_identity_for(variant: str, case_name: str) -> str:
        """This cell's identity, including any cell it consumes as an input.

        The multilevel SLV cell reads the matched Heston cell as its high control,
        so that Heston cell's identity is part of the SLV cell's. Without the
        link, a change to the control leaves the consumer looking reusable --
        which is exactly how the 35.5h fleet banked an SLV cell built on a
        gate-truncated control.
        """
        consumed = {}
        if (
            variant == "heston_slv"
            and case_name in SLV_MULTILEVEL_CASES
            and not args.quick
        ):
            consumed[f"heston/{case_name}"] = cell_identity(
                "heston",
                case_name,
                run_configuration,
                numerical_sha256=numerical_hash,
            )
        return cell_identity(
            variant,
            case_name,
            run_configuration,
            numerical_sha256=numerical_hash,
            consumed=consumed,
        )

    def compute_cell(variant: str, case: CaseSpec) -> dict:
        sampling = selected_sampling[variant]
        checkpoint_name = f"{variant}__{case.name}"
        identity = cell_identity_for(variant, case.name)
        cell = None
        if args.resume:
            cell = _load_checkpoint(
                args.output_dir,
                checkpoint_name,
                run_configuration_sha256=run_configuration_hash,
                kind="cell",
                identity_sha256=identity,
            )
            if cell is not None:
                print(f"[adi-greeks] resumed {variant}/{case.name}", flush=True)
        if cell is not None:
            return cell
        print(f"[adi-greeks] {variant}/{case.name}", flush=True)
        heston_bridge_profile = heston_spot_bridge_profile_by_case[case.name]
        slv_bridge_profile = slv_spot_bridge_profile_by_case[case.name]
        variant_seed = (
            reference_seeds["heston"]
            if variant == "heston"
            else reference_seeds["heston_slv_primary"]
        )
        cell_batches = _sampling_primary_batches_for_case(
            sampling, variant, case.name
        )
        cell = certify_case(
            variant,
            case,
            quick=args.quick,
            paths_per_batch=sampling["paths_per_batch"],
            batches=cell_batches,
            seed=variant_seed,
            hedge_inception_spot=args.hedge_inception_spot,
            sequential_policy=build_sequential_policy(
                args, variant, case.name, cap=cell_batches
            ),
            sequential_chunk_batches=args.sequential_chunk_batches,
            heston_spot_bridge_strata=heston_bridge_profile["strata"],
            heston_spot_bridge_dimensions=heston_bridge_profile["dimensions"],
            slv_spot_strata=slv_spot_strata,
            slv_spot_antithetic=slv_spot_antithetic,
            slv_spot_bridge_strata=slv_bridge_profile["strata"],
            slv_spot_bridge_dimensions=slv_bridge_profile["dimensions"],
            slv_mid_control_seed=reference_seeds["heston_slv_mid_control"],
            heston_high_cell=heston_cells_by_case.get(case.name),
            rqmc_batch_workers=(rqmc_batch_workers_by_variant_case[variant][case.name]),
        )
        _write_checkpoint(
            args.output_dir,
            checkpoint_name,
            run_configuration_sha256=run_configuration_hash,
            kind="cell",
            evidence=cell,
            identity_sha256=identity,
        )
        print(
            f"[adi-greeks] checkpointed {variant}/{case.name}: " f"{cell['status']}",
            flush=True,
        )
        return cell

    def compute_phase(
        variant: str, phase_cases: Sequence[CaseSpec], workers: int
    ) -> None:
        if not phase_cases:
            return
        if workers == 1:
            for case in phase_cases:
                cells_by_variant_case[(variant, case.name)] = compute_cell(
                    variant, case
                )
            return
        with ThreadPoolExecutor(
            max_workers=min(int(workers), len(phase_cases)),
            thread_name_prefix=f"adi-cell-{variant}",
        ) as pool:
            futures = {
                pool.submit(compute_cell, variant, case): case for case in phase_cases
            }
            for future in as_completed(futures):
                case = futures[future]
                cells_by_variant_case[(variant, case.name)] = future.result()

    if "heston" in selected_variants:
        # Feed one long, low-worker cell before short, high-worker cells so the
        # two-cell pool uses CPU without co-scheduling two large Sobol matrices.
        heston_execution_order = (
            "ordinary_full",
            "near_ko",
            "ordinary_decayed",
            "near_ki",
            "sigma_collapse",
            "near_expiry",
            "low_feller",
        )
        case_by_name = {case.name: case for case in cases}
        ordered_heston_cases = [
            case_by_name[name]
            for name in heston_execution_order
            if name in case_by_name
        ]
        compute_phase("heston", ordered_heston_cases, cell_workers)
        heston_cells_by_case.update(
            {case.name: cells_by_variant_case[("heston", case.name)] for case in cases}
        )
    if "heston_slv" in selected_variants:
        smooth_slv_cases = [
            case for case in cases if case.name not in SLV_MULTILEVEL_CASES
        ]
        hard_slv_cases = [case for case in cases if case.name in SLV_MULTILEVEL_CASES]
        compute_phase("heston_slv", smooth_slv_cases, cell_workers)
        compute_phase("heston_slv", hard_slv_cases, 1)

    cells = [
        cells_by_variant_case[(variant, case.name)]
        for variant in selected_variants
        for case in cases
    ]
    decisions = make_decisions(
        cells,
        anchors,
        quick=(args.quick or args.skip_anchors),
        variants=selected_variants,
        sampling_by_variant=selected_sampling,
        heston_spot_bridge_profile_by_case=(heston_spot_bridge_profile_by_case),
        slv_spot_strata=slv_spot_strata,
        slv_spot_antithetic=slv_spot_antithetic,
        slv_spot_bridge_strata=slv_spot_bridge_strata,
        slv_spot_bridge_profile_by_case=(slv_spot_bridge_profile_by_case),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
        "certification_mode": CERTIFICATION_MODE_FULL,
        "created_at": datetime.now().astimezone().isoformat(),
        "quick": bool(args.quick),
        "profile": (
            "quick (non-production)"
            if args.quick
            else (
                "diagnostic subset (non-admissible)"
                if args.skip_anchors
                or {case.name for case in cases}
                != {case.name for case in certification_cases(quick=False)}
                else "production"
            )
        ),
        "reference_seeds": reference_seeds,
        "paths_per_batch": int(default_paths),
        "batches": int(default_batches),
        "sampling_by_variant": selected_sampling,
        "confidence": CONFIDENCE,
        "python": runtime["python_version"],
        "runtime_environment": runtime,
        "implementation_sha256": implementation_hash,
        "numerical_implementation_sha256": numerical_hash,
        # Per-cell identities, so a reader can see what each cell's reuse was
        # predicated on. Kept beside the cells rather than inside them, so that
        # adding this does not perturb `_cell_evidence_sha256`, which the SLV
        # multilevel estimator uses to pin its Heston control.
        "cell_identities": {
            f"{variant}/{case.name}": cell_identity_for(variant, case.name)
            for variant in selected_variants
            for case in cases
        },
        "run_configuration_sha256": run_configuration_hash,
        "run_configuration": run_configuration,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "delta_cell_bound_contracts": DELTA_CELL_BOUND_CONTRACTS,
            "delta_bias_bound_contracts": DELTA_BIAS_BOUND_CONTRACTS,
            "gamma_cell_bound_contracts_for_1pct_move": GAMMA_CELL_BOUND_CONTRACTS,
            "hedge_inception_spot": float(args.hedge_inception_spot),
            "hedge_inception_spot_policy": (
                "strictest minimum from the frozen 27-inception MO cohort"
            ),
            "unresolved_route": "excluded_greek_unresolved",
            "heston_spot_bridge_profile_by_case": (heston_spot_bridge_profile_by_case),
            "slv_spot_bridge_profile_by_case": (slv_spot_bridge_profile_by_case),
            "qe_substeps_by_variant_case": (PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE),
            "slv_multilevel_policy": {
                "cases": sorted(SLV_MULTILEVEL_CASES),
                "mid_paths_per_batch": SLV_MID_CONTROL_PATHS_PER_BATCH,
                "mid_batches": SLV_MID_CONTROL_BATCHES,
                "frozen_control_weight": SLV_FROZEN_CONTROL_WEIGHT,
                "heston_control_weight": SLV_HESTON_CONTROL_WEIGHT,
            },
            "slv_spot_strata": slv_spot_strata,
            "slv_spot_antithetic": slv_spot_antithetic,
            "slv_spot_bridge_strata": slv_spot_bridge_strata,
            "rqmc_batch_workers": rqmc_batch_workers,
            "rqmc_batch_workers_by_variant_case": (rqmc_batch_workers_by_variant_case),
            "cell_workers": cell_workers,
            "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        },
        "anchors": anchors,
        "cells": cells,
        "decisions": decisions,
    }
    publish_payload(payload, args.output_dir)
    print(f"[adi-greeks] wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
