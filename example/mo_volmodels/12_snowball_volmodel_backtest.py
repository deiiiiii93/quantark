"""Stage 12 - Multi-inception snowball vol-model backtest runner (Phase 4).

Runs the production 3Y 000852.SH snowball (short, delta-hedged with IM
futures) over a fleet of monthly inceptions, pricing each day under six
model variants:

    flat_bsm       ATM IV at the product's REMAINING maturity, refreshed daily
    flat_bsm_quad  Engine control: flat_bsm's market data on the quadrature
                   engine instead of the 1D PDE (isolates engine from model)
    ts_bsm         ATM pillar term structure (TermStructureVolSurface)
    localvol       Dupire local vol off the full SABR-smoothed smile grid
    heston         Heston, per-day calibrated (Lewis, frozen mo_volmodels bounds)
    heston_slv     Heston-SLV, per-day calibrated leverage surface

Every variant of a given inception shares ONE set of contractual terms: the
fair coupon is solved once at inception under flat BSM (Gate G4), so the
comparison isolates the pricing/hedging model rather than the term sheet.

Engine routing follows the Gate G2 decision recorded by stage 11
(``output/pde_convergence_gate/gate_decision.json``): every one of the six
study variants -- ``flat_bsm``, ``flat_bsm_quad``, ``ts_bsm``, ``localvol``,
``heston``, ``heston_slv`` -- reads its own route from that file's
``variants`` map (keyed by the study variant name, not by ``vol_model``:
flat_bsm/flat_bsm_quad/ts_bsm all share ``vol_model="bsm"`` but can carry
independent verdicts). The two 2-D variants have an additional fail-closed
Stage-17 Greek admission (`output/adi_greek_certification/`): they run only
when Stage 11 admits the PDE PV and Stage 16 admits its delta/gamma. An
unresolved variant is recorded as `excluded_greek_unresolved`; it is never
redirected to a noisy daily RQMC hedge. See `GateRouting.solver_for`,
`ADIGreekRouting`, and `apply_adi_greek_admission`. The run manifest records
both decision files, both evidence hashes, and every excluded variant.

Trade lifecycle per inception: the replay runs from inception until knock-out,
maturity, or the end of the data window, whichever comes first.  Runs that hit
the data window first are recorded as ``censored_at_data_end`` rather than
silently treated as matured.

Outputs, per inception x variant, under ``--out-dir``:

    runs/<inception>/<variant>/states.csv      daily market + position + PnL
                              /greeks.csv      daily greeks path
                              /trades.csv      futures hedge trades
                              /rebalances.csv  rebalance decisions
                              /actions.csv     lifecycle actions (KO/KI/...)
                              /daily_events.csv, event_probabilities.csv
                              /calibration_records.json  per-day model fit
                              /run_summary.json
    inceptions.json   solved terms + coupon-solver diagnostics per inception
    run_manifest.json fleet configuration, gate provenance, per-run status

Run:
    .venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --gate-g3
    .venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest.replay import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    FuturesRollPolicy,
    SurfaceGridConfig,
    create_pricing_engine,
)
from quantark.backtest.replay.config import VolModelCalibrationConfig
from quantark.backtest.replay.market import derive_implied_dividend_yield
from quantark.backtest.replay.strategy_state import AutocallableDeltaHedgeStrategy
from quantark.param.vol.surface_history import VolSurfaceHistory
from quantark.backtest.transaction_costs import CompleteCostModel, ZeroCostModel
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE11_PATH = Path(__file__).resolve().parent / "11_pde_convergence_gate.py"
STAGE16_PATH = Path(__file__).resolve().parent / "16_adi_greek_certification.py"
STAGE17_PATH = Path(__file__).resolve().parent / "17_adi_slv_aggregate_certification.py"

# ---------------------------------------------------------------------------
# Fleet configuration
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"
DEFAULT_GATE_DECISION = PROJECT_ROOT / "output/pde_convergence_gate/gate_decision.json"
DEFAULT_ADI_GREEK_DECISION = (
    PROJECT_ROOT
    / "docs/modelvalidation/certificates/adi2d-snowball-greeks"
    / "2026-08-19/certificate.json"
)
# The legacy source: the raw stage-17 decision. Still accepted, but it sits
# beside 14.4 MB of Monte-Carlo rows that the repository deliberately does not
# carry, so it only resolves on a machine holding them.
LEGACY_ADI_GREEK_DECISION = (
    PROJECT_ROOT
    / "output/adi_greek_certification/adi_greek_certification_decision.json"
)
ADI_GREEK_DECISION_SCHEMA_VERSION = 12
DEFAULT_OUT_DIR = PROJECT_ROOT / "output/volmodel_backtest"

ADI_2D_PRODUCTION_ENGINE_CONTROLS = {
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

UNDERLYING_NAME = "CSI1000"


def _load_cohort():
    """Import the sibling cohort module (the stages are not a package)."""
    path = Path(__file__).resolve().parent / "cohort.py"
    spec = importlib.util.spec_from_file_location("mo_cohort", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_cohort"] = module
    spec.loader.exec_module(module)
    return module


cohort = _load_cohort()

# The frozen cohort as-of.  The replay window pins to THIS, never to the tail
# of the spot cache, which a live launchd job extends every weekday.
COHORT_ASOF = cohort.COHORT_ASOF

NOTIONAL = 50_000_000.0  # design doc term sheet
PRODUCT_QUANTITY = -1.0  # seller (short the snowball)

# Inception schedule: monthly starts; an inception is admitted only when the
# data window leaves it at least this much observable life, so every trade
# clears the 3-month KO lockout and sees >= 9 monthly observations.
MIN_OBSERVABLE_MONTHS = 12

# IM futures hedging costs (owner decision 2026-07-25): realistic CFFEX
# all-in - 0.5bp commission per side plus roughly half a tick of spread.
COST_PROPORTIONAL_RATE = 5e-5
COST_SPREAD_BPS = 1.0

# Fair-coupon solver (Gate G4).  The PV tolerance scales with notional; the
# solver fails closed rather than returning an unconverged/boundary coupon.
COUPON_LOWER = 0.0
COUPON_UPPER = 0.80
COUPON_PV_TOL_FRACTION = 1e-6  # of notional -> 50 CNY at 50M
COUPON_MAX_ITERATIONS = 60

# RQMC-QE reference configuration certified by the Gate G2 study.
MC_SEED = 20260723
MC_PATHS_PER_BATCH = 8192
MC_BATCHES = 16

# Quick mode: G3 smoke / CI.  Small MC and a short window.
QUICK_MC_PATHS_PER_BATCH = 512
QUICK_MC_BATCHES = 2
QUICK_MAX_DAYS = 25

SCHEMA_VERSION = 1

VARIANTS: Tuple[str, ...] = (
    "flat_bsm",
    "flat_bsm_quad",
    "ts_bsm",
    "localvol",
    "heston",
    "heston_slv",
)


@dataclass(frozen=True)
class VariantSpec:
    """How one model variant configures the daily pricing engine."""

    name: str
    vol_source: str
    surface_vol_mode: str
    vol_model: str
    description: str
    # Engine family for the DAILY PRICING engine.  Defaults to PDE so the
    # five original variants are untouched; flat_bsm_quad overrides it to
    # make an engine control that differs from flat_bsm in engine only.
    # The surface and event-stats engines stay 1D PDE for every variant.
    pricing_engine_type: EngineType = EngineType.PDE

    def uses_calibration(self) -> bool:
        return self.vol_model != "bsm"


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "flat_bsm": VariantSpec(
        name="flat_bsm",
        vol_source="surface",
        surface_vol_mode="flat_atm_remaining",
        vol_model="bsm",
        description="Flat BSM at the ATM IV of the product's remaining maturity",
    ),
    "flat_bsm_quad": VariantSpec(
        name="flat_bsm_quad",
        vol_source="surface",
        surface_vol_mode="flat_atm_remaining",
        vol_model="bsm",
        description=(
            "Engine control: flat_bsm's market data priced by FFT "
            "regime-switching quadrature instead of the 1D PDE"
        ),
        pricing_engine_type=EngineType.QUADRATURE,
    ),
    "ts_bsm": VariantSpec(
        name="ts_bsm",
        vol_source="surface",
        surface_vol_mode="term_structure",
        vol_model="bsm",
        description="BSM on the ATM pillar term structure",
    ),
    "localvol": VariantSpec(
        name="localvol",
        vol_source="surface",
        surface_vol_mode="full_grid",
        vol_model="localvol",
        description="Dupire local volatility from the full smile grid",
    ),
    "heston": VariantSpec(
        name="heston",
        vol_source="surface",
        surface_vol_mode="full_grid",
        vol_model="heston",
        description="Heston stochastic volatility, calibrated per day",
    ),
    "heston_slv": VariantSpec(
        name="heston_slv",
        vol_source="surface",
        surface_vol_mode="full_grid",
        vol_model="heston_slv",
        description="Heston-SLV with a per-day calibrated leverage surface",
    ),
}


# ---------------------------------------------------------------------------
# Stage 11 reuse (term sheet must be IDENTICAL to what Gate G2 certified)
# ---------------------------------------------------------------------------

_STAGE11 = None
_STAGE16 = None
_STAGE17 = None


def stage11():
    """Load stage 11 as a module; its term sheet is the certified one.

    The 3Y term sheet (KO/KI schedules, lockout, ACT/365 conventions) is
    defined once, in stage 11, and validated there by the convergence gate.
    Re-deriving it here would let the backtest silently drift away from what
    was certified, so it is imported instead.
    """
    global _STAGE11
    if _STAGE11 is None:
        spec = importlib.util.spec_from_file_location(
            "mo_pde_convergence_gate_11", STAGE11_PATH
        )
        if spec is None or spec.loader is None:
            raise ValidationError(f"cannot load stage 11 module at {STAGE11_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _STAGE11 = module
    return _STAGE11


def stage16():
    """Load the certification module for live source/runtime verification."""
    global _STAGE16
    if _STAGE16 is None:
        spec = importlib.util.spec_from_file_location(
            "mo_adi_greek_certification_16", STAGE16_PATH
        )
        if spec is None or spec.loader is None:
            raise ValidationError(f"cannot load stage 16 module at {STAGE16_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _STAGE16 = module
    return _STAGE16


def stage17():
    """Load the aggregate-only Greek amendment for live validation."""
    global _STAGE17
    if _STAGE17 is None:
        spec = importlib.util.spec_from_file_location(
            "mo_adi_slv_aggregate_certification_17", STAGE17_PATH
        )
        if spec is None or spec.loader is None:
            raise ValidationError(f"cannot load stage 17 module at {STAGE17_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _STAGE17 = module
    return _STAGE17


# ---------------------------------------------------------------------------
# Gate G2 routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRouting:
    """Per-variant solver routing read from the stage 11 gate decision."""

    decision_path: str
    evidence_sha256: Optional[str]
    routes: Dict[str, str]  # study variant name -> "pde" | "mc"
    pde_params: Dict[str, Dict[str, Any]]
    # Per-variant MC config the gate certified (paths_per_batch/batches/seed/
    # scheme/substeps_per_interval -- see _reference_params_block in stage
    # 11).  Defaults to {} so the several existing 4-positional-arg
    # GateRouting(...) constructions in the test suite keep working; the
    # fail-closed check for an MC-routed variant with no entry here lives in
    # make_engine_config, not here (mirrors solver_for's fail-closed style).
    mc_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def solver_for(self, variant: str) -> str:
        """Route for one of the six study variants.

        ``self.routes`` is keyed by the study VARIANT name (``flat_bsm``,
        ``flat_bsm_quad``, ``ts_bsm``, ``localvol``, ``heston``,
        ``heston_slv``) -- exactly the keys ``build_decision_payload``
        (stage 11) writes into the decision's ``variants`` map, per
        ``variant`` in ``GATE_PAIRS`` -- not by ``vol_model``.  That
        distinction matters because ``flat_bsm``, ``flat_bsm_quad`` and
        ``ts_bsm`` all share ``vol_model == "bsm"`` yet can carry
        independent gate verdicts.  There is no fallback for a missing or
        unrecognised route: after Task 4 every variant, including the four
        1D/quad ones, is inside the gate's scope, so an absent or malformed
        entry must raise rather than silently default to PDE.
        """
        route = self.routes.get(variant)
        if route not in ("pde", "mc"):
            raise ValidationError(
                f"gate decision has no usable route for variant={variant!r} "
                f"(got {route!r}); rerun stage 11 before the fleet"
            )
        return route

    def engine_options_for(self, variant: str) -> Dict[str, Any]:
        """PDE grid options for a variant the gate admitted to the PDE route.

        Only the 2D-ADI family (``heston``, ``heston_slv``) records
        ``n_x``/``n_v``/``n_t`` plus the certified variance-grid, V-drift,
        V=0-boundary, and dense-KI-clock controls in its ``pde_params``
        block; the four 1D/quad variants record their own ladder knob (``accuracy`` /
        ``grid_points``) instead (see ``_production_params_block`` in stage
        11), so the dict comprehension below is naturally empty for them --
        no name-based special case is needed.
        """
        if self.solver_for(variant) != "pde":
            return {}
        params = self.pde_params.get(variant, {})
        options = {
            key: int(params[key]) for key in ("n_x", "n_v", "n_t") if key in params
        }
        if variant in {"heston", "heston_slv"}:
            mismatches = {
                key: (params.get(key), expected)
                for key, expected in ADI_2D_PRODUCTION_ENGINE_CONTROLS.items()
                if params.get(key) != expected
            }
            if mismatches:
                raise ValidationError(
                    f"gate decision has stale 2-D production controls for "
                    f"variant={variant!r}: {mismatches}; rerun stage 11"
                )
            options.update(ADI_2D_PRODUCTION_ENGINE_CONTROLS)
        return options


@dataclass(frozen=True)
class ADIGreekRouting:
    """Production-only Greek admission for the two 2-D ADI variants."""

    decision_path: str
    evidence_sha256: str
    implementation_sha256: str
    run_configuration_sha256: str
    decision_sha256: str
    runtime_environment: Dict[str, str]
    production_engine_controls: Dict[str, Any]
    run_configuration: Dict[str, Any]
    routes: Dict[str, str]
    reasons: Dict[str, str]

    def route_for(self, variant: str) -> str:
        route = self.routes.get(variant)
        if route not in {"pde", "excluded_greek_unresolved"}:
            raise ValidationError(
                f"ADI Greek decision has no usable route for {variant!r}; "
                "run stage 17 production aggregate certification"
            )
        return route


def load_gate_routing(path: Path) -> GateRouting:
    try:
        payload = json.loads(Path(path).read_text())
    except OSError as exc:
        raise ValidationError(
            f"gate decision not readable at {path}; run stage 11 "
            "(11_pde_convergence_gate.py) before the fleet"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"gate decision {path} is not valid JSON: {exc}") from exc
    variants = payload.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValidationError(f"gate decision {path} has no 'variants' map")
    routes: Dict[str, str] = {}
    pde_params: Dict[str, Dict[str, Any]] = {}
    mc_params: Dict[str, Dict[str, Any]] = {}
    for name, entry in variants.items():
        if not isinstance(entry, dict):
            raise ValidationError(
                f"gate decision {path}: variant {name!r} is not an object"
            )
        if str(name) in {"heston", "heston_slv"}:
            gate = entry.get("gate")
            if (
                not isinstance(gate, dict)
                or gate.get("delta_authority") != "stage16"
                or gate.get("delta_required") is not False
            ):
                raise ValidationError(
                    f"gate decision {path}: variant {name!r} does not "
                    "delegate ADI Greek admission to Stage 16; rerun or "
                    "rescore stage 11"
                )
        routes[str(name)] = str(entry.get("route", ""))
        params = entry.get("pde_params")
        if isinstance(params, dict):
            pde_params[str(name)] = params
        mc = entry.get("mc_params")
        if isinstance(mc, dict):
            mc_params[str(name)] = mc
    return GateRouting(
        decision_path=str(path),
        evidence_sha256=payload.get("evidence_sha256"),
        routes=routes,
        pde_params=pde_params,
        mc_params=mc_params,
    )


ADI_GREEK_CERTIFICATE_STUDY = "adi2d-snowball-greeks"

# The certified candidate names, and the study variant each one admits.
CERTIFIED_CANDIDATE_VARIANTS = {
    "equity.snowball.heston_pde": "heston",
    "equity.snowball.heston_slv_pde": "heston_slv",
}


def load_adi_greek_admission(path: Path | str) -> ADIGreekRouting:
    """Load Greek admission from whichever artifact ``path`` names.

    A modelvalidation certificate always carries ``projected_sha256``; a
    stage-17 decision always carries ``schema_version``. Neither key is
    optional in its own format, so this dispatches on shape rather than
    guessing, and refuses a payload that is neither instead of trying both.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValidationError(
            f"ADI Greek admission artifact not readable at {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path} is not valid JSON: {exc}") from exc
    if "projected_sha256" in payload:
        return load_adi_greek_routing_from_certificate(path)
    if "schema_version" in payload:
        return load_adi_greek_routing(path)
    raise ValidationError(
        f"{path} is neither a modelvalidation certificate (projected_sha256) "
        "nor a stage-17 decision (schema_version)"
    )


def load_adi_greek_routing_from_certificate(path: Path | str) -> ADIGreekRouting:
    """Greek admission from the committed modelvalidation certificate.

    Preferred over ``load_adi_greek_routing``, which reads the raw stage-17
    payload.  Those payloads are 14.4 MB of Monte-Carlo row dumps and are
    deliberately NOT committed -- the repository publishes their identity
    instead -- so routing from them only ever worked on a machine that happened
    to hold them.  The certificate is committed, recomputes its own digest, and
    ``test/modelvalidation/test_banked_certificates.py`` re-runs both ADI
    solvers over all fourteen banked cells on every commit, so the claim that
    the engines still produce the certified numbers is continuously checked
    rather than asserted once.

    Fails closed at five seams: the modelvalidation validator (structure,
    verdicts, decision vocabulary, and a RECOMPUTED ``projected_sha256``), the
    study name, the quick flag, every cell's verdict, and every aggregate's
    verdict.  A candidate that is not ADMITTED routes to
    ``excluded_greek_unresolved``; stage 12 then excludes that variant outright
    rather than substituting daily MC Greeks.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValidationError(
            f"modelvalidation certificate not readable at {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"modelvalidation certificate {path} is not valid JSON: {exc}"
        ) from exc

    from quantark.modelvalidation import validate_payload as _validate_certificate

    try:
        _validate_certificate(payload)
    except Exception as exc:  # noqa: BLE001 - re-raised as this module's type
        raise ValidationError(
            f"modelvalidation certificate failed closed validation: {exc}"
        ) from exc

    study = payload.get("study") or {}
    if study.get("name") != ADI_GREEK_CERTIFICATE_STUDY:
        raise ValidationError(
            f"certificate {path} is study {study.get('name')!r}, not "
            f"{ADI_GREEK_CERTIFICATE_STUDY!r}"
        )
    if study.get("quick") is not False:
        raise ValidationError("certificate is quick/non-production evidence")

    verdicts = {cell.get("verdict") for cell in payload.get("cells") or []}
    if not verdicts or verdicts != {"PASS"}:
        raise ValidationError(
            f"certificate carries non-PASS cells: {sorted(v for v in verdicts)}"
        )
    aggregates = payload.get("aggregates") or []
    if not aggregates:
        raise ValidationError("certificate carries no aggregate verdict")
    for aggregate in aggregates:
        if aggregate.get("passed") is not True or (
            aggregate.get("within_bound") is not True
        ):
            raise ValidationError(
                "certificate aggregate did not pass for "
                f"{aggregate.get('candidate')!r}/{aggregate.get('quantity')!r}"
            )

    decisions = payload.get("decisions") or {}
    routes: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    for candidate, variant in CERTIFIED_CANDIDATE_VARIANTS.items():
        decision = decisions.get(candidate)
        if decision is None:
            raise ValidationError(
                f"certificate carries no decision for candidate {candidate!r}"
            )
        if decision == "ADMITTED":
            routes[variant] = "pde"
            reasons[variant] = (
                f"{candidate} ADMITTED by {ADI_GREEK_CERTIFICATE_STUDY} "
                f"({path.parent.name})"
            )
        else:
            routes[variant] = "excluded_greek_unresolved"
            reasons[variant] = (
                f"{candidate} decision is {decision!r}, not ADMITTED"
            )

    digests = (payload.get("imported") or {}).get("source_digests") or {}
    return ADIGreekRouting(
        decision_path=str(path),
        evidence_sha256=str(payload["projected_sha256"]),
        implementation_sha256=str(digests.get("stage16_implementation_sha256") or ""),
        run_configuration_sha256=str(
            digests.get("stage16_run_configuration_sha256") or ""
        ),
        decision_sha256=str(digests.get("stage17_decision_sha256") or ""),
        runtime_environment=dict(payload.get("runtime") or {}),
        # The certificate does not publish the resolved engine controls
        # machine-readably (its report.md carries them as prose), so this
        # records what IS published rather than re-deriving them from a
        # different authority and calling it provenance.
        production_engine_controls={},
        run_configuration={
            "source": "modelvalidation_certificate",
            "study": study.get("name"),
            "quantities": study.get("quantities"),
            "bounds": study.get("bounds"),
            "source_digests": dict(digests),
        },
        routes=routes,
        reasons=reasons,
    )


def load_adi_greek_routing(path: Path) -> ADIGreekRouting:
    """Load the hashed, production-sized Stage-17 amendment artifact."""
    try:
        payload = json.loads(Path(path).read_text())
    except OSError as exc:
        raise ValidationError(
            f"ADI Greek decision not readable at {path}; run "
            "17_adi_slv_aggregate_certification.py before any Heston/SLV fleet"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"ADI Greek decision {path} is not valid JSON: {exc}"
        ) from exc
    if payload.get("schema_version") != ADI_GREEK_DECISION_SCHEMA_VERSION:
        raise ValidationError(
            "ADI Greek decision uses a stale schema; rerun stage 17 with the "
            "aggregate-only paired-endpoint certification contract"
        )
    if payload.get("quick") is not False:
        raise ValidationError(
            "ADI Greek decision is quick/non-production evidence; rerun stage 17 "
            "without --quick"
        )
    certification = stage17()
    if certification.SCHEMA_VERSION != ADI_GREEK_DECISION_SCHEMA_VERSION:
        raise ValidationError(
            "Stage-12/Stage-17 ADI Greek schema mismatch in this checkout"
        )
    evidence_hash = payload.get("evidence_sha256")
    hashes = {
        "evidence_sha256": evidence_hash,
        "implementation_sha256": payload.get("implementation_sha256"),
        "run_configuration_sha256": payload.get("run_configuration_sha256"),
        "decision_sha256": payload.get("decision_sha256"),
    }
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in hashes.values()
    ):
        raise ValidationError("ADI Greek decision has incomplete hash provenance")

    evidence_path = Path(path).with_name("adi_greek_certification.json")
    try:
        evidence = json.loads(evidence_path.read_text())
    except OSError as exc:
        raise ValidationError(
            "ADI Greek decision lacks its sibling full evidence JSON; Stage 12 "
            "will not route from a compact decision alone"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"ADI Greek evidence {evidence_path} is not valid JSON: {exc}"
        ) from exc
    try:
        if evidence.get("evidence_sha256") != evidence_hash or (
            certification._projected_evidence_sha256(evidence) != evidence_hash
        ):
            raise ValueError("evidence hash mismatch")
        certification.validate_payload(evidence)
        expected_decision = certification.build_decision_payload(evidence)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"ADI Greek evidence failed closed validation: {exc}"
        ) from exc
    if payload != expected_decision:
        raise ValidationError(
            "ADI Greek decision does not exactly match its validated sibling evidence"
        )

    run_configuration = payload["run_configuration"]
    runtime = payload["runtime_environment"]
    production_controls = payload["production_engine_controls"]
    decision_hash = payload["decision_sha256"]
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict) or not decisions:
        raise ValidationError("ADI Greek decision has no decisions map")
    routes: Dict[str, str] = {}
    reasons: Dict[str, str] = {}
    for variant, row in decisions.items():
        if not isinstance(row, dict):
            raise ValidationError(
                f"ADI Greek decision entry {variant!r} is not an object"
            )
        route = str(row.get("route", ""))
        if route not in {"pde", "excluded_greek_unresolved"}:
            raise ValidationError(
                f"ADI Greek decision entry {variant!r} has invalid route {route!r}"
            )
        if route == "pde" and row.get("evidence_complete") is not True:
            raise ValidationError(
                f"ADI Greek decision entry {variant!r} admits PDE on incomplete evidence"
            )
        routes[str(variant)] = route
        reasons[str(variant)] = str(row.get("reason", ""))
    return ADIGreekRouting(
        decision_path=str(path),
        evidence_sha256=evidence_hash,
        implementation_sha256=hashes["implementation_sha256"],
        run_configuration_sha256=hashes["run_configuration_sha256"],
        decision_sha256=decision_hash,
        runtime_environment=dict(runtime),
        production_engine_controls=dict(production_controls),
        run_configuration=dict(run_configuration),
        routes=routes,
        reasons=reasons,
    )


def apply_adi_greek_admission(
    variants: Sequence[str],
    gate: GateRouting,
    greek_gate: ADIGreekRouting,
) -> Tuple[List[str], Dict[str, str]]:
    """Exclude unresolved 2-D variants; never substitute daily MC Greeks."""
    admitted: List[str] = []
    excluded: Dict[str, str] = {}
    for variant in variants:
        if variant not in {"heston", "heston_slv"}:
            admitted.append(variant)
            continue
        pv_route = gate.solver_for(variant)
        greek_route = greek_gate.route_for(variant)
        if pv_route == "pde" and greek_route == "pde":
            try:
                gate.engine_options_for(variant)
            except ValidationError as exc:
                excluded[variant] = str(exc)
            else:
                admitted.append(variant)
            continue
        reasons = []
        if pv_route != "pde":
            reasons.append(f"Stage 11 PV route is {pv_route!r}, not admitted PDE")
        if greek_route != "pde":
            reasons.append(
                greek_gate.reasons.get(variant)
                or "Stage 17 Greek evidence is unresolved"
            )
        excluded[variant] = "; ".join(reasons)
    return admitted, excluded


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

_HISTORY_CACHE: Dict[str, VolSurfaceHistory] = {}


def surface_history(history_dir: Path | str) -> VolSurfaceHistory:
    """One VolSurfaceHistory per directory per process.

    Artifacts are loaded lazily and sha-verified on first touch, so a 3Y run
    materialises ~730 of them (~40 MB).  Workers run several tasks in
    sequence over the SAME calendar, so caching per process avoids re-reading
    and re-hashing that set once per run.
    """
    key = str(history_dir)
    cached = _HISTORY_CACHE.get(key)
    if cached is None:
        cached = VolSurfaceHistory(key)
        _HISTORY_CACHE[key] = cached
    return cached


def load_spot_frame(history_dir: Path) -> pd.DataFrame:
    path = Path(history_dir) / "csi1000_spot.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    if "spot" not in frame.columns:
        raise ValidationError(f"{path} must have a 'spot' column")
    return frame.sort_values("date").reset_index(drop=True)


def load_futures_frame(history_dir: Path) -> pd.DataFrame:
    path = Path(history_dir) / "im_futures.csv"
    frame = pd.read_csv(path, parse_dates=["date", "expiry_date"])
    required = {"date", "contract", "futures_price", "expiry_date", "multiplier"}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"{path} missing columns: {sorted(missing)}")
    return frame.sort_values(["date", "expiry_date"]).reset_index(drop=True)


def atm_vol_channel(
    dates: Sequence[pd.Timestamp],
    history: VolSurfaceHistory,
    tenor_years: float = 1.0,
) -> pd.DataFrame:
    """Daily scalar vol channel: ATM IV at ``tenor_years`` off that day's surface.

    Surface-mode variants never price off this column, but the dataset's date
    calendar is the intersection of the spot/vol/rate channels, and the column
    is recorded in the daily state rows.  It is a real read from the admitted
    artifact (carry-forward per the manifest gap policy) rather than an
    invented constant.
    """
    rows = []
    for stamp in dates:
        artifact = history.surface_for(pd.Timestamp(stamp).date())
        surface = artifact.term_structure_vol_surface()
        rows.append(
            {
                "date": pd.Timestamp(stamp),
                "volatility": float(surface.get_vol(0.0, float(tenor_years), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def flat_rate_channel(dates: Sequence[pd.Timestamp], rate: float) -> pd.DataFrame:
    return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates], "rate": float(rate)})


def build_market_dataset(
    *,
    spot: pd.DataFrame,
    futures: pd.DataFrame,
    history: VolSurfaceHistory,
    rate: float,
) -> AutocallableMarketDataSet:
    vol = atm_vol_channel(list(spot["date"]), history)
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot,
        vol_data=vol,
        rate_data=flat_rate_channel(list(spot["date"]), rate),
        futures_data=futures,
        surface_history=history,
        metadata={
            "spot_symbol": "000852.SH",
            "futures_prefix": "IM",
            "vol_source": "mo_iv_surface_history",
            "vol_channel": "atm_1y_from_admitted_surface",
            "rate_source": "flat",
        },
    )


# ---------------------------------------------------------------------------
# Inception scheduling
# ---------------------------------------------------------------------------


def schedule_inceptions(
    *,
    calendar,
    data_start: date,
    data_end: date,
    first_admitted_surface: date,
    min_observable_months: int = MIN_OBSERVABLE_MONTHS,
    maturity_months: Optional[int] = None,
) -> List[date]:
    """Monthly inception dates with at least ``min_observable_months`` of data.

    Inceptions are month starts snapped forward to a trading day.  An
    inception is admitted only when it is inside the data window, has an
    admitted IV surface on or before it, and leaves at least
    ``min_observable_months`` of observable calendar before the data ends.
    Trades whose maturity exceeds the data end are still admitted; they are
    recorded as censored when the replay stops at the window edge.
    """
    s11 = stage11()
    maturity_months = int(
        maturity_months if maturity_months is not None else s11.MATURITY_MONTHS
    )
    if min_observable_months < 0:
        raise ValidationError("min_observable_months must be non-negative")

    out: List[date] = []
    cursor = date(data_start.year, data_start.month, 1)
    while cursor <= data_end:
        inception = calendar.next_trading_day(max(cursor, data_start))
        cursor = s11.add_months(cursor, 1)
        if inception > data_end or inception < data_start:
            continue
        if inception < first_admitted_surface:
            continue  # no surface to price or calibrate against
        horizon = s11.add_months(inception, min_observable_months)
        if horizon > data_end:
            continue
        if out and out[-1] == inception:
            continue
        out.append(inception)
    return out


# ---------------------------------------------------------------------------
# Product construction
# ---------------------------------------------------------------------------


def build_backtest_product(
    terms,
    *,
    initial_spot: float,
    coupon: float,
    notional: float = NOTIONAL,
) -> SnowballOption:
    """The stage-11 term sheet, sized to ``notional`` and carrying the coupon.

    Mirrors ``stage11.build_snowball_product`` field for field; it differs
    only in the three things a backtest needs and a pricing gate does not:
    the position size (``contract_multiplier``), the solved coupon, and
    ``initial_date`` (the lifecycle tracker resolves the observation schedule
    from it).  ``test_stage12_backtest_runner.py`` asserts the two products
    agree on every contractual field.
    """
    initial_spot = float(initial_spot)
    coupon = float(coupon)
    product = create_standard_snowball(
        initial_price=initial_spot,
        strike=initial_spot,
        maturity=float(terms.maturity_years),
        contract_multiplier=float(notional) / initial_spot,
        ko_barrier=terms.ko_pct * initial_spot,
        ko_rate=coupon,
        ki_barrier=terms.ki_pct * initial_spot,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=coupon,
        include_principal=False,
    )
    # The lifecycle tracker resolves the observation schedule off this date.
    product.initial_date = datetime.combine(terms.inception, datetime.min.time())
    return product


# ---------------------------------------------------------------------------
# Fair coupon (Gate G4)
# ---------------------------------------------------------------------------


@dataclass
class CouponSolution:
    coupon: float
    pv: float
    iterations: int
    bracket_low: float
    bracket_high: float
    pv_tolerance: float
    solved: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inception_pricing_env(
    *,
    history: VolSurfaceHistory,
    inception: date,
    spot: float,
    rate: float,
    remaining_years: float,
) -> PricingEnvironment:
    """Flat-BSM env at inception: ATM IV at the full tenor + parity-implied q.

    Matches what the ``flat_bsm`` variant builds on its first replay day
    (``surface_vol_mode="flat_atm_remaining"``), so the coupon is solved
    under the same model that will price the trade on day one.
    """
    artifact = history.surface_for(inception)
    atm = artifact.term_structure_vol_surface()
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot), asset_name=UNDERLYING_NAME),
        vol_surface=FlatVolSurface(
            volatility=float(atm.get_vol(0.0, float(remaining_years), 0.0))
        ),
        rate_curve=FlatRateCurve(rate=float(rate)),
        div_yield=artifact.term_structure_dividend_yield(rate=float(rate)),
        valuation_date=datetime.combine(inception, datetime.min.time()),
    )


def solve_fair_coupon(
    *,
    terms,
    initial_spot: float,
    env: PricingEnvironment,
    engine_config: AutocallableEngineConfig,
    notional: float = NOTIONAL,
    lower: float = COUPON_LOWER,
    upper: float = COUPON_UPPER,
    max_iterations: int = COUPON_MAX_ITERATIONS,
) -> CouponSolution:
    """Solve the coupon that makes the snowball's PV zero at inception.

    PV is affine in the coupon: neither the KO trigger nor the KI trigger
    depends on it, so the coupon leg and the rebate leg both scale linearly
    (measured: PV(0)=-3.61M, PV(0.2)=+1.17M, PV(0.4)=+5.95M on the first
    inception - the two increments agree to the cent).  The solver therefore
    uses false position, which lands on the root in ONE step when the
    function really is affine, and degrades gracefully to an Illinois-damped
    regula falsi if it is not.  Each evaluation is a full ~14s PDE price, so
    this is 3 prices instead of the ~40 a bisection would need.

    The affine step is a hypothesis, never an assumption: the returned coupon
    is always priced and checked against the PV tolerance.  Fails closed
    (Gate G4) - a coupon that is not bracketed, or not converged inside the
    tolerance when the iteration budget runs out, raises rather than being
    returned as an unconverged or boundary value.
    """
    tolerance = float(COUPON_PV_TOL_FRACTION) * float(notional)

    def value(coupon: float) -> float:
        product = build_backtest_product(
            terms, initial_spot=initial_spot, coupon=coupon, notional=notional
        )
        engine = create_pricing_engine(product, engine_config)
        pv = float(engine.price(product, env))
        if not math.isfinite(pv):
            raise ValidationError(
                f"fair-coupon solve produced a non-finite PV at coupon={coupon:.8f} "
                f"(inception {terms.inception.isoformat()})"
            )
        return pv

    return solve_affine_root(
        value,
        lower=float(lower),
        upper=float(upper),
        tolerance=tolerance,
        max_iterations=int(max_iterations),
        where=f"inception {terms.inception.isoformat()}",
    )


def solve_affine_root(
    value,
    *,
    lower: float,
    upper: float,
    tolerance: float,
    max_iterations: int,
    where: str = "",
    expand_limit: float = 5.0,
) -> CouponSolution:
    """Illinois-damped regula falsi for a root that is expected to be affine.

    Exact in one step when ``value`` really is affine; still a bracketing
    method (so it cannot diverge) when it is not.  Every returned root has
    been evaluated and checked against ``tolerance`` - the affine structure
    is exploited, never trusted.  Raises rather than returning an unconverged
    or boundary result.
    """
    low, high = float(lower), float(upper)
    f_low, f_high = value(low), value(high)
    while f_low * f_high > 0.0 and high < expand_limit:
        high *= 1.5
        f_high = value(high)
    if f_low * f_high > 0.0:
        raise ValidationError(
            f"root is not bracketed ({where}): "
            f"f({low:.6f})={f_low:,.2f}, f({high:.6f})={f_high:,.2f}"
        )
    if abs(f_low) <= tolerance:
        return CouponSolution(low, f_low, 0, float(lower), high, tolerance, True)
    if abs(f_high) <= tolerance:
        return CouponSolution(high, f_high, 0, float(lower), high, tolerance, True)

    bracket_high = high
    iterations = 0
    side = 0  # Illinois damping: which endpoint has gone stale
    for _ in range(int(max_iterations)):
        iterations += 1
        denom = f_high - f_low
        if abs(denom) <= 1e-30:
            raise ValidationError(
                f"root solve is degenerate ({where}): f is flat over "
                f"[{low:.6f}, {high:.6f}]"
            )
        guess = low - f_low * (high - low) / denom
        if not (min(low, high) < guess < max(low, high)):
            guess = 0.5 * (low + high)  # keep the iterate inside the bracket
        f_guess = value(guess)
        if abs(f_guess) <= tolerance:
            return CouponSolution(
                coupon=float(guess),
                pv=float(f_guess),
                iterations=iterations,
                bracket_low=float(lower),
                bracket_high=float(bracket_high),
                pv_tolerance=float(tolerance),
                solved=True,
            )
        if f_low * f_guess < 0.0:
            high, f_high = guess, f_guess
            if side == -1:
                f_low *= 0.5  # Illinois: halve the stale endpoint
            side = -1
        else:
            low, f_low = guess, f_guess
            if side == 1:
                f_high *= 0.5
            side = 1

    raise ValidationError(
        f"root solver did not converge ({where}) after {iterations} "
        f"iterations: bracket [{low:.8f}, {high:.8f}], "
        f"tolerance={tolerance:,.2f} (Gate G4 requires convergence)"
    )


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


def make_mc_params(paths_per_batch: int, batches: int, seed: int) -> MCParams:
    """Fixed-cost fixed-seed RQMC, matching the stage 11 gate reference."""
    return MCParams(
        num_paths=int(paths_per_batch),
        seed=int(seed),
        use_antithetic=False,
        rqmc_min_batches=int(batches),
        rqmc_max_batches=int(batches),
        rqmc_target_std=1e-12,
    )


def make_engine_config(
    variant: str,
    *,
    routing: GateRouting,
    calibration_cache_dir: Optional[Path] = None,
    quick: bool = False,
) -> AutocallableEngineConfig:
    """Daily pricing engine configuration for one variant."""
    spec = VARIANT_SPECS[variant]
    # Route by the study VARIANT name, not spec.vol_model: the decision's
    # variants map is keyed by variant (flat_bsm/flat_bsm_quad/ts_bsm all
    # share vol_model="bsm" but can carry independent verdicts) -- see
    # GateRouting.solver_for.
    solver = routing.solver_for(variant)

    # --quick is the G3 smoke / CI knob: a short window on a deliberately
    # tiny MC budget.  It is NOT Gate G2's certified configuration and must
    # never be used for a study run, so it keeps its own module-constant
    # paths/batches/seed rather than reading the gate's, and the engine
    # falls back to its own scheme/substeps defaults.
    mc_paths = QUICK_MC_PATHS_PER_BATCH if quick else MC_PATHS_PER_BATCH
    mc_batches = QUICK_MC_BATCHES if quick else MC_BATCHES
    mc_seed = MC_SEED
    mc_engine_options: Dict[str, Any] = {}

    if solver == "mc" and not quick:
        # G2 routed this variant to MC because its production engine's price
        # (and delta) disagreed with the reference computed under THIS exact
        # config -- paths/batches/seed AND scheme/substeps_per_interval.
        # Running the fleet on any other config computes a delta the gate
        # never certified, which is the whole defect this fixes. An absent
        # entry must raise, never silently fall back to this module's
        # constants or the engine's own defaults -- same fail-closed
        # principle as GateRouting.solver_for's missing-route check.
        gated = routing.mc_params.get(variant)
        if gated is None:
            raise ValidationError(
                f"gate decision has no mc_params for MC-routed variant={variant!r}; "
                "rerun stage 11 (11_pde_convergence_gate.py) before the fleet"
            )
        mc_paths = int(gated["paths_per_batch"])
        mc_batches = int(gated["batches"])
        mc_seed = int(gated["seed"])
        # Heston-only QE knobs; the gate's localvol entry has neither
        # (LocalVolSnowballMCEngine accepts neither kwarg), so only forward
        # keys the decision actually set -- passing substeps_per_interval=None
        # would TypeError inside the engine rather than falling back to its
        # default.
        #
        mc_engine_options = {
            key: gated[key]
            for key in ("substeps_per_interval",)
            if gated.get(key) is not None
        }
        # The QE scheme knob is ASYMMETRIC across the two Heston engines the
        # replay factory builds, so it cannot be forwarded under one name:
        #
        #   heston     -> HestonSnowballMCEngine      takes scheme=
        #   heston_slv -> HestonSLVQESnowballMCEngine takes martingale_correction=
        #
        # (engine_factory.py:253 and :277).  Passing the wrong one is not a
        # silent no-op: HestonSnowballMCEngine TypeErrors on
        # martingale_correction, and QESnowballMCEngine raises ValidationError
        # if "scheme" reaches its kwargs.  Both spellings mean the same
        # engine -- QESnowballMCEngine maps martingale_correction onto
        # scheme=QUADEXP_M internally -- which is why stage 11's
        # _make_mc_engine can pass martingale_correction to both and still be
        # the configuration recorded here as scheme="QUADEXP_M".
        if spec.vol_model == "heston":
            if gated.get("scheme") is not None:
                mc_engine_options["scheme"] = gated["scheme"]
        elif spec.vol_model == "heston_slv":
            if gated.get("martingale_correction") is not None:
                mc_engine_options["martingale_correction"] = gated[
                    "martingale_correction"
                ]

    calibration = None
    if spec.uses_calibration():
        calibration = VolModelCalibrationConfig(
            cache_dir=str(calibration_cache_dir) if calibration_cache_dir else None,
            slv_n_steps=12 if quick else 40,
            slv_n_x=61 if quick else 161,
            slv_n_z=31 if quick else 81,
        )

    return AutocallableEngineConfig(
        # Every variant keeps the SAME deterministic 1D snowball PDE for the
        # surface and event-stats engines, so those outputs stay comparable
        # across variants; only the daily pricing engine differs.  Pinned
        # explicitly (not left to fall back from pricing_engine_type) because
        # flat_bsm_quad routes its DAILY pricing engine to QUADRATURE while
        # keeping the reporting engines on PDE like every other variant.
        pricing_engine_type=spec.pricing_engine_type,
        surface_engine_type=EngineType.PDE,
        event_stats_engine_type=EngineType.PDE,
        pde_params=PDEParams(),
        quad_params=QuadParams(),
        mc_params=make_mc_params(mc_paths, mc_batches, mc_seed),
        vol_model_mc_method=MonteCarloMethod.RANDOMIZED_QUASI,
        vol_source=spec.vol_source,
        surface_vol_mode=spec.surface_vol_mode,
        vol_model=spec.vol_model,
        vol_model_solver=solver,
        vol_model_calibration=calibration,
        # engine_options_for is non-empty only on the PDE route (n_x/n_v/n_t)
        # and mc_engine_options only on the gated MC route (scheme/substeps),
        # so exactly one side of this merge is ever non-empty.
        vol_model_engine_options={
            **routing.engine_options_for(variant),
            **mc_engine_options,
        },
    )


def resolved_mc_engine_options(
    routing: GateRouting, variants: Sequence[str], *, quick: bool
) -> Dict[str, Dict[str, Any]]:
    """Per-MC-routed-variant ``vol_model_engine_options`` that actually ran.

    The gate decision's own payload note says production "may reduce path
    counts under a documented SE budget, recorded in the run manifest" --
    this is that record for the scheme/substeps half of the config (paths/
    batches/seed are already recorded as mc_paths_per_batch/mc_batches/
    mc_seed).  Built by calling make_engine_config itself rather than
    re-deriving the logic, so the manifest can never drift from what the
    fleet actually configured.
    """
    return {
        variant: make_engine_config(
            variant, routing=routing, quick=quick
        ).vol_model_engine_options
        for variant in variants
        if routing.solver_for(variant) == "mc"
    }


def make_cost_model(enabled: bool):
    if not enabled:
        return ZeroCostModel()
    return CompleteCostModel(
        fixed_commission=0.0,
        proportional_rate=COST_PROPORTIONAL_RATE,
        slippage_coefficient=0.0,
        spread_bps=COST_SPREAD_BPS,
    )


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def configure_numeric_threads() -> None:
    """One BLAS thread per worker; the fleet parallelises over runs."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single inception x variant replay. Picklable worker entry point."""
    configure_numeric_threads()
    started = time.perf_counter()
    inception = date.fromisoformat(task["inception"])
    variant = str(task["variant"])
    out_dir = Path(task["out_dir"])
    run_dir = out_dir / "runs" / inception.isoformat() / variant

    history = surface_history(task["history_dir"])
    spot = load_spot_frame(Path(task["history_dir"]))
    futures = load_futures_frame(Path(task["history_dir"]))

    window_start = pd.Timestamp(inception)
    window_end = pd.Timestamp(date.fromisoformat(task["window_end"]))
    spot = spot[(spot["date"] >= window_start) & (spot["date"] <= window_end)]
    futures = futures[
        (futures["date"] >= window_start) & (futures["date"] <= window_end)
    ]
    if task.get("max_days"):
        keep = list(spot["date"])[: int(task["max_days"])]
        spot = spot[spot["date"].isin(keep)]
        futures = futures[futures["date"].isin(keep)]
    spot = spot.reset_index(drop=True)
    futures = futures.reset_index(drop=True)
    if spot.empty:
        raise ValidationError(
            f"no spot data in window for inception {inception.isoformat()}"
        )

    market_data = build_market_dataset(
        spot=spot, futures=futures, history=history, rate=float(task["rate"])
    )
    dates = market_data.dates
    if len(dates) == 0:
        raise ValidationError(
            f"market dataset is empty for inception {inception.isoformat()}"
        )

    s11 = stage11()
    calendar = s11.TradingCalendar.from_spot_csv(
        Path(task["history_dir"]) / "csi1000_spot.csv"
    )
    terms = s11.build_snowball_terms(inception, calendar)
    product = build_backtest_product(
        terms,
        initial_spot=float(task["initial_spot"]),
        coupon=float(task["coupon"]),
        notional=float(task["notional"]),
    )

    routing = GateRouting(
        decision_path=task["gate"]["decision_path"],
        evidence_sha256=task["gate"]["evidence_sha256"],
        routes=task["gate"]["routes"],
        pde_params=task["gate"]["pde_params"],
        mc_params=task["gate"]["mc_params"],
    )
    engine_config = make_engine_config(
        variant,
        routing=routing,
        calibration_cache_dir=out_dir / "calibration_cache",
        quick=bool(task.get("quick")),
    )

    config = AutocallableBacktestConfig(
        product=product,
        market_data=market_data,
        engine_config=engine_config,
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=0.0, hedge_ratio=1.0, round_contracts=True
        ),
        roll_policy=FuturesRollPolicy(roll_days_before_expiry=5),
        transaction_cost_model=make_cost_model(bool(task["costs_enabled"])),
        product_quantity=float(task["product_quantity"]),
        underlying=UNDERLYING_NAME,
        start_date=pd.Timestamp(dates[0]).to_pydatetime(),
        end_date=pd.Timestamp(dates[-1]).to_pydatetime(),
        initial_product_price=0.0,
        surface_config=SurfaceGridConfig(
            spot_nodes=5, spot_width=0.05, q_nodes=3, q_width=0.005
        ),
        calculate_surfaces=bool(task.get("calculate_surfaces", False)),
        calculate_event_probabilities=bool(
            task.get("calculate_event_probabilities", True)
        ),
        metadata={
            "study": "snowball_volmodel_backtest",
            "inception": inception.isoformat(),
            "variant": variant,
            "vol_model": VARIANT_SPECS[variant].vol_model,
            "vol_model_solver": engine_config.vol_model_solver,
            "coupon": float(task["coupon"]),
            "notional": float(task["notional"]),
        },
    )

    results = AutocallableBacktestEngine(config).run()

    _write_frame(results.states_df, run_dir / "states.csv")
    _write_frame(results.greeks_df, run_dir / "greeks.csv")
    _write_frame(results.trades_df, run_dir / "trades.csv")
    _write_frame(results.rebalance_df, run_dir / "rebalances.csv")
    _write_frame(results.actions_df, run_dir / "actions.csv")
    _write_frame(results.daily_event_summary_df, run_dir / "daily_events.csv")
    _write_frame(results.event_probability_df, run_dir / "event_probabilities.csv")
    if bool(task.get("calculate_surfaces", False)):
        _write_frame(results.surfaces_df, run_dir / "surfaces.csv")

    calibration_records = list(getattr(results, "calibration_records", []) or [])
    _atomic_write_json(run_dir / "calibration_records.json", calibration_records)

    summary = summarize_run(
        results=results,
        terms=terms,
        inception=inception,
        variant=variant,
        engine_config=engine_config,
        dates=dates,
        coupon=float(task["coupon"]),
        notional=float(task["notional"]),
        calibration_records=calibration_records,
        elapsed=time.perf_counter() - started,
    )
    _atomic_write_json(run_dir / "run_summary.json", summary)
    return summary


def summarize_run(
    *,
    results,
    terms,
    inception: date,
    variant: str,
    engine_config: AutocallableEngineConfig,
    dates,
    coupon: float,
    notional: float,
    calibration_records: List[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    """Per-run outcome: lifecycle end state, PnL decomposition, provenance.

    Lifecycle state comes from the engine's own per-day flags (the
    ``knocked_out`` / ``knocked_in`` / ``matured`` columns it writes into
    every state row), not from parsing action labels: the flags ARE the
    lifecycle tracker's state, while the action log is a reporting view whose
    ``action_type`` vocabulary ("KO"/"KI"/"COUPON"/"MATURITY") is free to
    change.  The action log is used only to date the knock-out.
    """
    states = results.states_df
    actions = results.actions_df
    last_date = pd.Timestamp(dates[-1]).date()

    def final_flag(name: str) -> bool:
        if states is None or states.empty or name not in states.columns:
            return False
        return bool(states[name].iloc[-1])

    knocked_out = final_flag("knocked_out")
    knocked_in = final_flag("knocked_in")

    ko_date = None
    if (
        knocked_out
        and actions is not None
        and not actions.empty
        and "action_type" in actions.columns
    ):
        ko_rows = actions[actions["action_type"].astype(str).str.upper() == "KO"]
        if not ko_rows.empty:
            ko_date = str(ko_rows.index[0])

    # "Matured" means the replay actually reached the contractual maturity,
    # not merely that the flag was set on a knocked-out trade.
    matured = (not knocked_out) and (
        final_flag("matured") or last_date >= terms.maturity_date
    )
    censored = not knocked_out and not matured

    summary_metrics: Dict[str, Any] = {}
    try:
        summary_metrics = dict(results.get_summary())
    except Exception:  # noqa: BLE001 - summary is reporting, never fatal
        summary_metrics = {}

    return {
        "schema_version": SCHEMA_VERSION,
        "inception": inception.isoformat(),
        "variant": variant,
        "vol_model": VARIANT_SPECS[variant].vol_model,
        "vol_model_solver": engine_config.vol_model_solver,
        "surface_vol_mode": engine_config.surface_vol_mode,
        "coupon": float(coupon),
        "notional": float(notional),
        "maturity_date": terms.maturity_date.isoformat(),
        "maturity_years": float(terms.maturity_years),
        "first_date": pd.Timestamp(dates[0]).date().isoformat(),
        "last_date": last_date.isoformat(),
        "n_days": int(len(dates)),
        "lifecycle": {
            "knocked_out": bool(knocked_out),
            "knocked_in": bool(knocked_in),
            "ko_date": ko_date,
            "matured": bool(matured),
            "censored_at_data_end": bool(censored),
        },
        "n_calibration_records": len(calibration_records),
        "metrics": summary_metrics,
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# Fleet driver
# ---------------------------------------------------------------------------


def prepare_inceptions(
    *,
    history: VolSurfaceHistory,
    spot: pd.DataFrame,
    futures: pd.DataFrame,
    calendar,
    rate: float,
    notional: float,
    min_observable_months: int,
    data_end: Optional[date] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Schedule inceptions and solve each one's fair coupon (Gate G4).

    ``data_end`` defaults to the spot cache's last row, matching the
    pre-pin behaviour.  Callers that pin the replay window (``run_fleet``'s
    ``--data-end``) must pass the pinned value here too, or the fleet size
    keeps floating with the spot cache even though task windows are pinned.
    """
    s11 = stage11()
    data_start = pd.Timestamp(spot["date"].iloc[0]).date()
    if data_end is None:
        data_end = pd.Timestamp(spot["date"].iloc[-1]).date()
    admitted = history.admitted_dates

    inceptions = schedule_inceptions(
        calendar=calendar,
        data_start=data_start,
        data_end=data_end,
        first_admitted_surface=admitted[0],
        min_observable_months=min_observable_months,
    )
    if limit is not None:
        inceptions = inceptions[: int(limit)]

    spot_by_date = {
        pd.Timestamp(row.date).date(): float(row.spot)
        for row in spot.itertuples(index=False)
    }

    prepared: List[Dict[str, Any]] = []
    coupon_engine_config = make_engine_config(
        "flat_bsm",
        # Gate G4 solves the coupon under flat BSM on the 1D PDE regardless
        # of what Gate G2 decides for the fleet's other five variants, so
        # this pins "flat_bsm" -> "pde" explicitly rather than depending on
        # the (now removed) GateRouting short-circuit.
        routing=GateRouting("", None, {"flat_bsm": "pde"}, {}),
        calibration_cache_dir=None,
    )
    print(
        f"[coupons] solving fair coupons for {len(inceptions)} inceptions " "(Gate G4)",
        flush=True,
    )
    for index, inception in enumerate(inceptions, start=1):
        if inception not in spot_by_date:
            raise ValidationError(
                f"inception {inception.isoformat()} has no spot observation"
            )
        initial_spot = spot_by_date[inception]
        solve_started = time.perf_counter()
        terms = s11.build_snowball_terms(inception, calendar)
        env = inception_pricing_env(
            history=history,
            inception=inception,
            spot=initial_spot,
            rate=rate,
            remaining_years=float(terms.maturity_years),
        )
        solution = solve_fair_coupon(
            terms=terms,
            initial_spot=initial_spot,
            env=env,
            engine_config=coupon_engine_config,
            notional=notional,
        )
        implied_q = inception_implied_q(
            futures=futures, inception=inception, spot=initial_spot, rate=rate
        )
        print(
            f"  [{index}/{len(inceptions)}] {inception.isoformat()} "
            f"s0={initial_spot:,.2f} coupon={solution.coupon:.4%} "
            f"|PV|={abs(solution.pv):,.2f} "
            f"({solution.iterations} iters, {time.perf_counter() - solve_started:.1f}s)",
            flush=True,
        )
        prepared.append(
            {
                "inception": inception.isoformat(),
                "initial_spot": initial_spot,
                "coupon": solution.coupon,
                "coupon_solution": solution.to_dict(),
                "terms": terms.summary(),
                "maturity_date": terms.maturity_date.isoformat(),
                "atm_vol_at_inception": float(
                    env.vol_surface.get_vol(0.0, float(terms.maturity_years), 0.0)
                ),
                "futures_implied_q": implied_q,
                "surface_sha": history.sha_for(inception),
            }
        )
    return prepared


def inception_implied_q(
    *, futures: pd.DataFrame, inception: date, spot: float, rate: float
) -> Optional[float]:
    """Futures-basis implied dividend yield on the inception date (diagnostic)."""
    stamp = pd.Timestamp(inception)
    day = futures[futures["date"] == stamp]
    if day.empty:
        return None
    row = day.sort_values("expiry_date").iloc[0]
    ttm = (pd.Timestamp(row["expiry_date"]) - stamp).days / 365.0
    if ttm <= 0:
        return None
    _, implied_q = derive_implied_dividend_yield(
        rate=float(rate),
        spot=float(spot),
        futures_price=float(row["futures_price"]),
        time_to_maturity=float(ttm),
    )
    return float(implied_q)


def build_tasks(
    *,
    prepared: Sequence[Dict[str, Any]],
    variants: Sequence[str],
    routing: GateRouting,
    history_dir: Path,
    out_dir: Path,
    data_end: date,
    rate: float,
    notional: float,
    costs_enabled: bool,
    quick: bool,
    calculate_surfaces: bool,
    calculate_event_probabilities: bool,
    max_days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    gate_payload = {
        "decision_path": routing.decision_path,
        "evidence_sha256": routing.evidence_sha256,
        "routes": dict(routing.routes),
        "pde_params": dict(routing.pde_params),
        "mc_params": dict(routing.mc_params),
    }
    tasks: List[Dict[str, Any]] = []
    for entry in prepared:
        maturity = date.fromisoformat(entry["maturity_date"])
        window_end = min(maturity, data_end)
        for variant in variants:
            tasks.append(
                {
                    "inception": entry["inception"],
                    "variant": variant,
                    "initial_spot": entry["initial_spot"],
                    "coupon": entry["coupon"],
                    "window_end": window_end.isoformat(),
                    "history_dir": str(history_dir),
                    "out_dir": str(out_dir),
                    "gate": gate_payload,
                    "rate": float(rate),
                    "notional": float(notional),
                    "product_quantity": PRODUCT_QUANTITY,
                    "costs_enabled": bool(costs_enabled),
                    "quick": bool(quick),
                    # An explicit cap wins over --quick's default, so a timing
                    # smoke can shorten the window while keeping the PRODUCTION
                    # MC config -- --quick also shrinks paths/batches, which
                    # would corrupt the per-day cost it is meant to measure.
                    "max_days": (
                        int(max_days)
                        if max_days is not None
                        else (QUICK_MAX_DAYS if quick else None)
                    ),
                    "calculate_surfaces": bool(calculate_surfaces),
                    "calculate_event_probabilities": bool(
                        calculate_event_probabilities
                    ),
                }
            )
    return tasks


def execute_tasks(
    tasks: Sequence[Dict[str, Any]], *, workers: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run tasks (serially or over a process pool). Returns (summaries, failures)."""
    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if not tasks:
        return summaries, failures

    workers = max(1, min(int(workers), len(tasks)))
    if workers == 1:
        for task in tasks:
            label = f"{task['inception']} / {task['variant']}"
            print(f"[run] {label}", flush=True)
            try:
                summaries.append(run_one(task))
            except Exception as exc:  # noqa: BLE001 - one run must not kill the fleet
                failures.append(_failure_record(task, exc))
                print(f"[FAIL] {label}: {exc}", flush=True)
        return summaries, failures

    configure_numeric_threads()
    print(f"[parallel] {len(tasks)} runs over {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            label = f"{task['inception']} / {task['variant']}"
            try:
                summaries.append(future.result())
                print(f"[done] {label}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(_failure_record(task, exc))
                print(f"[FAIL] {label}: {exc}", flush=True)
    summaries.sort(key=lambda s: (s["inception"], s["variant"]))
    failures.sort(key=lambda f: (f["inception"], f["variant"]))
    return summaries, failures


def _failure_record(task: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    return {
        "inception": task["inception"],
        "variant": task["variant"],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-4000:],
    }


def build_run_manifest(
    *,
    cfg: Dict[str, Any],
    routing: GateRouting,
    prepared: Sequence[Dict[str, Any]],
    summaries: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
    elapsed: float,
) -> Dict[str, Any]:
    censored = sum(1 for s in summaries if s["lifecycle"]["censored_at_data_end"])
    knocked_out = sum(1 for s in summaries if s["lifecycle"]["knocked_out"])
    matured = sum(1 for s in summaries if s["lifecycle"]["matured"])
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "snowball_volmodel_backtest",
        "config": cfg,
        "term_sheet": {
            "underlying": "000852.SH",
            "notional": cfg["notional"],
            "product_quantity": PRODUCT_QUANTITY,
            "tenor_months": stage11().MATURITY_MONTHS,
            "ko_pct": stage11().KO_PCT,
            "ki_pct": stage11().KI_PCT,
            "lockout_months": stage11().LOCKOUT_MONTHS,
            "coupon": "solved per inception under flat BSM (Gate G4)",
        },
        "hedge_costs": {
            "enabled": cfg["costs_enabled"],
            "model": "CompleteCostModel" if cfg["costs_enabled"] else "ZeroCostModel",
            "proportional_rate": COST_PROPORTIONAL_RATE,
            "spread_bps": COST_SPREAD_BPS,
        },
        "gate_g2": {
            "decision_path": routing.decision_path,
            "evidence_sha256": routing.evidence_sha256,
            "routes": dict(routing.routes),
        },
        "adi_greek_certification": {
            "decision_path": cfg.get("adi_greek_decision"),
            "evidence_sha256": cfg.get("adi_greek_evidence_sha256"),
            "implementation_sha256": cfg.get("adi_greek_implementation_sha256"),
            "run_configuration_sha256": cfg.get("adi_greek_run_configuration_sha256"),
            "decision_sha256": cfg.get("adi_greek_decision_sha256"),
            "runtime_environment": cfg.get("adi_greek_runtime_environment"),
            "production_engine_controls": cfg.get(
                "adi_greek_production_engine_controls"
            ),
            "run_configuration": cfg.get("adi_greek_run_configuration"),
            "requested_variants": list(cfg.get("requested_variants", cfg["variants"])),
            "excluded_variants": dict(cfg.get("adi_greek_exclusions", {})),
        },
        "inceptions": list(prepared),
        "runs": list(summaries),
        "failures": list(failures),
        "counts": {
            "inceptions": len(prepared),
            "variants": len(cfg["variants"]),
            "runs_expected": len(prepared) * len(cfg["variants"]),
            "runs_completed": len(summaries),
            "runs_failed": len(failures),
            "knocked_out": knocked_out,
            "matured": matured,
            "censored_at_data_end": censored,
        },
        "elapsed_seconds": float(elapsed),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--gate-decision", default=str(DEFAULT_GATE_DECISION))
    parser.add_argument(
        "--adi-greek-decision",
        default=str(DEFAULT_ADI_GREEK_DECISION),
        help="ADI Greek admission artifact; required when heston or heston_slv "
        "is requested. Defaults to the COMMITTED modelvalidation certificate, "
        "which recomputes its own digest and is re-verified against both ADI "
        "solvers on every commit. A raw stage-17 decision is still accepted.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated subset of: " + ",".join(VARIANTS),
    )
    parser.add_argument(
        "--max-inceptions",
        type=int,
        default=None,
        help="cap the number of inceptions (timing runs)",
    )
    parser.add_argument(
        "--min-observable-months", type=int, default=MIN_OBSERVABLE_MONTHS
    )
    parser.add_argument(
        "--data-end",
        default=COHORT_ASOF.isoformat(),
        help=(
            "ISO date pinning the replay window end (default: the FROZEN "
            "COHORT_ASOF, never the last spot row). The daily calibration "
            "pipeline extends the spot cache every weekday, and data_end "
            "crossing 2026-08-01 admits a 28th inception -- measured "
            "2026-08-25, an unpinned window gave 28 against the 27 of record, "
            "which would void the banked coupons. Pinning is the default so "
            "forgetting the flag cannot silently resize the fleet."
        ),
    )
    parser.add_argument("--notional", type=float, default=NOTIONAL)
    parser.add_argument(
        "--no-costs", action="store_true", help="run with ZeroCostModel"
    )
    parser.add_argument("--surfaces", action="store_true")
    parser.add_argument("--no-event-probabilities", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small MC + short window (Gate G3 smoke / CI)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="cap replay days per run, WITHOUT shrinking the MC config the way "
        "--quick does. This is the timing-smoke knob: it measures the "
        "production per-replay-day cost on a short window. Overrides "
        f"--quick's default of {QUICK_MAX_DAYS} days when both are given.",
    )
    parser.add_argument(
        "--gate-g3",
        action="store_true",
        help="Gate G3: one inception x flat_bsm end-to-end sanity check",
    )
    return parser.parse_args(argv)


def run_fleet(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    history_dir = Path(args.history_dir)
    out_dir = Path(args.out_dir)
    rate = stage11().FLAT_RATE

    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_SPECS]
    if unknown:
        raise ValidationError(f"unknown variants: {unknown}; known: {list(VARIANTS)}")
    routing = load_gate_routing(Path(args.gate_decision))
    requested_variants = list(variants)
    adi_greek_routing = None
    adi_greek_exclusions: Dict[str, str] = {}
    if args.gate_g3:
        variants = ["flat_bsm"]
        requested_variants = list(variants)
    elif any(variant in {"heston", "heston_slv"} for variant in variants):
        adi_greek_routing = load_adi_greek_admission(Path(args.adi_greek_decision))
        variants, adi_greek_exclusions = apply_adi_greek_admission(
            variants,
            routing,
            adi_greek_routing,
        )
        if not variants:
            detail = "; ".join(
                f"{variant}: {reason}"
                for variant, reason in adi_greek_exclusions.items()
            )
            raise ValidationError(
                "all requested variants were excluded by fail-closed ADI Greek "
                f"admission: {detail}"
            )
    history = surface_history(history_dir)
    spot = load_spot_frame(history_dir)
    futures = load_futures_frame(history_dir)
    calendar = stage11().TradingCalendar.from_spot_csv(history_dir / "csi1000_spot.csv")
    data_end = pd.Timestamp(spot["date"].iloc[-1]).date()
    if args.data_end:
        pinned = date.fromisoformat(str(args.data_end))
        if pinned > data_end:
            raise ValidationError(
                f"--data-end {pinned} is beyond the spot cache ({data_end})"
            )
        data_end = pinned

    limit = 1 if args.gate_g3 else args.max_inceptions
    prepared = prepare_inceptions(
        history=history,
        spot=spot,
        futures=futures,
        calendar=calendar,
        rate=rate,
        notional=float(args.notional),
        min_observable_months=int(args.min_observable_months),
        data_end=data_end,
        limit=limit,
    )
    print(
        f"[schedule] {len(prepared)} inceptions x {len(variants)} variants "
        f"= {len(prepared) * len(variants)} runs",
        flush=True,
    )

    tasks = build_tasks(
        prepared=prepared,
        variants=variants,
        routing=routing,
        history_dir=history_dir,
        out_dir=out_dir,
        data_end=data_end,
        rate=rate,
        notional=float(args.notional),
        costs_enabled=not args.no_costs,
        quick=bool(args.quick),
        max_days=args.max_days,
        calculate_surfaces=bool(args.surfaces),
        calculate_event_probabilities=not args.no_event_probabilities,
    )
    summaries, failures = execute_tasks(tasks, workers=int(args.workers))

    cfg = {
        "history_dir": str(history_dir),
        "gate_decision": str(args.gate_decision),
        "adi_greek_decision": (
            None if adi_greek_routing is None else adi_greek_routing.decision_path
        ),
        "adi_greek_evidence_sha256": (
            None if adi_greek_routing is None else adi_greek_routing.evidence_sha256
        ),
        "adi_greek_implementation_sha256": (
            None
            if adi_greek_routing is None
            else adi_greek_routing.implementation_sha256
        ),
        "adi_greek_run_configuration_sha256": (
            None
            if adi_greek_routing is None
            else adi_greek_routing.run_configuration_sha256
        ),
        "adi_greek_decision_sha256": (
            None if adi_greek_routing is None else adi_greek_routing.decision_sha256
        ),
        "adi_greek_runtime_environment": (
            None
            if adi_greek_routing is None
            else dict(adi_greek_routing.runtime_environment)
        ),
        "adi_greek_production_engine_controls": (
            None
            if adi_greek_routing is None
            else dict(adi_greek_routing.production_engine_controls)
        ),
        "adi_greek_run_configuration": (
            None
            if adi_greek_routing is None
            else dict(adi_greek_routing.run_configuration)
        ),
        "requested_variants": requested_variants,
        "adi_greek_exclusions": adi_greek_exclusions,
        "out_dir": str(out_dir),
        "variants": variants,
        "min_observable_months": int(args.min_observable_months),
        "notional": float(args.notional),
        "costs_enabled": not args.no_costs,
        "quick": bool(args.quick),
        "gate_g3": bool(args.gate_g3),
        "rate": float(rate),
        "workers": int(args.workers),
        "mc_paths_per_batch": QUICK_MC_PATHS_PER_BATCH
        if args.quick
        else MC_PATHS_PER_BATCH,
        "mc_batches": QUICK_MC_BATCHES if args.quick else MC_BATCHES,
        "mc_seed": MC_SEED,
        # Per-variant scheme/substeps_per_interval actually resolved for the
        # MC-routed variants (Task 10, spec §5.5); {} for a run with no MC
        # route.  mc_paths_per_batch/mc_batches/mc_seed above stay as the
        # module-wide fallback -- see make_engine_config for how an
        # MC-routed variant's own gate values override them.
        "mc_engine_options": resolved_mc_engine_options(
            routing, variants, quick=bool(args.quick)
        ),
    }
    manifest = build_run_manifest(
        cfg=cfg,
        routing=routing,
        prepared=prepared,
        summaries=summaries,
        failures=failures,
        elapsed=time.perf_counter() - started,
    )
    _atomic_write_json(out_dir / "inceptions.json", list(prepared))
    _atomic_write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_fleet(args)
    counts = manifest["counts"]
    print(
        "\n[summary] {completed}/{expected} runs completed, {failed} failed "
        "({ko} KO, {mat} matured, {cens} censored)".format(
            completed=counts["runs_completed"],
            expected=counts["runs_expected"],
            failed=counts["runs_failed"],
            ko=counts["knocked_out"],
            mat=counts["matured"],
            cens=counts["censored_at_data_end"],
        ),
        flush=True,
    )
    print(f"[out] {Path(manifest['config']['out_dir']) / 'run_manifest.json'}")
    return 1 if counts["runs_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
