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

Production run::

    .venv/bin/python example/mo_volmodels/16_adi_greek_certification.py

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
    certify_equivalence,
    certify_signed_bias_from_batches,
)
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface


SCHEMA_VERSION = 8
# The 20260802-20260804 scrambles selected the estimator design, and 20260805
# exposed the need for the near-KI batch extension. Production evidence uses a
# fresh held-out scramble family so neither choice can bias its intervals.
SEED = 20260806
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
PRODUCTION_HESTON_BATCHES = 256
PRODUCTION_HESTON_BATCHES_BY_CASE = {
    "ordinary_full": PRODUCTION_HESTON_BATCHES,
    "ordinary_decayed": PRODUCTION_HESTON_BATCHES,
    "near_ko": PRODUCTION_HESTON_BATCHES,
    # Near KI is a discontinuity-dominated gamma regime.  The 256-scramble
    # held-out pilot passed every deterministic refinement gate but left the
    # paired target-to-fine QE-M bias interval inconclusive.  Keep the first
    # 256 scramble ids common with every other regime for the signed-bias gate,
    # then extend this cell alone until its reference interval is decisive.
    "near_ki": 2048,
    "low_feller": PRODUCTION_HESTON_BATCHES,
    "sigma_collapse": PRODUCTION_HESTON_BATCHES,
    "near_expiry": PRODUCTION_HESTON_BATCHES,
}
PRODUCTION_SLV_PATHS_PER_BATCH = 1024
PRODUCTION_SLV_BATCHES = 128
PRODUCTION_RQMC_BATCH_WORKERS = 4
PDE_REFINEMENT_RATIO_LIMIT = 1.25
PDE_REFINEMENT_NEGLIGIBLE_BOUND_FRACTION = 0.10
MIN_PRODUCTION_RQMC_BATCHES = 16
PRODUCTION_ENGINE_CONTROLS = {
    "grid_style": "concentrated",
    "v0_boundary": "degenerate_pde",
    "variance_grid_mode": "auto",
    "v_drift_scheme": "adaptive_upwind",
    "barrier_greek_steps_per_tick": 16,
    "greek_min_n_x": 300,
    "greek_min_n_v": 90,
    "greek_min_steps_per_year": 800,
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
    "quantark/validation/greek_certification.py",
    "quantark/volmodels/adi_core.py",
    "quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py",
    "quantark/asset/equity/engine/pde/pde_execution_adapters.py",
    "quantark/backtest/replay/engine_factory.py",
    "quantark/asset/equity/engine/mc/snowball_mc_engine.py",
    "quantark/asset/equity/engine/mc/snowball_vol_mc_engines.py",
    "quantark/montecarlo/qmc_rqmc_driver.py",
    "quantark/montecarlo/conditional_snowball.py",
    "quantark/montecarlo/qmc_qe_coupling.py",
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
        coarse = GridPoint(200, 60, max(80, int(math.ceil(400 * maturity))))
        target = GridPoint(300, 90, max(120, int(math.ceil(800 * maturity))))
        fine = GridPoint(450, 135, max(180, int(math.ceil(1600 * maturity))))
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
    rqmc_batch_workers: int = 1,
) -> PairedRQMCGreeksResult:
    spot = float(env.spot)
    specs = []
    for shifted_spot in (spot * (1.0 - bump), spot, spot * (1.0 + bump)):
        shifted_env = bumped_environment(env, shifted_spot)
        engine = make_mc_engine(
            variant,
            case,
            leverage,
            paths_per_batch=paths_per_batch,
            batches=batches,
            seed=seed,
            substeps=substeps,
            qe_draw_provider=qe_draw_provider,
            heston_spot_bridge_strata=heston_spot_bridge_strata,
            heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
            slv_spot_strata=slv_spot_strata,
            slv_spot_antithetic=slv_spot_antithetic,
            slv_spot_bridge_strata=slv_spot_bridge_strata,
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
    )


def combine_two_level_control(
    controlled: PairedRQMCGreeksResult,
    low_control: PairedRQMCGreeksResult,
    independent_control: PairedRQMCGreeksResult,
) -> PairedRQMCGreeksResult:
    """Return ``controlled-low control+independent control`` by scramble.

    The low component must be the exact conditional-control term already
    present in ``controlled``. Replacing it with an independent estimate of
    the same control expectation preserves unbiasedness. Combining inside each
    scramble keeps one auditable Student-t sample.
    """
    results = (controlled, low_control, independent_control)
    shapes = {result.batch_estimates.shape for result in results}
    if len(shapes) != 1 or next(iter(shapes))[1] != 5:
        raise ValueError("two-level Heston control batch shapes do not match")
    if len({result.batches_used for result in results}) != 1:
        raise ValueError("two-level Heston control batch counts do not match")
    if len({result.spot for result in results}) != 1 or len(
        {result.relative_bump for result in results}
    ) != 1:
        raise ValueError("two-level Heston control bump semantics do not match")

    estimates = (
        np.asarray(controlled.batch_estimates, dtype=float)
        - np.asarray(low_control.batch_estimates, dtype=float)
        + np.asarray(independent_control.batch_estimates, dtype=float)
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
        total_path_valuations=sum(
            result.total_path_valuations for result in results
        ),
        randomization_key=(
            "two_level_control("
            f"{controlled.randomization_key};"
            f"{low_control.randomization_key};"
            f"{independent_control.randomization_key})"
        ),
        batch_estimates=estimates,
        covariance=covariance,
    )


def coupled_qe_providers(
    *,
    seed: int,
    paths_per_batch: int,
    target_dt: np.ndarray,
    fine_dt: np.ndarray,
    reuse_count: int = 3,
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
        finer_bumps = [
            row
            for row in ladders["bump"]
            if float(row["bump"]) < SPOT_BUMP
        ]
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


def _pde_signed_refinement_components(
    ladders: dict, target: dict
) -> dict[str, dict]:
    components = {"delta": {}, "gamma": {}}
    for greek in components:
        for axis in ("n_x", "n_v", "n_t"):
            components[greek][axis] = (
                float(ladders["axes"][axis][-1][greek])
                - float(target[greek])
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
            values = [
                _economic_value(scale, greek, float(row[greek])) for row in rows
            ]
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


def certify_case(
    variant: str,
    case: CaseSpec,
    *,
    quick: bool,
    paths_per_batch: int,
    batches: int,
    seed: int,
    hedge_inception_spot: float,
    heston_spot_bridge_strata: int = HESTON_SPOT_BRIDGE_STRATA,
    heston_spot_bridge_dimensions: int = HESTON_SPOT_BRIDGE_DIMENSIONS,
    slv_spot_strata: int = SLV_SPOT_STRATA,
    slv_spot_antithetic: bool = SLV_SPOT_ANTITHETIC,
    slv_spot_bridge_strata: int = SLV_SPOT_BRIDGE_STRATA,
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
    raw_pde_signed = _pde_signed_refinement_components(
        ladder_evidence, pde_target
    )

    target_substeps, fine_substeps = ((1, 2) if quick else (4, 8))
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
        reuse_count=(1 if variant == "heston" else 3),
    )
    reference = paired_mc_reference(
        variant,
        case,
        product,
        env,
        leverage,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=target_substeps,
        bump=SPOT_BUMP,
        qe_draw_provider=target_provider,
        heston_spot_bridge_strata=heston_spot_bridge_strata,
        heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
        slv_spot_strata=slv_spot_strata,
        slv_spot_antithetic=slv_spot_antithetic,
        slv_spot_bridge_strata=slv_spot_bridge_strata,
        rqmc_batch_workers=rqmc_batch_workers,
    )
    reference_fine = paired_mc_reference(
        variant,
        case,
        product,
        env,
        leverage,
        paths_per_batch=paths_per_batch,
        batches=batches,
        seed=seed,
        substeps=fine_substeps,
        bump=SPOT_BUMP,
        qe_draw_provider=fine_provider,
        heston_spot_bridge_strata=heston_spot_bridge_strata,
        heston_spot_bridge_dimensions=heston_spot_bridge_dimensions,
        slv_spot_strata=slv_spot_strata,
        slv_spot_antithetic=slv_spot_antithetic,
        slv_spot_bridge_strata=slv_spot_bridge_strata,
        rqmc_batch_workers=rqmc_batch_workers,
    )
    scale = EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=hedge_inception_spot,
        study_notional=STUDY_NOTIONAL,
        hedge_multiplier=HEDGE_MULTIPLIER,
    )
    pde_refinement = _pde_refinement_diagnostics(ladder_evidence, scale)
    certifications = {}
    batch_differences = {}
    for greek, cell_bound in (
        ("delta", DELTA_CELL_BOUND_CONTRACTS),
        ("gamma", GAMMA_CELL_BOUND_CONTRACTS),
    ):
        pde_value = float(pde_target[greek])
        # Fine (8 substeps) is the oracle estimate. The coarser 4-substep
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

    diagnostic_engine = make_pde_engine(
        variant, case, ladders["target"], leverage
    )
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
        production_grid_policy = production_engine.greek_time_grid_policy(
            product, env
        )
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
                "conditioning" if variant == "heston" else
                "QE-M paired randomized Sobol with antithetic terminal and "
                "midpoint Brownian-bridge stratification plus path-frozen "
                "leverage conditional control"
            ),
            "primary": "fine",
            "target_substeps_per_interval": target_substeps,
            "fine_substeps_per_interval": fine_substeps,
            "heston_spot_bridge_strata": (
                int(heston_spot_bridge_strata)
                if variant == "heston"
                else None
            ),
            "heston_spot_bridge_dimensions": (
                int(heston_spot_bridge_dimensions)
                if variant == "heston"
                else None
            ),
            "slv_spot_strata": (
                int(slv_spot_strata) if variant == "heston_slv" else None
            ),
            "slv_spot_antithetic": (
                bool(slv_spot_antithetic) if variant == "heston_slv" else None
            ),
            "slv_spot_bridge_strata": (
                int(slv_spot_bridge_strata)
                if variant == "heston_slv"
                else None
            ),
            "slv_effective_spot_samples": (
                int(slv_spot_strata) * (2 if slv_spot_antithetic else 1)
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
        greek: float(old_certifications[greek]["pde"])
        for greek in ("delta", "gamma")
    }
    raw_pde_envelopes = _pde_envelope_components(
        cell["pde_ladders"], pde_target
    )
    raw_pde_signed = _pde_signed_refinement_components(
        cell["pde_ladders"], pde_target
    )
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
    """Recompute economics from complete schema-7 numerical evidence."""
    if payload.get("study") != "adi_2d_snowball_greek_certification":
        raise ValueError("certification study tag mismatch")
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            "schema-7 evidence with saved target and fine batches is required "
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
        slv_spot_strata=int(
            rescored.get("policy", {}).get("slv_spot_strata", 0)
        ),
        slv_spot_antithetic=bool(
            rescored.get("policy", {}).get("slv_spot_antithetic", False)
        ),
        slv_spot_bridge_strata=int(
            rescored.get("policy", {}).get("slv_spot_bridge_strata", 0)
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
    run_configuration["rescore_source_run_configuration_sha256"] = (
        source_run_configuration_hash
    )
    rescored["run_configuration"] = run_configuration
    rescored["run_configuration_sha256"] = _canonical_sha256(run_configuration)
    rescored["rescore_provenance"] = {
        "source_evidence_sha256": source_hash,
        "numerical_evidence_reused": True,
        "reason": (
            "separate normalized model spot from actual hedge inception spot"
        ),
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
            "status": _combined_status(check["status"] for check in vanilla_checks.values()),
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
                check["status"]
                for group in checks.values()
                for check in group.values()
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
            "status": _combined_status(check["status"] for check in unit_checks.values()),
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
            "status": _combined_status(check["status"] for check in det_checks.values()),
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


def _sampling_batches_for_case(
    sampling: dict, variant: str, case_name: str
) -> int:
    """Resolve the declared scramble count for one regime cell."""
    if variant == "heston":
        batches_by_case = sampling.get("batches_by_case")
        if isinstance(batches_by_case, dict) and case_name in batches_by_case:
            return int(batches_by_case[case_name])
    return int(sampling.get("batches", 0))


def _normalized_batches_by_case(sampling: dict) -> dict[str, int]:
    batches_by_case = sampling.get("batches_by_case")
    if not isinstance(batches_by_case, dict):
        return {}
    try:
        return {str(name): int(value) for name, value in batches_by_case.items()}
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
            sampling_complete = sampling_complete and int(
                sampling.get("paths_per_batch", 0)
            ) >= minimum_paths
            if variant == "heston":
                declared_profile = heston_spot_bridge_profile_by_case or {}
                sampling_complete = sampling_complete and (
                    int(sampling.get("batches", 0))
                    == PRODUCTION_HESTON_BATCHES
                    and _normalized_batches_by_case(sampling)
                    == PRODUCTION_HESTON_BATCHES_BY_CASE
                    and all(
                        actual_batches
                        == _sampling_batches_for_case(
                            sampling, variant, case_name
                        )
                        for case_name, actual_batches in (
                            batch_counts_by_case.items()
                        )
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
                sampling_complete = sampling_complete and (
                    int(sampling.get("batches", 0))
                    >= PRODUCTION_SLV_BATCHES
                    and all(
                        actual_batches == int(sampling.get("batches", 0))
                        for actual_batches in batch_counts_by_case.values()
                    )
                    and int(slv_spot_strata) == SLV_SPOT_STRATA
                    and bool(slv_spot_antithetic) == SLV_SPOT_ANTITHETIC
                    and int(slv_spot_bridge_strata)
                    == SLV_SPOT_BRIDGE_STRATA
                )
        evidence_complete = (
            not missing_anchors and not missing_cases and sampling_complete
        )
        cell_status = _combined_status(row["status"] for row in variant_rows)
        delta_batches = np.asarray(
            [
                row["batch_difference_contracts"]["delta"][:common_batches]
                for row in variant_rows
            ],
            dtype=float,
        )
        if delta_batches.ndim == 2 and delta_batches.shape[0] > 0:
            # Mean across cases *within* each scramble preserves cross-case
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
                    row["certifications"]["delta"][
                        "reference_substep_batch_contracts"
                    ][:common_batches]
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
                abs(aggregate_substep_mean)
                + aggregate_substep_half_width
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
                    f"{PRODUCTION_SLV_BATCHES} scrambles with terminal "
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
                        f"{name}={batches}"
                        for name, batches in extensions.items()
                    )
                    part += f" ({detail})"
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
    controls = payload.get("policy", {}).get(
        "production_engine_controls", {}
    )
    lines = [
        "# ADI 2-D Snowball Greek certification",
        "",
        f"- Profile: `{profile}`",
        f"- Seed: `{payload['seed']}`",
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
            + "`, `".join(
                f"{key}={value}" for key, value in controls.items()
            )
            + "`"
            if controls
            else "- Certified engine controls: unavailable"
        ),
        f"- Hedge inception spot: `{payload['policy']['hedge_inception_spot']}` "
        "(actual index level; numerical cases are normalized)",
        "- Verdict: fine QE-M 97.5% Student-t CI + paired 4→8-substep "
        "97.5% bias upper bound + separate `n_x`/`n_v`/`n_t` PDE envelopes",
        (
            "- Heston oracle: exact terminal spot-factor integration; "
            "case-specific residual Brownian-bridge profile="
            f"{payload['policy'].get('heston_spot_bridge_profile_by_case')}"
        ),
        (
            "- SLV oracle: unbiased path-frozen-leverage conditional control; "
            f"terminal strata={payload['policy'].get('slv_spot_strata')}, "
            f"antithetic={payload['policy'].get('slv_spot_antithetic')}, "
            f"midpoint strata={payload['policy'].get('slv_spot_bridge_strata')}; "
            "no PDE control variate"
        ),
        "- Simultaneous stochastic coverage: at least 95% by Bonferroni; the "
        "spot-bump ladder is a semantic diagnostic and is not counted as PDE error",
        "- Delta bound: 0.5 hedge contracts per cell; mean signed bias bound: 0.1 contracts",
        "- Gamma bound/unit: 0.5 change in futures hedge contracts under a 1% spot move",
        "",
        "## Routing decision",
        "",
        "| Variant | Route | Cells | Mean signed delta bias: estimate [interval] | Status | Reason |",
        "|---|---|---:|---:|---:|---|",
    ]
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
            "| Variant | Case | Tags | target grid | v grid | monotone V | fallback rows | Axis refinement | Delta: estimate [interval] | Delta | Gamma: estimate [interval] | Gamma | Cell |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cells"]:
        diagnostics = row["variance_operator"]
        delta = row["certifications"]["delta"]["verdict"]
        gamma = row["certifications"]["gamma"]["verdict"]
        grid = row["target_grid"]
        lines.append(
            f"| {row['variant']} | {row['case']['name']} | "
            f"{', '.join(row['case']['tags'])} | "
            f"{grid['n_x']}×{grid['n_v']}×{grid['n_t']} | "
            f"{diagnostics['variance_grid_mode']} | "
            f"{diagnostics['monotone']} | "
            f"{diagnostics['fallback_nodes']} | "
            f"{row['pde_refinement']['status']} | "
            f"{interval_text(delta)} | {delta['status']} | "
            f"{interval_text(gamma)} | {gamma['status']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Cell interval decomposition",
            "",
            "The substep column is the signed 4→8 mean ± its 97.5% half-width; "
            "the verdict uses `abs(mean) + half-width`.",
            "",
            "| Variant | Case | Greek | PDE | fine QE-M | fine SE | PDE−ref contracts | PDE envelope | substep mean ± half-width | total radius | interval | Status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["cells"]:
        for greek in ("delta", "gamma"):
            certification = row["certifications"][greek]
            component = certification[
                "reference_substep_components_contracts"
            ]
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


def validate_payload(payload: dict) -> None:
    """Fail closed before a malformed certification artifact is published."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certification schema version mismatch")
    if payload.get("study") != "adi_2d_snowball_greek_certification":
        raise ValueError("certification study tag mismatch")
    if int(payload.get("batches", 0)) < 2:
        raise ValueError("certification requires at least two RQMC batches")
    allowed_statuses = {status.value for status in EquivalenceStatus}
    hedge_inception_spot = float(
        payload.get("policy", {}).get("hedge_inception_spot", float("nan"))
    )
    if not np.isfinite(hedge_inception_spot) or hedge_inception_spot <= 0.0:
        raise ValueError("certification lacks a valid hedge inception spot")
    for cell in payload.get("cells", []):
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
        expected_substeps = (1, 2) if payload.get("quick") else (4, 8)
        actual_substeps = (
            int(reference.get("target_substeps_per_interval", 0)),
            int(reference.get("fine_substeps_per_interval", 0)),
        )
        if actual_substeps != expected_substeps or reference.get("primary") != "fine":
            raise ValueError("certification cell uses an invalid QE-M refinement contract")
        refinement = cell.get("pde_refinement", {})
        if refinement.get("status") not in allowed_statuses:
            raise ValueError("certification cell lacks valid PDE-axis refinement evidence")
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
                    raise ValueError(f"certification {greek} verdict has non-finite {key}")
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
            if int(decision.get("aggregate_common_scrambles", 0)) != (
                expected_common
            ):
                raise ValueError(
                    f"{variant}: invalid aggregate common-scramble count"
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
        _canonical_sha256(run_configuration)
        != payload["run_configuration_sha256"]
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
    if (
        run_configuration.get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
        or payload.get("policy", {}).get("production_engine_controls")
        != PRODUCTION_ENGINE_CONTROLS
    ):
        raise ValueError("certification uses stale production engine controls")
    if run_configuration.get(
        "heston_spot_bridge_profile_by_case"
    ) != payload.get("policy", {}).get("heston_spot_bridge_profile_by_case"):
        raise ValueError("certification Heston bridge profile metadata mismatch")
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
                if int(sampling["batches"]) < PRODUCTION_SLV_BATCHES:
                    raise ValueError(
                        f"{variant}: insufficient production sampling for PDE admission"
                    )
                policy = payload.get("policy", {})
                if (
                    int(policy.get("slv_spot_strata", 0))
                    != SLV_SPOT_STRATA
                    or bool(policy.get("slv_spot_antithetic", False))
                    != SLV_SPOT_ANTITHETIC
                    or int(policy.get("slv_spot_bridge_strata", 0))
                    != SLV_SPOT_BRIDGE_STRATA
                ):
                    raise ValueError(
                        "heston_slv: stale conditional sampling profile"
                    )
            else:
                if (
                    int(sampling["batches"]) != PRODUCTION_HESTON_BATCHES
                    or _normalized_batches_by_case(sampling)
                    != PRODUCTION_HESTON_BATCHES_BY_CASE
                ):
                    raise ValueError("heston: stale case-specific batch profile")
                if payload.get("policy", {}).get(
                    "heston_spot_bridge_profile_by_case"
                ) != HESTON_SPOT_BRIDGE_PROFILE_BY_CASE:
                    raise ValueError("heston: stale conditional sampling profile")
    for cell in payload.get("cells", []):
        variant = cell["variant"]
        case_name = str(cell.get("case", {}).get("name"))
        expected_batches = _sampling_batches_for_case(
            sampling_by_variant[variant], variant, case_name
        )
        if variant == "heston":
            declared_profile = run_configuration.get(
                "heston_spot_bridge_profile_by_case", {}
            ).get(case_name, {})
            actual_profile = {
                "strata": int(
                    cell.get("reference", {}).get(
                        "heston_spot_bridge_strata", 0
                    )
                ),
                "dimensions": int(
                    cell.get("reference", {}).get(
                        "heston_spot_bridge_dimensions", 0
                    )
                ),
            }
            if actual_profile != declared_profile:
                raise ValueError(
                    f"heston/{case_name}: conditional sampling profile mismatch"
                )
        for level in ("target", "fine"):
            level_evidence = cell.get("reference", {}).get(level, {})
            actual_batches = int(level_evidence.get("batches_used", 0))
            if actual_batches != expected_batches:
                raise ValueError(
                    f"{variant}: {level} batches do not match sampling metadata"
                )
            estimates = np.asarray(level_evidence.get("batch_estimates"), dtype=float)
            covariance = np.asarray(level_evidence.get("covariance"), dtype=float)
            if estimates.shape != (expected_batches, 5) or not np.all(
                np.isfinite(estimates)
            ):
                raise ValueError(f"{variant}: invalid {level} paired batch evidence")
            if covariance.shape != (5, 5) or not np.all(np.isfinite(covariance)):
                raise ValueError(f"{variant}: invalid {level} covariance evidence")
            control_estimates = level_evidence.get("control_batch_estimates")
            if variant == "heston_slv":
                control_estimates = np.asarray(control_estimates, dtype=float)
                if control_estimates.shape != (expected_batches, 5) or not np.all(
                    np.isfinite(control_estimates)
                ):
                    raise ValueError(
                        f"{variant}: invalid {level} conditional-control evidence"
                    )
            elif control_estimates is not None:
                raise ValueError(
                    f"{variant}: unexpected {level} conditional-control evidence"
                )
        for greek in ("delta", "gamma"):
            batches = np.asarray(
                cell["certifications"][greek].get(
                    "reference_substep_batch_contracts"
                ),
                dtype=float,
            )
            if batches.shape != (expected_batches,) or not np.all(
                np.isfinite(batches)
            ):
                raise ValueError(
                    f"{variant}: invalid {greek} substep batch evidence"
                )


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
) -> None:
    record = {
        "schema_version": SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification_checkpoint",
        "run_configuration_sha256": run_configuration_sha256,
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
):
    path = _checkpoint_path(output_dir, name)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid certification checkpoint {path}") from exc
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("study")
        != "adi_2d_snowball_greek_certification_checkpoint"
        or record.get("kind") != kind
    ):
        raise ValueError(f"checkpoint metadata mismatch: {path}")
    if record.get("run_configuration_sha256") != run_configuration_sha256:
        raise ValueError(
            f"checkpoint configuration mismatch: {path}; use a new output "
            "directory or rerun without --resume"
        )
    return record.get("evidence")


def build_decision_payload(payload: dict) -> dict:
    """Build a self-hashed, standalone routing decision."""
    decision = {
        "schema_version": SCHEMA_VERSION,
        "evidence_sha256": payload["evidence_sha256"],
        "implementation_sha256": payload["implementation_sha256"],
        "run_configuration_sha256": payload["run_configuration_sha256"],
        "run_configuration": payload["run_configuration"],
        "runtime_environment": payload["runtime_environment"],
        "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        "quick": payload["quick"],
        "decisions": payload["decisions"],
    }
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
    payload["evidence_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    json_text = (
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    report = render_markdown(payload)
    decision = build_decision_payload(payload)
    _atomic_write(output_dir / "adi_greek_certification.json", json_text)
    _atomic_write(output_dir / "adi_greek_certification.md", report)
    _atomic_write(
        output_dir / "adi_greek_certification_decision.json",
        json.dumps(decision, indent=2, sort_keys=True, default=_json_default)
        + "\n",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="non-production smoke profile")
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
        "--rqmc-batch-workers",
        type=int,
        help=(
            "parallel workers for independent RQMC scrambles "
            f"(production default: {PRODUCTION_RQMC_BATCH_WORKERS})"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not np.isfinite(args.hedge_inception_spot) or args.hedge_inception_spot <= 0.0:
        raise ValueError("hedge-inception-spot must be finite and positive")
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
            args.rqmc_batch_workers,
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
        payload = rescore_payload(source, args.hedge_inception_spot)
        publish_payload(payload, args.output_dir)
        print(f"[adi-greeks] rescored {args.rescore_evidence}", flush=True)
        print(f"[adi-greeks] wrote {args.output_dir}", flush=True)
        return 0

    heston_default_paths = (
        256 if args.quick else PRODUCTION_HESTON_PATHS_PER_BATCH
    )
    heston_default_batches = 4 if args.quick else PRODUCTION_HESTON_BATCHES
    slv_default_paths = 256 if args.quick else PRODUCTION_SLV_PATHS_PER_BATCH
    slv_default_batches = 4 if args.quick else PRODUCTION_SLV_BATCHES
    common_paths = args.paths_per_batch
    common_batches = args.batches
    default_paths = (
        heston_default_paths if common_paths is None else common_paths
    )
    default_batches = (
        heston_default_batches if common_batches is None else common_batches
    )
    sampling_by_variant = {
        "heston": {
            "paths_per_batch": int(
                (
                    heston_default_paths
                    if common_paths is None
                    else common_paths
                )
                if args.heston_paths_per_batch is None
                else args.heston_paths_per_batch
            ),
            "batches": int(
                (
                    heston_default_batches
                    if common_batches is None
                    else common_batches
                )
                if args.heston_batches is None
                else args.heston_batches
            ),
        },
        "heston_slv": {
            "paths_per_batch": int(
                (
                    slv_default_paths if common_paths is None else common_paths
                )
                if args.heston_slv_paths_per_batch is None
                else args.heston_slv_paths_per_batch
            ),
            "batches": int(
                (
                    slv_default_batches
                    if common_batches is None
                    else common_batches
                )
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
    sampling_by_variant["heston"]["batches_by_case"] = (
        heston_batches_by_case
    )
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
    for variant, sampling in sampling_by_variant.items():
        if sampling["paths_per_batch"] <= 0 or sampling["batches"] < 2:
            raise ValueError(
                f"{variant}: paths-per-batch must be positive and batches must be >= 2"
            )
        paths_per_batch = sampling["paths_per_batch"]
        if paths_per_batch & (paths_per_batch - 1):
            raise ValueError(
                f"{variant}: Sobol paths-per-batch must be a power of two"
            )
        if variant == "heston" and any(
            int(batches) < 2
            for batches in sampling["batches_by_case"].values()
        ):
            raise ValueError(
                "heston: every case-specific batch count must be >= 2"
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
    if rqmc_batch_workers < 1:
        raise ValueError("rqmc-batch-workers must be a positive integer")
    cases = certification_cases(quick=args.quick)
    if args.cases:
        wanted = set(args.cases)
        unknown = wanted - {case.name for case in certification_cases(quick=False)}
        if unknown:
            raise ValueError(f"unknown certification cases: {sorted(unknown)}")
        cases = [case for case in certification_cases(quick=False) if case.name in wanted]
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
            raise ValueError(
                "heston bridge dimensions > 1 require bridge strata > 1"
            )
        heston_spot_bridge_profile_by_case = {
            case.name: {
                "strata": override_strata,
                "dimensions": override_dimensions,
            }
            for case in cases
        }

    implementation_hash = implementation_sha256()
    runtime = runtime_environment()
    selected_sampling = {
        variant: sampling_by_variant[variant] for variant in args.variants
    }
    run_configuration = {
        "schema_version": SCHEMA_VERSION,
        "implementation_sha256": implementation_hash,
        "runtime_environment": runtime,
        "production_engine_controls": PRODUCTION_ENGINE_CONTROLS,
        "quick": bool(args.quick),
        "skip_anchors": bool(args.skip_anchors),
        "seed": int(args.seed),
        "hedge_inception_spot": float(args.hedge_inception_spot),
        "variants": list(args.variants),
        "cases": [case.as_dict() for case in cases],
        "sampling_by_variant": selected_sampling,
        "spot_bump": SPOT_BUMP,
        "full_bump_ladder": list(FULL_BUMP_LADDER),
        "stochastic_component_confidence": STOCHASTIC_COMPONENT_CONFIDENCE,
        "heston_spot_bridge_profile_by_case": (
            heston_spot_bridge_profile_by_case
        ),
        "slv_spot_strata": slv_spot_strata,
        "slv_spot_antithetic": slv_spot_antithetic,
        "slv_spot_bridge_strata": slv_spot_bridge_strata,
        "rqmc_batch_workers": rqmc_batch_workers,
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
    cells = []
    for variant in args.variants:
        sampling = selected_sampling[variant]
        for case in cases:
            checkpoint_name = f"{variant}__{case.name}"
            cell = None
            if args.resume:
                cell = _load_checkpoint(
                    args.output_dir,
                    checkpoint_name,
                    run_configuration_sha256=run_configuration_hash,
                    kind="cell",
                )
                if cell is not None:
                    print(
                        f"[adi-greeks] resumed {variant}/{case.name}", flush=True
                    )
            if cell is None:
                print(f"[adi-greeks] {variant}/{case.name}", flush=True)
                heston_bridge_profile = heston_spot_bridge_profile_by_case[
                    case.name
                ]
                cell = certify_case(
                    variant,
                    case,
                    quick=args.quick,
                    paths_per_batch=sampling["paths_per_batch"],
                    batches=_sampling_batches_for_case(
                        sampling, variant, case.name
                    ),
                    seed=args.seed,
                    hedge_inception_spot=args.hedge_inception_spot,
                    heston_spot_bridge_strata=heston_bridge_profile["strata"],
                    heston_spot_bridge_dimensions=heston_bridge_profile[
                        "dimensions"
                    ],
                    slv_spot_strata=slv_spot_strata,
                    slv_spot_antithetic=slv_spot_antithetic,
                    slv_spot_bridge_strata=slv_spot_bridge_strata,
                    rqmc_batch_workers=rqmc_batch_workers,
                )
                _write_checkpoint(
                    args.output_dir,
                    checkpoint_name,
                    run_configuration_sha256=run_configuration_hash,
                    kind="cell",
                    evidence=cell,
                )
                print(
                    f"[adi-greeks] checkpointed {variant}/{case.name}: "
                    f"{cell['status']}",
                    flush=True,
                )
            cells.append(cell)
    decisions = make_decisions(
        cells,
        anchors,
        quick=(args.quick or args.skip_anchors),
        variants=args.variants,
        sampling_by_variant=selected_sampling,
        heston_spot_bridge_profile_by_case=(
            heston_spot_bridge_profile_by_case
        ),
        slv_spot_strata=slv_spot_strata,
        slv_spot_antithetic=slv_spot_antithetic,
        slv_spot_bridge_strata=slv_spot_bridge_strata,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "adi_2d_snowball_greek_certification",
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
        "seed": int(args.seed),
        "paths_per_batch": int(default_paths),
        "batches": int(default_batches),
        "sampling_by_variant": selected_sampling,
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
            "gamma_cell_bound_contracts_for_1pct_move": GAMMA_CELL_BOUND_CONTRACTS,
            "hedge_inception_spot": float(args.hedge_inception_spot),
            "hedge_inception_spot_policy": (
                "strictest minimum from the frozen 27-inception MO cohort"
            ),
            "unresolved_route": "excluded_greek_unresolved",
            "heston_spot_bridge_profile_by_case": (
                heston_spot_bridge_profile_by_case
            ),
            "slv_spot_strata": slv_spot_strata,
            "slv_spot_antithetic": slv_spot_antithetic,
            "slv_spot_bridge_strata": slv_spot_bridge_strata,
            "rqmc_batch_workers": rqmc_batch_workers,
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
