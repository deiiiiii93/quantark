"""Stage 11 - engine-admission gate (G2) for the six snowball study variants.

Decides, per study variant (``flat_bsm``, ``flat_bsm_quad``, ``ts_bsm``,
``localvol``, ``heston``, ``heston_slv``), whether the production OTC backtest
may route pricing through its production engine or must fall back to an
independent reference method -- see ``GATE_PAIRS`` for the production/reference
pair table (§5.1).  For ``heston``/``heston_slv`` this is literally PDE-vs-MC
admission, as it always was: the 2D-ADI solvers (``HestonSnowballPDESolver`` /
``HestonSLVSnowballPDESolver``) against the QE Monte-Carlo engines
(``QESnowballMCEngine`` / ``HestonSLVQESnowballMCEngine``).  The other four
variants gate a 1D-PDE-vs-QUAD or PDE-vs-LV-MC pair the same way.  It prices
the production 3Y snowball (KO 103% monthly with a 3-month lockout -> 34
observations, KI 75% with daily-discrete monitoring from the trading calendar,
principal excluded, ko_rate = rebate_rate = 15%, r = 2% flat, q from the
artifact's parity forward pillars) on a sample of admitted IV-surface dates
spanning market regimes, compares each variant's production refinement ladder
(coarse/medium/fine) against its reference (RQMC with reported standard errors
for the three MC-referenced variants, a fixed finer deterministic config for
the other three), and emits:

- ``pde_convergence_gate.json`` : per date x case(full/decayed) x variant x
  level cell results (prices, diffs, tolerances, pass/fail, timings).
- ``gate_decision.json``        : per variant route ("pde"|"mc") + params +
  tolerance + sha256 evidence hash of the gate JSON (timings stripped).
- ``gate_report.md``            : readable summary tables + rationale.

Gate rule (per variant): the production engine is admitted only if the MEDIUM
grid passes ``|production - reference| <= max(2 * mc_se, TOL_ABS)`` (mc_se = 0
for a deterministic reference; TOL_ABS = 0.25% of notional) on EVERY sampled
date/case, shows no systematic sign bias (a biased production engine fails
even inside tolerance), the FINE grid also passes, |fine - medium| <=
TOL_ABS everywhere (no drift), AND the production engine's bumped delta
agrees with the reference's within half a futures contract's worth of
per-unit delta on every sampled date, with no systematic one-sided delta
bias across them (spec §5.3) -- a snowball is delta-hedged daily, so a
PV-only gate would admit an engine that manufactures spurious hedge trades
on every rebalance.  Any other outcome routes the variant away from its
production engine, with the ladder reported honestly.  See ``GATE_PAIRS``,
``delta_quantum_per_unit``, ``delta_cell_passed`` and ``detect_delta_bias``.

Determinism: fixed seeds and fixed Sobol batches everywhere; rerunning the
study reproduces identical JSON modulo ``"timings"`` fields (excluded from the
evidence hash).  This script never modifies engine/kernel files; it only reads
data and writes its own output artifacts (atomically).

Run:  .venv/bin/python example/mo_volmodels/11_pde_convergence_gate.py [--quick]
"""

from __future__ import annotations

import argparse
import calendar as _calendar_mod
import hashlib
import json
import math
import os
import tempfile
import time
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.backtest.replay.config import VolModelCalibrationConfig
from quantark.volmodels.calibration import (
    VOL_MODEL_HESTON,
    VOL_MODEL_HESTON_SLV,
    VOL_MODEL_LOCALVOL,
    VolModelCalibrator,
)
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.param import FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_divide
from quantark.volmodels.localvol import build_dupire_local_vol

# ---------------------------------------------------------------------------
# Frozen study configuration
# ---------------------------------------------------------------------------

# VARIANTS is derived from GATE_PAIRS (defined below, near the engine
# factories) so the study can never drift from the pair table -- see §5.1.
CASES = ("full", "decayed")
LEVELS = ("coarse", "medium", "fine")

# vol_model names the calibrator actually fits (the flat/term-structure BSM
# variants price directly off the environment's vol surface -- no per-day
# calibration, model=None).
_CALIBRATED_VARIANTS = (VOL_MODEL_LOCALVOL, VOL_MODEL_HESTON, VOL_MODEL_HESTON_SLV)

# Sample dates spanning regimes (YYYYMMDD).  Excluded dates are substituted
# with the nearest admitted date (and the substitution is logged).
DEFAULT_SAMPLE_DATES = (
    "20230515",  # 2023-05 calm regime
    "20231115",  # 2023 Q4 ordinary mid-regime
    "20240207",  # 2024-02 crash window (excluded -> substitutes to 2024-02-08)
    "20240614",  # 2024 mid ordinary
    "20241010",  # 2024-10 stimulus spike aftermath
    "20250113",  # 2025-01 ordinary
    "20250409",  # 2025-04 tariff crash
    "20260715",  # 2026-07 recent
)

FLAT_RATE = 0.02  # production flat rate channel
MATURITY_MONTHS = 36
LOCKOUT_MONTHS = 3  # first KO observation at month 3 -> 34 KO dates
KO_PCT = 1.03
KI_PCT = 0.75
KO_RATE = 0.15  # repo-standard snowball coupon (create_standard_snowball default)
ACT = 365.0  # year-fraction convention for calendar-derived observation dates

TOL_ABS_PCT = 0.25  # gate tolerance floor, % of notional
MC_SE_FACTOR = 2.0  # tolerance also covers 2x the reported MC standard error
BIAS_SIGN_FRACTION = 0.9  # >=90% same-signed diffs ...
BIAS_MEDIAN_FRACTION = 0.5  # ... with median |diff| >= 0.5 * TOL_ABS => systematic bias
SPOT_BUMP = 0.01  # delta bump (matches BumpConfig default)

SEED = 20260723
# Reference RQMC: 16 batches x 8192 paths = 131072 >= 100k, batch-spread SE.
#
# The reference runs QE-M (martingale-corrected QE, MC_MARTINGALE below) with
# 4 substeps per observation interval.  Plain QE on the raw observation grid
# carries a measured 0.078% of notional discretization bias against the same
# engine at 8 substeps -- small next to the 0.25% tolerance floor, but a
# reference should not spend any of the budget it is meant to measure against,
# and both corrections are cheap.  QUICK stays at 1 substep: it is a plumbing
# smoke whose route decision is already marked non-production-valid.
MC_MARTINGALE = True
MC_FULL = {"paths_per_batch": 8192, "batches": 16, "substeps_per_interval": 4}
MC_QUICK = {"paths_per_batch": 512, "batches": 4, "substeps_per_interval": 1}

# PDE ladder (n_x, n_v, n_t).  Coarse mirrors the stage-08 grid (90/36/96 at
# T=3).  Medium uses n_t = ceil(400*T) so dt <= 1/400y: strictly finer than the
# minimum 1-calendar-day gap between KI observation dates, which guarantees no
# two discrete events round to the same ADI step.  Fine doubles n_t and refines
# the spatial grid.  n_t scales with the case maturity; n_x/n_v are fixed.
LADDER_COARSE = (90, 36)
LADDER_MEDIUM = (200, 60)
LADDER_FINE = (300, 90)
LADDER_QUICK = (("coarse", 24, 12, 16), ("medium", 32, 16, 24), ("fine", 40, 20, 32))

# Decayed-case eligibility: remaining maturity window (years) at the decayed
# valuation date; outside this window the case is skipped (and logged).
DECAY_TARGET_MONTHS = 24
DECAY_MIN_REMAINING = 0.5
DECAY_MAX_REMAINING = 2.0

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, _calendar_mod.monthrange(y, m)[1])
    return date(y, m, day)


class TradingCalendar:
    """Trading-day calendar from the history's spot CSV.

    Beyond the last CSV date the calendar extends with plain weekdays (holiday
    schedule unknown); the extension is logged in the study metadata.
    """

    def __init__(self, days: Sequence[date]) -> None:
        if not days:
            raise ValueError("TradingCalendar needs at least one day")
        self._days = sorted(set(days))
        self._set = set(self._days)

    @classmethod
    def from_spot_csv(cls, path: os.PathLike | str) -> "TradingCalendar":
        days: List[date] = []
        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline()
            if not header.lower().startswith("date"):
                raise ValueError(f"spot CSV {path} missing 'date' header")
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                days.append(datetime.strptime(line.split(",")[0], "%Y-%m-%d").date())
        return cls(days)

    @property
    def first(self) -> date:
        return self._days[0]

    @property
    def last(self) -> date:
        return self._days[-1]

    def is_trading_day(self, d: date) -> bool:
        if d in self._set:
            return True
        return d > self._days[-1] and d.weekday() < 5

    def next_trading_day(self, d: date) -> date:
        """First trading day on or after ``d`` (weekday extension past the CSV)."""
        i = bisect_left(self._days, d)
        if i < len(self._days):
            return self._days[i]
        cur = d
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
        return cur

    def trading_days_between(self, start: date, end: date) -> List[date]:
        """All trading days t with ``start < t <= end``."""
        out: List[date] = []
        cur = start
        while cur < end:
            cur += timedelta(days=1)
            if self.is_trading_day(cur):
                out.append(cur)
        return out


def _parse_yyyymmdd(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse date {value!r} (expected YYYYMMDD or YYYY-MM-DD)")


def resolve_sample_dates(
    admitted: Sequence[date], requested: Sequence[date]
) -> List[Dict[str, Any]]:
    """Map each requested date to an admitted one (nearest on exclusion).

    Returns one record per requested date: ``{"requested", "date", "substituted"}``.
    Tie-breaking is deterministic (earlier admitted date wins).
    """
    admitted = sorted(admitted)
    if not admitted:
        raise ValueError("no admitted dates supplied")
    out = []
    for req in requested:
        if req in admitted:
            out.append(
                {"requested": req.isoformat(), "date": req.isoformat(), "substituted": False}
            )
            continue
        nearest = min(admitted, key=lambda a: (abs((a - req).days), a))
        out.append(
            {
                "requested": req.isoformat(),
                "date": nearest.isoformat(),
                "substituted": True,
            }
        )
    # De-duplicate resolved dates (two requests may map to the same admitted date).
    seen: set = set()
    deduped = []
    for rec in out:
        if rec["date"] in seen:
            continue
        seen.add(rec["date"])
        deduped.append(rec)
    return deduped


# ---------------------------------------------------------------------------
# Term sheet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnowballTerms:
    """Production snowball term sheet, as year fractions from ``valuation``.

    ``inception`` is the contractual strike/barrier fixing date; ``valuation``
    equals inception for the full trade and the decay date for the decayed
    variant (times are then shifted so t=0 is the decayed valuation).
    """

    inception: date
    valuation: date
    maturity_date: date
    maturity_years: float
    ko_times: Tuple[float, ...]  # monthly, first at ~month 3, last == maturity
    ki_times: Tuple[float, ...]  # daily-discrete, from the trading calendar
    ko_pct: float = KO_PCT
    ki_pct: float = KI_PCT
    ko_rate: float = KO_RATE

    def summary(self) -> Dict[str, Any]:
        return {
            "inception": self.inception.isoformat(),
            "valuation": self.valuation.isoformat(),
            "maturity_date": self.maturity_date.isoformat(),
            "maturity_years": self.maturity_years,
            "n_ko": len(self.ko_times),
            "n_ki": len(self.ki_times),
            "first_ko_time": self.ko_times[0] if self.ko_times else None,
            "ko_pct": self.ko_pct,
            "ki_pct": self.ki_pct,
            "ko_rate": self.ko_rate,
        }


def build_snowball_terms(
    inception: date,
    calendar: TradingCalendar,
    *,
    maturity_months: int = MATURITY_MONTHS,
    lockout_months: int = LOCKOUT_MONTHS,
    ko_pct: float = KO_PCT,
    ki_pct: float = KI_PCT,
    ko_rate: float = KO_RATE,
) -> SnowballTerms:
    """Build the 3Y production term sheet on ``inception``.

    KO: month anniversaries lockout..36 snapped to the next trading day
    (34 dates when lockout=3).  KI: every trading day in (inception, maturity].
    Times are ACT/365 year fractions from inception.
    """
    maturity_date = calendar.next_trading_day(add_months(inception, maturity_months))
    ko_dates: List[date] = []
    for m in range(lockout_months, maturity_months + 1):
        kd = calendar.next_trading_day(add_months(inception, m))
        if kd > maturity_date:
            kd = maturity_date
        ko_dates.append(kd)
    ko_dates = sorted(set(ko_dates))
    ki_days = calendar.trading_days_between(inception, maturity_date)
    t0 = inception
    maturity_years = (maturity_date - t0).days / ACT
    ko_times = tuple((d - t0).days / ACT for d in ko_dates)
    ki_times = tuple((d - t0).days / ACT for d in ki_days)
    if not ko_times or abs(ko_times[-1] - maturity_years) > 1e-12:
        raise ValueError("last KO observation must coincide with maturity")
    if not ki_times or ki_times[-1] > maturity_years + 1e-12:
        raise ValueError("KI schedule must end at or before maturity")
    return SnowballTerms(
        inception=inception,
        valuation=inception,
        maturity_date=maturity_date,
        maturity_years=maturity_years,
        ko_times=ko_times,
        ki_times=ki_times,
        ko_pct=ko_pct,
        ki_pct=ki_pct,
        ko_rate=ko_rate,
    )


def pick_decay_date(
    admitted: Sequence[date],
    terms: SnowballTerms,
    *,
    target_months: int = DECAY_TARGET_MONTHS,
    min_remaining: float = DECAY_MIN_REMAINING,
    max_remaining: float = DECAY_MAX_REMAINING,
) -> Optional[date]:
    """Latest admitted date <= inception + ``target_months`` with a usable remainder.

    Returns None (case skipped, logged) when no admitted date leaves a
    remaining maturity inside [min_remaining, max_remaining].
    """
    target = add_months(terms.inception, target_months)
    candidates = [
        a for a in admitted if terms.inception < a <= target and a < terms.maturity_date
    ]
    if not candidates:
        return None
    decay = max(candidates)
    remaining = (terms.maturity_date - decay).days / ACT
    if not (min_remaining <= remaining <= max_remaining):
        return None
    return decay


def decay_terms(terms: SnowballTerms, decay_date: date) -> SnowballTerms:
    """Shift the term sheet to a later valuation date (same absolute contract)."""
    t_shift = (decay_date - terms.inception).days / ACT
    if t_shift <= 0:
        raise ValueError("decay date must be after inception")
    ko = tuple(t - t_shift for t in terms.ko_times if t > t_shift)
    ki = tuple(t - t_shift for t in terms.ki_times if t > t_shift)
    remaining = terms.maturity_years - t_shift
    if not ko or not ki or remaining <= 0:
        raise ValueError("decayed term sheet has no remaining observations")
    return SnowballTerms(
        inception=terms.inception,
        valuation=decay_date,
        maturity_date=terms.maturity_date,
        maturity_years=remaining,
        ko_times=ko,
        ki_times=ki,
        ko_pct=terms.ko_pct,
        ki_pct=terms.ki_pct,
        ko_rate=terms.ko_rate,
    )


def build_snowball_product(terms: SnowballTerms, s0: float):
    """Create the SnowballOption: barriers/strike fixed off the INCEPTION spot."""
    s0 = float(s0)
    return create_standard_snowball(
        initial_price=s0,
        strike=s0,
        maturity=float(terms.maturity_years),
        contract_multiplier=1.0,
        ko_barrier=terms.ko_pct * s0,
        ko_rate=terms.ko_rate,
        ki_barrier=terms.ki_pct * s0,
        num_observations=len(terms.ko_times),
        is_reverse=False,
        ko_observation_dates=list(terms.ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(terms.ki_times),
        rebate_rate=terms.ko_rate,
        include_principal=False,
    )


# ---------------------------------------------------------------------------
# Pricing environment / calibration
# ---------------------------------------------------------------------------

def build_pricing_env(
    artifact: IvSurfaceArtifact,
    rate: float = FLAT_RATE,
    *,
    surface_vol_mode: str,
    remaining_maturity_years: Optional[float] = None,
) -> PricingEnvironment:
    """Production env: flat rate, dividend curve from the artifact's
    option-parity forward pillars.  ``surface_vol_mode`` selects the vol
    surface exactly as ``ProductReplay._vol_and_dividend`` does (spec §5.5,
    quantark/backtest/replay/product_replay.py:225-238) -- a variant's route
    is only independent evidence if it is priced off the SAME market data
    the fleet will use, and pricing every variant off ``grid_vol_surface()``
    is what made ``ts_bsm`` and ``flat_bsm`` bitwise identical (Task 9).  No
    default: a missing mode must fail loudly, not silently fall back to
    ``full_grid``.
    """
    if surface_vol_mode == "term_structure":
        vol_surface = artifact.term_structure_vol_surface()
    elif surface_vol_mode == "full_grid":
        vol_surface = artifact.grid_vol_surface()
    elif surface_vol_mode == "flat_atm_remaining":
        if remaining_maturity_years is None:
            raise ValidationError(
                "surface_vol_mode='flat_atm_remaining' requires remaining_maturity_years"
            )
        atm_term_structure = artifact.term_structure_vol_surface()
        vol_surface = FlatVolSurface(
            volatility=float(
                atm_term_structure.get_vol(0.0, remaining_maturity_years, 0.0)
            )
        )
    else:
        raise ValidationError(f"Unknown surface_vol_mode: {surface_vol_mode!r}")
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=rate),
        valuation_date=datetime.combine(artifact.trade_date, datetime.min.time()),
        spot_quote=SpotQuote(spot=float(artifact.s0)),
        vol_surface=vol_surface,
        div_yield=artifact.term_structure_dividend_yield(rate),
    )


def _envs_by_mode(
    artifact: IvSurfaceArtifact, rate: float, remaining_maturity_years: float
) -> Dict[str, PricingEnvironment]:
    """One env per DISTINCT ``surface_vol_mode`` in ``GATE_PAIRS`` (three),
    not one per variant (six): the three ``full_grid`` variants share a
    single env, so building per-variant would construct the same full-grid
    surface three times over.  ``_evaluate_case`` / ``_extras_first_date``
    look an env up by ``GATE_PAIRS[variant].surface_vol_mode``.
    """
    modes = {pair.surface_vol_mode for pair in GATE_PAIRS.values()}
    return {
        mode: build_pricing_env(
            artifact,
            rate,
            surface_vol_mode=mode,
            remaining_maturity_years=(
                remaining_maturity_years if mode == "flat_atm_remaining" else None
            ),
        )
        for mode in modes
    }


def make_calibrator(out_dir: Path, quick: bool) -> VolModelCalibrator:
    cfg = VolModelCalibrationConfig(
        cache_dir=str(out_dir / "calibration_cache"),
        slv_n_steps=12 if quick else 40,
        slv_n_x=61 if quick else 161,
        slv_n_z=31 if quick else 81,
    )
    return VolModelCalibrator(cfg)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _make_mc_params(mc_cfg: Dict[str, Any], seed: int) -> MCParams:
    return MCParams(
        num_paths=int(mc_cfg["paths_per_batch"]),
        seed=int(seed),
        use_antithetic=False,
        rqmc_min_batches=int(mc_cfg["batches"]),
        rqmc_max_batches=int(mc_cfg["batches"]),
        rqmc_target_std=1e-12,  # force all batches: fixed-cost, fixed-seed reference
    )


def _make_mc_engine(variant: str, model, mcp: MCParams, substeps: int):
    if variant == VOL_MODEL_HESTON:
        return QESnowballMCEngine(
            model.heston_params,
            params=mcp,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            substeps_per_interval=substeps,
            martingale_correction=MC_MARTINGALE,
        )
    return HestonSLVQESnowballMCEngine(
        model.heston_params,
        params=mcp,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=model.leverage_surface,
        eta=float(model.slv_eta if model.slv_eta is not None else 1.0),
        local_vol_surface=model.local_vol_surface,
        substeps_per_interval=substeps,
        martingale_correction=MC_MARTINGALE,
    )


def _make_pde_engine(variant: str, model, pdep: PDEParams, grid: Tuple[int, int, int]):
    n_x, n_v, n_t = grid
    if variant == VOL_MODEL_HESTON:
        return HestonSnowballPDESolver(
            model_params=model.heston_params, params=pdep, n_x=n_x, n_v=n_v, n_t=n_t
        )
    return HestonSLVSnowballPDESolver(
        model_params=model.heston_params,
        leverage_surface=model.leverage_surface,
        eta=float(model.slv_eta if model.slv_eta is not None else 1.0),
        params=pdep,
        n_x=n_x,
        n_v=n_v,
        n_t=n_t,
    )


# QUAD ladder over grid_points, and the 1D-PDE ladder over accuracy profiles.
# Both mirror the 2D ladder's shape: coarse -> medium -> fine, with 'medium'
# the level the fleet would actually run.  QUAD_REFERENCE_POINTS is a fixed
# reference config (>= 4x the QUAD production default of 1001) so a QUAD
# reference is genuinely finer than anything on the QUAD production ladder.
QUAD_LADDER = {"coarse": 2048, "medium": 4096, "fine": 8192}
QUAD_REFERENCE_POINTS = 16384          # >= 4x the QUAD production default
PDE1D_LADDER = {"coarse": "fast", "medium": "standard", "fine": "high"}


@dataclass(frozen=True)
class GatePair:
    """One variant's production engine and its independent reference.

    ``production`` / ``reference`` are labels recorded in the evidence; the
    builders are what actually run.  The two must be different numerical
    methods, or the cell is a common-mode comparison and proves nothing.

    Builder signature: ``(model, grid) -> engine``.  ``model`` is the
    CalibratedVolModel for calibrated variants and ``None`` for the BSM ones;
    ``grid`` is the ladder level's knob dict (see _production_grid below) for
    the production builder, and ``None`` for the reference builder (the
    reference is priced once at a fixed, finer configuration -- it is never
    laddered).  ``reference_is_mc`` drives whether a std error is expected: a
    deterministic reference has ``mc_se = None`` and the flat TOL_ABS floor.

    ``surface_vol_mode`` must match the fleet's ``VariantSpec.surface_vol_mode``
    for the same variant (see ``build_pricing_env`` and Task 9) -- no default,
    since defaulting is how a new variant silently gets the wrong surface.
    """

    production: str
    reference: str
    build_production: Callable[..., Any]
    build_reference: Callable[..., Any]
    reference_is_mc: bool
    surface_vol_mode: str


def _bsm_pde(accuracy: str = "standard"):
    """1D BSM PDE builder.  ``grid`` (production) overrides ``accuracy``;
    ``grid=None`` (reference) keeps the factory-bound fixed profile."""
    def build(model, grid):
        acc = accuracy if grid is None else grid["accuracy"]
        return SnowballPDESolver(PDEParams(accuracy=acc))
    return build


def _bsm_quad(grid_points: Optional[int] = None):
    """BSM quadrature builder.  ``grid`` (production) overrides ``grid_points``;
    ``grid=None`` (reference) keeps the factory-bound fixed count."""
    def build(model, grid):
        gp = grid_points if grid is None else grid["grid_points"]
        params = QuadParams() if gp is None else QuadParams(grid_points=gp)
        return SnowballQuadEngine(params=params)
    return build


GATE_PAIRS: Dict[str, GatePair] = {
    "flat_bsm": GatePair(
        production="pde_1d", reference="quad_high",
        build_production=_bsm_pde("standard"),
        build_reference=_bsm_quad(grid_points=QUAD_REFERENCE_POINTS),
        reference_is_mc=False,
        surface_vol_mode="flat_atm_remaining",
    ),
    "flat_bsm_quad": GatePair(
        production="quad", reference="pde_1d_high",
        build_production=_bsm_quad(),
        build_reference=_bsm_pde("high"),
        reference_is_mc=False,
        surface_vol_mode="flat_atm_remaining",
    ),
    "ts_bsm": GatePair(
        # Mirrors flat_bsm, not flat_bsm_quad: stage 12's VARIANT_SPECS["ts_bsm"]
        # keeps the PDE default (only flat_bsm_quad overrides pricing_engine_type
        # to QUADRATURE) -- see test_gate_prices_the_same_engine_family_the_fleet_will_run.
        # surface_vol_mode differs from flat_bsm though (term_structure, not
        # flat_atm_remaining) -- that's the whole point of Task 9.
        production="pde_1d", reference="quad_high",
        build_production=_bsm_pde("standard"),
        build_reference=_bsm_quad(grid_points=QUAD_REFERENCE_POINTS),
        reference_is_mc=False,
        surface_vol_mode="term_structure",
    ),
    "localvol": GatePair(
        production="pde_1d_lv", reference="lv_mc_rqmc",
        build_production=lambda model, grid: LocalVolSnowballPDESolver(
            params=PDEParams(accuracy=grid["accuracy"]),
            local_vol_surface=model.local_vol_surface),
        build_reference=lambda model, grid: LocalVolSnowballMCEngine(
            local_vol_surface=model.local_vol_surface,
            params=_make_mc_params(MC_FULL, SEED),
            method=MonteCarloMethod.RANDOMIZED_QUASI),
        reference_is_mc=True,
        surface_vol_mode="full_grid",
    ),
    "heston": GatePair(
        production="pde_2d_adi", reference="qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON, model, PDEParams(), (grid["n_x"], grid["n_v"], grid["n_t"])),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
        surface_vol_mode="full_grid",
    ),
    "heston_slv": GatePair(
        production="pde_2d_adi_slv", reference="slv_qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON_SLV, model, PDEParams(), (grid["n_x"], grid["n_v"], grid["n_t"])),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON_SLV, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
        surface_vol_mode="full_grid",
    ),
}

VARIANTS = tuple(GATE_PAIRS)


def pde_ladder(T: float, quick: bool = False) -> List[Tuple[str, int, int, int]]:
    """Refinement ladder [(level, n_x, n_v, n_t)] for a case of maturity ``T``.

    Medium n_t = ceil(400*T) => dt <= 1/400y < one calendar day, so no two
    discrete (daily KI / monthly KO) events can round to the same ADI step.
    """
    if quick:
        return [(level, nx, nv, nt) for level, nx, nv, nt in LADDER_QUICK]
    n_t_medium = max(8, int(math.ceil(400.0 * T)))
    n_t_coarse = max(32, int(round(32.0 * T)))
    return [
        ("coarse", LADDER_COARSE[0], LADDER_COARSE[1], n_t_coarse),
        ("medium", LADDER_MEDIUM[0], LADDER_MEDIUM[1], n_t_medium),
        ("fine", LADDER_FINE[0], LADDER_FINE[1], 2 * n_t_medium),
    ]


def _production_grid(variant: str, level: str, T: float, quick: bool) -> Dict[str, Any]:
    """Ladder knob for the PRODUCTION engine of ``variant`` at ``level``.

    2D ADI ladders over (n_x, n_v, n_t); QUAD over grid_points; the 1D PDE
    over its accuracy profile.  Returns a JSON-safe dict stored unchanged in
    ``cell["grid"]``, so the evidence keeps ONE schema across families and a
    reader can always see which knob moved.
    """
    pair = GATE_PAIRS[variant]
    if pair.production.startswith("pde_2d"):
        entry = next(g for g in pde_ladder(T, quick) if g[0] == level)
        return {"kind": "adi_2d", "n_x": entry[1], "n_v": entry[2], "n_t": entry[3]}
    if pair.production.startswith("quad"):
        return {"kind": "quad", "grid_points": QUAD_LADDER[level]}
    return {"kind": "pde_1d", "accuracy": PDE1D_LADDER[level]}


def event_key_collision_count(times: Sequence[float], T: float, n_t: int) -> int:
    """Number of discrete events lost to ADI tau-key collisions at this grid.

    Mirrors ``_Heston2DSnowballPDEBase._integer_tau_key`` exactly (round of the
    event's time-to-maturity in units of dt, clamped at 0).
    """
    dt = float(T) / int(n_t)
    keys = [
        int(round(max(float(T) - float(t), 0.0) / dt))
        for t in times
        if -1e-12 <= float(t) <= float(T) + 1e-12
    ]
    return len(keys) - len(set(keys))


# ---------------------------------------------------------------------------
# Gate logic (pure functions; unit-tested)
# ---------------------------------------------------------------------------

def gate_tolerance_pct(mc_se_pct: Optional[float], tol_abs_pct: float = TOL_ABS_PCT) -> float:
    """Per-cell tolerance in % of notional: max(2 x MC SE, TOL_ABS).

    ``mc_se_pct=None`` means the reference is DETERMINISTIC (QUAD or a finer
    PDE), not that its error is zero -- the tolerance is then the flat floor.
    """
    if mc_se_pct is None:
        return float(tol_abs_pct)
    return max(MC_SE_FACTOR * float(mc_se_pct), float(tol_abs_pct))


def gate_cell_passed(
    signed_diff_pct: float, mc_se_pct: Optional[float], tol_abs_pct: float = TOL_ABS_PCT
) -> bool:
    return abs(float(signed_diff_pct)) <= gate_tolerance_pct(mc_se_pct, tol_abs_pct)


def detect_systematic_bias(
    signed_diffs_pct: Sequence[float],
    tol_abs_pct: float = TOL_ABS_PCT,
    *,
    sign_fraction: float = BIAS_SIGN_FRACTION,
    median_fraction: float = BIAS_MEDIAN_FRACTION,
) -> Tuple[bool, Dict[str, Any]]:
    """Flag a one-sided PDE bias: >= sign_fraction same-signed diffs AND median
    |diff| >= median_fraction * TOL_ABS.  Needs >= 4 cells to be meaningful."""
    diffs = [float(x) for x in signed_diffs_pct if x is not None and math.isfinite(float(x))]
    n = len(diffs)
    info: Dict[str, Any] = {"n_cells": n, "sign_fraction": None, "median_abs_pct": None}
    if n < 4:
        info["note"] = "too few cells for bias detection"
        return False, info
    positives = sum(1 for x in diffs if x > 0.0)
    frac = max(positives, n - positives) / n
    median_abs = float(np.median(np.abs(np.asarray(diffs))))
    info.update({"sign_fraction": frac, "median_abs_pct": median_abs})
    biased = frac >= sign_fraction and median_abs >= median_fraction * float(tol_abs_pct)
    return biased, info


def detect_systematic_bias_bucketed(
    cells: Sequence[Dict[str, Any]],
    tol_abs_pct: float = TOL_ABS_PCT,
) -> Tuple[bool, Dict[str, Any]]:
    """Bias detection WITHIN maturity buckets, never pooled across them.

    The 2D PDE error changes sign with remaining maturity (spec §5.2), so a
    pooled sign fraction averages the two regimes and reads as unbiased while
    each bucket is unanimous.  A variant is biased if ANY bucket with enough
    cells is biased.  Buckets below detect_systematic_bias's 4-cell minimum
    are recorded and skipped -- they cannot flag, and they cannot mask.
    """
    buckets: Dict[str, Any] = {}
    any_biased = False
    for case in CASES:
        rows = [c for c in cells if c.get("case") == case]
        diffs = [c.get("signed_diff_pct") for c in rows]
        usable = [d for d in diffs if d is not None and math.isfinite(float(d))]
        if len(usable) < 4:
            buckets[case] = {"n_cells": len(usable), "skipped": True}
            continue
        biased, info = detect_systematic_bias(usable, tol_abs_pct)
        buckets[case] = {**info, "skipped": False, "biased": bool(biased)}
        any_biased = any_biased or bool(biased)
    return any_biased, {"buckets": buckets, "pooled_not_used": True}


# ---------------------------------------------------------------------------
# Feller-regime buckets (spec §7A.4(3), §7A.11).  A pooled Gate G2 verdict
# averages every regime the enforce_feller-constrained calibration produces
# into a number that describes no actual date -- 80% of dates sit right on
# the constraint boundary (ratio ~= 1.0) while a further 6.6% satisfy Feller
# by driving sigma to its bound (ratios up to 1.7e5), a deterministic-
# variance model with no smile dynamics.  Cut points are measured, not
# chosen:
#   0.5  -- section 7A.11's measured failure boundary: unconstrained fits
#           fail the gate at ratio <= 0.50 and pass above it (16.4% of the
#           cohort sits below 0.50).
#   10   -- section 7A.10(3)'s sigma-collapse marker: 6.6% of enforced fits
#           satisfy Feller by driving sigma to its lower bound, giving a
#           deterministic-variance model that fails the gate under BOTH
#           calibration policies.
# ---------------------------------------------------------------------------

FELLER_VIOLATED_BELOW = 0.5
FELLER_DEGENERATE_ABOVE = 10.0
FELLER_BUCKETS = ("violated", "boundary", "degenerate", "unknown")
FELLER_BUCKET_MIN_CELLS = 4  # same statistical floor as detect_systematic_bias


def feller_bucket(ratio: Optional[float]) -> str:
    """Bucket a Heston Feller ratio (2*kappa*theta/sigma**2) into a regime.

    ``None`` or non-finite -- an uncomputable ratio, or a non-Heston variant
    that never carries one -- buckets as "unknown", never as "boundary": a
    date whose regime cannot be determined must not read as the common,
    passing case.
    """
    if ratio is None or not math.isfinite(float(ratio)):
        return "unknown"
    ratio = float(ratio)
    if ratio < FELLER_VIOLATED_BELOW:
        return "violated"
    if ratio > FELLER_DEGENERATE_ABOVE:
        return "degenerate"
    return "boundary"


def _feller_bucket_summary(medium_cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-Feller-regime cell counts at the medium grid.

    Reporting only -- decide_route stays all-cells-must-pass regardless of
    bucket population; this exposes WHERE a variant fails, it does not
    license failure in an unpopular regime.  A bucket below
    FELLER_BUCKET_MIN_CELLS is flagged ``skipped`` (same precedent as
    detect_systematic_bias_bucketed's thin maturity buckets), but its real
    counts are reported regardless -- a thin bucket must not read as
    "clean" by having its numbers hidden.
    """
    out: Dict[str, Any] = {}
    for bucket in FELLER_BUCKETS:
        rows = [c for c in medium_cells if feller_bucket(c.get("feller_ratio")) == bucket]
        diffs = [c["abs_diff_pct"] for c in rows if c.get("abs_diff_pct") is not None]
        out[bucket] = {
            "n_cells": len(rows),
            "n_passed": sum(1 for c in rows if c.get("passed") is True),
            "max_abs_diff_pct": max(diffs) if diffs else None,
            "skipped": len(rows) < FELLER_BUCKET_MIN_CELLS,
        }
    return out


# ---------------------------------------------------------------------------
# Hedge-derived delta admission (spec §5.3).  The gate prices a 1-unit
# product; the backtest holds NOTIONAL/s0 index units.  One IM futures
# contract covers FUTURES_MULTIPLIER index points, so it moves
# FUTURES_MULTIPLIER * s0 / STUDY_NOTIONAL of per-unit delta.  s0 varies
# 1.49x across the 27 inceptions, so this is computed per date, never fixed.
# ---------------------------------------------------------------------------

FUTURES_MULTIPLIER = 200.0
STUDY_NOTIONAL = 50_000_000.0
DELTA_CELL_CONTRACTS = 0.5    # rounding provably absorbs less than this
DELTA_BIAS_CONTRACTS = 0.1    # accumulates over ~700 rebalances


def delta_quantum_per_unit(
    s0: float,
    notional: float = STUDY_NOTIONAL,
    multiplier: float = FUTURES_MULTIPLIER,
) -> float:
    """Per-unit delta moved by exactly one IM futures contract.

    A degenerate s0/notional collapses the quantum to 0.0 (safe_divide's
    fallback) rather than raising -- that makes delta_cell_passed
    unattainable except at exact agreement, i.e. it fails closed, never
    lenient.
    """
    return float(safe_divide(float(multiplier) * float(s0), float(notional)))


def delta_cell_passed(abs_diff: float, s0: float, **kw) -> bool:
    """Delta agreement within half a futures contract's worth of per-unit delta."""
    return abs(float(abs_diff)) <= DELTA_CELL_CONTRACTS * delta_quantum_per_unit(s0, **kw)


def detect_delta_bias(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """Mean SIGNED delta difference in contracts, against the bias bound.

    Signed, not absolute: contract rounding kills symmetric noise but lets a
    one-sided bias accumulate over the rebalance schedule.  Rows missing
    ``signed_diff``/``s0`` (an unevaluable delta cell, already recorded as
    ``passed: False`` at the row level) are excluded from the mean -- exactly
    like detect_systematic_bias_bucketed's thin buckets, they can neither
    flag bias nor mask it.
    """
    contracts = [
        float(safe_divide(float(r["signed_diff"]), delta_quantum_per_unit(float(r["s0"]))))
        for r in rows
        if r.get("signed_diff") is not None and r.get("s0")
    ]
    if not contracts:
        return False, {"n_rows": 0, "mean_signed_contracts": None}
    mean_signed = float(np.mean(contracts))
    return abs(mean_signed) > DELTA_BIAS_CONTRACTS, {
        "n_rows": len(contracts),
        "mean_signed_contracts": mean_signed,
        "max_abs_contracts": max(abs(c) for c in contracts),
        "bound_contracts": DELTA_BIAS_CONTRACTS,
    }


def _finite_or_error(value: Any, what: str) -> Optional[str]:
    """Return None when ``value`` is a finite float, else an error string.

    Non-finite engine prices (NaN/inf) must become recorded error cells so the
    study completes and reports honestly instead of dying in JSON assembly.
    """
    try:
        finite = value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False
    if finite:
        return None
    return f"non-finite {what}: {value!r}"


def decide_route(
    medium_cells: Sequence[Dict[str, Any]],
    fine_cells: Sequence[Dict[str, Any]],
    delta_rows: Sequence[Dict[str, Any]],
    tol_abs_pct: float = TOL_ABS_PCT,
) -> Dict[str, Any]:
    """Route decision for one variant from its medium/fine gate cells.

    PDE is admitted only if: every medium cell passes, no systematic bias at
    medium, every fine cell passes, |fine - medium| <= TOL_ABS on every paired
    cell (no drift), AND the production engine's delta agrees with the
    reference's within half a futures contract on every sampled date with no
    systematic one-sided delta bias (spec §5.3) -- a snowball is delta-hedged
    daily, so PV agreement alone is not sufficient evidence.  Any cell
    carrying an ``error`` counts as failed.  Zero evaluable cells never
    admits PDE (``all([])`` is vacuously True; ``delta_pass`` is explicitly
    forced False on empty ``delta_rows`` to mirror this).
    """
    delta_pass = all(r.get("passed") is True for r in delta_rows) if delta_rows else False
    delta_biased, delta_info = detect_delta_bias(delta_rows)
    feller_buckets = _feller_bucket_summary(medium_cells)
    if not medium_cells or not fine_cells:
        return {
            "route": "mc",
            "medium_pass": False,
            "fine_pass": False,
            "biased": False,
            "bias": {"n_cells": 0, "note": "empty ladder"},
            "drift_max_pct": 0.0,
            "drift_location": None,
            "delta_pass": delta_pass,
            "delta_biased": bool(delta_biased),
            "delta_info": delta_info,
            "feller_buckets": feller_buckets,
            "rationale": (
                "no cells evaluated for this variant "
                f"({len(medium_cells)} medium / {len(fine_cells)} fine) - "
                "refusing to admit PDE on empty evidence"
            ),
        }
    reasons: List[str] = []
    medium_pass = all(c.get("passed") is True for c in medium_cells)
    if not medium_pass:
        failing = [
            f"{c['date']}/{c['case']}"
            for c in medium_cells
            if c.get("passed") is not True
        ]
        reasons.append(f"medium grid fails on {len(failing)} cell(s): {', '.join(failing)}")
    biased, bias_info = detect_systematic_bias_bucketed(medium_cells, tol_abs_pct)
    if biased:
        offending = [
            case for case, b in bias_info["buckets"].items()
            if not b.get("skipped") and b.get("biased")
        ]
        detail = "; ".join(
            f"{case} (sign fraction {bias_info['buckets'][case]['sign_fraction']:.2f}, "
            f"median |diff| {bias_info['buckets'][case]['median_abs_pct']:.3f}% of notional)"
            for case in offending
        )
        reasons.append(
            f"systematic sign bias at medium grid within bucket(s): {detail}"
        )
    fine_pass = all(c.get("passed") is True for c in fine_cells)
    if not fine_pass:
        failing = [
            f"{c['date']}/{c['case']}" for c in fine_cells if c.get("passed") is not True
        ]
        reasons.append(f"fine grid fails on {len(failing)} cell(s): {', '.join(failing)}")
    drift_max = 0.0
    drift_where = None
    medium_by_key = {(c["date"], c["case"]): c for c in medium_cells}
    for f in fine_cells:
        m = medium_by_key.get((f["date"], f["case"]))
        if m is None or f.get("pde_price") is None or m.get("pde_price") is None:
            continue
        drift = abs(float(f["pde_price"]) - float(m["pde_price"])) / float(m["notional"]) * 100.0
        if drift > drift_max:
            drift_max, drift_where = drift, f"{f['date']}/{f['case']}"
    drift_ok = drift_max <= tol_abs_pct
    if not drift_ok:
        reasons.append(
            f"medium->fine drift {drift_max:.3f}% of notional at {drift_where} "
            f"exceeds TOL_ABS {tol_abs_pct:.3f}%"
        )
    if not delta_pass:
        if not delta_rows:
            reasons.append(
                "no delta evidence for this variant - refusing to admit PDE on "
                "empty evidence"
            )
        else:
            failing = [
                f"{r['date']}/{r.get('case', '?')}" for r in delta_rows if r.get("passed") is not True
            ]
            reasons.append(
                f"delta disagreement exceeds half a futures contract on "
                f"{len(failing)} cell(s): {', '.join(failing)}"
            )
    if delta_biased:
        reasons.append(
            f"systematic one-sided delta bias: mean "
            f"{delta_info['mean_signed_contracts']:+.3f} contracts exceeds the "
            f"{delta_info['bound_contracts']} contract bound"
        )
    route = "pde" if (
        medium_pass and not biased and fine_pass and drift_ok
        and delta_pass and not delta_biased
    ) else "mc"
    if route == "pde":
        rationale = (
            "medium grid inside tolerance on all sampled dates/cases, no sign "
            "bias, fine grid confirms (pass + no drift), and delta agrees "
            "with the reference within half a futures contract with no "
            "systematic bias"
        )
    else:
        rationale = "; ".join(reasons) if reasons else "route mc by default"
    return {
        "route": route,
        "medium_pass": bool(medium_pass),
        "fine_pass": bool(fine_pass),
        "biased": bool(biased),
        "bias": bias_info,
        "drift_max_pct": drift_max,
        "drift_location": drift_where,
        "delta_pass": bool(delta_pass),
        "delta_biased": bool(delta_biased),
        "delta_info": delta_info,
        "feller_buckets": feller_buckets,
        "rationale": rationale,
    }


def ladder_summary(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-level pass counts and diff statistics for honest ladder reporting."""
    out: Dict[str, Any] = {}
    for level in LEVELS:
        rows = [c for c in cells if c.get("level") == level]
        if not rows:
            continue
        diffs = [abs(float(c["signed_diff_pct"])) for c in rows if c.get("signed_diff_pct") is not None]
        signed = [float(c["signed_diff_pct"]) for c in rows if c.get("signed_diff_pct") is not None]
        out[level] = {
            "n_cells": len(rows),
            "n_passed": sum(1 for c in rows if c.get("passed") is True),
            "n_errors": sum(1 for c in rows if c.get("error")),
            "max_abs_diff_pct": max(diffs) if diffs else None,
            "mean_signed_diff_pct": float(np.mean(signed)) if signed else None,
        }
    return out


# ---------------------------------------------------------------------------
# Deterministic hashing / atomic writes / schemas
# ---------------------------------------------------------------------------

def strip_timings(obj: Any) -> Any:
    """Recursively drop every 'timings' subtree (non-deterministic fields)."""
    if isinstance(obj, dict):
        return {k: strip_timings(v) for k, v in obj.items() if k != "timings"}
    if isinstance(obj, list):
        return [strip_timings(v) for v in obj]
    return obj


def canonical_sha256(payload: Any) -> str:
    """sha256 of the canonical JSON of ``payload`` with timings stripped."""
    blob = json.dumps(
        strip_timings(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_CELL_REQUIRED = (
    "date", "case", "variant", "level", "grid", "pde_price", "mc_ref", "mc_se",
    "notional", "signed_diff_pct", "tol_pct", "passed",
)


def validate_gate_payload(payload: Dict[str, Any]) -> None:
    """Fail-closed schema check on pde_convergence_gate.json before writing."""
    for key in ("schema_version", "study", "config", "dates", "cells", "sanity"):
        if key not in payload:
            raise ValueError(f"gate payload missing key {key!r}")
    if payload["study"] != "pde_convergence_gate":
        raise ValueError("gate payload study tag mismatch")
    for cell in payload["cells"]:
        missing = [k for k in _CELL_REQUIRED if k not in cell]
        if missing:
            raise ValueError(f"gate cell missing keys {missing}")
        if cell["variant"] not in VARIANTS:
            raise ValueError(f"gate cell with unknown variant {cell['variant']!r}")
        if cell["level"] not in LEVELS:
            raise ValueError(f"gate cell with unknown level {cell['level']!r}")
        if cell["case"] not in CASES:
            raise ValueError(f"gate cell with unknown case {cell['case']!r}")
        if cell["error"] is None:
            required_finite = ["pde_price", "mc_ref", "signed_diff_pct"]
            # A deterministic reference (QUAD / finer PDE) has no std error by
            # construction -- only MC-referenced variants must report a finite one.
            if GATE_PAIRS[cell["variant"]].reference_is_mc:
                required_finite.append("mc_se")
            for key in required_finite:
                if cell[key] is None or not math.isfinite(float(cell[key])):
                    raise ValueError(f"gate cell {key} not finite without an error tag")


def validate_decision_payload(payload: Dict[str, Any]) -> None:
    for key in ("schema_version", "study", "variants", "tolerance", "evidence_sha256"):
        if key not in payload:
            raise ValueError(f"decision payload missing key {key!r}")
    if set(payload["variants"]) != set(VARIANTS):
        raise ValueError("decision payload must cover exactly the study variants")
    for variant, row in payload["variants"].items():
        if row["route"] not in ("pde", "mc"):
            raise ValueError(f"variant {variant}: route must be 'pde' or 'mc'")
        params_key = "pde_params" if row["route"] == "pde" else "mc_params"
        if params_key not in row:
            raise ValueError(f"variant {variant}: route params {params_key!r} missing")
        if not row.get("rationale"):
            raise ValueError(f"variant {variant}: rationale missing")


# ---------------------------------------------------------------------------
# Per-date study work (worker entry point)
# ---------------------------------------------------------------------------

def _bumped_pde_delta(engine, product, env, bump: float = SPOT_BUMP) -> float:
    """Central bumped delta (same formula as BaseEngine.calculate_greeks).

    Works for any DETERMINISTIC engine (1D PDE, 2D-ADI PDE, QUAD): built
    once by the caller, priced twice here.  Used for both the production
    engine and a deterministic reference (spec §5.3) -- a deterministic
    reference has no CRN to arrange, so it reuses this same central bump.
    """
    s0 = float(env.spot)
    env_up = deepcopy(env)
    env_up.spot_quote.spot = s0 * (1.0 + bump)
    env_dn = deepcopy(env)
    env_dn.spot_quote.spot = s0 * (1.0 - bump)
    return (float(engine.price(product, env_up)) - float(engine.price(product, env_dn))) / (
        2.0 * s0 * bump
    )


def _bumped_mc_delta(build_reference, model, product, env, bump: float = SPOT_BUMP) -> float:
    """Central bumped MC delta with common random numbers (same fixed seed).

    ``build_reference`` is ``GATE_PAIRS[variant].build_reference`` -- called
    here with ``grid=None`` (the reference is never laddered), driving this
    off the pair table instead of assuming a Heston/Heston-SLV engine.  Each
    reference builder bakes in the fixed module-level SEED, so rebuilding it
    fresh for each bump reproduces identical random draws (CRN) with no
    extra seed plumbing.
    """
    s0 = float(env.spot)
    env_up = deepcopy(env)
    env_up.spot_quote.spot = s0 * (1.0 + bump)
    env_dn = deepcopy(env)
    env_dn.spot_quote.spot = s0 * (1.0 - bump)
    eng_up = build_reference(model, None)
    eng_dn = build_reference(model, None)
    return (float(eng_up.price(product, env_up)) - float(eng_dn.price(product, env_dn))) / (
        2.0 * s0 * bump
    )


def _evaluate_case(
    *,
    date_iso: str,
    case_name: str,
    terms: SnowballTerms,
    s0_inception: float,
    envs: Dict[str, PricingEnvironment],
    models: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """MC references + PDE ladder + (full case) deltas for one date/case."""
    product = build_snowball_product(terms, s0_inception)
    notional = float(s0_inception)  # initial_price x contract_multiplier(=1)
    quick = bool(cfg["quick"])

    mc_refs: Dict[str, Any] = {}
    cells: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        pair = GATE_PAIRS[variant]
        model = models[variant]
        # Each variant prices off ITS OWN surface_vol_mode (spec §5.5) -- see
        # build_pricing_env; envs is keyed by mode, built once per date.
        env = envs[pair.surface_vol_mode]
        # A broken/non-finite reference must not kill the run: the variant's
        # cells are recorded as error cells (which count as gate failures).
        ref_err: Optional[str] = None
        ref_price: Optional[float] = None
        ref_se: Optional[float] = None
        ref_seconds: Optional[float] = None
        num_paths: Optional[int] = None
        batches_used: Optional[int] = None
        try:
            t0 = time.perf_counter()
            # Reference is priced once at its own fixed, finer configuration
            # (grid=None): it is never laddered, unlike the production engine.
            ref_engine = pair.build_reference(model, None)
            raw_price = float(ref_engine.price(product, env))
            ref_seconds = time.perf_counter() - t0
            if pair.reference_is_mc:
                result = ref_engine.get_last_result()
                raw_se = ref_engine.get_last_std_error()
                raw_se = float(raw_se) if raw_se is not None else float("nan")
                num_paths = int(result.num_paths)
                batches_used = int(result.batches_used)
                ref_err = _finite_or_error(raw_price, "reference price") or _finite_or_error(
                    raw_se, "reference std error"
                )
                if ref_err is None:
                    ref_price, ref_se = raw_price, raw_se
            else:
                ref_err = _finite_or_error(raw_price, "reference price")
                if ref_err is None:
                    ref_price = raw_price
        except Exception as exc:
            ref_err = f"{type(exc).__name__}: {exc}"
        mc_refs[variant] = {
            "price": ref_price,
            "std_error": ref_se,
            "num_paths": num_paths,
            "batches_used": batches_used,
            "price_pct_notional": (100.0 * ref_price / notional) if ref_price is not None else None,
            "se_pct_notional": (100.0 * ref_se / notional) if ref_se is not None else None,
            "error": ref_err,
            "timings": {"seconds": ref_seconds},
        }
        ref_se_pct = (100.0 * ref_se / notional) if ref_se is not None else None

        for level in LEVELS:
            grid = _production_grid(variant, level, terms.maturity_years, quick)
            # Event/step-key collisions are only meaningful where the gate
            # itself picks n_t (the 2D-ADI ladder); QUAD and the 1D PDE derive
            # their own internal time-stepping from grid_points/accuracy.
            n_t = grid.get("n_t")
            if n_t is not None:
                ki_collisions = event_key_collision_count(terms.ki_times, terms.maturity_years, n_t)
                ko_collisions = event_key_collision_count(terms.ko_times, terms.maturity_years, n_t)
            else:
                ki_collisions = ko_collisions = None
            cell = {
                "date": date_iso,
                "case": case_name,
                "variant": variant,
                "level": level,
                "grid": grid,
                "pde_price": None,
                "mc_ref": ref_price,
                "mc_se": ref_se,
                "mc_paths": num_paths,
                "notional": notional,
                "signed_diff_pct": None,
                "abs_diff_pct": None,
                "rel_diff_pct": None,
                # A deterministic reference's tolerance is the floor whether
                # or not it errored (TOL_ABS_PCT doesn't depend on it running).
                # A broken MC reference leaves the true tolerance unknown --
                # None, not a floor value that would misreport confidence.
                "tol_pct": (
                    gate_tolerance_pct(ref_se_pct, cfg["tol_abs_pct"])
                    if (not pair.reference_is_mc) or ref_se_pct is not None
                    else None
                ),
                "passed": False,
                "ki_event_key_collisions": ki_collisions,
                "ko_event_key_collisions": ko_collisions,
                "error": ref_err,  # broken reference: every cell fails closed
                "timings": {"pde_seconds": None, "mc_seconds": ref_seconds},
            }
            try:
                solver = pair.build_production(model, grid)
                t1 = time.perf_counter()
                raw_pde = float(solver.price(product, env))
                pde_seconds = time.perf_counter() - t1
                cell["timings"]["pde_seconds"] = pde_seconds
                pde_err = _finite_or_error(raw_pde, "production price")
                if pde_err is not None:
                    cell["error"] = (
                        pde_err if cell["error"] is None else f"{cell['error']}; {pde_err}"
                    )
                elif ref_err is None:
                    pde_price = raw_pde
                    signed = 100.0 * (pde_price - ref_price) / notional
                    cell.update(
                        {
                            "pde_price": pde_price,
                            "signed_diff_pct": signed,
                            "abs_diff_pct": abs(signed),
                            "rel_diff_pct": (
                                100.0 * abs(pde_price - ref_price) / abs(ref_price)
                                if ref_price
                                else None
                            ),
                            "passed": gate_cell_passed(signed, ref_se_pct, cfg["tol_abs_pct"]),
                        }
                    )
                else:
                    # Broken reference, finite production price: keep it as evidence.
                    cell["pde_price"] = raw_pde
            except Exception as exc:  # production failure is evidence against admission
                pde_msg = f"{type(exc).__name__}: {exc}"
                cell["error"] = (
                    pde_msg if cell["error"] is None else f"{cell['error']}; {pde_msg}"
                )
            cells.append(cell)

        # Gate G2 delta admission (spec §5.3): production engine bumped
        # centrally against the reference, built from GATE_PAIRS[variant]
        # the same way the PV cells above are -- not special-cased to any
        # one engine family.  Every variant carries this now; it drives
        # decide_route, it is not a report footnote.
        if case_name == "full":
            medium_grid = _production_grid(variant, "medium", terms.maturity_years, quick)
            delta_row = {
                "date": date_iso,
                "case": case_name,
                "variant": variant,
                "level": "medium",
                "s0": float(s0_inception),
                "delta_production": None,
                "delta_reference": None,
                "signed_diff": None,
                "abs_diff": None,
                "diff_contracts": None,
                "passed": False,        # fail closed until proven otherwise
                "error": None,
                "timings": {},
            }
            try:
                t1 = time.perf_counter()
                prod_engine = pair.build_production(model, medium_grid)
                delta_row["delta_production"] = _bumped_pde_delta(prod_engine, product, env)
                delta_row["timings"]["production_delta_seconds"] = time.perf_counter() - t1
            except Exception as exc:
                delta_row["error"] = f"production delta: {type(exc).__name__}: {exc}"
            try:
                t1 = time.perf_counter()
                if pair.reference_is_mc:
                    delta_row["delta_reference"] = _bumped_mc_delta(
                        pair.build_reference, model, product, env
                    )
                else:
                    ref_engine = pair.build_reference(model, None)
                    delta_row["delta_reference"] = _bumped_pde_delta(ref_engine, product, env)
                delta_row["timings"]["reference_delta_seconds"] = time.perf_counter() - t1
            except Exception as exc:
                delta_row["error"] = (
                    f"{delta_row['error']}; reference delta: {type(exc).__name__}: {exc}"
                    if delta_row["error"]
                    else f"reference delta: {type(exc).__name__}: {exc}"
                )
            for key in ("delta_production", "delta_reference"):
                err = _finite_or_error(delta_row[key], key)
                if err is not None:
                    delta_row["error"] = (
                        f"{delta_row['error']}; {err}" if delta_row["error"] else err
                    )
                    delta_row[key] = None
            if delta_row["delta_production"] is not None and delta_row["delta_reference"] is not None:
                signed = delta_row["delta_production"] - delta_row["delta_reference"]
                delta_row["signed_diff"] = signed
                delta_row["abs_diff"] = abs(signed)
                delta_row["diff_contracts"] = float(
                    safe_divide(signed, delta_quantum_per_unit(float(s0_inception)))
                )
                delta_row["passed"] = delta_cell_passed(abs(signed), float(s0_inception))
            else:
                delta_row["signed_diff"] = None
                delta_row["diff_contracts"] = None
                delta_row["passed"] = False        # fail closed
            deltas.append(delta_row)

    return {"mc_refs": mc_refs, "cells": cells, "deltas": deltas}


def _extras_first_date(
    *,
    date_iso: str,
    terms: SnowballTerms,
    s0_inception: float,
    envs: Dict[str, PricingEnvironment],
    artifact: IvSurfaceArtifact,
    models: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """First-date extras: substeps sensitivity + BSM-flat LV==BSM identity."""
    product = build_snowball_product(terms, s0_inception)
    notional = float(s0_inception)
    mcp = _make_mc_params(cfg["mc"], cfg["seed"])

    # (a) substeps_per_interval sensitivity.  The MC time grid already contains
    # every contractual (daily KI) observation date, so there is no discrete-
    # monitoring bias to remove; this documents the residual SDE-step
    # sensitivity of the QE discretization at one business day spacing.
    substeps_rows = []
    for variant in VARIANTS:
        # QE substeps_per_interval is a Heston/Heston-SLV discretization knob;
        # _make_mc_engine only builds those two engines, so (unlike the Gate
        # G2 delta check, which now runs for all six variants) this stays
        # scoped to the 2D-ADI pair.
        if not GATE_PAIRS[variant].production.startswith("pde_2d"):
            continue
        env = envs[GATE_PAIRS[variant].surface_vol_mode]  # both are full_grid
        row = {
            "date": date_iso,
            "case": "full",
            "variant": variant,
            "base_substeps": 1,
            "refined_substeps": 2,
            "base_price": None,
            "refined_price": None,
            "shift_pct_notional": None,
            "shift_over_se": None,
            "error": None,
        }
        try:
            base_engine = _make_mc_engine(variant, models[variant], mcp, 1)
            base = float(base_engine.price(product, env))
            base_se = base_engine.get_last_std_error()
            base_se = float(base_se) if base_se is not None else float("nan")
            ref_engine = _make_mc_engine(variant, models[variant], mcp, 2)
            refined = float(ref_engine.price(product, env))
            err = (
                _finite_or_error(base, "substeps base price")
                or _finite_or_error(refined, "substeps refined price")
                or _finite_or_error(base_se, "substeps base std error")
            )
            if err is not None:
                row["error"] = err
            else:
                row.update(
                    {
                        "base_price": base,
                        "refined_price": refined,
                        "shift_pct_notional": 100.0 * (refined - base) / notional,
                        "shift_over_se": abs(refined - base) / max(base_se, 1e-12),
                    }
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        substeps_rows.append(row)

    # (b) BSM-flat identity: a flat IV surface at the artifact's longest ATM
    # pillar must price identically under BSM MC and Dupire-LV MC (LV == flat).
    atm = max(artifact.atm_pillars, key=lambda p: p["T"])
    flat_vol = float(atm["atm_vol"])
    flat_surface = GridVolSurface(
        list(artifact.strikes),
        list(artifact.maturities),
        np.full((len(artifact.maturities), len(artifact.strikes)), flat_vol),
    )
    flat_env = PricingEnvironment(
        rate_curve=FlatRateCurve(rate=FLAT_RATE),
        # valuation_date is mode-independent (every envs[mode] is built off
        # the same artifact.trade_date); full_grid always exists in envs.
        valuation_date=envs["full_grid"].valuation_date,
        spot_quote=SpotQuote(spot=float(artifact.s0)),
        vol_surface=flat_surface,
        div_yield=artifact.term_structure_dividend_yield(FLAT_RATE),
    )
    lv_flat = build_dupire_local_vol(
        flat_surface,
        spot=float(artifact.s0),
        rate_curve=flat_env.rate_curve,
        div_yield=flat_env.get_div_yield,
    )
    mcp_id = _make_mc_params(
        {**cfg["mc"], "batches": max(2, cfg["mc"]["batches"] // 2)}, cfg["seed"] + 7
    )
    identity = {
        "date": date_iso,
        "flat_vol": flat_vol,
        "bsm_price": None,
        "lv_price": None,
        "diff_pct_notional": None,
        "combined_se_pct_notional": None,
        "consistent_within_2se": None,
        "error": None,
    }
    try:
        bsm_engine = SnowballMCEngine(params=mcp_id, method=MonteCarloMethod.RANDOMIZED_QUASI)
        bsm_price = float(bsm_engine.price(product, flat_env))
        bsm_se = bsm_engine.get_last_std_error()
        bsm_se = float(bsm_se) if bsm_se is not None else float("nan")
        lv_engine = LocalVolSnowballMCEngine(
            params=mcp_id,
            method=MonteCarloMethod.RANDOMIZED_QUASI,
            local_vol_surface=lv_flat,
        )
        lv_price = float(lv_engine.price(product, flat_env))
        lv_se = lv_engine.get_last_std_error()
        lv_se = float(lv_se) if lv_se is not None else float("nan")
        err = (
            _finite_or_error(bsm_price, "bsm identity price")
            or _finite_or_error(lv_price, "lv identity price")
            or _finite_or_error(bsm_se, "bsm identity std error")
            or _finite_or_error(lv_se, "lv identity std error")
        )
        if err is not None:
            identity["error"] = err
        else:
            se_comb = math.sqrt(bsm_se**2 + lv_se**2)
            identity.update(
                {
                    "bsm_price": bsm_price,
                    "lv_price": lv_price,
                    "diff_pct_notional": 100.0 * (lv_price - bsm_price) / notional,
                    "combined_se_pct_notional": 100.0 * se_comb / notional,
                    "consistent_within_2se": abs(lv_price - bsm_price) <= 2.0 * se_comb,
                }
            )
    except Exception as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
    return {"substeps_sensitivity": substeps_rows, "bsm_lv_identity": identity}


def _calibrate_models(calibrator: VolModelCalibrator, artifact: IvSurfaceArtifact) -> Dict[str, Any]:
    """One CalibratedVolModel per variant that needs one; ``None`` for BSM."""
    return {
        variant: (calibrator.calibrate(variant, artifact) if variant in _CALIBRATED_VARIANTS else None)
        for variant in VARIANTS
    }


def _calibration_record(model) -> Optional[Dict[str, Any]]:
    """Deterministic calibration record: drop volatile cache/timing fields.

    ``None`` for the BSM variants, which skip calibration entirely -- there
    is no record to report, not an empty one.
    """
    if model is None:
        return None
    volatile = {"cache_hit", "calibration_seconds", "per_expiry_rmse"}
    return {k: v for k, v in model.record.items() if k not in volatile}


_FELLER_TRACKED_VARIANTS = (VOL_MODEL_HESTON, VOL_MODEL_HESTON_SLV)


def _attach_feller_ratio(case: Dict[str, Any], models: Dict[str, Any]) -> None:
    """Copy the Heston-fit Feller ratio onto every heston/heston_slv cell and
    delta row (spec §7A.4/§7A.11), keyed off the SAME record
    ``_calibration_record`` reports -- no separate computation, so no
    separate safe_divide (the ratio is already produced by safe_divide in
    quantark.volmodels.calibration).  heston_slv shares the ratio because it
    is built ON TOP OF the calibrated Heston params, never refit on its own;
    every other variant, and a missing/uncalibrated Heston model, is
    explicitly None ("unknown" once bucketed) -- never silently omitted.
    """
    heston_model = models.get(VOL_MODEL_HESTON)
    ratio = heston_model.record.get("feller_ratio") if heston_model is not None else None
    for row in (*case["cells"], *case["deltas"]):
        row["feller_ratio"] = ratio if row["variant"] in _FELLER_TRACKED_VARIANTS else None


def process_date(cfg: Dict[str, Any], date_info: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: full study for one inception date.  Returns a JSON-safe record."""
    inception = _parse_yyyymmdd(date_info["date"])
    history = VolSurfaceHistory(cfg["history_dir"])
    artifact = history.surface_for(inception)  # inception is admitted by construction
    calendar = TradingCalendar.from_spot_csv(
        Path(cfg["history_dir"]) / cfg.get("spot_csv", "csi1000_spot.csv")
    )
    calibrator = make_calibrator(Path(cfg["out_dir"]), bool(cfg["quick"]))

    terms = build_snowball_terms(inception, calendar)
    envs = _envs_by_mode(artifact, cfg["flat_rate"], terms.maturity_years)
    models = _calibrate_models(calibrator, artifact)

    record: Dict[str, Any] = {
        "requested": date_info["requested"],
        "date": inception.isoformat(),
        "substituted": bool(date_info["substituted"]),
        "s0": float(artifact.s0),
        "surface_sha256": artifact.sha256,
        "max_listed_T": float(artifact.max_listed_T),
        "calendar_extension": terms.maturity_date > calendar.last,
        "terms": {"full": terms.summary()},
        "calibration": {
            variant: _calibration_record(models[variant]) for variant in VARIANTS
        },
        "cases": {},
        "extras": None,
        "timings": {},
    }

    t_case = time.perf_counter()
    record["cases"]["full"] = _evaluate_case(
        date_iso=inception.isoformat(),
        case_name="full",
        terms=terms,
        s0_inception=float(artifact.s0),
        envs=envs,
        models=models,
        cfg=cfg,
    )
    record["timings"]["full_seconds"] = time.perf_counter() - t_case
    _attach_feller_ratio(record["cases"]["full"], models)

    decay_date = pick_decay_date(history.admitted_dates, terms)
    if decay_date is None:
        record["terms"]["decayed"] = None
        record["cases"]["decayed"] = None
    else:
        decay_artifact = history.surface_for(decay_date)
        dterms = decay_terms(terms, decay_date)
        # decayed terms' OWN maturity_years (already t_shift-adjusted by
        # decay_terms), not a recomputed subtraction at this call site.
        denvs = _envs_by_mode(decay_artifact, cfg["flat_rate"], dterms.maturity_years)
        dmodels = _calibrate_models(calibrator, decay_artifact)
        record["terms"]["decayed"] = dterms.summary()
        record["decayed_calibration"] = {
            variant: _calibration_record(dmodels[variant]) for variant in VARIANTS
        }
        t_case = time.perf_counter()
        record["cases"]["decayed"] = _evaluate_case(
            date_iso=inception.isoformat(),
            case_name="decayed",
            terms=dterms,
            s0_inception=float(artifact.s0),  # contract notional fixed at inception
            envs=denvs,
            models=dmodels,
            cfg=cfg,
        )
        record["timings"]["decayed_seconds"] = time.perf_counter() - t_case
        _attach_feller_ratio(record["cases"]["decayed"], dmodels)

    if cfg.get("extras_date") == inception.isoformat():
        record["extras"] = _extras_first_date(
            date_iso=inception.isoformat(),
            terms=terms,
            s0_inception=float(artifact.s0),
            envs=envs,
            artifact=artifact,
            models=models,
            cfg=cfg,
        )
    return record


# ---------------------------------------------------------------------------
# Assembly / decision / report
# ---------------------------------------------------------------------------

def assemble_gate_payload(cfg: Dict[str, Any], date_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    for rec in date_records:
        for case_name in CASES:
            case = rec["cases"].get(case_name)
            if case is None:
                continue
            cells.extend(case["cells"])
            deltas.extend(case["deltas"])

    sanity_mc: List[Dict[str, Any]] = []
    for rec in date_records:
        for case_name in CASES:
            case = rec["cases"].get(case_name)
            if case is None:
                continue
            heston = case["mc_refs"][VOL_MODEL_HESTON]
            slv = case["mc_refs"][VOL_MODEL_HESTON_SLV]
            diff_pct = None
            if heston.get("price") is not None and slv.get("price") is not None:
                diff_pct = 100.0 * (slv["price"] - heston["price"]) / float(rec["s0"])
            sanity_mc.append(
                {
                    "date": rec["date"],
                    "case": case_name,
                    "diff_pct_notional": diff_pct,
                    "heston_fit_rmse_iv": rec["calibration"][VOL_MODEL_HESTON].get(
                        "overall_rmse_iv"
                    ),
                    "note": "Heston vs SLV MC differ only through the model fit "
                    "(SLV pins the smile; Heston approximates it) - a large gap "
                    "indicates calibration error, not engine error.",
                }
            )

    extras = next((rec["extras"] for rec in date_records if rec.get("extras")), None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "pde_convergence_gate",
        "config": {
            "history_dir": cfg["history_dir"],
            "quick": bool(cfg["quick"]),
            "flat_rate": cfg["flat_rate"],
            "seed": cfg["seed"],
            "mc": cfg["mc"],
            "tol_abs_pct": cfg["tol_abs_pct"],
            "mc_se_factor": MC_SE_FACTOR,
            "bias": {
                "sign_fraction": BIAS_SIGN_FRACTION,
                "median_fraction_of_tol": BIAS_MEDIAN_FRACTION,
            },
            "pde_ladder_rule": (
                "2D-ADI variants (heston/heston_slv): coarse=(90,36,round(32T)); "
                "medium=(200,60,ceil(400T)) - dt <= 1/400y so daily KI events cannot "
                "share an ADI step; fine=(300,90,2x medium). QUAD variants ladder over "
                "grid_points (quad_ladder); 1D-PDE variants ladder over accuracy "
                "(pde1d_ladder)."
            ),
            "quad_ladder": dict(QUAD_LADDER),
            "quad_reference_points": QUAD_REFERENCE_POINTS,
            "pde1d_ladder": dict(PDE1D_LADDER),
            "term_sheet": {
                "maturity_months": MATURITY_MONTHS,
                "lockout_months": LOCKOUT_MONTHS,
                "ko_pct": KO_PCT,
                "ki_pct": KI_PCT,
                "ko_rate": KO_RATE,
                "include_principal": False,
                "ki_monitoring": "daily-discrete (trading calendar, ACT/365)",
                "ko_monitoring": "monthly from month 3 (trading-calendar snapped)",
            },
        },
        "dates": date_records,
        "cells": cells,
        "deltas": deltas,
        "sanity": {
            "heston_vs_slv_mc": sanity_mc,
            "substeps_sensitivity": (extras or {}).get("substeps_sensitivity"),
            "bsm_lv_identity": (extras or {}).get("bsm_lv_identity"),
        },
        "timings": {
            "per_date_seconds": {
                rec["date"]: sum(
                    v for v in rec.get("timings", {}).values() if isinstance(v, (int, float))
                )
                for rec in date_records
            }
        },
    }
    validate_gate_payload(payload)
    return payload


_MC_ENGINE_NAME = {
    VOL_MODEL_HESTON: "QESnowballMCEngine",
    VOL_MODEL_HESTON_SLV: "HestonSLVQESnowballMCEngine",
    VOL_MODEL_LOCALVOL: "LocalVolSnowballMCEngine",
}


def _production_params_block(pair: GatePair, medium_grid: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Route-agnostic production-engine params for the decision payload.

    Only the 2D-ADI family maps onto ``AutocallableEngineConfig.vol_model_engine_options``
    (n_x/n_v/n_t); QUAD and 1D-PDE variants record their own ladder knob honestly
    instead of forcing them into that grid shape.
    """
    if medium_grid is not None and medium_grid.get("kind") == "adi_2d":
        return {
            "kind": "adi_2d",
            "label": pair.production,
            "n_x": medium_grid["n_x"],
            "n_v": medium_grid["n_v"],
            "n_t": medium_grid["n_t"],
            "n_t_rule": "ceil(400 * remaining_maturity_years)",
            "scheme": "craig_sneyd",
            "grid_style": "concentrated",
            "grid_focus": "auto",
            "event_projection": "cell_average",
            "event_rannacher_steps": 2,
            "theta": 0.5,
            "note": "maps to AutocallableEngineConfig.vol_model_engine_options "
            "(n_x/n_v/n_t) + PDEParams defaults; n_t is per-case: recompute as "
            "ceil(400 * remaining maturity in years) at each reprice - the "
            "emitted n_t is the full-3Y inception value",
        }
    return {"label": pair.production, **(medium_grid or {"kind": None})}


def _reference_params_block(pair: GatePair, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Route-agnostic reference-engine params for the decision payload."""
    if not pair.reference_is_mc:
        return {
            "kind": "deterministic",
            "label": pair.reference,
            "note": "no MC standard error; tolerance is the flat TOL_ABS floor",
        }
    params: Dict[str, Any] = {
        "kind": "mc",
        "label": pair.reference,
        "method": "randomized_quasi",
        "seed": cfg["seed"],
        "paths_per_batch": cfg["mc"]["paths_per_batch"],
        "batches": cfg["mc"]["batches"],
        "rqmc_min_batches": cfg["mc"]["batches"],
        "rqmc_max_batches": cfg["mc"]["batches"],
        "rqmc_target_std": 1e-12,
        "note": "REFERENCE-quality config used for the gate: RQMC pinned "
        "(rqmc_min_batches == rqmc_max_batches, rqmc_target_std ~ 0) so "
        "every batch runs and the batch-spread SE is honest. The "
        "production backtest runner (Phase 4) may reduce path counts "
        "under a documented SE budget, recorded in the run manifest.",
    }
    return params


def build_decision_payload(cfg: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    cells = gate["cells"]
    variants: Dict[str, Any] = {}
    for variant in VARIANTS:
        pair = GATE_PAIRS[variant]
        v_cells = [c for c in cells if c["variant"] == variant]
        medium = [c for c in v_cells if c["level"] == "medium"]
        fine = [c for c in v_cells if c["level"] == "fine"]
        v_deltas = [d for d in gate.get("deltas", []) if d["variant"] == variant]
        decision = decide_route(medium, fine, v_deltas, cfg["tol_abs_pct"])
        medium_grid = next(
            (c["grid"] for c in medium if c["case"] == "full"),
            medium[0]["grid"] if medium else None,
        )
        mc_params = _reference_params_block(pair, cfg)
        if pair.reference_is_mc:
            mc_params["engine"] = _MC_ENGINE_NAME.get(variant, pair.reference)
            if variant in (VOL_MODEL_HESTON, VOL_MODEL_HESTON_SLV):
                mc_params.update(
                    {
                        "substeps_per_interval": cfg["mc"]["substeps_per_interval"],
                        "scheme": "QUADEXP_M" if MC_MARTINGALE else "QUADEXP",
                        "martingale_correction": MC_MARTINGALE,
                    }
                )
        variants[variant] = {
            "route": decision["route"],
            "pde_params": _production_params_block(pair, medium_grid),
            "mc_params": mc_params,
            "ladder": ladder_summary(v_cells),
            "gate": {
                "medium_pass": decision["medium_pass"],
                "fine_pass": decision["fine_pass"],
                "biased": decision["biased"],
                "bias": decision["bias"],
                "drift_max_pct": decision["drift_max_pct"],
                "drift_location": decision["drift_location"],
                "delta_pass": decision["delta_pass"],
                "delta_biased": decision["delta_biased"],
                "delta_info": decision["delta_info"],
                "feller_buckets": decision["feller_buckets"],
            },
            "rationale": decision["rationale"],
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "pde_convergence_gate_decision",
        "quick_mode": bool(cfg["quick"]),
        "quick_mode_warning": (
            "quick mode uses smoke-test grids/paths; the route decision is NOT "
            "production-valid" if cfg["quick"] else None
        ),
        "tolerance": {
            "tol_abs_pct_notional": cfg["tol_abs_pct"],
            "mc_se_factor": MC_SE_FACTOR,
            "bias_sign_fraction": BIAS_SIGN_FRACTION,
            "bias_median_fraction_of_tol": BIAS_MEDIAN_FRACTION,
        },
        "mc_reference": {
            "method": "randomized_quasi",
            "scheme": "QUADEXP_M" if MC_MARTINGALE else "QUADEXP",
            "martingale_correction": MC_MARTINGALE,
            "seed": cfg["seed"],
            "paths_per_batch": cfg["mc"]["paths_per_batch"],
            "batches": cfg["mc"]["batches"],
            "total_paths": cfg["mc"]["paths_per_batch"] * cfg["mc"]["batches"],
            "substeps_per_interval": cfg["mc"]["substeps_per_interval"],
        },
        # Which calibration policy this evidence covers.  The replay config
        # gained heston_temporal_reference / heston_temporal_regularization /
        # slv_heston_override (spec §7A.12); they default off and the installed
        # scheduler does not enable them, so these gates DO cover production
        # today.  Recorded explicitly so a future reader cannot mistake lambda=0
        # evidence for coverage of lambda>0.
        "calibration_policy": {
            "heston_preset": "mo_frozen",
            "enforce_feller": True,
            "heston_temporal_regularization": 0.0,
            "slv_heston_override": None,
            "note": (
                "Independent daily calibration. This gate does NOT cover "
                "--temporal-smoothing (heston_temporal_regularization > 0) or "
                "an explicit slv_heston_override; enabling either is a re-gate "
                "trigger (spec §7A.12)."
            ),
        },
        "variants": variants,
        "evidence_file": "pde_convergence_gate.json",
        "evidence_sha256": canonical_sha256(gate),
    }
    validate_decision_payload(payload)
    return payload


def build_report(cfg: Dict[str, Any], gate: Dict[str, Any], decision: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# PDE convergence gate - Heston / Heston-SLV snowball")
    lines.append("")
    lines.append(
        "Trade: 3Y snowball on 000852.SH, KO 103% monthly from month 3 (34 obs), "
        "KI 75% daily-discrete (trading calendar), ko_rate 15%, principal excluded, "
        "r=2% flat, q from parity forward pillars."
    )
    lines.append(
        f"Reference: RQMC (scrambled Sobol) QE, {decision['mc_reference']['total_paths']} "
        f"paths ({decision['mc_reference']['batches']} batches x "
        f"{decision['mc_reference']['paths_per_batch']}), seed {decision['mc_reference']['seed']}. "
        f"Tolerance: max({MC_SE_FACTOR:g} x MC SE, {cfg['tol_abs_pct']}% of notional)."
    )
    lines.append("")
    if cfg["quick"]:
        lines.append("**QUICK MODE - smoke-test grids/paths; decisions are NOT production-valid.**")
        lines.append("")

    lines.append("## Route decision")
    lines.append("")
    lines.append(
        "| variant | route | medium pass | fine pass | bias | max drift % | "
        "delta pass | delta bias | rationale |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for variant, row in decision["variants"].items():
        g = row["gate"]
        lines.append(
            f"| {variant} | **{row['route']}** | {g['medium_pass']} | {g['fine_pass']} | "
            f"{g['biased']} | {g['drift_max_pct']:.3f} | {g['delta_pass']} | "
            f"{g['delta_biased']} | {row['rationale']} |"
        )
    lines.append("")

    lines.append("## Sample dates")
    lines.append("")
    lines.append("| requested | used | substituted | s0 | full case | decayed case |")
    lines.append("|---|---|---|---|---|---|")
    for rec in gate["dates"]:
        decayed = rec["terms"].get("decayed")
        decayed_txt = (
            f"{decayed['valuation']} (T={decayed['maturity_years']:.2f}y, "
            f"{decayed['n_ko']} KO/{decayed['n_ki']} KI)"
            if decayed
            else "skipped (no admitted date leaves 0.5-2.0y remaining)"
        )
        full = rec["terms"]["full"]
        lines.append(
            f"| {rec['requested']} | {rec['date']} | {rec['substituted']} | {rec['s0']:.2f} | "
            f"T={full['maturity_years']:.2f}y, {full['n_ko']} KO/{full['n_ki']} KI | {decayed_txt} |"
        )
    lines.append("")

    for variant in VARIANTS:
        row = decision["variants"][variant]
        lines.append(f"## {variant} ladder")
        lines.append("")
        lines.append("| level | grid (n_x,n_v,n_t@full) | cells | passed | errors | max abs diff % | mean signed diff % |")
        lines.append("|---|---|---|---|---|---|---|")
        grids = {
            c["level"]: c["grid"]
            for c in gate["cells"]
            if c["variant"] == variant and c["case"] == "full"
        }
        for level in LEVELS:
            summary = row["ladder"].get(level)
            if summary is None:
                continue
            grid = grids.get(level, {})
            max_diff = summary["max_abs_diff_pct"]
            mean_signed = summary["mean_signed_diff_pct"]
            max_txt = f"{max_diff:.4f}" if max_diff is not None else "n/a"
            mean_txt = f"{mean_signed:+.4f}" if mean_signed is not None else "n/a"
            lines.append(
                f"| {level} | ({grid.get('n_x')},{grid.get('n_v')},{grid.get('n_t')}) | "
                f"{summary['n_cells']} | {summary['n_passed']} | {summary['n_errors']} | "
                f"{max_txt} | {mean_txt} |"
            )
        lines.append("")
        lines.append("| date | case | MC ref % | SE % | medium PDE % | diff % (tol %) | pass |")
        lines.append("|---|---|---|---|---|---|---|")
        medium_cells = [c for c in gate["cells"] if c["variant"] == variant and c["level"] == "medium"]
        for c in medium_cells:
            mc_txt = (
                f"{100.0 * c['mc_ref'] / c['notional']:+.3f}"
                if c["mc_ref"] is not None
                else "n/a"
            )
            se_txt = (
                f"{100.0 * c['mc_se'] / c['notional']:.3f}"
                if c["mc_se"] is not None
                else "n/a"
            )
            if c["pde_price"] is None:
                pde_txt = f"ERROR: {c['error']}"
                diff_txt = "-"
            elif c["signed_diff_pct"] is None:
                pde_txt = f"{100.0 * c['pde_price'] / c['notional']:+.3f}"
                diff_txt = f"n/a ({c['error']})"
            else:
                pde_txt = f"{100.0 * c['pde_price'] / c['notional']:+.3f}"
                diff_txt = f"{c['signed_diff_pct']:+.3f} ({c['tol_pct']:.3f})"
            lines.append(
                f"| {c['date']} | {c['case']} | {mc_txt} | {se_txt} | "
                f"{pde_txt} | {diff_txt} | {c['passed']} |"
            )
        lines.append("")

    lines.append("## Delta agreement (Gate G2 criterion, spec §5.3)")
    lines.append("")
    deltas = gate.get("deltas") or []
    if deltas:
        lines.append(
            "| date | variant | delta production | delta reference | abs diff | "
            "diff (contracts) | pass |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for d in deltas:
            if d["delta_production"] is None or d["delta_reference"] is None:
                lines.append(
                    f"| {d['date']} | {d['variant']} | n/a ({d.get('error')}) | | | | "
                    f"{d['passed']} |"
                )
            else:
                lines.append(
                    f"| {d['date']} | {d['variant']} | {d['delta_production']:+.4f} | "
                    f"{d['delta_reference']:+.4f} | {d['abs_diff']:.4f} | "
                    f"{d['diff_contracts']:+.3f} | {d['passed']} |"
                )
        lines.append("")
    sanity = gate["sanity"]
    if sanity.get("substeps_sensitivity"):
        lines.append("substeps_per_interval sensitivity (first date, full case):")
        for row in sanity["substeps_sensitivity"]:
            if row.get("error") or row.get("shift_pct_notional") is None:
                lines.append(f"- {row['variant']}: n/a ({row.get('error') or 'no data'})")
            else:
                lines.append(
                    f"- {row['variant']}: substeps 1->2 shifts price by "
                    f"{row['shift_pct_notional']:+.4f}% of notional "
                    f"({row['shift_over_se']:.2f} x MC SE)"
                )
        lines.append("")
    identity = sanity.get("bsm_lv_identity")
    if identity:
        if identity.get("error") or identity.get("diff_pct_notional") is None:
            lines.append(
                f"BSM-flat LV==BSM identity ({identity['date']}): n/a "
                f"({identity.get('error') or 'no data'})."
            )
        else:
            lines.append(
                f"BSM-flat LV==BSM identity ({identity['date']}): diff "
                f"{identity['diff_pct_notional']:+.4f}% of notional vs combined SE "
                f"{identity['combined_se_pct_notional']:.4f}% "
                f"(consistent within 2 SE: {identity['consistent_within_2se']})."
            )
        lines.append("")
    lines.append(
        "Heston vs SLV MC sanity rows are in pde_convergence_gate.json "
        "(`sanity.heston_vs_slv_mc`); they differ by model fit, not by engine."
    )
    lines.append("")
    lines.append(f"Evidence sha256 (timings excluded): `{decision['evidence_sha256']}`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_study(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    history = VolSurfaceHistory(cfg["history_dir"])
    requested = [_parse_yyyymmdd(d) for d in cfg["dates"]]
    date_infos = resolve_sample_dates(history.admitted_dates, requested)
    if cfg["quick"]:
        date_infos = date_infos[:1]
    cfg = dict(cfg)
    cfg["extras_date"] = date_infos[0]["date"]

    t0 = time.perf_counter()
    workers = int(cfg.get("workers", 1))
    if workers > 1 and len(date_infos) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(process_date, [cfg] * len(date_infos), date_infos))
    else:
        records = [process_date(cfg, info) for info in date_infos]
    records.sort(key=lambda r: r["date"])

    gate = assemble_gate_payload(cfg, records)
    gate["timings"]["total_seconds"] = time.perf_counter() - t0
    decision = build_decision_payload(cfg, gate)
    report = build_report(cfg, gate, decision)
    return gate, decision, report


def _default_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    quick = bool(args.quick)
    return {
        "history_dir": str(Path(args.history_dir)),
        "out_dir": str(Path(args.out_dir)),
        "quick": quick,
        "workers": 1 if quick else int(args.workers),
        "dates": [d.strip() for d in args.dates.split(",") if d.strip()]
        if args.dates
        else list(DEFAULT_SAMPLE_DATES),
        "flat_rate": FLAT_RATE,
        "seed": SEED,
        "mc": dict(MC_QUICK if quick else MC_FULL),
        "tol_abs_pct": TOL_ABS_PCT,
        "spot_csv": "csi1000_spot.csv",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", default="example/mo_volmodels/data/history")
    parser.add_argument("--out-dir", default="output/pde_convergence_gate")
    parser.add_argument("--dates", default=None, help="comma-separated YYYYMMDD list")
    parser.add_argument("--quick", action="store_true", help="coarse everything, 1 date")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    cfg = _default_cfg(args)
    out_dir = Path(cfg["out_dir"])
    gate, decision, report = run_study(cfg)
    _atomic_write_json(out_dir / "pde_convergence_gate.json", gate)
    _atomic_write_json(out_dir / "gate_decision.json", decision)
    _atomic_write_text(out_dir / "gate_report.md", report)

    for variant, row in decision["variants"].items():
        print(f"[gate] {variant}: route={row['route']} - {row['rationale']}")
    print(f"[gate] wrote {out_dir/'pde_convergence_gate.json'}")
    print(f"[gate] wrote {out_dir/'gate_decision.json'}")
    print(f"[gate] wrote {out_dir/'gate_report.md'}")
    print(f"[gate] total seconds: {gate['timings']['total_seconds']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
