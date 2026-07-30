"""Stage 11 - PDE convergence gate for Heston / Heston-SLV snowball pricing.

Decides whether the production OTC backtest may route ``vol_model="heston"`` /
``"heston_slv"`` snowball pricing through the 2D-ADI PDE solvers
(``HestonSnowballPDESolver`` / ``HestonSLVSnowballPDESolver``) or must fall back
to the QE Monte-Carlo engines (``QESnowballMCEngine`` /
``HestonSLVQESnowballMCEngine``).  It prices the production 3Y snowball
(KO 103% monthly with a 3-month lockout -> 34 observations, KI 75% with
daily-discrete monitoring from the trading calendar, principal excluded,
ko_rate = rebate_rate = 15%, r = 2% flat, q from the artifact's parity forward
pillars) on a sample of admitted IV-surface dates spanning market regimes,
compares a PDE refinement ladder (coarse/medium/fine) against a high-quality
RQMC reference with reported standard errors, and emits:

- ``pde_convergence_gate.json`` : per date x case(full/decayed) x variant x
  level cell results (prices, diffs, tolerances, pass/fail, timings).
- ``gate_decision.json``        : per variant route ("pde"|"mc") + params +
  tolerance + sha256 evidence hash of the gate JSON (timings stripped).
- ``gate_report.md``            : readable summary tables + rationale.

Gate rule (per variant): PDE is admitted only if the MEDIUM grid passes
``|PDE - MC_ref| <= max(2 * MC_SE, TOL_ABS)`` (TOL_ABS = 0.25% of notional) on
EVERY sampled date/case, shows no systematic sign bias (a biased PDE fails even
inside tolerance), the FINE grid also passes, and |fine - medium| <= TOL_ABS
everywhere (no drift).  Any other outcome routes the variant to MC, with the
ladder reported honestly.  Deltas (PDE bumped vs MC CRN-bumped) are secondary
evidence, reported but not gated.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.backtest.otc.config import VolModelCalibrationConfig
from quantark.volmodels.calibration import (
    VOL_MODEL_HESTON,
    VOL_MODEL_HESTON_SLV,
    VolModelCalibrator,
)
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.volmodels.localvol import build_dupire_local_vol

# ---------------------------------------------------------------------------
# Frozen study configuration
# ---------------------------------------------------------------------------

VARIANTS = (VOL_MODEL_HESTON, VOL_MODEL_HESTON_SLV)
CASES = ("full", "decayed")
LEVELS = ("coarse", "medium", "fine")

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
MC_FULL = {"paths_per_batch": 8192, "batches": 16, "substeps_per_interval": 1}
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

def build_pricing_env(artifact: IvSurfaceArtifact, rate: float = FLAT_RATE) -> PricingEnvironment:
    """Production env: flat rate, dividend curve from the artifact's
    option-parity forward pillars."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=rate),
        valuation_date=datetime.combine(artifact.trade_date, datetime.min.time()),
        spot_quote=SpotQuote(spot=float(artifact.s0)),
        vol_surface=artifact.grid_vol_surface(),
        div_yield=artifact.term_structure_dividend_yield(rate),
    )


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
        )
    return HestonSLVQESnowballMCEngine(
        model.heston_params,
        params=mcp,
        method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=model.leverage_surface,
        eta=float(model.slv_eta if model.slv_eta is not None else 1.0),
        local_vol_surface=model.local_vol_surface,
        substeps_per_interval=substeps,
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

def gate_tolerance_pct(mc_se_pct: float, tol_abs_pct: float = TOL_ABS_PCT) -> float:
    """Per-cell tolerance in % of notional: max(2 x MC SE, TOL_ABS)."""
    return max(MC_SE_FACTOR * float(mc_se_pct), float(tol_abs_pct))


def gate_cell_passed(signed_diff_pct: float, mc_se_pct: float, tol_abs_pct: float = TOL_ABS_PCT) -> bool:
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
    tol_abs_pct: float = TOL_ABS_PCT,
) -> Dict[str, Any]:
    """Route decision for one variant from its medium/fine gate cells.

    PDE is admitted only if: every medium cell passes, no systematic bias at
    medium, every fine cell passes, and |fine - medium| <= TOL_ABS on every
    paired cell (no drift).  Any cell carrying an ``error`` counts as failed.
    Zero evaluable cells never admits PDE (``all([])`` is vacuously True).
    """
    if not medium_cells or not fine_cells:
        return {
            "route": "mc",
            "medium_pass": False,
            "fine_pass": False,
            "biased": False,
            "bias": {"n_cells": 0, "note": "empty ladder"},
            "drift_max_pct": 0.0,
            "drift_location": None,
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
    biased, bias_info = detect_systematic_bias(
        [c.get("signed_diff_pct") for c in medium_cells], tol_abs_pct
    )
    if biased:
        reasons.append(
            "systematic sign bias at medium grid "
            f"(sign fraction {bias_info['sign_fraction']:.2f}, "
            f"median |diff| {bias_info['median_abs_pct']:.3f}% of notional)"
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
    route = "pde" if (medium_pass and not biased and fine_pass and drift_ok) else "mc"
    if route == "pde":
        rationale = (
            "medium grid inside tolerance on all sampled dates/cases, no sign "
            "bias, fine grid confirms (pass + no drift)"
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
            for key in ("pde_price", "mc_ref", "mc_se", "signed_diff_pct"):
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
    """Central bumped delta (same formula as BaseEngine.calculate_greeks)."""
    s0 = float(env.spot)
    env_up = deepcopy(env)
    env_up.spot_quote.spot = s0 * (1.0 + bump)
    env_dn = deepcopy(env)
    env_dn.spot_quote.spot = s0 * (1.0 - bump)
    return (float(engine.price(product, env_up)) - float(engine.price(product, env_dn))) / (
        2.0 * s0 * bump
    )


def _bumped_mc_delta(variant, model, product, env, mcp, mc_cfg, bump: float = SPOT_BUMP) -> float:
    """Central bumped MC delta with common random numbers (same fixed seed)."""
    s0 = float(env.spot)
    env_up = deepcopy(env)
    env_up.spot_quote.spot = s0 * (1.0 + bump)
    env_dn = deepcopy(env)
    env_dn.spot_quote.spot = s0 * (1.0 - bump)
    eng_up = _make_mc_engine(variant, model, mcp, mc_cfg["substeps_per_interval"])
    eng_dn = _make_mc_engine(variant, model, mcp, mc_cfg["substeps_per_interval"])
    return (float(eng_up.price(product, env_up)) - float(eng_dn.price(product, env_dn))) / (
        2.0 * s0 * bump
    )


def _evaluate_case(
    *,
    date_iso: str,
    case_name: str,
    terms: SnowballTerms,
    s0_inception: float,
    env: PricingEnvironment,
    models: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """MC references + PDE ladder + (full case) deltas for one date/case."""
    product = build_snowball_product(terms, s0_inception)
    notional = float(s0_inception)  # initial_price x contract_multiplier(=1)
    quick = bool(cfg["quick"])
    mcp = _make_mc_params(cfg["mc"], cfg["seed"])
    pdep = PDEParams()  # remediated defaults: cell_average projection, 2 event-Rannacher steps
    ladder = pde_ladder(terms.maturity_years, quick)

    mc_refs: Dict[str, Any] = {}
    cells: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []

    for variant in VARIANTS:
        model = models[variant]
        # A broken/non-finite MC reference must not kill the run: the variant's
        # cells are recorded as error cells (which count as gate failures).
        mc_err: Optional[str] = None
        mc_price: Optional[float] = None
        mc_se: Optional[float] = None
        mc_seconds: Optional[float] = None
        num_paths: Optional[int] = None
        batches_used: Optional[int] = None
        try:
            t0 = time.perf_counter()
            engine = _make_mc_engine(variant, model, mcp, cfg["mc"]["substeps_per_interval"])
            raw_price = float(engine.price(product, env))
            mc_seconds = time.perf_counter() - t0
            result = engine.get_last_result()
            raw_se = engine.get_last_std_error()
            raw_se = float(raw_se) if raw_se is not None else float("nan")
            num_paths = int(result.num_paths)
            batches_used = int(result.batches_used)
            mc_err = _finite_or_error(raw_price, "mc reference price") or _finite_or_error(
                raw_se, "mc reference std error"
            )
            if mc_err is None:
                mc_price, mc_se = raw_price, raw_se
        except Exception as exc:
            mc_err = f"{type(exc).__name__}: {exc}"
        mc_refs[variant] = {
            "price": mc_price,
            "std_error": mc_se,
            "num_paths": num_paths,
            "batches_used": batches_used,
            "price_pct_notional": (100.0 * mc_price / notional) if mc_price is not None else None,
            "se_pct_notional": (100.0 * mc_se / notional) if mc_se is not None else None,
            "error": mc_err,
            "timings": {"seconds": mc_seconds},
        }
        mc_se_pct = (100.0 * mc_se / notional) if mc_se is not None else None

        for level, n_x, n_v, n_t in ladder:
            ki_collisions = event_key_collision_count(terms.ki_times, terms.maturity_years, n_t)
            ko_collisions = event_key_collision_count(terms.ko_times, terms.maturity_years, n_t)
            cell = {
                "date": date_iso,
                "case": case_name,
                "variant": variant,
                "level": level,
                "grid": {"n_x": n_x, "n_v": n_v, "n_t": n_t},
                "pde_price": None,
                "mc_ref": mc_price,
                "mc_se": mc_se,
                "mc_paths": num_paths,
                "notional": notional,
                "signed_diff_pct": None,
                "abs_diff_pct": None,
                "rel_diff_pct": None,
                "tol_pct": (
                    gate_tolerance_pct(mc_se_pct, cfg["tol_abs_pct"])
                    if mc_se_pct is not None
                    else None
                ),
                "passed": False,
                "ki_event_key_collisions": ki_collisions,
                "ko_event_key_collisions": ko_collisions,
                "error": mc_err,  # broken reference: every cell fails closed
                "timings": {"pde_seconds": None, "mc_seconds": mc_seconds},
            }
            solver = _make_pde_engine(variant, model, pdep, (n_x, n_v, n_t))
            try:
                t1 = time.perf_counter()
                raw_pde = float(solver.price(product, env))
                pde_seconds = time.perf_counter() - t1
                cell["timings"]["pde_seconds"] = pde_seconds
                pde_err = _finite_or_error(raw_pde, "pde price")
                if pde_err is not None:
                    cell["error"] = (
                        pde_err if cell["error"] is None else f"{cell['error']}; {pde_err}"
                    )
                elif mc_err is None:
                    pde_price = raw_pde
                    signed = 100.0 * (pde_price - mc_price) / notional
                    cell.update(
                        {
                            "pde_price": pde_price,
                            "signed_diff_pct": signed,
                            "abs_diff_pct": abs(signed),
                            "rel_diff_pct": (
                                100.0 * abs(pde_price - mc_price) / abs(mc_price)
                                if mc_price
                                else None
                            ),
                            "passed": gate_cell_passed(signed, mc_se_pct, cfg["tol_abs_pct"]),
                        }
                    )
                else:
                    # Broken reference, finite PDE: keep the price as evidence.
                    cell["pde_price"] = raw_pde
            except Exception as exc:  # PDE failure is evidence against the PDE route
                pde_msg = f"{type(exc).__name__}: {exc}"
                cell["error"] = (
                    pde_msg if cell["error"] is None else f"{cell['error']}; {pde_msg}"
                )
            cells.append(cell)

        # Secondary evidence: model-consistent delta agreement (full case only,
        # medium grid; PDE bumped vs MC CRN-bumped at the reference config).
        if case_name == "full":
            medium_grid = next((g for g in ladder if g[0] == "medium"), None)
            if medium_grid is not None:
                _, n_x, n_v, n_t = medium_grid
                solver = _make_pde_engine(variant, model, pdep, (n_x, n_v, n_t))
                delta_row = {
                    "date": date_iso,
                    "case": case_name,
                    "variant": variant,
                    "level": "medium",
                    "delta_pde": None,
                    "delta_mc": None,
                    "abs_diff": None,
                    "error": None,
                    "timings": {},
                }
                try:
                    t1 = time.perf_counter()
                    delta_row["delta_pde"] = _bumped_pde_delta(solver, product, env)
                    delta_row["timings"]["pde_delta_seconds"] = time.perf_counter() - t1
                except Exception as exc:
                    delta_row["error"] = f"pde delta: {type(exc).__name__}: {exc}"
                try:
                    t1 = time.perf_counter()
                    delta_row["delta_mc"] = _bumped_mc_delta(
                        variant, model, product, env, mcp, cfg["mc"]
                    )
                    delta_row["timings"]["mc_delta_seconds"] = time.perf_counter() - t1
                except Exception as exc:
                    delta_row["error"] = (
                        f"{delta_row['error']}; mc delta: {type(exc).__name__}: {exc}"
                        if delta_row["error"]
                        else f"mc delta: {type(exc).__name__}: {exc}"
                    )
                for key in ("delta_pde", "delta_mc"):
                    err = _finite_or_error(delta_row[key], key)
                    if err is not None:
                        delta_row["error"] = (
                            f"{delta_row['error']}; {err}" if delta_row["error"] else err
                        )
                        delta_row[key] = None
                if delta_row["delta_pde"] is not None and delta_row["delta_mc"] is not None:
                    delta_row["abs_diff"] = abs(
                        delta_row["delta_pde"] - delta_row["delta_mc"]
                    )
                deltas.append(delta_row)

    return {"mc_refs": mc_refs, "cells": cells, "deltas": deltas}


def _extras_first_date(
    *,
    date_iso: str,
    terms: SnowballTerms,
    s0_inception: float,
    env: PricingEnvironment,
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
        valuation_date=env.valuation_date,
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


def _calibration_record(model) -> Dict[str, Any]:
    """Deterministic calibration record: drop volatile cache/timing fields."""
    volatile = {"cache_hit", "calibration_seconds", "per_expiry_rmse"}
    return {k: v for k, v in model.record.items() if k not in volatile}


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
    env = build_pricing_env(artifact, cfg["flat_rate"])
    models = {variant: calibrator.calibrate(variant, artifact) for variant in VARIANTS}

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
        env=env,
        models=models,
        cfg=cfg,
    )
    record["timings"]["full_seconds"] = time.perf_counter() - t_case

    decay_date = pick_decay_date(history.admitted_dates, terms)
    if decay_date is None:
        record["terms"]["decayed"] = None
        record["cases"]["decayed"] = None
    else:
        decay_artifact = history.surface_for(decay_date)
        dterms = decay_terms(terms, decay_date)
        denv = build_pricing_env(decay_artifact, cfg["flat_rate"])
        dmodels = {variant: calibrator.calibrate(variant, decay_artifact) for variant in VARIANTS}
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
            env=denv,
            models=dmodels,
            cfg=cfg,
        )
        record["timings"]["decayed_seconds"] = time.perf_counter() - t_case

    if cfg.get("extras_date") == inception.isoformat():
        record["extras"] = _extras_first_date(
            date_iso=inception.isoformat(),
            terms=terms,
            s0_inception=float(artifact.s0),
            env=env,
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
                "coarse=(90,36,round(32T)); medium=(200,60,ceil(400T)) - dt <= 1/400y "
                "so daily KI events cannot share an ADI step; fine=(300,90,2x medium)"
            ),
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


def build_decision_payload(cfg: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    cells = gate["cells"]
    variants: Dict[str, Any] = {}
    for variant in VARIANTS:
        v_cells = [c for c in cells if c["variant"] == variant]
        medium = [c for c in v_cells if c["level"] == "medium"]
        fine = [c for c in v_cells if c["level"] == "fine"]
        decision = decide_route(medium, fine, cfg["tol_abs_pct"])
        medium_grid = next(
            (c["grid"] for c in medium if c["case"] == "full"),
            medium[0]["grid"] if medium else None,
        )
        variants[variant] = {
            "route": decision["route"],
            "pde_params": {
                "n_x": medium_grid["n_x"] if medium_grid else None,
                "n_v": medium_grid["n_v"] if medium_grid else None,
                "n_t": medium_grid["n_t"] if medium_grid else None,
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
            },
            "mc_params": {
                "engine": "QESnowballMCEngine"
                if variant == VOL_MODEL_HESTON
                else "HestonSLVQESnowballMCEngine",
                "method": "randomized_quasi",
                "seed": cfg["seed"],
                "paths_per_batch": cfg["mc"]["paths_per_batch"],
                "batches": cfg["mc"]["batches"],
                "rqmc_min_batches": cfg["mc"]["batches"],
                "rqmc_max_batches": cfg["mc"]["batches"],
                "rqmc_target_std": 1e-12,
                "substeps_per_interval": cfg["mc"]["substeps_per_interval"],
                "note": "REFERENCE-quality config used for the gate: RQMC pinned "
                "(rqmc_min_batches == rqmc_max_batches, rqmc_target_std ~ 0) so "
                "every batch runs and the batch-spread SE is honest. The "
                "production backtest runner (Phase 4) may reduce path counts "
                "under a documented SE budget, recorded in the run manifest.",
            },
            "ladder": ladder_summary(v_cells),
            "gate": {
                "medium_pass": decision["medium_pass"],
                "fine_pass": decision["fine_pass"],
                "biased": decision["biased"],
                "bias": decision["bias"],
                "drift_max_pct": decision["drift_max_pct"],
                "drift_location": decision["drift_location"],
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
            "scheme": "QUADEXP",
            "seed": cfg["seed"],
            "paths_per_batch": cfg["mc"]["paths_per_batch"],
            "batches": cfg["mc"]["batches"],
            "total_paths": cfg["mc"]["paths_per_batch"] * cfg["mc"]["batches"],
            "substeps_per_interval": cfg["mc"]["substeps_per_interval"],
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
    lines.append("| variant | route | medium pass | fine pass | bias | max drift % | rationale |")
    lines.append("|---|---|---|---|---|---|---|")
    for variant, row in decision["variants"].items():
        g = row["gate"]
        lines.append(
            f"| {variant} | **{row['route']}** | {g['medium_pass']} | {g['fine_pass']} | "
            f"{g['biased']} | {g['drift_max_pct']:.3f} | {row['rationale']} |"
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

    lines.append("## Secondary evidence")
    lines.append("")
    deltas = gate.get("deltas") or []
    if deltas:
        lines.append("| date | variant | delta PDE | delta MC (CRN) | abs diff |")
        lines.append("|---|---|---|---|---|")
        for d in deltas:
            if d["delta_pde"] is None or d["delta_mc"] is None:
                lines.append(
                    f"| {d['date']} | {d['variant']} | n/a ({d.get('error')}) | | |"
                )
            else:
                lines.append(
                    f"| {d['date']} | {d['variant']} | {d['delta_pde']:+.4f} | "
                    f"{d['delta_mc']:+.4f} | {d['abs_diff']:.4f} |"
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
