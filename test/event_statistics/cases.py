"""The fast cross-engine smoke matrix of the unified event statistics (spec AC20 fast, R3; T22).

One contract per row, priced by every engine that serves it, every number compared field by field against an MC
reference (MC is the reference, DESIGN §5):

- the MC reference: ``SEEDS`` independent PSEUDO runs of ``MC_PATHS`` paths each (independent batch means, spec
  AC21); a number's reference is the mean over the runs, its standard error the runs' spread / sqrt(runs); the
  expected knock-in time, a ratio, pools sum(P_k E_k) / sum(P_k) with a delta-method error. Every functional of the
  statistics (the cumulative knock-in table) gets its error the same way, from the runs;
- the vol-model references are unbiased in their stepping: local-vol MC with ``max_sde_step_days=1`` (its default,
  one log-Euler step per contractual interval, is biased on a skewed surface: plan ruling after T13b-diag), Heston
  MC with 32 QE substeps per contractual interval (one step per interval biases a quarterly digital by ~8.5 SE:
  T14 note);
- each grid engine runs at level ``L0`` in the smoke; its discretisation allowance per number is banked in
  ``allowances.json`` (``bank_allowances.py``): ``|L0 - L2|`` against a twice-refined level, with the convergence
  evidence ``|L1 - L2| < |L0 - L2|`` banked beside it (spec AC20: measured, never hand-picked).

The full DESIGN §5 matrix is T24's (``@pytest.mark.slow`` and ``QUANTARK_EVENT_STATS_MATRIX=1``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Mapping, Tuple

import numpy as np

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import HestonSnowballMCEngine, LocalVolSnowballMCEngine
from quantark.asset.equity.engine.pde import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams, QuadV2Params
from quantark.util.enum import CouponPayType

from test_asset.equity.engine_mc import mc_price_path_routes as routes

__all__ = ["CASES", "Case", "LEVELS", "MC_PATHS", "SEEDS", "fields_of", "mc_reference"]

#: independent PSEUDO runs of the MC reference and their paths (16 x 8192: every field's |z| measured <= 3, T22 report)
SEEDS = tuple(range(101, 117))
MC_PATHS = 8192
#: the grid levels: L0 runs in the smoke, L1 and L2 refine it (allowance |L0 - L2|, evidence |L1 - L2|)
LEVELS = ("L0", "L1", "L2")

HP = routes.HESTON
DCN_VALUATION = datetime(2024, 1, 15)
DCN_HESTON = routes.HESTON


def _pde(cls, lv=False):
    def at(points, steps):
        params = PDEParams() if points is None else PDEParams(grid=GridConfig(points=points, steps_per_day=steps))
        return (lambda: cls(params=params)) if lv else (lambda: cls(params))
    return {"L0": at(None, None), "L1": at(800, 8.0), "L2": at(1600, 16.0)}


def _quad_v2(cls):
    return {"L0": lambda: cls(QuadV2Params()), "L1": lambda: cls(QuadV2Params(order=10, cells_per_sd=3.0)),
            "L2": lambda: cls(QuadV2Params(order=12, cells_per_sd=4.0))}


def _quad_v1(cls):
    return {level: (lambda g=g: cls(QuadParams(grid_points=g)))
            for level, g in (("L0", 1001), ("L1", 4001), ("L2", 16001))}


def _adi(cls, slv=False):
    def at(n_x, n_v, n_t):
        extra = {"leverage_surface": routes.unit_leverage()} if slv else {}
        return lambda: cls(HP, n_x=n_x, n_v=n_v, n_t=n_t, **extra)
    return {"L0": at(120, 32, 200), "L1": at(240, 64, 400), "L2": at(480, 128, 800)}


def _dcn_pde(lv=False):
    def at(accuracy):
        return (lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy=accuracy))) if lv else \
            (lambda: DCNPDEEngine(PDEParams(accuracy=accuracy)))
    return {"L0": at("fast"), "L1": at("standard"), "L2": at("high")}


def _dcn_adi():
    return {"L0": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=61, n_v=21),
            "L1": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=121, n_v=41, substeps_per_interval=2),
            "L2": lambda: HestonDCNPDESolver(DCN_HESTON, n_x=241, n_v=81, substeps_per_interval=4)}


@dataclass(frozen=True)
class Case:
    """One contract, its market, its MC reference engine (``mc(seed)``) and the grid engines serving it."""
    family: str
    product: Callable[[], object]
    env: Callable[[], object]
    mc: Callable[[int], object]
    grid: Mapping[str, Mapping[str, Callable[[], object]]]


def _mc_params(seed):
    return MCParams(num_paths=MC_PATHS, seed=seed)


def _dcn_product():
    return routes.dcn(valuation=DCN_VALUATION)


CASES: Dict[str, Case] = {
    "flat_snowball_cki": Case(
        "snowball", routes.snowball, routes.flat_env, lambda s: SnowballMCEngine(params=_mc_params(s)),
        {"pde_1d": _pde(SnowballPDESolver), "quad_v2": _quad_v2(SnowballQuadEngineV2),
         "quad_v1": _quad_v1(SnowballQuadEngine)}),
    "flat_phoenix_memory_expiry_dki": Case(
        "phoenix", lambda: routes.phoenix(pay=CouponPayType.EXPIRY, ki="discrete"), routes.flat_env,
        lambda s: PhoenixMCEngine(params=_mc_params(s)),
        {"pde_1d": _pde(PhoenixPDESolver), "quad_v2": _quad_v2(PhoenixQuadEngineV2),
         "quad_v1": _quad_v1(PhoenixQuadEngine)}),
    "flat_ko_reset_absolute_dki": Case(
        "ko_reset", routes.ko_reset, routes.flat_env, lambda s: SnowballMCEngine(params=_mc_params(s)),
        {"pde_1d": _pde(KOResetSnowballPDESolver), "quad_v2": _quad_v2(KOResetSnowballQuadEngineV2),
         "quad_v1": _quad_v1(KOResetSnowballQuadEngine)}),
    "lv_snowball_dki": Case(
        "snowball", lambda: routes.snowball("discrete"), routes.grid_env,
        lambda s: LocalVolSnowballMCEngine(_mc_params(s), max_sde_step_days=1.0),
        {"local_vol_pde_1d": _pde(LocalVolSnowballPDESolver, lv=True)}),
    "heston_snowball_dki": Case(
        "snowball", lambda: routes.snowball("discrete"), routes.flat_env,
        lambda s: HestonSnowballMCEngine(HP, params=_mc_params(s), substeps_per_interval=32),
        {"heston_adi": _adi(HestonSnowballPDESolver), "slv_adi_unit_leverage": _adi(HestonSLVSnowballPDESolver,
                                                                                     slv=True)}),
    "dcn_flat": Case(
        "dcn", _dcn_product, lambda: routes.dcn_env(valuation=DCN_VALUATION),
        lambda s: DCNMCEngine(num_paths=MC_PATHS, seed=s, use_sobol=False, num_workers=1),
        {"dcn_pde": _dcn_pde()}),
    "dcn_local_vol": Case(
        "dcn", _dcn_product, lambda: routes.dcn_env(grid=True, valuation=DCN_VALUATION),
        lambda s: LocalVolDCNMCEngine(num_paths=MC_PATHS, seed=s, use_sobol=False, num_workers=1),
        {"dcn_local_vol_pde": _dcn_pde(lv=True)}),
    "dcn_heston": Case(
        "dcn", _dcn_product, lambda: routes.dcn_env(valuation=DCN_VALUATION),
        lambda s: HestonDCNMCEngine(model_params=DCN_HESTON, substeps_per_interval=4, num_paths=MC_PATHS, seed=s,
                                    use_sobol=False, num_workers=1),
        {"dcn_heston_adi": _dcn_adi()}),
}


# --- the numbers compared -------------------------------------------------------------------------------------------

#: every compared number but the knock-in table and the expected time (arrays align across engines by row)
SCALAR_AND_ROW_FIELDS = (
    "pv",
    "observations.probability", "observations.discounted_cash", "observations.cash_to_pay",
    "coupons.probability", "coupons.discounted_cash", "coupons.cash_to_pay",
    "maturity.probability", "maturity.discounted_cash", "maturity.cash_to_pay",
    "ki_settlement.probability", "ki_settlement.discounted_cash", "ki_settlement.cash_to_pay",
    "knock_in.probability", "knock_in.expected_loss", "expected_lifetime",
)
#: the cumulative knock-in table is compared at most this many rows apart (every row of a short table)
CDF_STRIDE = 21


def _get(stats, path):
    obj = stats
    for part in path.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part)
    return obj


def cdf_rows(size: int) -> np.ndarray:
    """The knock-in table rows whose cumulative probability is compared: every row of a table of at most 24 rows,
    else every ``CDF_STRIDE``-th row and the last."""
    if size <= 24:
        return np.arange(size)
    return np.unique(np.r_[np.arange(CDF_STRIDE - 1, size, CDF_STRIDE), size - 1])


def fields_of(stats) -> Dict[str, np.ndarray]:
    """``{path: float array}`` of the compared numbers (``ki_cdf``: the cumulative knock-in table at ``cdf_rows``;
    ``knock_in.expected_time`` kept apart: it is a ratio)."""
    out: Dict[str, np.ndarray] = {}
    for path in SCALAR_AND_ROW_FIELDS:
        value = _get(stats, path)
        if value is not None:
            out[path] = np.atleast_1d(np.asarray(value, dtype=float))
    table = stats.ki_table
    out["ki_table.times"] = np.asarray(table.times, dtype=float)
    out["ki_cdf"] = np.cumsum(np.asarray(table.probability, dtype=float))[cdf_rows(table.size)]
    if stats.knock_in.expected_time is not None:
        out["knock_in.expected_time"] = np.array([float(stats.knock_in.expected_time)])
    return out


@dataclass(frozen=True)
class Reference:
    """The MC reference: per number its mean over the independent runs and its standard error."""
    mean: Dict[str, np.ndarray]
    se: Dict[str, np.ndarray]
    runs: int
    knock_in_paths: float           # pooled knock-in path count (the expected-time ratio is gated only above 2)


def mc_reference(case: Case, seeds: Tuple[int, ...] = SEEDS) -> Reference:
    """The case's MC reference from ``len(seeds)`` independent PSEUDO runs (spec AC21: independent batch means)."""
    runs = [fields_of(case.mc(seed).calculate_event_stats(case.product(), case.env())) for seed in seeds]
    k = len(runs)
    mean, se = {}, {}
    for path in runs[0]:
        if path in ("knock_in.expected_time", "ki_table.times"):
            continue
        stack = np.vstack([r[path] for r in runs])
        mean[path] = stack.mean(axis=0)
        se[path] = stack.std(axis=0, ddof=1) / math.sqrt(k)
    times = runs[0]["ki_table.times"]
    for r in runs[1:]:
        np.testing.assert_array_equal(r["ki_table.times"], times)
    mean["ki_table.times"] = times
    probability = np.array([float(r["knock_in.probability"][0]) for r in runs])
    weighted = np.array([float(r["knock_in.probability"][0]) * float(r.get("knock_in.expected_time", [0.0])[0])
                         for r in runs])
    knock_in_paths = float(probability.sum() * MC_PATHS)
    if probability.sum() > 0.0:
        ratio = float(weighted.sum() / probability.sum())
        residual = weighted - ratio * probability
        mean["knock_in.expected_time"] = np.array([ratio])
        se["knock_in.expected_time"] = np.array(
            [math.sqrt(float(np.sum(residual ** 2)) / (k * (k - 1))) / float(probability.mean())])
    return Reference(mean=mean, se=se, runs=k, knock_in_paths=knock_in_paths)


def grid_statistics(case: Case, engine: str, level: str):
    return case.grid[engine][level]().calculate_event_stats(case.product(), case.env())
