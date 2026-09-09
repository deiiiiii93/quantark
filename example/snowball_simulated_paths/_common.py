"""Shared pieces of the simulated-path snowball study.

The product, the fair-coupon solver, the carry models and the hedge
policies are the q term-structure study's (``example/snowball_q_term_structure/_common.py``),
loaded here under a fixed module name so the two studies cannot drift.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.backtest.replay import AutocallableEngineConfig
from quantark.backtest.simulation import MarketPath, SnowballStressLibrary, StartState, stress_set
from quantark.backtest.simulation.results import jsonable
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parents[1]
Q_STUDY_DIR = PROJECT_ROOT / "example" / "snowball_q_term_structure"
DATA_DIR = STUDY_DIR / "data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "snowball_simulated_paths"
DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "output" / "snowball_q_term_structure"


def load_q_study():
    """The q term-structure study's helpers, once per process."""
    name = "q_term_structure_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Q_STUDY_DIR / "_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


Q = load_q_study()

MODELS = ("flat_active", "term_flat_q", "term_opt_tail")
HEDGES = ("front", "far")
BASELINE_CELL = "flat_active__front"

N_PATHS = 2000
N_DAYS = 275          # a 12-month product matures ~261 weekdays out; the engine refuses a shorter calendar
QUICK_PATHS = 40
MEAN_BLOCK_DAYS = 20
ANNUAL_DRIFT = 0.0
VOL_FLOOR = 0.08
CARRY_MODE = "changes"
SEED = 1
COST_BP = 1.0
SPOT_STEP = 0.0025
VOL_STEP = 0.01
Q_STEP = 0.0025
SURFACE_CACHE_BYTES = 2_000_000_000
STATE_CACHE_BYTES = 500_000_000
GATE_SURFACE = dict(sample_states=64, pv_tolerance_bp=25.0, delta_tolerance_hands=2.0)
GATE_LADDER = dict(sample_states=64, pv_tolerance_bp=10.0, delta_tolerance_hands=2.0)
CHECK_PATHS = 200
ORACLE_PATHS = 3


def cell_name(model: str, hedge: str) -> str:
    return f"{model}__{hedge}"


def engine_config(model: str, engine: str, *, quad_grid: int) -> AutocallableEngineConfig:
    """The replay engine config of one carry model on the PDE (life surface) or QUAD (repricing) engine."""
    if model not in Q.Q_MODELS:
        raise ValidationError(f"unknown carry model {model!r}; one of {tuple(Q.Q_MODELS)}")
    q_model = Q.Q_MODELS[model]
    if engine == "pde":
        kwargs: Dict[str, Any] = dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams())
    elif engine == "quad":
        kwargs = dict(pricing_engine_type=EngineType.QUADRATURE, quad_params=QuadParams(grid_points=int(quad_grid)))
    else:
        raise ValidationError("engine must be 'pde' or 'quad'")
    return AutocallableEngineConfig(
        dividend_source=q_model.dividend_source, futures_curve_extrapolation=q_model.extrapolation,
        futures_curve_min_tenor_days=int(q_model.min_tenor_days), **kwargs,
    )


def stress_paths(start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray) -> MarketPath:
    """The five designed adverse paths on the run calendar."""
    n = len(calendar)
    lib = SnowballStressLibrary
    # The stated lengths on the full calendar; shortened on a short one so
    # every scenario fits (each library method checks its own bound).
    scenarios = [
        lib.crash_into_ki(0.30, min(20, n // 2), n),
        lib.v_shape(0.28, min(20, n // 3), min(40, n // 3), n),
        lib.vol_spike(0.15, min(40, n - 2), n),
        lib.basis_blowout(-0.05, min(10, n // 2), n),
        lib.grind_up_to_ko(0.002, min(60, n // 2), n),
    ]
    return stress_set(scenarios, start=start, calendar=calendar, tenor_grid=tenor_grid)


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
