"""Shared pieces of the simulated-path snowball study.

Named ``_sim_common`` rather than ``_common``: the q study's stages import
their own ``_common`` by that bare name, and one pytest process running
both studies' tests would otherwise hand one study the other's module.

The product, the fair-coupon solver, the carry models and the hedge
policies are the q term-structure study's (``example/snowball_q_term_structure/_common.py``),
loaded here under a fixed module name so the two studies cannot drift.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.engine.pde.grid.config import GridConfig
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

#: The baseline is the q study's (the engine's historical flat-q default),
#: taken from it by name so the two studies cannot disagree on what it is called.
MODELS = (Q.BASELINE_MODEL, "term_flat_q", "term_opt_tail")
HEDGES = ("front", "far")
BASELINE_CELL = f"{Q.BASELINE_MODEL}__front"

N_PATHS = 2000
N_DAYS = 275          # a 12-month product matures ~261 weekdays out; the engine refuses a shorter calendar
QUICK_PATHS = 40
MEAN_BLOCK_DAYS = 20
ANNUAL_DRIFT = 0.0
VOL_FLOOR = 0.08
#: Carry is resampled as LEVELS, not daily changes: the curve holds the total
#: log basis per tenor and a year of resampled changes is a random walk that
#: doubles the history's cross-path dispersion (1.5Y pillar std 0.119 on day
#: 274 against 0.056 in history) and drove the 1M pillar to -9.8% (history
#: floor -5%), which a contract 25 days from expiry turns into an annualised
#: yield above 100% -- the term dividend curve refuses it, rightly.
CARRY_MODE = "levels"
SEED = 1
COST_BP = 1.0
SPOT_STEP = 0.0025
#: A surface (or ladder node) is solved at the vol bucket's centre, so the
#: bucket half-width times the vega is a PV gap the gate sees.  Measured on
#: the 1Y product at the 2026-09-09 start state: vega about 80 bp of notional
#: per vol point at the start spot and about 150 bp/pt near the knock-in
#: barrier; at a 0.01 step the gate saw 74 bp (0.48 pt off the centre) and
#: the 25 bp budget cannot hold.  0.002 keeps the vol term under 15 bp.
#: The yield term is about 70 bp per 1% of q, so 0.001 keeps it under 4 bp.
#: Measured at (0.0025, 0.000625) on 8 real paths: worst 12.7 bp / 0.40
#: hands over 64 sampled states, 942 solves against 682 at (0.01, 0.0025) --
#: on real paths vol and carry both move daily, so nearly every state is
#: its own bucket until the path count is in the thousands.
VOL_STEP = 0.002
Q_STEP = 0.001
SURFACE_CACHE_BYTES = 2_000_000_000
STATE_CACHE_BYTES = 500_000_000
#: 2 hands is calibrated on a typical state, whose delta runs about 40
#: hands; beside the knock-in on the last trading day the delta is twenty
#: times that, and an absolute rule there is twenty times tighter for no
#: stated reason.  1% of the state's own delta is the second budget, and
#: the larger of the two governs, so on a typical state (1% of 40 is 0.4
#: hands) the absolute rule still decides and this changes nothing.
#:
#: 1% is not arbitrary.  At the one state that needs it the gap runs 0.88%
#: at this mesh, 0.30% at 3201/32 and 0.37% at 6401/64 -- the EXACT solve
#: the gate scores against moves by about half a percent between meshes, so
#: a budget below that measures the mesh, not the provider.  Measured worst
#: designed state 7.15 hands against an 8.14 allowance; every other state
#: is 1.17 hands or less.
GATE_SURFACE = dict(sample_states=64, pv_tolerance_bp=25.0, delta_tolerance_hands=2.0,
                    delta_tolerance_rel=0.01)
GATE_LADDER = dict(sample_states=64, pv_tolerance_bp=10.0, delta_tolerance_hands=2.0)
#: The QUAD spot-ladder check is off by default: its 2026-09-09 failures were
#: the flat-node carry gap, structural, and Design B replaced the provider it
#: was checking.  Pass --check-paths to run it.
CHECK_PATHS = 0
#: Exact QUAD repricing on each cell's first paths, paired with the cell's
#: own provider on the same paths: the engine check (reported, not gated).
EXACT_CHECK_PATHS = 40
ORACLE_PATHS = 3
#: The life surface's spot domain as fractions of the initial spot.  A
#: surface is solved once at the start spot and read along the whole path,
#: so its domain is a PATH envelope, not the pricer's default vol-scaled
#: pricing envelope (which a 30% crash on a low-vol day would leave; the
#: readout then fails closed).  The upside is capped by the KO barrier.
#: Measured cost of the PDE mesh per NEW spot on a 1Y product (the exact
#: leg of the gate and the oracle pay it once per state): default vol-scaled
#: domain [0.35, 2.87] 1.07 s, [0.60, 1.60] 0.15 s, [0.40, 2.50] 1.08 s.
SURFACE_SPOT_RANGE = (0.40, 1.60)
#: The life-surface mesh.  A surface is solved once per market bucket and
#: read for every state in it, so accuracy here is bought per bucket, not
#: per state.
#:
#: Measured on the study's own day-zero gate, profile default (400 points,
#: 4 steps/day) against this mesh: worst PV gap 16.95 -> 5.61 bp, worst
#: delta gap 13.04 -> 7.15 hands, and every designed state away from expiry
#: 0.14 hands or better.  Against the Gaussian-transition quadrature
#: reference the surface converges at second order, and what is left of the
#: gate gap on those states is the vol/carry bucket term above, which no
#: mesh can remove.
#:
#: It is not slower.  Measured in the run rather than in isolation, a
#: two-path 275-day cell does 521 solves in 599 s at this mesh and 509 in
#: 592 s on the profile, because the grid binder rebuilds its layout for
#: every market bucket and that rebuild, not the sweep, is the solve.  An
#: isolated repeat solve looks cheap on the profile (0.15 s against 0.69 s)
#: only because the binder's layout cache is warm by the second call; the
#: fleet never sees that.  Doubling again to 3201/32 DOES cost: the
#: day-zero gate alone runs 33.7 s against 12.7 s here, and a two-path cell
#: was observed once at about 44 min.  What the fleet sees either way is
#: size: a surface here is four times a profile one, so
#: SURFACE_CACHE_BYTES holds 99 of them against 397, and at the path counts
#: where bucket reuse starts to pay that budget is what to raise.
#:
#: The one state this does not settle is the last trading day a fraction of
#: a percent above the knock-in barrier, where the surface and the fresh
#: solve the gate scores it against both oscillate as the barrier moves
#: within a cell: 7.15 hands here, 2.42 at 3201/32, 3.06 at 6401/64.  Since
#: that is not a sequence more mesh will settle, the extra mesh buys a
#: smaller number against a reference that cannot support it, and it is not
#: bought: GATE_SURFACE judges that state on its own delta scale instead.
SURFACE_POINTS = 1601
SURFACE_STEPS_PER_DAY = 16.0
#: 16 steps/day over this 1Y daily-observed product asks for 5060 intervals
#: and the engine's shipped cap of 5000 delivers 4799 -- space refined with
#: time standing still, which is what made the old ladder look divergent.
#: Sized with headroom; a calendar that outgrows it is REFUSED by the
#: surface pricer, not quietly coarsened.
SURFACE_MAX_STEPS = 8000
#: Stated beside ``points`` so the two are edited together: points above
#: this budget is an error from the grid layer, never a silent trim.
SURFACE_MAX_POINTS = 2000


def cell_name(model: str, hedge: str) -> str:
    return f"{model}__{hedge}"


def engine_config(
    model: str, engine: str, *, quad_grid: int, s0: Optional[float] = None,
    spot_range: Optional[Tuple[float, float]] = None, quad_align: str = "auto",
) -> AutocallableEngineConfig:
    """The replay engine config of one carry model on the PDE (life surface) or QUAD (repricing) engine.

    With ``s0`` the PDE grid spans ``spot_range`` (default
    ``SURFACE_SPOT_RANGE``) times it, so one surface covers every spot a
    bootstrap or stress path can read.

    ``quad_align`` is ``QuadParams.align_priority``.  Under the default
    ``"auto"`` the lattice pins whichever barrier is nearest spot in log
    space, so the target changes at ``sqrt(KI*KO)`` and a bumped delta
    within one bump of that level differences two differently aligned
    lattices; ``"ki"`` or ``"ko"`` pins one barrier for every evaluation.
    It reaches only the QUAD branch, so a PDE cell's fingerprint does not
    move with it.
    """
    if model not in Q.Q_MODELS:
        raise ValidationError(f"unknown carry model {model!r}; one of {tuple(Q.Q_MODELS)}")
    q_model = Q.Q_MODELS[model]
    if getattr(q_model, "dividend_policy", None) is not None:
        # The simulation inverts the ACTIVE hedge contract (dividend_yield_for_day
        # takes one contract); a model whose carry comes from another contract
        # would price something else under the same cell name.  Fail closed.
        raise ValidationError(
            f"model {model!r} takes its carry contract from policy {q_model.dividend_policy!r}; "
            "the simulation has no carry contract separate from the hedge"
        )
    if engine == "pde":
        bounds: Tuple[Optional[float], Optional[float]] = (None, None)
        if s0 is not None:
            lo, hi = spot_range if spot_range is not None else SURFACE_SPOT_RANGE
            bounds = (lo * float(s0), hi * float(s0))
        # The mesh is set here and not only under ``s0`` so the surface and
        # the fresh solves the gate scores it against share one grid; on two
        # grids the gate measures the difference between them.
        grid = GridConfig(bounds=bounds, points=SURFACE_POINTS,
                          steps_per_day=SURFACE_STEPS_PER_DAY, max_steps=SURFACE_MAX_STEPS,
                          max_points=SURFACE_MAX_POINTS)
        kwargs: Dict[str, Any] = dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams(grid=grid))
    elif engine == "quad":
        kwargs = dict(pricing_engine_type=EngineType.QUADRATURE,
                      quad_params=QuadParams(grid_points=int(quad_grid), align_priority=str(quad_align)))
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
