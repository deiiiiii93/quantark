"""
Generate a standalone HTML demo that explains how r, q, and vol affect the
quoted KO rate for a Snowball RFQ.

Structure:
- Standard Snowball
- 103 monthly KO
- 75 daily KI
- 2Y maturity
- principal excluded PV convention
- fair KO rate solved from Snowball PV + financing-leg PV = 0.0
"""

from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset.equity.engine.pde import SnowballPDESolver
from asset.equity.param import PDEParams
from asset.equity.product.option import (
    SnowballOption,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_standard_snowball,
    create_stepdown_snowball,
)
from asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from asset.equity.product.option.snowball_helpers import generate_ko_observation_dates
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import ObservationType, ProtectionType


OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo.html"
CSV_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_scenarios.csv"


def _linspace(start: float, stop: float, num_points: int) -> list[float]:
    if num_points < 2:
        return [start]
    step = (stop - start) / (num_points - 1)
    return [round(start + i * step, 6) for i in range(num_points)]


R_GRID = _linspace(0.01, 0.05, 4)
Q_GRID = _linspace(0.05, 0.15, 4)
VOL_GRID = _linspace(0.15, 0.35, 4)
TENOR_GRID = _linspace(2.0, 3.0, 4)
KO_GRID = _linspace(95.0, 110.0, 4)
KI_GRID = _linspace(60.0, 85.0, 4)

DEFAULT_R = 0.03
DEFAULT_Q = 0.10
DEFAULT_VOL = 0.20
DEFAULT_TENOR = 2.0
PDE_GRID_SIZE = 400
PDE_TIME_STEPS = 400
DEMO_BUSINESS_DAYS_PER_YEAR = 244
R_IMPACT_BUMP = 0.01
Q_IMPACT_BUMP = 0.01
KO_RATE_BOUNDS = (0.0, 5.0)
AFFINE_KO_RATE_PAIR = (0.0, 2.0)
PREPAYMENT = 100.0
BASE_KO_BARRIER = 103.0
BASE_KI_BARRIER = 75.0
KO_BARRIER_BUMP = 0.25
KI_BARRIER_BUMP = 0.25
DEFAULT_VARIANT = "standard"
VARIANTS = {
    "standard": {
        "family": "standard",
        "label": "Standard",
        "description": "Monthly 103 KO with daily 75 KI.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "european_ki": {
        "family": "european_ki",
        "label": "European KI",
        "description": "Monthly 103 KO with KI observed only at maturity.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "parachute": {
        "family": "parachute",
        "label": "Parachute",
        "description": "Monthly 103 KO that drops to 75 on the final KO observation.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "stepdown": {
        "family": "stepdown",
        "label": "Stepdown",
        "description": "Monthly KO barrier steps down from 103 toward 75 over the 2Y life.",
        "product_protection_type": "NONE",
        "interest_protection_type": "FULL",
    },
    "standard_partial": {
        "family": "standard",
        "label": "Standard Partial-Protected",
        "description": "Standard Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "european_ki_partial": {
        "family": "european_ki",
        "label": "European KI Partial-Protected",
        "description": "European KI Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "parachute_partial": {
        "family": "parachute",
        "label": "Parachute Partial-Protected",
        "description": "Parachute Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
    "stepdown_partial": {
        "family": "stepdown",
        "label": "Stepdown Partial-Protected",
        "description": "Stepdown Snowball with partial protection tied to the KI level.",
        "product_protection_type": "PARTIAL",
        "interest_protection_type": "PARTIAL",
    },
}

DEMO_VARIANT_KEYS = (
    "standard",
    "european_ki",
    "standard_partial",
    "european_ki_partial",
)
DEMO_VARIANTS = {key: VARIANTS[key] for key in DEMO_VARIANT_KEYS}


@dataclass(frozen=True)
class DemoMeta:
    generated_at: str
    engine: str
    solver_grid_size: int | None
    solver_time_steps: int | None
    structure: dict[str, Any]
    ranges: dict[str, list[float]]


@dataclass(frozen=True)
class ExactBarrierTask:
    scenario_id: int
    variant: str
    tenor_idx: int
    rate_idx: int
    q_idx: int
    vol_idx: int
    ko_idx: int
    ki_idx: int
    tenor: float
    rate: float
    div_yield: float
    vol: float
    ko_barrier: float
    ki_barrier: float


DEFAULT_HTML_UI_COPY = {
    "eyebrow_en": "PDE-backed RFQ explainer",
    "chip_engine_en": "Snowball PDE surface + interpolation",
    "cube_note_en": "The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.",
    "eyebrow_cn": "PDE 驱动 RFQ 解释器",
    "chip_engine_cn": "雪球 PDE 曲面 + 插值",
    "cube_note_cn": "页面内嵌较粗 PDE 曲面，并在浏览器端做插值。",
}


def build_product(ko_rate: float, variant: str) -> SnowballOption:
    return build_product_with_barriers(
        ko_rate=ko_rate,
        variant=variant,
        maturity=DEFAULT_TENOR,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def get_variant_config(variant: str) -> dict[str, str]:
    try:
        return VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {variant}") from exc


def get_partial_protection_rate(ki_barrier: float) -> float:
    return max(0.0, min(1.0, 1.0 - ki_barrier / 100.0))


def get_num_monthly_observations(maturity: float) -> int:
    return max(1, int(round(maturity * 12)))


def get_num_daily_ki_observations(maturity: float) -> int:
    return max(1, int(round(maturity * DEMO_BUSINESS_DAYS_PER_YEAR)))


def generate_daily_ki_observation_dates(maturity: float) -> list[float]:
    """Generate evenly spaced KI dates using the demo's 244 business-day convention."""
    num_observations = get_num_daily_ki_observations(maturity)
    return [(i + 1) / num_observations * maturity for i in range(num_observations)]


def build_product_with_barriers(
    ko_rate: float,
    variant: str,
    *,
    maturity: float,
    ko_barrier: float,
    ki_barrier: float,
) -> SnowballOption:
    variant_config = get_variant_config(variant)
    product_protection = ProtectionType[variant_config["product_protection_type"]]
    protection_rate = (
        get_partial_protection_rate(ki_barrier)
        if product_protection == ProtectionType.PARTIAL
        else 0.0
    )
    num_observations = get_num_monthly_observations(maturity)
    common = {
        "initial_price": 100.0,
        "strike": 100.0,
        "maturity": maturity,
        "contract_multiplier": 1.0,
        "ko_rate": ko_rate,
        "ki_barrier": ki_barrier,
        "is_reverse": False,
        "rebate_rate": ko_rate,
        "include_principal": False,
        "protection_type": product_protection,
        "protection_rate": protection_rate,
    }
    stepdown_rate = (ko_barrier - ki_barrier) / (max(num_observations - 1, 1) * 100.0)
    daily_ki_dates = generate_daily_ki_observation_dates(maturity)

    if variant_config["family"] == "standard":
        return create_standard_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_observations=num_observations,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )
    if variant_config["family"] == "european_ki":
        return create_european_ki_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_ko_observations=num_observations,
        )
    if variant_config["family"] == "parachute":
        return create_parachute_snowball(
            **common,
            ko_barrier=ko_barrier,
            num_observations=num_observations,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )
    if variant_config["family"] == "stepdown":
        return create_stepdown_snowball(
            **common,
            num_observations=num_observations,
            initial_ko_barrier=ko_barrier,
            stepdown_rate=stepdown_rate,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=daily_ki_dates,
        )

    raise ValueError(f"Unknown variant family for {variant}")


def build_protected_product(variant: str):
    return build_protected_product_with_barriers(
        variant=variant,
        maturity=DEFAULT_TENOR,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def build_protected_product_with_barriers(
    variant: str,
    *,
    maturity: float,
    ko_barrier: float,
    ki_barrier: float,
):
    variant_config = get_variant_config(variant)
    interest_protection = ProtectionType[variant_config["interest_protection_type"]]
    protection_rate = (
        get_partial_protection_rate(ki_barrier)
        if interest_protection == ProtectionType.PARTIAL
        else 0.0
    )
    num_observations = get_num_monthly_observations(maturity)
    if variant_config["family"] == "parachute":
        ko_barrier_value = [ko_barrier] * (num_observations - 1) + [ki_barrier]
    elif variant_config["family"] == "stepdown":
        stepdown_amount = (ko_barrier - ki_barrier) / max(num_observations - 1, 1)
        ko_barrier_value = [ko_barrier - i * stepdown_amount for i in range(num_observations)]
    else:
        ko_barrier_value = ko_barrier

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier_value,
            ko_rate=1.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=generate_ko_observation_dates(maturity, "monthly"),
            ki_barrier=None,
        ),
        payoff_config=PayoffConfig(
            rebate_rate=1.0,
            include_principal=False,
            protection_type=interest_protection,
            protection_rate=protection_rate,
        ),
        accrual_config=AccrualConfig(is_annualized=False),
        maturity=maturity,
        contract_multiplier=1.0,
        is_reverse=False,
    )


def build_env(rate: float, div_yield: float, vol: float) -> PricingEnvironment:
    return PricingEnvironment(
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(spot=100.0, asset_name="Snowball Demo"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
    )


def solve_fair_ko_rate_with_engine(
    engine: Any,
    *,
    rate: float,
    div_yield: float,
    vol: float,
    tenor: float,
    variant: str,
    ko_barrier: float = BASE_KO_BARRIER,
    ki_barrier: float = BASE_KI_BARRIER,
) -> dict[str, float]:
    """Solve the fair KO rate using the supplied pricing engine."""
    env = build_env(rate=rate, div_yield=div_yield, vol=vol)
    protected_pv = engine.price(
        build_protected_product_with_barriers(
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    interest_component_pv = PREPAYMENT - protected_pv
    target_snowball_pv = interest_component_pv
    k0, k1 = AFFINE_KO_RATE_PAIR
    p0 = engine.price(
        build_product_with_barriers(
            ko_rate=k0,
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    p1 = engine.price(
        build_product_with_barriers(
            ko_rate=k1,
            variant=variant,
            maturity=tenor,
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    slope = (p1 - p0) / (k1 - k0)
    if abs(slope) < 1e-12:
        raise ValueError("KO rate slope is numerically flat")
    fair_ko_rate = k0 + (target_snowball_pv - p0) / slope
    if not (KO_RATE_BOUNDS[0] <= fair_ko_rate <= KO_RATE_BOUNDS[1]):
        raise ValueError("Fair KO rate falls outside configured display bounds")
    combined_pv = target_snowball_pv - interest_component_pv
    return {
        "quoted_ko_rate": fair_ko_rate,
        "snowball_target_pv": target_snowball_pv,
        "interest_component_pv": interest_component_pv,
        "protected_snowball_pv": protected_pv,
        "combined_pv": combined_pv,
    }


def solve_fair_ko_rate(
    rate: float,
    div_yield: float,
    vol: float,
    tenor: float,
    variant: str,
    *,
    ko_barrier: float = BASE_KO_BARRIER,
    ki_barrier: float = BASE_KI_BARRIER,
    pde_params: PDEParams,
) -> dict[str, float]:
    engine = SnowballPDESolver(params=pde_params)
    return solve_fair_ko_rate_with_engine(
        engine,
        rate=rate,
        div_yield=div_yield,
        vol=vol,
        tenor=tenor,
        variant=variant,
        ko_barrier=ko_barrier,
        ki_barrier=ki_barrier,
    )


def serialize_engine(engine: Any) -> dict[str, Any]:
    """Serialize supported engine settings for worker-process construction."""
    if isinstance(engine, SnowballPDESolver):
        return {
            "engine_type": "pde",
            "params": {
                "grid_size": engine.params.grid_size,
                "time_steps": engine.params.time_steps,
            },
        }
    if type(engine).__name__ == "SnowballQuadEngine":
        return {
            "engine_type": "quad",
            "params": {
                "grid_points": engine.params.grid_points,
            },
        }
    raise ValueError(f"Unsupported engine for parallel exact-grid build: {type(engine).__name__}")


def build_engine_from_config(engine_config: dict[str, Any]) -> Any:
    """Construct a pricing engine from its serialized config."""
    engine_type = engine_config["engine_type"]
    params = engine_config["params"]
    if engine_type == "pde":
        return SnowballPDESolver(
            params=PDEParams(
                grid_size=params["grid_size"],
                time_steps=params["time_steps"],
            )
        )
    if engine_type == "quad":
        from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
        from asset.equity.param import QuadParams

        return SnowballQuadEngine(
            params=QuadParams(
                grid_points=params["grid_points"],
            )
        )
    raise ValueError(f"Unknown engine_type: {engine_type}")


_EXACT_BARRIER_WORKER_ENGINE: Any | None = None


def init_exact_barrier_worker(engine_config: dict[str, Any]) -> None:
    """Initialize one pricing engine per worker process."""
    global _EXACT_BARRIER_WORKER_ENGINE
    _EXACT_BARRIER_WORKER_ENGINE = build_engine_from_config(engine_config)


def solve_exact_barrier_task(task: ExactBarrierTask) -> tuple[ExactBarrierTask, dict[str, float] | None]:
    """Solve one exact KO/KI scenario inside a worker process."""
    if _EXACT_BARRIER_WORKER_ENGINE is None:
        raise RuntimeError("Exact barrier worker engine is not initialized.")
    try:
        result = solve_fair_ko_rate_with_engine(
            _EXACT_BARRIER_WORKER_ENGINE,
            rate=task.rate,
            div_yield=task.div_yield,
            vol=task.vol,
            tenor=task.tenor,
            variant=task.variant,
            ko_barrier=task.ko_barrier,
            ki_barrier=task.ki_barrier,
        )
    except Exception:
        result = None
    return (task, result)


def apply_barrier_adjustment(
    base_value: float | None,
    ko_sensitivity: float | None,
    ki_sensitivity: float | None,
    ko_barrier: float,
    ki_barrier: float,
) -> float | None:
    """Apply first-order KO/KI barrier adjustment around the base anchor."""
    if base_value is None:
        return None
    adjusted = base_value
    if ko_sensitivity is not None:
        adjusted += ko_sensitivity * (ko_barrier - BASE_KO_BARRIER)
    if ki_sensitivity is not None:
        adjusted += ki_sensitivity * (ki_barrier - BASE_KI_BARRIER)
    return adjusted


def expand_scenario_rows_with_barriers(
    anchor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand base-barrier rows onto an explicit KO/KI CSV export grid."""
    rows: list[dict[str, Any]] = []
    scenario_id = 0
    for row in anchor_rows:
        for ko_barrier in KO_GRID:
            for ki_barrier in KI_GRID:
                scenario_id += 1
                expanded = dict(row)
                expanded["scenario_id"] = scenario_id
                expanded["ko_barrier"] = ko_barrier
                expanded["ki_barrier"] = ki_barrier
                if row["product_protection_type"] == "PARTIAL":
                    expanded["product_protection_rate"] = get_partial_protection_rate(ki_barrier)
                if row["interest_protection_type"] == "PARTIAL":
                    expanded["interest_protection_rate"] = get_partial_protection_rate(ki_barrier)
                expanded["quoted_ko_rate"] = apply_barrier_adjustment(
                    row["quoted_ko_rate"],
                    row["quote_ko_sensitivity"],
                    row["quote_ki_sensitivity"],
                    ko_barrier,
                    ki_barrier,
                )
                expanded["interest_pv"] = apply_barrier_adjustment(
                    row["interest_pv"],
                    row["interest_ko_sensitivity"],
                    row["interest_ki_sensitivity"],
                    ko_barrier,
                    ki_barrier,
                )
                if expanded["interest_pv"] is not None:
                    expanded["combined_pv"] = 0.0
                    expanded["product_pv"] = expanded["interest_pv"]
                else:
                    expanded["combined_pv"] = None
                    expanded["product_pv"] = None
                # Keep the protected-leg anchor explicit so downstream users know what changed exactly.
                expanded["protected_snowball_pv"] = row["protected_snowball_pv"]
                rows.append(expanded)
    return rows


def build_scenario_row(
    *,
    scenario_id: int,
    variant: str,
    tenor: float,
    rate: float,
    div_yield: float,
    vol: float,
    ko_barrier: float,
    ki_barrier: float,
    variant_config: dict[str, str],
    result: dict[str, float] | None,
    quote_ko_sensitivity: float | None = None,
    quote_ki_sensitivity: float | None = None,
    interest_ko_sensitivity: float | None = None,
    interest_ki_sensitivity: float | None = None,
) -> dict[str, Any]:
    """Build one exported scenario row."""
    return {
        "scenario_id": scenario_id,
        "variant": variant,
        "tenor": tenor,
        "r": rate,
        "q": div_yield,
        "vol": vol,
        "ko_barrier": ko_barrier,
        "ki_barrier": ki_barrier,
        "product_protection_type": variant_config["product_protection_type"],
        "product_protection_rate": (
            get_partial_protection_rate(ki_barrier)
            if variant_config["product_protection_type"] == "PARTIAL"
            else 0.0
        ),
        "interest_protection_type": variant_config["interest_protection_type"],
        "interest_protection_rate": (
            get_partial_protection_rate(ki_barrier)
            if variant_config["interest_protection_type"] == "PARTIAL"
            else 0.0
        ),
        "quoted_ko_rate": None if result is None else result["quoted_ko_rate"],
        "product_pv": None if result is None else result["snowball_target_pv"],
        "interest_pv": None if result is None else result["interest_component_pv"],
        "combined_pv": None if result is None else result["combined_pv"],
        "protected_snowball_pv": None if result is None else result["protected_snowball_pv"],
        "quote_ko_sensitivity": quote_ko_sensitivity,
        "quote_ki_sensitivity": quote_ki_sensitivity,
        "interest_ko_sensitivity": interest_ko_sensitivity,
        "interest_ki_sensitivity": interest_ki_sensitivity,
    }


def build_exact_barrier_tasks() -> list[ExactBarrierTask]:
    """Enumerate exact KO/KI scenario tasks in deterministic export order."""
    tasks: list[ExactBarrierTask] = []
    scenario_id = 0
    for variant in DEMO_VARIANTS:
        for tenor_idx, tenor in enumerate(TENOR_GRID):
            for rate_idx, rate in enumerate(R_GRID):
                for q_idx, div_yield in enumerate(Q_GRID):
                    for vol_idx, vol in enumerate(VOL_GRID):
                        for ko_idx, ko_barrier in enumerate(KO_GRID):
                            for ki_idx, ki_barrier in enumerate(KI_GRID):
                                scenario_id += 1
                                tasks.append(
                                    ExactBarrierTask(
                                        scenario_id=scenario_id,
                                        variant=variant,
                                        tenor_idx=tenor_idx,
                                        rate_idx=rate_idx,
                                        q_idx=q_idx,
                                        vol_idx=vol_idx,
                                        ko_idx=ko_idx,
                                        ki_idx=ki_idx,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=ko_barrier,
                                        ki_barrier=ki_barrier,
                                    )
                                )
    return tasks


def make_nested_none(dimensions: list[int]) -> Any:
    """Create a nested list filled with None for the requested dimensions."""
    if len(dimensions) == 1:
        return [None] * dimensions[0]
    return [make_nested_none(dimensions[1:]) for _ in range(dimensions[0])]


def build_exact_barrier_cube_parallel(
    *,
    engine: Any,
    progress_label: str,
    parallel_workers: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the exact 6D KO/KI cube using all available worker processes."""
    worker_count = parallel_workers or max(1, os.cpu_count() or 1)
    tasks = build_exact_barrier_tasks()
    total = len(tasks)
    engine_config = serialize_engine(engine)
    variant_cubes: dict[str, Any] = {}
    dimensions = [
        len(TENOR_GRID),
        len(R_GRID),
        len(Q_GRID),
        len(VOL_GRID),
        len(KO_GRID),
        len(KI_GRID),
    ]
    for variant in DEMO_VARIANTS:
        variant_cubes[variant] = {
            "quote": make_nested_none(dimensions),
            "interest": make_nested_none(dimensions),
            "protected": make_nested_none(dimensions),
            "snowballTarget": make_nested_none(dimensions),
        }

    rows: list[dict[str, Any] | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_exact_barrier_worker,
        initargs=(engine_config,),
    ) as executor:
        futures = [executor.submit(solve_exact_barrier_task, task) for task in tasks]
        for future in as_completed(futures):
            task, result = future.result()
            done += 1
            print(
                f"[{done:05d}/{total}] {task.variant} fair ko_rate via {progress_label} for "
                f"T={task.tenor:.2f}, r={task.rate:.4f}, q={task.div_yield:.4f}, vol={task.vol:.4f}, "
                f"ko={task.ko_barrier:.1f}, ki={task.ki_barrier:.1f}"
            )
            variant_cube = variant_cubes[task.variant]
            idx = (
                task.tenor_idx,
                task.rate_idx,
                task.q_idx,
                task.vol_idx,
                task.ko_idx,
                task.ki_idx,
            )
            if result is not None:
                variant_cube["quote"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["quoted_ko_rate"]
                variant_cube["interest"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["interest_component_pv"]
                variant_cube["protected"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["protected_snowball_pv"]
                variant_cube["snowballTarget"][idx[0]][idx[1]][idx[2]][idx[3]][idx[4]][idx[5]] = result["snowball_target_pv"]

            rows[task.scenario_id - 1] = build_scenario_row(
                scenario_id=task.scenario_id,
                variant=task.variant,
                tenor=task.tenor,
                rate=task.rate,
                div_yield=task.div_yield,
                vol=task.vol,
                ko_barrier=task.ko_barrier,
                ki_barrier=task.ki_barrier,
                variant_config=get_variant_config(task.variant),
                result=result,
            )

    return (variant_cubes, [row for row in rows if row is not None])


def build_cube_with_engines(
    *,
    engine: Any,
    bump_engine: Any,
    progress_label: str | None = None,
    exact_barrier_grid: bool = False,
    parallel_workers: int | None = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variant_cubes: dict[str, Any] = {}
    anchor_rows: list[dict[str, Any]] = []
    engine_label = progress_label or type(engine).__name__
    if exact_barrier_grid and (parallel_workers or 1) > 1:
        return build_exact_barrier_cube_parallel(
            engine=engine,
            progress_label=engine_label,
            parallel_workers=parallel_workers,
        )
    barrier_points = len(KO_GRID) * len(KI_GRID) if exact_barrier_grid else 1
    total = (
        len(DEMO_VARIANTS)
        * len(TENOR_GRID)
        * len(R_GRID)
        * len(Q_GRID)
        * len(VOL_GRID)
        * barrier_points
    )
    done = 0
    for variant in DEMO_VARIANTS:
        variant_config = get_variant_config(variant)
        quote_cube: list[Any] = []
        interest_cube: list[Any] = []
        protected_cube: list[Any] = []
        target_cube: list[Any] = []
        quote_ko_sens_cube: list[Any] = []
        quote_ki_sens_cube: list[Any] = []
        interest_ko_sens_cube: list[Any] = []
        interest_ki_sens_cube: list[Any] = []
        for tenor in TENOR_GRID:
            quote_r_slice: list[Any] = []
            interest_r_slice: list[Any] = []
            protected_r_slice: list[Any] = []
            target_r_slice: list[Any] = []
            quote_ko_sens_r_slice: list[Any] = []
            quote_ki_sens_r_slice: list[Any] = []
            interest_ko_sens_r_slice: list[Any] = []
            interest_ki_sens_r_slice: list[Any] = []
            for rate in R_GRID:
                quote_q_slice: list[Any] = []
                interest_q_slice: list[Any] = []
                protected_q_slice: list[Any] = []
                target_q_slice: list[Any] = []
                quote_ko_sens_q_slice: list[Any] = []
                quote_ki_sens_q_slice: list[Any] = []
                interest_ko_sens_q_slice: list[Any] = []
                interest_ki_sens_q_slice: list[Any] = []
                for div_yield in Q_GRID:
                    quote_vol_slice: list[Any] = []
                    interest_vol_slice: list[Any] = []
                    protected_vol_slice: list[Any] = []
                    target_vol_slice: list[Any] = []
                    quote_ko_sens_vol_slice: list[Any] = []
                    quote_ki_sens_vol_slice: list[Any] = []
                    interest_ko_sens_vol_slice: list[Any] = []
                    interest_ki_sens_vol_slice: list[Any] = []
                    for vol in VOL_GRID:
                        if exact_barrier_grid:
                            quote_ko_slice: list[list[float | None]] = []
                            interest_ko_slice: list[list[float | None]] = []
                            protected_ko_slice: list[list[float | None]] = []
                            target_ko_slice: list[list[float | None]] = []
                            for ko_barrier in KO_GRID:
                                quote_ki_slice: list[float | None] = []
                                interest_ki_slice: list[float | None] = []
                                protected_ki_slice: list[float | None] = []
                                target_ki_slice: list[float | None] = []
                                for ki_barrier in KI_GRID:
                                    done += 1
                                    print(
                                        f"[{done:03d}/{total}] {variant} fair ko_rate via {engine_label} for "
                                        f"T={tenor:.2f}, r={rate:.4f}, q={div_yield:.4f}, vol={vol:.4f}, "
                                        f"ko={ko_barrier:.1f}, ki={ki_barrier:.1f}"
                                    )
                                    try:
                                        result = solve_fair_ko_rate_with_engine(
                                            engine,
                                            rate=rate,
                                            div_yield=div_yield,
                                            vol=vol,
                                            tenor=tenor,
                                            variant=variant,
                                            ko_barrier=ko_barrier,
                                            ki_barrier=ki_barrier,
                                        )
                                        quote_ki_slice.append(result["quoted_ko_rate"])
                                        interest_ki_slice.append(result["interest_component_pv"])
                                        protected_ki_slice.append(result["protected_snowball_pv"])
                                        target_ki_slice.append(result["snowball_target_pv"])
                                        anchor_rows.append(
                                            build_scenario_row(
                                                scenario_id=done,
                                                variant=variant,
                                                tenor=tenor,
                                                rate=rate,
                                                div_yield=div_yield,
                                                vol=vol,
                                                ko_barrier=ko_barrier,
                                                ki_barrier=ki_barrier,
                                                variant_config=variant_config,
                                                result=result,
                                            )
                                        )
                                    except Exception:
                                        quote_ki_slice.append(None)
                                        interest_ki_slice.append(None)
                                        protected_ki_slice.append(None)
                                        target_ki_slice.append(None)
                                        anchor_rows.append(
                                            build_scenario_row(
                                                scenario_id=done,
                                                variant=variant,
                                                tenor=tenor,
                                                rate=rate,
                                                div_yield=div_yield,
                                                vol=vol,
                                                ko_barrier=ko_barrier,
                                                ki_barrier=ki_barrier,
                                                variant_config=variant_config,
                                                result=None,
                                            )
                                        )
                                quote_ko_slice.append(quote_ki_slice)
                                interest_ko_slice.append(interest_ki_slice)
                                protected_ko_slice.append(protected_ki_slice)
                                target_ko_slice.append(target_ki_slice)
                            quote_vol_slice.append(quote_ko_slice)
                            interest_vol_slice.append(interest_ko_slice)
                            protected_vol_slice.append(protected_ko_slice)
                            target_vol_slice.append(target_ko_slice)
                        else:
                            done += 1
                            print(
                                f"[{done:03d}/{total}] {variant} fair ko_rate via {engine_label} for "
                                f"T={tenor:.2f}, r={rate:.4f}, q={div_yield:.4f}, vol={vol:.4f}"
                            )
                            try:
                                result = solve_fair_ko_rate_with_engine(
                                    engine,
                                    rate=rate,
                                    div_yield=div_yield,
                                    vol=vol,
                                    tenor=tenor,
                                    variant=variant,
                                    ko_barrier=BASE_KO_BARRIER,
                                    ki_barrier=BASE_KI_BARRIER,
                                )
                                quote_vol_slice.append(result["quoted_ko_rate"])
                                interest_vol_slice.append(result["interest_component_pv"])
                                protected_vol_slice.append(result["protected_snowball_pv"])
                                target_vol_slice.append(result["snowball_target_pv"])
                                try:
                                    ko_up = solve_fair_ko_rate_with_engine(
                                        bump_engine,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        tenor=tenor,
                                        variant=variant,
                                        ko_barrier=BASE_KO_BARRIER + KO_BARRIER_BUMP,
                                        ki_barrier=BASE_KI_BARRIER,
                                    )
                                    quote_ko_sens_vol_slice.append(
                                        (ko_up["quoted_ko_rate"] - result["quoted_ko_rate"])
                                        / KO_BARRIER_BUMP
                                    )
                                    interest_ko_sens_vol_slice.append(
                                        (ko_up["interest_component_pv"] - result["interest_component_pv"])
                                        / KO_BARRIER_BUMP
                                    )
                                except Exception:
                                    quote_ko_sens_vol_slice.append(None)
                                    interest_ko_sens_vol_slice.append(None)

                                try:
                                    ki_up = solve_fair_ko_rate_with_engine(
                                        bump_engine,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        tenor=tenor,
                                        variant=variant,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER + KI_BARRIER_BUMP,
                                    )
                                    quote_ki_sens_vol_slice.append(
                                        (ki_up["quoted_ko_rate"] - result["quoted_ko_rate"])
                                        / KI_BARRIER_BUMP
                                    )
                                    interest_ki_sens_vol_slice.append(
                                        (ki_up["interest_component_pv"] - result["interest_component_pv"])
                                        / KI_BARRIER_BUMP
                                    )
                                except Exception:
                                    quote_ki_sens_vol_slice.append(None)
                                    interest_ki_sens_vol_slice.append(None)
                                anchor_rows.append(
                                    build_scenario_row(
                                        scenario_id=done,
                                        variant=variant,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER,
                                        variant_config=variant_config,
                                        result=result,
                                        quote_ko_sensitivity=quote_ko_sens_vol_slice[-1],
                                        quote_ki_sensitivity=quote_ki_sens_vol_slice[-1],
                                        interest_ko_sensitivity=interest_ko_sens_vol_slice[-1],
                                        interest_ki_sensitivity=interest_ki_sens_vol_slice[-1],
                                    )
                                )
                            except Exception:
                                quote_vol_slice.append(None)
                                interest_vol_slice.append(None)
                                protected_vol_slice.append(None)
                                target_vol_slice.append(None)
                                quote_ko_sens_vol_slice.append(None)
                                quote_ki_sens_vol_slice.append(None)
                                interest_ko_sens_vol_slice.append(None)
                                interest_ki_sens_vol_slice.append(None)
                                anchor_rows.append(
                                    build_scenario_row(
                                        scenario_id=done,
                                        variant=variant,
                                        tenor=tenor,
                                        rate=rate,
                                        div_yield=div_yield,
                                        vol=vol,
                                        ko_barrier=BASE_KO_BARRIER,
                                        ki_barrier=BASE_KI_BARRIER,
                                        variant_config=variant_config,
                                        result=None,
                                    )
                                )
                    quote_q_slice.append(quote_vol_slice)
                    interest_q_slice.append(interest_vol_slice)
                    protected_q_slice.append(protected_vol_slice)
                    target_q_slice.append(target_vol_slice)
                    if not exact_barrier_grid:
                        quote_ko_sens_q_slice.append(quote_ko_sens_vol_slice)
                        quote_ki_sens_q_slice.append(quote_ki_sens_vol_slice)
                        interest_ko_sens_q_slice.append(interest_ko_sens_vol_slice)
                        interest_ki_sens_q_slice.append(interest_ki_sens_vol_slice)
                quote_r_slice.append(quote_q_slice)
                interest_r_slice.append(interest_q_slice)
                protected_r_slice.append(protected_q_slice)
                target_r_slice.append(target_q_slice)
                if not exact_barrier_grid:
                    quote_ko_sens_r_slice.append(quote_ko_sens_q_slice)
                    quote_ki_sens_r_slice.append(quote_ki_sens_q_slice)
                    interest_ko_sens_r_slice.append(interest_ko_sens_q_slice)
                    interest_ki_sens_r_slice.append(interest_ki_sens_q_slice)
            quote_cube.append(quote_r_slice)
            interest_cube.append(interest_r_slice)
            protected_cube.append(protected_r_slice)
            target_cube.append(target_r_slice)
            if not exact_barrier_grid:
                quote_ko_sens_cube.append(quote_ko_sens_r_slice)
                quote_ki_sens_cube.append(quote_ki_sens_r_slice)
                interest_ko_sens_cube.append(interest_ko_sens_r_slice)
                interest_ki_sens_cube.append(interest_ki_sens_r_slice)
        variant_cube = {
            "quote": quote_cube,
            "interest": interest_cube,
            "protected": protected_cube,
            "snowballTarget": target_cube,
        }
        if not exact_barrier_grid:
            variant_cube.update(
                {
                    "quoteKoSens": quote_ko_sens_cube,
                    "quoteKiSens": quote_ki_sens_cube,
                    "interestKoSens": interest_ko_sens_cube,
                    "interestKiSens": interest_ki_sens_cube,
                }
            )
        variant_cubes[variant] = variant_cube
    rows = anchor_rows if exact_barrier_grid else expand_scenario_rows_with_barriers(anchor_rows)
    return (variant_cubes, rows)


def build_cube(
    *, pde_params: PDEParams
) -> tuple[dict[str, dict[str, list[list[list[list[float | None]]]]]], list[dict[str, Any]]]:
    """Build the demo cube with the legacy PDE engine configuration."""
    bump_params = PDEParams(
        grid_size=max(50, pde_params.grid_size // 2),
        time_steps=max(80, pde_params.time_steps // 2),
    )
    return build_cube_with_engines(
        engine=SnowballPDESolver(params=pde_params),
        bump_engine=SnowballPDESolver(params=bump_params),
        progress_label="SnowballPDESolver",
    )


def write_scenario_csv(
    rows: list[dict[str, Any]],
    csv_output_path: Path = CSV_OUTPUT_PATH,
) -> None:
    """Write scenario PV table for downstream analysis."""
    fieldnames = [
        "scenario_id",
        "variant",
        "tenor",
        "r",
        "q",
        "vol",
        "ko_barrier",
        "ki_barrier",
        "product_protection_type",
        "product_protection_rate",
        "interest_protection_type",
        "interest_protection_rate",
        "quoted_ko_rate",
        "product_pv",
        "interest_pv",
        "combined_pv",
        "protected_snowball_pv",
        "quote_ko_sensitivity",
        "quote_ki_sensitivity",
        "interest_ko_sensitivity",
        "interest_ki_sensitivity",
    ]
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_demo_data_json(
    data: dict[str, Any],
    *,
    data_output_path: Path,
) -> None:
    """Write the embedded demo payload to a standalone JSON file."""
    data_output_path.parent.mkdir(parents=True, exist_ok=True)
    data_output_path.write_text(
        json.dumps(data, separators=(",", ":")),
        encoding="utf-8",
    )


def read_demo_data_json(data_input_path: Path) -> dict[str, Any]:
    """Read a standalone demo payload JSON file."""
    return json.loads(data_input_path.read_text(encoding="utf-8"))


def build_demo_data(
    *,
    cubes: dict[str, dict[str, list[list[list[list[float | None]]]]]],
    engine_name: str,
    solver_grid_size: int | None,
    solver_time_steps: int | None,
    exact_barrier_grid: bool = False,
) -> dict[str, Any]:
    """Build the embedded payload for one engine configuration."""
    return {
        "tenorGrid": TENOR_GRID,
        "rGrid": R_GRID,
        "qGrid": Q_GRID,
        "volGrid": VOL_GRID,
        "defaults": {
            "tenor": DEFAULT_TENOR,
            "r": DEFAULT_R,
            "q": DEFAULT_Q,
            "vol": DEFAULT_VOL,
            "variant": DEFAULT_VARIANT,
        },
        "koRateBounds": list(KO_RATE_BOUNDS),
        "variants": cubes,
        "variantMeta": DEMO_VARIANTS,
        "meta": asdict(
            DemoMeta(
                generated_at=datetime.utcnow().isoformat() + "Z",
                engine=engine_name,
                solver_grid_size=solver_grid_size,
                solver_time_steps=solver_time_steps,
                structure={
                    "spot": 100.0,
                    "strike": 100.0,
                    "maturity_years": DEFAULT_TENOR,
                    "tenor_years": TENOR_GRID,
                    "ko_barrier_grid": KO_GRID,
                    "ki_barrier_grid": KI_GRID,
                    "base_ko_barrier": 103.0,
                    "base_ki_barrier": 75.0,
                    "base_ko_frequency": "monthly",
                    "base_ki_frequency": "daily",
                    "exact_barrier_grid": exact_barrier_grid,
                    "include_principal": False,
                    "target_pv": 0.0,
                    "prepayment": PREPAYMENT,
                    "protected_leg_ko_rate": 1.0,
                    "protected_leg_protection": "VARIANT_SPECIFIC",
                    "protected_leg_include_principal": False,
                    "protected_leg_annualized": False,
                },
                ranges={
                    "tenor": TENOR_GRID,
                    "r": R_GRID,
                    "q": Q_GRID,
                    "vol": VOL_GRID,
                    "ko": KO_GRID,
                    "ki": KI_GRID,
                },
            )
        ),
    }


def render_html(data: dict[str, Any], ui_copy: dict[str, str] | None = None) -> str:
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Snowball RFQ Demo: Quoted KO Rate</title>
  <style>
    :root {
      --paper: #f4efe7;
      --ink: #172126;
      --muted: #5f6769;
      --panel: rgba(255, 252, 247, 0.76);
      --line: rgba(23, 33, 38, 0.14);
      --accent: #c75d2c;
      --accent-soft: rgba(199, 93, 44, 0.12);
      --teal: #0e7a72;
      --teal-soft: rgba(14, 122, 114, 0.14);
      --gold: #ac7f1f;
      --shadow: 0 28px 80px rgba(27, 30, 32, 0.12);
      --radius: 26px;
      --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", Helvetica, sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 20%, rgba(199, 93, 44, 0.12), transparent 24%),
        radial-gradient(circle at 82% 16%, rgba(14, 122, 114, 0.12), transparent 24%),
        radial-gradient(circle at 75% 78%, rgba(172, 127, 31, 0.12), transparent 26%),
        linear-gradient(180deg, #f9f4ed 0%, var(--paper) 100%);
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(23, 33, 38, 0.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 33, 38, 0.028) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.7), transparent 92%);
    }

    .page {
      width: min(1320px, calc(100vw - 40px));
      margin: 28px auto 48px;
      display: grid;
      gap: 20px;
    }

    .hero,
    .panel {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      backdrop-filter: blur(14px);
      box-shadow: var(--shadow);
    }

    .hero {
      padding: 28px 30px 26px;
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 24px;
      align-items: end;
    }

    .eyebrow {
      font: 700 12px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--teal);
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(2.6rem, 4vw, 4.6rem);
      line-height: 0.92;
      font-weight: 700;
      letter-spacing: -0.04em;
      max-width: 8.5ch;
    }

    .hero p {
      margin: 18px 0 0;
      max-width: 62ch;
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.98rem;
    }

    .hero-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-self: stretch;
    }

    .chip {
      padding: 14px 14px 12px;
      border-radius: 18px;
      border: 1px solid rgba(23, 33, 38, 0.08);
      background: rgba(255, 255, 255, 0.48);
    }

    .chip-label {
      font: 700 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      margin-bottom: 10px;
    }

    .chip-value {
      font-size: 1.02rem;
      font-weight: 700;
    }

    .dashboard {
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }

    .panel {
      padding: 22px;
    }

    .panel-title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 18px;
    }

    .panel-title h2 {
      margin: 0;
      font-size: 1.08rem;
      letter-spacing: 0.02em;
    }

    .panel-title span {
      font: 700 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
    }

    .result-stack {
      display: grid;
      gap: 14px;
    }

    .quote-card {
      padding: 18px 18px 16px;
      border-radius: 22px;
      border: 1px solid rgba(23, 33, 38, 0.08);
      background: linear-gradient(140deg, rgba(255, 255, 255, 0.82), rgba(255, 247, 240, 0.72));
    }

    .quote-label {
      font: 700 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
      margin-bottom: 14px;
    }

    .quote-value {
      display: flex;
      align-items: flex-end;
      gap: 12px;
      flex-wrap: wrap;
    }

    .quote-value strong {
      font-family: var(--serif);
      font-size: clamp(2.6rem, 5vw, 4.4rem);
      line-height: 0.9;
      letter-spacing: -0.05em;
      color: var(--accent);
    }

    .quote-value small {
      font-size: 0.96rem;
      color: var(--muted);
      max-width: 20ch;
      line-height: 1.45;
    }

    .formula {
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(23, 33, 38, 0.035);
      border: 1px solid rgba(23, 33, 38, 0.08);
      font-family: var(--mono);
      font-size: 0.88rem;
      line-height: 1.7;
      color: var(--ink);
    }

    .control {
      padding: 14px 0 8px;
      border-bottom: 1px dashed rgba(23, 33, 38, 0.10);
    }

    .control:last-child {
      border-bottom: 0;
    }

    .control-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }

    .control-head label {
      font-size: 0.98rem;
      font-weight: 700;
    }

    .control-value {
      font-family: var(--mono);
      color: var(--teal);
      font-weight: 700;
    }

    input[type="range"] {
      width: 100%;
      margin: 0;
      accent-color: var(--accent);
    }

    .control-foot {
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      color: var(--muted);
      font: 600 11px/1 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .impact-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .impact-card {
      padding: 14px 14px 12px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid rgba(23, 33, 38, 0.08);
      min-height: 122px;
    }

    .impact-card h3 {
      margin: 0 0 10px;
      font-size: 0.94rem;
      letter-spacing: 0.02em;
    }

    .impact-main {
      font-family: var(--serif);
      font-size: 2rem;
      line-height: 0.95;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }

    .impact-main.positive {
      color: var(--accent);
    }

    .impact-main.negative {
      color: var(--teal);
    }

    .impact-card p {
      margin: 0;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.5;
    }

    .heatmap-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .heatmap-card {
      padding: 14px;
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.52);
      border: 1px solid rgba(23, 33, 38, 0.08);
    }

    .heatmap-card h3 {
      margin: 0 0 6px;
      font-size: 0.96rem;
    }

    .heatmap-card p {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.45;
      min-height: 2.4em;
    }

    canvas {
      width: 100%;
      display: block;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(23, 33, 38, 0.08);
    }

    .legend {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      color: var(--muted);
      font: 700 10px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }

    .legend-bar {
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #0e7a72 0%, #efe5c7 55%, #c75d2c 100%);
      border: 1px solid rgba(23, 33, 38, 0.08);
    }

    .note {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }

    .note-block {
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(23, 33, 38, 0.035);
      border: 1px solid rgba(23, 33, 38, 0.08);
    }

    .note-block h3 {
      margin: 0 0 10px;
      font-size: 0.96rem;
    }

    .note-block p,
    .note-block ul {
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.6;
    }

    .note-block ul {
      padding-left: 18px;
    }

    @media (max-width: 1080px) {
      .hero,
      .dashboard,
      .note,
      .heatmap-grid {
        grid-template-columns: 1fr;
      }

      .impact-grid {
        grid-template-columns: 1fr;
      }

      .hero-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 720px) {
      .page {
        width: min(100vw - 18px, 1320px);
        margin: 10px auto 28px;
      }

      .hero,
      .panel {
        padding: 18px;
      }

      .hero-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
          <div class="eyebrow" id="lang-eyebrow">PDE-backed RFQ explainer</div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span id="lang-toggle-label" style="font:700 11px/1 var(--mono);letter-spacing:0.14em;text-transform:uppercase;color:var(--muted);">Language</span>
            <button id="lang-en" type="button" style="padding:8px 12px;border-radius:999px;border:1px solid rgba(23,33,38,0.12);background:rgba(255,255,255,0.9);font:700 11px/1 var(--mono);letter-spacing:0.12em;text-transform:uppercase;color:var(--ink);cursor:pointer;">EN</button>
            <button id="lang-cn" type="button" style="padding:8px 12px;border-radius:999px;border:1px solid rgba(23,33,38,0.12);background:rgba(255,255,255,0.55);font:700 11px/1 var(--mono);letter-spacing:0.12em;text-transform:uppercase;color:var(--ink);cursor:pointer;">CN</button>
          </div>
        </div>
        <h1 id="lang-hero-title">How <em>r</em>, <em>q</em>, and vol reprice a Snowball quote.</h1>
        <p id="lang-hero-body">
          This demo solves the <strong>fair KO rate</strong> for Snowball variants under a
          Chinese-market <strong>integrated financing quote convention</strong>. The dealer quotes
          one KO coupon that makes the PV of the <strong>ex-principal Snowball</strong>
          match the <strong>interest component</strong> implied by the selected protected
          financing leg.
        </p>
      </div>
      <div class="hero-grid">
        <div class="chip">
          <div class="chip-label" id="lang-chip-structure-label">Structure</div>
          <div class="chip-value" id="lang-chip-structure-value">103 monthly KO / 75 daily KI</div>
        </div>
        <div class="chip">
          <div class="chip-label" id="lang-chip-tenor-label">Tenor</div>
          <div class="chip-value" id="lang-chip-tenor-value">2.0Y selected, spot = 100, strike = 100</div>
        </div>
        <div class="chip">
          <div class="chip-label" id="lang-chip-quote-label">Quote Convention</div>
          <div class="chip-value" id="lang-chip-quote-value">Solve <code>Snowball PV − Interest PV = 0</code></div>
        </div>
        <div class="chip">
          <div class="chip-label" id="lang-chip-engine-label">Engine</div>
          <div class="chip-value" id="lang-chip-engine-value">Snowball PDE surface + interpolation</div>
        </div>
      </div>
    </section>

    <section class="dashboard">
      <div class="panel">
        <div class="panel-title">
          <h2 id="lang-market-title">Market Controls</h2>
          <span id="lang-market-tag">live quote</span>
        </div>
        <div class="result-stack">
          <div class="control">
            <div class="control-head">
              <label for="variant-select" id="lang-variant-label">Snowball variant</label>
              <div class="control-value" id="variant-badge">--</div>
            </div>
            <select id="variant-select" style="width:100%;padding:12px 14px;border-radius:14px;border:1px solid rgba(23,33,38,0.12);background:rgba(255,255,255,0.72);font:600 0.95rem/1 var(--sans);color:var(--ink);">
              <option value="standard">Standard</option>
              <option value="european_ki">European KI</option>
              <option value="standard_partial">Standard Partial-Protected</option>
              <option value="european_ki_partial">European KI Partial-Protected</option>
            </select>
            <div class="control-foot"><span id="variant-caption">--</span><span id="lang-variant-foot">structure</span></div>
          </div>

          <div class="quote-card">
            <div class="quote-label" id="lang-quote-label">Quoted KO Rate</div>
            <div class="quote-value">
              <strong id="ko-rate-value">--</strong>
              <small id="ko-rate-caption">Fair coupon that makes Snowball PV match the interest leg.</small>
            </div>
          </div>

          <div class="quote-card">
            <div class="quote-label" id="lang-financing-label">Financing Leg</div>
            <div class="quote-value">
              <strong id="interest-pv-value">--</strong>
              <small id="interest-pv-caption">Prepayment minus protected, ex-principal, unannualized-100% Snowball PV.</small>
            </div>
          </div>

          <div class="formula" id="lang-formula">
            <strong>Interest PV</strong> = Prepayment − PV(Protected Snowball, principal excluded, KO rate = 100% unannualized)<br />
            Solve <strong>ko_rate</strong> such that<br />
            <strong>V<sub>snowball, exN</sub>(r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />
            with monthly KO observations, daily KI observations, and a selectable 2Y-3Y tenor.
          </div>

          <div class="control">
            <div class="control-head">
              <label for="tenor-slider" id="lang-tenor-label">Tenor</label>
              <div class="control-value" id="tenor-value">--</div>
            </div>
            <input id="tenor-slider" type="range" min="2.0" max="3.0" step="0.01" value="2.0" />
            <div class="control-foot"><span>2.0Y</span><span>3.0Y</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="r-slider" id="lang-r-label">Risk-free rate <em>r</em></label>
              <div class="control-value" id="r-value">--</div>
            </div>
            <input id="r-slider" type="range" min="0.01" max="0.05" step="0.0005" value="0.03" />
            <div class="control-foot"><span>1.0%</span><span>5.0%</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="q-slider" id="lang-q-label">Dividend yield <em>q</em></label>
              <div class="control-value" id="q-value">--</div>
            </div>
            <input id="q-slider" type="range" min="0.05" max="0.15" step="0.0005" value="0.10" />
            <div class="control-foot"><span>5.0%</span><span>15.0%</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="rq-link-toggle" id="lang-rq-label">r-q link</label>
              <div class="control-value" id="rq-link-value">Off</div>
            </div>
            <label style="display:flex;align-items:center;gap:10px;padding:6px 0 2px;color:var(--muted);font-size:0.92rem;">
              <input id="rq-link-toggle" type="checkbox" style="width:18px;height:18px;accent-color:var(--teal);" />
              Preserve current <code style="font-family:var(--mono);">q - r</code> spread when either slider moves
            </label>
            <div class="control-foot"><span id="rq-link-caption">Independent moves</span><span id="lang-rq-foot">toggle</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="vol-slider" id="lang-vol-label">Flat vol <em>σ</em></label>
              <div class="control-value" id="vol-value">--</div>
            </div>
            <input id="vol-slider" type="range" min="0.15" max="0.35" step="0.0025" value="0.20" />
            <div class="control-foot"><span>15.0%</span><span>35.0%</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="ko-slider" id="lang-ko-label">KO barrier</label>
              <div class="control-value" id="ko-value">--</div>
            </div>
            <input id="ko-slider" type="range" min="95" max="110" step="0.25" value="103" />
            <div class="control-foot"><span>95</span><span>110</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="ki-slider" id="lang-ki-label">KI barrier</label>
              <div class="control-value" id="ki-value">--</div>
            </div>
            <input id="ki-slider" type="range" min="60" max="85" step="0.25" value="75" />
            <div class="control-foot"><span>60</span><span>85</span></div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">
          <h2 id="lang-impact-title">Local Impact</h2>
          <span id="lang-impact-tag">first-order intuition</span>
        </div>
        <div class="impact-grid">
          <article class="impact-card">
            <h3 id="lang-impact-r-title">If <em>r</em> moves +100bp</h3>
            <div class="impact-main" id="impact-r">--</div>
            <p id="impact-r-text"></p>
          </article>
          <article class="impact-card">
            <h3 id="lang-impact-q-title">If <em>q</em> moves +100bp</h3>
            <div class="impact-main" id="impact-q">--</div>
            <p id="impact-q-text"></p>
          </article>
          <article class="impact-card">
            <h3 id="lang-impact-vol-title">If vol moves +1 vol pt</h3>
            <div class="impact-main" id="impact-vol">--</div>
            <p id="impact-vol-text"></p>
          </article>
        </div>

      </div>
    </section>

    <section class="note">
      <div class="note-block">
        <h3 id="lang-note-interpretation">Interpretation</h3>
        <p id="summary-text"></p>
      </div>
      <div class="note-block">
        <h3 id="lang-note-build">Build Notes</h3>
        <ul id="lang-note-list">
          <li>Daily KI is modeled as 252 discrete business-day observations per year and scales with tenor.</li>
          <li>The interest leg is valued as the selected protected, principal-excluded Snowball with KO rate fixed at 100% unannualized.</li>
          <li>The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.</li>
          <li>This is a pricing explainer, not a production quoting front-end.</li>
        </ul>
      </div>
    </section>
  </div>

  <script>
    const DATA = __DATA__;

    const tenorSlider = document.getElementById("tenor-slider");
    const rSlider = document.getElementById("r-slider");
    const qSlider = document.getElementById("q-slider");
    const rqLinkToggle = document.getElementById("rq-link-toggle");
    const rqLinkValue = document.getElementById("rq-link-value");
    const rqLinkCaption = document.getElementById("rq-link-caption");
    const langEnButton = document.getElementById("lang-en");
    const langCnButton = document.getElementById("lang-cn");
    const volSlider = document.getElementById("vol-slider");
    const koSlider = document.getElementById("ko-slider");
    const kiSlider = document.getElementById("ki-slider");
    const variantSelect = document.getElementById("variant-select");
    const variantBadge = document.getElementById("variant-badge");
    const variantCaption = document.getElementById("variant-caption");
    const koRateValue = document.getElementById("ko-rate-value");
    const koRateCaption = document.getElementById("ko-rate-caption");
    const interestPvValue = document.getElementById("interest-pv-value");
    const interestPvCaption = document.getElementById("interest-pv-caption");

    const tenorValue = document.getElementById("tenor-value");
    const rValue = document.getElementById("r-value");
    const qValue = document.getElementById("q-value");
    const volValue = document.getElementById("vol-value");
    const koValue = document.getElementById("ko-value");
    const kiValue = document.getElementById("ki-value");

    const impactR = document.getElementById("impact-r");
    const impactQ = document.getElementById("impact-q");
    const impactVol = document.getElementById("impact-vol");
    const impactRText = document.getElementById("impact-r-text");
    const impactQText = document.getElementById("impact-q-text");
    const impactVolText = document.getElementById("impact-vol-text");
    const summaryText = document.getElementById("summary-text");

    let currentLang = "en";
    let updateQueued = false;

    const I18N = {
      en: {
        toggleLabel: "Language",
        eyebrow: "PDE-backed RFQ explainer",
        heroTitle: "How <em>r</em>, <em>q</em>, and vol reprice a Snowball quote.",
        heroBody: "This demo solves the <strong>fair KO rate</strong> for Snowball variants under a Chinese-market <strong>integrated financing quote convention</strong>. The dealer quotes one KO coupon that makes the PV of the <strong>ex-principal Snowball</strong> match the <strong>interest component</strong> implied by the selected protected financing leg.",
        chipStructureLabel: "Structure",
        chipStructureValue: "103 monthly KO / 75 daily KI",
        chipTenorLabel: "Tenor",
        chipTenorValue: (tenor) => `${tenor} selected, spot = 100, strike = 100`,
        chipQuoteLabel: "Quote Convention",
        chipQuoteValue: "Solve <code>Snowball PV − Interest PV = 0</code>",
        chipEngineLabel: "Engine",
        chipEngineValue: "Snowball PDE surface + interpolation",
        marketTitle: "Market Controls",
        marketTag: "live quote",
        variantLabel: "Snowball variant",
        variantFoot: "structure",
        quoteLabel: "Quoted KO Rate",
        financingLabel: "Financing Leg",
        formula: "<strong>Interest PV</strong> = Prepayment − PV(Protected Snowball, principal excluded, KO rate = 100% unannualized)<br />Solve <strong>ko_rate</strong> such that<br /><strong>V<sub>snowball, exN</sub>(T, r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />with monthly KO observations, daily KI observations, and a selectable 2Y-3Y tenor.",
        tenorLabel: "Tenor <em>T</em>",
        rLabel: "Risk-free rate <em>r</em>",
        qLabel: "Dividend yield <em>q</em>",
        rqLabel: "r-q link",
        rqFoot: "toggle",
        volLabel: "Flat vol <em>σ</em>",
        koLabel: "KO barrier",
        kiLabel: "KI barrier",
        impactTitle: "Local Impact",
        impactTag: "first-order intuition",
        impactRTitle: "If <em>r</em> moves +100bp",
        impactQTitle: "If <em>q</em> moves +100bp",
        impactVolTitle: "If vol moves +1 vol pt",
        noteInterpretation: "Interpretation",
        noteBuild: "Build Notes",
        notesList: [
          "Daily KI is modeled as 244 discrete business-day observations per year and scales with tenor.",
          "The interest leg is valued as the selected protected, principal-excluded Snowball with KO rate fixed at 100% unannualized.",
          "The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.",
          "This is a pricing explainer, not a production quoting front-end."
        ],
        rqOff: "Off",
        rqLinked: "Linked",
        rqIndependent: "Independent moves",
        rqHolding: (spread) => `Holding q-r spread = ${spread}`,
        variantActive: (desc) => `${desc} Barrier sliders active.`,
        variantPassive: (desc) => `${desc} Barrier sliders shown for standard reference only.`,
        quoteCaption: "Fair KO coupon needed to make Snowball PV match the embedded protected financing leg.",
        noQuoteCaption: "No positive fair KO rate available inside the embedded quote range.",
        noCombined: "Combined convention not available at this market point.",
        interestCaption: (prepayment, pv) => `Prepayment ${prepayment} minus protected ex-principal Snowball PV ${pv}.`,
        highRUp: "Higher r is lifting the fair coupon in this slice.",
        highRDown: "Higher r is easing the fair coupon because forward drift improves KO odds.",
        highQUp: "Higher q demands more coupon because forward carry deteriorates.",
        highQDown: "Higher q slightly reduces the required coupon in this slice.",
        highVolUp: "Higher vol demands more coupon to compensate for fatter downside KI risk.",
        highVolDown: "Higher vol slightly relaxes the quote in this slice.",
        summary: (variant, tenor, r, q, vol, ko, ki, quote, interestPv, targetPv, tail) =>
          `${variant}: at T=${tenor}, r=${r}, q=${q}, vol=${vol}, KO=${ko}, and KI=${ki}, the locally adjusted fair KO rate is ${quote}. The embedded interest leg contributes ${interestPv} of PV, computed as prepayment minus the selected protected ex-principal Snowball priced with unannualized 100% KO rate, so the quoted Snowball itself must also price to ${targetPv} for the quote to clear. ${tail}`,
        variantNames: {
          standard: "Standard",
          european_ki: "European KI",
          parachute: "Parachute",
          stepdown: "Stepdown",
          standard_partial: "Standard Partial-Protected",
          european_ki_partial: "European KI Partial-Protected",
          parachute_partial: "Parachute Partial-Protected",
          stepdown_partial: "Stepdown Partial-Protected"
        }
      },
      cn: {
        toggleLabel: "语言",
        eyebrow: "PDE 驱动 RFQ 解释器",
        heroTitle: "看懂 <em>r</em>、<em>q</em> 与波动率如何重定价雪球报价。",
        heroBody: "这个演示在中国市场常见的<strong>融资一体化报价口径</strong>下，求解雪球各变体的<strong>公平 KO 票息</strong>。交易员报价的单一 KO 票息，需要让<strong>不含本金雪球</strong>的 PV 与由所选<strong>保本融资腿</strong>隐含出来的<strong>利息成分</strong>相匹配。",
        chipStructureLabel: "结构",
        chipStructureValue: "103 月度敲出 / 75 日度敲入",
        chipTenorLabel: "期限",
        chipTenorValue: (tenor) => `${tenor}，现价 = 100，行权价 = 100`,
        chipQuoteLabel: "报价口径",
        chipQuoteValue: "求解 <code>雪球PV − 利息PV = 0</code>",
        chipEngineLabel: "引擎",
        chipEngineValue: "雪球 PDE 曲面 + 插值",
        marketTitle: "市场控制",
        marketTag: "实时报价",
        variantLabel: "雪球变体",
        variantFoot: "结构",
        quoteLabel: "KO 报价票息",
        financingLabel: "融资腿",
        formula: "<strong>利息 PV</strong> = 预付金 − PV(保本雪球，去本金，KO 票息 = 100% 非年化)<br />求解 <strong>ko_rate</strong> 使得<br /><strong>V<sub>snowball, exN</sub>(T, r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />结构为月度 KO、日度 KI，期限可在 2 年到 3 年间切换。",
        tenorLabel: "期限 <em>T</em>",
        rLabel: "无风险利率 <em>r</em>",
        qLabel: "分红率 <em>q</em>",
        rqLabel: "r-q 联动",
        rqFoot: "开关",
        volLabel: "平坦波动率 <em>σ</em>",
        koLabel: "KO 障碍",
        kiLabel: "KI 障碍",
        impactTitle: "局部影响",
        impactTag: "一阶直觉",
        impactRTitle: "<em>r</em> 上移 100bp",
        impactQTitle: "<em>q</em> 上移 100bp",
        impactVolTitle: "波动率上移 1 个 vol 点",
        noteInterpretation: "解读",
        noteBuild: "构建说明",
        notesList: [
          "日度 KI 按每年 244 个离散交易日建模，并随期限缩放。",
          "利息腿按所选保本、去本金、KO=100% 非年化的雪球估值。",
          "页面内嵌较粗 PDE 曲面，并在浏览器端做插值。",
          "这是一个定价解释器，不是生产级报价前端。"
        ],
        rqOff: "关闭",
        rqLinked: "联动",
        rqIndependent: "独立变动",
        rqHolding: (spread) => `保持 q-r 利差 = ${spread}`,
        variantActive: (desc) => `${desc} 障碍滑块已启用。`,
        variantPassive: (desc) => `${desc} 障碍滑块仅作标准结构参考。`,
        quoteCaption: "使雪球 PV 与内嵌保本融资腿相匹配所需的公平 KO 票息。",
        noQuoteCaption: "在当前内嵌区间内没有正的公平 KO 票息。",
        noCombined: "当前市场点下无法得到有效组合结果。",
        interestCaption: (prepayment, pv) => `预付金 ${prepayment} 减去去本金保本雪球 PV ${pv}。`,
        highRUp: "在这个切片里，更高的 r 会抬升公平票息。",
        highRDown: "在这个切片里，更高的 r 会通过改善向 103 KO 漂移而压低公平票息。",
        highQUp: "更高的 q 会恶化远期漂移，因此需要更高票息。",
        highQDown: "在这个切片里，更高的 q 略微降低所需票息。",
        highVolUp: "更高波动会放大下敲风险，因此需要更高票息补偿。",
        highVolDown: "在这个切片里，更高波动略微放松报价。",
        summary: (variant, tenor, r, q, vol, ko, ki, quote, interestPv, targetPv, tail) =>
          `${variant}：在 T=${tenor}、r=${r}、q=${q}、vol=${vol}、KO=${ko}、KI=${ki} 下，局部调整后的公平 KO 票息为 ${quote}。内嵌利息腿 PV 为 ${interestPv}，它来自“预付金减去所选保本、KO=100% 非年化的去本金雪球 PV”，因此报价雪球自身也必须定价到 ${targetPv} 才能满足当前口径。${tail}`,
        variantNames: {
          standard: "标准型",
          european_ki: "欧式 KI",
          parachute: "降落伞",
          stepdown: "递减敲出",
          standard_partial: "标准部分保本",
          european_ki_partial: "欧式 KI 部分保本",
          parachute_partial: "降落伞部分保本",
          stepdown_partial: "递减敲出部分保本"
        }
      }
    };

    function t(key, ...args) {
      const value = I18N[currentLang][key];
      return typeof value === "function" ? value(...args) : value;
    }

    function applyLanguage() {
      document.getElementById("lang-toggle-label").textContent = t("toggleLabel");
      document.getElementById("lang-eyebrow").textContent = t("eyebrow");
      document.getElementById("lang-hero-title").innerHTML = t("heroTitle");
      document.getElementById("lang-hero-body").innerHTML = t("heroBody");
      document.getElementById("lang-chip-structure-label").textContent = t("chipStructureLabel");
      document.getElementById("lang-chip-structure-value").textContent = t("chipStructureValue");
      document.getElementById("lang-chip-tenor-label").textContent = t("chipTenorLabel");
      document.getElementById("lang-chip-tenor-value").textContent = t("chipTenorValue", formatTenor(activeTenor()));
      document.getElementById("lang-chip-quote-label").textContent = t("chipQuoteLabel");
      document.getElementById("lang-chip-quote-value").innerHTML = t("chipQuoteValue");
      document.getElementById("lang-chip-engine-label").textContent = t("chipEngineLabel");
      document.getElementById("lang-chip-engine-value").textContent = t("chipEngineValue");
      document.getElementById("lang-market-title").textContent = t("marketTitle");
      document.getElementById("lang-market-tag").textContent = t("marketTag");
      document.getElementById("lang-variant-label").textContent = t("variantLabel");
      document.getElementById("lang-variant-foot").textContent = t("variantFoot");
      document.getElementById("lang-quote-label").textContent = t("quoteLabel");
      document.getElementById("lang-financing-label").textContent = t("financingLabel");
      document.getElementById("lang-formula").innerHTML = t("formula");
      document.getElementById("lang-tenor-label").innerHTML = t("tenorLabel");
      document.getElementById("lang-r-label").innerHTML = t("rLabel");
      document.getElementById("lang-q-label").innerHTML = t("qLabel");
      document.getElementById("lang-rq-label").textContent = t("rqLabel");
      document.getElementById("lang-rq-foot").textContent = t("rqFoot");
      document.getElementById("lang-vol-label").innerHTML = t("volLabel");
      document.getElementById("lang-ko-label").textContent = t("koLabel");
      document.getElementById("lang-ki-label").textContent = t("kiLabel");
      document.getElementById("lang-impact-title").textContent = t("impactTitle");
      document.getElementById("lang-impact-tag").textContent = t("impactTag");
      document.getElementById("lang-impact-r-title").innerHTML = t("impactRTitle");
      document.getElementById("lang-impact-q-title").innerHTML = t("impactQTitle");
      document.getElementById("lang-impact-vol-title").textContent = t("impactVolTitle");
      document.getElementById("lang-note-interpretation").textContent = t("noteInterpretation");
      document.getElementById("lang-note-build").textContent = t("noteBuild");
      const notes = t("notesList");
      document.getElementById("lang-note-list").innerHTML = notes.map((item) => `<li>${item}</li>`).join("");
      const options = variantSelect.options;
      for (let i = 0; i < options.length; i += 1) {
        const option = options[i];
        option.textContent = t("variantNames")[option.value] || option.textContent;
      }
      langEnButton.style.background = currentLang === "en" ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.55)";
      langCnButton.style.background = currentLang === "cn" ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.55)";
      refreshRQLinkUI();
    }

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function formatTenor(value) {
      const text = value.toFixed(2).replace(/\\.00$/, ".0").replace(/(\\.\\d)0$/, "$1");
      return `${text}Y`;
    }

    function formatPct(value, digits = 2) {
      return `${(value * 100).toFixed(digits)}%`;
    }

    function formatSignedBpsFromRate(deltaRate) {
      const bps = deltaRate * 10000;
      const sign = bps >= 0 ? "+" : "";
      return `${sign}${bps.toFixed(1)} bp`;
    }

    function sliderBounds(slider) {
      return {
        min: parseFloat(slider.min),
        max: parseFloat(slider.max),
      };
    }

    let rqLinkedSpread = parseFloat(qSlider.value) - parseFloat(rSlider.value);

    function refreshRQLinkUI() {
      rqLinkValue.textContent = rqLinkToggle.checked ? t("rqLinked") : t("rqOff");
      rqLinkCaption.textContent = rqLinkToggle.checked
        ? t("rqHolding", formatPct(rqLinkedSpread, 2))
        : t("rqIndependent");
    }

    function syncQFromR() {
      const qBounds = sliderBounds(qSlider);
      const linkedQ = clamp(
        parseFloat(rSlider.value) + rqLinkedSpread,
        qBounds.min,
        qBounds.max
      );
      qSlider.value = linkedQ.toFixed(3);
    }

    function syncRFromQ() {
      const rBounds = sliderBounds(rSlider);
      const linkedR = clamp(
        parseFloat(qSlider.value) - rqLinkedSpread,
        rBounds.min,
        rBounds.max
      );
      rSlider.value = linkedR.toFixed(3);
    }

    function locate(grid, value) {
      if (value <= grid[0]) {
        return { i: 0, t: 0 };
      }
      if (value >= grid[grid.length - 1]) {
        return { i: grid.length - 2, t: 1 };
      }
      for (let i = 0; i < grid.length - 1; i += 1) {
        if (value >= grid[i] && value <= grid[i + 1]) {
          const span = grid[i + 1] - grid[i];
          return { i, t: span === 0 ? 0 : (value - grid[i]) / span };
        }
      }
      return { i: grid.length - 2, t: 1 };
    }

    function activeVariantKey() {
      return variantSelect.value;
    }

    function activeTenor() {
      return parseFloat(tenorSlider.value);
    }

    function activeVariantData() {
      return DATA.variants[activeVariantKey()];
    }

    function sampleNested(node, locations, depth = 0) {
      const loc = locations[depth];
      const lowerNode = node[loc.i];
      const upperNode = node[loc.i + 1];

      if (depth === locations.length - 1) {
        if (lowerNode === null || upperNode === null) {
          return null;
        }
        return lerp(lowerNode, upperNode, loc.t);
      }

      if (lowerNode === null || upperNode === null) {
        return null;
      }
      const lowerValue = sampleNested(lowerNode, locations, depth + 1);
      const upperValue = sampleNested(upperNode, locations, depth + 1);
      if (lowerValue === null || upperValue === null) {
        return null;
      }
      return lerp(lowerValue, upperValue, loc.t);
    }

    function sampleCube(cube, tenor, rate, divYield, vol, koBarrier = null, kiBarrier = null) {
      const locations = [
        locate(DATA.tenorGrid, tenor),
        locate(DATA.rGrid, rate),
        locate(DATA.qGrid, divYield),
        locate(DATA.volGrid, vol),
      ];
      if (DATA.meta.structure.exact_barrier_grid) {
        locations.push(locate(DATA.meta.structure.ko_barrier_grid, koBarrier));
        locations.push(locate(DATA.meta.structure.ki_barrier_grid, kiBarrier));
      }
      return sampleNested(cube, locations);
    }

    function collectNonNullValues(node, values = []) {
      if (Array.isArray(node)) {
        node.forEach((child) => collectNonNullValues(child, values));
      } else if (node !== null) {
        values.push(node);
      }
      return values;
    }

    function currentRange() {
      return VARIANT_RANGES[activeVariantKey()];
    }

    const VARIANT_RANGES = Object.fromEntries(
      Object.entries(DATA.variants).map(([variantKey, variantData]) => {
        const flatValues = collectNonNullValues(variantData.quote);
        return [
          variantKey,
          {
            min: Math.min(...flatValues),
            max: Math.max(...flatValues),
          },
        ];
      })
    );

    function colorFor(value) {
      if (value === null) {
        return "#d6d0c7";
      }
      const range = currentRange();
      const denom = range.max - range.min;
      const t = clamp((value - range.min) / (denom === 0 ? 1 : denom), 0, 1);
      const hue = lerp(172, 18, t);
      const sat = lerp(66, 64, t);
      const light = lerp(36, 54, 1 - Math.abs(t - 0.55));
      return `hsl(${hue}, ${sat}%, ${light}%)`;
    }

    function sampleBarrierAware(baseCube, koSensCube, kiSensCube, tenor, rate, divYield, vol, koBarrier, kiBarrier) {
      const base = sampleCube(baseCube, tenor, rate, divYield, vol, koBarrier, kiBarrier);
      if (DATA.meta.structure.exact_barrier_grid) {
        return base;
      }
      if (base === null) {
        return null;
      }
      let adjusted = base;
      const koDelta = koBarrier - DATA.meta.structure.base_ko_barrier;
      const kiDelta = kiBarrier - DATA.meta.structure.base_ki_barrier;
      const koSens = sampleCube(koSensCube, tenor, rate, divYield, vol);
      const kiSens = sampleCube(kiSensCube, tenor, rate, divYield, vol);
      if (koSens !== null) {
        adjusted += koSens * koDelta;
      }
      if (kiSens !== null) {
        adjusted += kiSens * kiDelta;
      }
      return adjusted;
    }

    function describeDirection(delta, positiveDriverText, negativeDriverText) {
      if (delta > 0) {
        return positiveDriverText;
      }
      if (delta < 0) {
        return negativeDriverText;
      }
      return "Locally flat in this slice.";
    }

    function update() {
      const tenor = activeTenor();
      const rate = parseFloat(rSlider.value);
      const divYield = parseFloat(qSlider.value);
      const vol = parseFloat(volSlider.value);
      const koBarrier = parseFloat(koSlider.value);
      const kiBarrier = parseFloat(kiSlider.value);
      const variant = activeVariantKey();
      const variantData = activeVariantData();
      const variantMeta = DATA.variantMeta[variant];
      const quote = sampleBarrierAware(
        variantData.quote,
        variantData.quoteKoSens,
        variantData.quoteKiSens,
        tenor,
        rate,
        divYield,
        vol,
        koBarrier,
        kiBarrier,
      );
      const interestPv = sampleBarrierAware(
        variantData.interest,
        variantData.interestKoSens,
        variantData.interestKiSens,
        tenor,
        rate,
        divYield,
        vol,
        koBarrier,
        kiBarrier,
      );
      const protectedPv = sampleCube(
        variantData.protected,
        tenor,
        rate,
        divYield,
        vol,
        koBarrier,
        kiBarrier,
      );
      const snowballTargetPv = interestPv;

      tenorValue.textContent = formatTenor(tenor);
      document.getElementById("lang-chip-tenor-value").textContent = t("chipTenorValue", formatTenor(tenor));
      rValue.textContent = formatPct(rate);
      qValue.textContent = formatPct(divYield);
      volValue.textContent = formatPct(vol);
      koValue.textContent = koBarrier.toFixed(1);
      kiValue.textContent = kiBarrier.toFixed(1);
      variantBadge.textContent = variantMeta.label;
      variantCaption.textContent =
        variant === DATA.defaults.variant
          ? t("variantActive", variantMeta.description)
          : t("variantPassive", variantMeta.description);

      if (quote === null) {
        koRateValue.textContent = "N/A";
        koRateCaption.textContent = t("noQuoteCaption");
        interestPvValue.textContent = "N/A";
        interestPvCaption.textContent = t("noCombined");
        summaryText.textContent = t("noCombined");
        return;
      }

      koRateValue.textContent = formatPct(quote, 2);
      koRateCaption.textContent = t("quoteCaption");
      interestPvValue.textContent = `${interestPv >= 0 ? "+" : ""}${interestPv.toFixed(2)}`;
      const adjustedProtectedPv =
        interestPv === null ? protectedPv : DATA.meta.structure.prepayment - interestPv;
      interestPvCaption.textContent = t(
        "interestCaption",
        DATA.meta.structure.prepayment.toFixed(2),
        adjustedProtectedPv.toFixed(2)
      );

      const rUp = sampleBarrierAware(
        variantData.quote,
        variantData.quoteKoSens,
        variantData.quoteKiSens,
        tenor,
        clamp(rate + 0.01, DATA.rGrid[0], DATA.rGrid.at(-1)),
        divYield,
        vol,
        koBarrier,
        kiBarrier,
      );
      const qUp = sampleBarrierAware(
        variantData.quote,
        variantData.quoteKoSens,
        variantData.quoteKiSens,
        tenor,
        rate,
        clamp(divYield + 0.01, DATA.qGrid[0], DATA.qGrid.at(-1)),
        vol,
        koBarrier,
        kiBarrier,
      );
      const volUp = sampleBarrierAware(
        variantData.quote,
        variantData.quoteKoSens,
        variantData.quoteKiSens,
        tenor,
        rate,
        divYield,
        clamp(vol + 0.01, DATA.volGrid[0], DATA.volGrid.at(-1)),
        koBarrier,
        kiBarrier,
      );

      const dR = rUp === null ? 0 : rUp - quote;
      const dQ = qUp === null ? 0 : qUp - quote;
      const dVol = volUp === null ? 0 : volUp - quote;

      impactR.textContent = formatSignedBpsFromRate(dR);
      impactQ.textContent = formatSignedBpsFromRate(dQ);
      impactVol.textContent = formatSignedBpsFromRate(dVol);

      impactR.className = `impact-main ${dR >= 0 ? "positive" : "negative"}`;
      impactQ.className = `impact-main ${dQ >= 0 ? "positive" : "negative"}`;
      impactVol.className = `impact-main ${dVol >= 0 ? "positive" : "negative"}`;

      impactRText.textContent = describeDirection(
        dR,
        t("highRUp"),
        t("highRDown")
      );
      impactQText.textContent = describeDirection(
        dQ,
        t("highQUp"),
        t("highQDown")
      );
      impactVolText.textContent = describeDirection(
        dVol,
        t("highVolUp"),
        t("highVolDown")
      );

      const directionalText = [
        dR < 0 ? "Higher r tends to lower the quote by improving risk-neutral carry toward the 103 KO." : "Higher r is not helping the quote much in this slice.",
        dQ > 0 ? "Higher q tends to raise the quote because forward drift softens and KO becomes harder." : "q is muted here.",
        dVol > 0 ? "Higher vol pushes the quote up because downside KI risk dominates the upside KO benefit under the Snowball-versus-interest convention." : "vol is muted here.",
      ];

      summaryText.textContent =
        t(
          "summary",
          (t("variantNames")[variant] || variantMeta.label),
          formatTenor(tenor),
          formatPct(rate),
          formatPct(divYield),
          formatPct(vol),
          koBarrier.toFixed(1),
          kiBarrier.toFixed(1),
          formatPct(quote),
          interestPv.toFixed(2),
          snowballTargetPv.toFixed(2),
          directionalText.join(" ")
        );
    }

    function scheduleUpdate() {
      if (updateQueued) {
        return;
      }
      updateQueued = true;
      requestAnimationFrame(() => {
        updateQueued = false;
        update();
      });
    }

    variantSelect.value = DATA.defaults.variant;
    applyLanguage();

    rSlider.addEventListener("input", () => {
      if (rqLinkToggle.checked) {
        syncQFromR();
      }
      scheduleUpdate();
    });

    qSlider.addEventListener("input", () => {
      if (rqLinkToggle.checked) {
        syncRFromQ();
      }
      scheduleUpdate();
    });

    rqLinkToggle.addEventListener("change", () => {
      if (rqLinkToggle.checked) {
        rqLinkedSpread = parseFloat(qSlider.value) - parseFloat(rSlider.value);
      }
      refreshRQLinkUI();
      scheduleUpdate();
    });

    langEnButton.addEventListener("click", () => {
      currentLang = "en";
      applyLanguage();
      scheduleUpdate();
    });

    langCnButton.addEventListener("click", () => {
      currentLang = "cn";
      applyLanguage();
      scheduleUpdate();
    });

    [tenorSlider, volSlider, koSlider, kiSlider, variantSelect].forEach((slider) => {
      slider.addEventListener("input", scheduleUpdate);
    });

    scheduleUpdate();
  </script>
</body>
</html>
"""

    resolved_ui_copy = dict(DEFAULT_HTML_UI_COPY)
    if ui_copy:
        resolved_ui_copy.update(ui_copy)
    replacements = {
        DEFAULT_HTML_UI_COPY["eyebrow_en"]: resolved_ui_copy["eyebrow_en"],
        DEFAULT_HTML_UI_COPY["chip_engine_en"]: resolved_ui_copy["chip_engine_en"],
        DEFAULT_HTML_UI_COPY["cube_note_en"]: resolved_ui_copy["cube_note_en"],
        DEFAULT_HTML_UI_COPY["eyebrow_cn"]: resolved_ui_copy["eyebrow_cn"],
        DEFAULT_HTML_UI_COPY["chip_engine_cn"]: resolved_ui_copy["chip_engine_cn"],
        DEFAULT_HTML_UI_COPY["cube_note_cn"]: resolved_ui_copy["cube_note_cn"],
    }
    for source, target in replacements.items():
        template = template.replace(source, target)
    return template.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def write_demo_files(
    data: dict[str, Any],
    scenario_rows: list[dict[str, Any]],
    *,
    output_path: Path = OUTPUT_PATH,
    csv_output_path: Path = CSV_OUTPUT_PATH,
    ui_copy: dict[str, str] | None = None,
) -> None:
    """Write the demo HTML artifact and companion scenario CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(data, ui_copy=ui_copy), encoding="utf-8")
    write_scenario_csv(scenario_rows, csv_output_path=csv_output_path)


def main() -> None:
    pde_params = PDEParams(grid_size=PDE_GRID_SIZE, time_steps=PDE_TIME_STEPS)
    cubes, scenario_rows = build_cube(pde_params=pde_params)
    data = build_demo_data(
        cubes=cubes,
        engine_name="SnowballPDESolver",
        solver_grid_size=pde_params.grid_size,
        solver_time_steps=pde_params.time_steps,
    )
    write_demo_files(data, scenario_rows)
    print(f"Wrote demo HTML to {OUTPUT_PATH}")
    print(f"Wrote scenario CSV to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
