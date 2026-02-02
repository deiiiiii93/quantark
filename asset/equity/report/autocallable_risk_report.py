"""
Autocallable risk profile report (Snowball-first).

Generates:
- PV/Greeks surfaces with a focus on dividend/basis risk
- risk-neutral event probabilities and cashflow attribution (MC analyzer)
- historical replay and parametric shock PnL distributions (spot/q shocks applied to today)
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from asset.equity.analysis.autocallable_path_analyzer import (
    AutocallablePathAnalyzer,
    RiskNeutralSnowballEventStats,
    ShockPnLDistribution,
)
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.engine.event_stats import AutocallableEventStats
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import MCParams, PDEParams, QuadParams
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.report.plotting import save_heatmap, save_line_plot
from asset.equity.report.surfaces import (
    GridSpec,
    SurfaceSet,
    build_q_grid,
    build_spot_grid,
    build_vol_grid,
    compute_surfaces_from_pv,
    derivative_1d,
)
from asset.equity.riskmeasures import GreeksCalculator
from param import FlatVolSurface, SpotQuote
from param import TermStructureVolSurface
from param.div import ContinuousDividendYield, DividendYield, TermStructureDividendYield
from priceenv import PricingEnvironment
from util.enum import EquityGreek
from util.exceptions import ValidationError
from util.numerical import safe_divide, safe_log, safe_sqrt
from asset.equity.report.term_structure import (
    BucketedDividendYield,
    BucketedVolSurface,
    ScaledVolSurface,
    ShiftedDividendYield,
    SkewSmileVolSurface,
    TenorBucket,
    default_tenor_buckets,
)
from portfolio import Portfolio
from stresstest.equity.config import EquityStressConfig
from stresstest.equity.engine import EquityStressEngine
from stresstest.scenario.scenario import Scenario
from stresstest.scenario.scenario_builder import ScenarioBuilder
from stresstest.stress.stress_types import StressType


@dataclass(frozen=True)
class ReportResult:
    report_path: Path
    output_dir: Path


@dataclass(frozen=True)
class SkewSmileShock:
    skew: float = 0.0
    smile: float = 0.0


@dataclass(frozen=True)
class StressScenario:
    name: str
    spot_shock: float
    vol_shock: float
    q_shift: float


def _clone_env(
    env: PricingEnvironment,
    *,
    spot: Optional[float] = None,
    vol: Optional[float] = None,
    q: Optional[float] = None,
    vol_surface: Optional[object] = None,
    div_yield: Optional[object] = None,
    valuation_date: Optional[datetime] = None,
) -> PricingEnvironment:
    new_env = PricingEnvironment(
        rate_curve=env.rate_curve,
        valuation_date=valuation_date or env.valuation_date,
        spot_quote=env.spot_quote,
        vol_surface=env.vol_surface,
        div_yield=env.div_yield,
        day_count_convention=env.day_count_convention,
        bus_days_in_year=env.bus_days_in_year,
    )
    if spot is not None:
        new_env.spot_quote = SpotQuote(spot=float(spot))
    if vol_surface is not None:
        new_env.vol_surface = vol_surface
    elif vol is not None:
        new_env.vol_surface = FlatVolSurface(volatility=float(vol))
    if div_yield is not None:
        new_env.div_yield = div_yield
    elif q is not None:
        new_env.div_yield = ContinuousDividendYield(div_yield=float(q))
    return new_env


def _shift_dividend_yield(
    base_div_yield: DividendYield, shift: float
) -> DividendYield:
    if shift == 0.0:
        return base_div_yield
    if isinstance(base_div_yield, TermStructureDividendYield):
        yields = [max(0.0, float(y) + shift) for y in base_div_yield.yields]
        return TermStructureDividendYield(
            times=list(base_div_yield.times), yields=yields
        )
    if isinstance(base_div_yield, ContinuousDividendYield):
        return ContinuousDividendYield(
            div_yield=max(0.0, float(base_div_yield.div_yield) + shift)
        )
    return ShiftedDividendYield(base=base_div_yield, shift=shift)


def _scale_vol_surface(base_surface: object, scale: float) -> object:
    if isinstance(base_surface, TermStructureVolSurface):
        vols = [float(v) * scale for v in base_surface.vols]
        return TermStructureVolSurface(times=list(base_surface.times), vols=vols)
    if isinstance(base_surface, FlatVolSurface):
        return FlatVolSurface(volatility=float(base_surface.volatility) * scale)
    return ScaledVolSurface(base=base_surface, scale=scale)


def _select_snowball_pricing_engine(
    *,
    preference: Sequence[str],
    quad_params: Optional[QuadParams],
    pde_params: Optional[PDEParams],
    mc_params: Optional[MCParams],
) -> Tuple[BaseEngine, str]:
    if quad_params is None:
        # Report surfaces may require hundreds to thousands of reprices; use a smaller
        # quadrature grid by default to keep runtime reasonable.
        quad_params = QuadParams(grid_points=301)
    if pde_params is None:
        pde_params = PDEParams()
    if mc_params is None:
        mc_params = MCParams()

    tried = []
    for name in preference:
        engine_name = name.lower()
        tried.append(engine_name)
        try:
            if engine_name == "quad":
                return SnowballQuadEngine(params=quad_params), "quad"
            if engine_name == "pde":
                return SnowballPDESolver(params=pde_params), "pde"
            if engine_name == "mc":
                return SnowballMCEngine(params=mc_params), "mc"
        except Exception:
            continue

    raise ValidationError(f"Unable to select a Snowball engine from preference={list(preference)} (tried={tried})")


def _compute_pv_surface_sq(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    spot_grid: np.ndarray,
    q_grid: np.ndarray,
    base_div_yield: DividendYield,
    base_q: float,
) -> np.ndarray:
    pv = np.zeros((spot_grid.size, q_grid.size), dtype=float)
    for j, q in enumerate(q_grid):
        shift = float(q) - float(base_q)
        div_yield = ShiftedDividendYield(base=base_div_yield, shift=shift)
        for i, s in enumerate(spot_grid):
            env = _clone_env(pricing_env, spot=float(s), div_yield=div_yield)
            pv[i, j] = float(engine.price(product, env))
    return pv


def _compute_point_surfaces_sq(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    greeks_calculator: GreeksCalculator,
    spot_grid: np.ndarray,
    q_grid: np.ndarray,
    base_div_yield: DividendYield,
    base_q: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pv = np.zeros((spot_grid.size, q_grid.size), dtype=float)
    delta = np.zeros_like(pv)
    rhoq = np.zeros_like(pv)
    v_sq = np.zeros_like(pv)
    greek_list = [
        EquityGreek.PRICE,
        EquityGreek.DELTA,
        EquityGreek.DIVIDEND_RHO,
        EquityGreek.DELTA_Q,
    ]
    for j, q in enumerate(q_grid):
        shift = float(q) - float(base_q)
        div_yield = _shift_dividend_yield(base_div_yield, shift)
        for i, s in enumerate(spot_grid):
            env = _clone_env(pricing_env, spot=float(s), div_yield=div_yield)
            greeks = greeks_calculator.calculate(
                product, env, engine, greeks=greek_list
            )
            pv[i, j] = float(greeks["price"])
            delta[i, j] = float(greeks["delta"])
            rhoq[i, j] = float(greeks["dividend_rho"])
            v_sq[i, j] = float(greeks["delta_q"])
    return pv, delta, rhoq, v_sq


def _compute_pv_surface_sv(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    spot_grid: np.ndarray,
    vol_grid: np.ndarray,
    q: float,
    base_vol: float,
    base_div_yield: DividendYield,
    base_q: float,
) -> np.ndarray:
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for vol surfaces.")
    if base_vol <= 0:
        raise ValidationError(f"base_vol must be positive, got {base_vol}")
    pv = np.zeros((spot_grid.size, vol_grid.size), dtype=float)
    shift = float(q) - float(base_q)
    div_yield = ShiftedDividendYield(base=base_div_yield, shift=shift)
    for j, vol in enumerate(vol_grid):
        scale = float(vol) / float(base_vol)
        vol_surface = ScaledVolSurface(base=pricing_env.vol_surface, scale=scale)
        for i, s in enumerate(spot_grid):
            env = _clone_env(
                pricing_env,
                spot=float(s),
                vol_surface=vol_surface,
                div_yield=div_yield,
            )
            pv[i, j] = float(engine.price(product, env))
    return pv


def _compute_point_surfaces_sv(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    greeks_calculator: GreeksCalculator,
    spot_grid: np.ndarray,
    vol_grid: np.ndarray,
    q: float,
    base_vol: float,
    base_div_yield: DividendYield,
    base_q: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for vol surfaces.")
    if base_vol <= 0:
        raise ValidationError(f"base_vol must be positive, got {base_vol}")
    pv = np.zeros((spot_grid.size, vol_grid.size), dtype=float)
    rhoq = np.zeros_like(pv)
    vanna = np.zeros_like(pv)
    volga = np.zeros_like(pv)
    shift = float(q) - float(base_q)
    div_yield = _shift_dividend_yield(base_div_yield, shift)
    greek_list = [
        EquityGreek.PRICE,
        EquityGreek.DIVIDEND_RHO,
        EquityGreek.VANNA,
        EquityGreek.VOLGA,
    ]
    for j, vol in enumerate(vol_grid):
        scale = float(vol) / float(base_vol)
        vol_surface = _scale_vol_surface(pricing_env.vol_surface, scale)
        for i, s in enumerate(spot_grid):
            env = _clone_env(
                pricing_env,
                spot=float(s),
                vol_surface=vol_surface,
                div_yield=div_yield,
            )
            greeks = greeks_calculator.calculate(
                product, env, engine, greeks=greek_list
            )
            pv[i, j] = float(greeks["price"])
            rhoq[i, j] = float(greeks["dividend_rho"])
            vanna[i, j] = float(greeks["vanna"])
            volga[i, j] = float(greeks["volga"])
    return pv, rhoq, vanna, volga


def _compute_delta_surface_sq(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    greeks_calculator: GreeksCalculator,
    spot_grid: np.ndarray,
    q_grid: np.ndarray,
    base_div_yield: DividendYield,
    base_q: float,
) -> np.ndarray:
    delta = np.zeros((spot_grid.size, q_grid.size), dtype=float)
    greek_list = [EquityGreek.DELTA]
    for j, q in enumerate(q_grid):
        shift = float(q) - float(base_q)
        div_yield = _shift_dividend_yield(base_div_yield, shift)
        for i, s in enumerate(spot_grid):
            env = _clone_env(pricing_env, spot=float(s), div_yield=div_yield)
            greeks = greeks_calculator.calculate(
                product, env, engine, greeks=greek_list
            )
            delta[i, j] = float(greeks["delta"])
    return delta


def _render_event_stats_table(stats: RiskNeutralSnowballEventStats) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ko_time": stats.ko_times,
            "p_ko": stats.ko_prob,
            "p_survive": stats.survive_prob,
            "ed_ko_cf": stats.expected_discounted_ko_cf,
        }
    )
    return df


def _render_engine_event_stats_table(stats: AutocallableEventStats) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ko_time": stats.ko_times,
            "p_ko": stats.ko_probability,
            "p_survive": stats.survival_probability,
            "ed_ko_cf": stats.expected_discounted_ko_cashflow,
        }
    )
    return df


def _render_conditional_cashflow_table(
    ko_times: Sequence[float],
    ko_prob: Sequence[float],
    expected_discounted_ko_cf: Sequence[float],
) -> pd.DataFrame:
    ko_prob_arr = np.asarray(ko_prob, dtype=float)
    expected_cf = np.asarray(expected_discounted_ko_cf, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional_cf = np.where(ko_prob_arr > 0, expected_cf / ko_prob_arr, np.nan)
    return pd.DataFrame(
        {
            "ko_time": np.asarray(ko_times, dtype=float),
            "p_ko": ko_prob_arr,
            "ed_ko_cf": expected_cf,
            "conditional_ed_ko_cf": conditional_cf,
        }
    )


def _format_shock_label(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.0%}"


def _extract_stress(scenario: Scenario, parameters: set[str]):
    for stress in scenario.stresses:
        if stress.parameter.lower() in parameters:
            return stress
    return None


def _format_stress_value(stress) -> str:
    if stress is None:
        return "n/a"
    if stress.stress_type == StressType.PERCENTAGE:
        return f"{stress.stress_value:+.0%}"
    if stress.stress_type == StressType.ABSOLUTE:
        return f"{stress.stress_value:+.4f}"
    return f"{stress.stress_value:.4f}"


def _coerce_skew_smile_shock(value: Optional[object]) -> Optional[SkewSmileShock]:
    if value is None:
        return None
    if isinstance(value, SkewSmileShock):
        return value
    if isinstance(value, dict):
        return SkewSmileShock(
            skew=float(value.get("skew", 0.0)),
            smile=float(value.get("smile", 0.0)),
        )
    raise ValidationError("skew_smile_shock must be SkewSmileShock or dict with skew/smile.")


def _build_simple_scenario(
    *,
    name: str,
    spot_shock: float,
    vol_shock: float,
    q_shift: float,
    base_q: Optional[float] = None,
) -> Scenario:
    if base_q is not None and q_shift < 0.0:
        q_shift = max(q_shift, -float(base_q) + 1e-6)
    return (
        ScenarioBuilder()
        .name(name)
        .description("Autocallable report stress scenario")
        .spot_stress(spot_shock)
        .vol_stress(vol_shock)
        .div_yield_stress(q_shift, stress_type=StressType.ABSOLUTE)
        .build()
    )


def _coerce_stress_scenarios(
    value: Optional[object], *, base_q: Optional[float] = None
) -> Sequence[Scenario]:
    if value is None:
        return []
    if isinstance(value, list):
        scenarios: list[Scenario] = []
        for item in value:
            if isinstance(item, Scenario):
                scenarios.append(item)
            elif isinstance(item, StressScenario):
                scenarios.append(
                    _build_simple_scenario(
                        name=item.name,
                        spot_shock=item.spot_shock,
                        vol_shock=item.vol_shock,
                        q_shift=item.q_shift,
                        base_q=base_q,
                    )
                )
            elif isinstance(item, dict):
                if "stresses" in item:
                    builder = ScenarioBuilder().name(
                        str(item.get("name", "Scenario"))
                    ).description(str(item.get("description", "")))
                    for stress in item.get("stresses", []):
                        stress_type_raw = stress.get("stress_type", "percentage")
                        if isinstance(stress_type_raw, StressType):
                            stress_type = stress_type_raw
                        else:
                            stress_type = StressType(str(stress_type_raw).lower())
                        builder = builder.add_stress(
                            parameter=str(stress.get("parameter", "")),
                            stress_value=float(stress.get("stress_value", 0.0)),
                            stress_type=stress_type,
                            description=stress.get("description"),
                        )
                    scenarios.append(builder.build())
                else:
                    scenarios.append(
                        _build_simple_scenario(
                            name=str(item.get("name", "Scenario")),
                            spot_shock=float(item.get("spot_shock", 0.0)),
                            vol_shock=float(item.get("vol_shock", 0.0)),
                            q_shift=float(item.get("q_shift", 0.0)),
                            base_q=base_q,
                        )
                    )
            else:
                raise ValidationError("stress_scenarios entries must be dict or Scenario.")
        return scenarios
    raise ValidationError("stress_scenarios must be a list of dicts or Scenario.")


def _first_barrier_value(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, list):
        return float(value[0]) if value else None
    return float(value)


def _next_barrier_snapshot(
    *,
    barriers: Sequence[Optional[float]],
    observation_times: Sequence[Optional[float]],
    fallback_barrier: Optional[float],
    fallback_time: float,
) -> Tuple[Optional[float], Optional[float]]:
    times = [float(t) for t in observation_times if t is not None and t > 0.0]
    barrier = next((float(b) for b in barriers if b is not None), fallback_barrier)
    if times:
        return barrier, times[0]
    return barrier, fallback_time


def _barrier_distance_metrics(
    *,
    spot: float,
    barrier: Optional[float],
    time_to_barrier: Optional[float],
    pricing_env: PricingEnvironment,
    product: SnowballOption,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if barrier is None or time_to_barrier is None or time_to_barrier <= 0.0:
        return None, None, None
    vol = float(pricing_env.get_vol(product.strike, time_to_barrier))
    pct_distance = safe_divide(barrier, spot, fallback=0.0) - 1.0
    denom = vol * safe_sqrt(time_to_barrier)
    sigma_distance = safe_divide(safe_log(barrier / spot), denom, fallback=0.0)
    return barrier, pct_distance, sigma_distance


def _extract_barrier_profile(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    barrier_type: str,
) -> Tuple[Sequence[Optional[float]], Sequence[Optional[float]]]:
    try:
        if barrier_type == "ko":
            profile = product.get_ko_observation_profile(pricing_env)
        else:
            profile = product.get_ki_observation_profile(pricing_env)
        return (
            profile.get("barriers", []),
            profile.get("observation_times", []),
        )
    except Exception:
        config = product.barrier_config
        if barrier_type == "ko":
            times = config.ko_observation_dates or []
            if isinstance(config.ko_barrier, list):
                barriers = config.ko_barrier
            else:
                barriers = [config.ko_barrier] if config.ko_barrier is not None else []
                if times:
                    barriers = barriers * len(times)
            return barriers, times
        times = config.ki_observation_dates or []
        if isinstance(config.ki_barrier, list):
            barriers = config.ki_barrier
        else:
            barriers = [config.ki_barrier] if config.ki_barrier is not None else []
            if times:
                barriers = barriers * len(times)
        return barriers, times


def _format_barrier_watch(
    label: str,
    barrier: Optional[float],
    time_to_barrier: Optional[float],
    pct_distance: Optional[float],
    sigma_distance: Optional[float],
) -> str:
    if barrier is None or time_to_barrier is None or pct_distance is None or sigma_distance is None:
        return f"- {label}: n/a"
    return (
        f"- {label}: level={barrier:.6f}, T={time_to_barrier:.3f}y, "
        f"dist={pct_distance:.2%}, sigma={sigma_distance:.3f}"
    )


def _nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _is_post_ki_state(spot: float, ki_level: Optional[float], is_reverse: bool) -> bool:
    if ki_level is None:
        return False
    if is_reverse:
        return spot >= ki_level
    return spot <= ki_level


def _compute_scenario_ladder(
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    spot_shocks: Sequence[float],
    vol_shocks: Sequence[float],
) -> Tuple[pd.DataFrame, float, Tuple[float, float]]:
    base_spot = pricing_env.spot
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for scenario ladder.")
    base_pv = float(engine.price(product, pricing_env))

    rows = [_format_shock_label(s) for s in spot_shocks]
    cols = [_format_shock_label(v) for v in vol_shocks]
    ladder = pd.DataFrame(index=rows, columns=cols, dtype=float)

    worst_pnl = 0.0
    worst_cell = (0.0, 0.0)
    first = True

    for s_shock in spot_shocks:
        for v_shock in vol_shocks:
            spot = base_spot * (1.0 + s_shock)
            vol_surface = ScaledVolSurface(
                base=pricing_env.vol_surface, scale=1.0 + float(v_shock)
            )
            env = _clone_env(pricing_env, spot=spot, vol_surface=vol_surface)
            pv = float(engine.price(product, env))
            pnl = pv - base_pv
            ladder.loc[_format_shock_label(s_shock), _format_shock_label(v_shock)] = pnl
            if first or pnl < worst_pnl:
                worst_pnl = pnl
                worst_cell = (s_shock, v_shock)
                first = False

    return ladder, float(worst_pnl), worst_cell


def _compute_stress_scenarios(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_pv: float,
    scenarios: Sequence[Scenario],
) -> pd.DataFrame:
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for stress scenarios.")

    portfolio = Portfolio(
        portfolio_name="autocallable-report",
        pricing_environments={"UNDERLYING": pricing_env},
    )
    portfolio.add_position(
        product=product,
        quantity=1.0,
        entry_price=base_pv,
        underlying="UNDERLYING",
        engine=engine,
        entry_timestamp=pricing_env.valuation_date,
    )
    stress_engine = EquityStressEngine(
        config=EquityStressConfig(calculate_greeks=False, export_formats=[])
    )
    results = stress_engine.run_static_scenarios(portfolio, scenarios)

    rows = []
    for scenario_result in results.scenario_results:
        scenario = scenario_result.scenario
        spot_stress = _format_stress_value(_extract_stress(scenario, {"spot"}))
        vol_stress = _format_stress_value(
            _extract_stress(scenario, {"volatility", "vol"})
        )
        div_stress = _format_stress_value(
            _extract_stress(scenario, {"dividend_yield", "dividend", "div_yield"})
        )
        rows.append(
            {
                "scenario": scenario.name,
                "spot_stress": spot_stress,
                "vol_stress": vol_stress,
                "div_yield_stress": div_stress,
                "pnl": scenario_result.portfolio_pnl,
            }
        )
    return pd.DataFrame(rows)


def _compute_barrier_zoom_surfaces(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    barrier_level: float,
    vol_grid: np.ndarray,
    base_vol: float,
    base_div_yield: DividendYield,
    spot_nodes: int = 21,
    band_width: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pricing_env.vol_surface is None:
        raise ValidationError("vol_surface is required for barrier-zoom surfaces.")
    if base_vol <= 0:
        raise ValidationError(f"base_vol must be positive, got {base_vol}")
    spot_min = barrier_level * (1.0 - band_width)
    spot_max = barrier_level * (1.0 + band_width)
    spot_grid = np.linspace(spot_min, spot_max, spot_nodes)
    pv = np.zeros((spot_grid.size, vol_grid.size), dtype=float)
    for j, vol in enumerate(vol_grid):
        scale = float(vol) / float(base_vol)
        vol_surface = ScaledVolSurface(base=pricing_env.vol_surface, scale=scale)
        for i, spot in enumerate(spot_grid):
            env = _clone_env(
                pricing_env,
                spot=float(spot),
                vol_surface=vol_surface,
                div_yield=base_div_yield,
            )
            pv[i, j] = float(engine.price(product, env))

    delta = derivative_1d(pv, spot_grid, axis=0)
    gamma = derivative_1d(delta, spot_grid, axis=0)
    vega = derivative_1d(pv, vol_grid, axis=1)
    return spot_grid, vol_grid, gamma, vega


def _compute_vanna_volga(
    pv_sv: np.ndarray, spot_grid: np.ndarray, vol_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    delta = derivative_1d(pv_sv, spot_grid, axis=0)
    vega = derivative_1d(pv_sv, vol_grid, axis=1)
    vanna = derivative_1d(delta, vol_grid, axis=1)
    volga = derivative_1d(vega, vol_grid, axis=1)
    return vanna, volga


def _bucket_label(bucket: TenorBucket) -> str:
    return f"{bucket.label} ({bucket.start:.3g}-{bucket.end:.3g}y)"


def _compute_bucketed_greeks(
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    *,
    vol_bump: float = 0.01,
    div_bump: float = 1e-4,
) -> pd.DataFrame:
    maturity = product.get_maturity(pricing_env)
    buckets = default_tenor_buckets(maturity)
    base_pv = float(engine.price(product, pricing_env))

    rows = []
    for bucket in buckets:
        # Bucketed Vega
        if pricing_env.vol_surface is None:
            raise ValidationError("vol_surface is required for bucketed vega.")
        vol_surface = BucketedVolSurface(
            base=pricing_env.vol_surface,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=vol_bump,
        )
        env_vol = _clone_env(pricing_env, vol_surface=vol_surface)
        pv_vol = float(engine.price(product, env_vol))
        vega_bucket = (pv_vol - base_pv) / vol_bump

        # Bucketed Dividend Rho
        if pricing_env.div_yield is None:
            base_div = ContinuousDividendYield(div_yield=0.0)
        else:
            base_div = pricing_env.div_yield
        div_yield = BucketedDividendYield(
            base=base_div,
            bucket_start=bucket.start,
            bucket_end=bucket.end,
            bump=div_bump,
        )
        env_div = _clone_env(pricing_env, div_yield=div_yield)
        pv_div = float(engine.price(product, env_div))
        rho_q = (pv_div - base_pv) * (0.01 / div_bump)
        rho_b = -rho_q

        rows.append(
            {
                "bucket": _bucket_label(bucket),
                "bucket_vega": vega_bucket,
                "bucket_rho_q": rho_q,
                "bucket_rho_b": rho_b,
            }
        )

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".6g")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _render_pnl_summary(title: str, dist: ShockPnLDistribution) -> str:
    s = dist.summary()
    if s.get("count", 0.0) == 0.0:
        return f"### {title}\n\nNo data.\n"
    return (
        f"### {title}\n\n"
        f"- count: {int(s['count'])}\n"
        f"- mean: {s['mean']:.6f}\n"
        f"- std: {s['std']:.6f}\n"
        f"- p01/p05/p50/p95/p99: {s['p01']:.6f}, {s['p05']:.6f}, {s['p50']:.6f}, {s['p95']:.6f}, {s['p99']:.6f}\n"
    )


def generate_snowball_risk_report(
    *,
    product: SnowballOption,
    pricing_env: PricingEnvironment,
    output_dir: Path,
    grid_spec: Optional[GridSpec] = None,
    engine_preference: Sequence[str] = ("quad", "pde", "mc"),
    quad_params: Optional[QuadParams] = None,
    pde_params: Optional[PDEParams] = None,
    mc_params: Optional[MCParams] = None,
    analyzer_mc_params: Optional[MCParams] = None,
    historical_spot: Optional[Sequence[float]] = None,
    historical_q: Optional[Sequence[float]] = None,
    skew_smile_shock: Optional[SkewSmileShock] = None,
    stress_scenarios: Optional[Sequence[object]] = None,
    high_accuracy_surfaces: bool = False,
) -> ReportResult:
    if grid_spec is None:
        grid_spec = GridSpec()
    skew_smile_shock = _coerce_skew_smile_shock(skew_smile_shock)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    maturity = product.get_maturity(pricing_env)
    if maturity < 0:
        raise ValidationError(f"Product maturity must be non-negative, got {maturity}")

    base_spot = pricing_env.spot
    base_q = pricing_env.get_div_yield(maturity)
    base_vol = pricing_env.get_vol(product.strike, maturity)
    if pricing_env.div_yield is None:
        base_div_yield: DividendYield = ContinuousDividendYield(div_yield=0.0)
    else:
        base_div_yield = pricing_env.div_yield
    if stress_scenarios is None or len(stress_scenarios) == 0:
        stress_scenarios = _coerce_stress_scenarios(
            [
                {
                    "name": "Black Monday",
                    "spot_shock": -0.20,
                    "vol_shock": 0.50,
                    "q_shift": -0.02,
                },
                {
                    "name": "Slow Bleed (1M)",
                    "spot_shock": -0.05,
                    "vol_shock": -0.10,
                    "q_shift": 0.0,
                },
            ],
            base_q=base_q,
        )
    else:
        stress_scenarios = _coerce_stress_scenarios(stress_scenarios, base_q=base_q)

    ko_barriers, ko_times = _extract_barrier_profile(
        product=product, pricing_env=pricing_env, barrier_type="ko"
    )
    ki_barriers, ki_times = _extract_barrier_profile(
        product=product, pricing_env=pricing_env, barrier_type="ki"
    )
    ko_barrier, ko_time = _next_barrier_snapshot(
        barriers=ko_barriers,
        observation_times=ko_times,
        fallback_barrier=_first_barrier_value(product.barrier_config.ko_barrier),
        fallback_time=maturity,
    )
    ki_barrier, ki_time = _next_barrier_snapshot(
        barriers=ki_barriers,
        observation_times=ki_times,
        fallback_barrier=_first_barrier_value(product.barrier_config.ki_barrier),
        fallback_time=maturity,
    )
    ko_level, ko_pct, ko_sigma = _barrier_distance_metrics(
        spot=base_spot,
        barrier=ko_barrier,
        time_to_barrier=ko_time,
        pricing_env=pricing_env,
        product=product,
    )
    ki_level, ki_pct, ki_sigma = _barrier_distance_metrics(
        spot=base_spot,
        barrier=ki_barrier,
        time_to_barrier=ki_time,
        pricing_env=pricing_env,
        product=product,
    )
    barrier_watch = "Barrier Watch:\n" + "\n".join(
        [
            _format_barrier_watch("Next KO", ko_level, ko_time, ko_pct, ko_sigma),
            _format_barrier_watch("Next KI", ki_level, ki_time, ki_pct, ki_sigma),
        ]
    )

    engine, engine_used = _select_snowball_pricing_engine(
        preference=engine_preference,
        quad_params=quad_params,
        pde_params=pde_params,
        mc_params=mc_params,
    )
    base_pv = float(engine.price(product, pricing_env))
    skew_smile_pnl = None
    if skew_smile_shock is not None and (
        skew_smile_shock.skew != 0.0 or skew_smile_shock.smile != 0.0
    ):
        if pricing_env.vol_surface is None:
            raise ValidationError("vol_surface is required for skew/smile shocks.")
        skewed_surface = SkewSmileVolSurface(
            base=pricing_env.vol_surface,
            skew=skew_smile_shock.skew,
            smile=skew_smile_shock.smile,
        )
        skewed_env = _clone_env(pricing_env, vol_surface=skewed_surface)
        skew_smile_pnl = float(engine.price(product, skewed_env)) - base_pv

    spot_grid = build_spot_grid(base_spot, grid_spec)
    q_grid = build_q_grid(base_q, grid_spec)
    vol_grid = build_vol_grid(base_vol, grid_spec)

    greeks_calculator = (
        GreeksCalculator(params=engine.params) if high_accuracy_surfaces else None
    )

    if greeks_calculator is None:
        pv_sq = _compute_pv_surface_sq(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            spot_grid=spot_grid,
            q_grid=q_grid,
            base_div_yield=base_div_yield,
            base_q=base_q,
        )
        pv_sv = _compute_pv_surface_sv(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            spot_grid=spot_grid,
            vol_grid=vol_grid,
            q=base_q,
            base_vol=base_vol,
            base_div_yield=base_div_yield,
            base_q=base_q,
        )
        pv_sv_q_up = _compute_pv_surface_sv(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            spot_grid=spot_grid,
            vol_grid=vol_grid,
            q=base_q + grid_spec.q_bump_for_rho,
            base_vol=base_vol,
            base_div_yield=base_div_yield,
            base_q=base_q,
        )
        surfaces = compute_surfaces_from_pv(
            spot_grid=spot_grid,
            q_grid=q_grid,
            vol_grid=vol_grid,
            pv_sq=pv_sq,
            pv_sv=pv_sv,
            pv_sv_q_up=pv_sv_q_up,
            q_bump_for_rho=grid_spec.q_bump_for_rho,
        )
        vanna_sv, volga_sv = _compute_vanna_volga(pv_sv, spot_grid, vol_grid)
    else:
        pv_sq, delta_sq, rhoq_sq, v_sq = _compute_point_surfaces_sq(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            greeks_calculator=greeks_calculator,
            spot_grid=spot_grid,
            q_grid=q_grid,
            base_div_yield=base_div_yield,
            base_q=base_q,
        )
        pv_sv, rhoq_sv, vanna_sv, volga_sv = _compute_point_surfaces_sv(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            greeks_calculator=greeks_calculator,
            spot_grid=spot_grid,
            vol_grid=vol_grid,
            q=base_q,
            base_vol=base_vol,
            base_div_yield=base_div_yield,
            base_q=base_q,
        )
        surfaces = SurfaceSet(
            spot_grid=spot_grid,
            q_grid=q_grid,
            vol_grid=vol_grid,
            pv_sq=pv_sq,
            delta_sq=delta_sq,
            rhoq_sq=rhoq_sq,
            v_sq=v_sq,
            pv_sv=pv_sv,
            rhoq_sv=rhoq_sv,
        )
    gamma_sq = derivative_1d(surfaces.delta_sq, spot_grid, axis=0)

    charm_sq = None
    color_sq = None
    theta_base = None
    time_bump = float(grid_spec.time_bump_years)
    if time_bump > 0.0:
        bumped_date = pricing_env.valuation_date + timedelta(days=int(time_bump * 365))
        bumped_env = _clone_env(pricing_env, valuation_date=bumped_date)
        bumped_product = copy.deepcopy(product)
        dropped_all = bumped_product.time_shift(time_bump, bumped_date, bumped_env)
        if not dropped_all:
            bumped_maturity = bumped_product.get_maturity(bumped_env)
            if bumped_maturity > 0.0:
                pv_bumped = float(engine.price(bumped_product, bumped_env))
                theta_base = (pv_bumped - base_pv) / time_bump
                if bumped_env.div_yield is None:
                    bumped_div_yield: DividendYield = ContinuousDividendYield(div_yield=0.0)
                else:
                    bumped_div_yield = bumped_env.div_yield
                bumped_q = bumped_env.get_div_yield(bumped_maturity)
                if greeks_calculator is None:
                    pv_sq_bumped = _compute_pv_surface_sq(
                        product=bumped_product,
                        pricing_env=bumped_env,
                        engine=engine,
                        spot_grid=spot_grid,
                        q_grid=q_grid,
                        base_div_yield=bumped_div_yield,
                        base_q=bumped_q,
                    )
                    delta_bumped = derivative_1d(pv_sq_bumped, spot_grid, axis=0)
                else:
                    delta_bumped = _compute_delta_surface_sq(
                        product=bumped_product,
                        pricing_env=bumped_env,
                        engine=engine,
                        greeks_calculator=greeks_calculator,
                        spot_grid=spot_grid,
                        q_grid=q_grid,
                        base_div_yield=bumped_div_yield,
                        base_q=bumped_q,
                    )
                gamma_bumped = derivative_1d(delta_bumped, spot_grid, axis=0)
                charm_sq = (delta_bumped - surfaces.delta_sq) / time_bump
                color_sq = (gamma_bumped - gamma_sq) / time_bump

    vega_sv = derivative_1d(pv_sv, vol_grid, axis=1)
    spot_idx = _nearest_index(spot_grid, base_spot)
    q_idx = _nearest_index(q_grid, base_q)
    vol_idx = _nearest_index(vol_grid, base_vol)
    if greeks_calculator is None:
        base_delta = float(surfaces.delta_sq[spot_idx, q_idx])
        base_gamma = float(gamma_sq[spot_idx, q_idx])
        base_vega = float(vega_sv[spot_idx, vol_idx])
    else:
        base_greeks = greeks_calculator.calculate(
            product,
            pricing_env,
            engine,
            greeks=[EquityGreek.DELTA, EquityGreek.GAMMA, EquityGreek.VEGA],
        )
        base_delta = float(base_greeks["delta"])
        base_gamma = float(base_greeks["gamma"])
        base_vega = float(base_greeks["vega"])
    status = "Safe Zone"
    if ki_sigma is not None and abs(ki_sigma) < 0.5:
        status = "Knock-In Danger Zone"
    elif ko_sigma is not None and abs(ko_sigma) < 0.5:
        status = "Knock-Out Likely"

    barrier_zoom_lines = []
    if ko_level is not None:
        ko_spot_grid, ko_vol_grid, ko_gamma, ko_vega = _compute_barrier_zoom_surfaces(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            barrier_level=ko_level,
            vol_grid=vol_grid,
            base_vol=base_vol,
            base_div_yield=base_div_yield,
        )
        save_heatmap(
            x=ko_spot_grid,
            y=ko_vol_grid,
            z=ko_gamma,
            title="KO Barrier Zoom Gamma vs Spot×Vol",
            xlabel="Volatility σ",
            ylabel="Spot",
            colorbar_label="Gamma",
            path=plots_dir / "barrier_ko_gamma.png",
        )
        save_heatmap(
            x=ko_spot_grid,
            y=ko_vol_grid,
            z=ko_vega,
            title="KO Barrier Zoom Vega vs Spot×Vol",
            xlabel="Volatility σ",
            ylabel="Spot",
            colorbar_label="Vega",
            path=plots_dir / "barrier_ko_vega.png",
        )
        barrier_zoom_lines.append("- KO barrier: `plots/barrier_ko_gamma.png`, `plots/barrier_ko_vega.png`")

    if ki_level is not None:
        ki_spot_grid, ki_vol_grid, ki_gamma, ki_vega = _compute_barrier_zoom_surfaces(
            product=product,
            pricing_env=pricing_env,
            engine=engine,
            barrier_level=ki_level,
            vol_grid=vol_grid,
            base_vol=base_vol,
            base_div_yield=base_div_yield,
        )
        save_heatmap(
            x=ki_spot_grid,
            y=ki_vol_grid,
            z=ki_gamma,
            title="KI Barrier Zoom Gamma vs Spot×Vol",
            xlabel="Volatility σ",
            ylabel="Spot",
            colorbar_label="Gamma",
            path=plots_dir / "barrier_ki_gamma.png",
        )
        save_heatmap(
            x=ki_spot_grid,
            y=ki_vol_grid,
            z=ki_vega,
            title="KI Barrier Zoom Vega vs Spot×Vol",
            xlabel="Volatility σ",
            ylabel="Spot",
            colorbar_label="Vega",
            path=plots_dir / "barrier_ki_vega.png",
        )
        barrier_zoom_lines.append("- KI barrier: `plots/barrier_ki_gamma.png`, `plots/barrier_ki_vega.png`")

    barrier_zoom_section = "No barrier zoom plots generated."
    if barrier_zoom_lines:
        barrier_zoom_section = "\n".join(barrier_zoom_lines)

    # Scenario ladder
    ladder_spot_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10]
    ladder_vol_shocks = [-0.05, 0.0, 0.05]
    ladder_df, ladder_worst_pnl, ladder_worst_cell = _compute_scenario_ladder(
        product,
        pricing_env,
        engine,
        ladder_spot_shocks,
        ladder_vol_shocks,
    )
    stress_df = _compute_stress_scenarios(
        product=product,
        pricing_env=pricing_env,
        engine=engine,
        base_pv=base_pv,
        scenarios=stress_scenarios,
    )

    # Bucketed Greeks
    bucketed_df = _compute_bucketed_greeks(product, pricing_env, engine)

    # Risk-neutral event stats / cashflow attribution:
    # Prefer engine-provided stats (future QUAD/PDE implementations), else use MC analyzer.
    engine_stats = engine.calculate_event_stats(product, pricing_env)
    rn_stats: Optional[RiskNeutralSnowballEventStats] = None
    analyzer: Optional[AutocallablePathAnalyzer] = None
    if engine_stats is not None:
        event_df = _render_engine_event_stats_table(engine_stats)
    else:
        if analyzer_mc_params is None:
            analyzer_mc_params = MCParams(num_paths=20000, time_steps=252)
        analyzer = AutocallablePathAnalyzer(
            mc_params=analyzer_mc_params, q_bump=grid_spec.q_bump_for_rho
        )
        rn_stats = analyzer.analyze_snowball_risk_neutral(product, pricing_env)
        event_df = _render_event_stats_table(rn_stats)

    if engine_stats is not None:
        prob_ki = float(engine_stats.ki_probability)
        prob_ko_total = float(np.sum(engine_stats.ko_probability))
        conditional_df = _render_conditional_cashflow_table(
            engine_stats.ko_times,
            engine_stats.ko_probability,
            engine_stats.expected_discounted_ko_cashflow,
        )
    else:
        assert rn_stats is not None
        prob_ki = float(rn_stats.ki_probability)
        prob_ko_total = float(np.sum(rn_stats.ko_prob))
        conditional_df = _render_conditional_cashflow_table(
            rn_stats.ko_times,
            rn_stats.ko_prob,
            rn_stats.expected_discounted_ko_cf,
        )

    post_ki = _is_post_ki_state(base_spot, ki_level, product.is_reverse)
    if post_ki:
        lifecycle_context = (
            "Post-KI state (spot beyond KI barrier). Focus on Delta/Gamma and recovery risk.\n"
            f"- Recovery probability (proxy: KO before maturity): {prob_ko_total:.6f}"
        )
    else:
        lifecycle_context = (
            "Pre-KI state. Focus on KO vs KI probabilities.\n"
            f"- P(KO before maturity): {prob_ko_total:.6f}\n"
            f"- P(KI before maturity): {prob_ki:.6f}"
        )

    # Historical replay (optional): apply shocks to today's state (no roll-down)
    hist_section = ""
    if historical_spot is not None and historical_q is not None:
        if analyzer is None:
            if analyzer_mc_params is None:
                analyzer_mc_params = MCParams(num_paths=20000, time_steps=252)
            analyzer = AutocallablePathAnalyzer(
                mc_params=analyzer_mc_params, q_bump=grid_spec.q_bump_for_rho
            )

        def price_from_shock(spot_mult: float, q_shift: float) -> float:
            div_yield = ShiftedDividendYield(base=base_div_yield, shift=q_shift)
            shocked_env = _clone_env(
                pricing_env, spot=base_spot * spot_mult, div_yield=div_yield
            )
            return float(engine.price(product, shocked_env))

        dist = analyzer.historical_shock_pnl(
            base_price=base_pv,
            price_fn=price_from_shock,
            spot_series=historical_spot,
            q_series=historical_q,
            horizon_steps=1,
        )
        hist_section = _render_pnl_summary("Historical Shock PnL (1-step)", dist)
    else:
        hist_section = "## Historical analysis\nNo historical series provided."

    # Plots
    save_line_plot(
        x=spot_grid,
        y=surfaces.pv_sq[:, int(len(q_grid) / 2)],
        title="PV vs Spot (q fixed at q0)",
        xlabel="Spot",
        ylabel="PV",
        path=plots_dir / "pv_vs_spot.png",
    )
    save_heatmap(
        x=spot_grid,
        y=q_grid,
        z=surfaces.rhoq_sq,
        title="DividendRho (per 1% q) vs Spot×Dividend",
        xlabel="Dividend yield q",
        ylabel="Spot",
        colorbar_label="RhoQ",
        path=plots_dir / "rhoq_spot_div.png",
    )
    save_heatmap(
        x=spot_grid,
        y=vol_grid,
        z=surfaces.rhoq_sv,
        title="DividendRho (per 1% q) vs Spot×Vol",
        xlabel="Volatility σ",
        ylabel="Spot",
        colorbar_label="RhoQ",
        path=plots_dir / "rhoq_spot_vol.png",
    )
    save_heatmap(
        x=spot_grid,
        y=q_grid,
        z=surfaces.delta_sq,
        title="Delta vs Spot×Dividend",
        xlabel="Dividend yield q",
        ylabel="Spot",
        colorbar_label="Delta",
        path=plots_dir / "delta_spot_div.png",
    )
    save_heatmap(
        x=spot_grid,
        y=q_grid,
        z=surfaces.v_sq,
        title="Cross Sensitivity ∂²V/(∂S∂q) vs Spot×Dividend",
        xlabel="Dividend yield q",
        ylabel="Spot",
        colorbar_label="V_Sq",
        path=plots_dir / "cross_s_q.png",
    )
    save_heatmap(
        x=spot_grid,
        y=vol_grid,
        z=vanna_sv,
        title="Vanna ∂²V/(∂S∂σ) vs Spot×Vol",
        xlabel="Volatility σ",
        ylabel="Spot",
        colorbar_label="Vanna",
        path=plots_dir / "vanna_spot_vol.png",
    )
    save_heatmap(
        x=spot_grid,
        y=vol_grid,
        z=volga_sv,
        title="Volga ∂²V/(∂σ²) vs Spot×Vol",
        xlabel="Volatility σ",
        ylabel="Spot",
        colorbar_label="Volga",
        path=plots_dir / "volga_spot_vol.png",
    )
    charm_color_section = "Charm/Color not computed."
    if charm_sq is not None and color_sq is not None:
        save_heatmap(
            x=spot_grid,
            y=q_grid,
            z=charm_sq,
            title="Charm ∂Δ/∂t vs Spot×Dividend",
            xlabel="Dividend yield q",
            ylabel="Spot",
            colorbar_label="Charm",
            path=plots_dir / "charm_spot_div.png",
        )
        save_heatmap(
            x=spot_grid,
            y=q_grid,
            z=color_sq,
            title="Color ∂Γ/∂t vs Spot×Dividend",
            xlabel="Dividend yield q",
            ylabel="Spot",
            colorbar_label="Color",
            path=plots_dir / "color_spot_div.png",
        )
        charm_color_section = "- `plots/charm_spot_div.png`, `plots/color_spot_div.png`"

    # Markdown report
    report_md = output_dir / "risk_report.md"
    event_table_md = _df_to_markdown(event_df)

    stress_passed = (
        (stress_df["pnl"] > -0.5 * base_pv).all() if not stress_df.empty else True
    )
    stress_status = "passes" if stress_passed else "fails"

    basis_rho_note = (
        "Basis mapping (China index futures): define carry/basis `b = r - q`, so `RhoB = ∂V/∂b = -∂V/∂q = -RhoQ`."
    )
    ladder_md = _df_to_markdown(ladder_df)
    stress_md = _df_to_markdown(stress_df)
    bucketed_md = _df_to_markdown(bucketed_df)
    conditional_md = _df_to_markdown(conditional_df)
    if skew_smile_shock is None or (
        skew_smile_shock.skew == 0.0 and skew_smile_shock.smile == 0.0
    ):
        skew_smile_block = "Skew/smile shock not configured."
    else:
        skew_smile_block = (
            "Skew/smile shock model: "
            "`vol = base + skew * ln(K/S) + smile * ln(K/S)^2`\n"
            f"- skew: {skew_smile_shock.skew:.6f}\n"
            f"- smile: {skew_smile_shock.smile:.6f}\n"
            f"- PV impact: {skew_smile_pnl:.6f}"
        )
    theta_line = f"{theta_base:.6f}" if theta_base is not None else "n/a"
    executive_block = (
        f"- Status: {status}\n"
        f"- PV: {base_pv:.6f}; Delta: {base_delta:.6f}; Gamma: {base_gamma:.6f}; "
        f"Vega: {base_vega:.6f}; Theta: {theta_line}\n"
        f"{barrier_watch}"
    )
    if engine_stats is not None:
        event_block = (
            f"- PV (engine event stats): {engine_stats.pv:.6f}\n"
            f"- KI probability: {engine_stats.ki_probability:.6f}\n"
            f"- PV reconciliation error (PV - sum(expected discounted cashflows)): {engine_stats.reconciliation_error:.6g}\n\n"
            f"{event_table_md}"
        )
    else:
        assert rn_stats is not None
        event_block = (
            f"- MC PV (from expected discounted cashflows): {rn_stats.pv_mc:.6f} ± {rn_stats.std_error:.6f} (SE), paths={rn_stats.num_paths}\n"
            f"- KI probability: {rn_stats.ki_probability:.6f}\n"
            f"- PV reconciliation error (MC cashflows - MC engine price): {rn_stats.reconciliation_error:.6g}\n\n"
            f"{event_table_md}"
        )
    surface_mode = "point-greeks" if greeks_calculator is not None else "finite-difference"
    content = f"""# Autocallable Risk Profile Report (Snowball)

**Generated**: {datetime.now().isoformat(timespec="seconds")}

## 1. Executive Dashboard (Traffic Light)
*Goal: Instant situational awareness for the trader/manager.*

{executive_block}

**Risk Interpretation**: [ADD INTERPRETATION: e.g., The product is currently in a {status.lower()}. Main risk is { "gamma-pinning near KO" if "KO" in status else "delta-buyback on KI" if "KI" in status else "theta decay" }.]

## 2. Product & Market Snapshot
- **Engine used**: `{engine_used}`
- **Pricing Mode**: {"point-greeks" if greeks_calculator is not None else "finite-difference"}
- **PV (Base)**: {base_pv:.6f}
- **Spot S0**: {base_spot:.6f}
- **Dividend q0**: {base_q:.6f}
- **Vol σ0**: {base_vol:.6f}

## 3. Barrier Risk (Zoom)
*Goal: Understand "cliff-edge" effects near barriers.*

- Barrier zoom grid: ±2% around KO/KI level, spot nodes=21
{barrier_zoom_section}

**Interpretation**: Near-barrier Gamma spikes indicate significant hedging costs if the underlying oscillates around the barrier level.

## 4. Required Surfaces & Greeks
*Goal: Visualize sensitivities across the spot/vol/dividend space.*

- Surface mode: {surface_mode}
- **Dividend / Basis Risk**: {basis_rho_note}
- **DividendRho surfaces**: `plots/rhoq_spot_div.png`, `plots/rhoq_spot_vol.png`
- **Delta & Cross-Gamma**: `plots/delta_spot_div.png`, `plots/cross_s_q.png`
- **Advanced Volatility Risk**: Vanna/Volga surfaces: `plots/vanna_spot_vol.png`, `plots/volga_spot_vol.png`
- {skew_smile_block}
- **Higher-Order Time Greeks**: {charm_color_section}

**Interpretation**: High Vanna suggests that your Delta hedge will become unstable if volatility spikes.

## 5. Scenario Ladder (Spot × Vol)
*Goal: PnL impact of discrete market moves.*

- Spot shocks: {', '.join(_format_shock_label(s) for s in ladder_spot_shocks)}
- Vol shocks: {', '.join(_format_shock_label(v) for v in ladder_vol_shocks)}
- **Worst Case**: {_format_shock_label(ladder_worst_cell[0])} spot × {_format_shock_label(ladder_worst_cell[1])} vol results in PnL of **{ladder_worst_pnl:.6f}**.

{ladder_md}

**Interpretation**: The portfolio's largest vulnerability is a correlated move in spot and vol.

## 6. Bucketed Greeks (Term Structure)
*Goal: Identify risk concentration along the maturity curve.*

- Bucketed Vega is per +1 vol point (0.01).
- Bucketed Dividend Rho is per +1% dividend yield (0.01); Basis Rho = -Dividend Rho.

{bucketed_md}

**Interpretation**: Risk is concentrated in the {bucketed_df.iloc[bucketed_df['bucket_vega'].abs().argmax()]['bucket'] if not bucketed_df.empty else "long-end"} bucket.

## 7. Risk-Neutral Event Stats & Cashflow
*Goal: Expected lifecycle and cashflow timing.*

### Lifecycle Context
{lifecycle_context}

{event_block}

### Conditional Cashflow Projection
*Expected payoff given that Knock-Out occurs at each specific observation date.*

{conditional_md}

## 8. Historical & Stress Analysis
*Goal: Resilience against extreme market shocks.*

### Stress Scenarios
{stress_md}

{hist_section}

**Interpretation**: Stress tests indicate the product {stress_status} basic resilience checks at a 50% equity loss threshold.
"""
    report_md.write_text(content, encoding="utf-8")

    return ReportResult(report_path=report_md, output_dir=output_dir)


def _load_input_module(path: Path):
    spec = importlib.util.spec_from_file_location("risk_report_input", str(path))
    if spec is None or spec.loader is None:
        raise ValidationError(f"Unable to load input module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[misc]
    return module


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Snowball autocallable risk profile report.")
    p.add_argument("--input", type=str, required=True, help="Path to a python file providing build_product/build_env.")
    p.add_argument("--out", type=str, default=".gemini/tmp/snowball_risk_report", help="Output directory for report artifacts.")
    p.add_argument("--engine", type=str, default="quad,pde,mc", help="Engine preference order, comma-separated.")
    p.add_argument("--paths", type=int, default=20000, help="MC paths for event stats.")
    p.add_argument("--steps", type=int, default=252, help="MC time steps for event stats.")
    p.add_argument("--fast", action="store_true", help="Use smaller grids for quicker runs.")
    p.add_argument("--spot-nodes", type=int, default=None, help="Override spot grid nodes.")
    p.add_argument("--q-nodes", type=int, default=None, help="Override dividend grid nodes.")
    p.add_argument("--vol-nodes", type=int, default=None, help="Override vol grid nodes.")
    p.add_argument(
        "--high-accuracy",
        action="store_true",
        help="Compute surfaces using per-node GreeksCalculator (slower, higher fidelity).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    module = _load_input_module(input_path)
    if not hasattr(module, "build_product") or not hasattr(module, "build_pricing_env"):
        raise ValidationError("Input module must define build_product() and build_pricing_env().")

    product = module.build_product()
    env = module.build_pricing_env()
    if not isinstance(product, SnowballOption):
        raise ValidationError(f"build_product() must return SnowballOption for now, got {type(product).__name__}")
    if not isinstance(env, PricingEnvironment):
        raise ValidationError(f"build_pricing_env() must return PricingEnvironment, got {type(env).__name__}")

    hist_spot = getattr(module, "historical_spot", None)
    hist_q = getattr(module, "historical_q", None)
    skew_smile = _coerce_skew_smile_shock(getattr(module, "skew_smile_shock", None))
    stress_scenarios = getattr(module, "stress_scenarios", None)
    high_accuracy_surfaces = bool(
        getattr(module, "high_accuracy_surfaces", False)
    ) or bool(args.high_accuracy)

    grid = GridSpec()
    if args.fast:
        grid = GridSpec(spot_nodes=11, q_nodes=11, vol_nodes=11)
    if args.spot_nodes is not None:
        grid = replace(grid, spot_nodes=int(args.spot_nodes))
    if args.q_nodes is not None:
        grid = replace(grid, q_nodes=int(args.q_nodes))
    if args.vol_nodes is not None:
        grid = replace(grid, vol_nodes=int(args.vol_nodes))

    analyzer_params = MCParams(num_paths=int(args.paths), time_steps=int(args.steps))
    generate_snowball_risk_report(
        product=product,
        pricing_env=env,
        output_dir=out_dir,
        grid_spec=grid,
        engine_preference=[e.strip() for e in str(args.engine).split(",") if e.strip()],
        analyzer_mc_params=analyzer_params,
        historical_spot=hist_spot,
        historical_q=hist_q,
        skew_smile_shock=skew_smile,
        stress_scenarios=stress_scenarios,
        high_accuracy_surfaces=high_accuracy_surfaces,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
