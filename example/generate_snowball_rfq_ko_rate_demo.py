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

import json
import sys
import csv
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

R_GRID = [0.01, 0.03, 0.05]
Q_GRID = [0.08, 0.115, 0.15]
VOL_GRID = [0.15, 0.25, 0.35]

DEFAULT_R = 0.03
DEFAULT_Q = 0.10
DEFAULT_VOL = 0.20
KO_RATE_BOUNDS = (0.0, 5.0)
AFFINE_KO_RATE_PAIR = (0.0, 2.0)
PREPAYMENT = 100.0
BASE_KO_BARRIER = 103.0
BASE_KI_BARRIER = 75.0
KO_BARRIER_BUMP = 1.0
KI_BARRIER_BUMP = 1.0
DEFAULT_VARIANT = "standard"
VARIANTS = {
    "standard": {
        "label": "Standard",
        "description": "Monthly 103 KO with daily 75 KI.",
    },
    "european_ki": {
        "label": "European KI",
        "description": "Monthly 103 KO with KI observed only at maturity.",
    },
    "parachute": {
        "label": "Parachute",
        "description": "Monthly 103 KO that drops to 75 on the final KO observation.",
    },
    "stepdown": {
        "label": "Stepdown",
        "description": "Monthly KO barrier steps down from 103 toward 75 over the 2Y life.",
    },
}


@dataclass(frozen=True)
class DemoMeta:
    generated_at: str
    engine: str
    solver_grid_size: int
    solver_time_steps: int
    structure: dict[str, Any]
    ranges: dict[str, list[float]]


def build_product(ko_rate: float, variant: str) -> SnowballOption:
    return build_product_with_barriers(
        ko_rate=ko_rate,
        variant=variant,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def build_product_with_barriers(
    ko_rate: float,
    variant: str,
    *,
    ko_barrier: float,
    ki_barrier: float,
) -> SnowballOption:
    common = {
        "initial_price": 100.0,
        "strike": 100.0,
        "maturity": 2.0,
        "contract_multiplier": 1.0,
        "ko_barrier": ko_barrier,
        "ko_rate": ko_rate,
        "ki_barrier": ki_barrier,
        "is_reverse": False,
        "rebate_rate": ko_rate,
        "include_principal": False,
    }
    stepdown_rate = (ko_barrier - ki_barrier) / (23 * 100.0)

    if variant == "standard":
        return create_standard_snowball(
            **common,
            num_observations=24,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=generate_ko_observation_dates(2.0, "daily"),
        )
    if variant == "european_ki":
        return create_european_ki_snowball(
            **common,
            num_ko_observations=24,
        )
    if variant == "parachute":
        return create_parachute_snowball(
            **common,
            num_observations=24,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=generate_ko_observation_dates(2.0, "daily"),
        )
    if variant == "stepdown":
        return create_stepdown_snowball(
            **common,
            num_observations=24,
            initial_ko_barrier=ko_barrier,
            stepdown_rate=stepdown_rate,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=generate_ko_observation_dates(2.0, "daily"),
        )

    raise ValueError(f"Unknown variant: {variant}")


def build_full_protected_product(variant: str):
    return build_full_protected_product_with_barriers(
        variant=variant,
        ko_barrier=BASE_KO_BARRIER,
        ki_barrier=BASE_KI_BARRIER,
    )


def build_full_protected_product_with_barriers(
    variant: str,
    *,
    ko_barrier: float,
    ki_barrier: float,
):
    if variant == "parachute":
        ko_barrier_value = [ko_barrier] * 23 + [ki_barrier]
    elif variant == "stepdown":
        stepdown_amount = (ko_barrier - ki_barrier) / 23
        ko_barrier_value = [ko_barrier - i * stepdown_amount for i in range(24)]
    else:
        ko_barrier_value = ko_barrier

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier_value,
            ko_rate=1.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=generate_ko_observation_dates(2.0, "monthly"),
            ki_barrier=None,
        ),
        payoff_config=PayoffConfig(
            rebate_rate=1.0,
            include_principal=False,
            protection_type=ProtectionType.FULL,
        ),
        accrual_config=AccrualConfig(is_annualized=False),
        maturity=2.0,
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


def solve_fair_ko_rate(
    rate: float,
    div_yield: float,
    vol: float,
    variant: str,
    *,
    ko_barrier: float = BASE_KO_BARRIER,
    ki_barrier: float = BASE_KI_BARRIER,
    pde_params: PDEParams,
) -> dict[str, float]:
    env = build_env(rate=rate, div_yield=div_yield, vol=vol)
    engine = SnowballPDESolver(params=pde_params)
    protected_pv = engine.price(
        build_full_protected_product_with_barriers(
            variant=variant,
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
            ko_barrier=ko_barrier,
            ki_barrier=ki_barrier,
        ),
        env,
    )
    p1 = engine.price(
        build_product_with_barriers(
            ko_rate=k1,
            variant=variant,
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


def build_cube(
    *, pde_params: PDEParams
) -> tuple[dict[str, dict[str, list[list[list[float | None]]]]], list[dict[str, Any]]]:
    variant_cubes: dict[str, dict[str, list[list[list[float | None]]]]] = {}
    scenario_rows: list[dict[str, Any]] = []
    bump_params = PDEParams(
        grid_size=max(50, pde_params.grid_size // 2),
        time_steps=max(80, pde_params.time_steps // 2),
    )
    total = len(VARIANTS) * len(R_GRID) * len(Q_GRID) * len(VOL_GRID)
    done = 0
    for variant in VARIANTS:
        quote_cube: list[list[list[float | None]]] = []
        interest_cube: list[list[list[float | None]]] = []
        protected_cube: list[list[list[float | None]]] = []
        target_cube: list[list[list[float | None]]] = []
        quote_ko_sens_cube: list[list[list[float | None]]] = []
        quote_ki_sens_cube: list[list[list[float | None]]] = []
        interest_ko_sens_cube: list[list[list[float | None]]] = []
        interest_ki_sens_cube: list[list[list[float | None]]] = []
        for rate in R_GRID:
            quote_q_slice: list[list[float | None]] = []
            interest_q_slice: list[list[float | None]] = []
            protected_q_slice: list[list[float | None]] = []
            target_q_slice: list[list[float | None]] = []
            quote_ko_sens_q_slice: list[list[float | None]] = []
            quote_ki_sens_q_slice: list[list[float | None]] = []
            interest_ko_sens_q_slice: list[list[float | None]] = []
            interest_ki_sens_q_slice: list[list[float | None]] = []
            for div_yield in Q_GRID:
                quote_vol_slice: list[float | None] = []
                interest_vol_slice: list[float | None] = []
                protected_vol_slice: list[float | None] = []
                target_vol_slice: list[float | None] = []
                quote_ko_sens_vol_slice: list[float | None] = []
                quote_ki_sens_vol_slice: list[float | None] = []
                interest_ko_sens_vol_slice: list[float | None] = []
                interest_ki_sens_vol_slice: list[float | None] = []
                for vol in VOL_GRID:
                    done += 1
                    print(
                        f"[{done:02d}/{total}] {variant} fair ko_rate for "
                        f"r={rate:.4f}, q={div_yield:.4f}, vol={vol:.4f}"
                    )
                    try:
                        result = solve_fair_ko_rate(
                            rate=rate,
                            div_yield=div_yield,
                            vol=vol,
                            variant=variant,
                            ko_barrier=BASE_KO_BARRIER,
                            ki_barrier=BASE_KI_BARRIER,
                            pde_params=pde_params,
                        )
                        quote_vol_slice.append(result["quoted_ko_rate"])
                        interest_vol_slice.append(result["interest_component_pv"])
                        protected_vol_slice.append(result["protected_snowball_pv"])
                        target_vol_slice.append(result["snowball_target_pv"])
                        if variant == DEFAULT_VARIANT:
                            try:
                                ko_up = solve_fair_ko_rate(
                                    rate=rate,
                                    div_yield=div_yield,
                                    vol=vol,
                                    variant=variant,
                                    ko_barrier=BASE_KO_BARRIER + KO_BARRIER_BUMP,
                                    ki_barrier=BASE_KI_BARRIER,
                                    pde_params=bump_params,
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
                                ki_up = solve_fair_ko_rate(
                                    rate=rate,
                                    div_yield=div_yield,
                                    vol=vol,
                                    variant=variant,
                                    ko_barrier=BASE_KO_BARRIER,
                                    ki_barrier=BASE_KI_BARRIER + KI_BARRIER_BUMP,
                                    pde_params=bump_params,
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
                        else:
                            quote_ko_sens_vol_slice.append(None)
                            interest_ko_sens_vol_slice.append(None)
                            quote_ki_sens_vol_slice.append(None)
                            interest_ki_sens_vol_slice.append(None)
                        scenario_rows.append(
                            {
                                "scenario_id": done,
                                "variant": variant,
                                "r": rate,
                                "q": div_yield,
                                "vol": vol,
                                "ko_barrier": BASE_KO_BARRIER,
                                "ki_barrier": BASE_KI_BARRIER,
                                "quoted_ko_rate": result["quoted_ko_rate"],
                                "product_pv": result["snowball_target_pv"],
                                "interest_pv": result["interest_component_pv"],
                                "combined_pv": result["combined_pv"],
                                "protected_snowball_pv": result["protected_snowball_pv"],
                                "quote_ko_sensitivity": quote_ko_sens_vol_slice[-1],
                                "quote_ki_sensitivity": quote_ki_sens_vol_slice[-1],
                                "interest_ko_sensitivity": interest_ko_sens_vol_slice[-1],
                                "interest_ki_sensitivity": interest_ki_sens_vol_slice[-1],
                            }
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
                        scenario_rows.append(
                            {
                                "scenario_id": done,
                                "variant": variant,
                                "r": rate,
                                "q": div_yield,
                                "vol": vol,
                                "ko_barrier": BASE_KO_BARRIER,
                                "ki_barrier": BASE_KI_BARRIER,
                                "quoted_ko_rate": None,
                                "product_pv": None,
                                "interest_pv": None,
                                "combined_pv": None,
                                "protected_snowball_pv": None,
                                "quote_ko_sensitivity": None,
                                "quote_ki_sensitivity": None,
                                "interest_ko_sensitivity": None,
                                "interest_ki_sensitivity": None,
                            }
                        )
                quote_q_slice.append(quote_vol_slice)
                interest_q_slice.append(interest_vol_slice)
                protected_q_slice.append(protected_vol_slice)
                target_q_slice.append(target_vol_slice)
                quote_ko_sens_q_slice.append(quote_ko_sens_vol_slice)
                quote_ki_sens_q_slice.append(quote_ki_sens_vol_slice)
                interest_ko_sens_q_slice.append(interest_ko_sens_vol_slice)
                interest_ki_sens_q_slice.append(interest_ki_sens_vol_slice)
            quote_cube.append(quote_q_slice)
            interest_cube.append(interest_q_slice)
            protected_cube.append(protected_q_slice)
            target_cube.append(target_q_slice)
            quote_ko_sens_cube.append(quote_ko_sens_q_slice)
            quote_ki_sens_cube.append(quote_ki_sens_q_slice)
            interest_ko_sens_cube.append(interest_ko_sens_q_slice)
            interest_ki_sens_cube.append(interest_ki_sens_q_slice)
        variant_cubes[variant] = {
            "quote": quote_cube,
            "interest": interest_cube,
            "protected": protected_cube,
            "snowballTarget": target_cube,
            "quoteKoSens": quote_ko_sens_cube,
            "quoteKiSens": quote_ki_sens_cube,
            "interestKoSens": interest_ko_sens_cube,
            "interestKiSens": interest_ki_sens_cube,
        }
    return (variant_cubes, scenario_rows)


def write_scenario_csv(rows: list[dict[str, Any]]) -> None:
    """Write scenario PV table for downstream analysis."""
    fieldnames = [
        "scenario_id",
        "variant",
        "r",
        "q",
        "vol",
        "ko_barrier",
        "ki_barrier",
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
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_html(data: dict[str, Any]) -> str:
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
          This demo solves the <strong>fair KO rate</strong> for a standard Snowball under a
          Chinese-market <strong>integrated financing quote convention</strong>. The dealer quotes
          one KO coupon that makes the PV of the <strong>ex-principal Snowball</strong>
          match the <strong>interest component</strong> implied by a full-protected
          Snowball financing leg.
        </p>
      </div>
      <div class="hero-grid">
        <div class="chip">
          <div class="chip-label" id="lang-chip-structure-label">Structure</div>
          <div class="chip-value" id="lang-chip-structure-value">103 monthly KO / 75 daily KI</div>
        </div>
        <div class="chip">
          <div class="chip-label" id="lang-chip-tenor-label">Tenor</div>
          <div class="chip-value" id="lang-chip-tenor-value">2.0Y, spot = 100, strike = 100</div>
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
              <option value="parachute">Parachute</option>
              <option value="stepdown">Stepdown</option>
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
              <small id="interest-pv-caption">Prepayment minus full-protected, ex-principal, unannualized-100% Snowball PV.</small>
            </div>
          </div>

          <div class="formula" id="lang-formula">
            <strong>Interest PV</strong> = Prepayment − PV(Full-Protected Snowball, principal excluded, KO rate = 100% unannualized)<br />
            Solve <strong>ko_rate</strong> such that<br />
            <strong>V<sub>snowball, exN</sub>(r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />
            with monthly KO observations, daily KI observations, and 2Y tenor.
          </div>

          <div class="control">
            <div class="control-head">
              <label for="r-slider" id="lang-r-label">Risk-free rate <em>r</em></label>
              <div class="control-value" id="r-value">--</div>
            </div>
            <input id="r-slider" type="range" min="0.01" max="0.05" step="0.001" value="0.03" />
            <div class="control-foot"><span>1.0%</span><span>5.0%</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="q-slider" id="lang-q-label">Dividend yield <em>q</em></label>
              <div class="control-value" id="q-value">--</div>
            </div>
            <input id="q-slider" type="range" min="0.08" max="0.15" step="0.001" value="0.10" />
            <div class="control-foot"><span>8.0%</span><span>15.0%</span></div>
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
            <input id="vol-slider" type="range" min="0.15" max="0.35" step="0.005" value="0.20" />
            <div class="control-foot"><span>15.0%</span><span>35.0%</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="ko-slider" id="lang-ko-label">KO barrier</label>
              <div class="control-value" id="ko-value">--</div>
            </div>
            <input id="ko-slider" type="range" min="95" max="110" step="0.5" value="103" />
            <div class="control-foot"><span>95</span><span>110</span></div>
          </div>

          <div class="control">
            <div class="control-head">
              <label for="ki-slider" id="lang-ki-label">KI barrier</label>
              <div class="control-value" id="ki-value">--</div>
            </div>
            <input id="ki-slider" type="range" min="60" max="85" step="0.5" value="75" />
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
            <h3 id="lang-impact-r-title">If <em>r</em> moves +50bp</h3>
            <div class="impact-main" id="impact-r">--</div>
            <p id="impact-r-text"></p>
          </article>
          <article class="impact-card">
            <h3 id="lang-impact-q-title">If <em>q</em> moves +50bp</h3>
            <div class="impact-main" id="impact-q">--</div>
            <p id="impact-q-text"></p>
          </article>
          <article class="impact-card">
            <h3 id="lang-impact-vol-title">If vol moves +1 vol pt</h3>
            <div class="impact-main" id="impact-vol">--</div>
            <p id="impact-vol-text"></p>
          </article>
        </div>

        <div class="panel-title" style="margin-top: 24px;">
          <h2 id="lang-surface-title">Response Surfaces</h2>
          <span id="lang-surface-tag">same convention, different slices</span>
        </div>
        <div class="heatmap-grid">
          <article class="heatmap-card">
            <h3><em>r</em> vs <em>q</em></h3>
            <p id="heatmap-rq-caption"></p>
            <canvas id="heatmap-rq" width="360" height="300"></canvas>
          </article>
          <article class="heatmap-card">
            <h3><em>r</em> vs vol</h3>
            <p id="heatmap-rv-caption"></p>
            <canvas id="heatmap-rv" width="360" height="300"></canvas>
          </article>
          <article class="heatmap-card">
            <h3><em>q</em> vs vol</h3>
            <p id="heatmap-qv-caption"></p>
            <canvas id="heatmap-qv" width="360" height="300"></canvas>
          </article>
        </div>
        <div class="legend">
          <span>lower quote</span>
          <div class="legend-bar"></div>
          <span>higher quote</span>
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
          <li>Daily KI is modeled as 504 discrete business-day observations over 2 years.</li>
          <li>The interest leg is valued as a full-protected, principal-excluded Snowball with KO rate fixed at 100% unannualized.</li>
          <li>The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.</li>
          <li>This is a pricing explainer, not a production quoting front-end.</li>
        </ul>
      </div>
    </section>
  </div>

  <script>
    const DATA = __DATA__;

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

    const rValue = document.getElementById("r-value");
    const qValue = document.getElementById("q-value");
    const volValue = document.getElementById("vol-value");
    const koValue = document.getElementById("ko-value");
    const kiValue = document.getElementById("ki-value");

    const heatmapRQ = document.getElementById("heatmap-rq");
    const heatmapRV = document.getElementById("heatmap-rv");
    const heatmapQV = document.getElementById("heatmap-qv");

    const impactR = document.getElementById("impact-r");
    const impactQ = document.getElementById("impact-q");
    const impactVol = document.getElementById("impact-vol");
    const impactRText = document.getElementById("impact-r-text");
    const impactQText = document.getElementById("impact-q-text");
    const impactVolText = document.getElementById("impact-vol-text");
    const summaryText = document.getElementById("summary-text");

    const heatmapRQCaption = document.getElementById("heatmap-rq-caption");
    const heatmapRVCaption = document.getElementById("heatmap-rv-caption");
    const heatmapQVCaption = document.getElementById("heatmap-qv-caption");
    let currentLang = "en";

    const I18N = {
      en: {
        toggleLabel: "Language",
        eyebrow: "PDE-backed RFQ explainer",
        heroTitle: "How <em>r</em>, <em>q</em>, and vol reprice a Snowball quote.",
        heroBody: "This demo solves the <strong>fair KO rate</strong> for a standard Snowball under a Chinese-market <strong>integrated financing quote convention</strong>. The dealer quotes one KO coupon that makes the PV of the <strong>ex-principal Snowball</strong> match the <strong>interest component</strong> implied by a full-protected Snowball financing leg.",
        chipStructureLabel: "Structure",
        chipStructureValue: "103 monthly KO / 75 daily KI",
        chipTenorLabel: "Tenor",
        chipTenorValue: "2.0Y, spot = 100, strike = 100",
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
        formula: "<strong>Interest PV</strong> = Prepayment − PV(Full-Protected Snowball, principal excluded, KO rate = 100% unannualized)<br />Solve <strong>ko_rate</strong> such that<br /><strong>V<sub>snowball, exN</sub>(r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />with monthly KO observations, daily KI observations, and 2Y tenor.",
        rLabel: "Risk-free rate <em>r</em>",
        qLabel: "Dividend yield <em>q</em>",
        rqLabel: "r-q link",
        rqFoot: "toggle",
        volLabel: "Flat vol <em>σ</em>",
        koLabel: "KO barrier",
        kiLabel: "KI barrier",
        impactTitle: "Local Impact",
        impactTag: "first-order intuition",
        impactRTitle: "If <em>r</em> moves +50bp",
        impactQTitle: "If <em>q</em> moves +50bp",
        impactVolTitle: "If vol moves +1 vol pt",
        surfaceTitle: "Response Surfaces",
        surfaceTag: "same convention, different slices",
        noteInterpretation: "Interpretation",
        noteBuild: "Build Notes",
        notesList: [
          "Daily KI is modeled as 504 discrete business-day observations over 2 years.",
          "The interest leg is valued as a full-protected, principal-excluded Snowball with KO rate fixed at 100% unannualized.",
          "The HTML embeds a coarse PDE-solved cube and interpolates between nodes in-browser.",
          "This is a pricing explainer, not a production quoting front-end."
        ],
        rqOff: "Off",
        rqLinked: "Linked",
        rqIndependent: "Independent moves",
        rqHolding: (spread) => `Holding q-r spread = ${spread}`,
        variantActive: (desc) => `${desc} Barrier sliders active.`,
        variantPassive: (desc) => `${desc} Barrier sliders shown for standard reference only.`,
        quoteCaption: "Fair KO coupon needed to make Snowball PV match the embedded interest leg.",
        noQuoteCaption: "No positive fair KO rate available inside the embedded quote range.",
        noCombined: "Combined convention not available at this market point.",
        interestCaption: (prepayment, pv) => `Prepayment ${prepayment} minus protected ex-principal Snowball PV ${pv}.`,
        heatmapVol: (v) => `Slice at vol = ${v}.`,
        heatmapQ: (q) => `Slice at q = ${q}.`,
        heatmapR: (r) => `Slice at r = ${r}.`,
        highRUp: "Higher r is lifting the fair coupon in this slice.",
        highRDown: "Higher r is easing the fair coupon because forward drift improves KO odds.",
        highQUp: "Higher q demands more coupon because forward carry deteriorates.",
        highQDown: "Higher q slightly reduces the required coupon in this slice.",
        highVolUp: "Higher vol demands more coupon to compensate for fatter downside KI risk.",
        highVolDown: "Higher vol slightly relaxes the quote in this slice.",
        summary: (variant, r, q, vol, ko, ki, quote, interestPv, targetPv, tail) =>
          `${variant}: at r=${r}, q=${q}, vol=${vol}, KO=${ko}, and KI=${ki}, the locally adjusted fair KO rate is ${quote}. The embedded interest leg contributes ${interestPv} of PV, computed as prepayment minus a full-protected ex-principal Snowball priced with unannualized 100% KO rate, so the quoted Snowball itself must also price to ${targetPv} for the quote to clear. ${tail}`,
        variantNames: {
          standard: "Standard",
          european_ki: "European KI",
          parachute: "Parachute",
          stepdown: "Stepdown"
        }
      },
      cn: {
        toggleLabel: "语言",
        eyebrow: "PDE 驱动 RFQ 解释器",
        heroTitle: "看懂 <em>r</em>、<em>q</em> 与波动率如何重定价雪球报价。",
        heroBody: "这个演示在中国市场常见的<strong>融资一体化报价口径</strong>下，求解标准雪球的<strong>公平 KO 票息</strong>。交易员报价的单一 KO 票息，需要让<strong>不含本金雪球</strong>的 PV 与由<strong>全保本雪球融资腿</strong>隐含出来的<strong>利息成分</strong>相匹配。",
        chipStructureLabel: "结构",
        chipStructureValue: "103 月度敲出 / 75 日度敲入",
        chipTenorLabel: "期限",
        chipTenorValue: "2.0 年，现价 = 100，行权价 = 100",
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
        formula: "<strong>利息 PV</strong> = 预付金 − PV(全保本雪球，去本金，KO 票息 = 100% 非年化)<br />求解 <strong>ko_rate</strong> 使得<br /><strong>V<sub>snowball, exN</sub>(r, q, σ; ko_rate) − Interest PV = 0.0</strong><br />结构为月度 KO、日度 KI、2 年期限。",
        rLabel: "无风险利率 <em>r</em>",
        qLabel: "分红率 <em>q</em>",
        rqLabel: "r-q 联动",
        rqFoot: "开关",
        volLabel: "平坦波动率 <em>σ</em>",
        koLabel: "KO 障碍",
        kiLabel: "KI 障碍",
        impactTitle: "局部影响",
        impactTag: "一阶直觉",
        impactRTitle: "<em>r</em> 上移 50bp",
        impactQTitle: "<em>q</em> 上移 50bp",
        impactVolTitle: "波动率上移 1 个 vol 点",
        surfaceTitle: "响应曲面",
        surfaceTag: "同一口径，不同切片",
        noteInterpretation: "解读",
        noteBuild: "构建说明",
        notesList: [
          "日度 KI 采用 2 年 504 个离散交易日观察。",
          "利息腿按全保本、去本金、KO=100% 非年化的雪球估值。",
          "页面内嵌较粗 PDE 曲面，并在浏览器端做插值。",
          "这是一个定价解释器，不是生产级报价前端。"
        ],
        rqOff: "关闭",
        rqLinked: "联动",
        rqIndependent: "独立变动",
        rqHolding: (spread) => `保持 q-r 利差 = ${spread}`,
        variantActive: (desc) => `${desc} 障碍滑块已启用。`,
        variantPassive: (desc) => `${desc} 障碍滑块仅作标准结构参考。`,
        quoteCaption: "使雪球 PV 与内嵌利息腿相匹配所需的公平 KO 票息。",
        noQuoteCaption: "在当前内嵌区间内没有正的公平 KO 票息。",
        noCombined: "当前市场点下无法得到有效组合结果。",
        interestCaption: (prepayment, pv) => `预付金 ${prepayment} 减去去本金保本雪球 PV ${pv}。`,
        heatmapVol: (v) => `固定 vol = ${v} 的切片。`,
        heatmapQ: (q) => `固定 q = ${q} 的切片。`,
        heatmapR: (r) => `固定 r = ${r} 的切片。`,
        highRUp: "在这个切片里，更高的 r 会抬升公平票息。",
        highRDown: "在这个切片里，更高的 r 会通过改善向 103 KO 漂移而压低公平票息。",
        highQUp: "更高的 q 会恶化远期漂移，因此需要更高票息。",
        highQDown: "在这个切片里，更高的 q 略微降低所需票息。",
        highVolUp: "更高波动会放大下敲风险，因此需要更高票息补偿。",
        highVolDown: "在这个切片里，更高波动略微放松报价。",
        summary: (variant, r, q, vol, ko, ki, quote, interestPv, targetPv, tail) =>
          `${variant}：在 r=${r}、q=${q}、vol=${vol}、KO=${ko}、KI=${ki} 下，局部调整后的公平 KO 票息为 ${quote}。内嵌利息腿 PV 为 ${interestPv}，它来自“预付金减去 KO=100% 非年化的去本金保本雪球 PV”，因此报价雪球自身也必须定价到 ${targetPv} 才能满足当前口径。${tail}`,
        variantNames: {
          standard: "标准型",
          european_ki: "欧式 KI",
          parachute: "降落伞",
          stepdown: "递减敲出"
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
      document.getElementById("lang-chip-tenor-value").textContent = t("chipTenorValue");
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
      document.getElementById("lang-surface-title").textContent = t("surfaceTitle");
      document.getElementById("lang-surface-tag").textContent = t("surfaceTag");
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

    function activeVariantData() {
      return DATA.variants[activeVariantKey()];
    }

    function cubeValue(cube, i, j, k) {
      return cube[i][j][k];
    }

    function trilinear(cube, rate, divYield, vol) {
      const rLoc = locate(DATA.rGrid, rate);
      const qLoc = locate(DATA.qGrid, divYield);
      const vLoc = locate(DATA.volGrid, vol);

      const corners = [
        cubeValue(cube, rLoc.i, qLoc.i, vLoc.i),
        cubeValue(cube, rLoc.i + 1, qLoc.i, vLoc.i),
        cubeValue(cube, rLoc.i, qLoc.i + 1, vLoc.i),
        cubeValue(cube, rLoc.i + 1, qLoc.i + 1, vLoc.i),
        cubeValue(cube, rLoc.i, qLoc.i, vLoc.i + 1),
        cubeValue(cube, rLoc.i + 1, qLoc.i, vLoc.i + 1),
        cubeValue(cube, rLoc.i, qLoc.i + 1, vLoc.i + 1),
        cubeValue(cube, rLoc.i + 1, qLoc.i + 1, vLoc.i + 1),
      ];

      if (corners.some((value) => value === null)) {
        return null;
      }

      const c00 = lerp(corners[0], corners[1], rLoc.t);
      const c10 = lerp(corners[2], corners[3], rLoc.t);
      const c01 = lerp(corners[4], corners[5], rLoc.t);
      const c11 = lerp(corners[6], corners[7], rLoc.t);
      const c0 = lerp(c00, c10, qLoc.t);
      const c1 = lerp(c01, c11, qLoc.t);
      return lerp(c0, c1, vLoc.t);
    }

    function currentRange() {
      const flatValues = activeVariantData().quote.flat(2).filter((value) => value !== null);
      return {
        min: Math.min(...flatValues),
        max: Math.max(...flatValues),
      };
    }

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

    function applyBarrierAdjustment(base, koSens, kiSens, koBarrier, kiBarrier) {
      if (base === null) {
        return null;
      }
      let adjusted = base;
      const koDelta = koBarrier - DATA.meta.structure.base_ko_barrier;
      const kiDelta = kiBarrier - DATA.meta.structure.base_ki_barrier;
      if (koSens !== null) {
        adjusted += koSens * koDelta;
      }
      if (kiSens !== null) {
        adjusted += kiSens * kiDelta;
      }
      return adjusted;
    }

    function drawHeatmap(canvas, sampleFn, xLabel, yLabel, xCurrent, yCurrent, xRange, yRange) {
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      const margin = { top: 18, right: 18, bottom: 34, left: 40 };
      const innerWidth = width - margin.left - margin.right;
      const innerHeight = height - margin.top - margin.bottom;
      const cols = 34;
      const rows = 28;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(255, 255, 255, 0.84)";
      ctx.fillRect(0, 0, width, height);

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const x = xRange[0] + (col / (cols - 1)) * (xRange[1] - xRange[0]);
          const y = yRange[0] + ((rows - 1 - row) / (rows - 1)) * (yRange[1] - yRange[0]);
          const value = sampleFn(x, y);
          ctx.fillStyle = colorFor(value);
          const px = margin.left + (col / cols) * innerWidth;
          const py = margin.top + (row / rows) * innerHeight;
          ctx.fillRect(px, py, innerWidth / cols + 1, innerHeight / rows + 1);
        }
      }

      ctx.strokeStyle = "rgba(23, 33, 38, 0.18)";
      ctx.lineWidth = 1;
      ctx.strokeRect(margin.left, margin.top, innerWidth, innerHeight);

      const crossX = margin.left + ((xCurrent - xRange[0]) / (xRange[1] - xRange[0])) * innerWidth;
      const crossY = margin.top + (1 - (yCurrent - yRange[0]) / (yRange[1] - yRange[0])) * innerHeight;

      ctx.strokeStyle = "rgba(255, 255, 255, 0.86)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(crossX, margin.top);
      ctx.lineTo(crossX, margin.top + innerHeight);
      ctx.moveTo(margin.left, crossY);
      ctx.lineTo(margin.left + innerWidth, crossY);
      ctx.stroke();

      ctx.fillStyle = "#172126";
      ctx.font = "11px var(--mono)";
      ctx.textAlign = "center";
      ctx.fillText(xLabel, margin.left + innerWidth / 2, height - 8);
      ctx.save();
      ctx.translate(12, margin.top + innerHeight / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();

      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(23, 33, 38, 0.72)";
      ctx.fillText(formatPct(xRange[0]), margin.left, height - 18);
      ctx.textAlign = "right";
      ctx.fillText(formatPct(xRange[1]), margin.left + innerWidth, height - 18);

      ctx.textAlign = "left";
      ctx.fillText(formatPct(yRange[0]), 6, margin.top + innerHeight);
      ctx.fillText(formatPct(yRange[1]), 6, margin.top + 10);
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
      const rate = parseFloat(rSlider.value);
      const divYield = parseFloat(qSlider.value);
      const vol = parseFloat(volSlider.value);
      const koBarrier = parseFloat(koSlider.value);
      const kiBarrier = parseFloat(kiSlider.value);
      const variant = activeVariantKey();
      const variantData = activeVariantData();
      const variantMeta = DATA.variantMeta[variant];
      const quote = applyBarrierAdjustment(
        trilinear(variantData.quote, rate, divYield, vol),
        trilinear(variantData.quoteKoSens, rate, divYield, vol),
        trilinear(variantData.quoteKiSens, rate, divYield, vol),
        koBarrier,
        kiBarrier,
      );
      const interestPv = applyBarrierAdjustment(
        trilinear(variantData.interest, rate, divYield, vol),
        trilinear(variantData.interestKoSens, rate, divYield, vol),
        trilinear(variantData.interestKiSens, rate, divYield, vol),
        koBarrier,
        kiBarrier,
      );
      const protectedPv = trilinear(variantData.protected, rate, divYield, vol);
      const snowballTargetPv = interestPv;

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
      interestPvCaption.textContent = t(
        "interestCaption",
        DATA.meta.structure.prepayment.toFixed(2),
        protectedPv.toFixed(2)
      );

      const rUp = applyBarrierAdjustment(
        trilinear(variantData.quote, clamp(rate + 0.005, DATA.rGrid[0], DATA.rGrid.at(-1)), divYield, vol),
        trilinear(variantData.quoteKoSens, clamp(rate + 0.005, DATA.rGrid[0], DATA.rGrid.at(-1)), divYield, vol),
        trilinear(variantData.quoteKiSens, clamp(rate + 0.005, DATA.rGrid[0], DATA.rGrid.at(-1)), divYield, vol),
        koBarrier,
        kiBarrier,
      );
      const qUp = applyBarrierAdjustment(
        trilinear(variantData.quote, rate, clamp(divYield + 0.005, DATA.qGrid[0], DATA.qGrid.at(-1)), vol),
        trilinear(variantData.quoteKoSens, rate, clamp(divYield + 0.005, DATA.qGrid[0], DATA.qGrid.at(-1)), vol),
        trilinear(variantData.quoteKiSens, rate, clamp(divYield + 0.005, DATA.qGrid[0], DATA.qGrid.at(-1)), vol),
        koBarrier,
        kiBarrier,
      );
      const volUp = applyBarrierAdjustment(
        trilinear(variantData.quote, rate, divYield, clamp(vol + 0.01, DATA.volGrid[0], DATA.volGrid.at(-1))),
        trilinear(variantData.quoteKoSens, rate, divYield, clamp(vol + 0.01, DATA.volGrid[0], DATA.volGrid.at(-1))),
        trilinear(variantData.quoteKiSens, rate, divYield, clamp(vol + 0.01, DATA.volGrid[0], DATA.volGrid.at(-1))),
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

      heatmapRQCaption.textContent = t("heatmapVol", formatPct(vol));
      heatmapRVCaption.textContent = t("heatmapQ", formatPct(divYield));
      heatmapQVCaption.textContent = t("heatmapR", formatPct(rate));

      drawHeatmap(
        heatmapRQ,
        (x, y) => applyBarrierAdjustment(
          trilinear(variantData.quote, x, y, vol),
          trilinear(variantData.quoteKoSens, x, y, vol),
          trilinear(variantData.quoteKiSens, x, y, vol),
          koBarrier,
          kiBarrier,
        ),
        "r",
        "q",
        rate,
        divYield,
        [DATA.rGrid[0], DATA.rGrid.at(-1)],
        [DATA.qGrid[0], DATA.qGrid.at(-1)]
      );
      drawHeatmap(
        heatmapRV,
        (x, y) => applyBarrierAdjustment(
          trilinear(variantData.quote, x, divYield, y),
          trilinear(variantData.quoteKoSens, x, divYield, y),
          trilinear(variantData.quoteKiSens, x, divYield, y),
          koBarrier,
          kiBarrier,
        ),
        "r",
        "σ",
        rate,
        vol,
        [DATA.rGrid[0], DATA.rGrid.at(-1)],
        [DATA.volGrid[0], DATA.volGrid.at(-1)]
      );
      drawHeatmap(
        heatmapQV,
        (x, y) => applyBarrierAdjustment(
          trilinear(variantData.quote, rate, x, y),
          trilinear(variantData.quoteKoSens, rate, x, y),
          trilinear(variantData.quoteKiSens, rate, x, y),
          koBarrier,
          kiBarrier,
        ),
        "q",
        "σ",
        divYield,
        vol,
        [DATA.qGrid[0], DATA.qGrid.at(-1)],
        [DATA.volGrid[0], DATA.volGrid.at(-1)]
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

    variantSelect.value = DATA.defaults.variant;
    applyLanguage();

    rSlider.addEventListener("input", () => {
      if (rqLinkToggle.checked) {
        syncQFromR();
      }
      update();
    });

    qSlider.addEventListener("input", () => {
      if (rqLinkToggle.checked) {
        syncRFromQ();
      }
      update();
    });

    rqLinkToggle.addEventListener("change", () => {
      if (rqLinkToggle.checked) {
        rqLinkedSpread = parseFloat(qSlider.value) - parseFloat(rSlider.value);
      }
      refreshRQLinkUI();
      update();
    });

    langEnButton.addEventListener("click", () => {
      currentLang = "en";
      applyLanguage();
      update();
    });

    langCnButton.addEventListener("click", () => {
      currentLang = "cn";
      applyLanguage();
      update();
    });

    [volSlider, koSlider, kiSlider, variantSelect].forEach((slider) => {
      slider.addEventListener("input", update);
    });

    update();
  </script>
</body>
</html>
"""

    return template.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def main() -> None:
    pde_params = PDEParams(grid_size=100, time_steps=150)
    cubes, scenario_rows = build_cube(pde_params=pde_params)
    data = {
        "rGrid": R_GRID,
        "qGrid": Q_GRID,
        "volGrid": VOL_GRID,
        "defaults": {
            "r": DEFAULT_R,
            "q": DEFAULT_Q,
            "vol": DEFAULT_VOL,
            "variant": DEFAULT_VARIANT,
        },
        "koRateBounds": list(KO_RATE_BOUNDS),
        "variants": cubes,
        "variantMeta": VARIANTS,
        "meta": asdict(
            DemoMeta(
                generated_at=datetime.utcnow().isoformat() + "Z",
                engine="SnowballPDESolver",
                solver_grid_size=pde_params.grid_size,
                solver_time_steps=pde_params.time_steps,
                structure={
                    "spot": 100.0,
                    "strike": 100.0,
                    "maturity_years": 2.0,
                    "base_ko_barrier": 103.0,
                    "base_ki_barrier": 75.0,
                    "base_ko_frequency": "monthly",
                    "base_ki_frequency": "daily",
                    "include_principal": False,
                    "target_pv": 0.0,
                    "prepayment": PREPAYMENT,
                    "protected_leg_ko_rate": 1.0,
                    "protected_leg_protection": "FULL",
                    "protected_leg_include_principal": False,
                    "protected_leg_annualized": False,
                },
                ranges={"r": R_GRID, "q": Q_GRID, "vol": VOL_GRID},
            )
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_html(data), encoding="utf-8")
    write_scenario_csv(scenario_rows)
    print(f"Wrote demo HTML to {OUTPUT_PATH}")
    print(f"Wrote scenario CSV to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
