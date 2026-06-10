"""
Generate a Quad-1001 Greeks-vs-spot comparison report for Snowball variants.

The report compares:
- Snowball variants: standard DKI, European KI, step-down KO, parachute, airbag
- Tenors: 1y, 2y, 3y
- KO/KI combinations: 100/80, 103/75, 103/70, 105/75

Outputs are written to ``output/doc/snowball_greeks_spot_comparison_quad_1001``
by default, including a CSV data cube, PNG line charts, Markdown report, and
HTML report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


import numpy as np
import pandas as pd

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import BumpConfig, EngineParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_airbag_snowball,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_standard_snowball,
    create_stepdown_snowball,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment


INITIAL_PRICE = 100.0
STRIKE = 100.0
KO_RATE = 0.15
RATE = 0.02
DIVIDEND_YIELD = 0.03
VOLATILITY = 0.22
VALUATION_DATE = datetime(2026, 1, 5)

TENORS = (1, 2, 3)
GREEK_COLUMNS = ("price", "delta", "gamma", "vega", "dividend_rho")

GREEK_LABELS = {
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega (+1 vol point)",
    "dividend_rho": "Dividend Rho (+1% q)",
}

VARIANT_DESCRIPTIONS = {
    "standard_dki": "Standard Snowball, continuous down-and-in monitoring.",
    "european_ki": "European KI Snowball, KI checked only at maturity.",
    "stepdown": "Step-down Snowball, KO barrier decreases 0.5% of initial per month.",
    "parachute": "Parachute Snowball, final KO barrier drops to the KI level.",
    "airbag": "Airbag Snowball, reduced participation below 60% spot.",
}

VARIANT_LABELS = {
    "standard_dki": "Standard DKI",
    "european_ki": "European KI",
    "stepdown": "Step-down",
    "parachute": "Parachute",
    "airbag": "Airbag",
}

KO_KI_COMBOS = (
    {"combo": "KO100_KI80", "label": "KO 100 / KI 80", "ko": 100.0, "ki": 80.0},
    {"combo": "KO103_KI75", "label": "KO 103 / KI 75", "ko": 103.0, "ki": 75.0},
    {"combo": "KO103_KI70", "label": "KO 103 / KI 70", "ko": 103.0, "ki": 70.0},
    {"combo": "KO105_KI75", "label": "KO 105 / KI 75", "ko": 105.0, "ki": 75.0},
)

COLORS = {
    "standard_dki": "#2563eb",
    "european_ki": "#059669",
    "stepdown": "#d97706",
    "parachute": "#7c3aed",
    "airbag": "#dc2626",
}

COMBO_COLORS = {
    "KO100_KI80": "#dc2626",
    "KO103_KI75": "#2563eb",
    "KO103_KI70": "#059669",
    "KO105_KI75": "#7c3aed",
}

_ENGINE: SnowballQuadEngine | None = None
_CALCULATOR: GreeksCalculator | None = None


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    grid_points: int
    workers: int
    spot_points: int
    spot_min: float
    spot_max: float
    spot_bump: float
    vol_bump: float
    div_bump: float
    rate_bump: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Quad-1001 Snowball Greeks-vs-spot comparison report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/doc/snowball_greeks_spot_comparison_quad_1001"),
        help="Directory for the generated report artifacts.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1001,
        help="Quadrature grid points. Use 1001 for the requested Quad 1001 engine.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 8)),
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--spot-points",
        type=int,
        default=25,
        help="Base spot grid size before adding KO/KI/initial anchors.",
    )
    parser.add_argument("--spot-min", type=float, default=60.0)
    parser.add_argument("--spot-max", type=float, default=120.0)
    parser.add_argument(
        "--spot-bump",
        type=float,
        default=0.005,
        help="Relative spot bump for Delta/Gamma finite differences.",
    )
    parser.add_argument(
        "--vol-bump",
        type=float,
        default=0.01,
        help="Absolute volatility bump. Vega is the price change for this bump.",
    )
    parser.add_argument("--div-bump", type=float, default=0.0001)
    parser.add_argument("--rate-bump", type=float, default=0.0001)
    return parser.parse_args()


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        output_dir=args.output_dir,
        grid_points=args.grid_points,
        workers=args.workers,
        spot_points=args.spot_points,
        spot_min=args.spot_min,
        spot_max=args.spot_max,
        spot_bump=args.spot_bump,
        vol_bump=args.vol_bump,
        div_bump=args.div_bump,
        rate_bump=args.rate_bump,
    )


def _init_worker(
    grid_points: int,
    spot_bump: float,
    vol_bump: float,
    div_bump: float,
    rate_bump: float,
) -> None:
    global _ENGINE, _CALCULATOR
    _ENGINE = SnowballQuadEngine(params=QuadParams(grid_points=grid_points))
    _CALCULATOR = GreeksCalculator(
        params=EngineParams(
            bump_config=BumpConfig(
                spot_bump=spot_bump,
                vol_bump=vol_bump,
                div_bump=div_bump,
                rate_bump=rate_bump,
            )
        )
    )


def _build_env(spot: float) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=FlatVolSurface(volatility=VOLATILITY),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIVIDEND_YIELD),
        valuation_date=VALUATION_DATE,
    )


def _build_product(
    *,
    variant: str,
    tenor: int,
    ko_barrier: float,
    ki_barrier: float,
):
    common_kwargs = {
        "initial_price": INITIAL_PRICE,
        "strike": STRIKE,
        "maturity": float(tenor),
        "contract_multiplier": 1.0,
        "ko_rate": KO_RATE,
        "ki_barrier": float(ki_barrier),
        "is_reverse": False,
        "include_principal": False,
    }
    observations = 12 * int(tenor)

    if variant == "standard_dki":
        return create_standard_snowball(
            ko_barrier=float(ko_barrier),
            num_observations=observations,
            **common_kwargs,
        )
    if variant == "european_ki":
        return create_european_ki_snowball(
            ko_barrier=float(ko_barrier),
            num_ko_observations=observations,
            **common_kwargs,
        )
    if variant == "stepdown":
        return create_stepdown_snowball(
            initial_ko_barrier=float(ko_barrier),
            stepdown_rate=0.005,
            num_observations=observations,
            **common_kwargs,
        )
    if variant == "parachute":
        return create_parachute_snowball(
            ko_barrier=float(ko_barrier),
            num_observations=observations,
            **common_kwargs,
        )
    if variant == "airbag":
        return create_airbag_snowball(
            ko_barrier=float(ko_barrier),
            airbag_barrier=60.0,
            participation_rate=1.0,
            airbag_participation_rate=0.5,
            num_observations=observations,
            **common_kwargs,
        )
    raise ValueError(f"Unknown variant: {variant}")


def _compute_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _ENGINE is None or _CALCULATOR is None:
        raise RuntimeError("Worker is not initialized.")
    product = _build_product(
        variant=str(task["variant"]),
        tenor=int(task["tenor"]),
        ko_barrier=float(task["ko_barrier"]),
        ki_barrier=float(task["ki_barrier"]),
    )
    env = _build_env(float(task["spot"]))
    values = _CALCULATOR.calculate(
        product,
        env,
        _ENGINE,
        greeks=GREEK_COLUMNS,
    )
    return {
        **dict(task),
        "price": float(values["price"]),
        "delta": float(values["delta"]),
        "gamma": float(values["gamma"]),
        "vega": float(values["vega"]),
        "dividend_rho": float(values["dividend_rho"]),
    }


def build_spot_grid(config: RunConfig) -> np.ndarray:
    base = np.linspace(config.spot_min, config.spot_max, config.spot_points)
    anchors = [INITIAL_PRICE, config.spot_min, config.spot_max]
    for combo in KO_KI_COMBOS:
        anchors.extend([float(combo["ko"]), float(combo["ki"])])
    values = sorted({round(float(x), 10) for x in np.concatenate([base, anchors])})
    return np.array(values, dtype=float)


def build_tasks(spot_grid: np.ndarray) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for combo in KO_KI_COMBOS:
        for tenor in TENORS:
            for variant in VARIANT_LABELS:
                for spot in spot_grid:
                    tasks.append(
                        {
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
                            "tenor": tenor,
                            "combo": combo["combo"],
                            "combo_label": combo["label"],
                            "ko_barrier": float(combo["ko"]),
                            "ki_barrier": float(combo["ki"]),
                            "spot": float(spot),
                        }
                    )
    return tasks


def compute_data(config: RunConfig, tasks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = time.time()
    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_init_worker,
        initargs=(
            config.grid_points,
            config.spot_bump,
            config.vol_bump,
            config.div_bump,
            config.rate_bump,
        ),
    ) as executor:
        futures = [executor.submit(_compute_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx == 1 or idx % 100 == 0 or idx == len(futures):
                elapsed = time.time() - start
                print(f"Computed {idx}/{len(futures)} points in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    return df.sort_values(["combo", "tenor", "variant", "spot"]).reset_index(drop=True)


def _require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def save_variant_facet_plot(
    *,
    df: pd.DataFrame,
    greek: str,
    output_path: Path,
) -> None:
    plt = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(TENORS),
        len(KO_KI_COMBOS),
        figsize=(18, 11),
        sharex=True,
    )

    for row_idx, tenor in enumerate(TENORS):
        for col_idx, combo in enumerate(KO_KI_COMBOS):
            ax = axes[row_idx, col_idx]
            subset = df[(df["tenor"] == tenor) & (df["combo"] == combo["combo"])]
            for variant in VARIANT_LABELS:
                variant_df = subset[subset["variant"] == variant].sort_values("spot")
                ax.plot(
                    variant_df["spot"].to_numpy(),
                    variant_df[greek].to_numpy(),
                    label=VARIANT_LABELS[variant],
                    linewidth=1.8,
                    color=COLORS[variant],
                )
            ax.axvline(float(combo["ki"]), color="#111827", linestyle="--", linewidth=0.8, alpha=0.55)
            ax.axvline(float(combo["ko"]), color="#111827", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.axvline(INITIAL_PRICE, color="#6b7280", linestyle="-.", linewidth=0.8, alpha=0.45)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{tenor}Y | {combo['label']}", fontsize=10)
            if row_idx == len(TENORS) - 1:
                ax.set_xlabel("Spot")
            if col_idx == 0:
                ax.set_ylabel(GREEK_LABELS[greek])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(VARIANT_LABELS), frameon=False)
    fig.suptitle(f"{GREEK_LABELS[greek]} vs Spot by Snowball Variant", fontsize=15)
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_combo_facet_plot(
    *,
    df: pd.DataFrame,
    greek: str,
    variant: str,
    output_path: Path,
) -> None:
    plt = _require_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(TENORS), figsize=(15, 4.6), sharex=True)

    for idx, tenor in enumerate(TENORS):
        ax = axes[idx]
        subset = df[(df["tenor"] == tenor) & (df["variant"] == variant)]
        for combo in KO_KI_COMBOS:
            combo_df = subset[subset["combo"] == combo["combo"]].sort_values("spot")
            ax.plot(
                combo_df["spot"].to_numpy(),
                combo_df[greek].to_numpy(),
                label=combo["label"],
                linewidth=1.8,
                color=COMBO_COLORS[str(combo["combo"])],
            )
        ax.axvline(INITIAL_PRICE, color="#6b7280", linestyle="-.", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{tenor}Y", fontsize=10)
        ax.set_xlabel("Spot")
        if idx == 0:
            ax.set_ylabel(GREEK_LABELS[greek])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(KO_KI_COMBOS), frameon=False)
    fig.suptitle(
        f"{GREEK_LABELS[greek]} vs Spot by KO/KI - {VARIANT_LABELS[variant]}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_plots(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    paths: dict[str, str] = {}

    for greek in GREEK_LABELS:
        path = plot_dir / f"facet_variants_{greek}.png"
        save_variant_facet_plot(df=df, greek=greek, output_path=path)
        paths[f"facet_variants_{greek}"] = str(path.relative_to(output_dir))

    combo_dir = plot_dir / "combo_slices"
    for greek in GREEK_LABELS:
        for variant in VARIANT_LABELS:
            path = combo_dir / f"{variant}_{greek}_by_combo.png"
            save_combo_facet_plot(
                df=df,
                greek=greek,
                variant=variant,
                output_path=path,
            )
            paths[f"{variant}_{greek}_by_combo"] = str(path.relative_to(output_dir))

    return paths


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown_table(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(record.get(column, "")) for column in columns) + " |"
        for record in records
    ]
    return "\n".join([header, divider, *rows])


def build_key_table(df: pd.DataFrame) -> list[dict[str, str]]:
    mask = (
        (df["combo"] == "KO103_KI75")
        & np.isclose(df["spot"], INITIAL_PRICE)
    )
    records: list[dict[str, str]] = []
    subset = df[mask].sort_values(["tenor", "variant"])
    for _, row in subset.iterrows():
        records.append(
            {
                "Tenor": f"{int(row['tenor'])}Y",
                "Variant": str(row["variant_label"]),
                "Price": _format_float(row["price"]),
                "Delta": _format_float(row["delta"]),
                "Gamma": _format_float(row["gamma"], 5),
                "Vega": _format_float(row["vega"]),
                "DivRho": _format_float(row["dividend_rho"]),
            }
        )
    return records


def build_interpretation(df: pd.DataFrame) -> list[str]:
    base = df[(df["combo"] == "KO103_KI75") & np.isclose(df["spot"], INITIAL_PRICE)]
    gamma_peak = df.iloc[df["gamma"].abs().idxmax()]
    vega_peak = df.iloc[df["vega"].abs().idxmax()]
    rho_peak = df.iloc[df["dividend_rho"].abs().idxmax()]

    observations = [
        (
            "The strongest Gamma concentration appears at "
            f"S={_format_float(gamma_peak['spot'], 2)} for "
            f"{gamma_peak['variant_label']} {int(gamma_peak['tenor'])}Y "
            f"({gamma_peak['combo_label']}), with Gamma={_format_float(gamma_peak['gamma'], 5)}."
        ),
        (
            "The largest absolute Vega in this grid is "
            f"{_format_float(vega_peak['vega'])} at S={_format_float(vega_peak['spot'], 2)} for "
            f"{vega_peak['variant_label']} {int(vega_peak['tenor'])}Y "
            f"({vega_peak['combo_label']})."
        ),
        (
            "The largest absolute dividend sensitivity is "
            f"{_format_float(rho_peak['dividend_rho'])} at S={_format_float(rho_peak['spot'], 2)} for "
            f"{rho_peak['variant_label']} {int(rho_peak['tenor'])}Y "
            f"({rho_peak['combo_label']})."
        ),
    ]

    if not base.empty:
        delta_ranges = []
        for tenor in TENORS:
            tenor_base = base[base["tenor"] == tenor]
            delta_ranges.append(
                f"{tenor}Y Delta range "
                f"{_format_float(tenor_base['delta'].min())} to "
                f"{_format_float(tenor_base['delta'].max())}"
            )
        observations.append(
            "At S=100 under the base KO 103 / KI 75 structure, "
            + "; ".join(delta_ranges)
            + "."
        )

    return observations


def write_markdown_report(
    *,
    df: pd.DataFrame,
    config: RunConfig,
    plot_paths: Mapping[str, str],
    elapsed_seconds: float,
    data_csv: Path,
    metadata_json: Path,
) -> Path:
    report_path = config.output_dir / "snowball_greeks_spot_comparison_quad_1001.md"
    key_table = build_key_table(df)
    observations = build_interpretation(df)

    variant_rows = [
        {"Variant": VARIANT_LABELS[key], "Definition": desc}
        for key, desc in VARIANT_DESCRIPTIONS.items()
    ]
    combo_rows = [
        {
            "Combination": combo["label"],
            "KO": _format_float(float(combo["ko"]), 1),
            "KI": _format_float(float(combo["ki"]), 1),
        }
        for combo in KO_KI_COMBOS
    ]

    lines = [
        "# Snowball Greeks vs Spot Comparison - Quad 1001",
        "",
        "## Scope",
        "",
        (
            "This report compares Snowball Greeks as line charts over spot for five "
            "Snowball variants, three tenors (1Y, 2Y, 3Y), and four KO/KI barrier "
            "combinations. Pricing uses `SnowballQuadEngine` with "
            f"`QuadParams(grid_points={config.grid_points})`."
        ),
        "",
        "## Method",
        "",
        "- Market: S0=100, strike=100, flat volatility=22%, risk-free rate=2%, dividend yield=3%.",
        "- Payoff convention: ex-principal, annual KO coupon=15%, monthly KO observations.",
        (
            "- Greeks: finite-difference `GreeksCalculator`; "
            f"spot bump={config.spot_bump:.2%}, vol bump={config.vol_bump:.2%}, "
            f"rate/dividend bump={config.rate_bump:.2%}/{config.div_bump:.2%}."
        ),
        "- Vega is shown as the price change for a +1 vol point bump; dividend rho is price change for a +1% dividend-yield shift.",
        "",
        "## Product Variants",
        "",
        _markdown_table(variant_rows, ["Variant", "Definition"]),
        "",
        "## KO/KI Combinations",
        "",
        _markdown_table(combo_rows, ["Combination", "KO", "KI"]),
        "",
        "## Key Readouts",
        "",
        *[f"- {item}" for item in observations],
        "",
        "## Base Spot Table",
        "",
        "Base point: S=100, KO 103 / KI 75.",
        "",
        _markdown_table(key_table, ["Tenor", "Variant", "Price", "Delta", "Gamma", "Vega", "DivRho"]),
        "",
        "## Main Line Charts",
        "",
        "Chart markers: dashed vertical line = KI barrier, dotted vertical line = KO barrier, dash-dot vertical line = S0.",
        "",
    ]

    for greek in GREEK_LABELS:
        key = f"facet_variants_{greek}"
        lines.extend(
            [
                f"### {GREEK_LABELS[greek]}",
                "",
                f"![{GREEK_LABELS[greek]} vs Spot]({plot_paths[key]})",
                "",
            ]
        )

    lines.extend(
        [
            "## KO/KI Appendix Charts",
            "",
            (
                "Appendix charts compare KO/KI combinations for each variant and tenor. "
                "They are saved under `plots/combo_slices/` in the output directory."
            ),
            "",
            "## Artifacts",
            "",
            f"- Data cube: `{data_csv.relative_to(config.output_dir)}`",
            f"- Metadata: `{metadata_json.relative_to(config.output_dir)}`",
            f"- Runtime: {_format_float(elapsed_seconds, 1)} seconds with {config.workers} workers.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_html_report(
    *,
    markdown_path: Path,
    config: RunConfig,
    plot_paths: Mapping[str, str],
    key_table: Sequence[Mapping[str, str]],
    observations: Sequence[str],
) -> Path:
    html_path = config.output_dir / "index.html"

    def table_html(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
        rows = ["<table><thead><tr>"]
        rows.extend(f"<th>{column}</th>" for column in columns)
        rows.append("</tr></thead><tbody>")
        for record in records:
            rows.append("<tr>")
            rows.extend(f"<td>{record.get(column, '')}</td>" for column in columns)
            rows.append("</tr>")
        rows.append("</tbody></table>")
        return "".join(rows)

    style = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }
    h1, h2, h3 { color: #111827; }
    p, li { line-height: 1.5; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }
    th { background: #f3f4f6; }
    img { width: 100%; max-width: 1500px; border: 1px solid #e5e7eb; }
    .chart { margin: 28px 0 38px; }
    .meta { color: #4b5563; }
    """
    variant_rows = [
        {"Variant": VARIANT_LABELS[key], "Definition": desc}
        for key, desc in VARIANT_DESCRIPTIONS.items()
    ]
    combo_rows = [
        {
            "Combination": combo["label"],
            "KO": _format_float(float(combo["ko"]), 1),
            "KI": _format_float(float(combo["ki"]), 1),
        }
        for combo in KO_KI_COMBOS
    ]

    chart_html = []
    for greek in GREEK_LABELS:
        key = f"facet_variants_{greek}"
        chart_html.append(
            f'<section class="chart"><h3>{GREEK_LABELS[greek]}</h3>'
            f'<img alt="{GREEK_LABELS[greek]} vs Spot" src="{plot_paths[key]}"></section>'
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Snowball Greeks vs Spot Comparison - Quad 1001</title>
  <style>{style}</style>
</head>
<body>
  <h1>Snowball Greeks vs Spot Comparison - Quad 1001</h1>
  <p class="meta">Generated from {markdown_path.name}. Engine: SnowballQuadEngine / QuadParams(grid_points={config.grid_points}).</p>
  <h2>Scope</h2>
  <p>Comparison across five Snowball variants, tenors 1Y/2Y/3Y, and four KO/KI combinations. Market assumptions: S0=100, strike=100, vol=22%, r=2%, q=3%, KO coupon=15%, ex-principal payoff convention.</p>
  <h2>Product Variants</h2>
  {table_html(variant_rows, ["Variant", "Definition"])}
  <h2>KO/KI Combinations</h2>
  {table_html(combo_rows, ["Combination", "KO", "KI"])}
  <h2>Key Readouts</h2>
  <ul>{''.join(f'<li>{item}</li>' for item in observations)}</ul>
  <h2>Base Spot Table</h2>
  <p>Base point: S=100, KO 103 / KI 75.</p>
  {table_html(key_table, ["Tenor", "Variant", "Price", "Delta", "Gamma", "Vega", "DivRho"])}
  <h2>Main Line Charts</h2>
  <p>Chart markers: dashed vertical line = KI barrier, dotted vertical line = KO barrier, dash-dot vertical line = S0.</p>
  {''.join(chart_html)}
  <h2>Artifacts</h2>
  <p>Full data cube: <code>snowball_greeks_spot_data.csv</code>. KO/KI appendix charts are under <code>plots/combo_slices/</code>.</p>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def write_metadata(
    *,
    config: RunConfig,
    spot_grid: np.ndarray,
    tasks_count: int,
    elapsed_seconds: float,
    plot_paths: Mapping[str, str],
) -> Path:
    metadata_path = config.output_dir / "metadata.json"
    payload = {
        "engine": "SnowballQuadEngine",
        "quad_params": {"grid_points": config.grid_points},
        "market": {
            "initial_price": INITIAL_PRICE,
            "strike": STRIKE,
            "rate": RATE,
            "dividend_yield": DIVIDEND_YIELD,
            "volatility": VOLATILITY,
            "valuation_date": VALUATION_DATE.isoformat(),
        },
        "finite_difference": {
            "spot_bump": config.spot_bump,
            "vol_bump": config.vol_bump,
            "rate_bump": config.rate_bump,
            "div_bump": config.div_bump,
        },
        "tenors": list(TENORS),
        "variants": VARIANT_DESCRIPTIONS,
        "ko_ki_combos": list(KO_KI_COMBOS),
        "spot_grid": [float(x) for x in spot_grid],
        "tasks_count": tasks_count,
        "workers": config.workers,
        "elapsed_seconds": elapsed_seconds,
        "plots": dict(plot_paths),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    config = build_run_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    spot_grid = build_spot_grid(config)
    tasks = build_tasks(spot_grid)
    print(
        f"Running {len(tasks)} Greek points with Quad grid={config.grid_points}, "
        f"workers={config.workers}, spot nodes={len(spot_grid)}"
    )

    df = compute_data(config, tasks)
    data_csv = config.output_dir / "snowball_greeks_spot_data.csv"
    df.to_csv(data_csv, index=False)

    plot_paths = save_plots(df, config.output_dir)
    elapsed = time.time() - start
    metadata_path = write_metadata(
        config=config,
        spot_grid=spot_grid,
        tasks_count=len(tasks),
        elapsed_seconds=elapsed,
        plot_paths=plot_paths,
    )
    markdown_path = write_markdown_report(
        df=df,
        config=config,
        plot_paths=plot_paths,
        elapsed_seconds=elapsed,
        data_csv=data_csv,
        metadata_json=metadata_path,
    )
    html_path = write_html_report(
        markdown_path=markdown_path,
        config=config,
        plot_paths=plot_paths,
        key_table=build_key_table(df),
        observations=build_interpretation(df),
    )

    print(f"Wrote data cube: {data_csv}")
    print(f"Wrote markdown report: {markdown_path}")
    print(f"Wrote HTML report: {html_path}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
