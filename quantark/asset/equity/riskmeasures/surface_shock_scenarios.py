"""Typed surface-shock scenario cells on the execution framework
(Phase 5; spec section 13, "Port the solution surface-risk workflow").

This module replaces the solution-side ``surface_risk_parallel.py``
pattern — a mutable ``_WORKER`` global dict, ``os.environ`` mutation for
engine worker counts, and reconstruction of cell semantics by PARSING
scenario names (``name.startswith("tenor_")`` etc.) — with:

- ``SurfaceShockCell``: a typed cell (model x mode x optional tenor or
  moneyness bucket). ``cell_scenario_id`` RENDERS a stable id from the
  typed fields; nothing ever parses it back.
- a registered transformer (``equity-surface-shock/v1``) whose shocked
  snapshot exists for mutation-footprint verification (spec 10.2);
- a registered ``value_kind="float"`` runner that drives the SAME
  ``run_surface_shock_pipeline`` entry point the production pipeline
  uses, with engine worker counts travelling as EXPLICIT payload values.

The typed cell spans the full model x mode x bucket Cartesian space —
combinations the solution's name grammar could not express (for example
``heston`` x ``recalibrate`` x tenor bucket) plan and price through the
same transformer/runner. ``build_surface_shock_cells`` reproduces the
solution's exact production menu (4 global cells + LV-recalibrate tenor
cells + LV-recalibrate moneyness cells).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from quantark.execution.contracts import ScenarioSpec
from quantark.execution.scenario import registries

__all__ = [
    "SURFACE_SHOCK_RUNNER_ID",
    "SURFACE_SHOCK_TRANSFORMER_ID",
    "SurfaceShockCell",
    "SurfaceShockInputs",
    "build_surface_shock_cells",
    "cell_scenario_id",
    "cells_to_scenario_specs",
    "surface_shock_economics",
]

SURFACE_SHOCK_TRANSFORMER_ID = "equity-surface-shock/v1"
SURFACE_SHOCK_RUNNER_ID = "equity-surface-shock/v1"


@dataclass(frozen=True)
class SurfaceShockCell:
    """Typed shock cell: the name-parsing replacement."""

    model: str                                    # "local_vol" | "heston"
    mode: str                                     # "frozen" | "recalibrate"
    dsigma: float
    tenor_bucket: Optional[Tuple[float, float]] = None
    moneyness_bucket: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class SurfaceShockInputs:
    """Immutable base-input snapshot a registered factory rebuilds in any
    process: cleaned quotes, curves, the product, and EXPLICIT engine
    settings (num_paths/seed/engine_workers as payload pairs — never
    environment variables)."""

    cleaned: object          # CleanedQuoteSet
    spot: float
    rate_curve: object
    carry_curve: object
    product: object
    engine_settings: tuple   # sorted (key, value) pairs


def cell_scenario_id(cell: SurfaceShockCell) -> str:
    """Stable display id RENDERED from typed fields; never parsed back."""
    parts = [cell.model, cell.mode]
    if cell.tenor_bucket is not None:
        parts.append(
            f"tenor_{cell.tenor_bucket[0]:.6f}_{cell.tenor_bucket[1]:.6f}"
        )
    if cell.moneyness_bucket is not None:
        parts.append(
            "moneyness_"
            f"{cell.moneyness_bucket[0]:+.2f}_{cell.moneyness_bucket[1]:+.2f}"
        )
    return "surface_shock." + ".".join(parts)


def build_surface_shock_cells(
    tenors: Sequence[float],
    moneyness_buckets: Sequence[Tuple[float, float]],
    dsigma: float,
) -> tuple:
    """The solution's exact production menu (data-driven count:
    4 + len(tenors) + len(moneyness_buckets))."""
    cells = [
        SurfaceShockCell("local_vol", "frozen", dsigma),
        SurfaceShockCell("local_vol", "recalibrate", dsigma),
        SurfaceShockCell("heston", "frozen", dsigma),
        SurfaceShockCell("heston", "recalibrate", dsigma),
    ]
    for tenor in tenors:
        cells.append(
            SurfaceShockCell(
                "local_vol", "recalibrate", dsigma,
                tenor_bucket=(tenor - 1e-6, tenor + 1e-6),
            )
        )
    for low, high in moneyness_buckets:
        cells.append(
            SurfaceShockCell(
                "local_vol", "recalibrate", dsigma,
                moneyness_bucket=(low, high),
            )
        )
    return tuple(cells)


def cells_to_scenario_specs(cells: Sequence[SurfaceShockCell]) -> tuple:
    specs = []
    for cell in cells:
        parameters = (
            ("dsigma", cell.dsigma),
            ("mode", cell.mode),
            ("model", cell.model),
            ("moneyness_bucket", cell.moneyness_bucket),
            ("tenor_bucket", cell.tenor_bucket),
        )
        specs.append(
            ScenarioSpec(
                scenario_id=cell_scenario_id(cell),
                transformer_id=SURFACE_SHOCK_TRANSFORMER_ID,
                parameters=tuple(sorted(parameters)),
                mutation_tags=frozenset({"vol_surface"}),
                required_capabilities=frozenset(
                    {f"runner:{SURFACE_SHOCK_RUNNER_ID}"}
                ),
            )
        )
    return tuple(specs)


def _cleaned_iv_tree(cleaned) -> tuple:
    """Canonicalizable rendering of the cleaned IV nodes (the shock
    layer): the mutation-footprint component for ``vol_surface``."""
    return tuple(
        (
            expiry_t,
            tuple(
                (quote.log_moneyness, quote.iv)
                for quote in cleaned.slices[expiry_t]
            ),
        )
        for expiry_t in sorted(cleaned.slices)
    )


def surface_shock_transformer(base: SurfaceShockInputs, parameters: dict):
    """Pure: returns a new snapshot with the shocked cleaned quotes; used
    by the planner to VERIFY the declared {vol_surface} footprint."""
    from quantark.asset.equity.riskmeasures.surface_shock_pipeline import (
        shock_cleaned_ivs,
    )

    shocked = shock_cleaned_ivs(
        base.cleaned,
        parameters["dsigma"],
        tenor_bucket=parameters.get("tenor_bucket"),
        moneyness_bucket=parameters.get("moneyness_bucket"),
    )
    return dataclasses.replace(base, cleaned=shocked)


def _env_builder(base: SurfaceShockInputs):
    from quantark.param import SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import DayCountConvention

    def build(surface):
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=base.spot),
            vol_surface=surface,
            rate_curve=base.rate_curve,
            div_yield=None,
            valuation_date=base.cleaned.valuation_date,
            day_count_convention=DayCountConvention.ACT_365,
        )
        env.div_yield = base.carry_curve.to_dividend_yield(base.rate_curve)
        return env

    return build


def _engine_factory_from_settings(settings: dict):
    from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
        HestonDCNMCEngine,
        LocalVolDCNMCEngine,
    )

    num_paths = settings["num_paths"]
    seed = settings["seed"]
    engine_workers = settings.get("engine_workers", 1)

    def factory(model, artifact):
        if model == "local_vol":
            return LocalVolDCNMCEngine(
                local_vol_surface=artifact, num_paths=num_paths, seed=seed,
                num_workers=engine_workers,
            )
        return HestonDCNMCEngine(
            model_params=artifact, num_paths=num_paths, seed=seed,
            num_workers=engine_workers,
        )

    return factory


def _flatten(prefix: str, mapping: dict, out: list) -> None:
    for key in sorted(mapping):
        value = mapping[key]
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            _flatten(path, value, out)
        elif isinstance(value, list):
            out.append((path, tuple(value)))
        else:
            out.append((path, value))


def surface_shock_economics(result) -> tuple:
    """Explicitly tiered field mapping (spec 13.4): economic fields flat,
    plan-dependent diagnostics under the ``numerical.`` prefix. The
    complete-payload validator compares every field."""
    economics = [
        ("pv_base", result.pv_base),
        ("pv_shocked", result.pv_shocked),
        ("pnl", result.pnl),
        ("no_arb_passed", result.no_arb_passed),
        ("mode", result.mode),
        ("model", result.model),
        ("notes", tuple(result.notes)),
        ("shock.layer", result.shock["layer"]),
        ("shock.dsigma", result.shock["dsigma"]),
        ("shock.tenor_bucket", (
            tuple(result.shock["tenor_bucket"])
            if result.shock["tenor_bucket"] else None
        )),
        ("shock.moneyness_bucket", (
            tuple(result.shock["moneyness_bucket"])
            if result.shock["moneyness_bucket"] else None
        )),
    ]
    numerical: list = []
    if result.calibration:
        _flatten("numerical.calibration", result.calibration, numerical)
    if result.artifact_diagnostics:
        _flatten("numerical.artifact", result.artifact_diagnostics, numerical)
    return tuple(economics) + tuple(numerical)


def surface_shock_runner(cell, resolved, child_context):
    """One auditable cell through the SAME production entry point.

    The pipeline applies the shock itself (its shock path stays the
    single implementation); the transformer's shocked snapshot exists for
    planner footprint verification.
    """
    from quantark.asset.equity.riskmeasures.surface_shock_pipeline import (
        SurfaceShockMode,
        run_surface_shock_pipeline,
    )

    base = resolved.base_inputs
    parameters = dict(cell.parameters)
    settings = dict(base.engine_settings)
    calibration_config = settings.get("heston_calibration_config")
    result = run_surface_shock_pipeline(
        product=base.product,
        base_env_builder=_env_builder(base),
        cleaned=base.cleaned,
        spot=base.spot,
        rate_curve=base.rate_curve,
        carry_curve=base.carry_curve,
        model=parameters["model"],
        mode=SurfaceShockMode(parameters["mode"]),
        engine_factory=_engine_factory_from_settings(settings),
        dsigma=parameters["dsigma"],
        tenor_bucket=parameters.get("tenor_bucket"),
        moneyness_bucket=parameters.get("moneyness_bucket"),
        max_calibration_rmse_iv=settings.get(
            "max_calibration_rmse_iv", 0.02
        ),
        heston_calibration_config=(
            dict(calibration_config)
            if calibration_config is not None else None
        ),
    )
    return float(result.pnl), surface_shock_economics(result), None


registries.register_transformer(
    SURFACE_SHOCK_TRANSFORMER_ID,
    surface_shock_transformer,
    allowed_tags=frozenset({"vol_surface"}),
    components=(
        ("vol_surface", lambda s: _cleaned_iv_tree(s.cleaned)),
        ("spot", lambda s: s.spot),
        ("model_params", lambda s: s.engine_settings),
    ),
)
registries.register_runner(
    SURFACE_SHOCK_RUNNER_ID, surface_shock_runner, value_kind="float",
)
