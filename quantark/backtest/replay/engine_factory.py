"""
Pricing engine factory for OTC autocallable backtests.
"""

from typing import Any, Optional

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine,
    OneTouchAnalyticalEngine,
    RangeAccrualAnalyticalEngine,
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import EngineParams, MCParams, PDEParams, QuadParams
from quantark.asset.equity.param.pde_removed_names import removed_name_error
from quantark.asset.equity.product.option import (
    AsianOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    EuropeanVanillaOption,
    OneTouchOption,
    RangeAccrualOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

from .config import AutocallableEngineConfig
from quantark.volcalibration.calibrate import CalibratedVolModel


def _create_analytical_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    params = EngineParams()
    if isinstance(product, EuropeanVanillaOption):
        return BlackScholesEngine(params=params)
    if isinstance(product, AsianOption):
        return AsianOptionAnalyticalEngine(params=params, method=config.method)
    if isinstance(product, RangeAccrualOption):
        return RangeAccrualAnalyticalEngine(params=params)
    if isinstance(product, BarrierOption):
        return BarrierAnalyticalEngine(params=params)
    if isinstance(product, DoubleBarrierOption):
        return DoubleBarrierOptionAnalyticalEngine(params=params)
    if isinstance(product, OneTouchOption):
        return OneTouchAnalyticalEngine(params=params)
    if isinstance(product, SingleSharkfinOption):
        return SingleSharkfinOptionAnalyticalEngine(params=params)
    if isinstance(product, DoubleSharkfinOption):
        return DoubleSharkfinOptionAnalyticalEngine(params=params)
    raise ValidationError(f"Unsupported analytical product type: {type(product).__name__}")


def _apply_bump_overrides(engine, delta_bump_size, gamma_bump_size):
    """Translate per-run bump overrides into the engine's own BumpConfig.

    The replay never bumps by hand (spec §7): overrides land on a COPY of the
    engine's params so shared params objects (config.pde_params reused across
    products) are never mutated.
    """
    if delta_bump_size is None and gamma_bump_size is None:
        return engine
    from dataclasses import replace

    from quantark.asset.equity.param import BumpConfig

    base = engine.params.get_effective_bump_config()
    spot = float(delta_bump_size) if delta_bump_size is not None else base.spot_bump
    gamma = (
        float(gamma_bump_size)
        if gamma_bump_size is not None
        else base.gamma_spot_bump  # a delta-only override must not erase it
    )
    engine.params = replace(
        engine.params,
        bump_config=replace(base, spot_bump=spot, gamma_spot_bump=gamma),
    )
    return engine


def _dcn_mc_kwargs(mc_params: MCParams, method: Any) -> dict:
    """DCN MC engines take their own arguments (no ``MCParams``): paths, seed, Sobol (``use_qmc``) or antithetic;
    one batch, one worker. A ``method`` has no DCN meaning and is refused rather than ignored."""
    if method is not None:
        raise ValidationError(f"the DCN MC engines take no method, got {method!r}")
    return dict(num_paths=int(mc_params.num_paths), seed=int(mc_params.seed), use_sobol=bool(mc_params.use_qmc),
                use_antithetic=bool(mc_params.use_antithetic), num_batches=1, num_workers=1)


def _dcn_mc_engine(cls, mc_params: MCParams, method: Any, **kwargs) -> BaseEngine:
    return cls(**_dcn_mc_kwargs(mc_params, method), **kwargs)


def _create_dcn_vol_model_engine(vol_model, solver, calibrated, pde_params, mc_params, mc_method, options):
    """The DCN's vol-model engines (A6): local-vol MC / PDE and Heston MC / PDE; no SLV DCN route exists."""
    if vol_model == "localvol":
        if calibrated.local_vol_surface is None:
            raise ValidationError("vol_model='localvol' requires calibrated.local_vol_surface")
        if solver == "pde":
            return LocalVolDCNPDEEngine(local_vol_surface=calibrated.local_vol_surface,
                                        params=pde_params or PDEParams(), **options)
        return _dcn_mc_engine(LocalVolDCNMCEngine, mc_params or MCParams(), mc_method,
                              local_vol_surface=calibrated.local_vol_surface, **options)
    if vol_model == "heston":
        if calibrated.heston_params is None:
            raise ValidationError("vol_model='heston' requires calibrated.heston_params")
        if solver == "pde":
            return HestonDCNPDESolver(model_params=calibrated.heston_params, **options)
        return _dcn_mc_engine(HestonDCNMCEngine, mc_params or MCParams(), mc_method,
                              model_params=calibrated.heston_params, **options)
    # TODO(dcn-slv): no SLV DCN engine exists (spec non-goal); refused, never priced with another model
    raise ValidationError(f"no DCN engine for vol_model={vol_model!r}")


def create_autocallable_engine(
    *,
    product: Any,
    engine_type: EngineType,
    config: AutocallableEngineConfig,
    method: Any = None,
) -> BaseEngine:
    """Create a product-compatible pricing engine for backtest replay."""
    if engine_type == EngineType.ANALYTICAL:
        return _create_analytical_engine(product, config)

    if engine_type == EngineType.PDE:
        selected = method if method is not None else config.method
        if selected is not None:
            raise removed_name_error("PDEEngine", "method")
        if isinstance(product, DCNOption):
            return DCNPDEEngine(params=config.pde_params or PDEParams())
        return PDEEngine(params=config.pde_params or PDEParams())

    if engine_type == EngineType.MONTE_CARLO:
        mc_params = config.mc_params or MCParams()
        selected_method = method if method is not None else config.method
        if isinstance(product, DCNOption):
            return _dcn_mc_engine(DCNMCEngine, mc_params, selected_method)
        if isinstance(product, EuropeanVanillaOption):
            return EuropeanMCEngine(params=mc_params, method=selected_method)
        if isinstance(product, PhoenixOption):
            return PhoenixMCEngine(params=mc_params, method=selected_method)
        if isinstance(product, SnowballOption):
            return SnowballMCEngine(params=mc_params, method=selected_method)
        raise ValidationError(f"Unsupported MC product type: {type(product).__name__}")

    if engine_type == EngineType.QUADRATURE:
        quad_params = config.quad_params or QuadParams()
        if isinstance(product, EuropeanVanillaOption):
            selected_method = method if method is not None else config.method
            return EuropeanQuadEngine(params=quad_params, method=selected_method)
        if isinstance(product, PhoenixOption):
            return PhoenixQuadEngine(params=quad_params)
        if isinstance(product, KnockOutResetSnowballOption):
            # SnowballQuadEngine ignores schedule 2 (a knocked-in contract priced 4.195 for -1.744)
            return KOResetSnowballQuadEngine(params=quad_params)
        if isinstance(product, SnowballOption):
            return SnowballQuadEngine(params=quad_params)
        raise ValidationError(f"Unsupported Quad product type: {type(product).__name__}")

    raise ValidationError(f"Unsupported engine type: {engine_type}")


def create_pricing_engine(
    product: Any,
    config: AutocallableEngineConfig,
    *,
    delta_bump_size: float | None = None,
    gamma_bump_size: float | None = None,
) -> BaseEngine:
    engine = create_autocallable_engine(
        product=product,
        engine_type=config.pricing_engine_type,
        config=config,
    )
    return _apply_bump_overrides(engine, delta_bump_size, gamma_bump_size)


def create_surface_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    return create_autocallable_engine(
        product=product,
        engine_type=config.resolve_surface_engine_type(),
        config=config,
        method=None,
    )


def create_event_stats_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    return create_autocallable_engine(
        product=product,
        engine_type=config.resolve_event_stats_engine_type(),
        config=config,
        method=None,
    )


def create_mc_event_stats_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    fallback = AutocallableEngineConfig(
        pricing_engine_type=EngineType.MONTE_CARLO,
        mc_params=config.mc_params or MCParams(num_paths=5000, seed=42),
    )
    return create_autocallable_engine(
        product=product,
        engine_type=EngineType.MONTE_CARLO,
        config=fallback,
    )


def create_vol_model_engine(
    *,
    vol_model: str,
    solver: str,
    calibrated: CalibratedVolModel,
    pde_params: Optional[PDEParams] = None,
    mc_params: Optional[MCParams] = None,
    mc_method: Any = None,
    engine_options: Optional[dict] = None,
    delta_bump_size: Optional[float] = None,
    gamma_bump_size: Optional[float] = None,
    product: Any = None,
) -> BaseEngine:
    """Create a snowball engine wired to a per-day calibrated vol model.

    The calibrated model is FROZEN into the engine (prebuilt LV surface /
    Heston params / leverage surface), so spot-bump greeks reprice with the
    same calibrated model and never trigger a recalibration.  ``eta`` for
    the SLV variants comes from the calibration (``calibrated.slv_eta``) so
    engine and leverage surface always agree.

    ``engine_options`` are passed through to the solver constructor (e.g.
    ``n_x``/``n_v``/``n_t`` for the 2D Heston PDE solvers); unknown options
    raise ``ValidationError`` (fail-closed, never silently dropped).
    """
    options = dict(engine_options or {})
    if not isinstance(calibrated, CalibratedVolModel):
        raise ValidationError(
            f"calibrated must be a CalibratedVolModel, got {type(calibrated).__name__}"
        )
    if calibrated.variant != vol_model:
        raise ValidationError(
            f"calibrated variant {calibrated.variant!r} does not match "
            f"requested vol_model {vol_model!r}"
        )
    if solver not in ("pde", "mc"):
        raise ValidationError(f"solver must be 'pde' or 'mc', got {solver!r}")

    if isinstance(product, DCNOption):
        try:
            return _apply_bump_overrides(_create_dcn_vol_model_engine(
                vol_model, solver, calibrated, pde_params, mc_params, mc_method, options,
            ), delta_bump_size, gamma_bump_size)
        except TypeError as exc:
            raise ValidationError(
                f"Invalid vol_model_engine_options for ({vol_model!r}, {solver!r}): {exc}"
            ) from exc

    try:
        if vol_model == "localvol":
            if calibrated.local_vol_surface is None:
                raise ValidationError(
                    "vol_model='localvol' requires calibrated.local_vol_surface"
                )
            if solver == "pde":
                return _apply_bump_overrides(LocalVolSnowballPDESolver(
                    params=pde_params or PDEParams(),
                    local_vol_surface=calibrated.local_vol_surface,
                    **options,
                ), delta_bump_size, gamma_bump_size)
            return _apply_bump_overrides(LocalVolSnowballMCEngine(
                params=mc_params or MCParams(),
                method=mc_method,
                local_vol_surface=calibrated.local_vol_surface,
                **options,
            ), delta_bump_size, gamma_bump_size)

        if vol_model == "heston":
            if calibrated.heston_params is None:
                raise ValidationError(
                    "vol_model='heston' requires calibrated.heston_params"
                )
            if solver == "pde":
                return _apply_bump_overrides(HestonSnowballPDESolver(
                    model_params=calibrated.heston_params,
                    params=pde_params or PDEParams(),
                    **options,
                ), delta_bump_size, gamma_bump_size)
            return _apply_bump_overrides(HestonSnowballMCEngine(
                model_params=calibrated.heston_params,
                params=mc_params or MCParams(),
                method=mc_method,
                **options,
            ), delta_bump_size, gamma_bump_size)

        if vol_model == "heston_slv":
            if calibrated.heston_params is None or calibrated.leverage_surface is None:
                raise ValidationError(
                    "vol_model='heston_slv' requires calibrated.heston_params "
                    "and calibrated.leverage_surface"
                )
            eta = float(
                calibrated.slv_eta if calibrated.slv_eta is not None else 1.0
            )
            if solver == "pde":
                return _apply_bump_overrides(HestonSLVSnowballPDESolver(
                    model_params=calibrated.heston_params,
                    leverage_surface=calibrated.leverage_surface,
                    eta=eta,
                    params=pde_params or PDEParams(),
                    **options,
                ), delta_bump_size, gamma_bump_size)
            return _apply_bump_overrides(HestonSLVQESnowballMCEngine(
                model_params=calibrated.heston_params,
                params=mc_params or MCParams(),
                leverage_surface=calibrated.leverage_surface,
                eta=eta,
                method=mc_method,
                **options,
            ), delta_bump_size, gamma_bump_size)
    except TypeError as exc:
        raise ValidationError(
            f"Invalid vol_model_engine_options for ({vol_model!r}, {solver!r}): {exc}"
        ) from exc

    raise ValidationError(f"Unknown vol_model: {vol_model!r}")
