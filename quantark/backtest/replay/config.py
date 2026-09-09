"""
Configuration objects for OTC autocallable backtests.
"""

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Literal, Optional

from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

from .market import AutocallableMarketDataSet
from .strategy_state import AutocallableDeltaHedgeStrategy


# Canonical home: quantark.backtest.futures_ledger (re-exported here).
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: F401,E402


# VolModelCalibrationConfig and HESTON_PRESETS moved to
# quantark.volcalibration.config in 0.4.0: the vol-calibration engine must not
# depend on the backtest package config shape.  Re-exported here until 0.5.0.
from quantark.volcalibration.config import (  # noqa: F401,E402
    HESTON_PRESETS,
    VolModelCalibrationConfig,
)

@dataclass
class AutocallableEngineConfig:
    """
    Engine selection for pricing, surfaces, and event stats.

    ``vol_source`` selects the daily vol channel used by
    ``ProductReplay.build_env``:

    - ``"scalar"`` (default): historical behavior — one flat vol from the
      market ``volatility`` column.
    - ``"surface"``: reprice daily against the admitted IV-surface artifact
      (requires ``market_data.surface_history``); ``surface_vol_mode`` picks
      the vol object built from the artifact:

      - ``"flat_atm_remaining"``: ATM term structure sampled at the product's
        remaining maturity, wrapped in a flat surface (refreshed daily).
      - ``"term_structure"``: ATM pillar term structure (strike-independent).
      - ``"full_grid"``: full strike x maturity smile grid.

    ``vol_model`` selects the daily pricing model (Task 2.3):

    - ``"bsm"`` (default): the classic factory engine (``pricing_engine_type``).
    - ``"localvol"`` / ``"heston"`` / ``"heston_slv"``: the day's pricing
      engine is a per-day calibrated vol-model snowball engine
      (``create_vol_model_engine``), calibrated once per surface artifact
      (sha-keyed cache) and used for both the base price and the
      bumped-greek reprices of that day.  Requires ``vol_source="surface"``
      and forces ``surface_vol_mode="full_grid"``: the LV variants consume
      the full smile grid, and the Heston variants price from their own
      model dynamics (the env vol object is unused by those engines, so
      full_grid keeps the recorded provenance consistent).
      ``pricing_engine_type`` still drives the surface/event-stats engines.

    ``method`` belongs to ``pricing_engine_type`` (a ``PDEMethod`` for PDE, a
    ``MonteCarloMethod`` for MC, ...) and is also what the surface and
    event-stats engines receive.  A vol-model MC route needs its OWN method
    (e.g. ``MonteCarloMethod.RANDOMIZED_QUASI``) which is generally NOT valid
    for ``pricing_engine_type``: a run that prices Heston on RQMC while
    keeping the deterministic PDE surface/event-stats engines cannot express
    both through one field.  ``vol_model_mc_method`` is that second slot; it
    falls back to ``method`` when unset, so existing configurations are
    unchanged.

    ``dividend_source`` selects the daily dividend/carry channel:

    - ``None`` (default): the historical behaviour — scalar vol mode prices
      off ONE flat yield implied from the active hedge contract (simple
      compounding, floored at zero); surface vol mode prices off the
      artifact's option-implied forward pillars.
    - ``"active_contract"``: the scalar-mode flat yield, explicitly.
    - ``"futures_curve"``: every listed contract on the day with at least
      ``futures_curve_min_tenor_days`` calendar days to expiry inverted
      through ``IndexFuturesCurve`` — continuous compounding, signed, linear
      in nodal q.  Beyond the last listed tenor ``futures_curve_extrapolation``
      picks ``"flat_q"`` (the endpoint zero yield is held) or
      ``"flat_forward_carry"`` (the last segment's forward carry continues,
      via ``ForwardCarryCurve``) or ``"surface_forward_carry"`` (the chain is
      held in log-forward space and continued past its last contract with
      the admitted IV-surface artifact's option-implied FORWARD carry, level
      matched at the join, then flat forward carry; needs
      ``market_data.surface_history``).  The minimum tenor (default one week,
      at least one day: a contract expiring today has no defined yield)
      drops delivery-week contracts, whose annualised basis is noise (a 1%
      basis two days out reads as a 180% yield); a day left with ONE eligible
      contract prices off that node as a flat continuous signed yield — the
      exact one-node limit of both extrapolation conventions — and a day
      with none fails closed.  The state row's ``pricing_q`` is the term
      yield at the product's remaining maturity (for a multi-product book:
      the FIRST product's; pricing itself samples each product's own).
    - ``"surface_forwards"``: the admitted IV-surface artifact's parity
      forward pillars as a ``TermStructureDividendYield``, usable with the
      scalar vol channel (requires ``market_data.surface_history``).

    The two term sources are incompatible with ``fixed_dividend_yield`` and
    with the flat-q surface grid (``calculate_surfaces``); both are rejected
    at config time rather than silently swapping the model.
    """

    pricing_engine_type: EngineType = EngineType.PDE
    method: Optional[Any] = None
    vol_model_mc_method: Optional[Any] = None
    pde_params: Optional[PDEParams] = None
    mc_params: Optional[MCParams] = None
    quad_params: Optional[QuadParams] = None
    surface_engine_type: Optional[EngineType] = None
    event_stats_engine_type: Optional[EngineType] = None
    vol_source: Literal["scalar", "surface"] = "scalar"
    surface_vol_mode: Literal[
        "flat_atm_remaining", "term_structure", "full_grid"
    ] = "flat_atm_remaining"
    vol_model: Literal["bsm", "localvol", "heston", "heston_slv"] = "bsm"
    vol_model_solver: Literal["pde", "mc"] = "pde"
    vol_model_calibration: Optional[VolModelCalibrationConfig] = None
    vol_model_engine_options: dict[str, Any] = field(default_factory=dict)
    event_stats_fallback: Literal["none", "mc"] = "none"
    dividend_source: Optional[
        Literal["active_contract", "futures_curve", "surface_forwards"]
    ] = None
    futures_curve_extrapolation: Literal[
        "flat_q", "flat_forward_carry", "surface_forward_carry"
    ] = "flat_q"
    futures_curve_min_tenor_days: int = 7

    def uses_term_dividend_source(self) -> bool:
        """True when the day's dividend object is a term structure chosen by
        ``dividend_source`` (as opposed to the legacy scalar channel)."""
        return self.dividend_source in ("futures_curve", "surface_forwards")

    def __post_init__(self) -> None:
        supported = {
            EngineType.ANALYTICAL,
            EngineType.PDE,
            EngineType.MONTE_CARLO,
            EngineType.QUADRATURE,
        }
        if self.pricing_engine_type not in supported:
            raise ValidationError(
                "pricing_engine_type must be ANALYTICAL, PDE, MONTE_CARLO, or QUADRATURE"
            )
        if self.surface_engine_type is not None and self.surface_engine_type not in supported:
            raise ValidationError(
                "surface_engine_type must be ANALYTICAL, PDE, MONTE_CARLO, or QUADRATURE"
            )
        if (
            self.event_stats_engine_type is not None
            and self.event_stats_engine_type not in supported
        ):
            raise ValidationError(
                "event_stats_engine_type must be PDE, MONTE_CARLO, or QUADRATURE"
            )
        if self.vol_source not in ("scalar", "surface"):
            raise ValidationError("vol_source must be 'scalar' or 'surface'")
        if self.surface_vol_mode not in (
            "flat_atm_remaining",
            "term_structure",
            "full_grid",
        ):
            raise ValidationError(
                "surface_vol_mode must be 'flat_atm_remaining', "
                "'term_structure', or 'full_grid'"
            )
        if self.vol_model not in ("bsm", "localvol", "heston", "heston_slv"):
            raise ValidationError(
                "vol_model must be 'bsm', 'localvol', 'heston', or 'heston_slv'"
            )
        if self.vol_model_solver not in ("pde", "mc"):
            raise ValidationError("vol_model_solver must be 'pde' or 'mc'")
        if self.event_stats_fallback not in ("none", "mc"):
            raise ValidationError("event_stats_fallback must be 'none' or 'mc'")
        if self.dividend_source not in (
            None,
            "active_contract",
            "futures_curve",
            "surface_forwards",
        ):
            raise ValidationError(
                "dividend_source must be None, 'active_contract', "
                "'futures_curve', or 'surface_forwards'"
            )
        if self.futures_curve_extrapolation not in (
            "flat_q", "flat_forward_carry", "surface_forward_carry"
        ):
            raise ValidationError(
                "futures_curve_extrapolation must be 'flat_q', 'flat_forward_carry' "
                "or 'surface_forward_carry'"
            )
        if int(self.futures_curve_min_tenor_days) < 1:
            # a contract expiring today has T = 0 and no defined yield
            raise ValidationError("futures_curve_min_tenor_days must be at least 1")
        if self.vol_model_calibration is None:
            self.vol_model_calibration = VolModelCalibrationConfig()
        elif not isinstance(self.vol_model_calibration, VolModelCalibrationConfig):
            raise ValidationError(
                "vol_model_calibration must be a VolModelCalibrationConfig"
            )
        if self.vol_model != "bsm":
            if self.vol_source != "surface":
                raise ValidationError(
                    "vol_model != 'bsm' requires vol_source='surface' "
                    "(per-day calibration is keyed by surface artifact sha)"
                )
            # Vol-model engines consume either the full smile grid (LV) or
            # their own model dynamics (Heston/SLV); forcing full_grid keeps
            # the env vol object consistent for the daily record.
            self.surface_vol_mode = "full_grid"

    def resolve_vol_model_mc_method(self) -> Optional[Any]:
        """MC method for the vol-model engines (falls back to ``method``)."""
        if self.vol_model_mc_method is not None:
            return self.vol_model_mc_method
        return self.method

    def resolve_surface_engine_type(self) -> EngineType:
        if self.surface_engine_type is not None:
            return self.surface_engine_type
        if self.pricing_engine_type == EngineType.MONTE_CARLO:
            return EngineType.QUADRATURE
        return self.pricing_engine_type

    def resolve_event_stats_engine_type(self) -> EngineType:
        if self.event_stats_engine_type is not None:
            return self.event_stats_engine_type
        return self.pricing_engine_type


def _validate_term_dividend_source(
    engine_config: AutocallableEngineConfig,
    *,
    fixed_dividend_yield: Optional[float],
    calculate_surfaces: bool,
    dividend_roll_policy: Optional[FuturesRollPolicy] = None,
) -> None:
    """Reject run options that would silently flatten a term dividend source.

    ``fixed_dividend_yield`` is a scalar override of the active-contract
    yield, and the spot x q surface grid reprices on flat ``q`` nodes around a
    scalar centre; under ``futures_curve`` / ``surface_forwards`` either one
    would swap the pricing model behind the recorded provenance.
    """
    if not engine_config.uses_term_dividend_source():
        return
    source = engine_config.dividend_source
    if fixed_dividend_yield is not None:
        raise ValidationError(
            f"fixed_dividend_yield cannot be combined with dividend_source="
            f"{source!r}; the term structure IS the dividend input"
        )
    if calculate_surfaces:
        raise ValidationError(
            f"calculate_surfaces is not supported with dividend_source={source!r}: "
            "the spot x q surface grid prices on flat q nodes"
        )
    if dividend_roll_policy is not None:
        raise ValidationError(
            f"dividend_roll_policy cannot be combined with dividend_source="
            f"{source!r}: a term source reads the WHOLE chain, so naming one "
            "contract for the carry would be silently ignored"
        )


@dataclass
class SurfaceGridConfig:
    """Compact surface grid configuration."""

    spot_nodes: int = 5
    spot_width: float = 0.05
    q_nodes: int = 3
    q_width: float = 0.01

    def __post_init__(self) -> None:
        if self.spot_nodes < 3:
            raise ValidationError("spot_nodes must be at least 3")
        if self.q_nodes < 1:
            raise ValidationError("q_nodes must be positive")
        if self.spot_width <= 0:
            raise ValidationError("spot_width must be positive")
        if self.q_width < 0:
            raise ValidationError("q_width must be non-negative")


@dataclass
class AutocallableBacktestConfig:
    """
    Full configuration for an OTC autocallable historical hedge replay.
    """

    product: Any
    market_data: AutocallableMarketDataSet
    engine_config: AutocallableEngineConfig = field(
        default_factory=AutocallableEngineConfig
    )
    strategy: Optional[Any] = None
    roll_policy: FuturesRollPolicy = field(default_factory=FuturesRollPolicy)
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    product_quantity: float = -1.0
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_product_price: Optional[float] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = True
    calculate_event_probabilities: bool = True
    terminate_on_lifecycle_end: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Appended AFTER metadata so existing positional construction keeps its slots.
    pnl_explain: Optional[Any] = None   # PnLExplainConfig; None = no explain, no behaviour change
    # Which contract the FLAT carry channel is inverted from.  None (default)
    # = the contract the hedge holds, which is the historical behaviour: the
    # engine hands one selected row to both the hedge trade and build_env.
    # Set it to price the carry off a contract the hedge does not trade, e.g.
    # read from the longest listed contract while the delta sits in the front
    # month.  A term dividend_source reads the WHOLE chain and would ignore
    # it, so that combination is rejected rather than silently dropped.
    dividend_roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self) -> None:
        if self.product is None:
            raise ValidationError("product is required")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.product_quantity == 0:
            raise ValidationError("product_quantity must be non-zero")
        if self.fixed_dividend_yield is not None and not math.isfinite(
            float(self.fixed_dividend_yield)
        ):
            raise ValidationError("fixed_dividend_yield must be finite")
        _validate_term_dividend_source(
            self.engine_config,
            fixed_dividend_yield=self.fixed_dividend_yield,
            calculate_surfaces=self.calculate_surfaces,
            dividend_roll_policy=self.dividend_roll_policy,
        )
        for field_name in ("delta_bump_size", "gamma_bump_size"):
            bump = getattr(self, field_name)
            if bump is None:
                continue
            bump_value = float(bump)
            if not math.isfinite(bump_value) or bump_value <= 0.0:
                raise ValidationError(f"{field_name} must be a positive finite value")


@dataclass
class ReplayProduct:
    product: Any
    quantity: float
    position_id: int
    has_lifecycle: bool
    initial_price: Optional[float] = None

    def __post_init__(self):
        if self.product is None:
            raise ValidationError("ReplayProduct.product is required")
        if self.quantity == 0:
            raise ValidationError("ReplayProduct.quantity must be non-zero")


@dataclass
class HedgeSpec:
    kind: str = "futures"
    multiplier: float = 1.0
    roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self):
        if self.kind not in ("futures", "spot"):
            raise ValidationError(f"HedgeSpec.kind must be futures|spot, got {self.kind}")
        if self.kind == "futures" and self.roll_policy is None:
            self.roll_policy = FuturesRollPolicy()


@dataclass
class ReplayBacktestConfig:
    products: list[ReplayProduct]
    market_data: AutocallableMarketDataSet
    hedge: HedgeSpec = field(default_factory=HedgeSpec)
    engine_config: AutocallableEngineConfig = field(default_factory=AutocallableEngineConfig)
    strategy: Any = None
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = False
    calculate_event_probabilities: bool = True
    terminate_on_lifecycle_end: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    # Appended AFTER metadata so existing positional construction keeps its slots.
    pnl_explain: Optional[Any] = None   # PnLExplainConfig; None = no explain, no behaviour change
    # Which contract the FLAT carry channel is inverted from.  None (default)
    # = the contract the hedge holds, which is the historical behaviour: the
    # engine hands one selected row to both the hedge trade and build_env.
    # Set it to price the carry off a contract the hedge does not trade, e.g.
    # read from the longest listed contract while the delta sits in the front
    # month.  A term dividend_source reads the WHOLE chain and would ignore
    # it, so that combination is rejected rather than silently dropped.
    dividend_roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self):
        if not self.products:
            raise ValidationError("ReplayBacktestConfig.products must be non-empty")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.strategy is None:
            self.strategy = AutocallableDeltaHedgeStrategy()
        _validate_term_dividend_source(
            self.engine_config,
            fixed_dividend_yield=self.fixed_dividend_yield,
            calculate_surfaces=self.calculate_surfaces,
            dividend_roll_policy=self.dividend_roll_policy,
        )
        if self.engine_config.vol_model != "bsm":
            # create_vol_model_engine builds Snowball engines only; pricing a
            # Phoenix/European with one would be silently wrong (fail-closed).
            from quantark.asset.equity.product.option.snowball_option import (
                SnowballOption,
            )

            for bp in self.products:
                if not isinstance(bp.product, SnowballOption):
                    raise ValidationError(
                        "vol_model != 'bsm' supports SnowballOption products "
                        f"only; position_id={bp.position_id} holds "
                        f"{type(bp.product).__name__}. Wire the corresponding "
                        "vol-model engines before booking other product types."
                    )


# Compatible aliases (canonical names above).
BookProduct = ReplayProduct
BookAutocallableBacktestConfig = ReplayBacktestConfig
