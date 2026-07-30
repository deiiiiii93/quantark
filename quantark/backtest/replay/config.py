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
from quantark.volmodels.calibration import HESTON_PRESETS


# Canonical home: quantark.backtest.futures_ledger (re-exported here).
from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: F401,E402


@dataclass
class VolModelCalibrationConfig:
    """
    Calibration options for the vol-model variants (Task 2.4).

    ``cache_dir`` selects the persistent calibration cache (JSON files
    keyed by surface sha + variant + config fingerprint); ``None`` means
    in-memory caching only.  ``records_path`` (optional) makes the engine
    persist per-day calibration records as one JSON file per run.
    ``heston_preset`` selects the frozen Heston calibration configuration
    (only ``"mo_frozen"`` — the mo_volmodels suite values).  ``slv_eta`` /
    ``slv_n_steps`` / ``slv_n_x`` / ``slv_n_z`` steer the forward
    Fokker-Planck leverage calibration (suite defaults 1.0 / 40 / 161 /
    81).  ``shared_store`` is an optional caller-held dict shared across
    runs as a common in-memory cache.
    """

    cache_dir: Optional[str] = None
    records_path: Optional[str] = None
    heston_preset: str = "mo_frozen"
    heston_max_nfev: int = 200
    slv_eta: float = 1.0
    slv_n_steps: int = 40
    slv_n_x: int = 161
    slv_n_z: int = 81
    shared_store: Optional[MutableMapping] = None

    def __post_init__(self) -> None:
        if self.heston_preset not in HESTON_PRESETS:
            raise ValidationError(
                f"heston_preset must be one of {sorted(HESTON_PRESETS)}"
            )
        if not isinstance(self.heston_max_nfev, int) or self.heston_max_nfev <= 0:
            raise ValidationError("heston_max_nfev must be a positive integer")
        if not math.isfinite(float(self.slv_eta)) or float(self.slv_eta) < 0.0:
            raise ValidationError("slv_eta must be non-negative and finite")
        if int(self.slv_n_steps) < 1:
            raise ValidationError("slv_n_steps must be >= 1")
        if int(self.slv_n_x) < 3:
            raise ValidationError("slv_n_x must be >= 3")
        if int(self.slv_n_z) < 3:
            raise ValidationError("slv_n_z must be >= 3")
        if self.shared_store is not None and not isinstance(
            self.shared_store, MutableMapping
        ):
            raise ValidationError("shared_store must be a mutable mapping")


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
        for field_name in ("delta_bump_size", "gamma_bump_size"):
            bump = getattr(self, field_name)
            if bump is None:
                continue
            bump_value = float(bump)
            if not math.isfinite(bump_value) or bump_value <= 0.0:
                raise ValidationError(f"{field_name} must be a positive finite value")


@dataclass
class BookProduct:
    product: Any
    quantity: float
    position_id: int
    has_lifecycle: bool
    initial_price: Optional[float] = None

    def __post_init__(self):
        if self.product is None:
            raise ValidationError("BookProduct.product is required")
        if self.quantity == 0:
            raise ValidationError("BookProduct.quantity must be non-zero")


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
class BookAutocallableBacktestConfig:
    products: list[BookProduct]
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
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.products:
            raise ValidationError("BookAutocallableBacktestConfig.products must be non-empty")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.strategy is None:
            self.strategy = AutocallableDeltaHedgeStrategy()
