"""
Configuration objects for OTC autocallable backtests.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from asset.equity.param import MCParams, PDEParams, QuadParams
from backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from util.enum.engine_enums import EngineType
from util.exceptions import ValidationError

from .market import AutocallableMarketDataSet


@dataclass
class FuturesRollPolicy:
    """
    Rule for selecting and rolling Chinese equity index futures contracts.
    """

    roll_days_before_expiry: int = 5

    def __post_init__(self) -> None:
        if self.roll_days_before_expiry < 0:
            raise ValidationError("roll_days_before_expiry must be non-negative")

    def select_contract(
        self, futures_slice, valuation_date, current_contract: Optional[str] = None
    ):
        """
        Select the active contract from a daily futures-chain slice.
        """
        valuation_date = valuation_date.normalize()
        rows = futures_slice.copy()
        rows = rows[rows["expiry_date"] > valuation_date]
        if rows.empty:
            raise ValidationError(
                f"No non-expired futures contract on {valuation_date.date()}"
            )

        if current_contract is not None:
            current = rows[rows["contract"] == current_contract]
            if not current.empty:
                current_row = current.sort_values("expiry_date").iloc[0]
                days_to_expiry = (
                    current_row["expiry_date"] - valuation_date
                ).days
                if days_to_expiry > self.roll_days_before_expiry:
                    return current_row

        min_expiry = valuation_date + timedelta(days=self.roll_days_before_expiry)
        candidates = rows[rows["expiry_date"] > min_expiry]
        if candidates.empty:
            candidates = rows
        return candidates.sort_values(["expiry_date", "contract"]).iloc[0]


@dataclass
class AutocallableEngineConfig:
    """
    Engine selection for pricing, surfaces, and event stats.
    """

    pricing_engine_type: EngineType = EngineType.PDE
    method: Optional[Any] = None
    pde_params: Optional[PDEParams] = None
    mc_params: Optional[MCParams] = None
    quad_params: Optional[QuadParams] = None
    surface_engine_type: Optional[EngineType] = None
    event_stats_engine_type: Optional[EngineType] = None

    def __post_init__(self) -> None:
        supported = {
            EngineType.PDE,
            EngineType.MONTE_CARLO,
            EngineType.QUADRATURE,
        }
        if self.pricing_engine_type not in supported:
            raise ValidationError(
                "pricing_engine_type must be PDE, MONTE_CARLO, or QUADRATURE"
            )
        if self.surface_engine_type is not None and self.surface_engine_type not in supported:
            raise ValidationError(
                "surface_engine_type must be PDE, MONTE_CARLO, or QUADRATURE"
            )
        if (
            self.event_stats_engine_type is not None
            and self.event_stats_engine_type not in supported
        ):
            raise ValidationError(
                "event_stats_engine_type must be PDE, MONTE_CARLO, or QUADRATURE"
            )

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
