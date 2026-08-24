"""
Base class for pricing engines.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import update_wrapper
import inspect
from math import isfinite
from typing import Dict, Optional, Sequence, TYPE_CHECKING
import numpy as np
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
from quantark.util.enum.engine_enums import EngineType
from quantark.util.numerical import is_close

if TYPE_CHECKING:
    from quantark.asset.equity.lifecycle import EquityOptionLifecycleState


def _install_lifecycle_keyword(cls, method_name: str) -> None:
    """Add the shared keyword/guard to legacy concrete equity methods."""
    original = cls.__dict__.get(method_name)
    if original is None:
        return
    signature = inspect.signature(original)
    if (
        "product" not in signature.parameters
        or "lifecycle_state" in signature.parameters
    ):
        return
    if not cls.__module__.startswith("quantark.asset.equity.engine."):
        return

    parameters = list(signature.parameters.values())
    lifecycle_parameter = inspect.Parameter(
        "lifecycle_state",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
    )
    for index, parameter in enumerate(parameters):
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parameters.insert(index, lifecycle_parameter)
            break
    else:
        parameters.append(lifecycle_parameter)
    public_signature = signature.replace(parameters=parameters)

    def lifecycle_aware_method(
        self,
        *args,
        lifecycle_state=None,
        **kwargs,
    ):
        bound = signature.bind_partial(self, *args, **kwargs)
        product = bound.arguments.get("product")
        if product is not None:
            from quantark.asset.equity.engine.settlement_support import (
                validate_settlement_capability,
            )

            validate_settlement_capability(
                self,
                product,
                lifecycle_state,
            )
        return original(self, *args, **kwargs)

    update_wrapper(lifecycle_aware_method, original)
    lifecycle_aware_method.__signature__ = public_signature
    setattr(cls, method_name, lifecycle_aware_method)


class BaseEngine(ABC):
    """
    Abstract base class for all pricing engines.

    Engines are responsible for computing prices and Greeks for derivatives.

    Attributes:
        engine_type: The type category of this engine (ANALYTICAL, MONTE_CARLO, PDE, etc.)
    """

    engine_type: EngineType = EngineType.ANALYTICAL
    supports_spot_greeks_grid = False
    settlement_support = SettlementSupport.NONE
    supports_lifecycle_state = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for method_name in ("price", "price_with_events", "calculate_greeks"):
            _install_lifecycle_keyword(cls, method_name)

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize the engine.

        Args:
            params: Engine configuration parameters
        """
        self.params = params if params is not None else EngineParams()

    @abstractmethod
    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state: Optional["EquityOptionLifecycleState"] = None,
    ) -> float:
        """
        Calculate the price of the product.

        Args:
            product: The derivative product to price
            pricing_env: Pricing environment with market data

        Returns:
            Product price
        """
        pass

    def create_bump_context(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> "BaseEngine":
        """
        Return an engine context for finite-difference bump repricing.

        Engines with market-dependent numerical grids can override this hook to
        freeze the base valuation grid/domain before bumped repricing. Engines
        without such state explicitly fall back to the current repricing engine.
        """
        return self

    def execute(self, request, context):
        """Route a framework ``PricingRequest`` through the execution kernel.

        Non-abstract compatibility entry point (execution-framework spec
        section 5.4). Existing subclasses need no change; the kernel resolves
        a capability adapter for this engine and falls back to the serial
        LegacyPriceAdapter. Direct ``price``/``price_detailed`` calls are
        unaffected.
        """
        from quantark.execution.kernel import ExecutionKernel

        return ExecutionKernel.dispatch(self, request, context)

    def price_with_events(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        emit_distribution: bool = True,
        streams: "Optional[frozenset]" = None,
        *,
        lifecycle_state: Optional["EquityOptionLifecycleState"] = None,
    ) -> "PricingResult":
        """
        Return product NPV and an event distribution for cash-leg valuation.

        Engines that already implement calculate_event_stats are adapted to the
        generalized EventDistribution. Engines without event stats fall back to
        a maturity-only distribution, which is sufficient for deterministic and
        full-schedule cash legs.

        ``streams`` (the EventType set the caller needs, [§11.1]) is honored by
        engines that support column pruning (the PDE autocallable solvers and
        the Snowball/Phoenix QUAD engines, which override this method); other
        engines ignore it and return the full distribution.
        """
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult
        from quantark.asset.equity.engine.settlement_support import (
            validate_settlement_capability,
        )

        validate_settlement_capability(self, product, lifecycle_state)

        if emit_distribution:
            stats = self.calculate_event_stats(product, pricing_env)
            if stats is not None:
                return PricingResult(
                    npv=float(stats.pv),
                    event_distribution=EventDistribution.from_autocallable_stats(stats),
                )

        if lifecycle_state is None:
            npv = self.price(product, pricing_env)
        else:
            npv = self.price(product, pricing_env, lifecycle_state=lifecycle_state)
        return PricingResult(
            npv=npv,
            event_distribution=EventDistribution.trivial(product.get_maturity(pricing_env)),
        )

    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state: Optional["EquityOptionLifecycleState"] = None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method.

        This default implementation uses bump-and-reprice.
        Subclasses can override to provide analytical Greeks.

        Args:
            product: The derivative product
            pricing_env: Pricing environment with market data

        Returns:
            Dictionary of Greeks
        """
        from copy import deepcopy

        bump_config = self.params.get_effective_bump_config()
        spot_bump = bump_config.spot_bump
        gamma_bump = (
            bump_config.gamma_spot_bump
            if bump_config.gamma_spot_bump is not None
            else spot_bump
        )
        # Resolve the numerical domain once at the base market. PDE engines use
        # this hook to freeze their spatial layout so delta/gamma measure market
        # sensitivity rather than a mixture of market and grid movement.
        bump_engine = self.create_bump_context(product, pricing_env)
        if bump_engine is None:
            bump_engine = self

        # Ask for the keyword only when a state is actually supplied: engine
        # subclasses defined outside quantark.asset.equity.engine.* never get
        # the lifecycle-keyword retrofit, and a None state must not break them.
        def _price(env):
            if lifecycle_state is None:
                return bump_engine.price(product, env)
            return bump_engine.price(product, env, lifecycle_state=lifecycle_state)

        base_price = _price(pricing_env)
        greeks = {"price": base_price}

        # Delta: dV/dS
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1 + spot_bump
        price_up = _price(env_up)

        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1 - spot_bump
        price_down = _price(env_down)

        delta = (price_up - price_down) / (2 * pricing_env.spot * spot_bump)
        greeks["delta"] = delta

        # Gamma: d²V/dS² — optionally on its own bump width
        if is_close(gamma_bump, spot_bump, rel_tol=1e-5, abs_tol=1e-8):
            gamma_up, gamma_down = price_up, price_down
        else:
            env_gup = deepcopy(pricing_env)
            env_gup.spot_quote.spot *= 1 + gamma_bump
            gamma_up = _price(env_gup)

            env_gdown = deepcopy(pricing_env)
            env_gdown.spot_quote.spot *= 1 - gamma_bump
            gamma_down = _price(env_gdown)
        gamma = (gamma_up - 2 * base_price + gamma_down) / (
            pricing_env.spot * gamma_bump
        ) ** 2
        greeks["gamma"] = gamma

        return greeks

    def calculate_spot_greeks_curve(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot_levels: Sequence[float],
    ) -> list[dict[str, float | str]]:
        """Calculate Delta and Gamma along spot levels.

        Engines with a reusable valuation grid override this method. The default
        contract deliberately falls back to point Greeks so every engine remains
        supported.
        """
        spots = np.asarray([float(spot) for spot in spot_levels], dtype=float)
        if spots.size == 0:
            return []
        if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
            raise ValueError("spot levels must be positive and finite")
        if self.supports_spot_greeks_grid:
            self._last_spot_greeks_grid = None
            self.price(product, pricing_env)
            cached = self._last_spot_greeks_grid
            if cached is not None:
                grid_spots, grid_prices = cached
                deltas = np.gradient(grid_prices, grid_spots, edge_order=2)
                gammas = np.gradient(deltas, grid_spots, edge_order=2)
                return [
                    {
                        "spot": float(spot),
                        "price": float(np.interp(spot, grid_spots, grid_prices)),
                        "delta": float(np.interp(spot, grid_spots, deltas)),
                        "gamma": float(np.interp(spot, grid_spots, gammas)),
                        "calculation_mode": "engine_grid",
                    }
                    for spot in spots
                ]

        rows: list[dict[str, float | str]] = []
        for raw_spot in spots:
            spot = float(raw_spot)
            if not isfinite(spot) or spot <= 0.0:
                raise ValueError(f"spot levels must be positive and finite, got {raw_spot}")
            env = deepcopy(pricing_env)
            env.spot_quote.spot = spot
            greeks = self.calculate_greeks(product, env)
            price = greeks.get("price")
            if price is None:
                price = self.price(product, env)
            rows.append(
                {
                    "spot": spot,
                    "price": float(price),
                    "delta": float(greeks.get("delta", 0.0)),
                    "gamma": float(greeks.get("gamma", 0.0)),
                    "calculation_mode": "reprice",
                }
            )
        return rows

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """
        Optionally provide per-observation event stats and cashflow decomposition.

        Engines MAY override this method to provide per-observation probabilities and
        expected discounted cashflows for autocallable products. This enables faster
        reporting (especially for QUAD/PDE engines) compared to Monte Carlo analyzers.

        Default behavior: return None (not supported).
        """
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}()"
