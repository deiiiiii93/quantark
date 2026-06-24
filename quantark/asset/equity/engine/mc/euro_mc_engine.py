"""
Monte Carlo pricing engine for European vanilla options.
"""

import math
from typing import Optional, Union, Tuple
import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import safe_exp

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from quantark.asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig


class EuropeanMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for European vanilla options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    NOTE: numerical Black engine — a smile surface is collapsed to a single
    strike-selected constant vol (sigma = get_vol(K, T)). Not smile-aware; for
    smile-consistent dynamics use LV / SLV / Heston / SABR Monte-Carlo.

    Usage:
        # Preferred: Two-level enum pattern
        engine = EuropeanMCEngine(
            params=MCParams(num_paths=10000, time_steps=252),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = EuropeanMCEngine(
            params=MCParams(num_paths=10000),
            method=MonteCarloMethod.QUASI
        )

        # Backward compatibility: String
        engine = EuropeanMCEngine(method="quasi")

    The engine creates a GBMPathGenerator internally based on the pricing
    environment and MCParams configuration.
    """

    engine_type = EngineType.MONTE_CARLO

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
    ):
        """
        Initialize Monte Carlo engine.

        Args:
            params: Monte Carlo configuration parameters (MCParams)
            method: Monte Carlo method selection, one of:
                - EngineType.MONTE_CARLO(MonteCarloMethod.XXX) (preferred)
                - MonteCarloMethod.XXX
                - String: "pseudo", "quasi", "randomized_quasi"
                - None: defaults to MonteCarloMethod.PSEUDO

        Raises:
            ValidationError: If method is invalid or params are invalid
        """
        if params is None:
            params = MCParams()

        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )

        super().__init__(params)

        if method is None:
            self.method = self.DEFAULT_METHOD
        elif isinstance(method, tuple):
            engine_type, mc_method = method
            if engine_type != EngineType.MONTE_CARLO:
                raise ValidationError(
                    f"Expected EngineType.MONTE_CARLO, got {engine_type}"
                )
            if not isinstance(mc_method, MonteCarloMethod):
                raise ValidationError(
                    f"Expected MonteCarloMethod, got {type(mc_method).__name__}"
                )
            self.method = mc_method
        elif isinstance(method, MonteCarloMethod):
            self.method = method
        elif isinstance(method, str):
            try:
                self.method = MonteCarloMethod[method.upper()]
            except KeyError:
                valid_methods = [m.name for m in MonteCarloMethod]
                raise ValidationError(
                    f"Invalid method string '{method}'. Valid methods: {valid_methods}"
                )
        else:
            raise ValidationError(
                f"Invalid method type {type(method).__name__}. "
                "Expected MonteCarloMethod, tuple, str, or None"
            )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a European vanilla option using Monte Carlo simulation.

        Args:
            product: European vanilla option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a European vanilla option
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"EuropeanMCEngine only supports EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        from quantark.param.vol.collapse_guard import guard_constant_vol
        guard_constant_vol(
            pricing_env.vol_surface, type(self).__name__,
            strict=getattr(self.params, "strict_smile", False),
        )

        self._validate_inputs(S, K, T, r, q, sigma)

        if T < 1e-10:
            return product.get_payoff(S)

        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            price, std_error = self._price_rqmc(
                product, S, K, T, r, q, sigma
            )
        else:
            price, std_error = self._price_mc_or_qmc(
                product, S, K, T, r, q, sigma
            )

        contract_multiplier = product.contract_multiplier
        price *= contract_multiplier
        std_error *= contract_multiplier
        self._last_std_error = std_error

        if price < 0:
            raise PricingError(f"Negative price computed: {price}")

        lower_bound = self._european_lower_bound(product, S, K, T, r, q)
        if price < lower_bound - 1e-6:
            raise PricingError(
                f"Price ({price:.6f}) below discounted European lower bound "
                f"({lower_bound:.6f})"
            )

        return price

    def _european_lower_bound(
        self,
        product: EuropeanVanillaOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
    ) -> float:
        """Calculate the discounted no-arbitrage lower bound for a European option."""
        spot_pv = S * safe_exp(-q * T)
        strike_pv = K * safe_exp(-r * T)
        if product.is_call():
            lower_bound = max(spot_pv - strike_pv, 0.0)
        else:
            lower_bound = max(strike_pv - spot_pv, 0.0)
        return lower_bound * product.contract_multiplier

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if q < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {q}")

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        num_paths: Optional[int] = None,
    ) -> GBMPathGenerator:
        """
        Create a GBMPathGenerator configured for the current method.

        Args:
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            T: Time to maturity

        Returns:
            Configured GBMPathGenerator
        """
        params = self.params
        effective_num_paths = params.num_paths if num_paths is None else int(num_paths)
        if effective_num_paths <= 0:
            raise ValidationError(
                f"num_paths must be positive, got {effective_num_paths}"
            )

        if self.method == MonteCarloMethod.PSEUDO:
            random_stream = PseudoRandomNormalGenerator(seed=params.seed)
            is_qmc = False
        elif self.method in (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI):
            random_stream = SobolNormalGenerator(base_seed=params.seed)
            is_qmc = True
        else:
            raise ValidationError(f"Unknown Monte Carlo method: {self.method}")

        vr_config = None
        if params.use_antithetic and not is_qmc:
            vr_config = VarianceReductionConfig(antithetic=True)

        generator = GBMPathGenerator(
            initial_value=S,
            vol=sigma,
            rrf=r,
            div=q,
            maturity=T,
            time_steps=params.time_steps,
            num_paths=effective_num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=False,
            vr_config=vr_config,
            is_qmc=is_qmc,
        )

        return generator

    def _calculate_payoffs(
        self, product: EuropeanVanillaOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate option payoffs from terminal prices.

        Args:
            product: European vanilla option
            terminal_prices: Array of terminal spot prices, shape (num_paths,)

        Returns:
            Array of payoffs, shape (num_paths,)
        """
        K = product.strike

        if product.is_call():
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)

        return payoffs

    def _price_mc_or_qmc(
        self,
        product: EuropeanVanillaOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> Tuple[float, float]:
        """
        Price using normal MC or QMC (non-randomized).

        Args:
            product: European vanilla option
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Returns:
            Tuple of (price, standard_error)
        """
        generator = self._create_path_generator(S, r, q, sigma, T)

        paths, aux = generator.generate_paths(return_aux=True)

        terminal_prices = paths[:, -1]

        payoffs = self._calculate_payoffs(product, terminal_prices)

        discount_factor = math.exp(-r * T)
        discounted_payoffs = discount_factor * payoffs

        mean_payoff = float(discounted_payoffs.mean())
        std_payoff = float(discounted_payoffs.std(ddof=1))

        std_error = std_payoff / math.sqrt(len(payoffs))

        return mean_payoff, std_error

    def _price_rqmc(
        self,
        product: EuropeanVanillaOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> Tuple[float, float]:
        """
        Price using Randomized QMC with adaptive batching.

        Args:
            product: European vanilla option
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Returns:
            Tuple of (price, standard_error)
        """
        discount_factor = math.exp(-r * T)

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            terminal_prices = paths[:, -1]
            payoffs = self._calculate_payoffs(product, terminal_prices)
            discounted_payoffs = discount_factor * payoffs
            return discounted_payoffs

        params = self.params
        max_batches = getattr(
            params, "rqmc_max_batches", getattr(params, "max_batches", 32)
        )
        min_batches = getattr(
            params, "rqmc_min_batches", getattr(params, "min_batches", 4)
        )
        if hasattr(params, "resolve_rqmc_target_std"):
            target_std = params.resolve_rqmc_target_std(product=product)
        else:
            target_std = getattr(params, "target_std", 1e-4)
        if hasattr(params, "resolve_rqmc_paths_per_batch"):
            per_batch_paths = params.resolve_rqmc_paths_per_batch(
                max_batches=max_batches
            )
        else:
            per_batch_paths = params.num_paths

        generator = self._create_path_generator(
            S, r, q, sigma, T, num_paths=per_batch_paths
        )

        result = run_rqmc(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            target_std=target_std,
            min_batches=min_batches,
        )

        self._last_rqmc_result = result

        return result.price, result.std_error

    def get_last_std_error(self) -> Optional[float]:
        """
        Get the standard error from the last pricing run.

        Returns:
            Standard error, or None if no pricing has been performed yet
        """
        return getattr(self, '_last_std_error', None)

    def get_last_rqmc_result(self):
        """
        Get the full RQMC result from the last RQMC pricing run.

        Returns:
            RQMCResult object, or None if last pricing was not RQMC
        """
        return getattr(self, '_last_rqmc_result', None)

    def __repr__(self):
        return f"EuropeanMCEngine(method={self.method.name})"
