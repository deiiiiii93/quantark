"""
Monte Carlo pricing engine for Asian options.

This engine prices Asian options using Monte Carlo simulation with support for:
- Fixed strike (average price option) and floating strike (average strike option)
- Arithmetic and geometric averaging
- Custom or uniform observation schedules
- Historical (past) observations with recorded prices
- Three Monte Carlo methods (PSEUDO, QUASI, RANDOMIZED_QUASI)
- Vectorized NumPy operations for efficiency

For options where valuation_date > initial_date, past observations with recorded
prices are combined with simulated future prices for averaging calculations.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option.asian_option import AsianOption
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AveragingType, AsianStrikeType
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType
from quantark.util.exceptions import ValidationError, PricingError
from quantark.util.numerical import is_zero

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from quantark.asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig


@dataclass
class AsianMCResult:
    """Result container for Asian option MC pricing."""

    price: float
    std_error: float
    num_paths: int
    average_price_mean: float  # Mean of simulated averages
    average_price_std: float  # Std of simulated averages
    num_past_observations: int = 0  # Number of already-observed prices
    num_future_observations: int = 0  # Number of simulated observations
    batches_used: Optional[int] = None


class AsianOptionMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for Asian options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Asian options are path-dependent options where the payoff depends on the
    average price of the underlying asset over a specified observation period.

    Two main variants:
    - Fixed strike (average price option): payoff based on average vs strike
    - Floating strike (average strike option): payoff based on final spot vs average

    Usage:
        # Preferred: Two-level enum pattern
        engine = AsianOptionMCEngine(
            params=MCParams(num_paths=100000, time_steps=252),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = AsianOptionMCEngine(
            params=MCParams(num_paths=100000),
            method=MonteCarloMethod.QUASI
        )

        # Backward compatibility: String
        engine = AsianOptionMCEngine(method="quasi")

    The engine creates a GBMPathGenerator internally based on the pricing
    environment and product observation schedule.
    """

    engine_type = EngineType.MONTE_CARLO

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
    ):
        """
        Initialize Asian option Monte Carlo engine.

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

        # Result storage
        self._last_result: Optional[AsianMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an Asian option using Monte Carlo simulation.

        Args:
            product: Asian option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not an AsianOption
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, AsianOption):
            raise PricingError(
                f"AsianOptionMCEngine only supports AsianOption, "
                f"got {type(product).__name__}"
            )

        # Extract market data
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, K)
        self._df = make_df_fn(pricing_env)

        self._validate_inputs(S, K, T, r, q, sigma, product)

        # Handle near-expiry case
        if is_zero(T):
            # At expiry, return payoff with current spot as both spot and average
            return product.get_payoff(S, average=S)

        # Price using appropriate method
        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            result = self._price_rqmc(product, pricing_env, S, K, T, r, q, sigma)
        else:
            result = self._price_mc_or_qmc(product, pricing_env, S, K, T, r, q, sigma)

        self._last_result = result

        if result.price < 0:
            raise PricingError(f"Negative price computed: {result.price}")

        return result.price

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        product: AsianOption,
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")

        # Strike validation only for fixed strike options
        if product.is_fixed_strike() and K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")

    def _build_observation_grid(
        self,
        product: AsianOption,
        pricing_env: PricingEnvironment,
        T: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Build time grid aligned with future observation times.

        Uses resolve_observations() to separate past (already observed) from
        future (to be simulated) observations.

        Args:
            product: Asian option product
            pricing_env: Pricing environment with valuation date
            T: Time to maturity

        Returns:
            Tuple of:
            - all_times: Sorted unique times including future observations and maturity
            - dt_array: Time increments between times
            - obs_indices: Indices into all_times for future observations
            - past_prices: Array of already-observed prices
            - weights: Normalized averaging weights aligned with [past, future] order
            - total_observations: Total number of observations for averaging
        """
        # Use resolve_observations to get past and future observations
        past_prices_list, past_weights, future_times_list, future_weights, total_observations = (
            product.resolve_observations(pricing_env)
        )

        past_prices = np.array(past_prices_list)
        future_times = np.array(future_times_list)
        # Weights are aligned with the average's column order: past first, then
        # future (matching how _compute_averages concatenates the price columns).
        weights = np.array(list(past_weights) + list(future_weights), dtype=float)

        # If no future observations, we only need to compute payoff from past prices
        if len(future_times) == 0:
            # No simulation needed, return empty arrays
            return (
                np.array([T]),
                np.array([T]),
                np.array([], dtype=int),
                past_prices,
                weights,
                total_observations,
            )

        # Ensure maturity is included in the time grid
        all_times_set = set(future_times.tolist()) | {T}
        all_times = np.array(sorted(all_times_set))

        # Build dt_array
        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        # Find indices for future observation times
        obs_indices = np.searchsorted(all_times, future_times)

        return all_times, dt_array, obs_indices, past_prices, weights, total_observations

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ) -> GBMPathGenerator:
        """
        Create a GBMPathGenerator configured for the observation grid.

        Args:
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            T: Time to maturity
            dt_array: Non-uniform time increments
            batch_id: Batch identifier for RQMC

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
            seed = params.seed + (batch_id or 0) * 1000
            random_stream = PseudoRandomNormalGenerator(seed=seed)
            is_qmc = False
        elif self.method in (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI):
            random_stream = SobolNormalGenerator(base_seed=params.seed)
            is_qmc = True
        else:
            raise ValidationError(f"Unknown Monte Carlo method: {self.method}")

        # Use antithetic variates for non-QMC
        vr_config = None
        if params.use_antithetic and not is_qmc:
            vr_config = VarianceReductionConfig(antithetic=True)

        term_ctx = getattr(self, "_term_ctx", None)
        if term_ctx is not None:
            env_ctx, ref_strike = term_ctx
            term = build_mc_term_inputs(
                env_ctx, ref_strike=ref_strike, maturity=T,
                time_steps=len(dt_array), dt_array=dt_array,
            )
            vol_in, rrf_in, div_in = term.vol, term.rrf, term.div
        else:
            vol_in, rrf_in, div_in = sigma, r, q

        generator = GBMPathGenerator(
            initial_value=S,
            vol=vol_in,
            rrf=rrf_in,
            div=div_in,
            maturity=T,
            time_steps=len(dt_array),
            num_paths=effective_num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=False,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

        return generator

    def _compute_averages(
        self,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        past_prices: np.ndarray,
        weights: np.ndarray,
        total_observations: int,
        averaging_type: AveragingType,
    ) -> np.ndarray:
        """
        Compute the (weighted) average price over all observations per path.

        Combines past (already observed) prices with simulated future prices
        to compute the full average. ``weights`` are normalized averaging weights
        aligned with the [past, future] column order and summing to 1.

        Args:
            paths: Simulated paths, shape (num_paths, num_times + 1)
            obs_indices: Indices into paths for future observations
            past_prices: Array of already-observed prices, shape (num_past,)
            weights: Normalized weights, shape (num_past + num_future,)
            total_observations: Total number of observations for averaging
            averaging_type: ARITHMETIC or GEOMETRIC

        Returns:
            Array of averages, shape (num_paths,)
        """
        num_paths = paths.shape[0]
        num_past = len(past_prices)
        num_future = len(obs_indices)

        # Handle case with no future observations
        if num_future == 0:
            # All observations are past - return constant (weighted) average
            avg = self._weighted_average_1d(
                np.asarray(past_prices, dtype=float), weights, averaging_type
            )
            return np.full(num_paths, avg)

        # Extract simulated prices at future observation times (offset by 1 for t=0)
        future_prices = paths[:, obs_indices + 1]  # (num_paths, num_future)

        # Combine past and future prices
        if num_past > 0:
            # Broadcast past_prices across all paths
            past_broadcast = np.broadcast_to(
                past_prices, (num_paths, num_past)
            )  # (num_paths, num_past)
            all_prices = np.concatenate(
                [past_broadcast, future_prices], axis=1
            )  # (num_paths, total_observations)
        else:
            all_prices = future_prices

        # Compute weighted average (weights aligned with all_prices columns)
        if averaging_type == AveragingType.ARITHMETIC:
            return all_prices @ weights
        else:  # GEOMETRIC
            # Weighted log-mean for numerical stability
            log_prices = np.log(all_prices)
            return np.exp(log_prices @ weights)

    @staticmethod
    def _weighted_average_1d(
        prices: np.ndarray, weights: np.ndarray, averaging_type: AveragingType
    ) -> float:
        """Weighted scalar average of a 1-D price vector (weights sum to 1)."""
        if len(prices) == 0 or len(weights) == 0:
            # An empty schedule has no average to take: np.dot([], []) would
            # silently return 0.0 (and geometric exp(0)==1.0), fabricating a
            # payoff against a non-existent average. Reject it explicitly.
            raise ValidationError(
                "Cannot compute a weighted average over an empty observation "
                "schedule (no past prices and no future fixings)."
            )
        if averaging_type == AveragingType.ARITHMETIC:
            return float(np.dot(weights, prices))
        log_prices = np.log(prices)
        return float(np.exp(np.dot(weights, log_prices)))

    def _compute_payoffs(
        self,
        product: AsianOption,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        past_prices: np.ndarray,
        weights: np.ndarray,
        total_observations: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute payoffs for all paths based on strike type and averaging.

        Args:
            product: AsianOption product
            paths: Simulated paths, shape (num_paths, num_times + 1)
            obs_indices: Indices into paths for future observations
            past_prices: Array of already-observed prices
            weights: Normalized averaging weights aligned with [past, future]
            total_observations: Total number of observations for averaging

        Returns:
            Tuple of:
            - payoffs: Undiscounted payoffs, shape (num_paths,)
            - averages: Computed averages, shape (num_paths,)
        """
        # Compute averages (combining past and future prices)
        averages = self._compute_averages(
            paths, obs_indices, past_prices, weights, total_observations, product.averaging_type
        )

        # Get terminal spot prices
        terminal_spots = paths[:, -1]

        # Compute payoffs based on strike type
        if product.is_fixed_strike():
            # Fixed strike (average price option):
            # Call: max(average - K, 0), Put: max(K - average, 0)
            K = product.strike
            if product.is_call():
                payoffs = np.maximum(averages - K, 0.0)
            else:
                payoffs = np.maximum(K - averages, 0.0)
        else:
            # Floating strike (average strike option):
            # Call: max(spot - average, 0), Put: max(average - spot, 0)
            if product.is_call():
                payoffs = np.maximum(terminal_spots - averages, 0.0)
            else:
                payoffs = np.maximum(averages - terminal_spots, 0.0)

        return payoffs, averages

    def _price_mc_or_qmc(
        self,
        product: AsianOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> AsianMCResult:
        """
        Price using normal MC or QMC (non-randomized).
        """
        # Build observation grid (separates past and future observations)
        all_times, dt_array, obs_indices, past_prices, weights, total_observations = (
            self._build_observation_grid(product, pricing_env, T)
        )

        num_past = len(past_prices)
        num_future = len(obs_indices)

        # Handle special case: all observations are in the past
        if num_future == 0:
            # Compute (weighted) average from past prices only
            avg = self._weighted_average_1d(
                np.asarray(past_prices, dtype=float), weights, product.averaging_type
            )

            # Compute payoff
            payoff = product.get_payoff(S, average=avg)

            # Discount payoff
            discount_factor = self._df(T)
            price = discount_factor * payoff

            return AsianMCResult(
                price=price,
                std_error=0.0,  # No simulation uncertainty
                num_paths=0,
                average_price_mean=avg,
                average_price_std=0.0,
                num_past_observations=num_past,
                num_future_observations=0,
            )

        # Create path generator for future observations
        maturity_for_sim = all_times[-1]  # Time to last future observation
        generator = self._create_path_generator(
            S, r, q, sigma, maturity_for_sim, dt_array
        )

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False)

        # Compute payoffs (combines past and future prices)
        payoffs, averages = self._compute_payoffs(
            product, paths, obs_indices, past_prices, weights, total_observations
        )

        # Discount payoffs
        discount_factor = self._df(T)
        discounted_payoffs = discount_factor * payoffs

        # Compute price and standard error
        price = float(discounted_payoffs.mean())
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs))

        return AsianMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            average_price_mean=float(averages.mean()),
            average_price_std=float(averages.std(ddof=1)),
            num_past_observations=num_past,
            num_future_observations=num_future,
        )

    def _price_rqmc(
        self,
        product: AsianOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> AsianMCResult:
        """
        Price using Randomized QMC with adaptive batching.
        """
        # Build observation grid (separates past and future observations)
        all_times, dt_array, obs_indices, past_prices, weights, total_observations = (
            self._build_observation_grid(product, pricing_env, T)
        )

        num_past = len(past_prices)
        num_future = len(obs_indices)

        # Handle special case: all observations are in the past
        if num_future == 0:
            # Same as MC case - no simulation needed (weighted average)
            avg = self._weighted_average_1d(
                np.asarray(past_prices, dtype=float), weights, product.averaging_type
            )

            payoff = product.get_payoff(S, average=avg)
            discount_factor = self._df(T)
            price = discount_factor * payoff

            return AsianMCResult(
                price=price,
                std_error=0.0,
                num_paths=0,
                average_price_mean=avg,
                average_price_std=0.0,
                num_past_observations=num_past,
                num_future_observations=0,
            )

        params = self.params
        max_batches = getattr(
            params, "rqmc_max_batches", getattr(params, "max_batches", 32)
        )
        min_batches = getattr(
            params, "rqmc_min_batches", getattr(params, "min_batches", 4)
        )
        if hasattr(params, "resolve_rqmc_target_std"):
            target_std = params.resolve_rqmc_target_std(
                product=product, pricing_env=pricing_env
            )
        else:
            target_std = getattr(params, "target_std", 1e-4)
        if hasattr(params, "resolve_rqmc_paths_per_batch"):
            per_batch_paths = params.resolve_rqmc_paths_per_batch(
                max_batches=max_batches
            )
        else:
            per_batch_paths = params.num_paths

        # Create path generator (will be used with different batch_ids)
        maturity_for_sim = all_times[-1]
        generator = self._create_path_generator(
            S,
            r,
            q,
            sigma,
            maturity_for_sim,
            dt_array,
            num_paths=per_batch_paths,
        )

        discount_factor = self._df(T)

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            payoffs, _ = self._compute_payoffs(
                product, paths, obs_indices, past_prices, weights, total_observations
            )
            return discount_factor * payoffs

        result = run_rqmc(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            target_std=target_std,
            min_batches=min_batches,
        )

        # Run one more batch to get average statistics
        paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
        _, averages = self._compute_payoffs(
            product, paths, obs_indices, past_prices, weights, total_observations
        )

        return AsianMCResult(
            price=result.price,
            std_error=result.std_error,
            num_paths=result.total_paths,
            average_price_mean=float(averages.mean()),
            average_price_std=float(averages.std(ddof=1)),
            num_past_observations=num_past,
            num_future_observations=num_future,
            batches_used=result.batches_used,
        )

    def get_last_result(self) -> Optional[AsianMCResult]:
        """
        Get the full result from the last pricing run.

        Returns:
            AsianMCResult object, or None if no pricing has been performed
        """
        return self._last_result

    def get_last_std_error(self) -> Optional[float]:
        """
        Get the standard error from the last pricing run.

        Returns:
            Standard error, or None if no pricing has been performed
        """
        if self._last_result is None:
            return None
        return self._last_result.std_error

    def __repr__(self):
        return f"AsianOptionMCEngine(method={self.method.name})"
