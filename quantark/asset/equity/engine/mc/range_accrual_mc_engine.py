"""
Monte Carlo pricing engine for Range Accrual options.

This engine prices Range Accrual options using Monte Carlo simulation with support for:
- Weighted observations (e.g., Friday = 3 for weekend carry)
- Historical (past) observations with recorded in-range outcomes
- Time-varying upper and lower barriers
- Reverse mode (pay when outside range instead of inside)
- Annualized or non-annualized accrual rates
- Three Monte Carlo methods (PSEUDO, QUASI, RANDOMIZED_QUASI)
- Vectorized NumPy operations for efficiency

For options where valuation_date > initial_date, past observations with recorded
outcomes are combined with simulated future observations for payoff calculations.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero


@dataclass
class RangeAccrualMCResult:
    """Result container for Range Accrual option MC pricing."""

    price: float
    std_error: float
    num_paths: int
    in_range_ratio_mean: float  # Mean of in-range weight ratios across paths
    in_range_ratio_std: float  # Std of in-range weight ratios across paths
    num_past_observations: int = 0  # Number of already-observed outcomes
    num_future_observations: int = 0  # Number of simulated observations
    past_in_range_weights: float = 0.0  # Accumulated in-range weights from past
    total_weights: float = 0.0  # Total weight sum
    batches_used: Optional[int] = None


class RangeAccrualMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for Range Accrual options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Range Accrual options pay based on the proportion of observations where the
    underlying stays within a defined price range. The payoff formula is:

        Payoff = initial_price * contract_multiplier * accrual_rate
                 * (sum_in_range_weights / sum_total_weights) * year_fraction

    Usage:
        # Preferred: Two-level enum pattern
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=100000),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=100000),
            method=MonteCarloMethod.QUASI
        )

        # Backward compatibility: String
        engine = RangeAccrualMCEngine(method="quasi")

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
        Initialize Range Accrual Monte Carlo engine.

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
        self._last_result: Optional[RangeAccrualMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a Range Accrual option using Monte Carlo simulation.

        Args:
            product: Range Accrual option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a RangeAccrualOption
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, RangeAccrualOption):
            raise PricingError(
                f"RangeAccrualMCEngine only supports RangeAccrualOption, "
                f"got {type(product).__name__}"
            )

        # Extract market data
        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.initial_price, T)

        self._validate_inputs(S, T, r, q, sigma, product)

        # Handle near-expiry case
        if is_zero(T):
            # At expiry, use product's payoff calculation with known weights
            past_in_range, past_total = product.get_past_accrual(pricing_env)
            total_weights = product.get_total_weights()
            return product.get_payoff(
                S,
                in_range_weights=past_in_range,
                total_weights=total_weights,
                pricing_env=pricing_env,
            )

        # Price using appropriate method
        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            result = self._price_rqmc(product, pricing_env, S, T, r, q, sigma)
        else:
            result = self._price_mc_or_qmc(product, pricing_env, S, T, r, q, sigma)

        self._last_result = result

        # Range accrual payoffs are non-negative by construction
        if result.price < 0:
            raise PricingError(f"Negative price computed: {result.price}")

        return result.price

    def _validate_inputs(
        self,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        product: RangeAccrualOption,
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")

        if product.range_config is None:
            raise ValidationError("range_config is required for Range Accrual option")

    def _build_observation_grid(
        self,
        product: RangeAccrualOption,
        pricing_env: PricingEnvironment,
        T: float,
    ) -> Tuple[
        np.ndarray,  # all_times
        np.ndarray,  # dt_array
        np.ndarray,  # future_obs_indices
        List[Tuple[float, bool]],  # past_observations: (weight, in_range)
        List[Tuple[float, float, int]],  # future_observations: (weight, time, obs_idx)
        float,  # total_weights
    ]:
        """
        Build time grid aligned with future observation times.

        Uses resolve_observations() to separate past (already observed) from
        future (to be simulated) observations.

        Args:
            product: Range Accrual option product
            pricing_env: Pricing environment with valuation date
            T: Time to maturity

        Returns:
            Tuple of:
            - all_times: Sorted unique times including future observations and maturity
            - dt_array: Time increments between times
            - future_obs_indices: Indices into paths for future observations
            - past_observations: List of (weight, in_range) for past observations
            - future_observations: List of (weight, time, obs_idx) for future observations
            - total_weights: Sum of all observation weights
        """
        past_obs, future_obs, total_weights = product.resolve_observations(pricing_env)

        # If no future observations, we only need to compute payoff from past data
        if len(future_obs) == 0:
            return (
                np.array([T]),
                np.array([T]),
                np.array([], dtype=int),
                past_obs,
                future_obs,
                total_weights,
            )

        # Extract future observation times
        future_times = np.array([t for _, t, _ in future_obs])

        # Ensure maturity is included in the time grid
        all_times_set = set(future_times.tolist()) | {T}
        all_times = np.array(sorted(all_times_set))

        # Build dt_array
        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        # Find indices for future observation times in the path array
        # Paths have shape (num_paths, num_times + 1) where index 0 is t=0
        future_obs_indices = np.searchsorted(all_times, future_times) + 1

        return (
            all_times,
            dt_array,
            future_obs_indices,
            past_obs,
            future_obs,
            total_weights,
        )

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
            num_paths: Override for number of paths

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

        # Use antithetic variates for non-QMC (no barriers to worry about)
        vr_config = None
        if params.use_antithetic and not is_qmc:
            vr_config = VarianceReductionConfig(antithetic=True)

        generator = GBMPathGenerator(
            initial_value=S,
            vol=sigma,
            rrf=r,
            div=q,
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

    def _check_in_range(
        self,
        product: RangeAccrualOption,
        spots: np.ndarray,
        future_obs: List[Tuple[float, float, int]],
    ) -> np.ndarray:
        """
        Check which observations are in range for all paths.

        Args:
            product: Range Accrual option product
            spots: Spot prices at observation times, shape (num_paths, num_future_obs)
            future_obs: List of (weight, time, obs_idx) for future observations

        Returns:
            Boolean array of shape (num_paths, num_future_obs) indicating in-range
        """
        num_paths, num_future = spots.shape
        in_range = np.zeros((num_paths, num_future), dtype=bool)

        for i, (_, _, obs_idx) in enumerate(future_obs):
            # Get barriers for this observation
            upper = product.range_config.get_upper_barrier(obs_idx)
            lower = product.range_config.get_lower_barrier(obs_idx)

            # Check if in range
            spot_col = spots[:, i]
            in_range_col = (spot_col >= lower) & (spot_col <= upper)

            # Apply reverse mode if configured
            if product.range_config.is_reverse:
                in_range_col = ~in_range_col

            in_range[:, i] = in_range_col

        return in_range

    def _compute_payoffs(
        self,
        product: RangeAccrualOption,
        pricing_env: PricingEnvironment,
        paths: np.ndarray,
        future_obs_indices: np.ndarray,
        past_obs: List[Tuple[float, bool]],
        future_obs: List[Tuple[float, float, int]],
        total_weights: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute payoffs for all paths.

        Args:
            product: Range Accrual option product
            pricing_env: Pricing environment
            paths: Simulated paths, shape (num_paths, num_times + 1)
            future_obs_indices: Indices into paths for future observations
            past_obs: List of (weight, in_range) for past observations
            future_obs: List of (weight, time, obs_idx) for future observations
            total_weights: Sum of all observation weights

        Returns:
            Tuple of:
            - payoffs: Undiscounted payoffs, shape (num_paths,)
            - in_range_ratios: In-range weight ratios, shape (num_paths,)
        """
        num_paths = paths.shape[0]
        num_future = len(future_obs)

        # Calculate past contribution (same for all paths)
        past_in_range_weights = sum(w for w, in_range in past_obs if in_range)

        # Handle case with no future observations
        if num_future == 0:
            in_range_ratio = past_in_range_weights / total_weights if total_weights > 0 else 0.0
            in_range_ratios = np.full(num_paths, in_range_ratio)
        else:
            # Extract simulated prices at future observation times
            future_spots = paths[:, future_obs_indices]  # (num_paths, num_future)

            # Check in-range status for all future observations
            in_range = self._check_in_range(product, future_spots, future_obs)

            # Get future weights as array
            future_weights = np.array([w for w, _, _ in future_obs])

            # Calculate in-range weights for each path
            # Weight is added if in_range is True
            future_in_range_weights = (in_range * future_weights).sum(axis=1)

            # Total in-range weights = past + future
            total_in_range_weights = past_in_range_weights + future_in_range_weights

            # Compute in-range ratio
            in_range_ratios = total_in_range_weights / total_weights if total_weights > 0 else np.zeros(num_paths)

        # Compute payoffs using product's formula
        # Payoff = initial_price * contract_multiplier * accrual_rate * ratio * year_fraction
        year_fraction = product.get_year_fraction(pricing_env)
        accrual_rate = product.range_config.accrual_rate

        payoffs = (
            product.initial_price
            * product.contract_multiplier
            * accrual_rate
            * in_range_ratios
            * year_fraction
        )

        return payoffs, in_range_ratios

    def _price_mc_or_qmc(
        self,
        product: RangeAccrualOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> RangeAccrualMCResult:
        """
        Price using normal MC or QMC (non-randomized).
        """
        # Build observation grid
        (
            all_times,
            dt_array,
            future_obs_indices,
            past_obs,
            future_obs,
            total_weights,
        ) = self._build_observation_grid(product, pricing_env, T)

        num_past = len(past_obs)
        num_future = len(future_obs)
        past_in_range_weights = sum(w for w, in_range in past_obs if in_range)

        # Handle special case: all observations are in the past
        if num_future == 0:
            # Compute payoff from past observations only
            in_range_ratio = past_in_range_weights / total_weights if total_weights > 0 else 0.0

            year_fraction = product.get_year_fraction(pricing_env)
            accrual_rate = product.range_config.accrual_rate
            payoff = (
                product.initial_price
                * product.contract_multiplier
                * accrual_rate
                * in_range_ratio
                * year_fraction
            )

            # Discount payoff
            discount_factor = math.exp(-r * T)
            price = discount_factor * payoff

            return RangeAccrualMCResult(
                price=price,
                std_error=0.0,  # No simulation uncertainty
                num_paths=0,
                in_range_ratio_mean=in_range_ratio,
                in_range_ratio_std=0.0,
                num_past_observations=num_past,
                num_future_observations=0,
                past_in_range_weights=past_in_range_weights,
                total_weights=total_weights,
            )

        # Create path generator for future observations
        maturity_for_sim = all_times[-1]
        generator = self._create_path_generator(
            S, r, q, sigma, maturity_for_sim, dt_array
        )

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False)

        # Compute payoffs
        payoffs, in_range_ratios = self._compute_payoffs(
            product,
            pricing_env,
            paths,
            future_obs_indices,
            past_obs,
            future_obs,
            total_weights,
        )

        # Discount payoffs
        discount_factor = math.exp(-r * T)
        discounted_payoffs = discount_factor * payoffs

        # Compute price and standard error
        price = float(discounted_payoffs.mean())
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs))

        return RangeAccrualMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            in_range_ratio_mean=float(in_range_ratios.mean()),
            in_range_ratio_std=float(in_range_ratios.std(ddof=1)),
            num_past_observations=num_past,
            num_future_observations=num_future,
            past_in_range_weights=past_in_range_weights,
            total_weights=total_weights,
        )

    def _price_rqmc(
        self,
        product: RangeAccrualOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> RangeAccrualMCResult:
        """
        Price using Randomized QMC with adaptive batching.
        """
        # Build observation grid
        (
            all_times,
            dt_array,
            future_obs_indices,
            past_obs,
            future_obs,
            total_weights,
        ) = self._build_observation_grid(product, pricing_env, T)

        num_past = len(past_obs)
        num_future = len(future_obs)
        past_in_range_weights = sum(w for w, in_range in past_obs if in_range)

        # Handle special case: all observations are in the past
        if num_future == 0:
            in_range_ratio = past_in_range_weights / total_weights if total_weights > 0 else 0.0

            year_fraction = product.get_year_fraction(pricing_env)
            accrual_rate = product.range_config.accrual_rate
            payoff = (
                product.initial_price
                * product.contract_multiplier
                * accrual_rate
                * in_range_ratio
                * year_fraction
            )

            discount_factor = math.exp(-r * T)
            price = discount_factor * payoff

            return RangeAccrualMCResult(
                price=price,
                std_error=0.0,
                num_paths=0,
                in_range_ratio_mean=in_range_ratio,
                in_range_ratio_std=0.0,
                num_past_observations=num_past,
                num_future_observations=0,
                past_in_range_weights=past_in_range_weights,
                total_weights=total_weights,
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

        # Create path generator
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

        discount_factor = math.exp(-r * T)

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            payoffs, _ = self._compute_payoffs(
                product,
                pricing_env,
                paths,
                future_obs_indices,
                past_obs,
                future_obs,
                total_weights,
            )
            return discount_factor * payoffs

        result = run_rqmc(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            target_std=target_std,
            min_batches=min_batches,
        )

        # Run one more batch to get ratio statistics
        paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
        _, in_range_ratios = self._compute_payoffs(
            product,
            pricing_env,
            paths,
            future_obs_indices,
            past_obs,
            future_obs,
            total_weights,
        )

        return RangeAccrualMCResult(
            price=result.price,
            std_error=result.std_error,
            num_paths=result.total_paths,
            in_range_ratio_mean=float(in_range_ratios.mean()),
            in_range_ratio_std=float(in_range_ratios.std(ddof=1)),
            num_past_observations=num_past,
            num_future_observations=num_future,
            past_in_range_weights=past_in_range_weights,
            total_weights=total_weights,
            batches_used=result.batches_used,
        )

    def get_last_result(self) -> Optional[RangeAccrualMCResult]:
        """
        Get the full result from the last pricing run.

        Returns:
            RangeAccrualMCResult object, or None if no pricing has been performed
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
        return f"RangeAccrualMCEngine(method={self.method.name})"
