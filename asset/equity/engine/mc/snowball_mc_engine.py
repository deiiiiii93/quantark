"""
Monte Carlo pricing engine for Snowball (autocallable) options.

This engine prices snowball options using Monte Carlo simulation with support for:
- Standard and reverse snowball structures
- Discrete KO observations with time-varying barriers and rates
- Discrete or continuous KI monitoring
- INSTANT or EXPIRY coupon payment timing
- Vectorized NumPy operations for efficiency
- Optional Dask parallelization for batch processing
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import MCParams
from priceenv import PricingEnvironment
from util.enum import (
    ObservationType,
    CouponPayType,
    BarrierType,
)
from util.enum.engine_enums import MonteCarloMethod, EngineType
from util.exceptions import ValidationError, PricingError

from asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig


# Optional Dask import
try:
    from dask import delayed, compute
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class SnowballMCResult:
    """Result container for Snowball MC pricing."""

    price: float
    std_error: float
    num_paths: int
    ko_probability: float
    v0_probability: float
    v1_probability: float
    avg_ko_time: Optional[float] = None
    batches_used: Optional[int] = None


class SnowballMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for Snowball (autocallable) options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Usage:
        # Preferred: Two-level enum pattern
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000, time_steps=252),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000),
            method=MonteCarloMethod.QUASI
        )

        # With optional Dask parallelization
        engine = SnowballMCEngine(
            params=MCParams(num_paths=100000),
            use_dask=True,
            num_batches=8
        )

    The engine creates a GBMPathGenerator internally based on the pricing
    environment and product observation schedule.
    """

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO
    DEFAULT_KI_STEPS_PER_YEAR = 252  # Daily monitoring for continuous KI

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_dask: bool = False,
        num_batches: int = 4,
    ):
        """
        Initialize Snowball Monte Carlo engine.

        Args:
            params: Monte Carlo configuration parameters (MCParams)
            method: Monte Carlo method selection, one of:
                - EngineType.MONTE_CARLO(MonteCarloMethod.XXX) (preferred)
                - MonteCarloMethod.XXX
                - String: "pseudo", "quasi", "randomized_quasi"
                - None: defaults to MonteCarloMethod.PSEUDO
            use_dask: Enable Dask parallel processing (requires Dask installed)
            num_batches: Number of batches for parallel processing

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

        # Parse method
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

        # Dask configuration
        self.use_dask = use_dask and DASK_AVAILABLE
        if use_dask and not DASK_AVAILABLE:
            warnings.warn(
                "Dask requested but not installed. Falling back to single-threaded NumPy.",
                UserWarning,
            )
        self.num_batches = num_batches

        # Result storage
        self._last_result: Optional[SnowballMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a Snowball option using Monte Carlo simulation.

        Args:
            product: Snowball option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a SnowballOption
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, SnowballOption):
            raise PricingError(
                f"SnowballMCEngine only supports SnowballOption, "
                f"got {type(product).__name__}"
            )

        # Extract market data
        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(product.strike, T)

        self._validate_inputs(S, T, r, q, sigma, product)

        # Handle near-expiry case
        if T < 1e-10:
            return product.get_payoff(S, pricing_env)

        # Price using appropriate method
        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            result = self._price_rqmc(product, pricing_env, S, T, r, q, sigma)
        elif self.use_dask and self.num_batches > 1:
            result = self._price_parallel(product, pricing_env, S, T, r, q, sigma)
        else:
            result = self._price_mc_or_qmc(product, pricing_env, S, T, r, q, sigma)

        self._last_result = result

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
        product: SnowballOption,
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if q < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {q}")

        # Validate observation schedule exists
        if product.barrier_config.ko_observation_type == ObservationType.DISCRETE:
            if (
                product.barrier_config.ko_observation_schedule is None
                and product.barrier_config.ko_observation_dates is None
            ):
                raise ValidationError(
                    "KO observation schedule or dates required for discrete monitoring"
                )

    def _build_time_grid(
        self, product: SnowballOption, pricing_env: PricingEnvironment, T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build time grid aligned with observation dates.

        Returns:
            Tuple of:
            - all_times: Sorted unique observation times including maturity
            - dt_array: Time increments between observations
            - ko_indices: Indices into all_times for KO observations
            - ki_indices: Indices into all_times for KI observations (or fine grid)
        """
        # Get KO observation times
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_times = ko_profile["observation_times"]

        # Get KI observation times (if applicable)
        ki_times = []
        ki_continuous = (
            product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
            or product.barrier_config.ki_continuous
        )

        if product.has_ki_barrier:
            if ki_continuous:
                # Generate fine grid for continuous monitoring
                num_ki_steps = int(T * self.DEFAULT_KI_STEPS_PER_YEAR) + 1
                ki_times = list(np.linspace(0, T, num_ki_steps + 1)[1:])
            else:
                ki_profile = product.get_ki_observation_profile(pricing_env)
                ki_times = ki_profile["observation_times"]

        # Combine all times and ensure maturity is included
        all_times_set = set(ko_times) | set(ki_times) | {T}
        all_times = np.array(sorted(all_times_set))

        # Build dt_array
        times_with_zero = np.concatenate([[0.0], all_times])
        dt_array = np.diff(times_with_zero)

        # Find indices for KO and KI observations
        ko_indices = np.searchsorted(all_times, ko_times)
        if ki_continuous:
            # All times except t=0 are KI observation points
            ki_indices = np.arange(len(all_times))
        elif ki_times:
            ki_indices = np.searchsorted(all_times, ki_times)
        else:
            ki_indices = np.array([], dtype=int)

        return all_times, dt_array, ko_indices, ki_indices

    def _create_path_generator(
        self,
        S: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
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

        if self.method == MonteCarloMethod.PSEUDO:
            seed = params.seed + (batch_id or 0) * 1000
            random_stream = PseudoRandomNormalGenerator(seed=seed)
            is_qmc = False
        elif self.method in (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI):
            random_stream = SobolNormalGenerator(base_seed=params.seed)
            is_qmc = True
        else:
            raise ValidationError(f"Unknown Monte Carlo method: {self.method}")

        # No antithetic variates for barrier options (breaks correlation structure)
        vr_config = None

        generator = GBMPathGenerator(
            initial_value=S,
            vol=sigma,
            rrf=r,
            div=q,
            maturity=T,
            time_steps=len(dt_array),
            num_paths=params.num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=False,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

        return generator

    def _check_ko_barriers(
        self,
        paths: np.ndarray,
        ko_indices: np.ndarray,
        ko_barriers: np.ndarray,
        is_reverse: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized KO barrier checking.

        Args:
            paths: Simulated paths, shape (num_paths, num_times + 1)
            ko_indices: Indices into paths for KO observations
            ko_barriers: KO barrier levels, shape (num_ko_obs,)
            is_reverse: True for reverse snowball (DOWN barrier)

        Returns:
            Tuple of:
            - ko_triggered: Boolean array (num_paths,) indicating if KO was triggered
            - first_ko_idx: Index of first KO trigger (-1 if never triggered)
        """
        # Extract prices at KO observation times (offset by 1 for t=0)
        ko_prices = paths[:, ko_indices + 1]  # (num_paths, num_ko_obs)

        # Vectorized barrier check
        if is_reverse:
            # Reverse snowball: DOWN barrier (KO if price <= barrier)
            ko_hit = ko_prices <= ko_barriers
        else:
            # Standard snowball: UP barrier (KO if price >= barrier)
            ko_hit = ko_prices >= ko_barriers

        # Find first KO time per path
        ko_triggered = ko_hit.any(axis=1)
        first_ko_idx = np.full(len(paths), -1, dtype=int)

        if ko_triggered.any():
            # argmax returns first True in each row
            first_ko_idx[ko_triggered] = np.argmax(ko_hit[ko_triggered], axis=1)

        return ko_triggered, first_ko_idx

    def _check_ki_barriers(
        self,
        paths: np.ndarray,
        ki_indices: np.ndarray,
        ki_barriers: Union[float, np.ndarray],
        is_reverse: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized KI barrier checking.

        Args:
            paths: Simulated paths, shape (num_paths, num_times + 1)
            ki_indices: Indices into paths for KI observations
            ki_barrier: KI barrier level (single value for now)
            is_reverse: True for reverse snowball (UP barrier for KI)

        Returns:
            Tuple of:
            - ki_triggered: Boolean array (num_paths,) indicating if KI was triggered
            - first_ki_idx: Index of first KI trigger (-1 if never triggered)
        """
        if len(ki_indices) == 0:
            return np.zeros(len(paths), dtype=bool), np.full(len(paths), -1, dtype=int)

        # Extract prices at KI observation times (offset by 1 for t=0)
        ki_prices = paths[:, ki_indices + 1]  # (num_paths, num_ki_obs)

        # Extract prices at KI observation times (offset by 1 for t=0)
        ki_prices = paths[:, ki_indices + 1]  # (num_paths, num_ki_obs)
        num_ki_obs_times = ki_prices.shape[1]

        ki_barriers_effective = np.array(ki_barriers)

        # If ki_barriers has a single value (i.e., scalar or [scalar]), broadcast it
        if ki_barriers_effective.shape == () or ki_barriers_effective.shape == (1,):
            ki_barriers_aligned = np.full(num_ki_obs_times, ki_barriers_effective.item())
        else:
            # If it's an array with multiple values, it must match the number of observation times
            if ki_barriers_effective.shape[0] != num_ki_obs_times:
                raise ValidationError(
                    f"ki_barriers array (shape {ki_barriers_effective.shape[0]}) "
                    f"does not match number of KI observation times ({num_ki_obs_times})"
                )
            ki_barriers_aligned = ki_barriers_effective

        # Vectorized barrier check
        if is_reverse:
            # Reverse snowball: UP barrier for KI (KI if price >= barrier)
            ki_hit = ki_prices >= ki_barriers_aligned
        else:
            # Standard snowball: DOWN barrier for KI (KI if price <= barrier)
            ki_hit = ki_prices <= ki_barriers_aligned

        # Find first KI time per path
        ki_triggered = ki_hit.any(axis=1)
        first_ki_idx = np.full(len(paths), -1, dtype=int)

        if ki_triggered.any():
            first_ki_idx[ki_triggered] = np.argmax(ki_hit[ki_triggered], axis=1)

        return ki_triggered, first_ki_idx

    def _compute_payoffs(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        paths: np.ndarray,
        all_times: np.ndarray,
        ko_indices: np.ndarray,
        ki_indices: np.ndarray,
        r: float,
        T: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Compute payoffs for all paths based on their terminal state.

        Args:
            product: SnowballOption product
            pricing_env: Pricing environment
            paths: Simulated paths, shape (num_paths, num_times + 1)
            all_times: All observation times
            ko_indices: Indices for KO observations
            ki_indices: Indices for KI observations
            r: Risk-free rate
            T: Time to maturity

        Returns:
            Tuple of:
            - payoffs: Undiscounted payoffs, shape (num_paths,)
            - settlement_times: Settlement time for each path, shape (num_paths,)
            - stats: Dictionary with probability statistics
        """
        num_paths = len(paths)

        # Get resolved observation profiles
        ko_profile = product.get_ko_observation_profile(pricing_env)
        ko_barriers = np.array(ko_profile["barriers"])
        ko_payoffs_schedule = np.array(ko_profile["payoffs"])
        ko_times = np.array(ko_profile["observation_times"])
        ko_settlement_times = np.array(ko_profile["settlement_times"])

        # Get KI barriers (can be scalar or array)
        ki_barriers_val = None
        if product.has_ki_barrier:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_barriers_val = np.array(ki_profile["barriers"])

        # Check barriers
        ko_triggered, first_ko_idx = self._check_ko_barriers(
            paths, ko_indices, ko_barriers, product.is_reverse
        )

        ki_triggered = np.zeros(num_paths, dtype=bool)
        first_ki_idx = np.full(num_paths, -1, dtype=int)
        if ki_barriers_val is not None:
            ki_triggered, first_ki_idx = self._check_ki_barriers(
                paths, ki_indices, ki_barriers_val, product.is_reverse
            )

        # Handle disable_ko_after_ki logic
        if product.barrier_config.disable_ko_after_ki and ki_barriers_val is not None:
            # Get times for comparison
            ko_trigger_times = np.where(
                first_ko_idx >= 0,
                ko_times[first_ko_idx],
                np.inf,
            )
            ki_trigger_times = np.where(
                first_ki_idx >= 0,
                all_times[ki_indices[first_ki_idx]] if len(ki_indices) > 0 else np.inf,
                np.inf,
            )

            # KO is only valid if it happens before KI
            ko_before_ki = ko_trigger_times < ki_trigger_times
            ko_valid = ko_triggered & ko_before_ki
        else:
            ko_valid = ko_triggered

        # Classify paths into states
        is_ko = ko_valid
        is_v0 = ~is_ko & ~ki_triggered
        is_v1 = ~is_ko & ki_triggered

        # Initialize payoffs and settlement times
        payoffs = np.zeros(num_paths)
        settlement_times = np.full(num_paths, T)

        # KO payoffs
        if is_ko.any():
            ko_idx_for_payoff = first_ko_idx[is_ko]
            payoffs[is_ko] = ko_payoffs_schedule[ko_idx_for_payoff]

            # Settlement time depends on coupon pay type
            if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:
                settlement_times[is_ko] = ko_settlement_times[ko_idx_for_payoff]
            # else: EXPIRY - settlement at maturity (already set)

        # V0 payoffs (never KO, never KI)
        if is_v0.any():
            terminal_spots = paths[is_v0, -1]
            v0_payoffs = np.array([
                product.get_maturity_payoff_v0(spot, pricing_env)
                for spot in terminal_spots
            ])
            payoffs[is_v0] = v0_payoffs

        # V1 payoffs (never KO, KI happened)
        if is_v1.any():
            terminal_spots = paths[is_v1, -1]
            v1_payoffs = np.array([
                product.get_maturity_payoff_v1(spot, pricing_env)
                for spot in terminal_spots
            ])
            payoffs[is_v1] = v1_payoffs

        # Compute statistics
        stats = {
            "ko_probability": float(is_ko.mean()),
            "v0_probability": float(is_v0.mean()),
            "v1_probability": float(is_v1.mean()),
        }

        if is_ko.any():
            avg_ko_time = float(ko_times[first_ko_idx[is_ko]].mean())
            stats["avg_ko_time"] = avg_ko_time
        else:
            stats["avg_ko_time"] = None

        return payoffs, settlement_times, stats

    def _price_mc_or_qmc(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using normal MC or QMC (non-randomized).
        """
        # Build time grid
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        # Create path generator
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False)

        # Compute payoffs
        payoffs, settlement_times, stats = self._compute_payoffs(
            product, pricing_env, paths, all_times, ko_indices, ki_indices, r, T
        )

        # Discount payoffs
        discount_factors = np.exp(-r * settlement_times)
        discounted_payoffs = payoffs * discount_factors

        # Compute price and standard error
        price = float(discounted_payoffs.mean())
        std_payoff = float(discounted_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(payoffs))

        return SnowballMCResult(
            price=price,
            std_error=std_error,
            num_paths=len(paths),
            ko_probability=stats["ko_probability"],
            v0_probability=stats["v0_probability"],
            v1_probability=stats["v1_probability"],
            avg_ko_time=stats.get("avg_ko_time"),
        )

    def _price_single_batch(
        self,
        batch_id: int,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        all_times: np.ndarray,
        dt_array: np.ndarray,
        ko_indices: np.ndarray,
        ki_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Price a single batch of paths for parallel processing.
        """
        # Create path generator for this batch
        generator = self._create_path_generator(
            S, r, q, sigma, T, dt_array, batch_id=batch_id
        )

        # Generate paths
        paths, _ = generator.generate_paths(return_aux=False, batch_id=batch_id)

        # Compute payoffs
        payoffs, settlement_times, stats = self._compute_payoffs(
            product, pricing_env, paths, all_times, ko_indices, ki_indices, r, T
        )

        # Discount payoffs
        discount_factors = np.exp(-r * settlement_times)
        discounted_payoffs = payoffs * discount_factors

        return discounted_payoffs, stats

    def _price_parallel(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using Dask parallel batch processing.
        """
        # Build time grid (shared across batches)
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        # Create delayed tasks for each batch
        batch_results = [
            delayed(self._price_single_batch)(
                batch_id=i,
                product=product,
                pricing_env=pricing_env,
                S=S,
                T=T,
                r=r,
                q=q,
                sigma=sigma,
                all_times=all_times,
                dt_array=dt_array,
                ko_indices=ko_indices,
                ki_indices=ki_indices,
            )
            for i in range(self.num_batches)
        ]

        # Compute all batches in parallel
        results = compute(*batch_results)

        # Aggregate results
        all_payoffs = []
        all_stats = {"ko_probability": [], "v0_probability": [], "v1_probability": []}

        for discounted_payoffs, stats in results:
            all_payoffs.append(discounted_payoffs)
            all_stats["ko_probability"].append(stats["ko_probability"])
            all_stats["v0_probability"].append(stats["v0_probability"])
            all_stats["v1_probability"].append(stats["v1_probability"])

        combined_payoffs = np.concatenate(all_payoffs)
        total_paths = len(combined_payoffs)

        price = float(combined_payoffs.mean())
        std_payoff = float(combined_payoffs.std(ddof=1))
        std_error = std_payoff / math.sqrt(total_paths)

        return SnowballMCResult(
            price=price,
            std_error=std_error,
            num_paths=total_paths,
            ko_probability=float(np.mean(all_stats["ko_probability"])),
            v0_probability=float(np.mean(all_stats["v0_probability"])),
            v1_probability=float(np.mean(all_stats["v1_probability"])),
            batches_used=self.num_batches,
        )

    def _price_rqmc(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        S: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> SnowballMCResult:
        """
        Price using Randomized QMC with adaptive batching.
        """
        # Build time grid
        all_times, dt_array, ko_indices, ki_indices = self._build_time_grid(
            product, pricing_env, T
        )

        # Create path generator (will be used with different batch_ids)
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)

        def pricer_fn(paths, aux):
            """Pricer function for RQMC driver."""
            payoffs, settlement_times, _ = self._compute_payoffs(
                product, pricing_env, paths, all_times, ko_indices, ki_indices, r, T
            )
            discount_factors = np.exp(-r * settlement_times)
            return payoffs * discount_factors

        params = self.params
        max_batches = getattr(params, "max_batches", 32)
        target_std = getattr(params, "target_std", 1e-4)
        min_batches = getattr(params, "min_batches", 4)

        result = run_rqmc(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            target_std=target_std,
            min_batches=min_batches,
        )

        # Run one more batch to get statistics
        paths, _ = generator.generate_paths(return_aux=False, batch_id=0)
        _, _, stats = self._compute_payoffs(
            product, pricing_env, paths, all_times, ko_indices, ki_indices, r, T
        )

        return SnowballMCResult(
            price=result.price,
            std_error=result.std_error,
            num_paths=result.total_paths,
            ko_probability=stats["ko_probability"],
            v0_probability=stats["v0_probability"],
            v1_probability=stats["v1_probability"],
            batches_used=result.batches_used,
        )

    def get_last_result(self) -> Optional[SnowballMCResult]:
        """
        Get the full result from the last pricing run.

        Returns:
            SnowballMCResult object, or None if no pricing has been performed
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
        dask_str = f", use_dask={self.use_dask}" if self.use_dask else ""
        return f"SnowballMCEngine(method={self.method.name}{dask_str})"
