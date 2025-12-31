"""
Monte Carlo pricing engine for American vanilla options using LSM.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option import AmericanOption
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import MCParams
from priceenv import PricingEnvironment
from util.enum.engine_enums import MonteCarloMethod, EngineType
from util.exceptions import ValidationError, PricingError
from util.numerical import (
    Tolerance,
    is_zero,
    is_finite,
    safe_divide,
    safe_exp,
    safe_power,
    safe_sqrt,
    validate_positive,
    validate_non_negative,
)

from asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from asset.equity.process.bsm.qmc_rqmc_driver import RQMCResult, run_rqmc
from asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig


@dataclass
class AmericanMCResult:
    """Result container for American option MC pricing."""

    price: float
    std_error: float
    num_paths: int
    early_exercise_ratio: Optional[float] = None
    avg_exercise_time: Optional[float] = None
    batches_used: Optional[int] = None


class AmericanOptionMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for American vanilla options using LSM.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Usage:
        # Preferred: Two-level enum pattern
        engine = AmericanOptionMCEngine(
            params=MCParams(num_paths=100000, time_steps=252),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
        )

        # Alternative: Direct method enum
        engine = AmericanOptionMCEngine(
            params=MCParams(num_paths=100000),
            method=MonteCarloMethod.QUASI
        )

        # Backward compatibility: String
        engine = AmericanOptionMCEngine(method="quasi")
    """

    engine_type = EngineType.MONTE_CARLO

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        regression_degree: int = 2,
        min_regression_points: int = 25,
    ):
        """
        Initialize American option Monte Carlo engine.

        Args:
            params: Monte Carlo configuration parameters (MCParams)
            method: Monte Carlo method selection, one of:
                - EngineType.MONTE_CARLO(MonteCarloMethod.XXX) (preferred)
                - MonteCarloMethod.XXX
                - String: "pseudo", "quasi", "randomized_quasi"
                - None: defaults to MonteCarloMethod.PSEUDO
            regression_degree: Polynomial degree for LSM regression basis
            min_regression_points: Minimum in-the-money paths for regression

        Raises:
            ValidationError: If method or regression settings are invalid
        """
        if params is None:
            params = MCParams()

        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )

        if regression_degree < 1:
            raise ValidationError(
                f"regression_degree must be >= 1, got {regression_degree}"
            )
        if min_regression_points < 1:
            raise ValidationError(
                "min_regression_points must be positive, "
                f"got {min_regression_points}"
            )
        if min_regression_points < regression_degree + 1:
            raise ValidationError(
                "min_regression_points must be at least regression_degree + 1, "
                f"got {min_regression_points}"
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

        self.regression_degree = regression_degree
        self.min_regression_points = min_regression_points
        self._last_result: Optional[AmericanMCResult] = None
        self._last_rqmc_result: Optional[RQMCResult] = None

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an American vanilla option using Monte Carlo simulation.

        Args:
            product: American vanilla option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not an AmericanOption
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, AmericanOption):
            raise PricingError(
                f"AmericanOptionMCEngine only supports AmericanOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        self._validate_inputs(S, K, T, r, q, sigma)

        if is_zero(T):
            return product.get_payoff(S)

        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            result = self._price_rqmc(product, S, K, T, r, q, sigma)
        else:
            result = self._price_mc_or_qmc(product, S, K, T, r, q, sigma)

        self._last_result = result

        if result.price < 0.0:
            raise PricingError(f"Negative price computed: {result.price}")

        intrinsic = product.intrinsic_value(S)
        return max(result.price, intrinsic)

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> None:
        """Validate pricing inputs."""
        validate_positive(S, "spot")
        validate_positive(K, "strike")
        validate_non_negative(T, "time_to_maturity")
        validate_positive(sigma, "volatility")
        validate_non_negative(q, "dividend_yield")
        if not is_finite(r):
            raise ValidationError(f"risk_free_rate must be finite, got {r}")

    def _create_path_generator(
        self, S: float, r: float, q: float, sigma: float, T: float
    ) -> GBMPathGenerator:
        """Create a GBMPathGenerator configured for the current method."""
        params = self.params

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

        return GBMPathGenerator(
            initial_value=S,
            vol=sigma,
            rrf=r,
            div=q,
            maturity=T,
            time_steps=params.time_steps,
            num_paths=params.num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=False,
            vr_config=vr_config,
            is_qmc=is_qmc,
        )

    def _price_mc_or_qmc(
        self,
        product: AmericanOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> AmericanMCResult:
        """Price using normal MC or QMC (non-randomized)."""
        generator = self._create_path_generator(S, r, q, sigma, T)
        paths, _ = generator.generate_paths(return_aux=False)

        discount_factors = safe_exp(-r * generator.dt_vector)

        payoffs, exercise_steps = self._lsm_discounted_payoffs(
            product=product,
            paths=paths,
            discount_factors=discount_factors,
            strike=K,
            return_exercise_steps=True,
        )

        price = float(payoffs.mean())
        if payoffs.shape[0] > 1:
            std_payoff = float(payoffs.std(ddof=1))
            std_error = safe_divide(
                std_payoff, safe_sqrt(payoffs.shape[0]), fallback=0.0
            )
        else:
            std_error = 0.0

        exercise_times = self._exercise_times_in_years(
            exercise_steps=exercise_steps,
            times=generator.times,
            maturity=T,
        )
        early_exercise_ratio = float(np.mean(exercise_steps < generator.time_steps))
        avg_exercise_time = float(exercise_times.mean())

        return AmericanMCResult(
            price=price,
            std_error=std_error,
            num_paths=payoffs.shape[0],
            early_exercise_ratio=early_exercise_ratio,
            avg_exercise_time=avg_exercise_time,
        )

    def _price_rqmc(
        self,
        product: AmericanOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> AmericanMCResult:
        """Price using Randomized QMC with adaptive batching."""
        generator = self._create_path_generator(S, r, q, sigma, T)
        discount_factors = safe_exp(-r * generator.dt_vector)

        def pricer_fn(paths, aux):
            return self._lsm_discounted_payoffs(
                product=product,
                paths=paths,
                discount_factors=discount_factors,
                strike=K,
                return_exercise_steps=False,
            )

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

        self._last_rqmc_result = result

        return AmericanMCResult(
            price=result.price,
            std_error=result.std_error,
            num_paths=result.total_paths,
            early_exercise_ratio=None,
            avg_exercise_time=None,
            batches_used=result.batches_used,
        )

    def _lsm_discounted_payoffs(
        self,
        product: AmericanOption,
        paths: np.ndarray,
        discount_factors: np.ndarray,
        strike: float,
        return_exercise_steps: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply Longstaff-Schwartz regression to estimate optimal exercise.

        Returns discounted cashflows at time 0 and optionally exercise steps.
        """
        num_paths, num_steps_plus_one = paths.shape
        time_steps = num_steps_plus_one - 1

        payoffs = self._intrinsic_values(product, paths[:, -1], strike)
        exercise_steps = np.full(num_paths, time_steps, dtype=int)

        for step in range(time_steps - 1, 0, -1):
            payoffs = payoffs * discount_factors[step]

            spot = paths[:, step]
            exercise_values = self._intrinsic_values(product, spot, strike)
            in_the_money = exercise_values > Tolerance.ZERO

            if not np.any(in_the_money):
                continue

            if np.count_nonzero(in_the_money) >= self.min_regression_points:
                continuation = self._estimate_continuation(
                    spots=spot[in_the_money],
                    cashflows=payoffs[in_the_money],
                    strike=strike,
                )
            else:
                continuation = payoffs[in_the_money]

            exercise_now = exercise_values[in_the_money] > continuation
            if np.any(exercise_now):
                exercise_indices = np.where(in_the_money)[0][exercise_now]
                payoffs[exercise_indices] = exercise_values[exercise_indices]
                exercise_steps[exercise_indices] = step

        payoffs = payoffs * discount_factors[0]

        if return_exercise_steps:
            return payoffs, exercise_steps
        return payoffs

    def _estimate_continuation(
        self, spots: np.ndarray, cashflows: np.ndarray, strike: float
    ) -> np.ndarray:
        """Estimate continuation value via polynomial regression."""
        normalized = safe_divide(spots, strike, fallback=0.0)
        design = self._build_regression_matrix(normalized)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, cashflows, rcond=None)
            continuation = design @ coeffs
        except np.linalg.LinAlgError:
            return cashflows

        if not np.all(is_finite(continuation)):
            return cashflows

        return continuation

    def _build_regression_matrix(self, x: np.ndarray) -> np.ndarray:
        """Build polynomial regression basis matrix."""
        columns = [np.ones_like(x)]
        for power in range(1, self.regression_degree + 1):
            columns.append(safe_power(x, power))
        return np.column_stack(columns)

    @staticmethod
    def _intrinsic_values(
        product: AmericanOption, spot: np.ndarray, strike: float
    ) -> np.ndarray:
        """Compute intrinsic values for call/put options."""
        if product.is_call():
            return np.maximum(spot - strike, 0.0)
        return np.maximum(strike - spot, 0.0)

    @staticmethod
    def _exercise_times_in_years(
        exercise_steps: np.ndarray, times: np.ndarray, maturity: float
    ) -> np.ndarray:
        """Convert exercise step indices to exercise times."""
        max_step = times.shape[0]
        indices = np.clip(exercise_steps - 1, 0, max_step - 1)
        exercise_times = times[indices]
        exercise_times = np.where(exercise_steps == max_step, maturity, exercise_times)
        return exercise_times

    def get_last_result(self) -> Optional[AmericanMCResult]:
        """Get the result from the last pricing run."""
        return self._last_result

    def get_last_rqmc_result(self) -> Optional[RQMCResult]:
        """Get the RQMC result from the last RQMC pricing run."""
        return self._last_rqmc_result

    def __repr__(self) -> str:
        return f"AmericanOptionMCEngine(method={self.method.name})"
