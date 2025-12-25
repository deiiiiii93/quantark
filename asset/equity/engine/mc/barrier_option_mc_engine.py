"""
Monte Carlo pricing engine for barrier options.
"""

import math
from typing import Optional, Union, Tuple

import numpy as np

from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option import BarrierOption
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import MCParams
from priceenv import PricingEnvironment
from util.enum import ObservationType, ObservationAggregation
from util.enum.engine_enums import MonteCarloMethod, EngineType
from util.exceptions import ValidationError, PricingError
from util.numerical import (
    is_zero,
    is_close,
    validate_positive,
    validate_non_negative,
    safe_exp,
    safe_log,
)

from asset.equity.process.bsm.qmc_brownian_bridge import compute_step_crossing_probabilities
from asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig


class BarrierOptionMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for barrier options.

    Supports three Monte Carlo methods:
    - PSEUDO: Standard Monte Carlo with pseudorandom numbers
    - QUASI: Quasi-Monte Carlo with Sobol sequences
    - RANDOMIZED_QUASI: Randomized QMC with adaptive batching

    Optional Brownian-bridge handling can be enabled for continuous
    monitoring to approximate barrier crossings between time steps.
    """

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_brownian_bridge: bool = False,
    ):
        """
        Initialize Monte Carlo engine for barrier options.

        Args:
            params: Monte Carlo configuration parameters (MCParams)
            method: Monte Carlo method selection, one of:
                - EngineType.MONTE_CARLO(MonteCarloMethod.XXX) (preferred)
                - MonteCarloMethod.XXX
                - String: "pseudo", "quasi", "randomized_quasi"
                - None: defaults to MonteCarloMethod.PSEUDO
            use_brownian_bridge: Enable Brownian-bridge handling for continuous monitoring

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

        if not isinstance(use_brownian_bridge, bool):
            raise ValidationError(
                f"use_brownian_bridge must be bool, got {type(use_brownian_bridge).__name__}"
            )
        self.use_brownian_bridge = use_brownian_bridge

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a barrier option using Monte Carlo simulation.

        Args:
            product: Barrier option to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a barrier option
            ValidationError: If pricing parameters are invalid
        """
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierOptionMCEngine only supports BarrierOption, "
                f"got {type(product).__name__}"
            )

        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        self._validate_inputs(S, K, T, r, q, sigma, product)

        if is_zero(T):
            return self._price_expiry_payoff(product, S, r, T)

        if product.observation_type != ObservationType.EXPIRY and product.is_barrier_hit(S):
            if product.is_knock_out:
                return self._price_immediate_knock_out(product, r, T)
            return self._price_vanilla_mc(product, pricing_env)

        if product.observation_type == ObservationType.EXPIRY:
            if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
                price, std_error = self._price_rqmc(
                    product, pricing_env, S, K, T, r, q, sigma
                )
            else:
                price, std_error = self._price_expiry_mc(
                    product, S, K, T, r, q, sigma
                )
        elif self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            price, std_error = self._price_rqmc(
                product, pricing_env, S, K, T, r, q, sigma
            )
        else:
            price, std_error = self._price_mc_or_qmc(
                product, pricing_env, S, K, T, r, q, sigma
            )

        self._last_std_error = std_error

        if price < 0.0:
            raise PricingError(f"Negative price computed: {price}")

        return price

    def _validate_inputs(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        product: BarrierOption,
    ) -> None:
        """Validate pricing inputs."""
        validate_positive(S, "spot")
        validate_positive(K, "strike")
        validate_non_negative(T, "maturity")
        validate_positive(sigma, "volatility")
        validate_non_negative(q, "dividend_yield")
        validate_positive(product.barrier, "barrier")
        validate_non_negative(product.rebate, "rebate")
        validate_positive(product.participation_rate, "participation_rate")

    def _price_immediate_knock_out(
        self, product: BarrierOption, r: float, T: float
    ) -> float:
        if product.pay_at_hit:
            return product.rebate
        return product.rebate * safe_exp(-r * T)

    def _price_vanilla_mc(
        self, product: BarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        from asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
        from asset.equity.product.option import EuropeanVanillaOption

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

        engine = EuropeanMCEngine(params=self.params, method=self.method)
        price = engine.price(vanilla, pricing_env)
        self._last_std_error = engine.get_last_std_error()
        self._last_rqmc_result = engine.get_last_rqmc_result()
        return price * product.participation_rate

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
            use_brownian_bridge=self.use_brownian_bridge,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

        return generator

    def _build_continuous_grid(self, T: float) -> Tuple[np.ndarray, np.ndarray]:
        time_steps = int(self.params.time_steps)
        if time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {time_steps}")
        dt_array = np.full(time_steps, T / float(time_steps), dtype=float)
        times = np.cumsum(dt_array)
        return times, dt_array

    def _build_discrete_grid(
        self, product: BarrierOption, pricing_env: PricingEnvironment, T: float
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        ObservationAggregation,
    ]:
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise ValidationError("Discrete monitoring requires ObservationSchedule.")

        resolved = schedule.resolve(
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.rebate,
            require_single=True,
        )

        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        if np.any(obs_times < 0.0) or np.any(obs_times > T):
            raise ValidationError("Observation times must be within [0, maturity].")
        sorted_obs = np.sort(obs_times)
        if np.any(np.diff(sorted_obs) <= 0.0):
            raise ValidationError("Observation times must be strictly increasing.")

        if is_close(sorted_obs[-1], T):
            all_times = sorted_obs
        else:
            all_times = np.concatenate([sorted_obs, [T]])

        dt_array = np.diff(np.concatenate([[0.0], all_times]))
        obs_indices = np.searchsorted(all_times, obs_times)
        barriers = np.array([rec.barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time if rec.settlement_time is not None else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        return (
            all_times,
            dt_array,
            obs_indices,
            barriers,
            payoffs,
            settlement_times,
            schedule.aggregation_mode,
        )

    def _calculate_vanilla_payoff(
        self, product: BarrierOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        K = product.strike
        if product.is_call():
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)
        return payoffs * product.participation_rate

    def _price_expiry_payoff(
        self, product: BarrierOption, spot: float, r: float, T: float
    ) -> float:
        hit = product.is_barrier_hit(spot)
        payoff = product.get_payoff(spot) * product.participation_rate
        if product.is_knock_out:
            value = product.rebate if hit else payoff
        else:
            value = payoff if hit else product.rebate
        return value * safe_exp(-r * T)

    def _expiry_payoffs(
        self, product: BarrierOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        hit = terminal_prices >= product.barrier if product.is_up_barrier else terminal_prices <= product.barrier
        vanilla = self._calculate_vanilla_payoff(product, terminal_prices)
        if product.is_knock_out:
            return np.where(hit, product.rebate, vanilla)
        return np.where(hit, vanilla, product.rebate)

    def _compute_bridge_step_hit_probabilities(
        self,
        paths: np.ndarray,
        barrier: float,
        sigma: float,
        times: np.ndarray,
    ) -> np.ndarray:
        return compute_step_crossing_probabilities(paths, barrier, sigma, times)

    def _expected_rebate_at_hit(
        self,
        step_hit_prob: np.ndarray,
        times: np.ndarray,
        r: float,
        rebate: float,
    ) -> np.ndarray:
        survival_before = np.cumprod(1.0 - step_hit_prob, axis=1)
        survival_before = np.concatenate(
            [np.ones((step_hit_prob.shape[0], 1)), survival_before[:, :-1]], axis=1
        )
        first_hit_prob = survival_before * step_hit_prob
        discount = safe_exp(-r * times).reshape(1, -1)
        return rebate * np.sum(first_hit_prob * discount, axis=1)

    def _aggregate_discrete_hit_payoffs(
        self,
        hit_matrix: np.ndarray,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        aggregation: ObservationAggregation,
        r: float,
        pay_at_hit: bool,
        T: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hit_any = hit_matrix.any(axis=1)
        if hit_matrix.size == 0:
            return np.zeros(hit_matrix.shape[0], dtype=float), hit_any

        if aggregation == ObservationAggregation.STOP_FIRST_HIT:
            first_idx = np.argmax(hit_matrix, axis=1)
            payoff = payoffs[first_idx]
            if pay_at_hit:
                discount = safe_exp(-r * settlement_times[first_idx])
            else:
                discount = safe_exp(-r * T)
            discounted = payoff * discount
            discounted[~hit_any] = 0.0
            return discounted, hit_any

        if aggregation == ObservationAggregation.ACCUMULATE:
            if pay_at_hit:
                discount = safe_exp(-r * settlement_times)
                discounted = np.sum(hit_matrix * (payoffs * discount), axis=1)
            else:
                total = np.sum(hit_matrix * payoffs, axis=1)
                discounted = total * safe_exp(-r * T)
            return discounted, hit_any

        if aggregation in (ObservationAggregation.BEST, ObservationAggregation.WORST):
            if pay_at_hit:
                discount = safe_exp(-r * settlement_times)
                value_matrix = hit_matrix * (payoffs * discount)
                if aggregation == ObservationAggregation.BEST:
                    discounted = value_matrix.max(axis=1)
                else:
                    value_matrix = np.where(hit_matrix, value_matrix, np.inf)
                    discounted = value_matrix.min(axis=1)
            else:
                value_matrix = hit_matrix * payoffs
                if aggregation == ObservationAggregation.BEST:
                    best = value_matrix.max(axis=1)
                    discounted = best * safe_exp(-r * T)
                else:
                    value_matrix = np.where(hit_matrix, value_matrix, np.inf)
                    worst = value_matrix.min(axis=1)
                    discounted = worst * safe_exp(-r * T)
            discounted[~hit_any] = 0.0
            return discounted, hit_any

        raise ValidationError(f"Unknown aggregation mode: {aggregation}")

    def _discounted_payoffs_continuous(
        self,
        product: BarrierOption,
        paths: np.ndarray,
        times: np.ndarray,
        r: float,
        T: float,
        sigma: float,
    ) -> np.ndarray:
        terminal_prices = paths[:, -1]
        vanilla_payoffs = self._calculate_vanilla_payoff(product, terminal_prices)
        df_T = safe_exp(-r * T)

        if self.use_brownian_bridge:
            step_hit_prob = self._compute_bridge_step_hit_probabilities(
                paths, product.barrier, sigma, times
            )
            survival_prob = np.prod(1.0 - step_hit_prob, axis=1)

            if product.is_knock_out:
                if product.pay_at_hit:
                    rebate_leg = self._expected_rebate_at_hit(
                        step_hit_prob, times, r, product.rebate
                    )
                    return rebate_leg + survival_prob * vanilla_payoffs * df_T
                return (
                    survival_prob * vanilla_payoffs
                    + (1.0 - survival_prob) * product.rebate
                ) * df_T

            return (
                (1.0 - survival_prob) * vanilla_payoffs
                + survival_prob * product.rebate
            ) * df_T

        hit_matrix = (
            paths[:, 1:] >= product.barrier
            if product.is_up_barrier
            else paths[:, 1:] <= product.barrier
        )
        hit_any = hit_matrix.any(axis=1)

        if product.is_knock_out:
            if product.pay_at_hit:
                first_idx = np.argmax(hit_matrix, axis=1)
                hit_time = times[first_idx]
                rebate_payoff = product.rebate * safe_exp(-r * hit_time)
                rebate_payoff[~hit_any] = 0.0
                return np.where(hit_any, rebate_payoff, vanilla_payoffs * df_T)
            return np.where(hit_any, product.rebate, vanilla_payoffs) * df_T

        return np.where(hit_any, vanilla_payoffs, product.rebate) * df_T

    def _discounted_payoffs_discrete(
        self,
        product: BarrierOption,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        barriers: np.ndarray,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        aggregation: ObservationAggregation,
        r: float,
        T: float,
    ) -> np.ndarray:
        terminal_prices = paths[:, -1]
        vanilla_payoffs = self._calculate_vanilla_payoff(product, terminal_prices)
        df_T = safe_exp(-r * T)

        obs_prices = paths[:, obs_indices + 1]
        hit_matrix = obs_prices >= barriers if product.is_up_barrier else obs_prices <= barriers

        hit_payoffs, hit_any = self._aggregate_discrete_hit_payoffs(
            hit_matrix,
            payoffs,
            settlement_times,
            aggregation,
            r,
            product.pay_at_hit,
            T,
        )

        if product.is_knock_out:
            return np.where(hit_any, hit_payoffs, vanilla_payoffs * df_T)

        return np.where(hit_any, vanilla_payoffs * df_T, product.rebate * df_T)

    def _price_mc_or_qmc(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> Tuple[float, float]:
        if product.observation_type == ObservationType.DISCRETE:
            (
                times,
                dt_array,
                obs_indices,
                barriers,
                payoffs,
                settlement_times,
                aggregation,
            ) = self._build_discrete_grid(product, pricing_env, T)
            generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_discrete(
                product,
                paths,
                obs_indices,
                barriers,
                payoffs,
                settlement_times,
                aggregation,
                r,
                T,
            )
        else:
            times, dt_array = self._build_continuous_grid(T)
            generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_continuous(
                product, paths, times, r, T, sigma
            )

        mean_payoff = float(discounted.mean())
        std_payoff = float(discounted.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(discounted))

        return mean_payoff, std_error

    def _price_rqmc(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> Tuple[float, float]:
        if product.observation_type == ObservationType.EXPIRY:
            dt_array = np.array([T], dtype=float)

            def pricer_fn(paths, aux):
                terminal_prices = paths[:, -1]
                payoffs = self._expiry_payoffs(product, terminal_prices)
                return payoffs * safe_exp(-r * T)

        elif product.observation_type == ObservationType.DISCRETE:
            (
                times,
                dt_array,
                obs_indices,
                barriers,
                payoffs,
                settlement_times,
                aggregation,
            ) = self._build_discrete_grid(product, pricing_env, T)

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_discrete(
                    product,
                    paths,
                    obs_indices,
                    barriers,
                    payoffs,
                    settlement_times,
                    aggregation,
                    r,
                    T,
                )
        else:
            times, dt_array = self._build_continuous_grid(T)

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_continuous(
                    product, paths, times, r, T, sigma
                )

        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)

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

        return result.price, result.std_error

    def _price_expiry_mc(
        self,
        product: BarrierOption,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
    ) -> Tuple[float, float]:
        dt_array = np.array([T], dtype=float)
        generator = self._create_path_generator(S, r, q, sigma, T, dt_array)
        paths, _ = generator.generate_paths(return_aux=True)
        terminal_prices = paths[:, -1]

        payoffs = self._expiry_payoffs(product, terminal_prices)
        discounted = payoffs * safe_exp(-r * T)

        mean_payoff = float(discounted.mean())
        std_payoff = float(discounted.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(discounted))

        return mean_payoff, std_error

    def get_last_std_error(self) -> Optional[float]:
        """
        Get the standard error from the last pricing run.

        Returns:
            Standard error, or None if no pricing has been performed yet
        """
        return getattr(self, "_last_std_error", None)

    def get_last_rqmc_result(self):
        """
        Get the full RQMC result from the last RQMC pricing run.

        Returns:
            RQMCResult object, or None if last pricing was not RQMC
        """
        return getattr(self, "_last_rqmc_result", None)

    def __repr__(self) -> str:
        return (
            f"BarrierOptionMCEngine(method={self.method.name}, "
            f"brownian_bridge={self.use_brownian_bridge})"
        )
