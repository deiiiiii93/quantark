"""
Monte Carlo pricing engine for double sharkfin options.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_brownian_bridge import (
    compute_step_crossing_probabilities,
)
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_variance_reduction import VarianceReductionConfig
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import DoubleSharkfinOption
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationAggregation, ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    is_close,
    is_zero,
    safe_exp,
    validate_non_negative,
    validate_positive,
)


class DoubleSharkfinOptionMCEngine(BaseEngine):
    """
    Monte Carlo engine for DoubleSharkfinOption.

    Supports pseudorandom MC, Sobol QMC, and randomized QMC. Continuous
    monitoring checks the simulated path grid, with optional Brownian-bridge
    crossing probabilities between grid points.
    """

    engine_type = EngineType.MONTE_CARLO

    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_brownian_bridge: bool = False,
    ):
        """
        Initialize double sharkfin Monte Carlo engine.

        Args:
            params: Monte Carlo configuration.
            method: Monte Carlo method selection.
            use_brownian_bridge: Enable bridge crossing probabilities for
                continuous monitoring.

        Raises:
            ValidationError: If method or parameters are invalid.
        """
        if params is None:
            params = MCParams()
        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )

        super().__init__(params)
        self.method = self._resolve_method(method)
        if not isinstance(use_brownian_bridge, bool):
            raise ValidationError(
                f"use_brownian_bridge must be bool, got {type(use_brownian_bridge).__name__}"
            )
        self.use_brownian_bridge = use_brownian_bridge

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a double sharkfin option by Monte Carlo simulation.

        Args:
            product: DoubleSharkfinOption to price.
            pricing_env: Market data environment.

        Returns:
            Present value scaled by product.contract_multiplier.
        """
        if not isinstance(product, DoubleSharkfinOption):
            raise PricingError(
                "DoubleSharkfinOptionMCEngine only supports DoubleSharkfinOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(product.strike, maturity)
        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, product.strike)
        self._df = make_df_fn(pricing_env)


        self._validate_inputs(spot, maturity, rate, div, vol, product)

        if is_zero(maturity):
            self._last_std_error = 0.0
            return product.get_payoff(spot)

        if product.observation_type != ObservationType.EXPIRY and product.is_barrier_hit(spot):
            self._last_std_error = 0.0
            return self._price_already_hit(product, rate, maturity)

        if self.method == MonteCarloMethod.RANDOMIZED_QUASI:
            price, std_error = self._price_rqmc(
                product, pricing_env, spot, maturity, rate, div, vol
            )
        elif product.observation_type == ObservationType.EXPIRY:
            price, std_error = self._price_expiry_mc(
                product, spot, maturity, rate, div, vol
            )
        else:
            price, std_error = self._price_mc_or_qmc(
                product, pricing_env, spot, maturity, rate, div, vol
            )

        self._last_std_error = std_error
        if price < 0.0:
            raise PricingError(f"Negative price computed: {price}")
        return price

    def _resolve_method(
        self, method: Union[str, MonteCarloMethod, tuple, None]
    ) -> MonteCarloMethod:
        """Resolve constructor method argument into a MonteCarloMethod."""
        if method is None:
            return self.DEFAULT_METHOD
        if isinstance(method, tuple):
            engine_type, mc_method = method
            if engine_type != EngineType.MONTE_CARLO:
                raise ValidationError(
                    f"Expected EngineType.MONTE_CARLO, got {engine_type}"
                )
            if not isinstance(mc_method, MonteCarloMethod):
                raise ValidationError(
                    f"Expected MonteCarloMethod, got {type(mc_method).__name__}"
                )
            return mc_method
        if isinstance(method, MonteCarloMethod):
            return method
        if isinstance(method, str):
            try:
                return MonteCarloMethod[method.upper()]
            except KeyError:
                valid_methods = [m.name for m in MonteCarloMethod]
                raise ValidationError(
                    f"Invalid method string '{method}'. Valid methods: {valid_methods}"
                )
        raise ValidationError(
            f"Invalid method type {type(method).__name__}. "
            "Expected MonteCarloMethod, tuple, str, or None"
        )

    def _validate_inputs(
        self,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        product: DoubleSharkfinOption,
    ) -> None:
        """Validate pricing inputs."""
        validate_positive(spot, "spot")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_positive(product.strike, "strike")
        validate_positive(product.lower_barrier, "lower_barrier")
        validate_positive(product.upper_barrier, "upper_barrier")
        validate_non_negative(product.participation_rate, "participation_rate")
        validate_non_negative(product.knock_out_rebate, "knock_out_rebate")
        validate_non_negative(product.no_hit_rebate, "no_hit_rebate")
        validate_positive(product.contract_multiplier, "contract_multiplier")

        if product.lower_barrier >= product.upper_barrier:
            raise ValidationError("lower_barrier must be less than upper_barrier.")
        if not (product.lower_barrier < product.strike < product.upper_barrier):
            raise ValidationError(
                "Double sharkfin strike must be inside the barrier corridor."
            )
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def _price_already_hit(
        self, product: DoubleSharkfinOption, rate: float, maturity: float
    ) -> float:
        """Value a continuously monitored product whose spot has already hit."""
        if product.pay_at_hit:
            return product.get_barrier_payoff()
        return product.get_barrier_payoff() * self._df(maturity)

    def _create_path_generator(
        self,
        spot: float,
        rate: float,
        div: float,
        vol: float,
        maturity: float,
        dt_array: np.ndarray,
        batch_id: Optional[int] = None,
        num_paths: Optional[int] = None,
    ) -> GBMPathGenerator:
        """Create a GBM path generator for the selected MC method."""
        effective_num_paths = self.params.num_paths if num_paths is None else int(num_paths)
        if effective_num_paths <= 0:
            raise ValidationError(
                f"num_paths must be positive, got {effective_num_paths}"
            )

        if self.method == MonteCarloMethod.PSEUDO:
            seed = self.params.seed + (batch_id or 0) * 1000
            random_stream = PseudoRandomNormalGenerator(seed=seed)
            is_qmc = False
        elif self.method in (MonteCarloMethod.QUASI, MonteCarloMethod.RANDOMIZED_QUASI):
            random_stream = SobolNormalGenerator(base_seed=self.params.seed)
            is_qmc = True
        else:
            raise ValidationError(f"Unknown Monte Carlo method: {self.method}")

        vr_config = None
        if self.params.use_antithetic and not is_qmc:
            vr_config = VarianceReductionConfig(antithetic=True)

        term_ctx = getattr(self, "_term_ctx", None)
        if term_ctx is not None:
            env_ctx, ref_strike = term_ctx
            term = build_mc_term_inputs(
                env_ctx, ref_strike=ref_strike, maturity=maturity,
                time_steps=len(dt_array), dt_array=dt_array,
            )
            vol_in, rrf_in, div_in = term.vol, term.rrf, term.div
        else:
            vol_in, rrf_in, div_in = vol, rate, div

        return GBMPathGenerator(
            initial_value=spot,
            vol=vol_in,
            rrf=rrf_in,
            div=div_in,
            maturity=maturity,
            time_steps=len(dt_array),
            num_paths=effective_num_paths,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=self.use_brownian_bridge,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

    def _build_continuous_grid(self, maturity: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build a uniform monitoring grid for continuous path simulation."""
        time_steps = int(self.params.time_steps)
        if time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {time_steps}")
        dt_array = np.full(time_steps, maturity / float(time_steps), dtype=float)
        times = np.cumsum(dt_array)
        return times, dt_array

    def _build_discrete_grid(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        maturity: float,
    ):
        """Build a simulation grid aligned with discrete observation times."""
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise ValidationError("Discrete monitoring requires ObservationSchedule.")

        resolved = schedule.resolve(
            pricing_env,
            default_upper=product.upper_barrier,
            default_lower=product.lower_barrier,
            default_payoff=product.knock_out_rebate,
            require_double=True,
        )
        obs_times = np.array([rec.observation_time for rec in resolved], dtype=float)
        if np.any(obs_times < 0.0) or np.any(obs_times > maturity):
            raise ValidationError("Observation times must be within [0, maturity].")

        sorted_obs = np.sort(obs_times)
        if np.any(np.diff(sorted_obs) <= 0.0):
            raise ValidationError("Observation times must be strictly increasing.")

        if is_close(sorted_obs[-1], maturity):
            all_times = sorted_obs
        else:
            all_times = np.concatenate([sorted_obs, [maturity]])

        dt_array = np.diff(np.concatenate([[0.0], all_times]))
        obs_indices = np.searchsorted(all_times, obs_times)
        lower_barriers = np.array([rec.lower_barrier for rec in resolved], dtype=float)
        upper_barriers = np.array([rec.upper_barrier for rec in resolved], dtype=float)
        payoffs = np.array([rec.payoff for rec in resolved], dtype=float)
        settlement_times = np.array(
            [
                rec.settlement_time
                if rec.settlement_time is not None
                else rec.observation_time
                for rec in resolved
            ],
            dtype=float,
        )

        return (
            all_times,
            dt_array,
            obs_indices,
            lower_barriers,
            upper_barriers,
            payoffs,
            settlement_times,
            schedule.aggregation_mode,
        )

    def _no_hit_unit_payoff(
        self, product: DoubleSharkfinOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        """Vectorized no-hit payoff before contract multiplier."""
        if product.is_call():
            capped_spot = np.minimum(terminal_prices, product.upper_barrier)
            intrinsic = np.maximum(capped_spot - product.strike, 0.0)
        else:
            capped_spot = np.maximum(terminal_prices, product.lower_barrier)
            intrinsic = np.maximum(product.strike - capped_spot, 0.0)
        return product.no_hit_rebate + product.participation_rate * intrinsic

    def _expiry_payoffs(
        self, product: DoubleSharkfinOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        """Undiscounted unit payoffs for expiry monitoring."""
        hit = (terminal_prices >= product.upper_barrier) | (
            terminal_prices <= product.lower_barrier
        )
        no_hit = self._no_hit_unit_payoff(product, terminal_prices)
        return np.where(hit, product.knock_out_rebate, no_hit)

    def _aggregate_discrete_hit_payoffs(
        self,
        hit_matrix: np.ndarray,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        aggregation: ObservationAggregation,
        rate: float,
        pay_at_hit: bool,
        maturity: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate hit payoffs for discrete observations."""
        hit_any = hit_matrix.any(axis=1)
        if hit_matrix.size == 0:
            return np.zeros(hit_matrix.shape[0], dtype=float), hit_any

        if aggregation == ObservationAggregation.STOP_FIRST_HIT:
            first_idx = np.argmax(hit_matrix, axis=1)
            payoff = payoffs[first_idx]
            discount = (
                self._df(settlement_times[first_idx])
                if pay_at_hit
                else self._df(maturity)
            )
            discounted = payoff * discount
            discounted[~hit_any] = 0.0
            return discounted, hit_any

        if aggregation == ObservationAggregation.ACCUMULATE:
            if pay_at_hit:
                discount = self._df(settlement_times)
                discounted = np.sum(hit_matrix * (payoffs * discount), axis=1)
            else:
                total = np.sum(hit_matrix * payoffs, axis=1)
                discounted = total * self._df(maturity)
            return discounted, hit_any

        if aggregation in (ObservationAggregation.BEST, ObservationAggregation.WORST):
            if pay_at_hit:
                discount = self._df(settlement_times)
                value_matrix = hit_matrix * (payoffs * discount)
                if aggregation == ObservationAggregation.BEST:
                    discounted = value_matrix.max(axis=1)
                else:
                    value_matrix = np.where(hit_matrix, value_matrix, np.inf)
                    discounted = value_matrix.min(axis=1)
            else:
                value_matrix = hit_matrix * payoffs
                if aggregation == ObservationAggregation.BEST:
                    discounted = value_matrix.max(axis=1) * self._df(maturity)
                else:
                    value_matrix = np.where(hit_matrix, value_matrix, np.inf)
                    discounted = value_matrix.min(axis=1) * self._df(maturity)
            discounted[~hit_any] = 0.0
            return discounted, hit_any

        raise ValidationError(f"Unknown aggregation mode: {aggregation}")

    def _discounted_payoffs_continuous(
        self,
        product: DoubleSharkfinOption,
        paths: np.ndarray,
        times: np.ndarray,
        rate: float,
        maturity: float,
        vol: float,
    ) -> np.ndarray:
        """Discounted unit payoffs for continuous monitoring."""
        terminal_prices = paths[:, -1]
        no_hit_payoffs = self._no_hit_unit_payoff(product, terminal_prices)
        discount_maturity = self._df(maturity)
        hit_matrix = (paths[:, 1:] >= product.upper_barrier) | (
            paths[:, 1:] <= product.lower_barrier
        )

        if self.use_brownian_bridge:
            upper_prob = compute_step_crossing_probabilities(
                paths, product.upper_barrier, vol, times
            )
            lower_prob = compute_step_crossing_probabilities(
                paths, product.lower_barrier, vol, times
            )
            endpoint_hit = hit_matrix.astype(float)
            step_hit_prob = 1.0 - (1.0 - upper_prob) * (1.0 - lower_prob)
            step_hit_prob = np.maximum(step_hit_prob, endpoint_hit)
            step_hit_prob = np.clip(step_hit_prob, 0.0, 1.0)
            survival_prob = np.prod(1.0 - step_hit_prob, axis=1)

            if product.pay_at_hit:
                hit_leg = self._expected_rebate_at_hit(
                    step_hit_prob, times, rate, product.knock_out_rebate
                )
                return hit_leg + survival_prob * no_hit_payoffs * discount_maturity
            return (
                survival_prob * no_hit_payoffs
                + (1.0 - survival_prob) * product.knock_out_rebate
            ) * discount_maturity

        hit_any = hit_matrix.any(axis=1)
        if product.pay_at_hit:
            first_idx = np.argmax(hit_matrix, axis=1)
            hit_time = times[first_idx]
            hit_payoff = product.knock_out_rebate * self._df(hit_time)
            hit_payoff[~hit_any] = 0.0
            return np.where(hit_any, hit_payoff, no_hit_payoffs * discount_maturity)

        return np.where(hit_any, product.knock_out_rebate, no_hit_payoffs) * discount_maturity

    def _expected_rebate_at_hit(
        self,
        step_hit_prob: np.ndarray,
        times: np.ndarray,
        rate: float,
        rebate: float,
    ) -> np.ndarray:
        """Expected discounted rebate from first-hit step probabilities."""
        survival_before = np.cumprod(1.0 - step_hit_prob, axis=1)
        survival_before = np.concatenate(
            [np.ones((step_hit_prob.shape[0], 1)), survival_before[:, :-1]], axis=1
        )
        first_hit_prob = survival_before * step_hit_prob
        discount = self._df(times).reshape(1, -1)
        return rebate * np.sum(first_hit_prob * discount, axis=1)

    def _discounted_payoffs_discrete(
        self,
        product: DoubleSharkfinOption,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        lower_barriers: np.ndarray,
        upper_barriers: np.ndarray,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        aggregation: ObservationAggregation,
        rate: float,
        maturity: float,
    ) -> np.ndarray:
        """Discounted unit payoffs for discrete monitoring."""
        terminal_prices = paths[:, -1]
        no_hit_payoffs = self._no_hit_unit_payoff(product, terminal_prices)
        discount_maturity = self._df(maturity)

        obs_prices = paths[:, obs_indices + 1]
        hit_matrix = (obs_prices >= upper_barriers) | (obs_prices <= lower_barriers)
        hit_payoffs, hit_any = self._aggregate_discrete_hit_payoffs(
            hit_matrix,
            payoffs,
            settlement_times,
            aggregation,
            rate,
            product.pay_at_hit,
            maturity,
        )

        return np.where(hit_any, hit_payoffs, no_hit_payoffs * discount_maturity)

    def _price_mc_or_qmc(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        """Price with pseudorandom MC or non-randomized QMC."""
        if product.observation_type == ObservationType.DISCRETE:
            (
                times,
                dt_array,
                obs_indices,
                lower_barriers,
                upper_barriers,
                payoffs,
                settlement_times,
                aggregation,
            ) = self._build_discrete_grid(product, pricing_env, maturity)
            generator = self._create_path_generator(
                spot, rate, div, vol, maturity, dt_array
            )
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_discrete(
                product,
                paths,
                obs_indices,
                lower_barriers,
                upper_barriers,
                payoffs,
                settlement_times,
                aggregation,
                rate,
                maturity,
            )
        elif product.observation_type == ObservationType.CONTINUOUS:
            times, dt_array = self._build_continuous_grid(maturity)
            generator = self._create_path_generator(
                spot, rate, div, vol, maturity, dt_array
            )
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_continuous(
                product, paths, times, rate, maturity, vol
            )
        else:
            return self._price_expiry_mc(product, spot, maturity, rate, div, vol)

        return self._summarize_discounted(product, discounted)

    def _price_rqmc(
        self,
        product: DoubleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        """Price with randomized QMC adaptive batching."""
        if product.observation_type == ObservationType.EXPIRY:
            dt_array = np.array([maturity], dtype=float)

            def pricer_fn(paths, aux):
                terminal_prices = paths[:, -1]
                return self._expiry_payoffs(product, terminal_prices) * safe_exp(
                    -rate * maturity
                )

        elif product.observation_type == ObservationType.DISCRETE:
            (
                times,
                dt_array,
                obs_indices,
                lower_barriers,
                upper_barriers,
                payoffs,
                settlement_times,
                aggregation,
            ) = self._build_discrete_grid(product, pricing_env, maturity)

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_discrete(
                    product,
                    paths,
                    obs_indices,
                    lower_barriers,
                    upper_barriers,
                    payoffs,
                    settlement_times,
                    aggregation,
                    rate,
                    maturity,
                )

        else:
            times, dt_array = self._build_continuous_grid(maturity)

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_continuous(
                    product, paths, times, rate, maturity, vol
                )

        max_batches = getattr(
            self.params, "rqmc_max_batches", getattr(self.params, "max_batches", 32)
        )
        min_batches = getattr(
            self.params, "rqmc_min_batches", getattr(self.params, "min_batches", 4)
        )
        target_std = self.params.resolve_rqmc_target_std(
            product=product, pricing_env=pricing_env
        )
        per_batch_paths = self.params.resolve_rqmc_paths_per_batch(
            max_batches=max_batches
        )
        generator = self._create_path_generator(
            spot,
            rate,
            div,
            vol,
            maturity,
            dt_array,
            num_paths=per_batch_paths,
        )
        result = run_rqmc(
            pricer_fn=pricer_fn,
            path_generator=generator,
            max_batches=max_batches,
            target_std=target_std,
            min_batches=min_batches,
        )
        self._last_rqmc_result = result
        multiplier = product.contract_multiplier
        return result.price * multiplier, result.std_error * multiplier

    def _price_expiry_mc(
        self,
        product: DoubleSharkfinOption,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        """Price expiry monitoring with a single terminal simulation step."""
        dt_array = np.array([maturity], dtype=float)
        generator = self._create_path_generator(spot, rate, div, vol, maturity, dt_array)
        paths, _ = generator.generate_paths(return_aux=True)
        terminal_prices = paths[:, -1]
        discounted = self._expiry_payoffs(product, terminal_prices) * safe_exp(
            -rate * maturity
        )
        return self._summarize_discounted(product, discounted)

    def _summarize_discounted(
        self, product: DoubleSharkfinOption, discounted: np.ndarray
    ) -> Tuple[float, float]:
        """Return scaled mean price and standard error."""
        mean_payoff = float(discounted.mean())
        if len(discounted) <= 1:
            return mean_payoff * product.contract_multiplier, 0.0
        std_payoff = float(discounted.std(ddof=1))
        std_error = std_payoff / math.sqrt(len(discounted))
        multiplier = product.contract_multiplier
        return mean_payoff * multiplier, std_error * multiplier

    def get_last_std_error(self) -> Optional[float]:
        """Get the standard error from the last pricing run."""
        return getattr(self, "_last_std_error", None)

    def get_last_rqmc_result(self):
        """Get the full RQMC result from the last RQMC pricing run."""
        return getattr(self, "_last_rqmc_result", None)

    def __repr__(self):
        return "DoubleSharkfinOptionMCEngine()"
