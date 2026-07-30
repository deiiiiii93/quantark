"""
Monte Carlo pricing engine for single sharkfin options.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
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
from quantark.asset.equity.product.option import SingleSharkfinOption
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.execution.errors import CapabilityError
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationAggregation, ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    is_close,
    is_zero,
    validate_non_negative,
    validate_positive,
)


class SingleSharkfinOptionMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for SingleSharkfinOption.

    Supports pseudo-random, quasi-Monte Carlo, and randomized QMC methods.
    Continuous monitoring uses path-grid monitoring by default and can enable
    Brownian-bridge crossing probabilities with ``use_brownian_bridge=True``.
    """

    engine_type = EngineType.MONTE_CARLO
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    # Request-scoped term-structure context (set at every public pricing
    # entry point from ITS pricing_env; never carried across calls).
    # Private helpers called outside a public entry fall back to their
    # scalar arguments (None ctx) or fail loudly (None _df) rather than
    # silently reusing a previous environment. Engine instances are not
    # safe for concurrent pricing calls (pre-existing engine contract).
    _term_ctx = None
    _df = None
    _terminal_df = None
    _pricing_env = None
    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
        use_brownian_bridge: bool = False,
    ):
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
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        """
        Price a single sharkfin option using Monte Carlo simulation.
        """
        if not isinstance(product, SingleSharkfinOption):
            raise PricingError(
                "SingleSharkfinOptionMCEngine only supports SingleSharkfinOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            self._last_std_error = 0.0
            return fixed_pv
        terminal_timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        spot = pricing_env.spot
        strike = product.strike
        maturity = product.get_maturity(pricing_env)
        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(strike, maturity)
        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, strike)
        self._df = make_df_fn(pricing_env)
        self._terminal_df = terminal_timing.payment_df
        self._pricing_env = pricing_env

        self._validate_inputs(spot, strike, maturity, rate, div, vol, product)

        if is_zero(maturity):
            self._last_std_error = 0.0
            return (
                product.get_payoff(spot) * self._terminal_df
                + pending_pv
            )

        if (
            product.observation_type != ObservationType.EXPIRY
            and product.is_barrier_hit(spot)
        ):
            self._last_std_error = 0.0
            return (
                self._price_immediate_knock_out(product, pricing_env)
                + pending_pv
            )

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

        multiplier = product.contract_multiplier
        price *= multiplier
        std_error *= multiplier
        self._last_std_error = std_error

        if price < 0.0:
            raise PricingError(f"Negative price computed: {price}")

        return price + pending_pv

    def _resolve_method(
        self, method: Union[str, MonteCarloMethod, tuple, None]
    ) -> MonteCarloMethod:
        """Resolve Monte Carlo method from enum, two-level enum, string, or None."""
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
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        product: SingleSharkfinOption,
    ) -> None:
        """Validate pricing inputs."""
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_positive(product.barrier, "barrier")
        validate_non_negative(product.participation_rate, "participation_rate")
        validate_non_negative(product.knock_out_rebate, "knock_out_rebate")
        validate_non_negative(product.no_hit_rebate, "no_hit_rebate")
        validate_positive(product.contract_multiplier, "contract_multiplier")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def _price_immediate_knock_out(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
    ) -> float:
        if product.pay_at_hit:
            payment_df = SettlementResolver.resolve_contingent(
                product,
                SettlementRequest(
                    kind=CashflowKind.HIT,
                    determination_date=pricing_env.valuation_date,
                    determination_time=0.0,
                    cashflow_id="single-sharkfin:already-hit",
                ),
                pricing_env,
            ).payment_df
        else:
            payment_df = self._terminal_df
        return (
            product.knock_out_rebate
            * payment_df
            * product.contract_multiplier
        )

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
        num_paths_eff = self.params.num_paths if num_paths is None else int(num_paths)
        if num_paths_eff <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths_eff}")

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
            num_paths=num_paths_eff,
            model="bsm",
            random_stream=random_stream,
            use_brownian_bridge=self.use_brownian_bridge,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

    def _build_continuous_grid(self, maturity: float) -> Tuple[np.ndarray, np.ndarray]:
        time_steps = int(self.params.time_steps)
        if time_steps <= 0:
            raise ValidationError(f"time_steps must be positive, got {time_steps}")
        dt_array = np.full(time_steps, maturity / float(time_steps), dtype=float)
        return np.cumsum(dt_array), dt_array

    def _build_discrete_grid(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
        maturity: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        schedule = product.observation_schedule
        if schedule is None or not schedule.records:
            raise ValidationError("Discrete monitoring requires ObservationSchedule.")
        if schedule.aggregation_mode != ObservationAggregation.STOP_FIRST_HIT:
            raise ValidationError(
                "Single sharkfin MC supports STOP_FIRST_HIT aggregation only."
            )

        resolved = schedule.resolve(
            pricing_env,
            default_barrier=product.barrier,
            default_payoff=product.knock_out_rebate,
            require_single=True,
            product=product,
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
        barriers = np.array([rec.barrier for rec in resolved], dtype=float)
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
        return dt_array, obs_indices, barriers, payoffs, settlement_times

    def _no_hit_payoff(
        self, product: SingleSharkfinOption, terminal_prices: np.ndarray
    ) -> np.ndarray:
        if product.is_call():
            capped_spot = np.minimum(terminal_prices, product.barrier)
            intrinsic = np.maximum(capped_spot - product.strike, 0.0)
        else:
            capped_spot = np.maximum(terminal_prices, product.barrier)
            intrinsic = np.maximum(product.strike - capped_spot, 0.0)
        return product.no_hit_rebate + product.participation_rate * intrinsic

    def _price_expiry_payoff(
        self,
        product: SingleSharkfinOption,
        terminal_prices: np.ndarray,
    ) -> np.ndarray:
        hit = (
            terminal_prices >= product.barrier
            if product.is_call()
            else terminal_prices <= product.barrier
        )
        return np.where(
            hit,
            product.knock_out_rebate,
            self._no_hit_payoff(product, terminal_prices),
        )

    def _price_expiry_mc(
        self,
        product: SingleSharkfinOption,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        dt_array = np.array([maturity], dtype=float)
        generator = self._create_path_generator(spot, rate, div, vol, maturity, dt_array)
        paths, _ = generator.generate_paths(return_aux=True)
        discounted = (
            self._price_expiry_payoff(product, paths[:, -1])
            * self._terminal_df
        )
        return self._mean_and_std_error(discounted)

    def _discounted_payoffs_discrete(
        self,
        product: SingleSharkfinOption,
        paths: np.ndarray,
        obs_indices: np.ndarray,
        barriers: np.ndarray,
        payoffs: np.ndarray,
        settlement_times: np.ndarray,
        rate: float,
        maturity: float,
    ) -> np.ndarray:
        terminal_prices = paths[:, -1]
        obs_prices = paths[:, obs_indices + 1]
        hit_matrix = (
            obs_prices >= barriers if product.is_call() else obs_prices <= barriers
        )
        hit_any = hit_matrix.any(axis=1)
        first_idx = np.argmax(hit_matrix, axis=1)

        hit_payoff = payoffs[first_idx]
        if product.pay_at_hit:
            hit_discount = self._df(settlement_times[first_idx])
        else:
            hit_discount = self._terminal_df
        discounted_hit = hit_payoff * hit_discount
        discounted_hit[~hit_any] = 0.0

        discounted_no_hit = (
            self._no_hit_payoff(product, terminal_prices) * self._terminal_df
        )
        return np.where(hit_any, discounted_hit, discounted_no_hit)

    def _discounted_payoffs_continuous(
        self,
        product: SingleSharkfinOption,
        paths: np.ndarray,
        times: np.ndarray,
        rate: float,
        maturity: float,
        vol: float,
    ) -> np.ndarray:
        terminal_prices = paths[:, -1]
        no_hit_payoff = self._no_hit_payoff(product, terminal_prices)
        df_t = self._terminal_df

        if self.use_brownian_bridge:
            step_hit_prob = compute_step_crossing_probabilities(
                paths, product.barrier, vol, times
            )
            survival_prob = np.prod(1.0 - step_hit_prob, axis=1)
            no_hit_leg = survival_prob * no_hit_payoff * df_t

            if product.pay_at_hit:
                survival_before = np.cumprod(1.0 - step_hit_prob, axis=1)
                survival_before = np.concatenate(
                    [
                        np.ones((step_hit_prob.shape[0], 1)),
                        survival_before[:, :-1],
                    ],
                    axis=1,
                )
                first_hit_prob = survival_before * step_hit_prob
                discount = self._event_payment_dfs(
                    product, times
                ).reshape(1, -1)
                hit_leg = product.knock_out_rebate * np.sum(
                    first_hit_prob * discount, axis=1
                )
            else:
                hit_leg = (1.0 - survival_prob) * product.knock_out_rebate * df_t

            return hit_leg + no_hit_leg

        hit_matrix = (
            paths[:, 1:] >= product.barrier
            if product.is_call()
            else paths[:, 1:] <= product.barrier
        )
        hit_any = hit_matrix.any(axis=1)

        if product.pay_at_hit:
            first_idx = np.argmax(hit_matrix, axis=1)
            hit_time = times[first_idx]
            hit_payoff = (
                product.knock_out_rebate
                * self._event_payment_dfs(product, hit_time)
            )
            hit_payoff[~hit_any] = 0.0
        else:
            hit_payoff = product.knock_out_rebate * df_t

        no_hit_value = no_hit_payoff * df_t
        return np.where(hit_any, hit_payoff, no_hit_value)

    def _event_payment_dfs(
        self,
        product: SingleSharkfinOption,
        determination_times: np.ndarray,
    ) -> np.ndarray:
        """Resolve event payment discount factors on the simulated hit grid."""
        try:
            return np.asarray(
                [
                    SettlementResolver.resolve_contingent(
                        product,
                        SettlementRequest(
                            kind=CashflowKind.HIT,
                            determination_time=float(time),
                            cashflow_id=f"simulated-hit[{index}]",
                        ),
                        self._pricing_env,
                    ).payment_df
                    for index, time in enumerate(determination_times)
                ],
                dtype=float,
            )
        except ValidationError as exc:
            raise CapabilityError(
                f"{type(self).__name__} cannot resolve delayed first-hit "
                "settlement from a numeric simulation grid"
            ) from exc

    def _price_mc_or_qmc(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        if product.observation_type == ObservationType.DISCRETE:
            dt_array, obs_indices, barriers, payoffs, settlement_times = (
                self._build_discrete_grid(product, pricing_env, maturity)
            )
            generator = self._create_path_generator(
                spot, rate, div, vol, maturity, dt_array
            )
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_discrete(
                product,
                paths,
                obs_indices,
                barriers,
                payoffs,
                settlement_times,
                rate,
                maturity,
            )
        else:
            times, dt_array = self._build_continuous_grid(maturity)
            generator = self._create_path_generator(
                spot, rate, div, vol, maturity, dt_array
            )
            paths, _ = generator.generate_paths(return_aux=True)
            discounted = self._discounted_payoffs_continuous(
                product, paths, times, rate, maturity, vol
            )
        return self._mean_and_std_error(discounted)

    def _price_rqmc(
        self,
        product: SingleSharkfinOption,
        pricing_env: PricingEnvironment,
        spot: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
    ) -> Tuple[float, float]:
        if product.observation_type == ObservationType.EXPIRY:
            dt_array = np.array([maturity], dtype=float)

            def pricer_fn(paths, aux):
                return (
                    self._price_expiry_payoff(product, paths[:, -1])
                    * self._terminal_df
                )

        elif product.observation_type == ObservationType.DISCRETE:
            dt_array, obs_indices, barriers, payoffs, settlement_times = (
                self._build_discrete_grid(product, pricing_env, maturity)
            )

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_discrete(
                    product,
                    paths,
                    obs_indices,
                    barriers,
                    payoffs,
                    settlement_times,
                    rate,
                    maturity,
                )

        else:
            times, dt_array = self._build_continuous_grid(maturity)

            def pricer_fn(paths, aux):
                return self._discounted_payoffs_continuous(
                    product, paths, times, rate, maturity, vol
                )

        max_batches = self.params.rqmc_max_batches
        min_batches = self.params.rqmc_min_batches
        target_std = self.params.resolve_rqmc_target_std(
            product=product, pricing_env=pricing_env
        )
        paths_per_batch = self.params.resolve_rqmc_paths_per_batch(
            max_batches=max_batches
        )
        generator = self._create_path_generator(
            spot, rate, div, vol, maturity, dt_array, num_paths=paths_per_batch
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

    def _mean_and_std_error(self, discounted: np.ndarray) -> Tuple[float, float]:
        mean_payoff = float(discounted.mean())
        if len(discounted) <= 1:
            return mean_payoff, 0.0
        std_payoff = float(discounted.std(ddof=1))
        return mean_payoff, std_payoff / math.sqrt(len(discounted))

    def get_last_std_error(self) -> Optional[float]:
        """Return the standard error from the most recent pricing run."""
        return getattr(self, "_last_std_error", None)

    def get_last_rqmc_result(self):
        """Return the full RQMC result from the most recent RQMC run."""
        return getattr(self, "_last_rqmc_result", None)

    def __repr__(self) -> str:
        return (
            f"SingleSharkfinOptionMCEngine(method={self.method.name}, "
            f"brownian_bridge={self.use_brownian_bridge})"
        )
