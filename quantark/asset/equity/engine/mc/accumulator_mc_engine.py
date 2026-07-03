"""
Monte Carlo pricing engine for accumulator options.

This engine provides the exact discrete-monitoring benchmark for the accumulator,
complementing the analytical decomposition (which relies on the
Broadie-Glasserman-Kou barrier-shift approximation for discrete monitoring).

For each simulated path the engine walks the observation dates in order:

* TERMINATION: the first observation with ``spot >= KO`` knocks out the whole
  contract; observations strictly before it accrue, the rebate is paid at the hit
  time, and later observations are dropped.
* SINGLE_DAY: an observation with ``spot >= KO`` cancels only that day's accrual;
  the contract continues.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_variance_reduction import (
    VarianceReductionConfig,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import AccumulatorOption
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs, make_df_fn
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AccumulatorKnockOutType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    is_close,
    is_zero,
    validate_non_negative,
    validate_positive,
)


class AccumulatorMCEngine(BaseEngine):
    """
    Monte Carlo pricing engine for :class:`AccumulatorOption`.

    Supports pseudo-random and quasi-Monte Carlo path generation. Monitoring is
    discrete on the product's observation schedule (the natural accumulator
    convention).
    """

    engine_type = EngineType.MONTE_CARLO
    DEFAULT_METHOD = MonteCarloMethod.PSEUDO

    def __init__(
        self,
        params: Optional[MCParams] = None,
        method: Union[str, MonteCarloMethod, tuple, None] = None,
    ):
        if params is None:
            params = MCParams()
        if not isinstance(params, MCParams):
            raise ValidationError(
                f"params must be MCParams instance, got {type(params).__name__}"
            )
        super().__init__(params)
        self.method = self._resolve_method(method)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an accumulator option using Monte Carlo simulation.

        Args:
            product: An :class:`AccumulatorOption` instance.
            pricing_env: Market data environment.

        Returns:
            Present value of the accumulator.

        Raises:
            PricingError: If the product type is unsupported.
            ValidationError: If pricing inputs are invalid.
        """
        if not isinstance(product, AccumulatorOption):
            raise PricingError(
                "AccumulatorMCEngine only supports AccumulatorOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        strike = product.strike
        maturity = product.get_maturity(pricing_env)

        if is_zero(maturity):
            # At expiry only locked-in accrual and the deterministic terminal
            # extra-shares leg remain -- no rate/dividend/volatility needed.
            self._last_std_error = 0.0
            return product.get_realized_accrual() + self._extra_shares_intrinsic(
                product, spot
            )

        rate = pricing_env.get_rate(maturity)
        div = pricing_env.get_div_yield(maturity)
        vol = pricing_env.get_vol(strike, maturity)
        # Term-structure context for this pricing call (see term_inputs.py)
        self._term_ctx = (pricing_env, strike)
        self._df = make_df_fn(pricing_env)


        self._validate_inputs(spot, strike, maturity, rate, div, vol, product)

        realized = product.get_realized_accrual()
        if product.settlement_at_expiry:
            realized *= float(pricing_env.get_discount_factor(maturity))

        times = product.get_observation_times()
        if not times:
            # No accrual fixings, but a terminal extra-shares leg may still pay.
            extra_val, std_error = self._price_terminal_extra(
                product, pricing_env, spot, strike, maturity, rate, div, vol
            )
            self._last_std_error = std_error
            return realized + extra_val

        price, std_error = self._price_mc(
            product, pricing_env, spot, strike, maturity, rate, div, vol, times
        )
        price += realized
        self._last_std_error = std_error
        return price

    def _extra_shares_intrinsic(self, product: AccumulatorOption, spot: float) -> float:
        """Deterministic terminal value of the extra-shares leg at expiry."""
        extra = product.extra_shares_at_expiry
        if extra <= 0.0 or spot >= product.knock_out_barrier:
            return 0.0
        return -extra * max(product.strike - spot, 0.0) * product.contract_multiplier

    def _price_terminal_extra(
        self, product, pricing_env, spot, strike, maturity, rate, div, vol
    ) -> Tuple[float, float]:
        """Value the terminal extra-shares leg alone (no accrual fixings)."""
        extra = product.extra_shares_at_expiry
        if extra <= 0.0:
            return 0.0, 0.0
        dt_array = np.array([maturity], dtype=float)
        generator = self._create_path_generator(spot, rate, div, vol, maturity, dt_array)
        paths, _ = generator.generate_paths(return_aux=True)
        terminal = paths[:, -1]
        put = np.maximum(strike - terminal, 0.0)
        alive = terminal < product.knock_out_barrier  # expiry up-and-out check
        df_T = float(pricing_env.get_discount_factor(maturity))
        discounted = -extra * put * product.contract_multiplier * df_T * alive
        return self._mean_and_std_error(discounted)

    # ------------------------------------------------------------------
    # Monte Carlo core
    # ------------------------------------------------------------------

    def _price_mc(
        self,
        product: AccumulatorOption,
        pricing_env: PricingEnvironment,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        times: list,
    ) -> Tuple[float, float]:
        obs_times = np.asarray(times, dtype=float)
        # Grid hits every observation date and the contract maturity.
        if is_close(float(obs_times[-1]), maturity):
            all_times = obs_times.copy()
        else:
            all_times = np.concatenate([obs_times, [maturity]])
        dt_array = np.diff(np.concatenate([[0.0], all_times]))
        obs_indices = np.searchsorted(all_times, obs_times)

        generator = self._create_path_generator(
            spot, rate, div, vol, maturity, dt_array
        )
        paths, _ = generator.generate_paths(return_aux=True)

        obs_prices = paths[:, obs_indices + 1]  # (num_paths, n_obs)
        terminal_prices = paths[:, -1]

        daily = product.daily_share_accumulation
        mult = product.contract_multiplier
        ko = product.knock_out_barrier

        # Per-observation accrual assuming the day is alive (full economic value).
        gain = obs_prices - strike
        leverage = np.where(obs_prices >= strike, 1.0, product.gearing)
        accrual = leverage * gain * daily * mult  # (num_paths, n_obs)

        # Discount factor for each observation: settle on the observation date,
        # or defer all accruals to maturity when settlement_at_expiry.
        if product.settlement_at_expiry:
            df_obs = np.full(
                obs_times.shape, float(pricing_env.get_discount_factor(maturity))
            )
        else:
            df_obs = np.array(
                [float(pricing_env.get_discount_factor(t)) for t in obs_times]
            )

        hit_matrix = obs_prices >= ko  # (num_paths, n_obs)

        if product.knock_out_type == AccumulatorKnockOutType.TERMINATION:
            discounted = self._terminate_payoffs(
                product, pricing_env, accrual, df_obs, hit_matrix, obs_times,
                terminal_prices, strike, maturity, mult,
            )
        else:
            discounted = self._single_day_payoffs(
                product, pricing_env, accrual, df_obs, hit_matrix,
                terminal_prices, strike, maturity, mult,
            )

        return self._mean_and_std_error(discounted)

    def _terminate_payoffs(
        self, product, pricing_env, accrual, df_obs, hit_matrix, obs_times,
        terminal_prices, strike, maturity, mult,
    ) -> np.ndarray:
        """Discounted path payoffs for TERMINATION knock-out."""
        n_obs = hit_matrix.shape[1]
        hit_any = hit_matrix.any(axis=1)
        first_idx = np.argmax(hit_matrix, axis=1)

        # Accrue observations strictly before the first knock-out; for paths that
        # never knock out, accrue every observation.
        col = np.arange(n_obs)[None, :]
        accrue_mask = np.where(
            hit_any[:, None], col < first_idx[:, None], True
        )
        discounted = np.sum(accrual * df_obs[None, :] * accrue_mask, axis=1)

        # Knock-out rebate paid at the hit time (payment_at_hit).
        rebate_full = product.get_knock_out_rebate_cash() * mult
        if rebate_full != 0.0:
            hit_times = obs_times[first_idx]
            df_hit = np.array(
                [float(pricing_env.get_discount_factor(t)) for t in hit_times]
            )
            discounted = discounted + np.where(hit_any, rebate_full * df_hit, 0.0)

        # Extra shares at expiry: short up-and-out put, alive only if neither an
        # observation nor the terminal spot breached the barrier. (The terminal
        # check is uniform with the SINGLE_DAY and no-fixings paths; it never
        # changes the payoff here since KO > strike makes the put zero whenever
        # the terminal barrier is breached.)
        extra = product.extra_shares_at_expiry
        if extra > 0.0:
            put = np.maximum(strike - terminal_prices, 0.0)
            df_T = float(pricing_env.get_discount_factor(maturity))
            alive = (~hit_any) & (terminal_prices < product.knock_out_barrier)
            discounted = discounted - extra * put * mult * df_T * alive

        return discounted

    def _single_day_payoffs(
        self, product, pricing_env, accrual, df_obs, hit_matrix,
        terminal_prices, strike, maturity, mult,
    ) -> np.ndarray:
        """Discounted path payoffs for SINGLE_DAY knock-out."""
        # Accrue every observation whose spot is below the barrier.
        alive_mask = ~hit_matrix
        discounted = np.sum(accrual * df_obs[None, :] * alive_mask, axis=1)

        # Extra shares at expiry: short up-and-out put checked only at expiry.
        extra = product.extra_shares_at_expiry
        if extra > 0.0:
            put = np.maximum(strike - terminal_prices, 0.0)
            df_T = float(pricing_env.get_discount_factor(maturity))
            alive = terminal_prices < product.knock_out_barrier
            discounted = discounted - extra * put * mult * df_T * alive

        return discounted

    # ------------------------------------------------------------------
    # Path generation & helpers
    # ------------------------------------------------------------------

    def _create_path_generator(
        self, spot, rate, div, vol, maturity, dt_array
    ) -> GBMPathGenerator:
        """Create a GBM path generator for the selected MC method."""
        num_paths = int(self.params.num_paths)
        if num_paths <= 0:
            raise ValidationError(f"num_paths must be positive, got {num_paths}")

        if self.method == MonteCarloMethod.PSEUDO:
            random_stream = PseudoRandomNormalGenerator(seed=self.params.seed)
            is_qmc = False
        elif self.method in (
            MonteCarloMethod.QUASI,
            MonteCarloMethod.RANDOMIZED_QUASI,
        ):
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
            num_paths=num_paths,
            model="bsm",
            random_stream=random_stream,
            vr_config=vr_config,
            is_qmc=is_qmc,
            dt_array=dt_array,
        )

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
                valid = [m.name for m in MonteCarloMethod]
                raise ValidationError(
                    f"Invalid method string '{method}'. Valid methods: {valid}"
                )
        raise ValidationError(
            f"Invalid method type {type(method).__name__}. "
            "Expected MonteCarloMethod, tuple, str, or None"
        )

    def _validate_inputs(
        self, spot, strike, maturity, rate, div, vol, product
    ) -> None:
        """Validate pricing inputs."""
        validate_positive(spot, "spot")
        validate_positive(strike, "strike")
        validate_non_negative(maturity, "maturity")
        validate_positive(vol, "volatility")
        validate_positive(product.knock_out_barrier, "knock_out_barrier")
        validate_non_negative(product.gearing, "gearing")
        validate_non_negative(
            product.daily_share_accumulation, "daily_share_accumulation"
        )
        validate_positive(product.contract_multiplier, "contract_multiplier")
        if abs(rate) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {rate}")

    def _mean_and_std_error(self, discounted: np.ndarray) -> Tuple[float, float]:
        mean_payoff = float(discounted.mean())
        if len(discounted) <= 1:
            return mean_payoff, 0.0
        std_payoff = float(discounted.std(ddof=1))
        return mean_payoff, std_payoff / math.sqrt(len(discounted))

    def get_last_std_error(self) -> Optional[float]:
        """Return the standard error from the most recent pricing run."""
        return getattr(self, "_last_std_error", None)

    def __repr__(self) -> str:
        return f"AccumulatorMCEngine(method={self.method.name})"
