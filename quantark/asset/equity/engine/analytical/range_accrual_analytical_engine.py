"""
Analytical pricing engine for Range Accrual options via digital option decomposition.

Under GBM, a Range Accrual decomposes into a portfolio of digital options by
linearity of expectation. Each observation contributes independently:

    E[accrual_ratio] = (1/W) * sum_i w_i * P(L_i <= S(t_i) <= U_i)

where P(L <= S(t) <= U) = N(d2_L) - N(d2_U) with standard BSM d2 terms.

The price is:
    Price = exp(-r*T) * S_0 * M * c * tau * E[accrual_ratio]

Historical observations are incorporated deterministically.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.param import EngineParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.range_accrual_option import RangeAccrualOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_log


@dataclass
class RangeAccrualAnalyticalResult:
    """Result container for Range Accrual analytical pricing."""

    price: float
    expected_accrual_ratio: float
    per_observation_probs: List[float]  # P_i for each future observation
    past_in_range_weights: float
    future_expected_in_range_weights: float
    total_weights: float
    num_past_observations: int
    num_future_observations: int


class RangeAccrualAnalyticalEngine(BaseEngine):
    """
    Closed-form pricing for Range Accrual options via digital decomposition.

    Decomposes the Range Accrual payoff into a sum of digital option
    probabilities under Black-Scholes-Merton. Each observation at time t_i
    with barriers [L_i, U_i] contributes:

        P_i = N(d2_L) - N(d2_U)   (standard mode)
        P_i = 1 - [N(d2_L) - N(d2_U)]  (reverse mode)

    where d2(K, t) = [ln(S/K) + (r - q - sigma^2/2)*t] / (sigma*sqrt(t))

    Supports:
    - Weighted observations (e.g., Friday=3 for weekend carry)
    - Historical observations with known in-range outcomes
    - Time-varying barriers (per-observation or scalar)
    - Reverse mode (pay when outside range)
    - Annualized or non-annualized accrual rates
    - Per-observation vol term structure
    """

    engine_type = EngineType.ANALYTICAL
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    supports_lifecycle_state = True

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_OBS_TIME = 30.0

    def __init__(self, params: Optional[EngineParams] = None):
        super().__init__(params)
        self._last_result: Optional[RangeAccrualAnalyticalResult] = None

    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> float:
        """
        Price a Range Accrual option analytically.

        Args:
            product: RangeAccrualOption to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product type is unsupported
            ValidationError: If input parameters are invalid
            NumericalError: If numerical computation fails
        """
        if not isinstance(product, RangeAccrualOption):
            raise PricingError(
                f"RangeAccrualAnalyticalEngine only supports RangeAccrualOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv
        timing = resolve_terminal_timing(product, pricing_env)
        pending_pv = pending_receivable_pv(lifecycle_state, pricing_env)

        if product.range_config is None:
            raise ValidationError("range_config is required for Range Accrual option")

        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)

        self._validate_inputs(S, T, r, q)

        # Handle near-expiry: all observations are effectively past
        if is_zero(T):
            past_in_range, _ = product.get_past_accrual(pricing_env)
            total_weights = product.get_total_weights()
            return (
                product.get_payoff(
                    S,
                    in_range_weights=past_in_range,
                    total_weights=total_weights,
                    pricing_env=pricing_env,
                )
                * timing.delay_df
                + pending_pv
            )

        # Resolve past vs future observations
        past_obs, future_obs, total_weights = product.resolve_observations(pricing_env)

        # Accumulate deterministic past contribution
        past_in_range_weights = sum(w for w, in_range in past_obs if in_range)

        # Compute expected future in-range weights analytically
        future_expected, per_obs_probs = self._compute_future_expected_weights(
            product, pricing_env, S, r, q, future_obs
        )

        # Expected accrual ratio
        total_in_range = past_in_range_weights + future_expected
        if is_zero(total_weights):
            expected_ratio = 0.0
        else:
            expected_ratio = total_in_range / total_weights

        # Compute price
        year_fraction = product.get_year_fraction(pricing_env)
        accrual_rate = product.range_config.accrual_rate
        discount = timing.payment_df

        price = (
            discount
            * product.initial_price
            * product.contract_multiplier
            * accrual_rate
            * expected_ratio
            * year_fraction
        )

        self._last_result = RangeAccrualAnalyticalResult(
            price=price,
            expected_accrual_ratio=expected_ratio,
            per_observation_probs=per_obs_probs,
            past_in_range_weights=past_in_range_weights,
            future_expected_in_range_weights=future_expected,
            total_weights=total_weights,
            num_past_observations=len(past_obs),
            num_future_observations=len(future_obs),
        )

        return price + pending_pv

    def _compute_future_expected_weights(
        self,
        product: RangeAccrualOption,
        pricing_env: PricingEnvironment,
        S: float,
        r: float,
        q: float,
        future_obs: List[Tuple[float, float, int]],
    ) -> Tuple[float, List[float]]:
        """
        Compute expected in-range weights for future observations.

        For each future observation, computes:
            P_i = N(d2_L) - N(d2_U)  (or 1 - that for reverse mode)

        Uses vectorized NumPy for efficiency.

        Args:
            product: Range Accrual option
            pricing_env: Pricing environment
            S: Current spot price
            r: Risk-free rate
            q: Dividend yield
            future_obs: List of (weight, time, obs_idx) for future observations

        Returns:
            Tuple of (total_expected_in_range_weight, per_obs_probabilities)
        """
        if len(future_obs) == 0:
            return 0.0, []

        config = product.range_config
        assert config is not None  # Validated by caller

        n = len(future_obs)
        weights = np.empty(n)
        times = np.empty(n)
        lowers = np.empty(n)
        uppers = np.empty(n)
        sigmas = np.empty(n)

        for i, (w, t, obs_idx) in enumerate(future_obs):
            weights[i] = w
            times[i] = t
            lowers[i] = config.get_lower_barrier(obs_idx)
            uppers[i] = config.get_upper_barrier(obs_idx)
            sigmas[i] = pricing_env.get_vol(product.initial_price, t)

        # Handle observations with near-zero time or near-zero vol deterministically
        probs = np.empty(n)
        degenerate = (times < self.MIN_MATURITY) | (sigmas < self.MIN_VOL)

        if np.any(degenerate):
            deg_mask = degenerate
            # For near-zero time: check current spot
            # For near-zero vol: check forward spot
            fwd = np.where(
                times[deg_mask] < self.MIN_MATURITY,
                S,
                S * np.exp((r - q) * times[deg_mask]),
            )
            in_range = (fwd >= lowers[deg_mask]) & (fwd <= uppers[deg_mask])
            probs[deg_mask] = in_range.astype(float)

        # Process non-degenerate observations vectorized
        normal = ~degenerate
        if np.any(normal):
            t_n = times[normal]
            sig_n = sigmas[normal]
            l_n = lowers[normal]
            u_n = uppers[normal]

            sqrt_t = np.sqrt(t_n)
            drift = (r - q - 0.5 * sig_n * sig_n) * t_n
            denom = sig_n * sqrt_t

            # d2 for lower barrier: P(S(t) >= L)
            d2_L = (np.log(S / l_n) + drift) / denom
            # d2 for upper barrier: P(S(t) >= U)
            d2_U = (np.log(S / u_n) + drift) / denom

            probs[normal] = stats.norm.cdf(d2_L) - stats.norm.cdf(d2_U)

        # Apply reverse mode
        if config.is_reverse:
            probs = 1.0 - probs

        # Clamp probabilities to [0, 1] for numerical safety
        np.clip(probs, 0.0, 1.0, out=probs)

        # Expected in-range weight = sum(w_i * P_i)
        expected_weight = float(np.dot(weights, probs))
        per_obs_probs = probs.tolist()

        return expected_weight, per_obs_probs

    def _validate_inputs(
        self, S: float, T: float, r: float, q: float
    ) -> None:
        """Validate pricing inputs."""
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if abs(r) > 1.0:
            raise ValidationError(
                f"Risk-free rate outside reasonable bounds: {r}"
            )

    def get_last_result(self) -> Optional[RangeAccrualAnalyticalResult]:
        """Get the full result from the last pricing run."""
        return self._last_result

    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> Dict[str, float]:
        """
        Calculate Greeks using analytical formulas where possible.

        Delta and gamma are computed analytically from the derivative of N(d2).
        Falls back to base class bump-and-reprice for vega, theta, rho.

        Args:
            product: Range Accrual option
            pricing_env: Pricing environment

        Returns:
            Dictionary of Greeks: price, delta, gamma
        """
        if not isinstance(product, RangeAccrualOption):
            raise PricingError(
                f"RangeAccrualAnalyticalEngine only supports RangeAccrualOption, "
                f"got {type(product).__name__}"
            )
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return {"price": fixed_pv, "delta": 0.0, "gamma": 0.0}
        timing = resolve_terminal_timing(product, pricing_env)

        config = product.range_config
        if config is None:
            raise ValidationError("range_config is required")

        S = pricing_env.spot
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)

        base_price = self.price(
            product,
            pricing_env,
            lifecycle_state=lifecycle_state,
        )
        greeks: Dict[str, float] = {"price": base_price}

        # Near-expiry: delta/gamma are zero (payoff is flat w.r.t. spot)
        if is_zero(T):
            greeks["delta"] = 0.0
            greeks["gamma"] = 0.0
            return greeks

        _, future_obs, total_weights = product.resolve_observations(pricing_env)

        if len(future_obs) == 0 or is_zero(total_weights):
            greeks["delta"] = 0.0
            greeks["gamma"] = 0.0
            return greeks

        # Analytical delta: dPrice/dS
        # dP_i/dS = [n(d2_L) - n(d2_U)] / (S * sigma_i * sqrt(t_i))
        # where n(.) is the standard normal PDF
        delta_sum = 0.0
        gamma_sum = 0.0

        for w, t, obs_idx in future_obs:
            if t < self.MIN_MATURITY:
                continue

            sigma = pricing_env.get_vol(product.initial_price, t)
            if sigma < self.MIN_VOL:
                continue

            lower = config.get_lower_barrier(obs_idx)
            upper = config.get_upper_barrier(obs_idx)

            sqrt_t = math.sqrt(t)
            sig_sqrt_t = sigma * sqrt_t
            drift = (r - q - 0.5 * sigma * sigma) * t

            d2_L = (safe_log(S / lower) + drift) / sig_sqrt_t
            d2_U = (safe_log(S / upper) + drift) / sig_sqrt_t

            n_d2_L = float(stats.norm.pdf(d2_L))
            n_d2_U = float(stats.norm.pdf(d2_U))

            # dP_i/dS
            dp_ds = (n_d2_L - n_d2_U) / (S * sig_sqrt_t)

            # d^2P_i/dS^2  (second derivative of N(d2(K)) w.r.t. S)
            # d/dS[n(d2)/(S*sig*sqrt_t)] =
            #   -d2*n(d2)/(S^2*sig^2*t) - n(d2)/(S^2*sig*sqrt_t)
            s2 = S * S
            sig2_t = sigma * sigma * t
            d2p_ds2 = (
                -(d2_L * n_d2_L - d2_U * n_d2_U) / (s2 * sig2_t)
                - (n_d2_L - n_d2_U) / (s2 * sig_sqrt_t)
            )

            sign = -1.0 if config.is_reverse else 1.0
            delta_sum += w * dp_ds * sign
            gamma_sum += w * d2p_ds2 * sign

        # Scale by payoff parameters
        year_fraction = product.get_year_fraction(pricing_env)
        accrual_rate = config.accrual_rate
        discount = timing.payment_df
        scale = (
            discount
            * product.initial_price
            * product.contract_multiplier
            * accrual_rate
            * year_fraction
            / total_weights
        )

        greeks["delta"] = scale * delta_sum
        greeks["gamma"] = scale * gamma_sum

        return greeks

    def __repr__(self):
        return "RangeAccrualAnalyticalEngine()"
