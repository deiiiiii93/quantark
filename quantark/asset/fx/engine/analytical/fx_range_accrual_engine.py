"""
Analytical engine for FX range accruals via range-digital decomposition.

By linearity of expectation a range-accrual coupon decomposes into one range
digital per fixing:

    E[in-range fraction] = (1 / W) * sum_i w_i * P(L_i < X_{t_i} < U_i)

and the present value of the domestic-currency coupon is

    PV = notional * accrual_rate * year_fraction * (1 / W)
         * [ sum_past w_i * 1{in range} * DF_i
           + sum_future w_i * P_i * DF_i ]

where ``DF_i`` discounts the fixing's contribution to today (the maturity
discount factor for EXPIRY pay, the fixing-date discount factor for INSTANT
pay). Each ``P_i`` is computed under Garman-Kohlhagen dynamics with the parity
forward ``F(t) = S * DF_f(t) / DF_d(t)``.

Two per-fixing pricing methods are supported (``FxRangeAccrualMethod``):

- DIGITAL_COMBINATION: closed-form ``N(d2_L) - N(d2_U)`` at the surface vol.
  Exact under a flat / Black-Scholes surface; the sticky-strike, level-only
  approximation under a Vanna-Volga smile.
- CALL_PUT_SPREAD: static replication ``[C(K-h) - C(K+h)] / (2h)`` with the
  vanilla legs priced through the surface, capturing the smile skew under
  Vanna-Volga. Under a flat surface it reproduces DIGITAL_COMBINATION.

Smile note: the project's Vanna-Volga surface is single-tenor (its ``tau`` is an
anchor; ``time_to_maturity`` is advisory). A multi-fixing accrual therefore
applies one smile shape across all fixing times. Anchor the surface near the
accrual horizon; widely dispersed fixings are an approximation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from scipy.stats import norm

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.engine.analytical.garman_kohlhagen_engine import (
    GarmanKohlhagenEngine,
)
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_range_accrual_option import (
    FxRangeAccrualOption,
)
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import CouponPayType, FxRangeAccrualMethod, OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_log, safe_sqrt


@dataclass
class FxRangeAccrualResult:
    """Full breakdown of an FX range accrual pricing run."""

    price: float
    expected_accrual_ratio: float
    per_observation_probs: List[float]
    past_in_range_weights: float
    future_expected_in_range_weights: float
    total_weights: float
    num_past_observations: int
    num_future_observations: int


class FxRangeAccrualAnalyticalEngine(BaseFxEngine):
    """
    Closed-form FX range accrual pricing via range-digital decomposition.

    Supports weighted and per-fixing barriers, historical (settled) fixings,
    reverse-range accrual, EXPIRY / INSTANT coupon timing, and both flat-BS and
    Vanna-Volga surfaces under either per-fixing pricing method.
    """

    engine_type = EngineType.ANALYTICAL

    MIN_VOL = 1e-3
    MIN_MATURITY = 1e-10

    # Relative strike bump for the call/put-spread replication of each digital.
    _H_REL = 1e-4
    _H_FLOOR = 1e-6

    def __init__(
        self,
        method: Union[str, FxRangeAccrualMethod, tuple, None] = None,
        params: Optional[FxEngineParams] = None,
    ):
        super().__init__(params)
        self.method = self._resolve_method(method)
        self._vanilla_engine = GarmanKohlhagenEngine()
        self._last_result: Optional[FxRangeAccrualResult] = None

    @staticmethod
    def _resolve_method(
        method: Union[str, FxRangeAccrualMethod, tuple, None],
    ) -> FxRangeAccrualMethod:
        """Normalise the method argument (default: DIGITAL_COMBINATION)."""
        if method is None:
            return FxRangeAccrualMethod.DIGITAL_COMBINATION
        if isinstance(method, tuple):
            engine_type, m = method
            if engine_type != EngineType.ANALYTICAL:
                raise ValidationError(
                    f"Expected EngineType.ANALYTICAL, got {engine_type}"
                )
            if not isinstance(m, FxRangeAccrualMethod):
                raise ValidationError(
                    f"Expected FxRangeAccrualMethod, got {type(m).__name__}"
                )
            return m
        if isinstance(method, FxRangeAccrualMethod):
            return method
        if isinstance(method, str):
            try:
                return FxRangeAccrualMethod(method.lower())
            except ValueError:
                valid = ", ".join(m.value for m in FxRangeAccrualMethod)
                raise ValidationError(
                    f"Invalid method '{method}'. Valid methods: {valid}"
                )
        raise ValidationError(f"Unsupported method type: {type(method).__name__}")

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(self, product: BaseFxProduct, fx_env: FxPricingEnvironment) -> float:
        """Price an FX range accrual; full breakdown via :meth:`get_last_result`."""
        option = self._check_product(product)
        config = option.range_config

        spot = fx_env.effective_spot()
        if spot <= 0:
            raise ValidationError(f"Spot must be positive, got {spot}")

        T_pay = option.get_delivery(fx_env)
        year_fraction = option.get_year_fraction(fx_env)
        scale = (
            option.notional
            * config.accrual_rate
            * year_fraction
            * self._conversion_factor(option)
        )

        T = option.get_maturity(fx_env)
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")

        # Near-expiry: every fixing is settled; the coupon is deterministic.
        if is_zero(T):
            past_in_range, _ = option.get_past_accrual(fx_env)
            total = option.get_total_weights()
            ratio = 0.0 if is_zero(total) else past_in_range / total
            price = scale * ratio * self._payment_df(fx_env, T_pay)
            self._last_result = FxRangeAccrualResult(
                price=price,
                expected_accrual_ratio=ratio,
                per_observation_probs=[],
                past_in_range_weights=past_in_range,
                future_expected_in_range_weights=0.0,
                total_weights=total,
                num_past_observations=0,
                num_future_observations=0,
            )
            return price

        past_obs, future_obs, total_weights = option.resolve_observations(fx_env)
        if is_zero(total_weights):
            raise ValidationError("Total observation weight must be positive")

        pay_type = config.accrual_pay_type
        df_pay = self._payment_df(fx_env, T_pay)

        # Past (settled) fixings. Under EXPIRY they fold into the final coupon
        # discounted from maturity; under INSTANT they were already paid and so
        # carry no present value.
        if pay_type == CouponPayType.EXPIRY:
            past_in_range = sum(w for w, ok in past_obs if ok)
            past_pv_weight = past_in_range * df_pay
        else:
            past_in_range = sum(w for w, ok in past_obs if ok)
            past_pv_weight = 0.0

        # Future fixings.
        future_pv_weight, future_expected, per_obs_probs = self._future_contributions(
            option, fx_env, future_obs, df_pay
        )

        price = scale * (past_pv_weight + future_pv_weight) / total_weights

        expected_ratio = (
            0.0
            if is_zero(total_weights)
            else (past_in_range + future_expected) / total_weights
        )
        self._last_result = FxRangeAccrualResult(
            price=price,
            expected_accrual_ratio=expected_ratio,
            per_observation_probs=per_obs_probs,
            past_in_range_weights=past_in_range,
            future_expected_in_range_weights=future_expected,
            total_weights=total_weights,
            num_past_observations=len(past_obs),
            num_future_observations=len(future_obs),
        )
        return price

    def _future_contributions(
        self,
        option: FxRangeAccrualOption,
        fx_env: FxPricingEnvironment,
        future_obs: List[Tuple[float, float, int]],
        df_pay: float,
    ) -> Tuple[float, float, List[float]]:
        """
        Discounted in-range weight, expected in-range weight, and per-fixing
        probabilities for the future fixings.
        """
        if len(future_obs) == 0:
            return 0.0, 0.0, []

        config = option.range_config
        instant = config.accrual_pay_type == CouponPayType.INSTANT

        pv_weight = 0.0
        expected_weight = 0.0
        probs: List[float] = []

        for w, t, obs_idx in future_obs:
            lower, upper = option.barriers_for(obs_idx)
            p = self._range_probability(fx_env, lower, upper, t)
            if config.is_reverse:
                p = 1.0 - p
            p = min(1.0, max(0.0, p))

            df_i = self._payment_df(fx_env, t) if instant else df_pay
            pv_weight += w * p * df_i
            expected_weight += w * p
            probs.append(p)

        return pv_weight, expected_weight, probs

    def _range_probability(
        self,
        fx_env: FxPricingEnvironment,
        lower: float,
        upper: float,
        t: float,
    ) -> float:
        """
        ``P(lower < X_t < upper)`` for one fixing, before reverse-mode handling.

        Degenerate fixings (vanishing time or vol) collapse to a deterministic
        forward in-range indicator. Otherwise the selected method is applied at
        each barrier with that barrier's own implied vol.
        """
        sigma_l = fx_env.get_vol(lower, t)
        sigma_u = fx_env.get_vol(upper, t)
        if t < self.MIN_MATURITY or min(sigma_l, sigma_u) < self.MIN_VOL:
            fwd = fx_env.get_forward(t)
            return 1.0 if (lower <= fwd <= upper) else 0.0

        prob_above_l = self._prob_above_strike(fx_env, lower, sigma_l, t)
        prob_above_u = self._prob_above_strike(fx_env, upper, sigma_u, t)
        return prob_above_l - prob_above_u

    def _prob_above_strike(
        self, fx_env: FxPricingEnvironment, strike: float, sigma: float, t: float
    ) -> float:
        """``P(X_t > strike)`` by the configured per-fixing method."""
        if self.method == FxRangeAccrualMethod.DIGITAL_COMBINATION:
            fwd = self._forward_for_prob(fx_env, strike, sigma, t)
            sig_sqrt_t = sigma * float(safe_sqrt(t))
            d2 = (
                float(safe_log(fwd / strike)) - 0.5 * sigma * sigma * t
            ) / sig_sqrt_t
            return float(norm.cdf(d2))
        return self._replicated_prob(fx_env, strike, t)

    def _replicated_prob(
        self, fx_env: FxPricingEnvironment, strike: float, t: float
    ) -> float:
        """
        ``P(X_t > strike)`` via a centered call-spread replication of the digital.

        The cash-or-nothing digital call equals ``-dC/dK`` and is approximated by
        ``[C(K-h) - C(K+h)] / (2h)`` with both vanilla legs priced through the
        surface (so the Vanna-Volga skew enters). Dividing by the leg discount
        factor recovers the (undiscounted) risk-neutral probability.
        """
        h = max(self._H_FLOOR, self._H_REL * strike)
        if strike - h <= 0.0:
            raise PricingError(
                f"Replication strike bump h={h} too large for strike {strike}"
            )
        c_dn = self._call_leg_price(fx_env, strike - h, t)
        c_up = self._call_leg_price(fx_env, strike + h, t)
        cash_digital = (c_dn - c_up) / (2.0 * h)  # = df(t) * N_smile(d2)
        df_t = self._leg_discount_df(fx_env, t)
        if is_zero(df_t):
            raise PricingError("Vanishing discount factor in replication")
        return cash_digital / df_t

    # ------------------------------------------------------------------
    # Override hooks (the quanto engine substitutes the drift / discounting /
    # conversion through these; the vanilla engine uses the GK defaults).
    # ------------------------------------------------------------------

    def _forward_for_prob(
        self, fx_env: FxPricingEnvironment, strike: float, sigma: float, t: float
    ) -> float:
        """Forward used in the digital-combination ``d2`` (parity forward)."""
        return fx_env.get_forward(t)

    def _call_leg_price(
        self, fx_env: FxPricingEnvironment, strike: float, t: float
    ) -> float:
        """Smile-consistent unit-foreign-notional vanilla call expiring at ``t``."""
        vanilla = FxVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            maturity=t,
            delivery=t,
            notional_foreign=1.0,
        )
        return self._vanilla_engine.price(vanilla, fx_env)

    def _leg_discount_df(self, fx_env: FxPricingEnvironment, t: float) -> float:
        """Discount factor embedded in the replication legs (domestic curve)."""
        return fx_env.get_domestic_df(t)

    def _payment_df(self, fx_env: FxPricingEnvironment, t: float) -> float:
        """Discount factor for a coupon paid at time ``t`` (domestic curve)."""
        return fx_env.get_domestic_df(t)

    def _conversion_factor(self, option: FxRangeAccrualOption) -> float:
        """Fixed payout conversion factor (1.0 for a domestic-currency coupon)."""
        return 1.0

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def calculate_greeks(
        self, product: BaseFxProduct, fx_env: FxPricingEnvironment
    ) -> Dict[str, float]:
        """FX Greeks by central bump-and-reprice (base-class conventions).

        Bump-and-reprice keeps both pricing methods and the Vanna-Volga
        re-anchoring consistent: the smile re-binds to the bumped market before
        each reprice, so delta/gamma capture the sticky-delta smile dynamics.
        """
        self._check_product(product)
        return super().calculate_greeks(product, fx_env)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxRangeAccrualOption:
        if not isinstance(product, FxRangeAccrualOption):
            raise PricingError(
                f"FxRangeAccrualAnalyticalEngine only supports "
                f"FxRangeAccrualOption, got {type(product).__name__}"
            )
        # Specialized subclasses (quanto / foreign coupon) price under a
        # different measure or currency; route them to their own engine rather
        # than silently valuing them as a plain domestic-coupon accrual.
        if type(product) is not FxRangeAccrualOption:
            raise PricingError(
                f"FxRangeAccrualAnalyticalEngine prices plain domestic-coupon "
                f"range accruals; use the engine matching "
                f"{type(product).__name__}"
            )
        if product.range_config is None:
            raise ValidationError("range_config is required")
        return product

    def get_last_result(self) -> Optional[FxRangeAccrualResult]:
        """Full breakdown from the most recent :meth:`price` call."""
        return self._last_result

    def __repr__(self):
        return f"FxRangeAccrualAnalyticalEngine(method={self.method.value})"
