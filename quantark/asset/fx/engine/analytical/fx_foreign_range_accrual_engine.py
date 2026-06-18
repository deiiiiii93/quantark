"""
Analytical engine for FX foreign-currency-coupon range accruals.

The coupon is paid in the foreign (base) currency. Receiving one foreign unit
at payment is worth ``X_T`` domestic units there, so each in-range fixing is a
cash-or-nothing digital *under the foreign measure*: its in-the-money
probability is ``N(d1)`` rather than the domestic ``N(d2)``, and one unit of
coupon paid at ``t`` is worth ``S_eff * DF_foreign(t)`` domestic today.

For a fixing at ``t_i`` paid at ``T_pay >= t_i`` the value is

    PV_i = S_eff * DF_foreign(T_pay) * [N(d1_L) - N(d1_U)]

with ``d1`` evaluated at the *fixing-date* forward ``F(0, t_i)`` and variance
``sigma^2 * t_i`` (a martingale argument collapses the fixing-before-payment
case to the fixing-date forward while the foreign unit discounts to ``T_pay``).

Reuses :class:`FxRangeAccrualAnalyticalEngine` and overrides only the per-fixing
probability (``N(d1)`` via the asset-or-nothing identity
``AoN(>K) = call(K) + K * cashdigital(K)``, so the Vanna-Volga skew enters under
the spread method) and the coupon value factor. Both per-fixing methods and both
surfaces are supported exactly as in the domestic case.
"""

from scipy.stats import norm

from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import (
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_foreign_range_accrual_option import (
    FxForeignRangeAccrualOption,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxRangeAccrualMethod
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_zero, safe_log, safe_sqrt


class FxForeignRangeAccrualAnalyticalEngine(FxRangeAccrualAnalyticalEngine):
    """
    Closed-form FX foreign-coupon range accrual pricing (value in domestic ccy).

    Inherits the full range-digital decomposition and substitutes the foreign
    measure (``N(d1)``) and the foreign-unit value factor ``S_eff * DF_foreign``
    through the base engine's override hooks.
    """

    # ------------------------------------------------------------------
    # Foreign-measure override hooks
    # ------------------------------------------------------------------

    def _prob_above_strike(
        self, fx_env: FxPricingEnvironment, strike: float, sigma: float, t: float
    ) -> float:
        """Foreign-measure ``P(X_t > strike) = N(d1)`` by the configured method."""
        if self.method == FxRangeAccrualMethod.DIGITAL_COMBINATION:
            fwd = fx_env.get_forward(t)
            sig_sqrt_t = sigma * float(safe_sqrt(t))
            d1 = (
                float(safe_log(fwd / strike)) + 0.5 * sigma * sigma * t
            ) / sig_sqrt_t
            return float(norm.cdf(d1))
        return self._asset_or_nothing_prob(fx_env, strike, t)

    def _asset_or_nothing_prob(
        self, fx_env: FxPricingEnvironment, strike: float, t: float
    ) -> float:
        """
        ``N_smile(d1) = AoN(>K) / (S_eff * DF_foreign(t))`` via static replication.

        The asset-or-nothing call ``AoN(>K)`` (pays ``X_t`` domestic if in money)
        decomposes as ``call(K) + K * cashdigital(K)``; pricing the legs through
        the surface captures the smile skew. Dividing by ``S_eff * DF_foreign(t)``
        (the value of one foreign unit) recovers the foreign-measure probability.
        """
        h = max(self._H_FLOOR, self._H_REL * strike)
        if strike - h <= 0.0:
            raise PricingError(
                f"Replication strike bump h={h} too large for strike {strike}"
            )
        c_k = self._call_leg_price(fx_env, strike, t)
        c_dn = self._call_leg_price(fx_env, strike - h, t)
        c_up = self._call_leg_price(fx_env, strike + h, t)
        cash_digital = (c_dn - c_up) / (2.0 * h)  # = DF_dom(t) * N_smile(d2)
        aon = c_k + strike * cash_digital  # = S_eff * DF_foreign(t) * N_smile(d1)

        denom = fx_env.effective_spot() * fx_env.get_foreign_df(t)
        if is_zero(denom):
            raise PricingError("Vanishing foreign-unit value in replication")
        return aon / denom

    def _payment_df(self, fx_env: FxPricingEnvironment, t: float) -> float:
        """Domestic present value of one foreign-currency unit paid at ``t``."""
        return fx_env.effective_spot() * fx_env.get_foreign_df(t)

    # ------------------------------------------------------------------
    # Product check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxForeignRangeAccrualOption:
        if not isinstance(product, FxForeignRangeAccrualOption):
            raise PricingError(
                f"FxForeignRangeAccrualAnalyticalEngine only supports "
                f"FxForeignRangeAccrualOption, got {type(product).__name__}"
            )
        if product.range_config is None:
            raise ValidationError("range_config is required")
        return product

    def __repr__(self):
        return f"FxForeignRangeAccrualAnalyticalEngine(method={self.method.value})"
