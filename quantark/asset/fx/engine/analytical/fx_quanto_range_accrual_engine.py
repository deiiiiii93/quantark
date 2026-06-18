"""
Analytical engine for FX quanto range accruals.

A quanto range accrual is a range accrual whose domestic-currency coupon is
paid in a third settlement currency at a fixed ``quanto_fx_rate``. Under the
settlement-currency measure the underlying pair's drift is shifted by the
quanto adjustment ``mu_q = sign * rho * sigma * sigma_quanto`` (sign set by the
conversion orientation), so each fixing's range digital is evaluated against
the quanto-adjusted forward ``F_q(t) = F(t) * exp(mu_q * t)``. The coupon is
converted at ``quanto_fx_rate`` and discounted on the settlement curve.

This reuses :class:`FxRangeAccrualAnalyticalEngine` wholesale and overrides only
the forward, the replication leg, the discount curve, and the conversion
factor. Both per-fixing methods (DIGITAL_COMBINATION / CALL_PUT_SPREAD) and both
surfaces (flat BS / Vanna-Volga) are supported exactly as in the vanilla case.

Smile note: ``mu_q`` uses each barrier's own implied vol ``sigma(K)``, matching
the existing quanto vanilla/digital engines (which use the strike vol). The
Vanna-Volga surface remains single-tenor.
"""

from typing import Optional, Union

from quantark.asset.fx.engine.analytical.fx_range_accrual_engine import (
    FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.engine.analytical.fx_quanto_vanilla_engine import (
    GarmanKohlhagenQuantoEngine,
    _quanto_forward,
    _require_quanto,
)
from quantark.asset.fx.engine.base_fx_engine import FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.asset.fx.product.option.fx_quanto_range_accrual_option import (
    FxQuantoRangeAccrualOption,
)
from quantark.asset.fx.product.option.fx_quanto_vanilla_option import (
    FxQuantoVanillaOption,
)
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import FxRangeAccrualMethod, OptionType
from quantark.util.exceptions import PricingError, ValidationError


class FxQuantoRangeAccrualAnalyticalEngine(FxRangeAccrualAnalyticalEngine):
    """
    Closed-form FX quanto range accrual pricing in the settlement currency.

    Inherits the full range-digital decomposition (weights, per-fixing
    barriers, settled fixings, reverse mode, EXPIRY/INSTANT pay) and substitutes
    the quanto drift, settlement-curve discounting, and fixed conversion through
    the base engine's override hooks.
    """

    def __init__(
        self,
        method: Union[str, FxRangeAccrualMethod, tuple, None] = None,
        params: Optional[FxEngineParams] = None,
    ):
        super().__init__(method=method, params=params)
        # Replication legs must carry the quanto drift, so price them with the
        # quanto vanilla engine rather than plain Garman-Kohlhagen.
        self._vanilla_engine = GarmanKohlhagenQuantoEngine()

    # ------------------------------------------------------------------
    # Quanto override hooks
    # ------------------------------------------------------------------

    def _forward_for_prob(
        self, fx_env: FxPricingEnvironment, strike: float, sigma: float, t: float
    ) -> float:
        """Quanto-adjusted forward using this barrier's implied vol."""
        quanto = _require_quanto(fx_env)
        return _quanto_forward(fx_env, quanto, sigma, t)

    def _call_leg_price(
        self, fx_env: FxPricingEnvironment, strike: float, t: float
    ) -> float:
        """Unit-notional quanto vanilla call (quanto rate 1.0; conversion applied
        separately) expiring at ``t`` — its ``-dC/dK`` is ``df_settle * N(d2_q)``.
        """
        leg = FxQuantoVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            quanto_fx_rate=1.0,
            maturity=t,
            delivery=t,
            notional_foreign=1.0,
        )
        return self._vanilla_engine.price(leg, fx_env)

    def _leg_discount_df(self, fx_env: FxPricingEnvironment, t: float) -> float:
        """Replication legs discount on the settlement curve."""
        return _require_quanto(fx_env).settlement_curve.get_discount_factor(t)

    def _payment_df(self, fx_env: FxPricingEnvironment, t: float) -> float:
        """Coupon discounts on the settlement curve."""
        return _require_quanto(fx_env).settlement_curve.get_discount_factor(t)

    def _conversion_factor(self, option: FxQuantoRangeAccrualOption) -> float:
        """Fixed domestic-to-settlement conversion of the coupon."""
        return option.quanto_fx_rate

    # ------------------------------------------------------------------
    # Product check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_product(product: BaseFxProduct) -> FxQuantoRangeAccrualOption:
        if not isinstance(product, FxQuantoRangeAccrualOption):
            raise PricingError(
                f"FxQuantoRangeAccrualAnalyticalEngine only supports "
                f"FxQuantoRangeAccrualOption, got {type(product).__name__}"
            )
        if product.range_config is None:
            raise ValidationError("range_config is required")
        return product

    def __repr__(self):
        return f"FxQuantoRangeAccrualAnalyticalEngine(method={self.method.value})"
