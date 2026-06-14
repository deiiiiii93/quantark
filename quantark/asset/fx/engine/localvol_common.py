"""Shared helpers for FX local-volatility engines (v1 restrictions, surface, sizing)."""

from __future__ import annotations

from typing import Optional

from quantark.param import GridVolSurface
from quantark.priceenv import FxPricingEnvironment
from quantark.util.numerical import is_close
from quantark.util.exceptions import PricingError
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol


def check_fx_v1_restrictions(product, fx_env: FxPricingEnvironment) -> None:
    """Reject FX configurations outside the v1 local-vol scope (no silent flattening)."""
    if getattr(fx_env, "market_forward", None) is not None:
        raise PricingError("FX local-vol v1 does not support market_forward override")
    if getattr(fx_env, "spot_days", 0) != 0:
        raise PricingError("FX local-vol v1 requires spot_days == 0")
    tau = float(product.get_maturity(fx_env))
    tau_delivery = float(product.get_delivery(fx_env))
    if not is_close(tau, tau_delivery, abs_tol=1e-9):
        raise PricingError(
            f"FX local-vol v1 requires expiry == delivery (got {tau} vs {tau_delivery})"
        )


def build_fx_local_vol(
    fx_env: FxPricingEnvironment, prebuilt: Optional[LocalVolSurface]
) -> LocalVolSurface:
    if prebuilt is not None:
        return prebuilt
    if not isinstance(fx_env.vol_surface, GridVolSurface):
        raise PricingError(
            "FX local-vol engine requires a GridVolSurface (market IV grid) or a "
            "prebuilt LocalVolSurface"
        )
    return build_dupire_local_vol(
        fx_env.vol_surface, spot=float(fx_env.effective_spot()),
        rate_curve=fx_env.domestic_curve,
        div_yield=lambda t: float(fx_env.foreign_curve.get_rate(t)),
    )


def fx_contract_value(product, fx_env: FxPricingEnvironment, unit_price: float) -> float:
    """Scale a per-unit-foreign domestic price by notional, participation, annualization,
    matching GarmanKohlhagenEngine sizing (the per-unit price already includes the
    domestic discount factor to delivery)."""
    return (
        float(product.notional)
        * float(unit_price)
        * float(product.participation_rate)
        * float(product.annualization_factor(fx_env))
    )
