"""Risk-stack bridge for Total Return Swaps.

The realized-cashflow :class:`TotalReturnSwapEngine` prices a TRS from an observed
price path and returns a per-period cashflow table. The risk modules (Position,
VaR, stress, dynamic scenario, backtest) speak a different language: a
mark-to-market value as a function of a :class:`PricingEnvironment`, plus
bump-revaluation greeks.

:class:`TRSValuationEngine` bridges the two. It maps a ``PricingEnvironment``'s
spot (and, opt-in, funding rate) onto the *valuation-date slice* of the TRS price
path, re-runs the cashflow engine, and reads the valuation-date ``present_value``
as the swap mark-to-market. A TRS is delta-one on its float leg, so spot delta is
recovered by bumping that valuation-date observation and re-pricing, and the
funding-rate exposure by bumping ``fix_leg.rate``.

Because the bridge consumes a standard ``PricingEnvironment``, the revaluation
machinery in VaR / stress / dynamic-scenario operates on a swap position with no
changes to those engines.
"""

from dataclasses import replace
from datetime import datetime
from typing import Dict, Optional

from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.product.swap.trs_params import PricingParams, TRSParams
from quantark.asset.equity.engine.cashflow.total_return_swap_engine import (
    TotalReturnSwapEngine,
)
from quantark.util.exceptions import MarketDataError, ValidationError

# Present value is extracted at high precision so finite-difference greeks are not
# swamped by the engine's display rounding.
_PV_PRECISION = 10


class TRSValuationEngine:
    """Marks a single-asset TRS to market under a :class:`PricingEnvironment`.

    Args:
        base_params: The contractual TRS parameters (observed path, legs, events,
            margin, valuation date).
        funding_rate_ref: Reference funding rate the contractual ``fix_leg.rate``
            was struck against. When ``None`` (default) the financing leg is a
            fixed contractual rate and env rate shocks do not move the base MtM
            (``rho`` is still available by directly bumping ``fix_leg.rate``).
            When supplied, the financing leg floats: ``financing(env) =
            fix_leg.rate + (env.get_rate(tenor) - funding_rate_ref)``.
    """

    def __init__(
        self, base_params: TRSParams, *, funding_rate_ref: Optional[float] = None
    ) -> None:
        if base_params.pricing is None:
            raise ValidationError("TRS pricing parameters must be provided")
        self._base = base_params
        self._engine = TotalReturnSwapEngine()
        self._funding_rate_ref = funding_rate_ref
        self._val_date = base_params.pricing.valuation_date
        if self._val_date not in base_params.asset.asset_prices.index:
            raise ValidationError(
                f"valuation date {self._val_date!r} is not an observed price date; "
                "spot revaluation requires the valuation date to be a path pivot"
            )

    # ------------------------------------------------------------------ #
    # Parameter cloning / pricing
    # ------------------------------------------------------------------ #
    def _clone(
        self, spot: Optional[float] = None, financing_rate: Optional[float] = None
    ) -> TRSParams:
        """Return a TRS parameter copy with the valuation-date spot and/or the
        financing rate overridden. Calendars/events/margin are shared (never
        mutated); only the price series and fixed-leg rate are replaced."""
        prices = self._base.asset.asset_prices.copy()
        if spot is not None:
            prices.loc[self._val_date] = spot
        new_asset = replace(self._base.asset, asset_prices=prices)

        fix_leg = self._base.fix_leg
        if financing_rate is not None:
            fix_leg = replace(self._base.fix_leg, rate=financing_rate)

        return replace(self._base, asset=new_asset, fix_leg=fix_leg)

    def _present_value(self, params: TRSParams) -> float:
        """Run the cashflow engine and read the valuation-date present value."""
        pricing: PricingParams = params.pricing  # type: ignore[assignment]
        params = replace(params, pricing=replace(pricing, output_mode="spot"))
        df = self._engine.price(params, precision=_PV_PRECISION)
        pv = df.iloc[-1]["present_value"]
        if isinstance(pv, str):
            raise MarketDataError(
                "TRS present_value is non-numeric at the valuation date; "
                "the valuation-date row produced no priced cashflow"
            )
        return float(pv)

    # ------------------------------------------------------------------ #
    # Public mark-to-market / greeks
    # ------------------------------------------------------------------ #
    def remaining_tenor(self) -> float:
        """Year fraction from the valuation date to contract maturity (ACT/365)."""
        end = max(self._base.fix_leg.end_date, self._base.float_leg.end_date)
        start = datetime.strptime(self._val_date, "%Y-%m-%d")
        finish = datetime.strptime(end, "%Y-%m-%d")
        return max((finish - start).days / 365.0, 0.0)

    def _financing_rate(self, pricing_env: Optional[PricingEnvironment]) -> Optional[float]:
        """Financing rate implied by the env, or ``None`` to keep contractual."""
        if self._funding_rate_ref is None or pricing_env is None:
            return None
        env_rate = pricing_env.get_rate(self.remaining_tenor())
        return self._base.fix_leg.rate + (env_rate - self._funding_rate_ref)

    def mark_to_market(
        self,
        pricing_env: Optional[PricingEnvironment] = None,
        *,
        spot: Optional[float] = None,
        financing_rate: Optional[float] = None,
    ) -> float:
        """Mark-to-market value (valuation-date present value) of one swap.

        Spot is taken from ``pricing_env`` unless ``spot`` is given explicitly;
        the financing rate follows the funding-flow-through rule unless overridden.
        """
        if spot is None and pricing_env is not None:
            spot = pricing_env.spot
        if financing_rate is None:
            financing_rate = self._financing_rate(pricing_env)
        return self._present_value(self._clone(spot=spot, financing_rate=financing_rate))

    def greeks(
        self,
        pricing_env: PricingEnvironment,
        *,
        spot_bump: float = 0.01,
        rate_bump: float = 1e-4,
    ) -> Dict[str, float]:
        """Per-contract greeks by central finite differences.

        ``spot_bump`` is a fractional shift of spot; ``rate_bump`` is an absolute
        shift of the financing rate. Returns the equity core-greek keys (vega and
        theta are zero — a TRS has no volatility dependence).
        """
        spot = pricing_env.spot
        fin = self._financing_rate(pricing_env)
        base = self._present_value(self._clone(spot=spot, financing_rate=fin))

        h = spot * spot_bump
        if h <= 0.0:
            raise ValidationError("spot must be positive to compute swap greeks")
        up = self._present_value(self._clone(spot=spot + h, financing_rate=fin))
        down = self._present_value(self._clone(spot=spot - h, financing_rate=fin))
        delta = (up - down) / (2.0 * h)
        gamma = (up - 2.0 * base + down) / (h * h)

        # Rho: bump the financing rate directly (available even when the env
        # funding flow-through is off, i.e. funding_rate_ref is None).
        fin_base = fin if fin is not None else self._base.fix_leg.rate
        rho_up = self._present_value(
            self._clone(spot=spot, financing_rate=fin_base + rate_bump)
        )
        rho_down = self._present_value(
            self._clone(spot=spot, financing_rate=fin_base - rate_bump)
        )
        rho = (rho_up - rho_down) / (2.0 * rate_bump)

        return {
            "price": base,
            "delta": delta,
            "gamma": gamma,
            "vega": 0.0,
            "theta": 0.0,
            "rho": rho,
        }
