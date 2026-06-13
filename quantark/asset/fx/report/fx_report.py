"""
Formatted reporting for FX options.
"""

from typing import Optional

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "CHF": "CHF",
    "CAD": "C$",
    "AUD": "A$",
    "NZD": "NZ$",
    "HKD": "HK$",
    "SGD": "S$",
    "KRW": "₩",
    "INR": "₹",
    "BRL": "R$",
    "MXN": "Mex$",
}

_RULE = "-" * 80
_DOUBLE_RULE = "=" * 80


def get_currency_symbol(currency_code: str) -> str:
    """
    Currency symbol for a currency code, falling back to the code itself.
    """
    code = currency_code.upper()
    return CURRENCY_SYMBOLS.get(code, code)


def format_fx_option_report(
    product: BaseFxProduct,
    fx_env: FxPricingEnvironment,
    engine: BaseFxEngine,
    include_greeks: bool = True,
    display_mode: str = "codes",
) -> str:
    """
    Render a formatted pricing report for an FX option.

    Args:
        product: FX option product (vanilla, digital, or quanto)
        fx_env: FX pricing environment
        engine: Pricing engine matching the product
        include_greeks: Include the Greeks section
        display_mode: "codes" (USD) or "symbols" ($) for currency display

    Returns:
        Multi-line report string

    Raises:
        ValueError: If display_mode is invalid
    """
    if display_mode not in ("codes", "symbols"):
        raise ValueError(
            f"display_mode must be 'codes' or 'symbols', got '{display_mode}'"
        )

    pair = product.currency_pair
    if display_mode == "symbols":
        dom_display = get_currency_symbol(pair.domestic)
        for_display = get_currency_symbol(pair.foreign)
    else:
        dom_display = pair.domestic
        for_display = pair.foreign

    greeks: Optional[dict] = None
    if include_greeks:
        greeks = engine.calculate_greeks(product, fx_env)
        value = greeks["price"]
    else:
        value = engine.price(product, fx_env)

    tau = product.get_maturity(fx_env)

    lines = [
        _DOUBLE_RULE,
        f"{type(product).__name__} - Pricing Results",
        _DOUBLE_RULE,
        "",
        f"Currency Pair: {pair}",
    ]

    option_type = getattr(product, "option_type", None)
    if option_type is not None:
        lines.append(f"Option Type: {option_type}")
    lines.append("")

    lines.extend(
        [
            "Market Parameters:",
            _RULE,
            f"  Spot Rate:              {fx_env.spot:,.6f}  ({for_display}/{dom_display})",
        ]
    )
    strike = getattr(product, "strike", None)
    if strike is not None:
        lines.append(
            f"  Strike:                 {strike:,.6f}  ({for_display}/{dom_display})"
        )
        lines.append(f"  Volatility:             {fx_env.get_vol(strike, tau):.4%}")
    lines.extend(
        [
            f"  Domestic Rate:          {fx_env.get_domestic_rate(tau):.4%}  ({pair.domestic})",
            f"  Foreign Rate:           {fx_env.get_foreign_rate(tau):.4%}  ({pair.foreign})",
            f"  Time to Expiry:         {tau:.4f} years",
            "",
        ]
    )

    notional = getattr(product, "notional", None)
    payout = getattr(product, "payout", None)
    if notional is not None or payout is not None:
        lines.extend(["Notional Information:", _RULE])
        if notional is not None:
            lines.append(
                f"  Notional (Foreign):     {notional:,.2f}  ({for_display})"
            )
        if payout is not None:
            lines.append(f"  Payout:                 {payout:,.2f}")
        lines.append("")

    lines.extend(
        [
            "Pricing Results:",
            _RULE,
            f"  Option Value:           {value:,.6f}  ({dom_display})",
            "",
        ]
    )

    if greeks is not None:
        lines.extend(
            [
                "Risk Measures (Greeks):",
                _RULE,
                f"  Delta:                  {greeks['delta']:,.6f}  ({for_display})",
                f"  Forward Delta:          {greeks['fwd_delta']:,.6f}  ({for_display})",
                f"  Gamma:                  {greeks['gamma']:,.6f}",
                f"  Vega (per 1% vol):      {greeks['vega']:,.6f}  ({dom_display})",
                f"  Theta (daily):          {greeks['theta']:,.6f}  ({dom_display})",
                f"  Rho Domestic:           {greeks['rho_dom']:,.6f}  ({dom_display})",
                f"  Rho Foreign:            {greeks['rho_for']:,.6f}  ({for_display})",
                "",
            ]
        )

    lines.append(_DOUBLE_RULE)
    return "\n".join(lines)
