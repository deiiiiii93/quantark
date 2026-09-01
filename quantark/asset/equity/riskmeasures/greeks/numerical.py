"""Bump-based scalar greeks.

Bodies moved verbatim from GreeksCalculator (R2); each function takes the
facade instance ``calc`` for bump config and engine-mode resolution. The
facade's public ``calculate_numerical_*`` methods delegate here.
"""

from copy import deepcopy
from typing import Dict, Optional, Tuple

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.asset.equity.riskmeasures.greeks.registry import DEFAULT_SET, REGISTRY
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def numerical_delta(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    spot_prices: Optional[Tuple[float, float]] = None,
    bump: Optional[float] = None,
) -> float:
    """Numerical delta using central spot bump."""
    bump = bump if bump is not None else calc._bump_config.spot_bump
    base_price, price_up_spot, price_down_spot = bump_envs.spot_bumped_prices(
        product,
        pricing_env,
        engine,
        bump,
        base_price=base_price,
        reuse=spot_prices,
    )
    return bump_envs.calculate_sensitivity(
        base_price,
        price_up_spot,
        price_down_spot,
        bump=bump,
        scale=pricing_env.spot,
        mode="central",
    )


def numerical_gamma(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    spot_prices: Optional[Tuple[float, float]] = None,
    bump: Optional[float] = None,
) -> float:
    """Numerical gamma using central spot bump."""
    bump = bump if bump is not None else calc._bump_config.spot_bump
    base_price, price_up_spot, price_down_spot = bump_envs.spot_bumped_prices(
        product,
        pricing_env,
        engine,
        bump,
        base_price=base_price,
        reuse=spot_prices,
    )
    return bump_envs.calculate_sensitivity(
        base_price,
        price_up_spot,
        price_down_spot,
        bump=bump,
        scale=pricing_env.spot,
        mode="second_order",
    )


def numerical_vega(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    vol_bump: Optional[float] = None,
) -> float:
    """Numerical vega from a vol bump."""
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    vol_bump = vol_bump if vol_bump is not None else calc._bump_config.vol_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", pricing_env.spot)
    current_vol = pricing_env.get_vol(strike, T)
    env_up_vol = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=1.0
    )
    price_up_vol = engine.price(product, env_up_vol)
    return bump_envs.calculate_sensitivity(
        base_price, price_up_vol, bump=vol_bump, mode="one_sided"
    )


def numerical_volga(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    vol_bump: Optional[float] = None,
) -> float:
    """Numerical volga (second derivative wrt vol) using vol bumps."""
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    vol_bump = vol_bump if vol_bump is not None else calc._bump_config.vol_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", pricing_env.spot)
    current_vol = pricing_env.get_vol(strike, T)

    if current_vol - vol_bump <= 0:
        env_up = bump_envs.build_vol_bumped_env(
            pricing_env, product, current_vol, vol_bump, direction=1.0
        )
        vega_base = numerical_vega(
            calc, product, pricing_env, engine, base_price=base_price, vol_bump=vol_bump
        )
        vega_up = numerical_vega(
            calc, product, env_up, engine, base_price=None, vol_bump=vol_bump
        )
        return (vega_up - vega_base) / vol_bump

    env_up = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=1.0
    )
    env_down = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=-1.0
    )
    price_up = engine.price(product, env_up)
    price_down = engine.price(product, env_down)
    return bump_envs.calculate_sensitivity(
        base_price, price_up, price_down, bump=vol_bump, mode="second_order"
    )


def numerical_vanna(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    vol_bump: Optional[float] = None,
) -> float:
    """Numerical vanna (cross derivative wrt spot and vol)."""
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    vol_bump = vol_bump if vol_bump is not None else calc._bump_config.vol_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", pricing_env.spot)
    current_vol = pricing_env.get_vol(strike, T)

    env_up = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=1.0
    )
    env_down = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=-1.0
    )

    if current_vol - vol_bump <= 0:
        base_delta = numerical_delta(
            calc,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            bump=calc._bump_config.spot_bump,
        )
        delta_up = numerical_delta(
            calc,
            product,
            env_up,
            engine,
            base_price=base_price,
            bump=calc._bump_config.spot_bump,
        )
        return (delta_up - base_delta) / vol_bump

    delta_up = numerical_delta(
        calc,
        product,
        env_up,
        engine,
        base_price=base_price,
        bump=calc._bump_config.spot_bump,
    )
    delta_down = numerical_delta(
        calc,
        product,
        env_down,
        engine,
        base_price=base_price,
        bump=calc._bump_config.spot_bump,
    )
    return (delta_up - delta_down) / (2.0 * vol_bump)


def numerical_theta(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    time_bump_days: Optional[int] = None,
    time_bump_mode: Optional[str] = None,
) -> float:
    """
    Numerical theta via time bump with observation schedule handling.

    Theta date advancement is controlled by BumpConfig.time_bump_mode:
    "calendar_days" preserves legacy calendar-date bumps, "business_days"
    advances by valid pricing-calendar business days, and "auto" uses
    business days for BUSINESS_DAYS pricing environments with a calendar.
    """
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    time_bump_days = (
        time_bump_days
        if time_bump_days is not None
        else calc._bump_config.time_bump_days
    )
    time_bump_mode = (
        time_bump_mode
        if time_bump_mode is not None
        else getattr(calc._bump_config, "time_bump_mode", "auto")
    )
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    product_theta = deepcopy(product)
    env_theta = deepcopy(pricing_env)
    current_maturity = product.get_maturity(pricing_env)

    bumped_date, time_bump, resolved_mode = bump_envs.advance_theta_bump(
        pricing_env, time_bump_days, time_bump_mode
    )

    if time_bump <= 0.0:
        if current_maturity <= 0.0:
            return 0.0
        if resolved_mode == "business_days":
            raise ValidationError(
                "Business-day theta bump did not advance time: "
                f"valuation_date={pricing_env.valuation_date}, "
                f"bumped_date={bumped_date}, time_bump_days={time_bump_days}"
            )
        return 0.0
    if current_maturity <= time_bump:
        return 0.0

    env_theta.valuation_date = bumped_date
    dropped_all_observations = product_theta.time_shift(
        time_bump, bumped_date, env_theta
    )

    if dropped_all_observations:
        return 0.0

    price_theta = engine.price(product_theta, env_theta)
    return price_theta - base_price


def numerical_rho(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    rate_bump: Optional[float] = None,
) -> float:
    """Numerical rho from a rate bump (per 1% rate change)."""
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    rate_bump = rate_bump if rate_bump is not None else calc._bump_config.rate_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    env_up_rate = deepcopy(pricing_env)
    from quantark.param.rrf import FlatRateCurve

    T = product.get_maturity(pricing_env)
    current_rate = pricing_env.get_rate(T)
    env_up_rate.rate_curve = FlatRateCurve(current_rate + rate_bump)
    price_up_rate = engine.price(product, env_up_rate)
    raw = bump_envs.calculate_sensitivity(
        base_price, price_up_rate, bump=rate_bump, mode="one_sided"
    )
    return raw * (0.01 / rate_bump)


def numerical_dividend_rho(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    div_bump: Optional[float] = None,
) -> float:
    """
    Numerical dividend_rho (psi) from dividend yield bump.

    Measures price sensitivity to dividend yield changes:
        dividend_rho = dV/dq

    Returns:
        Dividend rho value (price change per 1% div_yield change).
        Negative for call options (higher div = lower call price).
        Positive for put options (higher div = higher put price).
    """
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    div_bump = div_bump if div_bump is not None else calc._bump_config.div_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    current_div = pricing_env.get_div_yield(T)
    env_up_div = bump_envs.build_div_bumped_env(
        pricing_env, product, current_div, div_bump, direction=1.0
    )
    price_up_div = engine.price(product, env_up_div)
    raw = bump_envs.calculate_sensitivity(
        base_price, price_up_div, bump=div_bump, mode="one_sided"
    )
    return raw * (0.01 / div_bump)


def numerical_delta_q(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    div_bump: Optional[float] = None,
    base_delta: Optional[float] = None,
) -> float:
    """Numerical dDelta/dq via dividend yield bumps."""
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    div_bump = div_bump if div_bump is not None else calc._bump_config.div_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    current_div = pricing_env.get_div_yield(T)

    if base_delta is None:
        base_delta = numerical_delta(
            calc,
            product,
            pricing_env,
            engine,
            base_price=base_price,
            bump=calc._bump_config.spot_bump,
        )

    env_up = bump_envs.build_div_bumped_env(
        pricing_env, product, current_div, div_bump, direction=1.0
    )
    env_down = bump_envs.build_div_bumped_env(
        pricing_env, product, current_div, div_bump, direction=-1.0
    )
    delta_up = numerical_delta(
        calc,
        product,
        env_up,
        engine,
        base_price=base_price,
        bump=calc._bump_config.spot_bump,
    )
    delta_down = numerical_delta(
        calc,
        product,
        env_down,
        engine,
        base_price=base_price,
        bump=calc._bump_config.spot_bump,
    )
    return (delta_up - delta_down) / (2.0 * div_bump)


def numerical_speed(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    bump: Optional[float] = None,
) -> float:
    """Numerical speed (d3V/dS3) via a 4-point stencil on the spot axis.

    Uses single-level relative bumps only: the V(S(1±h)) legs are the same
    prices delta/gamma use, and V(S(1±2h)) adds two pricings. No nested
    bumped-env chains, so no compounded-bump ambiguity.
    """
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    bump = bump if bump is not None else calc._bump_config.spot_bump
    base_price, price_up, price_down = bump_envs.spot_bumped_prices(
        product, pricing_env, engine, bump, base_price=base_price
    )

    env_up2 = deepcopy(pricing_env)
    env_up2.spot_quote.spot *= 1 + 2.0 * bump
    price_up2 = engine.price(product, env_up2)

    env_down2 = deepcopy(pricing_env)
    env_down2.spot_quote.spot *= 1 - 2.0 * bump
    price_down2 = engine.price(product, env_down2)

    h = pricing_env.spot * bump
    return (
        price_up2 - 2.0 * price_up + 2.0 * price_down - price_down2
    ) / (2.0 * h**3)


def numerical_zomma(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    vol_bump: Optional[float] = None,
) -> float:
    """Numerical zomma (dGamma/dsigma) via gamma at vol-bumped envs.

    Inner gammas go through get_delta_gamma, so greeks_mode ENGINE/AUTO
    grid readout is honored. Falls back to a one-sided-up difference when
    sigma - vol_bump would be non-positive (same guard as vanna/volga).
    """
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    vol_bump = vol_bump if vol_bump is not None else calc._bump_config.vol_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", pricing_env.spot)
    current_vol = pricing_env.get_vol(strike, T)

    env_up = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=1.0
    )

    if current_vol - vol_bump <= 0:
        _, _, gamma_base = get_delta_gamma(
            calc, product, pricing_env, engine, base_price
        )
        _, _, gamma_up = get_delta_gamma(calc, product, env_up, engine, None)
        return (gamma_up - gamma_base) / vol_bump

    env_down = bump_envs.build_vol_bumped_env(
        pricing_env, product, current_vol, vol_bump, direction=-1.0
    )
    _, _, gamma_up = get_delta_gamma(calc, product, env_up, engine, None)
    _, _, gamma_down = get_delta_gamma(calc, product, env_down, engine, None)
    return (gamma_up - gamma_down) / (2.0 * vol_bump)


def numerical_dividend_volga(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    div_bump: Optional[float] = None,
) -> float:
    """Numerical dividend volga (d2V/dq2) via central dividend-yield bumps.

    Second-order convexity in the carry input; central second difference
    like volga. No positivity guard: a negative dividend yield is a
    legitimate carry input, unlike volatility.
    """
    engine = bump_envs.resolve_bump_engine(product, pricing_env, engine)
    div_bump = div_bump if div_bump is not None else calc._bump_config.div_bump
    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    T = product.get_maturity(pricing_env)
    current_div = pricing_env.get_div_yield(T)
    env_up = bump_envs.build_div_bumped_env(
        pricing_env, product, current_div, div_bump, direction=1.0
    )
    env_down = bump_envs.build_div_bumped_env(
        pricing_env, product, current_div, div_bump, direction=-1.0
    )
    price_up = engine.price(product, env_up)
    price_down = engine.price(product, env_down)
    return bump_envs.calculate_sensitivity(
        base_price, price_up, price_down, bump=div_bump, mode="second_order"
    )


def get_delta_gamma(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float],
) -> Tuple[float, float, float]:
    """Get base price, delta, and gamma via engine or bump method."""
    if calc._should_use_engine_greeks(engine):
        engine_greeks = engine.calculate_greeks(product, pricing_env)
        if base_price is None:
            base_price = engine_greeks["price"]
        return base_price, engine_greeks["delta"], engine_greeks["gamma"]

    base_price = bump_envs.ensure_base_price(product, pricing_env, engine, base_price)
    spot_prices = bump_envs.spot_bumped_prices(
        product, pricing_env, engine, calc._bump_config.spot_bump, base_price=base_price
    )[1:]

    delta = numerical_delta(
        calc,
        product,
        pricing_env,
        engine,
        base_price=base_price,
        spot_prices=spot_prices,
        bump=calc._bump_config.spot_bump,
    )
    gamma = numerical_gamma(
        calc,
        product,
        pricing_env,
        engine,
        base_price=base_price,
        spot_prices=spot_prices,
        bump=calc._bump_config.spot_bump,
    )

    return base_price, delta, gamma


def linear_greeks(product: BaseEquityProduct, price: float) -> Dict[str, float]:
    """
    Greeks for linear (delta-one) products.

    Delta-one products have trivial Greeks: delta = 1.0, everything else
    0.0 (no optionality). Values derive from the registry's linear_value so
    the default key set cannot drift from the request surface.
    """
    greeks = {"price": price}
    for name in DEFAULT_SET:
        if name == "price":
            continue
        greeks[name] = REGISTRY[name].linear_value
    return greeks
