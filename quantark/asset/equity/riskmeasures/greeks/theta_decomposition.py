"""Theta component decomposition (estimate + exact zeroed-r/q repricing).

Bodies moved verbatim from GreeksCalculator (R1c pure code motion).
"""

from copy import deepcopy
from typing import Dict, Optional

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.priceenv import PricingEnvironment
from quantark.util.numerical import is_zero


def estimate_theta_components(
    theta: float,
    rho: float,
    dividend_rho: float,
    r: float,
    q: float,
    T: float,
    rate_bump: float = 0.01,
    dividend_bump: float = 0.01,
) -> Dict[str, float]:
    """
    Fast estimation of theta components from existing Greeks.

    Uses the relationships between theta components and other Greeks:
        r_theta ≈ -r/T * rho / rate_bump (corrected for scale and daily conversion)
        q_theta ≈ -q/T * dividend_rho / dividend_bump (corrected for scale and daily conversion)
        convexity_theta ≈ theta - r_theta - q_theta

    This is an approximation that avoids repricing. For exact decomposition,
    use exact_theta_components() instead.

    Args:
        theta: Total theta (per day)
        rho: Rho (sensitivity to rate, per 1% change)
        dividend_rho: Dividend rho (sensitivity to dividend yield, per 1% change)
        r: Interest rate (annual)
        q: Dividend yield (annual)
        T: Time to maturity in years
        rate_bump: Rate scale of the rho input (default: 1% = 0.01)
        dividend_bump: Dividend scale of the dividend_rho input (default: 1% = 0.01)

    Returns:
        Dictionary with convexity_theta, r_theta, q_theta (all per day)
    """
    if is_zero(T):
        return {
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
        }

    # Rho/Dividend Rho are per rate_bump/dividend_bump size, so divide by scale.
    # Divide by 365 to convert annual rate decay to daily theta equivalent.
    # Divide by T to cancel out the T term in Rho (Rho = dV/dr = T * dV/d(rT) approx).
    r_theta = -r / T * (rho / rate_bump) / 365
    q_theta = -q / T * (dividend_rho / dividend_bump) / 365
    convexity_theta = theta - r_theta - q_theta

    return {
        "convexity_theta": convexity_theta,
        "r_theta": r_theta,
        "q_theta": q_theta,
    }


def exact_theta_components(
    calc,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    base_price: Optional[float] = None,
    time_bump_days: Optional[int] = None,
    time_bump_mode: Optional[str] = None,
    theta: Optional[float] = None,
) -> Dict[str, float]:
    """
    Exact numerical theta decomposition via repricing with zeroed r/q.

    Reprices theta under (r=0,q=0), (q=0), (r=0) and allocates the r/q
    interaction term symmetrically (Shapley attribution), so the three
    components sum to the total theta exactly:
        convexity_theta = theta(0,0)
        r_theta = 1/2 [ (theta(r,0)-theta(0,0)) + (theta(r,q)-theta(0,q)) ]
        q_theta = 1/2 [ (theta(0,q)-theta(0,0)) + (theta(r,q)-theta(r,0)) ]

    Note: This is computationally expensive (3 extra theta evaluations,
    plus the total theta when not supplied) and should be treated as a
    slow path. For fast estimation, use estimate_theta_components().

    Args:
        calc: GreeksCalculator facade instance
        product: The derivative product
        pricing_env: Pricing environment
        engine: Pricing engine
        base_price: Pre-calculated base price
        time_bump_days: Time bump in days
        time_bump_mode: Date-advance mode for the theta step
        theta: Total theta under the same step, if already computed

    Returns:
        Dictionary with convexity_theta, r_theta, q_theta (all per day)
    """
    from quantark.param.div import ContinuousDividendYield
    from quantark.param.rrf import FlatRateCurve

    time_bump_days = (
        time_bump_days
        if time_bump_days is not None
        else calc._bump_config.time_bump_days
    )

    T = product.get_maturity(pricing_env)
    if is_zero(T):
        return {
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
        }

    # Create environments with zeroed r and/or q
    env_no_r = deepcopy(pricing_env)
    env_no_r.rate_curve = FlatRateCurve(0.0)

    env_no_q = deepcopy(pricing_env)
    env_no_q.div_yield = ContinuousDividendYield(0.0)

    env_no_rq = deepcopy(pricing_env)
    env_no_rq.rate_curve = FlatRateCurve(0.0)
    env_no_rq.div_yield = ContinuousDividendYield(0.0)

    # Calculate theta in each environment
    theta_no_rq = calc.calculate_numerical_theta(
        product, env_no_rq, engine,
        time_bump_days=time_bump_days, time_bump_mode=time_bump_mode,
    )
    theta_no_q = calc.calculate_numerical_theta(
        product, env_no_q, engine,
        time_bump_days=time_bump_days, time_bump_mode=time_bump_mode,
    )
    theta_no_r = calc.calculate_numerical_theta(
        product, env_no_r, engine,
        time_bump_days=time_bump_days, time_bump_mode=time_bump_mode,
    )
    if theta is None:
        theta = calc.calculate_numerical_theta(
            product, pricing_env, engine,
            base_price=base_price,
            time_bump_days=time_bump_days, time_bump_mode=time_bump_mode,
        )

    # Symmetric (Shapley) allocation of the r/q interaction: the three
    # components reconcile to the total theta exactly.
    convexity_theta = theta_no_rq
    r_theta = 0.5 * ((theta_no_q - theta_no_rq) + (theta - theta_no_r))
    q_theta = 0.5 * ((theta_no_r - theta_no_rq) + (theta - theta_no_q))

    return {
        "convexity_theta": convexity_theta,
        "r_theta": r_theta,
        "q_theta": q_theta,
    }
