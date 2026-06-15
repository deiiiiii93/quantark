"""
Black-Scholes barrier building blocks for Vanna-Volga: Reiner-Rubinstein
one-touch / no-touch probabilities and single-barrier survival probabilities.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _mu_from_b(b: float, vol: float) -> float:
    return (b - 0.5 * vol * vol) / (vol * vol)


def one_touch_hit_prob(
    spot: float,
    barrier: float,
    vol: float,
    tau: float,
    b: float,
    is_up: bool,
) -> float:
    """Probability that a GBM with drift ``b`` hits the barrier before T.

    Reiner-Rubinstein expiry-pay cash-or-nothing one-touch probability.
    """
    if tau <= 0.0:
        return (
            1.0
            if (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
            else 0.0
        )
    if vol <= 0.0:
        # Deterministic drift path S(t) = spot * exp(b*t) is monotonic in t, so
        # its extreme over [0, tau] is at an endpoint. The barrier is touched
        # iff that extreme reaches it.
        s_T = spot * math.exp(b * tau)
        if is_up:
            hit = max(spot, s_T) >= barrier
        else:
            hit = min(spot, s_T) <= barrier
        return 1.0 if hit else 0.0

    mu = _mu_from_b(b, vol)
    sqrt_tau = math.sqrt(tau)
    log_H_over_S = math.log(barrier / spot)

    x2 = math.log(spot / barrier) / (vol * sqrt_tau) + (1.0 + mu) * vol * sqrt_tau
    y2 = math.log(barrier / spot) / (vol * sqrt_tau) + (1.0 + mu) * vol * sqrt_tau

    if is_up:
        phi = 1.0
        eta = -1.0
    else:
        phi = -1.0
        eta = 1.0

    term1 = norm.cdf(phi * x2 - phi * vol * sqrt_tau)
    # term2 = (H/S)^(2 mu) * N(...). For small vol with strong carry the power
    # overflows before the vanishing CDF can offset it, so evaluate the product
    # in log space (the CDF tail is handled stably by ``norm.logcdf``).
    log_term2 = 2.0 * mu * log_H_over_S + float(norm.logcdf(eta * y2 - eta * vol * sqrt_tau))
    term2 = math.exp(log_term2) if log_term2 < 700.0 else float("inf")
    p_hit = float(term1 + term2)
    return max(0.0, min(1.0, p_hit))


def no_touch_price(
    spot: float,
    barrier: float,
    vol: float,
    tau: float,
    r_discount: float,
    b_drift: float,
    is_up: bool,
) -> float:
    """Single-barrier no-touch paying 1 at expiry if the barrier is not hit."""
    df = math.exp(-r_discount * tau)
    p_hit = one_touch_hit_prob(spot, barrier, vol, tau, b_drift, is_up)
    return df * (1.0 - p_hit)


def survival_probability_single(
    spot: float,
    barrier: float,
    rd: float,
    rf: float,
    vol: float,
    tau: float,
    is_up: bool,
) -> float:
    """Single-barrier survival probability: average of domestic/foreign measures.

    p_surv^d = NT^d / DF_d with b = rd - rf
    p_surv^f = NT^f / DF_f with b = rd - rf + sigma^2
    """
    if tau <= 0.0:
        # At expiry there is no remaining time to hit: survival is 0 only if the
        # barrier is already breached, otherwise 1.
        breached = (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
        return 0.0 if breached else 1.0

    nt_d = no_touch_price(
        spot=spot, barrier=barrier, vol=vol, tau=tau,
        r_discount=rd, b_drift=rd - rf, is_up=is_up,
    )
    df_d = math.exp(-rd * tau)
    p_surv_d = nt_d / df_d

    nt_f = no_touch_price(
        spot=spot, barrier=barrier, vol=vol, tau=tau,
        r_discount=rf, b_drift=(rd - rf + vol * vol), is_up=is_up,
    )
    df_f = math.exp(-rf * tau)
    p_surv_f = nt_f / df_f

    return max(0.0, min(1.0, 0.5 * (p_surv_d + p_surv_f)))


__all__ = ["one_touch_hit_prob", "no_touch_price", "survival_probability_single"]
