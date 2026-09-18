"""Exact-GBM autocallable oracle. No QuantArk or reference-pricer imports.

Each confidence interval is across independent scrambled Sobol replicates,
using Student t. Monitoring is only at the specified dates, except for the
explicit continuous-KI case, which uses exact conditional bridge hit draws.
All prices are per 100 initial notional, excluding principal.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm, qmc, t as student_t


@dataclass
class Case:
    name: str
    product: str = "snowball"
    spot: float = 100.0
    strike: float = 100.0
    maturity: float = 1.0
    rate: float = 0.03
    div: float = 0.01
    vol: float = 0.20
    coupon: float = 0.12
    ko_times: list = field(default_factory=lambda: [i / 12 for i in range(1, 13)])
    ko_levels: list = field(default_factory=lambda: [103.0] * 12)
    ki_times: list = field(default_factory=lambda: [i / 12 for i in range(1, 13)])
    ki_level: float = 75.0
    coupon_level: float = 85.0
    already_ki: bool = False
    loss_cap: float | None = None
    memory: bool = False
    continuous_ki: bool = False
    term: bool = False
    reverse: bool = False


TERM_TIMES = np.array([0.25, 0.5, 1.0, 2.0])
TERM_RATES = np.array([0.02, 0.035, 0.025, 0.03])
TERM_DIVS = np.array([-0.015, 0.02, -0.005, 0.01])
TERM_VOLS = np.array([0.22, 0.18, 0.24, 0.20])


def cumulants(c, times):
    times = np.asarray(times, dtype=float)
    if not c.term:
        return c.rate * times, c.div * times, c.vol**2 * times
    r = np.interp(times, TERM_TIMES, TERM_RATES) * times
    q = np.interp(times, TERM_TIMES, TERM_DIVS) * times
    w = np.interp(times, TERM_TIMES, TERM_VOLS**2 * TERM_TIMES)
    w = np.where(times <= TERM_TIMES[0], TERM_VOLS[0]**2 * times, w)
    w = np.where(times >= TERM_TIMES[-1], TERM_VOLS[-1]**2 * times, w)
    return r, q, w


def vanilla_put(c):
    r, q, w = (float(x) for x in cumulants(c, c.maturity))
    fwd = c.spot * math.exp(r - q)
    if w == 0:
        return math.exp(-r) * max(c.strike - fwd, 0.0)
    d1 = (math.log(fwd / c.strike) + w / 2) / math.sqrt(w)
    d2 = d1 - math.sqrt(w)
    return math.exp(-r) * (c.strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def coupon_only_exact(c):
    r, q, w = cumulants(c, c.ko_times)
    d2 = (np.log(c.spot / c.coupon_level) + r - q - w / 2) / np.sqrt(w)
    amounts = 100 * c.coupon * np.diff([0.0] + c.ko_times)
    return float(np.sum(amounts * np.exp(-r) * norm.cdf(d2)))


def estimate(c, power=17, replicates=8, seed=20260911):
    times = np.unique(c.ko_times + ([] if c.continuous_ki else c.ki_times) + [c.maturity])
    r, q, w = cumulants(c, np.r_[0.0, times])
    dr, dq, dw = np.diff(r), np.diff(q), np.diff(w)
    if np.any(dw < -1e-14):
        raise ValueError("Negative forward variance in independent MC input")
    ko_map = {round(x, 12): i for i, x in enumerate(c.ko_times)}
    ki_set = {round(x, 12) for x in c.ki_times}
    cp_amounts = 100 * c.coupon * np.diff([0.0] + c.ko_times)
    dim = len(times) * (2 if c.continuous_ki else 1)
    estimates = []
    forward_estimates = []
    put_estimates = []
    replicate_components = []
    npaths = 2**power
    block = min(npaths, 2**14)
    for rep in range(replicates):
        sobol = qmc.Sobol(d=dim, scramble=True, seed=seed + 1009 * rep)
        sums = np.zeros(5)
        for start in range(0, npaths, block):
            size = min(block, npaths - start)
            u = sobol.random(size)
            z = ndtri(np.clip(u[:, :len(times)], 1e-15, 1 - 1e-15))
            spot = np.full(size, c.spot)
            alive = np.ones(size, dtype=bool)
            initial_hit = c.spot >= c.ki_level if c.reverse else c.spot <= c.ki_level
            knocked = np.full(size, c.already_ki or (c.continuous_ki and initial_hit))
            ko_value = np.zeros(size)
            coupon_value = np.zeros(size)
            missed = np.zeros(size)
            for j, obs in enumerate(times):
                old_spot = spot
                spot = spot * np.exp(dr[j] - dq[j] - dw[j] / 2 + math.sqrt(max(dw[j], 0)) * z[:, j])
                if c.continuous_ki:
                    hits = ((old_spot >= c.ki_level) | (spot >= c.ki_level)) if c.reverse else ((old_spot <= c.ki_level) | (spot <= c.ki_level))
                    if dw[j] > 0:
                        p = np.exp(np.minimum(0, -2 * np.log(old_spot / c.ki_level) * np.log(spot / c.ki_level) / dw[j]))
                        hits |= u[:, len(times) + j] < p
                    knocked |= hits
                elif round(float(obs), 12) in ki_set:
                    knocked |= (spot >= c.ki_level if c.reverse else spot <= c.ki_level)
                k = ko_map.get(round(float(obs), 12))
                if k is not None:
                    if c.product == "phoenix":
                        entitled = alive & (spot <= c.coupon_level if c.reverse else spot >= c.coupon_level)
                        amount = cp_amounts[k] + (missed if c.memory else 0)
                        coupon_value += entitled * amount * math.exp(-r[j + 1])
                        if c.memory:
                            missed = np.where(entitled, 0, missed + cp_amounts[k])
                    hit_ko = alive & (spot <= c.ko_levels[k] if c.reverse else spot >= c.ko_levels[k])
                    if c.product == "snowball":
                        ko_value += hit_ko * (100 * c.coupon * obs) * math.exp(-r[j + 1])
                    alive &= ~hit_ko
            loss = np.maximum(spot - c.strike if c.reverse else c.strike - spot, 0.0)
            if c.loss_cap is not None:
                loss = np.minimum(loss, c.loss_cap)
            downside = -loss * alive * knocked * math.exp(-r[-1])
            if c.product == "snowball":
                coupon_value += alive * ~knocked * (100 * c.coupon * c.maturity) * math.exp(-r[-1])
            sums += [np.sum(ko_value), np.sum(coupon_value), np.sum(downside), np.sum(spot), np.sum(np.maximum(c.strike - spot, 0) * math.exp(-r[-1]))]
        sums /= npaths
        estimates.append(float(sum(sums[:3])))
        replicate_components.append(sums[:3].tolist())
        forward_estimates.append(float(sums[3]))
        put_estimates.append(float(sums[4]))

    def summary(values):
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(replicates))
        half = float(student_t.ppf(0.975, replicates - 1) * se)
        return {"mean": mean, "se": se, "half95": half, "lo95": mean - half, "hi95": mean + half}

    return {**summary(estimates), "replicate_values": estimates, "replicate_components": replicate_components,
            "paths": npaths * replicates, "replicates": replicates, "seed": seed,
            "vanilla_put": {**summary(put_estimates), "exact": vanilla_put(c)},
            "terminal_forward": {**summary(forward_estimates), "exact": c.spot * math.exp(r[-1] - q[-1])}}
