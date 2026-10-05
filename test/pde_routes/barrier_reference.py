"""Exact reference for a discretely monitored single-barrier option under flat GBM (no PDE code).

A backward recursion over the observation dates. The alive region at a date is integrated with
composite Gauss-Legendre whose panels end on the barrier and on the strike, so every integrand is
smooth on a panel; the breach mass is the normal tail in closed form. The valuation-date step is
differentiated analytically, so delta and gamma are pointwise. Panel width is a fraction of the
shortest interval's sigma sqrt(dt); the kernel is applied in row blocks over the columns within
ten standard deviations (the rest is below 1e-22).

Knock-in = vanilla - knock-out(rebate 0) + rebate * DF(0,T) * P(never breached)   (rebate at expiry
when the option never knocks in: the kernel's and the product's convention).

Banked from the 3C evidence `l_ref.py` and `l_ref_term.py` (spec 3C §7.2); not production code.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _npdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _nodes(lo, hi, breaks, width, order):
    edges = [lo] + sorted(b for b in breaks if lo < b < hi) + [hi]
    gx, gw = np.polynomial.legendre.leggauss(order)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        panels = max(1, int(np.ceil((b - a) / width)))
        e = np.linspace(a, b, panels + 1)
        half, mid = 0.5 * np.diff(e), 0.5 * (e[:-1] + e[1:])
        xs.append((mid[:, None] + half[:, None] * gx[None, :]).ravel())
        ws.append((half[:, None] * gw[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


def _bs(call, spot, strike, T, vol, r, q):
    sd = vol * np.sqrt(T)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * vol * vol) * T) / sd
    d2 = d1 - sd
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    gamma = dq * _npdf(d1) / (spot * sd)
    if call:
        return spot * dq * ndtr(d1) - strike * dr * ndtr(d2), dq * ndtr(d1), gamma
    return strike * dr * ndtr(-d2) - spot * dq * ndtr(-d1), -dq * ndtr(-d1), gamma


class _Kernel:
    """The Gaussian transition over ``dt`` on the nodes, in row blocks limited to +-10 sd."""

    def __init__(self, x, w, mu, vol, dt, block=256):
        self.sd = vol * np.sqrt(dt)
        self.mean = x + mu * dt
        self.blocks = []
        for a in range(0, len(x), block):
            b = min(a + block, len(x))
            j0 = int(np.searchsorted(x, self.mean[a] - 10.0 * self.sd))
            j1 = int(np.searchsorted(x, self.mean[b - 1] + 10.0 * self.sd))
            z = (x[None, j0:j1] - self.mean[a:b, None]) / self.sd
            self.blocks.append((a, b, j0, j1, _npdf(z) * (w[None, j0:j1] / self.sd)))

    def apply(self, v):
        out = np.empty_like(v)
        for a, b, j0, j1, k in self.blocks:
            out[a:b] = k @ v[j0:j1]
        return out


def knock_out(*, payoff, strike, lower, upper, cash, at_hit, obs, spots, vol, r, q,
              width_sd=0.4, order=12, pad_sd=10.0):
    """{spot: (pv, delta, gamma)} of the knock-out monitored on ``obs`` (the last date is maturity).

    payoff: 'call' | 'put' | 'one' | 'zero' paid to a survivor at maturity; ``cash`` is paid on a
    breach, at the breach date (``at_hit``) or at maturity.
    """
    obs = np.asarray(obs, dtype=float)
    T = float(obs[-1])
    steps = np.diff(np.concatenate([[0.0], obs]))
    assert np.all(steps > 0.0)
    mu = r - q - 0.5 * vol * vol
    anchors = [np.log(float(s)) for s in spots] + ([np.log(strike)] if payoff in ("call", "put") else [])
    span = pad_sd * vol * np.sqrt(T) + abs(mu) * T
    lo = np.log(lower) if lower else min(anchors) - span
    hi = np.log(upper) if upper else max(anchors) + span
    width = width_sd * vol * np.sqrt(float(steps.min()))
    x, w = _nodes(lo, hi, anchors, min(width, 0.0125), order)
    s = np.exp(x)
    v = {"call": np.maximum(s - strike, 0.0), "put": np.maximum(strike - s, 0.0),
         "one": np.ones_like(s), "zero": np.zeros_like(s)}[payoff]

    def cash_at(k):
        return cash if at_hit else cash * np.exp(-r * (T - obs[k]))

    def breach(mean, sd, order_=0):
        out = 0.0
        if upper:
            d = (mean - hi) / sd
            out = out + (ndtr(d), _npdf(d) / sd, -d * _npdf(d) / sd ** 2)[order_]
        if lower:
            d = (lo - mean) / sd
            out = out + (ndtr(d), -_npdf(d) / sd, -d * _npdf(d) / sd ** 2)[order_]
        return out

    kernels = {}
    for k in range(len(obs) - 1, 0, -1):            # from observation k back to observation k-1
        dt = float(obs[k] - obs[k - 1])
        key = round(dt, 12)
        if key not in kernels:
            kernels[key] = _Kernel(x, w, mu, vol, dt)
        kern = kernels[key]
        v = np.exp(-r * dt) * (kern.apply(v) + breach(kern.mean, kern.sd) * cash_at(k))

    out = {}
    t1 = float(obs[0])
    sd = vol * np.sqrt(t1)
    df = np.exp(-r * t1)
    c1 = cash_at(0)
    for spot in spots:
        m = np.log(float(spot)) + mu * t1
        z = (x - m) / sd
        base = _npdf(z) / sd * w
        pv = df * (base @ v + breach(m, sd) * c1)
        vx = df * ((base * z / sd) @ v + breach(m, sd, 1) * c1)
        vxx = df * ((base * (z * z - 1.0) / sd ** 2) @ v + breach(m, sd, 2) * c1)
        out[float(spot)] = (float(pv), float(vx / spot), float((vxx - vx) / spot ** 2))
    return out, len(x)


def barrier_reference(*, kind, call, strike, level, rebate, at_hit, obs, spots, vol, r, q, **kw):
    """{spot: (pv, delta, gamma)} of UO / DO / UI / DI on ``obs``; participation 1."""
    up = kind[0] == "U"
    edges = dict(lower=None if up else level, upper=level if up else None)
    common = dict(strike=strike, obs=obs, spots=spots, vol=vol, r=r, q=q, **edges, **kw)
    payoff = "call" if call else "put"
    if kind[1] == "O":
        return knock_out(payoff=payoff, cash=rebate, at_hit=at_hit, **common)[0]
    ko, _ = knock_out(payoff=payoff, cash=0.0, at_hit=False, **common)
    T = float(obs[-1])
    out = {}
    surv = knock_out(payoff="one", cash=0.0, at_hit=False, **common)[0] if rebate else None
    for spot in spots:
        van = _bs(call, float(spot), strike, T, vol, r, q)
        vals = [van[i] - ko[float(spot)][i] for i in range(3)]
        if rebate:
            vals = [vals[i] + rebate * surv[float(spot)][i] for i in range(3)]   # survival value is discounted to T
        out[float(spot)] = tuple(vals)
    return out


# --- term curves (l_ref_term.py) ---------------------------------------------------------------
#The exact discrete-barrier recursion of l_ref.py under DETERMINISTIC TERM CURVES (review 1, P2).
#
#Same quadrature (panels ending on the barrier and the strike, breach mass in closed form). The
#market enters through two functions of time only:
#  df(t)      discount factor to t
#  growth(t)  forward growth factor, F(t) = S0 * growth(t)   (exp of the integrated r - q)
#so each interval [t_{k-1}, t_k] has its own log-drift ln(growth(t_k)/growth(t_{k-1})) - sigma^2 dt/2
#and its own discount df(t_k)/df(t_{k-1}). Cash on a breach at observation k is paid at `pay_time(k)`
#(the observation itself for pay-at-hit, otherwise the terminal payment time) and the survivor's
#payoff at `terminal_pay` (>= the last observation): delayed payments are discount-factor ratios.
#Constant vol. With flat curves it reproduces l_ref.knock_out (self-test in x6).

def knock_out_term(*, payoff, strike, lower, upper, cash, at_hit, obs, spots, vol, df, growth, terminal_pay=None,
                   width_sd=0.4, order=12, pad_sd=10.0):
    """{spot: (pv, delta, gamma)}; payoff in 'call' | 'put' | 'one' | 'zero'."""
    obs = np.asarray(obs, dtype=float)
    T = float(obs[-1])
    t_pay = T if terminal_pay is None else float(terminal_pay)
    edges_t = np.concatenate([[0.0], obs])
    steps = np.diff(edges_t)
    assert np.all(steps > 0.0)
    drift = np.array([np.log(growth(b) / growth(a)) - 0.5 * vol * vol * (b - a) for a, b in zip(edges_t[:-1], edges_t[1:])])
    disc = np.array([df(b) / df(a) for a, b in zip(edges_t[:-1], edges_t[1:])])
    anchors = [np.log(float(s)) for s in spots] + ([np.log(strike)] if payoff in ("call", "put") else [])
    span = pad_sd * vol * np.sqrt(T) + float(np.abs(drift).sum())
    lo = np.log(lower) if lower else min(anchors) - span
    hi = np.log(upper) if upper else max(anchors) + span
    width = width_sd * vol * np.sqrt(float(steps.min()))
    x, w = _nodes(lo, hi, anchors, min(width, 0.0125), order)
    s = np.exp(x)
    settle = df(t_pay) / df(T)                               # the survivor is paid at the terminal payment time
    v = {"call": np.maximum(s - strike, 0.0), "put": np.maximum(strike - s, 0.0),
         "one": np.ones_like(s), "zero": np.zeros_like(s)}[payoff] * settle

    def cash_at(k):                                          # value at observation k of the breach cash
        return cash if at_hit else cash * df(t_pay) / df(float(obs[k]))

    def breach(mean, sd, order_=0):
        out = 0.0
        if upper:
            d = (mean - hi) / sd
            out = out + (ndtr(d), _npdf(d) / sd, -d * _npdf(d) / sd ** 2)[order_]
        if lower:
            d = (lo - mean) / sd
            out = out + (ndtr(d), -_npdf(d) / sd, -d * _npdf(d) / sd ** 2)[order_]
        return out

    for k in range(len(obs) - 1, 0, -1):                     # from observation k back to observation k-1
        dt = float(steps[k])
        kern = _Kernel(x, w, float(drift[k]) / dt, vol, dt)   # one kernel per interval: its own drift
        v = disc[k] * (kern.apply(v) + breach(kern.mean, kern.sd) * cash_at(k))

    out = {}
    t1 = float(obs[0])
    sd = vol * np.sqrt(t1)
    c1 = cash_at(0)
    for spot in spots:
        m = np.log(float(spot)) + float(drift[0])
        z = (x - m) / sd
        base = _npdf(z) / sd * w
        pv = disc[0] * (base @ v + breach(m, sd) * c1)
        vx = disc[0] * ((base * z / sd) @ v + breach(m, sd, 1) * c1)
        vxx = disc[0] * ((base * (z * z - 1.0) / sd ** 2) @ v + breach(m, sd, 2) * c1)
        out[float(spot)] = (float(pv), float(vx / spot), float((vxx - vx) / spot ** 2))
    return out, len(x)


def black_scholes_term(call, spot, strike, T, vol, df, growth, terminal_pay=None):
    """European value with integrated carry: forward S0 growth(T), paid at `terminal_pay`."""
    fwd = spot * growth(T)
    sd = vol * np.sqrt(T)
    d1 = (np.log(fwd / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    pay = df(T if terminal_pay is None else terminal_pay)
    return float(pay * (fwd * ndtr(d1) - strike * ndtr(d2))) if call else float(pay * (strike * ndtr(-d2) - fwd * ndtr(-d1)))


def barrier_reference_term(*, kind, call, strike, level, rebate, at_hit, obs, spots, vol, df, growth):
    """{spot: pv} of UO / DO / UI / DI; a knock-in's rebate is paid at maturity when never knocked in."""
    up = kind[0] == "U"
    edges = dict(lower=None if up else level, upper=level if up else None)
    common = dict(strike=strike, obs=obs, spots=spots, vol=vol, df=df, growth=growth, **edges)
    payoff = "call" if call else "put"
    if kind[1] == "O":
        return {s: v[0] for s, v in knock_out_term(payoff=payoff, cash=rebate, at_hit=at_hit, **common)[0].items()}
    ko = knock_out_term(payoff=payoff, cash=0.0, at_hit=False, **common)[0]
    surv = knock_out_term(payoff="one", cash=0.0, at_hit=False, **common)[0] if rebate else None
    T = float(obs[-1])
    return {float(s): black_scholes_term(call, float(s), strike, T, vol, df, growth) - ko[float(s)][0]
            + (rebate * surv[float(s)][0] if rebate else 0.0) for s in spots}
