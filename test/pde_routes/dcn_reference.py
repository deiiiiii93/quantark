"""Exact DCN value under flat GBM: a two-state (never knocked in / knocked in) recursion over the
contract's DAILY grid. No engine code: the dates, accruals and payment times are the product's
own date arithmetic (`build_dcn_grid_context`), the event rule is the contract's forward statement
(`engine/mc/dcn_payoff.py`: at a date, knock-in first, then knock-out, then the coupon).

Between two dates the transition is the Gaussian of the log-return, integrated by composite
Gauss-Legendre on panels that end on the knock-in barrier, the coupon barrier, the knock-out
barrier and the loss strike, so every integrand is smooth on a panel. The kernel of each distinct
calendar gap is a sparse matrix cut at `kcut` standard deviations. The valuation-date value, delta
and gamma are read by integrating the first date's surface against the kernel centred on the spot
and its two analytic derivatives, so no node has to sit on the spot.

Three legs are carried as columns (coupon, knock-out, loss); the value is their sum.

Banked from the 3C evidence `dcn_exact.py` and `dcn_exact_term.py` (spec 3C §7.2); not production code.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface

SQRT_2PI = np.sqrt(2.0 * np.pi)


def _nodes(edges, width, order):
    gx, gw = np.polynomial.legendre.leggauss(order)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        panels = max(1, int(np.ceil((b - a) / width)))
        e = np.linspace(a, b, panels + 1)
        half, mid = 0.5 * np.diff(e), 0.5 * (e[:-1] + e[1:])
        xs.append((mid[:, None] + half[:, None] * gx[None, :]).ravel())
        ws.append((half[:, None] * gw[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


def _flat_market(env):
    assert isinstance(env.vol_surface, FlatVolSurface), "flat vol only"
    assert isinstance(env.rate_curve, FlatRateCurve), "flat rate only"
    assert isinstance(env.div_yield, ContinuousDividendYield), "flat dividend yield only"
    return float(env.get_vol(float(env.spot), 1.0)), float(env.get_rate(1.0)), float(env.get_div_yield(1.0))


def _kernel(x, w, mu_dt, s, kcut):
    """Sparse K[i, j] = w_j * pdf((x_j - x_i - mu_dt) / s) / s, cut at |z| <= kcut."""
    centre = x + mu_dt
    a = np.searchsorted(x, centre - kcut * s, side="left")
    b = np.searchsorted(x, centre + kcut * s, side="right")
    length = b - a
    indptr = np.concatenate([[0], np.cumsum(length)])
    cols = np.arange(indptr[-1]) - np.repeat(indptr[:-1], length) + np.repeat(a, length)
    z = (x[cols] - np.repeat(centre, length)) / s
    data = w[cols] * np.exp(-0.5 * z * z) / (s * SQRT_2PI)
    return sp.csr_matrix((data, cols, indptr), shape=(x.size, x.size))


def dcn_exact(product, env, spots=None, *, width=0.006, order=12, nsig=8.0, kcut=8.5):
    """Unsigned legs, value, delta and gamma at each spot (default: the environment's).

    Returns a dict of arrays over `spots`; `pv` carries the product's direction sign, as the
    engines' `pv` does. `conservation` is the largest |row sum - 1| of a kernel over the rows
    whose support lies inside the domain (a quadrature check, not a tolerance).
    """
    vol, r, q = _flat_market(env)
    ctx = build_dcn_grid_context(product)
    times = np.asarray(ctx.times, dtype=float)
    assert times[0] == 0.0 and np.all(np.diff(times) > 0), "the daily grid starts at the valuation date"
    spots = np.atleast_1d(np.asarray([float(env.spot)] if spots is None else spots, dtype=float))
    mu = r - q - 0.5 * vol * vol
    T = float(times[-1])
    b_ki, b_c, b_ko, k_loss = (np.log(float(v)) for v in
                               (product.ki_barrier, product.coupon_barrier, product.ko_barrier, product.k_loss))
    x_lo, x_hi = float(np.log(spots.min())), float(np.log(spots.max()))
    reach = nsig * vol * np.sqrt(T) + abs(mu) * T
    lo = min(b_ki, x_lo) - reach
    hi = max(b_ko, k_loss, x_hi) + reach
    x, w = _nodes(sorted({lo, b_ki, b_c, b_ko, k_loss, hi}), width, order)
    s_nodes = np.exp(x)
    ki, ko, cp = x <= b_ki, x >= b_ko, x >= b_c          # GL nodes never sit on an edge

    notional, s0, part = float(product.notional), float(product.initial_price), float(product.participation)
    df = lambda t: np.exp(-r * float(t))                  # noqa: E731  (flat curve)
    obs_at = {int(c): j for j, c in enumerate(ctx.obs_cols)}

    def amounts(j, t_obs):
        cpn = ko_amt = None
        if ctx.obs_is_coupon[j]:
            cpn = part * product.coupon_rate * ctx.coupon_accruals[j] * notional * df(ctx.coupon_pay_times[j]) / df(t_obs)
        if ctx.obs_is_ko[j]:
            ko_amt = part * product.ko_coupon_rate * ctx.ko_accruals[j] * notional * df(ctx.ko_pay_times[j]) / df(t_obs)
        return cpn, ko_amt

    def events(v0, v1, j, t_obs):
        """The contract at one date, backward: columns (coupon, knock-out, loss)."""
        if j is not None:
            cpn, ko_amt = amounts(j, t_obs)
            if cpn is not None:
                v0[cp, 0] += cpn                           # never knocked in, at or above the coupon barrier
            if ko_amt is not None:
                for v in (v0, v1):                         # knock-out has priority over the coupon
                    v[ko, :] = 0.0
                    v[ko, 1] = ko_amt
        v0[ki, :] = v1[ki, :]                              # the daily knock-in, tested before the knock-out
        return v0, v1

    # maturity: the loss leg of a knocked-in survivor, settled at the loss payment date
    v1 = np.zeros((x.size, 3))
    v1[:, 2] = -(notional / s0) * part * np.maximum(np.exp(k_loss) - s_nodes, 0.0) * df(ctx.loss_pay_time) / df(T)
    v0 = np.zeros((x.size, 3))
    v0, v1 = events(v0, v1, obs_at.get(times.size - 1), T)

    kernels, conservation = {}, 0.0
    for i in range(times.size - 2, 0, -1):                 # propagate to date i, then its events
        dt = float(times[i + 1] - times[i])
        key = round(dt * 365.0)
        if key not in kernels:
            s = vol * np.sqrt(dt)
            kernels[key] = _kernel(x, w, mu * dt, s, kcut)
            inside = (x + mu * dt - kcut * s > lo) & (x + mu * dt + kcut * s < hi)
            rows = np.asarray(kernels[key].sum(axis=1)).ravel()
            conservation = max(conservation, float(np.max(np.abs(rows[inside] - 1.0))))
        disc = np.exp(-r * dt)
        v0, v1 = disc * (kernels[key] @ v0), disc * (kernels[key] @ v1)
        v0, v1 = events(v0, v1, obs_at.get(i), float(times[i]))

    # valuation date: the kernel centred on each spot, with its x-derivatives
    dt = float(times[1] - times[0])
    s = vol * np.sqrt(dt)
    xs = np.log(spots)
    z = (x[None, :] - xs[:, None] - mu * dt) / s
    k = w[None, :] * np.exp(-0.5 * z * z) / (s * SQRT_2PI) * np.exp(-r * dt)
    knocked_in = bool(product.knocked_in_at_valuation)
    out = {"spot": spots}
    legs = np.empty((spots.size, 3)); vx = np.empty(spots.size); vxx = np.empty(spots.size)
    for n, spot in enumerate(spots):
        surface = v1 if (knocked_in or spot <= float(product.ki_barrier)) else v0
        legs[n] = k[n] @ surface
        total = surface.sum(axis=1)
        vx[n] = (k[n] * z[n] / s) @ total
        vxx[n] = (k[n] * (z[n] * z[n] - 1.0) / (s * s)) @ total
    sign = float(product.direction_sign)
    out.update(coupon=legs[:, 0], ko=legs[:, 1], loss=legs[:, 2], pv=sign * legs.sum(axis=1),
               delta=sign * vx / spots, gamma=sign * (vxx - vx) / (spots * spots),
               conservation=conservation, nodes=int(x.size), dates=int(times.size))
    return out


# --- term curves (dcn_exact_term.py): per-step drift and discount from the environment -----------

def _market(env):
    """Constant vol; deterministic rate and dividend curves through the environment's own accessors:
    df(t) the discount factor, growth(t) = exp((r(t) - q(t)) t) with r, q the zero rates to t."""
    assert isinstance(env.vol_surface, FlatVolSurface), "flat vol only"
    vol = float(env.get_vol(float(env.spot), 1.0))
    df = lambda t: 1.0 if float(t) <= 0.0 else float(env.get_discount_factor(float(t)))            # noqa: E731
    growth = lambda t: 1.0 if float(t) <= 0.0 else float(np.exp((env.get_rate(float(t)) - env.get_div_yield(float(t))) * float(t)))  # noqa: E731
    return vol, df, growth


def dcn_exact_term(product, env, spots=None, *, width=0.006, order=12, nsig=8.0, kcut=8.5):
    """Unsigned legs, value, delta and gamma at each spot (default: the environment's).

    Returns a dict of arrays over `spots`; `pv` carries the product's direction sign, as the
    engines' `pv` does. `conservation` is the largest |row sum - 1| of a kernel over the rows
    whose support lies inside the domain (a quadrature check, not a tolerance).
    """
    vol, df, growth = _market(env)
    ctx = build_dcn_grid_context(product)
    times = np.asarray(ctx.times, dtype=float)
    assert times[0] == 0.0 and np.all(np.diff(times) > 0), "the daily grid starts at the valuation date"
    spots = np.atleast_1d(np.asarray([float(env.spot)] if spots is None else spots, dtype=float))
    T = float(times[-1])
    b_ki, b_c, b_ko, k_loss = (np.log(float(v)) for v in
                               (product.ki_barrier, product.coupon_barrier, product.ko_barrier, product.k_loss))
    x_lo, x_hi = float(np.log(spots.min())), float(np.log(spots.max()))
    drift = np.array([np.log(growth(b) / growth(a)) - 0.5 * vol * vol * (b - a) for a, b in zip(times[:-1], times[1:])])
    reach = nsig * vol * np.sqrt(T) + float(np.abs(drift).sum())
    lo = min(b_ki, x_lo) - reach
    hi = max(b_ko, k_loss, x_hi) + reach
    x, w = _nodes(sorted({lo, b_ki, b_c, b_ko, k_loss, hi}), width, order)
    s_nodes = np.exp(x)
    ki, ko, cp = x <= b_ki, x >= b_ko, x >= b_c          # GL nodes never sit on an edge

    notional, s0, part = float(product.notional), float(product.initial_price), float(product.participation)
    obs_at = {int(c): j for j, c in enumerate(ctx.obs_cols)}

    def amounts(j, t_obs):
        cpn = ko_amt = None
        if ctx.obs_is_coupon[j]:
            cpn = part * product.coupon_rate * ctx.coupon_accruals[j] * notional * df(ctx.coupon_pay_times[j]) / df(t_obs)
        if ctx.obs_is_ko[j]:
            ko_amt = part * product.ko_coupon_rate * ctx.ko_accruals[j] * notional * df(ctx.ko_pay_times[j]) / df(t_obs)
        return cpn, ko_amt

    def events(v0, v1, j, t_obs):
        """The contract at one date, backward: columns (coupon, knock-out, loss)."""
        if j is not None:
            cpn, ko_amt = amounts(j, t_obs)
            if cpn is not None:
                v0[cp, 0] += cpn                           # never knocked in, at or above the coupon barrier
            if ko_amt is not None:
                for v in (v0, v1):                         # knock-out has priority over the coupon
                    v[ko, :] = 0.0
                    v[ko, 1] = ko_amt
        v0[ki, :] = v1[ki, :]                              # the daily knock-in, tested before the knock-out
        return v0, v1

    # maturity: the loss leg of a knocked-in survivor, settled at the loss payment date
    v1 = np.zeros((x.size, 3))
    v1[:, 2] = -(notional / s0) * part * np.maximum(np.exp(k_loss) - s_nodes, 0.0) * df(ctx.loss_pay_time) / df(T)
    v0 = np.zeros((x.size, 3))
    v0, v1 = events(v0, v1, obs_at.get(times.size - 1), T)

    kernels, conservation = {}, 0.0
    for i in range(times.size - 2, 0, -1):                 # propagate to date i, then its events
        dt = float(times[i + 1] - times[i])
        key = (round(dt * 365.0), round(float(drift[i]), 14))     # one kernel per distinct (dt, drift)
        if key not in kernels:
            s = vol * np.sqrt(dt)
            kernels = {key: _kernel(x, w, float(drift[i]), s, kcut)} if len(kernels) > 8 else {**kernels, key: _kernel(x, w, float(drift[i]), s, kcut)}
            inside = (x + drift[i] - kcut * s > lo) & (x + drift[i] + kcut * s < hi)
            rows = np.asarray(kernels[key].sum(axis=1)).ravel()
            conservation = max(conservation, float(np.max(np.abs(rows[inside] - 1.0))))
        disc = df(times[i + 1]) / df(times[i])
        v0, v1 = disc * (kernels[key] @ v0), disc * (kernels[key] @ v1)
        v0, v1 = events(v0, v1, obs_at.get(i), float(times[i]))

    # valuation date: the kernel centred on each spot, with its x-derivatives
    dt = float(times[1] - times[0])
    s = vol * np.sqrt(dt)
    xs = np.log(spots)
    z = (x[None, :] - xs[:, None] - float(drift[0])) / s
    k = w[None, :] * np.exp(-0.5 * z * z) / (s * SQRT_2PI) * (df(times[1]) / df(times[0]))
    knocked_in = bool(product.knocked_in_at_valuation)
    out = {"spot": spots}
    legs = np.empty((spots.size, 3)); vx = np.empty(spots.size); vxx = np.empty(spots.size)
    for n, spot in enumerate(spots):
        surface = v1 if (knocked_in or spot <= float(product.ki_barrier)) else v0
        legs[n] = k[n] @ surface
        total = surface.sum(axis=1)
        vx[n] = (k[n] * z[n] / s) @ total
        vxx[n] = (k[n] * (z[n] * z[n] - 1.0) / (s * s)) @ total
    sign = float(product.direction_sign)
    out.update(coupon=legs[:, 0], ko=legs[:, 1], loss=legs[:, 2], pv=sign * legs.sum(axis=1),
               delta=sign * vx / spots, gamma=sign * (vxx - vx) / (spots * spots),
               conservation=conservation, nodes=int(x.size), dates=int(times.size))
    return out
