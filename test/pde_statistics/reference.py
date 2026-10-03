"""Exact event statistics of a Snowball or (memory-free) Phoenix under flat GBM: a two-state
(not knocked in / knocked in) recursion over the observation dates with indicator columns.
No engine code is used; the cashflows are the product's own.

Columns, all undiscounted expectations at t = 0 (the discounting is applied at the end
from the flat curve, which is exact because every payment date is deterministic):
  KO_k   P(knock-out at observation k)
  C_k    P(coupon paid at observation k)                      (Phoenix; a simultaneous KO still pays)
  KIS    P(settles knocked in) = P(knock-in AND never knocked out)
  M      E[maturity payoff * 1{never knocked out}]            (the maturity leg)
  KIE    P(the knock-in barrier is breached at any time in [0, T]), regardless of knock-out
         (the MC / PDE "ki_ever" definition: the path is monitored after a knock-out too)

Between two dates the transition kernel is the Gaussian of the log-return; under continuous
knock-in the not-knocked-in state uses the Brownian-bridge stay-above probability
1 - exp(-2 (x - b)(y - b) / (sigma^2 dt)) (drift-free). The knock-out region at a date is the
nodal absorption on panels that end on the barrier. Every integral is composite Gauss-Legendre on
panels that end on the knock-in barrier, the strike and the coupon barrier, so each integrand
is smooth on a panel. The quadrature error is reported as the difference to a run with twice
the panel width.

Banked from the 3B evidence directory `exact_stats.py` (spec 3B §5.7); not production code.
"""
from __future__ import annotations

import numpy as np

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface
from quantark.util.enum import ObservationType


def _npdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _nodes(edges, width, order=12):
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
    T = 1.0
    return float(env.spot), float(env.get_vol(100.0, T)), float(env.get_rate(T)), float(env.get_div_yield(T))


def _product_terms(product, env):
    """Dates, barriers and cashflows read from the product (the same records the engines resolve)."""
    records = product.resolve_ko_observations(env)
    ko_times = np.array([float(r.observation_time) for r in records])
    ko_barriers = np.array([float(r.barrier) for r in records])
    ko_payoffs = np.array([float(r.payoff) if r.payoff is not None else 0.0 for r in records])
    ko_settle = np.array([float(r.settlement_time) if r.settlement_time is not None else float(r.observation_time)
                          for r in records])
    assert np.all(ko_barriers == ko_barriers[0]), "constant knock-out barrier only (closed-form tail)"
    bc = product.barrier_config
    ki_barrier = float(bc.ki_barrier)
    ki_continuous = bool(bc.ki_continuous or bc.ki_observation_type == ObservationType.CONTINUOUS)
    if ki_continuous:
        ki_times = np.array([])
    else:
        prof = product.get_ki_observation_profile(env)
        ki_times = np.array([float(t) for t in prof["observation_times"]])
        bars = prof.get("barriers")
        if bars:
            assert all(abs(float(b) - ki_barrier) < 1e-12 for b in bars), "constant knock-in barrier only"
    assert not (bc.disable_ko_after_ki and product.has_ki_barrier), "knock-out must survive knock-in here"
    coupon = None
    if isinstance(product, PhoenixOption):
        assert not product.has_memory_coupon, "memory coupons are not in this recursion (QUAD V2 is the reference)"
        cb = product.coupon_config.coupon_barrier
        n = len(records)
        bars = np.array([float(b) for b in cb]) if isinstance(cb, list) else np.full(n, float(cb))
        assert np.all(bars == bars[0]), "constant coupon barrier only"
        amounts = np.array([float(product.get_coupon_payoff(i)) for i in range(n)])
        pay_times = ko_settle.copy()       # INSTANT coupons settle with their observation's record
        from quantark.util.enum import CouponPayType
        assert product.coupon_config.coupon_pay_type == CouponPayType.INSTANT, "instant coupons only"
        coupon = dict(barrier=float(bars[0]), amounts=amounts, pay_times=pay_times)
    assert not bool(getattr(product, "is_reverse", False)), "reverse products (down knock-out, up knock-in) are not implemented"
    assert ko_times.size and ko_times[0] > 1e-12, "an observation at the valuation date is not implemented (the engines decide it at the spot)"
    terminal = resolve_terminal_timing(product, env)
    T = float(product.get_maturity(env))
    T_pay = float(terminal.payment_time)   # the maturity payoff settles at the resolved terminal payment time
    assert T_pay >= T - 1e-12, "terminal payment before maturity"
    assert abs(ko_times[-1] - T) < 1e-12, "the last observation is the maturity"
    return dict(ko_times=ko_times, ko_barrier=float(ko_barriers[0]), ko_payoffs=ko_payoffs, ko_settle=ko_settle,
                ki_barrier=ki_barrier, ki_continuous=ki_continuous, ki_times=ki_times, coupon=coupon, T=T, T_pay=T_pay)


def exact_event_stats(product, env, *, width=0.006, kie_width=0.01, order=12):
    spot, vol, r, q = _flat_market(env)
    terms = _product_terms(product, env)
    T, b = terms["T"], np.log(terms["ki_barrier"])
    x_ko, x0 = np.log(terms["ko_barrier"]), np.log(spot)
    x_k = np.log(float(product.strike))
    coupon = terms["coupon"]
    knocked_in_now = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
    if terms["ki_continuous"] and x0 <= b:
        knocked_in_now = True

    # event dates: the union of knock-out, knock-in and coupon dates, as fractions of a year
    dates = set(np.round(terms["ko_times"], 12).tolist())
    if not terms["ki_continuous"]:
        dates |= set(np.round(terms["ki_times"], 12).tolist())
    dates = np.array(sorted(dates))
    ko_at = {round(float(t), 12): k for k, t in enumerate(terms["ko_times"])}
    ki_at = set() if terms["ki_continuous"] else {round(float(t), 12) for t in terms["ki_times"]}
    n_ko = len(terms["ko_times"])

    # columns
    names = [f"KO_{k}" for k in range(n_ko)]
    if coupon is not None:
        names += [f"C_{k}" for k in range(n_ko)]
    names += ["KIS", "M", "ONE"]
    col = {name: i for i, name in enumerate(names)}
    n_cols = len(names)

    # nodes below the knock-out barrier, panels ending on the barriers and the strike
    drift = abs(r - q - 0.5 * vol * vol) * T
    lo = min(b, x_k, x0) - 8.0 * vol * np.sqrt(T) - drift
    hi = max(x_ko, x0) + 8.0 * vol * np.sqrt(T) + drift
    edges = sorted({lo, b, x_k, x_ko, hi} | ({np.log(coupon["barrier"])} if coupon else set()))
    x, w = _nodes(edges, width, order)
    s = np.exp(x)
    alive = x > b                                           # the not-knocked-in state exists here (continuous KI)
    below_b = x <= b
    ko_region = x >= x_ko                                   # a knock-out date absorbs this region (nodal, inclusive)

    def kernels(x_from, dt):
        mu, sd = (r - q - 0.5 * vol * vol) * dt, vol * np.sqrt(dt)
        mean = x_from[:, None] + mu
        free = _npdf((x[None, :] - mean) / sd) / sd * w[None, :]
        if terms["ki_continuous"]:
            cross = np.exp(-2.0 * np.maximum(x_from[:, None] - b, 0.0) * np.maximum(x[None, :] - b, 0.0) / (sd * sd))
            stay = free * (1.0 - cross)                      # zero wherever y <= b
            return free, stay
        return free, None

    def apply_events(W_in, W_out, t, terminal=False):
        """The events of date ``t`` on the post-continuation values: knock-out, coupon, knock-in."""
        key = round(float(t), 12)
        k = ko_at.get(key)
        if k is not None:
            W_in[ko_region, :] = 0.0
            W_out[ko_region, :] = 0.0
            W_in[ko_region, col[f"KO_{k}"]] = 1.0
            W_out[ko_region, col[f"KO_{k}"]] = 1.0
        if coupon is not None and k is not None:
            mask = s >= coupon["barrier"]
            W_in[mask, col[f"C_{k}"]] = 1.0
            W_out[mask, col[f"C_{k}"]] = 1.0
        if key in ki_at:
            W_out[below_b, :] = W_in[below_b, :]
        return W_in, W_out


    # terminal values at the maturity date, before its events
    W_in = np.zeros((x.size, n_cols))
    W_out = np.zeros((x.size, n_cols))
    W_in[:, col["KIS"]] = 1.0
    W_in[:, col["ONE"]] = 1.0
    W_out[:, col["ONE"]] = 1.0
    W_in[:, col["M"]] = np.array([product.get_maturity_payoff_v1(float(v), pricing_env=env) for v in s])
    W_out[:, col["M"]] = np.array([product.get_maturity_payoff_v0(float(v), pricing_env=env) for v in s])
    W_in, W_out = apply_events(W_in, W_out, dates[-1], terminal=True)

    def step(x_from, dt, W_in, W_out):
        free, stay = kernels(x_from, dt)
        V_in = free @ W_in
        if stay is not None:
            touched = free @ W_in - stay @ W_in
            V_out = stay @ W_out + touched
        else:
            V_out = free @ W_out
        return V_in, V_out

    for m in range(len(dates) - 1, 0, -1):
        dt = dates[m] - dates[m - 1]
        V_in, V_out = step(x, dt, W_in, W_out)
        if terms["ki_continuous"]:
            V_out[~alive, :] = V_in[~alive, :]
        W_in, W_out = apply_events(V_in, V_out, dates[m - 1])
    V_in, V_out = step(np.array([x0]), dates[0], W_in, W_out)
    v = (V_in if knocked_in_now else V_out)[0]

    mass_lost = 1.0 - float(np.sum([v[col[f"KO_{k}"]] for k in range(n_ko)]) + v[col["ONE"]])
    if abs(mass_lost) > 1e-10:
        raise ValueError(f"integration domain loses probability mass: {mass_lost:.3e} (widen the domain)")

    # knock-in ever: no knock-out absorption
    xe, we = _nodes(sorted({lo, b, hi}), kie_width, order)
    U = np.zeros(xe.size)
    ONE_E = np.ones(xe.size)
    for m in range(len(dates) - 1, -1, -1):
        dt = dates[m] - (dates[m - 1] if m > 0 else 0.0)
        x_from = xe if m > 0 else np.array([x0])
        mu, sd = (r - q - 0.5 * vol * vol) * dt, vol * np.sqrt(dt)
        mean = x_from[:, None] + mu
        free = _npdf((xe[None, :] - mean) / sd) / sd * we[None, :]
        if terms["ki_continuous"]:
            cross = np.exp(-2.0 * np.maximum(x_from[:, None] - b, 0.0) * np.maximum(xe[None, :] - b, 0.0) / (sd * sd))
            stay = free * (1.0 - cross)
            Unew = (free - stay) @ np.ones(xe.size) + stay @ U
        else:
            Ua = U.copy()
            if round(float(dates[m]), 12) in ki_at:
                Ua[xe <= b] = 1.0
            Unew = free @ Ua
        ONE_E_new = free @ ONE_E
        if m > 0:
            Unew[xe <= b] = 1.0 if terms["ki_continuous"] else Unew[xe <= b]
            U = Unew
            ONE_E = ONE_E_new
        else:
            kie = float(Unew[0]) if not knocked_in_now else 1.0
            kie_mass_lost = 1.0 - float(ONE_E_new[0])
    if abs(kie_mass_lost) > 1e-10:
        raise ValueError(f"knock-in-ever domain loses probability mass: {kie_mass_lost:.3e}")

    df = lambda t: float(np.exp(-r * t))
    ko_p = np.array([v[col[f"KO_{k}"]] for k in range(n_ko)])
    ko_cf = ko_p * terms["ko_payoffs"] * np.array([df(t) for t in terms["ko_settle"]])
    out = dict(ko_times=terms["ko_times"].tolist(), ko_probability=ko_p.tolist(),
               expected_discounted_ko_cashflow=ko_cf.tolist(),
               survival_probability=np.maximum(0.0, 1.0 - np.cumsum(ko_p)).tolist(),
               ki_survive_knocked_in_probability=float(v[col["KIS"]]), ki_ever_probability=kie,
               expected_discounted_maturity_cashflow=float(df(terms["T_pay"]) * v[col["M"]]),
               mass_lost=mass_lost, kie_mass_lost=kie_mass_lost)
    pv = float(ko_cf.sum() + out["expected_discounted_maturity_cashflow"])
    if coupon is not None:
        cp = np.array([v[col[f"C_{k}"]] for k in range(n_ko)])
        ccf = cp * coupon["amounts"] * np.array([df(t) for t in coupon["pay_times"]])
        out["coupon_probability"], out["expected_discounted_coupon_cashflow"] = cp.tolist(), ccf.tolist()
        pv += float(ccf.sum())
    out["pv"] = pv
    for k, val in out.items():
        if not np.all(np.isfinite(np.asarray(val, dtype=float))):
            raise ValueError(f"non-finite reference output in {k}")
    return out


def exact_with_error(product, env, width=0.006, kie_width=0.01):
    fine = exact_event_stats(product, env, width=width, kie_width=kie_width)
    coarse = exact_event_stats(product, env, width=2 * width, kie_width=2 * kie_width)
    err = {}
    for k, v in fine.items():
        if k in ("mass_lost", "kie_mass_lost"):
            continue
        if isinstance(v, list) and v and isinstance(v[0], float):
            err[k] = float(np.max(np.abs(np.asarray(v) - np.asarray(coarse[k]))))
        elif isinstance(v, float):
            err[k] = abs(v - coarse[k])
    fine["quadrature_error"] = err
    return fine
