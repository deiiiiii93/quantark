"""C08 (Q3): the second mechanism -- the reflection factor overflows on the near-zero-variance columns (reverse).

``_ContinuousKIFirstPassage._geometry`` forms  G = exp(-2 mu_e a / sig2) * [m Phi(m/s) + s phi(m/s)],  m = mu_e dt - a,
s = sqrt(sig2 dt). On a reverse (up) barrier mu_e = -mu < 0 when r - q > sig2 / 2, so the factor is exp(+...). The 2D
v = 0 column floors sig2 at 1e-10 (s ~ 1e-6): a live node within 10 s of the barrier (a < ~1e-5 in log spot) enters the
band and the factor overflows to inf while the bracket underflows to 0: inf x 0 = NaN, written into V0 (C07: KI 114.45 on
the production PV grid, n_x 200, nearest node 0.0009 cells). The half-cell floor does not touch it (it chooses k*, not G).

The product is bounded: exp(-2 mu_e a / sig2) phi(m / s) = phi((mu_e dt + a) / s) (completing the square), and
exp(-2 mu_e a / sig2) Phi(m / s) = exp(-2 mu_e a / sig2 + log Phi(m / s)). PROPOSED evaluation, ``_g_guarded``: the
same closed form in that one-exponent form ONLY on the band nodes where the naive exponent exceeds 700 (np.exp
overflows past ~709), so every value that is finite today keeps its bits.

Arms on KI 114.45 (production PV on (200, 60, 800)) and the C03 / C07 divergent cases: as-is, floor only (C02), floor
+ guarded reflection. Inertness of the guarded reflection, float.hex(), on the C05 set.
"""
import math

import numpy as np
from scipy.special import log_ndtr, ndtr

import c02_q3_common as q3
from c02_q3_common import BSM_LIMIT, REV_C, REV_D, STD_C, px
from c05_production_and_inertness import MEDIUM, PROD, fmt
from common import HP, env, snowball, solver
from quantark.asset.equity.engine.pde import snowball_pde_solver as sps
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EXP_GUARD = 700.0


def _g_guarded(mu_e, dt, ab, sig2, s_step):
    m = mu_e * dt - ab
    t = m / s_step
    expo = -2.0 * mu_e * ab / sig2
    G = np.empty_like(ab)
    naive = expo <= _EXP_GUARD
    if naive.any():
        tn = t[naive]
        bracket = m[naive] * ndtr(tn) + s_step * np.exp(-0.5 * tn * tn) / _SQRT_2PI
        G[naive] = np.exp(expo[naive]) * bracket                               # today's arithmetic, bit for bit
    big = ~naive
    if big.any():
        u = (mu_e * dt + ab[big]) / s_step
        G[big] = (m[big] * np.exp(expo[big] + log_ndtr(t[big]))
                  + s_step * np.exp(-0.5 * u * u) / _SQRT_2PI)
    return G


def _geometry_fixed(self, dt, mu, sig2, s_vec, barrier, floor_at_barrier=True):
    key = (round(dt, 12), round(mu, 12), round(sig2, 12), round(float(barrier), 12), len(s_vec))
    cached = self._cache.get(key)
    if cached is not None:
        return cached
    result = (None, None, None)
    if dt > 0.0 and sig2 > 0.0 and barrier > 0.0:
        log_s = np.log(np.asarray(s_vec, dtype=float))
        log_b = np.log(float(barrier))
        if self._is_reverse:
            a, mu_e = log_b - log_s, -mu
        else:
            a, mu_e = log_s - log_b, mu
        s_step = np.sqrt(sig2 * dt)
        band = np.flatnonzero((a > 0.0) & (a < self._BAND_STDS * s_step))
        if band.size:
            G = np.zeros(len(s_vec), dtype=float)
            G[band] = _g_guarded(mu_e, dt, a[band], sig2, s_step)
            live = np.flatnonzero(a > 0.0)
            if live.size > 1:
                near = live[np.argsort(a[live], kind="stable")] if floor_at_barrier else live
                resolved = live[a[live] >= 0.5 * (a[near[1]] - a[near[0]])]
                if resolved.size:
                    live = resolved
            k_star = int(live[np.argmin(np.abs(a[live] - s_step))])
            result = (G, k_star, float(a[k_star]))
    self._cache[key] = result
    return result


def arms(label, f):
    out = {"as-is": f()}
    with q3.barrier_cell_floor():
        out["floor"] = f()
    sps._ContinuousKIFirstPassage._geometry = _geometry_fixed
    try:
        out["floor + guarded reflection"] = f()
    finally:
        sps._ContinuousKIFirstPassage._geometry = q3._ORIG_GEOM
    print(f"{label}: " + "  |  ".join(f"{k} {v}" for k, v in out.items()), flush=True)


def guarded_only_hex(f):
    a = f()
    sps._ContinuousKIFirstPassage._geometry = (
        lambda self, *args: _geometry_fixed(self, *args, floor_at_barrier=False))
    try:
        b = f()
    finally:
        sps._ContinuousKIFirstPassage._geometry = q3._ORIG_GEOM
    return f"{a!r} vs {b!r}  {'bit-identical' if float(a).hex() == float(b).hex() else f'MOVED {b - a:+.3e}'}"


def greeks(spec):
    try:
        g = solver(HP, *MEDIUM, **PROD).calculate_greeks(snowball(**spec), env())
        return " ".join(f"{k} {fmt(float(v))}" for k, v in g.items())
    except Exception as exc:  # noqa: BLE001 - probe
        return f"{type(exc).__name__}"


if __name__ == "__main__":
    np.seterr(all="ignore")
    ki114 = dict(REV_C, ki=114.45)
    arms("KI 114.45 production PV (200, 60, 800)", lambda: px(MEDIUM, spec=ki114, **PROD))
    arms("KI 114.45 production Greeks", lambda: greeks(ki114))
    arms("KI 139.80 production Greeks", lambda: greeks(dict(REV_C, ki=139.80)))
    for n in ((312, 48, 192), (317, 48, 192), (384, 48, 192), (322, 48, 192)):
        arms(f"KI 125 n={n}", lambda n=n: px(n))
    print("inertness of the guarded reflection alone (library floor kept), float.hex():", flush=True)
    n = (384, 48, 192)
    print("   2D rev continuous (288,48,192)", guarded_only_hex(lambda: solver(HP, 288, 48, 192).price(
        snowball(**REV_C), env())), flush=True)
    print("   2D std continuous   ", guarded_only_hex(lambda: solver(HP, *n).price(snowball(**STD_C), env())),
          flush=True)
    print("   2D rev discrete     ", guarded_only_hex(lambda: solver(HP, *n).price(snowball(**REV_D), env())),
          flush=True)
    print("   2D rev BSM limit    ", guarded_only_hex(lambda: solver(BSM_LIMIT, *n).price(snowball(**REV_C), env())),
          flush=True)
    print("   1D std continuous   ", guarded_only_hex(lambda: SnowballPDESolver().price(snowball(**STD_C), env())),
          flush=True)
    print("   1D rev continuous   ", guarded_only_hex(lambda: SnowballPDESolver().price(snowball(**REV_C), env())),
          flush=True)
