"""High-precision references for the double no-touch probability, evaluated at the EXACT double inputs
(review 3 finding 1). mpmath is NOT a project dependency: put it on PYTHONPATH.
Banked from the 3C evidence `w_ref.py` (spec 3C §7.2); not production code. mpmath is imported inside each
function, so importing this module never needs it.

`image(...)`: the image series, every float taken as the rational it is, with working precision set from
the largest exponent a term forms (a product of a huge and a tiny mpf loses nothing; a sum of their logs
in doubles loses everything, which is the finding). A difference of two normal CDFs is formed from
the two tails on the side where they are small (`_ndiff`): at any working precision Phi(19) - Phi(1000)
computed directly is 0, and the first version of this file returned 0.5 for 0.4795 that way. `bracket(...)`: the one-barrier first-passage closed
forms; P_U + P_L - 1 <= P <= min(P_U, P_L), tight whenever one barrier is remote."""


def _setup(spot, T, r, q, vol, lo, up):
    import mpmath as mp

    mp.mp.dps = 400
    S, T_, r_, q_, v, L, U = (mp.mpf(z) for z in (spot, T, r, q, vol, lo, up))
    a, x = mp.log(U / L), mp.log(S / L)
    s = v * mp.sqrt(T_)
    mu = r_ - q_ - v * v / 2
    return a, x, s, mu * T_, mu / (v * v)


def _ndiff(hi, lo):
    """Phi(hi) - Phi(lo), hi > lo, from tails."""
    import mpmath as mp

    if lo > 0:
        return mp.ncdf(-lo) - mp.ncdf(-hi)
    if hi < 0:
        return mp.ncdf(hi) - mp.ncdf(lo)
    return 1 - mp.ncdf(-hi) - mp.ncdf(lo)


def image(spot, T, r, q, vol, lo, up, digits=60):
    import mpmath as mp

    a, x, s, m, alpha = _setup(spot, T, r, q, vol, lo, up)
    tau = (s / a) ** 2
    pairs = int(mp.sqrt(70 * tau)) + 3                       # exp(-2 K^2 / tau) < 1e-60
    big = max(abs(2 * alpha * (pairs + 1) * a), ((2 * pairs + 2) * a + abs(m)) ** 2 / (2 * s * s), mp.mpf(10))
    mp.mp.dps = int(digits + mp.log10(big) + 15)
    tot = mp.mpf(0)
    for k in range(-pairs, pairs + 1):
        sh = 2 * k * a
        tot += mp.exp(2 * alpha * k * a) * _ndiff((a - x - sh - m) / s, (-x - sh - m) / s)
        tot -= mp.exp(2 * alpha * (k * a - x)) * _ndiff((a + x - sh - m) / s, (x - sh - m) / s)
    return tot


def bracket(spot, T, r, q, vol, lo, up, digits=60):
    import mpmath as mp

    a, x, s, m, alpha = _setup(spot, T, r, q, vol, lo, up)
    d_up, d_lo = a - x, x
    big = max(abs(2 * alpha * a), (a + abs(m)) ** 2 / (2 * s * s), mp.mpf(10))
    mp.mp.dps = int(digits + mp.log10(big) + 15)
    # hit probabilities (each a sum of two tails), so that nothing is lost when they are tiny
    h_up = mp.ncdf(-(d_up - m) / s) + mp.exp(2 * alpha * d_up) * mp.ncdf((-d_up - m) / s)
    h_lo = mp.ncdf(-(d_lo + m) / s) + mp.exp(-2 * alpha * d_lo) * mp.ncdf((-d_lo + m) / s)
    return 1 - h_up - h_lo, 1 - max(h_up, h_lo)
