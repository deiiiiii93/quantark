"""Option (e): evaluate the LAST transition at spot instead of interpolating it.

The engine diffuses the value surface to t=0 on the grid and then reads it
off with np.interp at log-moneyness 0.  But the final transition is an
explicit smooth function of the readout coordinate:

    V(x) = prefactor * scale * sum_j u_j * omega(x - x_j)  +  tail(x)

with u_j the already-weighted nodal values, omega the Gaussian kernel and
tail the closed-form erfc term.  Evaluating THAT at x = 0 is one off-grid
row of the operator the engine already applies -- not a new quadrature rule
and not interpolation.  QuadratureCore._calculate_final_value already works
this way (quad_core.py:591); only the bespoke snowball path interpolates.

Runtime patched, no engine edit.
"""
from __future__ import annotations
import math

import numpy as np
from scipy.special import erfc

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from candidates import cubic_interpolate

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005

_ORIG_DIFFUSE = SnowballQuadEngine._diffuse_fft
_ORIG_INTERP = QuadratureMath.interpolate

REGISTRY: dict[int, tuple] = {}
KEEP: list = []
STATS = {"pointwise": 0, "fallback": 0}


def diffuse_recording(self, values, math_utils, omega_array, prefactor,
                      p_lr, p_ur, p0, alpha, beta, tau_step):
    out = _ORIG_DIFFUSE(self, values, math_utils, omega_array, prefactor,
                        p_lr, p_ur, p0, alpha, beta, tau_step)
    if out.ndim == 1:
        REGISTRY[id(out)] = (
            np.asarray(values).copy(), math_utils, float(prefactor),
            int(p_lr), int(p_ur), int(p0), float(alpha), float(beta),
            float(tau_step),
        )
        KEEP.append(out)
    return out


def pointwise(self, values, x: float = 0.0) -> float:
    entry = REGISTRY.get(id(values))
    if entry is None:
        STATS["fallback"] += 1
        return _ORIG_INTERP(self, values, x)
    STATS["pointwise"] += 1
    src, mu, prefactor, p_lr, p_ur, p0, alpha, beta, tau = entry
    grid = mu.grid
    u = mu.simpson_weights(src, p_lr, p_ur, p0)[: len(grid)]
    z = x - grid
    omega = np.exp(-(z * z) / (4.0 * tau) - alpha * z)
    scale = mu.h if mu.integration_rule == "trapezoid" else mu.h / 3.0
    main = prefactor * scale * float(np.dot(u, omega))
    sqrt_tau = math.sqrt(tau)
    u_left = (x - grid[0] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    u_right = (x - grid[-1] + 2.0 * tau * alpha) / (2.0 * sqrt_tau)
    tail_scale = 0.5 * math.exp(tau * (alpha * alpha - beta))
    tail = src[0] * tail_scale * erfc(u_left) + src[-1] * tail_scale * erfc(-u_right)
    return float(main + tail)


def run(label, interp, spots, record=False):
    if record:
        SnowballQuadEngine._diffuse_fft = diffuse_recording
    if interp is not None:
        QuadratureMath.interpolate = interp
    REGISTRY.clear()
    KEEP.clear()
    STATS.update(pointwise=0, fallback=0)
    try:
        product = repro.build(TTM)
        eng = SnowballQuadEngine(
            QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
        )
        errs, px = [], []
        for s in spots:
            s = float(s)
            row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            ref = row["delta"] / M_REF
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            px.append(eng.price(product, repro.env_at(s)))
            errs.append((up - dn) / (2 * s * BUMP) / M_REF - ref)
        errs = np.asarray(errs)
        note = ""
        if record:
            note = f"  [pointwise {STATS['pointwise']}, fallback {STATS['fallback']}]"
        print(f"  {label:<30} ptp {np.ptp(errs):8.4f}  mean|.| "
              f"{np.abs(errs).mean():8.4f}{note}")
        return np.asarray(px)
    finally:
        SnowballQuadEngine._diffuse_fft = _ORIG_DIFFUSE
        QuadratureMath.interpolate = _ORIG_INTERP


def main():
    ki = repro.KI
    for lo, hi, tag in (
        (ki + 240.0, ki + 280.0, "12 to 14 cells above the KI barrier"),
        (ki + 2.0, ki + 42.0, "0.1 to 2.1 cells above"),
    ):
        spots = np.linspace(lo, hi, 42)
        print(tag)
        base = run("linear (current)", None, spots)
        run("cubic readout", cubic_interpolate, spots)
        tr = run("final transition at spot", pointwise, spots, record=True)
        rel = np.abs(tr - base) / np.abs(base)
        print(f"    price move vs current: max {rel.max():.3e} rel "
              f"({rel.max() * 1e4:.3f} bp of price)")
        print()


if __name__ == "__main__":
    main()
