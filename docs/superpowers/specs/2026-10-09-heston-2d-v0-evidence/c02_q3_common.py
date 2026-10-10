"""Shared probes for the Q3 diagnosis (C02-C05): the reverse continuous-KI Heston snowball KO 97 / KI 125 (a02).

Runtime patches, each restorable:
* ``correction_off()``  -- the 2D continuous-KI crossing correction (FIRST_PASSAGE, ``_apply_continuous_ki_correction``)
  returns U unchanged; the nodal mask still applies.
* ``barrier_cell_floor()`` -- the half-cell lambda floor of ``_ContinuousKIFirstPassage._geometry`` measured on the
  cell AT THE BARRIER on both orientations. The library measures ``a[live[1]] - a[live[0]]``: the two live nodes of
  lowest index. On a standard (down) product they are the two nearest the barrier; on a reverse (up) product the
  live side is BELOW the barrier, so they are the two lowest-spot nodes, ``a`` decreases with the index, the
  difference is negative and the floor never removes anything. The patch sorts the live nodes by ``a`` first, so on
  a standard product it takes the very same two nodes (bit-identical by construction).
"""
import contextlib
import math

import numpy as np
from scipy.special import ndtr

from common import HP, T, env, snowball, solver
from quantark.asset.equity.engine.pde import snowball_pde_solver as sps
from quantark.asset.equity.engine.pde.vol_continuous_ki import Heston2DBarrierCrossingMixin
from quantark.volmodels.heston import HestonParams

REV_C = dict(ko=97.0, ki=125.0, continuous=True, rebate_rate=0.15, reverse=True)
REV_D = dict(REV_C, continuous=False)
STD_C = dict(ko=103.0, ki=75.0, continuous=True, rebate_rate=0.15)
BSM_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=-0.5)

_ORIG_CORR = Heston2DBarrierCrossingMixin._apply_continuous_ki_correction
_ORIG_GEOM = sps._ContinuousKIFirstPassage._geometry
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _geometry_barrier_cell(self, dt, mu, sig2, s_vec, barrier):
    """``_ContinuousKIFirstPassage._geometry`` with the half-cell floor measured at the barrier (one line differs)."""
    key = (round(dt, 12), round(mu, 12), round(sig2, 12), round(float(barrier), 12), len(s_vec))
    cached = self._cache.get(key)
    if cached is not None:
        return cached
    result = (None, None, None)
    if dt > 0.0 and sig2 > 0.0 and barrier > 0.0:
        log_s = np.log(np.asarray(s_vec, dtype=float))
        log_b = np.log(float(barrier))
        if self._is_reverse:
            a = log_b - log_s
            mu_e = -mu
        else:
            a = log_s - log_b
            mu_e = mu
        s_step = np.sqrt(sig2 * dt)
        band = np.flatnonzero((a > 0.0) & (a < self._BAND_STDS * s_step))
        if band.size:
            ab = a[band]
            m = mu_e * dt - ab
            t = m / s_step
            bracket = m * ndtr(t) + s_step * np.exp(-0.5 * t * t) / _SQRT_2PI
            G = np.zeros(len(s_vec), dtype=float)
            G[band] = np.exp(-2.0 * mu_e * ab / sig2) * bracket
            live = np.flatnonzero(a > 0.0)
            if live.size > 1:
                near = live[np.argsort(a[live], kind="stable")]          # nearest the barrier first
                resolved = live[a[live] >= 0.5 * (a[near[1]] - a[near[0]])]   # THE CHANGED LINE
                if resolved.size:
                    live = resolved
            k_star = int(live[np.argmin(np.abs(a[live] - s_step))])
            result = (G, k_star, float(a[k_star]))
    self._cache[key] = result
    return result


@contextlib.contextmanager
def correction_off():
    Heston2DBarrierCrossingMixin._apply_continuous_ki_correction = lambda self, U, core, tau, barrier, v1: U
    try:
        yield
    finally:
        Heston2DBarrierCrossingMixin._apply_continuous_ki_correction = _ORIG_CORR


@contextlib.contextmanager
def barrier_cell_floor():
    sps._ContinuousKIFirstPassage._geometry = _geometry_barrier_cell
    try:
        yield
    finally:
        sps._ContinuousKIFirstPassage._geometry = _ORIG_GEOM


def px(n, hp=HP, spec=REV_C, **kw):
    try:
        v = solver(hp, *n, **kw).price(snowball(**spec), env())
    except Exception as exc:  # noqa: BLE001 - probe
        return f"{type(exc).__name__}"
    return f"{v:.6f}" if math.isfinite(v) and abs(v) < 1e6 else f"DIVERGED {v:.3e}"


def barrier_geometry(n, hp=HP, spec=REV_C, **kw):
    """The S grid around the KI barrier: the live node nearest the barrier, its distance in cells, the cell the
    library's floor measures (negative = no floor) and the k* the library picks on the v0 column."""
    s = solver(hp, *n, **kw)
    p = snowball(**spec)
    s._prepare_state(p, env(), T, True)
    core = s._make_core(p, env(), T)
    S = np.asarray(core.S_grid, dtype=float)
    b = float(spec["ki"])
    reverse = bool(spec.get("reverse"))
    a = (math.log(b) - np.log(S)) if reverse else (np.log(S) - math.log(b))
    live = np.flatnonzero(a > 0.0)
    near = live[np.argsort(a[live])]
    cell_at_barrier = float(a[near[1]] - a[near[0]])
    lib_cell = float(a[live[1]] - a[live[0]])
    return dict(nearest=float(S[near[0]]), a_nearest=float(a[near[0]]), cells=float(a[near[0]] / cell_at_barrier),
                cell=cell_at_barrier, lib_cell=lib_cell, core=core, S=S, a=a)
