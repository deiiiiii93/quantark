"""Benchmark public-package routes for the ADI tridiagonal sweeps.

Real shapes from the production march (near_ko, 300x135):
  S-sweep: 135 DISTINCT systems of size 300 (coefficients vary per V-slice)
  V-sweep: ONE system of size 135, 300 right-hand sides (x-independent operator)

Routes:
  A. current solve_tridiag_batch (pure-NumPy Thomas, Python loop)
  B. block-concatenated scipy.linalg.solve_banded (one LAPACK dgtsv call)
  C. block-concatenated lapack.dgttrf ONCE (cacheable per dt/theta/interval)
     + lapack.dgttrs per solve
  D. per-slice lapack.dgtsv Python loop (naive scipy route, for reference)
  V1. solve_banded multi-RHS for the V-sweep
  V2. cached dgttrf + dgttrs multi-RHS for the V-sweep
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.linalg import lapack

sys.path.insert(0, "/private/tmp/quant-ark-adi-greek-certification")
from quantark.util.numerical.tridiag import solve_tridiag_batch

rng = np.random.default_rng(42)

N_SYS, N = 135, 300          # S-sweep shape
N_RHS_V, N_V = 300, 135      # V-sweep shape
S_SOLVES = 5396              # per near_ko march (measured)
V_SOLVES = 5396

def make_systems(n_sys, n):
    """Diagonally dominant implicit-step-like systems (full-length convention)."""
    sub = -0.4 * rng.random((n_sys, n))
    sup = -0.4 * rng.random((n_sys, n))
    diag = 1.0 + np.abs(sub) + np.abs(sup) + 0.05 * rng.random((n_sys, n))
    sub[:, 0] = 0.0
    sup[:, -1] = 0.0
    rhs = rng.random((n_sys, n))
    return sub, diag, sup, rhs

def concat_bands(sub, diag, sup):
    """Pack n_sys tridiag systems into one block-diagonal banded system."""
    n_sys, n = diag.shape
    d = diag.ravel()
    dl = sub.ravel()[1:]          # sub[:,0]==0 supplies the zero seam entries
    du = sup.ravel()[:-1]         # sup[:,-1]==0 likewise
    return dl, d, du

def bench(fn, repeats=60):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn()
    dt = (time.perf_counter() - t0) / repeats
    return dt, out

sub, diag, sup, rhs = make_systems(N_SYS, N)

# --- A: current implementation
tA, xA = bench(lambda: solve_tridiag_batch(sub, diag, sup, rhs))

# --- B: concatenated solve_banded (packing inside the timed region)
def route_B():
    dl, d, du = concat_bands(sub, diag, sup)
    ab = np.zeros((3, d.size))
    ab[0, 1:] = du
    ab[1, :] = d
    ab[2, :-1] = dl
    return solve_banded((1, 1), ab, rhs.ravel(), check_finite=False).reshape(N_SYS, N)
tB, xB = bench(route_B)

# --- C: concatenated gttrf cached + gttrs per solve
dl0, d0, du0 = concat_bands(sub, diag, sup)
_dl, _d, _du, _du2, _ipiv, info = lapack.dgttrf(dl0, d0, du0)
assert info == 0
def route_C():
    x, info = lapack.dgttrs(_dl, _d, _du, _du2, _ipiv, rhs.ravel())
    return x.reshape(N_SYS, N)
tC, xC = bench(route_C)

# --- D: per-slice dgtsv loop
def route_D():
    out = np.empty_like(rhs)
    for j in range(N_SYS):
        _, _, _, x, info = lapack.dgtsv(sub[j, 1:], diag[j], sup[j, :-1], rhs[j])
        out[j] = x
    return out
tD, xD = bench(route_D, repeats=20)

print(f"S-sweep: {N_SYS} distinct systems x N={N}   ({S_SOLVES} solves per march)")
print(f"  A  solve_tridiag_batch (current)      {tA*1e3:7.3f} ms   -> {tA*S_SOLVES:6.2f} s/march")
print(f"  B  concat + solve_banded (1 dgtsv)    {tB*1e3:7.3f} ms   -> {tB*S_SOLVES:6.2f} s/march   maxdiff={np.max(np.abs(xB-xA)):.2e}")
print(f"  C  concat + cached gttrf, gttrs/step  {tC*1e3:7.3f} ms   -> {tC*S_SOLVES:6.2f} s/march   maxdiff={np.max(np.abs(xC-xA)):.2e}")
print(f"  D  per-slice dgtsv python loop        {tD*1e3:7.3f} ms   -> {tD*S_SOLVES:6.2f} s/march   maxdiff={np.max(np.abs(xD-xA)):.2e}")

# --- V-sweep: one matrix, many RHS
subv, diagv, supv, _ = make_systems(1, N_V)
a1, b1, c1 = subv[0], diagv[0], supv[0]
rhsV = rng.random((N_RHS_V, N_V))

tVA, xVA = bench(lambda: solve_tridiag_batch(
    np.broadcast_to(a1, (N_RHS_V, N_V)), np.broadcast_to(b1, (N_RHS_V, N_V)),
    np.broadcast_to(c1, (N_RHS_V, N_V)), rhsV))

def route_V1():
    ab = np.zeros((3, N_V))
    ab[0, 1:] = c1[:-1]
    ab[1, :] = b1
    ab[2, :-1] = a1[1:]
    return solve_banded((1, 1), ab, rhsV.T, check_finite=False).T
tV1, xV1 = bench(route_V1)

_dlv, _dv, _duv, _du2v, _ipivv, info = lapack.dgttrf(a1[1:], b1, c1[:-1])
assert info == 0
def route_V2():
    x, info = lapack.dgttrs(_dlv, _dv, _duv, _du2v, _ipivv, rhsV.T)
    return x.T
tV2, xV2 = bench(route_V2)

print(f"\nV-sweep: 1 system x N={N_V}, {N_RHS_V} RHS   ({V_SOLVES} solves per march)")
print(f"  A  solve_tridiag_batch (current)      {tVA*1e3:7.3f} ms   -> {tVA*V_SOLVES:6.2f} s/march")
print(f"  V1 solve_banded multi-RHS             {tV1*1e3:7.3f} ms   -> {tV1*V_SOLVES:6.2f} s/march   maxdiff={np.max(np.abs(xV1-xVA)):.2e}")
print(f"  V2 cached gttrf + gttrs multi-RHS     {tV2*1e3:7.3f} ms   -> {tV2*V_SOLVES:6.2f} s/march   maxdiff={np.max(np.abs(xV2-xVA)):.2e}")

best_S = min(tB, tC)
best_V = min(tV1, tV2)
now = tA * S_SOLVES + tVA * V_SOLVES
after = best_S * S_SOLVES + best_V * V_SOLVES
print(f"\ntridiag total per march:  now {now:.1f} s  ->  scipy routes {after:.1f} s"
      f"   (march was 19.5 s total; other {19.5-now:.1f} s unchanged)")
print(f"projected march: {19.5 - now + after:.1f} s  ({19.5/(19.5-now+after):.1f}x overall)")
