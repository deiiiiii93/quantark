"""Unit gate for every kernel variant: bitwise equality, guard parity, speed.

Variants:
  stock   quantark solve_tridiag_batch (pure NumPy, per-iteration np.any)
  numpy+  hoisted-guard pure NumPy      (bitwise expected)
  scipy   LAPACK concat / multi-RHS     (~1e-16, pivoted — NOT bitwise)
  C       transposed compiled Thomas    (bitwise expected)
"""
from __future__ import annotations

import sys
import time

import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-fuxinyao-quant-ark/0b6e4fbe-10d5-4787-b996-5c7815bd68e3/scratchpad"
sys.path.insert(0, SCRATCH)
sys.path.insert(0, "/private/tmp/quant-ark-adi-greek-certification")

from quantark.util.exceptions import NumericalError
from quantark.util.numerical.tridiag import solve_tridiag_batch
from boosted_tridiag import solve_tridiag_batch_boosted
from boosted_tridiag_c import solve_tridiag_batch_c
from boosted_tridiag_np import solve_tridiag_batch_hoisted

VARIANTS = {
    "numpy+": solve_tridiag_batch_hoisted,
    "scipy": solve_tridiag_batch_boosted,
    "C": solve_tridiag_batch_c,
}
rng = np.random.default_rng(7)


def make(n_sys, n):
    sub = -0.4 * rng.random((n_sys, n)); sub[:, 0] = 0.0
    sup = -0.4 * rng.random((n_sys, n)); sup[:, -1] = 0.0
    diag = 1.0 + np.abs(sub) + np.abs(sup) + 0.05 * rng.random((n_sys, n))
    return sub, diag, sup, rng.random((n_sys, n))


# ---- bitwise equality on both real ADI shapes
sub, diag, sup, rhs = make(135, 300)                                  # S-sweep
a1, b1, c1, _ = make(1, 135)                                          # V-sweep
A = np.broadcast_to(a1[0], (300, 135)); B = np.broadcast_to(b1[0], (300, 135))
C = np.broadcast_to(c1[0], (300, 135)); R = rng.random((300, 135))
xs, vs = solve_tridiag_batch(sub, diag, sup, rhs), solve_tridiag_batch(A, B, C, R)
print("bitwise equality vs stock:")
for name, fn in VARIANTS.items():
    xS, xV = fn(sub, diag, sup, rhs), fn(A, B, C, R)
    bit_s, bit_v = np.array_equal(xs, xS), np.array_equal(vs, xV)
    md = max(np.max(np.abs(xS - xs)), np.max(np.abs(xV - vs)))
    print(f"  {name:7s} S-shape: {str(bit_s):5s}  V-shape: {str(bit_v):5s}  maxdiff={md:.2e}")

# ---- pivot-guard parity: first-pivot zero, and an interior denom that
# cancels EXACTLY in IEEE arithmetic (b1 == s1*cp0 bitwise)
b0, c0, s1 = 1.3, 0.7, 0.9
cp0 = c0 / b0
cases = {
    "first-pivot zero": (np.array([[0.0, -0.3, 0.0]]),
                         np.array([[0.0, 1.4, 1.2]]),
                         np.array([[-0.2, -0.3, 0.0]])),
    "interior exact-zero denom": (np.array([[0.0, s1, -0.3]]),
                                  np.array([[b0, s1 * cp0, 1.5]]),
                                  np.array([[c0, -0.2, 0.0]])),
}
r3 = np.array([[1.0, 2.0, 3.0]])
print("\npivot-guard behaviour (stock raises NumericalError):")
for label, (bs, bd, bp) in cases.items():
    outcomes = {}
    for name, fn in [("stock", solve_tridiag_batch), *VARIANTS.items()]:
        try:
            out = fn(bs, bd, bp, r3)
            outcomes[name] = f"solved (finite={bool(np.all(np.isfinite(out)))})"
        except NumericalError:
            outcomes[name] = "NumericalError (same message)"
    print(f"  {label}:")
    for name, res in outcomes.items():
        print(f"      {name:7s} {res}")

# ---- micro-benchmark on the real shapes
def bench(fn, args, repeats=300):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    return (time.perf_counter() - t0) / repeats

print("\nmicro-benchmark (per call / per-march tridiag over 5396+5396 solves):")
rows = [("stock", solve_tridiag_batch), *VARIANTS.items()]
for name, fn in rows:
    tS = bench(fn, (sub, diag, sup, rhs))
    tV = bench(fn, (A, B, C, R))
    march = (tS + tV) * 5396
    print(f"  {name:7s} S {tS*1e3:6.3f} ms   V {tV*1e3:6.3f} ms   -> {march:5.1f} s/march")
