"""Is option (d) constructible, and how much does it disturb the envelope?

Instead of SHIFTING the grid onto the barrier (which un-pins it from spot),
STRETCH the extent slightly so the barrier lands on the lattice that already
contains spot.  With an odd node count and no shift, log-moneyness 0 is the
centre node exactly, so the readout needs no interpolation at all.

Requirement: log(B/S) = k * h with h = 2*log_c/(n-1), i.e.
   (n-1) * |log(B/S)| / (2*log_c) = k, an integer.
n is already free above the adaptive floor, and log_c is a heuristic
envelope, so search n for a near-integer and absorb the residual into log_c.
"""
from __future__ import annotations
import math
import numpy as np

S0, KI = 6734.0, 0.75 * 6734.0
VOL, TTM, NUM_STD = 0.1774233515693156, 0.295890, 10.0


def log_c_of(vol, ttm, num_std):
    return num_std * vol * math.sqrt(ttm) + (1.0 + 0.5 * vol * vol) * ttm


def main():
    print("centre node exactness, odd linspace:")
    for n in (401, 683, 1581):
        g = np.linspace(-1.2656, 1.2656, n)
        print(f"  n={n:5d}  grid[(n-1)//2] = {g[(n - 1) // 2]:.3e}")

    log_c = log_c_of(VOL, TTM, NUM_STD)
    print(f"\nlog_c = {log_c:.8f}  (num_std_devs = {NUM_STD})")
    print(f"{'spot':>9} {'log(B/S)':>11} {'best n':>7} {'k':>5} "
          f"{'|k - exact|':>12} {'log_c stretch':>14} {'equiv num_std':>14}")
    for spot in (5306.986, 5250.0, 5400.0, 5600.0, 6000.0):
        target = abs(math.log(KI / spot))
        best = None
        for n in range(683, 683 + 400, 2):  # odd counts at or above the floor
            exact = (n - 1) * target / (2.0 * log_c)
            k = round(exact)
            if k <= 0:
                continue
            err = abs(exact - k)
            if best is None or err < best[0]:
                best = (err, n, k, exact)
        err, n, k, exact = best
        # absorb the residual into the envelope
        log_c_new = (n - 1) * target / (2.0 * k)
        stretch = log_c_new / log_c - 1.0
        equiv_std = (log_c_new - (1.0 + 0.5 * VOL * VOL) * TTM) / (
            VOL * math.sqrt(TTM)
        )
        print(f"{spot:9.2f} {math.log(KI / spot):11.6f} {n:7d} {k:5d} "
              f"{err:12.2e} {stretch:+14.3e} {equiv_std:14.4f}")


if __name__ == "__main__":
    main()
