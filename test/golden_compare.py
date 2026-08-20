"""Cross-architecture golden comparison helpers.

The frozen goldens throughout this suite were captured on ONE machine — every
such test's module docstring calls them "same-machine references", not
cross-platform bit claims. Bitwise float equality does not hold across CPU
architectures / BLAS / libm builds: the x86_64 CI runners differ from the ARM64
machine the goldens were frozen on by the last 1-2 ULP (~1e-14 relative) on a
single operation. Measured 2026-08-19 across 326 banked certificate anchors,
an autocallable PDE/QUAD solve marching hundreds of steps through barrier events
amplifies that to a median 2.0e-12 and a worst case 2.2e-11 -- three orders
above the per-operation figure, with no outlier of a different order. The 1e-9
bound below still clears that by ~44x; do not tighten it on the strength of the
per-operation number alone. These
helpers compare a live payload against a frozen golden with a tight relative
tolerance that absorbs that cross-arch ULP noise while still catching any
genuine numerical regression (>~1e-8).

IMPORTANT: same-machine invariants — e.g. two engine schemes that must be
byte-identical when run on the SAME machine — stay exact ``==`` at the call
site. Do NOT route those comparisons through here; only frozen goldens
(captured elsewhere) need the tolerance.
"""
import math

# ~44x headroom over the worst measured cross-arch drift (2.2e-11, iterative
# barrier solves), yet tight enough to fail on a real numerical regression of
# ~1e-8 or larger.
GOLDEN_REL_TOL = 1e-9
# Absorbs cross-arch noise on values whose magnitude is ~0 (survival
# probabilities, deep-OTM greeks), where a relative tolerance has no purchase.
GOLDEN_ABS_TOL = 1e-12


def close(live, golden, *, rel_tol=GOLDEN_REL_TOL, abs_tol=GOLDEN_ABS_TOL) -> bool:
    """Recursively compare a live payload against a frozen golden.

    Floats compare with :func:`math.isclose` (``rel_tol``/``abs_tol``); bools,
    ints, strings and ``None`` require exact equality. dicts must share keys and
    lists/tuples must share length, with each element compared recursively.
    """
    # bool is an int subclass but is a discrete flag -> exact match only.
    if isinstance(live, bool) or isinstance(golden, bool):
        return live == golden
    if isinstance(live, float) or isinstance(golden, float):
        return math.isclose(
            float(live), float(golden), rel_tol=rel_tol, abs_tol=abs_tol
        )
    if isinstance(live, dict) and isinstance(golden, dict):
        if live.keys() != golden.keys():
            return False
        return all(
            close(live[k], golden[k], rel_tol=rel_tol, abs_tol=abs_tol)
            for k in golden
        )
    if isinstance(live, (list, tuple)) and isinstance(golden, (list, tuple)):
        if len(live) != len(golden):
            return False
        return all(
            close(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(live, golden)
        )
    return live == golden


def assert_close(live, golden, *, rel_tol=GOLDEN_REL_TOL, abs_tol=GOLDEN_ABS_TOL,
                 msg: str = "") -> None:
    """Assert ``live`` matches frozen ``golden`` within cross-arch tolerance."""
    assert close(live, golden, rel_tol=rel_tol, abs_tol=abs_tol), (
        f"{msg}\nlive   = {live!r}\ngolden = {golden!r}"
    )
