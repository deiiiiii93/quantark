"""Canonical value fingerprints (spec section 10.1)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.execution.cache.fingerprint import (
    Uncanonicalizable,
    fingerprint,
    try_fingerprint,
)


def test_equal_valued_dataclasses_share_fingerprint():
    from quantark.param import FlatRateCurve
    from quantark.param.div import ContinuousDividendYield

    assert fingerprint(FlatRateCurve(rate=0.05)) == fingerprint(
        FlatRateCurve(rate=0.05)
    )
    assert fingerprint(FlatRateCurve(rate=0.05)) != fingerprint(
        FlatRateCurve(rate=0.051)
    )
    # class identity participates: same single-float field value, different type
    assert fingerprint(FlatRateCurve(rate=0.05)) != fingerprint(
        ContinuousDividendYield(div_yield=0.05)
    )


def test_grid_surface_and_composites():
    from quantark.param import GridVolSurface

    def grid():
        return GridVolSurface(
            strikes=[90.0, 100.0, 110.0],
            maturities=[0.5, 1.0],
            iv_grid=np.full((2, 3), 0.2),
        )

    assert fingerprint(grid()) == fingerprint(grid())
    changed = grid()
    changed.iv_grid = np.full((2, 3), 0.21)
    assert fingerprint(grid()) != fingerprint(changed)
    assert fingerprint((grid(), datetime(2026, 1, 1))) == fingerprint(
        (grid(), datetime(2026, 1, 1))
    )


def test_unregistered_type_is_uncanonicalizable():
    class Opaque:
        pass

    with pytest.raises(Uncanonicalizable):
        fingerprint(Opaque())
    assert try_fingerprint(Opaque()) is None
    assert try_fingerprint({"k": Opaque()}) is None


def test_float_precision_and_nan_distinct():
    assert fingerprint(0.1) != fingerprint(0.1 + 1e-17) or (0.1 == 0.1 + 1e-17)
    assert fingerprint(float("nan")) == fingerprint(float("nan"))
    assert fingerprint(1.0) != fingerprint(1)  # float vs int tagged apart
