import pytest

from quantark.asset.equity.report.term_structure import (
    BucketedDividendYield,
    BucketedVolSurface,
    ScaledVolSurface,
    ShiftedDividendYield,
)
from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield
from quantark.param.vol import FlatVolSurface, TermStructureVolSurface


def test_bucketed_vol_surface_bump_only_in_bucket():
    base = FlatVolSurface(volatility=0.2)
    bucket = BucketedVolSurface(base=base, bucket_start=0.0, bucket_end=0.5, bump=0.05)

    assert bucket.get_vol(100.0, 0.25, 100.0) == pytest.approx(0.25)
    assert bucket.get_vol(100.0, 0.75, 100.0) == pytest.approx(0.2)


def test_bucketed_dividend_yield_allows_signed_carry():
    base = ContinuousDividendYield(div_yield=0.01)
    bucket = BucketedDividendYield(base=base, bucket_start=0.0, bucket_end=0.5, bump=-0.02)

    assert bucket.get_yield(0.25) == pytest.approx(-0.01)  # no zero floor
    assert bucket.get_yield(0.75) == pytest.approx(0.01)


def test_scaled_vol_surface_scales_term_structure():
    base = TermStructureVolSurface(times=[0.5, 1.0], vols=[0.2, 0.3])
    scaled = ScaledVolSurface(base=base, scale=1.1)

    assert scaled.get_vol(100.0, 0.75, 100.0) == pytest.approx(0.2978814082, rel=1e-6)


def test_shifted_dividend_yield_applies_parallel_shift():
    base = TermStructureDividendYield(times=[0.5, 1.0], yields=[0.02, 0.03])
    shifted = ShiftedDividendYield(base=base, shift=0.01)
    shifted_down = ShiftedDividendYield(base=base, shift=-0.05)

    assert shifted.get_yield(0.75) == pytest.approx(0.035)
    assert shifted_down.get_yield(0.75) == pytest.approx(-0.025)  # no zero floor
