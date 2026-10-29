"""Andreasen–Huge local-vol construction: calibration, lazy reference object, sampled view."""
import math
from pathlib import Path

import numpy as np
import pytest

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.localvol.andreasen_huge import (
    AndreasenHugeConfig,
    AndreasenHugeLocalVol,
    calibrate_andreasen_huge,
)

FIXTURE = Path(__file__).resolve().parents[1] / "golden/modelvalidation/data/iv_surface_20240208.json"
R = 0.02


def _fixture():
    art = IvSurfaceArtifact.from_file(FIXTURE)
    q = float(art.implied_q_pillars(R)[1][-1])
    return art.grid_vol_surface(), float(art.s0), q


@pytest.fixture(scope="module")
def fixture_ah():
    grid, spot, q = _fixture()
    return calibrate_andreasen_huge(grid, spot, FlatRateCurve(R), lambda t: q), grid, spot, q


def test_fixture_calibration_is_exact_on_both_sides(fixture_ah):
    ah, grid, _, _ = fixture_ah
    assert isinstance(ah, AndreasenHugeLocalVol)
    assert len(ah.report) == 2 * len(grid.maturities)
    assert {r.side for r in ah.report} == {"call", "put"}
    for rec in ah.report:
        assert rec.worst_vol_error <= 1e-10, rec
        assert rec.excluded == 0 and rec.quotes == len(grid.strikes)


def test_discrete_model_reprices_market_calls_and_puts(fixture_ah):
    ah, grid, _, _ = fixture_ah
    worst = ah.discrete_repricing_errors()
    assert worst.shape == (len(grid.maturities), 2)
    assert np.max(worst) <= 1e-10


def _flat_on_fixture(vol=0.25):
    grid, spot, q = _fixture()
    flat = GridVolSurface(list(grid.strikes), list(grid.maturities), np.full(np.shape(grid.iv_grid), vol))
    return flat, spot, q


def test_default_raises_on_unidentifiable_quotes_naming_the_expiry():
    flat, spot, q = _flat_on_fixture()
    t0 = float(flat.maturities[0])
    with pytest.raises(NumericalError, match=f"expiry {t0:.6f}"):
        calibrate_andreasen_huge(flat, spot, FlatRateCurve(R), lambda t: q)


def test_quote_rule_is_opt_in_and_reported():
    flat, spot, q = _flat_on_fixture()
    ah = calibrate_andreasen_huge(flat, spot, FlatRateCurve(R), lambda t: q,
                                  AndreasenHugeConfig(max_abs_d=4.0))
    first = [r for r in ah.report if math.isclose(r.expiry, float(flat.maturities[0]))]
    assert first and all(r.excluded > 0 for r in first)
    assert all(r.worst_vol_error <= 1e-10 for r in ah.report)
    later = [r for r in ah.report if r.expiry > 0.5]
    assert all(r.excluded == 0 for r in later)


def test_local_vol_beyond_last_expiry_is_flat_in_time(fixture_ah):
    ah, grid, spot, _ = fixture_ah
    t_last = float(grid.maturities[-1])
    s = spot * np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    assert np.array_equal(ah.local_vol(s, t_last + 0.5), ah.local_vol(s, t_last))
    assert np.array_equal(ah.local_vol(s, t_last + 3.0), ah.local_vol(s, t_last))


def test_local_vol_at_time_zero_is_the_first_theta_of_the_otm_side():
    grid, spot, _ = _fixture()
    ah = calibrate_andreasen_huge(grid, spot, FlatRateCurve(0.0), lambda t: 0.0)    # flat zero carry: F = S0
    s = np.asarray(grid.strikes, dtype=float)[[2, 10, 20, 22, 23, 26, 30]]         # quoted strikes are grid nodes
    k_put, th_put = ah.theta_knots(0, "put")
    k_call, th_call = ah.theta_knots(0, "call")
    expect = np.where(s < spot, np.interp(np.log(s), np.log(k_put), th_put),
                      np.interp(np.log(s), np.log(k_call), th_call))
    assert np.allclose(ah.local_vol(s, 0.0), expect, rtol=1e-12, atol=0.0)


def test_local_vol_accepts_array_times(fixture_ah):
    ah, grid, spot, _ = fixture_ah
    t = np.array([0.01, 0.05, 0.3])
    s = np.full(3, spot)
    got = ah.local_vol(s, t)
    assert got.shape == (3,)
    for i in range(3):
        assert got[i] == ah.local_vol(spot, float(t[i]))


def _term_structure():
    grid, spot, _ = _fixture()
    pillars = [0.02, 0.25, 0.5, 1.0]
    rate = LinearRateCurve(list(zip(pillars, [0.01, 0.02, 0.025, 0.03])))
    div = TermStructureDividendYield(times=pillars, yields=[0.09, 0.07, 0.06, 0.05])
    return grid, spot, rate, div


def test_forward_is_the_curve_forward_at_every_expiry():
    grid, spot, rate, div = _term_structure()
    ah = calibrate_andreasen_huge(grid, spot, rate, div.get_yield)
    for t in grid.maturities:
        t = float(t)
        expect = spot * math.exp((rate.get_rate(t) - div.get_yield(t)) * t)
        assert ah.forward(t) == pytest.approx(expect, rel=1e-12)


def test_engine_facing_map_uses_forward_moneyness():
    """D3b: sigma(t, S) = sigma_AH(t, S * F_AH(t) / F_env(t)) — mid-segment the two forwards differ."""
    grid, spot, rate, div = _term_structure()
    ah = calibrate_andreasen_huge(grid, spot, rate, div.get_yield)
    t0, t1 = float(grid.maturities[2]), float(grid.maturities[3])
    t = 0.5 * (t0 + t1)
    f_env = spot * math.exp((rate.get_rate(t) - div.get_yield(t)) * t)
    assert not math.isclose(ah.forward(t), f_env, rel_tol=1e-6)          # the curves bow mid-segment
    s = spot * np.array([0.85, 0.95, 1.0, 1.08])
    assert np.allclose(ah.local_vol(s, t), ah._calibration_local_vol(s * ah.forward(t) / f_env, t),
                       rtol=1e-14, atol=0.0)


def test_cell_check_rejects_grids_too_coarse_for_the_drift():
    grid, spot, _ = _fixture()
    with pytest.raises(ValidationError, match="strike gap"):
        calibrate_andreasen_huge(grid, spot, FlatRateCurve(0.02), lambda t: -0.6,
                                 AndreasenHugeConfig(cells_per_gap=1, theta_bounds=(0.05, 5.0)))


@pytest.mark.parametrize("bad", [
    dict(strikes=[100.0], maturities=[0.5], iv=[[0.2]]),
    dict(strikes=[100.0, 90.0], maturities=[0.5], iv=[[0.2, 0.2]]),
    dict(strikes=[90.0, 100.0], maturities=[0.5], iv=[[0.2, -0.1]]),
])
def test_input_validation(bad):
    surface = GridVolSurface(bad["strikes"], bad["maturities"], np.array(bad["iv"], dtype=float)) \
        if sorted(bad["strikes"]) == bad["strikes"] and min(np.ravel(bad["iv"])) > 0 and len(bad["strikes"]) > 1 \
        else _RawGrid(bad)
    with pytest.raises(ValidationError):
        calibrate_andreasen_huge(surface, 100.0, FlatRateCurve(R), lambda t: 0.0)


class _RawGrid:
    """Duck-typed grid that bypasses GridVolSurface's own validation, so the builder's checks are tested."""

    def __init__(self, d):
        self.strikes = d["strikes"]
        self.maturities = d["maturities"]
        self.iv_grid = np.array(d["iv"], dtype=float)


@pytest.mark.parametrize("kw", [
    dict(cells_per_gap=0), dict(rows_per_segment=0), dict(domain_log_width=0.0),
    dict(theta_bounds=(0.5, 0.1)), dict(exact_tol=0.0), dict(max_abs_d=-1.0),
    dict(density_floor=1.0), dict(jump_width=0.0), dict(theta_bounds=(0.0, 1.0)),
])
def test_config_validation(kw):
    with pytest.raises(ValidationError):
        AndreasenHugeConfig(**kw)
