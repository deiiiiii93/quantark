"""Two-surface snowball PDE march: fully implicit on zero-diffusion steps; legacy prices unchanged."""
from datetime import datetime

import pytest
from golden_compare import GOLDEN_REL_TOL

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

from test_snowball_pde import create_pricing_env, create_reverse_snowball, create_standard_snowball


def test_legacy_flat_surface_prices_are_unchanged():
    # pinned before the zero-diffusion theta fix: flat surfaces have no zero-diffusion sets
    # (frozen on the banking machine; a PDE march drifts by ULPs across architectures)
    frozen = lambda bits: pytest.approx(float.fromhex(bits), rel=GOLDEN_REL_TOL)
    assert SnowballPDESolver(PDEParams()).price(create_standard_snowball(), create_pricing_env()) == frozen("0x1.deda181017793p+19")
    assert SnowballPDESolver(PDEParams()).price(create_reverse_snowball(), create_pricing_env()) == frozen("0x1.cce78e3a0e01bp+19")


def _sessions_only_lunch_context(sse_calendar, sse_sessions):
    prod = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    profile = VarianceProfile.sessions_only(sse_sessions, 244)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    return resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 15, 12, 0, tzinfo=SHANGHAI)),
                                                    session_calendar=sse_sessions, variance_profile=profile,
                                                    fixings=tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])))


def test_zero_diffusion_steps_march_fully_implicit(sse_calendar, sse_sessions, monkeypatch):
    ctx = _sessions_only_lunch_context(sse_calendar, sse_sessions)
    solver = SnowballPDESolver(PDEParams())
    captured, zero_sets = [], {}
    march = solver._time_stepping_two_surface
    banded, matrices = solver._get_banded_system, solver._get_matrices

    def spy_march(*args, **kwargs):
        zero_sets["sets"] = kwargs["step_coeffs"].zero_diffusion_sets
        return march(*args, **kwargs)

    def spy_banded(l, c, u, dt, theta, coeff_key=0):
        captured.append((coeff_key, theta))
        return banded(l, c, u, dt, theta, coeff_key=coeff_key)

    def spy_matrices(I_int, A, dt, theta, coeff_key=0):
        captured.append((coeff_key, theta))
        return matrices(I_int, A, dt, theta, coeff_key=coeff_key)

    monkeypatch.setattr(solver, "_time_stepping_two_surface", spy_march)
    monkeypatch.setattr(solver, "_get_banded_system", spy_banded)
    monkeypatch.setattr(solver, "_get_matrices", spy_matrices)
    price = solver.price(ctx.numerical.product, ctx.pricing_env)
    zero = zero_sets["sets"]
    on_zero = [theta for key, theta in captured if key in zero]
    assert zero and on_zero, "the sessions-only profile must produce zero-diffusion steps"
    assert all(theta == 1.0 for theta in on_zero), sorted(set(on_zero))
    assert any(theta != 1.0 for key, theta in captured if key not in zero)       # diffusive steps keep their schedule
    assert 0.0 < price < 20.0
