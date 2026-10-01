"""The problem identity and what moves it (spec 2 §5.9, §9.3, §14)."""
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.grid import GridBinder, GridConfig, GridRequest, MarketSnapshot
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.calendar import CalendarType, create_calendar

from pde_bitwise.capture_book import env_flat, env_grid_vol, env_with_spot

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def _bind(**request):
    binder = GridBinder("standard", GridConfig(time_steps=3))
    market = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)
    base = dict(tau=1.0, bound_anchors=(100.0, 100.0), critical_prices=(100.0,), hard_lower=None, hard_upper=None,
                event_times=())
    base.update(request)
    return binder.bind(GridRequest(**base), market)


def test_a_budget_with_no_node_a_readout_node_and_an_event_node():
    # §14: three requests on one budget; the fill is the same, the problem is not
    plain = _bind()
    readout = _bind(extra_times=(2.0 / 3.0,))
    event = _bind(event_times=(2.0 / 3.0,))
    assert plain.time.t.tobytes() == readout.time.t.tobytes() == event.time.t.tobytes()
    assert len({plain.fingerprint, readout.fingerprint, event.fingerprint}) == 3
    stepper = CrankNicolson().resolved(("full_step", 1, 2))
    assert not np.array_equal(stepper.schedule(readout), stepper.schedule(event))


def test_a_holiday_on_the_product_settlement_calendar_moves_the_identity():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit

    from intraday.conftest import dated_snowball, flat_env

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    t0 = datetime(2026, 3, 16)
    p = dated_snowball(create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028)), t0)
    p.settlement_convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=cal)
    env = flat_env(t0)
    s = SnowballPDESolver(PDEParams())
    key, problem = s._observation_cache_key(env, p, "ko"), s.explain(p, env).identity["problem"]
    cal.holidays.add(datetime(2026, 4, 17))   # inside the schedule, on the product's own calendar
    assert s._observation_cache_key(env, p, "ko") != key
    assert SnowballPDESolver(PDEParams()).explain(p, env).identity["problem"] != problem


def test_frozen_bump_and_fresh_solve_at_the_same_bumped_spot_differ():
    p, env = make_snowball(SB), env_flat()
    bumped = env_with_spot(env, 101.0)
    frozen = SnowballPDESolver(PDEParams()).create_bump_context(p, env).explain(p, bumped).identity
    fresh = SnowballPDESolver(PDEParams()).explain(p, bumped).identity
    for name in ("layout", "coefficient_key", "problem"):
        assert frozen[name] != fresh[name], name


def _lv_env(vol):
    from quantark.param import GridVolSurface

    env = env_grid_vol()
    surf = env.vol_surface
    env.vol_surface = GridVolSurface(list(surf.strikes), list(surf.maturities),
                                     np.full(np.shape(surf.iv_grid), vol))
    return env


def test_prebuilt_local_vol_surfaces_key_the_coefficients():
    p = make_snowball(SB)
    env = env_grid_vol()
    keys = []
    for vol in (0.15, 0.35):
        surface = LocalVolSnowballPDESolver(PDEParams())._build_surface(_lv_env(vol))
        keys.append(LocalVolSnowballPDESolver(PDEParams(), local_vol_surface=surface)
                    .explain(p, env).identity["coefficient_key"])
    assert keys[0] is not None and keys[0] != keys[1]


def test_a_local_vol_bump_instance_carries_the_prebuilt_surface():
    p, env = make_snowball(SB), env_grid_vol()
    surface = LocalVolSnowballPDESolver(PDEParams())._build_surface(env)
    solver = LocalVolSnowballPDESolver(PDEParams(), local_vol_surface=surface)
    clone = solver.create_bump_context(p, env)
    assert clone is not solver and clone._prebuilt is solver._prebuilt


def test_operators_with_equal_evidence_have_different_identities():
    from pde_core.components_fixtures import ShiftedOperator

    p, env = make_snowball(SB), env_flat()
    a, b = ShiftedOperator(0.0), ShiftedOperator(1e-9)
    assert a.evidence() == b.evidence() == {}
    from quantark.asset.equity.engine.pde.core.identity import component_fingerprint

    assert component_fingerprint(a) != component_fingerprint(b)
    ka = SnowballPDESolver(PDEParams(operator=a)).explain(p, env).identity["coefficient_key"]
    kb = SnowballPDESolver(PDEParams(operator=b)).explain(p, env).identity["coefficient_key"]
    assert ka is not None and ka != kb


def test_a_readout_moves_the_problem_but_not_the_factorization():
    from pde_core.components_fixtures import NearestNodeReadout

    p, env = make_snowball(SB), env_flat()
    std = SnowballPDESolver(PDEParams()).explain(p, env).identity
    near = SnowballPDESolver(PDEParams(readout=NearestNodeReadout())).explain(p, env).identity
    assert std["problem"] != near["problem"]
    assert std["factorization"] == near["factorization"] is not None
