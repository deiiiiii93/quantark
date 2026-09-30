"""prepare → march → read for the base family; the knock-in composite (spec 2 §5.1-§5.3, decision 21)."""
from quantark.asset.equity.engine.pde import BarrierPDESolver, EuropeanPDESolver
from quantark.asset.equity.engine.pde.core.prepare import (CompositeProblem, Operation, OutputRequest, prepare,
                                                          run)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
from quantark.util.enum import BarrierType, ObservationType, OptionType

from pde_bitwise.capture_book import env_flat, expired_barrier


def replace_barrier_type(p, barrier_type):
    return BarrierOption(strike=p.strike, maturity=p.maturity, option_type=p.option_type, barrier=p.barrier,
                         barrier_type=barrier_type, observation_type=p.observation_type)


def _up_in_call():
    return BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=120.0,
                         barrier_type=BarrierType.UP_IN, observation_type=ObservationType.CONTINUOUS)


def test_knock_in_prepares_two_problems_on_their_own_domains():
    comp = prepare(BarrierPDESolver(PDEParams()), _up_in_call(), env_flat(), OutputRequest(Operation.PRICE))
    assert isinstance(comp, CompositeProblem) and len(comp.parts) == 2
    vanilla, ko = comp.parts
    assert vanilla.layout.spatial.bounds[1] > 200.0          # automatic envelope (measured 222.554…)
    assert ko.layout.spatial.bounds[1] == 120.0              # the barrier's hard edge


def test_knock_in_composition_is_vanilla_minus_knock_out_for_price_and_greeks():
    ki = _up_in_call()
    env = env_flat()
    ko = replace_barrier_type(ki, BarrierType.UP_OUT)
    van = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g_ki = BarrierPDESolver(PDEParams()).calculate_greeks(ki, env)
    g_ko = BarrierPDESolver(PDEParams()).calculate_greeks(ko, env)
    g_v = EuropeanPDESolver(PDEParams()).calculate_greeks(van, env)
    for k in ("price", "delta", "gamma"):
        assert g_ki[k] == ki.participation_rate * g_v[k] - g_ko[k]


def test_expired_product_has_a_terminal_plan_and_no_layout():
    # The plan's dated BarrierOption valued after its exercise date raises in
    # get_maturity on every operation (Task 1 ruling); a tenor product aged to
    # maturity 0 reaches the short-circuit.
    p = expired_barrier()
    env = env_flat()
    prep = prepare(BarrierPDESolver(PDEParams()), p, env, OutputRequest(Operation.PRICE))
    assert prep.terminal is not None and prep.layout is None
    assert run(prep) == BarrierPDESolver(PDEParams()).price(p, env)


# --- explain (Task 11; spec 2 §4.7, §9.3) ----------------------------------------
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.execution import PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.exceptions import PricingError, ValidationError

from pde_bitwise.capture_book import (env_after_maturity, expired_snowball, immediate_ko_env,
                                      immediate_ko_snowball)

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def test_explain_reports_the_cap_on_a_five_year_standard_snowball():
    p = make_snowball(dict(SB, months=60, maturity=5.0))
    plan = SnowballPDESolver(PDEParams()).explain(p, env_flat())
    # measured on the pre-branch base too: 5040 requested, scaled under max_steps=5000 to 4980
    assert (plan.time.requested_steps, plan.time.actual_steps, plan.time.fill_scaled) == (5040, 4980, True)
    assert plan.time.actual_steps <= plan.settings["grid.max_steps"]["effective"]


def test_explain_reports_the_cap_on_a_three_year_high_snowball():
    p = make_snowball(dict(SB, months=36, maturity=3.0))
    plan = SnowballPDESolver(PDEParams(accuracy="high")).explain(p, env_flat())
    assert (plan.time.requested_steps, plan.time.actual_steps) == (6048, 4968)   # measured, base too


def test_explain_raises_where_price_would():
    bad = make_snowball(SB)
    s = SnowballPDESolver(PDEParams(grid=GridConfig(points=3, eps_crit=1.0)))   # measured: price raises
    with pytest.raises(ValidationError):
        s.price(bad, env_flat())
    with pytest.raises(ValidationError):
        s.explain(bad, env_flat())


def test_terminal_plan_on_an_expired_and_a_knocked_out_product():
    product = expired_snowball()
    plan = SnowballPDESolver(PDEParams()).explain(product, env_after_maturity(product))
    assert plan.layout_bound is False
    assert plan.terminal == {"price": "expired", "greeks": "expired", "price_with_events": "expired",
                             "event_stats": "expired", "life_surface": "raises ValidationError"}
    plan = SnowballPDESolver(PDEParams()).explain(immediate_ko_snowball(), immediate_ko_env())
    assert plan.terminal["price"] == "knocked_out_at_valuation"
    assert plan.terminal["event_stats"] == "marches"          # decision 16: no short-circuit
    assert plan.terminal["life_surface"] == "raises ValidationError"


def test_a_marched_plan_has_every_section():
    plan = SnowballPDESolver(PDEParams()).explain(make_snowball(SB), env_flat())
    assert plan.layout_bound is True and plan.parts == ()
    for name in ("route", "settings", "space", "time", "events", "scheme", "outputs", "identity", "terminal"):
        assert name in plan.to_dict()
    assert plan.identity["fingerprintable"] is True
    assert "theta_by_step" in plan.to_dict()["scheme"] and "theta_by_step" not in plan.scheme
    assert "space" in str(plan) and "identity" in str(plan)


def test_a_knock_in_plan_has_its_parts():
    plan = BarrierPDESolver(PDEParams()).explain(_up_in_call(), env_flat())
    assert len(plan.parts) == 2
    assert plan.route["composition"] == "knock-in = participation × vanilla − knock-out"


def _record_problem_identities(monkeypatch):
    import quantark.asset.equity.engine.pde.core.prepare as prep_mod
    seen, real = [], prep_mod.march

    def spy(prepared):
        seen.append(prepared.identity["problem"])
        return real(prepared)

    monkeypatch.setattr(prep_mod, "march", spy)
    return seen


def test_every_entry_point_reaches_the_explained_problem_identity(monkeypatch):
    p, env = make_snowball(SB), env_flat()
    want = SnowballPDESolver(PDEParams()).explain(p, env).identity["problem"]
    assert want is not None
    seen = _record_problem_identities(monkeypatch)
    SnowballPDESolver(PDEParams()).price(p, env)                                   # seen[0]
    SnowballPDESolver(PDEParams()).create_bump_context(p, env).price(p, env)       # seen[1]: same spot, same layout
    with PricingSession() as session:                                              # seen[2]: session execute
        session.execute(SnowballPDESolver(PDEParams()), PricingRequest(product=p, pricing_env=env))
    SnowballPDESolver(PDEParams()).calculate_greeks(p, env)                        # seen[3]
    assert seen[:3] == [want, want, want]
    assert seen[3] != want      # the output request is part of the problem identity


def _split(a, b, env_a, env_b):
    ia = SnowballPDESolver(PDEParams()).explain(a, env_a).identity
    ib = SnowballPDESolver(PDEParams()).explain(b, env_b).identity
    for shared in ("grid_artefact", "coefficient_key", "factorization"):
        assert ia[shared] is not None and ia[shared] == ib[shared], shared
    assert ia["problem"] != ib["problem"]


def test_a_coupon_splits_only_the_problem():
    _split(make_snowball(dict(SB, ko_rate=0.15, rebate_rate=0.15)),
           make_snowball(dict(SB, ko_rate=0.30, rebate_rate=0.30)), env_flat(), env_flat())


def test_knocked_in_splits_only_the_problem():
    alive, knocked = make_snowball(SB), make_snowball(SB)
    knocked._otc_lifecycle_knocked_in = True
    _split(alive, knocked, env_flat(), env_flat())


def test_a_settlement_lag_splits_only_the_problem():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit

    lagged = make_snowball(SB)   # a tenor product settles on a year-fraction lag
    lagged.settlement_convention = SettlementConvention(lag=0.01, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    _split(make_snowball(SB), lagged, env_flat(), env_flat())


def test_routes_without_explain():
    from quantark.asset.equity.engine.pde import LocalVolPDESolver
    from quantark.asset.equity.engine.pde.barrier_vol_pde_solvers import LocalVolBarrierPDESolver
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
    from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine

    for cls in (DCNPDEEngine, LocalVolDCNPDEEngine, LocalVolPDESolver, LocalVolBarrierPDESolver):
        assert not hasattr(cls, "explain"), cls.__name__


def test_the_2d_solvers_refuse_explain():
    from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import HestonPhoenixPDESolver
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver
    from quantark.volmodels.heston import HestonParams

    hp = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.6)
    for cls in (HestonSnowballPDESolver, HestonPhoenixPDESolver):
        with pytest.raises(PricingError, match="explain covers the 1D core routes"):
            cls(hp).explain(make_snowball(SB), env_flat())


def test_pde_engine_explain_delegates():
    from quantark.asset.equity.engine.pde_engine import PDEEngine

    p = make_snowball(SB)
    assert (PDEEngine(PDEParams()).explain(p, env_flat()).identity["problem"]
            == SnowballPDESolver(PDEParams()).explain(p, env_flat()).identity["problem"])
