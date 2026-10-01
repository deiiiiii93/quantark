"""The damping form is part of every layout and artefact key (spec 3A §5.3 "Identity", decision 6)."""
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.param import PDEParams
from quantark.execution import PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_snowball import make_environment, make_snowball

from pde_3a.products import ENV, SNOWBALL

FULL = CrankNicolson(damping="full_step", terminal_damping_steps=1, event_damping_steps=2)
HALF = CrankNicolson(damping="half_step", terminal_damping_steps=1, event_damping_steps=1)


def _product():
    return make_snowball(dict(SNOWBALL, ki_monitoring="discrete"))


def _direct(stepper):
    # the default factorization cache, as a session solve runs: a cache-off solve of this
    # discrete-KI snowball differs from cache-on by ~2e-12 already on 0.7.0 (not 3A's)
    s = SnowballPDESolver(PDEParams(stepper=stepper))
    return s.price(_product(), make_environment(ENV)), s.explain(_product(), make_environment(ENV))


@pytest.mark.parametrize("order", [(FULL, HALF), (HALF, FULL)])
def test_one_session_prices_both_forms_as_fresh_solves(order):
    env = make_environment(ENV)
    with PricingSession() as session:
        outcomes = [session.execute(SnowballPDESolver(PDEParams(stepper=st)), PricingRequest(product=_product(),
                                                                                          pricing_env=env))
                    for st in order]
    for st, outcome in zip(order, outcomes):
        price, _ = _direct(st)
        assert outcome.value == price
    fps = [_direct(st)[1].identity["grid_artefact"] for st in order]
    assert fps[0] != fps[1]


@pytest.mark.parametrize("stepper", [FULL, HALF])
def test_bump_context_and_calendar_rebind_keep_the_refinement(stepper):
    env = make_environment(ENV)
    solver = SnowballPDESolver(PDEParams(stepper=stepper))
    ctx = solver.create_bump_context(_product(), env)
    assert ctx._frozen_base_layout.request.refinement == stepper.time_refinement()
    rebound = solver.grid_binder.rebind_time(ctx._frozen_base_layout, ctx._frozen_base_layout.request)
    assert rebound.request.refinement == stepper.time_refinement()
    assert rebound.time.actual_steps == ctx._frozen_base_layout.time.actual_steps
