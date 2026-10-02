"""The damping class is a function of the contract alone (spec 3A decision 12, §5.3)."""
import copy
import pickle

import pytest

from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.pde import (BarrierPDESolver, KOResetSnowballPDESolver, PhoenixPDESolver,
                                              SnowballPDESolver)
from quantark.asset.equity.engine.pde.core import prepare as prepare_mod
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.engine.pde.core.identity import component_fingerprint
from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_snowball import make_environment, make_snowball

from pde_3a.products import ENV, KO_RESET, PHOENIX, SNOWBALL, dates, PRODUCTS


def _explicit(key):
    form, term, event = DAMPING_DEFAULTS[key]
    return PDEParams(stepper=CrankNicolson(damping=form, terminal_damping_steps=term, event_damping_steps=event))


def _products(ki):
    mon = {"ki_monitoring": "continuous" if ki == "cont" else "discrete"}
    return {
        "snowball": (SnowballPDESolver, make_snowball(dict(SNOWBALL, **mon))),
        "phoenix": (PhoenixPDESolver, make_phoenix(dict(PHOENIX, ki_barrier=85.0, coupon_barrier=90.0, **mon))),
        "ko_reset": (KOResetSnowballPDESolver, make_ko_reset(dict(KO_RESET, ki_barrier=85.0, **mon))),
    }


@pytest.mark.parametrize("ki,key", [("cont", "core-continuous-ki"), ("disc", "core")])
@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
@pytest.mark.parametrize("spot", [100.0, 84.0])          # alive, and knocked in at valuation (continuous)
def test_autocallables_resolve_by_their_knock_in_monitoring(ki, key, family, spot):
    cls, product = _products(ki)[family]
    env = make_environment(dict(ENV, spot=spot))
    assert cls(PDEParams()).adapter.damping_key(product) == key
    assert cls(PDEParams()).price(product, env) == cls(_explicit(key)).price(product, env)


def test_a_continuous_flag_without_a_knock_in_barrier_is_core(monkeypatch):
    cls, product = _products("cont")["snowball"]
    monkeypatch.setattr(type(product), "has_ki_barrier", property(lambda self: False))
    assert cls(PDEParams()).adapter.damping_key(product) == "core"


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_base_family_products_are_core(name):
    product = PRODUCTS[name][1](dates(1.0 / 12.0))
    env = make_environment(ENV)
    assert BarrierPDESolver(PDEParams()).adapter.damping_key(product) == "core"
    assert PDEEngine(params=PDEParams()).price(product, env) == PDEEngine(params=_explicit("core")).price(product, env)


def test_explain_reports_the_form_in_force():
    _, product = _products("disc")["snowball"]
    plan = SnowballPDESolver(PDEParams()).explain(product, make_environment(ENV))
    assert plan.scheme["resolved"]["damping"] == DAMPING_DEFAULTS["core"][0]
    assert plan.settings["stepper.damping"]["effective"] == DAMPING_DEFAULTS["core"][0]


@pytest.mark.parametrize("ki", ["cont", "disc"])
@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
@pytest.mark.parametrize("spot", [85.5, 84.9])          # a down bump knocks in; an up bump un-knocks
def test_bumps_across_the_barrier_keep_one_stepper_and_one_layout(ki, family, spot, monkeypatch):
    cls, product = _products(ki)[family]
    env = make_environment(dict(ENV, spot=spot))
    seen = []
    orig = prepare_mod.march

    def spy(prepared):
        seen.append((prepared.identity["layout"], component_fingerprint(prepared.stepper)))
        return orig(prepared)

    monkeypatch.setattr(prepare_mod, "march", spy)
    GreeksCalculator().calculate_numerical_greeks(product=product, pricing_env=env, engine=cls(PDEParams()),
                                                  greeks=["delta", "gamma"])
    assert len({s for _, s in seen}) == 1, "a bump changed the stepper"
    assert len({lay for lay, _ in seen}) == 1, "a bump changed the layout"


@pytest.mark.parametrize("ki", ["cont", "disc"])
@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
@pytest.mark.parametrize("bump", ["spot_down", "spot_up", "vol", "rate"])
def test_a_bumped_price_equals_a_fresh_solve_on_the_frozen_layout(ki, family, bump):
    cls, product = _products(ki)[family]
    base_env = make_environment(dict(ENV, spot=85.5))
    over = {"spot_down": dict(spot=85.5 * 0.99), "spot_up": dict(spot=85.5 * 1.01),
            "vol": dict(spot=85.5, vol=ENV["vol"] + 0.01), "rate": dict(spot=85.5, rate=ENV["rate"] + 0.0001)}[bump]
    bumped = make_environment(dict(ENV, **over))
    solver = cls(PDEParams())
    ctx = solver.create_bump_context(product, base_env)
    resolved = solver.adapter.stepper(solver.params, product)
    fresh = cls(PDEParams(stepper=resolved))
    fresh._frozen_base_layout = ctx._frozen_base_layout
    assert ctx.price(product, bumped) == fresh.price(product, bumped)


def test_a_half_step_solver_survives_copy_and_pickle():
    # Review Focus 3
    _, product = _products("disc")["snowball"]
    env = make_environment(ENV)
    solver = SnowballPDESolver(PDEParams(stepper=CrankNicolson(damping="half_step")))
    price = solver.price(product, env)
    for clone in (copy.deepcopy(solver), pickle.loads(pickle.dumps(solver))):
        assert clone.price(product, env) == price
    request = solver.explain(product, env).to_dict()["time"]["request"]
    assert pickle.loads(pickle.dumps(request)) == request


class _HalfStepCustom(CrankNicolson):
    """A custom stepper that states its own refinement (Review Focus 4)."""


class _NoRefinementStepper:
    """A custom stepper without ``time_refinement``: the built-in arithmetic, no damping."""

    def schedule(self, layout):
        import numpy as np
        return np.full(layout.time.actual_steps, 0.5)

    def step(self, state, s, dt, theta, boundaries):
        return CrankNicolson.step(CrankNicolson(), state, s, dt, theta, boundaries)

    def evidence(self):
        return {}


def test_custom_steppers_on_a_core_route():
    _, product = _products("disc")["snowball"]
    env = make_environment(ENV)
    plan = SnowballPDESolver(PDEParams(stepper=_NoRefinementStepper())).explain(product, env)
    assert plan.to_dict()["time"]["request"].refinement.is_zero
    custom = _HalfStepCustom(damping="half_step", terminal_damping_steps=1, event_damping_steps=1)
    plan = SnowballPDESolver(PDEParams(stepper=custom)).explain(product, env)
    assert plan.to_dict()["time"]["request"].refinement == custom.time_refinement()
    assert plan.time.actual_steps > plan.time.requested_steps


def test_a_subclass_without_a_form_names_the_fields_it_must_set():
    # final review, Important 1: a subclass is a custom stepper, used as given (route_applicability),
    # so the route never fills its damping form. 0.7.0 priced (1, 2) with no form; the refusal must
    # say what to set, not "call resolved()", which the caller never calls.
    _, product = _products("disc")["snowball"]
    env = make_environment(ENV)
    legacy = _HalfStepCustom(terminal_damping_steps=1, event_damping_steps=2)
    with pytest.raises(ValidationError, match=r"damping.*'full_step'") as raised:
        SnowballPDESolver(PDEParams(stepper=legacy)).price(product, env)
    assert "resolved()" not in str(raised.value)
    fixed = _HalfStepCustom(damping="full_step", terminal_damping_steps=1, event_damping_steps=2)
    builtin = CrankNicolson(damping="full_step", terminal_damping_steps=1, event_damping_steps=2)
    assert (SnowballPDESolver(PDEParams(stepper=fixed)).price(product, env)
            == SnowballPDESolver(PDEParams(stepper=builtin)).price(product, env))
