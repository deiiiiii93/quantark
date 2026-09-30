"""Component conformance: the three slots accept any conforming object (spec 2 §4.1, §5.8)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.pde import EuropeanPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.core.components import CrankNicolson
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.execution import PricingRequest, PricingSession
from quantark.modelvalidation.builders.equity_snowball import SnowballPDECandidate, make_snowball
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical.comparison import is_close
from quantark.util.numerical.constants import Tolerance

from pde_bitwise.capture_book import env_flat
from pde_core.components_fixtures import (BackwardEuler, InPlaceStepper, NaNStepper, NearestNodeReadout,
                                          PlainStepper, ShiftedOperator, WrongShapeStepper)

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def _european():
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)


def _price(**params):
    return EuropeanPDESolver(PDEParams(**params)).price(_european(), env_flat())


@pytest.mark.parametrize("slot", [dict(stepper=BackwardEuler()), dict(operator=ShiftedOperator(shift=0.0)),
                                  dict(readout=NearestNodeReadout())])
def test_each_custom_slot_prices_a_european(slot):
    pv = _price(**slot)
    assert np.isfinite(pv) and pv > 0.0


def test_backward_euler_agrees_with_the_built_in_at_theta_one():
    built_in = _price(stepper=CrankNicolson(theta=1.0, terminal_damping_steps=0, event_damping_steps=0))
    custom = _price(stepper=BackwardEuler())
    assert is_close(custom, built_in, rel_tol=Tolerance.RELATIVE)


@pytest.mark.parametrize("stepper", [NaNStepper(), WrongShapeStepper()])
def test_a_bad_step_raises_naming_the_stepper(stepper):
    with pytest.raises(NumericalError, match=type(stepper).__name__):
        _price(stepper=stepper)


def test_a_stepper_without_a_value_fingerprint_prices_but_has_no_identity():
    assert _price(stepper=PlainStepper()) > 0.0
    solver = SnowballPDESolver(PDEParams(stepper=PlainStepper()))
    assert solver.explain(make_snowball(SB), env_flat()).identity["problem"] is None


def test_a_session_stores_no_factorization_for_an_unfingerprintable_stepper():
    solver = EuropeanPDESolver(PDEParams(stepper=PlainStepper()))
    with PricingSession() as session:
        for _ in range(2):
            session.execute(solver, PricingRequest(product=_european(), pricing_env=env_flat()))
        kinds = [d.kind for d in session.context.artifact_cache._entries]
    assert "pde-factorization-pack" not in kinds


def test_a_study_candidate_cannot_be_built_from_a_params_object():
    # from_config cannot name a custom component, and a PDEParams object is not a declaration
    with pytest.raises(ValidationError):
        SnowballPDECandidate(environment_params={}, product_params={}, quantities=("pv",),
                             params=PDEParams(stepper=PlainStepper()))


def test_the_state_a_stepper_is_handed_is_read_only():
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        _price(stepper=InPlaceStepper())
    assert _price(stepper=BackwardEuler()) == _price(stepper=BackwardEuler())
