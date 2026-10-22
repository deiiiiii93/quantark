"""A convertible's hazard rate defaults to its credit spread through the credit triangle.

GS and TF read ``credit_spread``; jump-diffusion and the HW trinomial read a default intensity. Without
an explicit ``hazard_rate`` the intensity is ``credit_spread / (1 - recovery_rate)``, so a bond priced on
either family carries the same credit; an explicit ``hazard_rate`` (zero included) still wins.
"""
import dataclasses
from datetime import datetime

import pytest

from quantark.asset.bond.engine.convertible import ConvertibleBondEngine
from quantark.asset.bond.engine.pde.convertible import ConvertibleBondJumpDiffusionEngine, ConvertibleBondPDEParams
from quantark.asset.bond.engine.tree.convertible import ConvertibleBondTreeParams, ConvertibleBondTrinomialEngine
from quantark.asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTrinomialVolScheme
from quantark.asset.bond.product.convertible import ConvertibleBond
from quantark.asset.credit.conventions import spread_to_hazard
from quantark.param.quote import SpotQuote
from quantark.param.rrf import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ConvertibleBondMethod
from quantark.util.exceptions import ValidationError

ENV = PricingEnvironment(
    valuation_date=datetime(2024, 6, 1),
    spot_quote=SpotQuote(spot=12.0),
    vol_surface=FlatVolSurface(volatility=0.30),
    rate_curve=FlatRateCurve(rate=0.05),
)


def _bond(**credit):
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        **credit,
    )


class TestSpreadToHazard:
    def test_credit_triangle(self):
        assert spread_to_hazard(0.02, 0.4) == pytest.approx(0.02 / 0.6, abs=1e-15)

    def test_full_recovery_is_undefined(self):
        with pytest.raises(ValidationError):
            spread_to_hazard(0.02, 1.0)


class TestEffectiveHazardRate:
    def test_derived_from_spread_without_hazard(self):
        bond = _bond(credit_spread=0.02, recovery_rate=0.4)
        assert bond.hazard_rate is None
        assert bond.effective_hazard_rate == spread_to_hazard(0.02, 0.4)

    def test_explicit_hazard_wins(self):
        assert _bond(credit_spread=0.02, hazard_rate=0.01).effective_hazard_rate == 0.01

    def test_explicit_zero_hazard_wins(self):
        assert _bond(credit_spread=0.02, hazard_rate=0.0).effective_hazard_rate == 0.0

    def test_no_credit_inputs_mean_no_default(self):
        assert _bond().effective_hazard_rate == 0.0

    def test_replacing_the_spread_rederives_the_hazard(self):
        bond = dataclasses.replace(_bond(credit_spread=0.02, recovery_rate=0.4), credit_spread=0.03)
        assert bond.effective_hazard_rate == spread_to_hazard(0.03, 0.4)

    def test_full_recovery_needs_an_explicit_hazard(self):
        bond = _bond(credit_spread=0.02, recovery_rate=1.0)  # constructs: GS and TF never read the hazard
        with pytest.raises(ValidationError):
            bond.effective_hazard_rate
        assert _bond(credit_spread=0.02, recovery_rate=1.0, hazard_rate=0.01).effective_hazard_rate == 0.01

    def test_negative_hazard_is_rejected(self):
        with pytest.raises(ValidationError):
            _bond(hazard_rate=-0.01)


def _jd(bond):
    params = ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=200)
    return ConvertibleBondJumpDiffusionEngine(ENV, params).price_with_details(bond).dirty_price


def _trinomial(scheme):
    def price(bond):
        params = ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=scheme)
        return ConvertibleBondTrinomialEngine(ENV, params).price_with_details(bond).dirty_price
    return price


HAZARD_ENGINES = {
    "jump-diffusion": _jd,
    **{f"trinomial-{s.value}": _trinomial(s) for s in ConvertibleBondTrinomialVolScheme},
}


@pytest.mark.parametrize("engine", list(HAZARD_ENGINES))
def test_hazard_engines_price_the_derived_hazard(engine):
    derived = _bond(credit_spread=0.02, recovery_rate=0.4)
    explicit = _bond(credit_spread=0.02, recovery_rate=0.4, hazard_rate=spread_to_hazard(0.02, 0.4))
    price = HAZARD_ENGINES[engine]
    assert price(derived) == price(explicit)
    assert price(derived) < price(_bond(recovery_rate=0.4))  # the spread now carries default risk


@pytest.mark.parametrize("method", [ConvertibleBondMethod.JUMP_DIFFUSION, ConvertibleBondMethod.TRINOMIAL_HW])
def test_cs01_bumps_the_derived_hazard(method):
    derived = _bond(credit_spread=0.02, recovery_rate=0.4)
    explicit = _bond(credit_spread=0.02, recovery_rate=0.4, hazard_rate=spread_to_hazard(0.02, 0.4))
    engine = ConvertibleBondEngine(ENV, method=method)
    assert engine.cs01(derived) == engine.cs01(explicit)
    assert engine.cs01(derived) > 0.0
