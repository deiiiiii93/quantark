"""One resolver for the per-leg annualization flags (accrual follow-ups §6, user decision 2026-10-09).

An explicit ``is_annualized_ko`` / ``is_annualized_rebate`` / ``is_annualized_ki`` always wins. Unset (``None``):
the knock-out coupon and the rebate follow ``is_annualized``; the knock-in payoff stays flat. The construction
check is the accrual spec's R5 alone, on the effective flags.
"""
import math
from datetime import timedelta

import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad import SnowballQuadEngine, SnowballQuadEngineV2
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.asset.equity.product.option import accrual
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from .conftest import T0, dates, env, ko_reset, phoenix, snowball, times


# -- the resolver ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("config, expected", [
    (AccrualConfig(), (True, True, False)),
    (AccrualConfig(is_annualized=False), (False, False, False)),
    (AccrualConfig(is_annualized=False, is_annualized_ko=True, is_annualized_rebate=True, is_annualized_ki=True),
     (True, True, True)),
    (AccrualConfig(is_annualized_ko=False, is_annualized_rebate=False, is_annualized_ki=False), (False, False, False)),
    (AccrualConfig(is_annualized_ki=True), (True, True, True)),
])
def test_an_explicit_flag_wins_and_none_follows_the_leg_default(config, expected):
    p = snowball(accrual=config)
    assert tuple(accrual.leg_annualized(p, leg) for leg in ("ko", "rebate", "knock_in")) == expected


def test_an_unknown_leg_is_refused():
    with pytest.raises(ValidationError, match="leg"):
        accrual.leg_annualized(snowball(), "coupon")


# -- the cash readers --------------------------------------------------------------------------------------------

def test_a_default_snowball_annualizes_its_rebate_and_keeps_its_knock_in_flat():
    p = create_standard_snowball(initial_price=100.0, strike=100.0, maturity=2.0, num_observations=24)
    e = env()
    assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=e), 0.15 * 100.0 * 2.0)
    assert is_close(p.get_maturity_payoff_v1(60.0, pricing_env=e), -40.0)


def test_is_annualized_false_or_an_explicit_false_keeps_the_rebate_flat():
    for config in (AccrualConfig(is_annualized=False), AccrualConfig(is_annualized_rebate=False)):
        p = snowball(accrual=config, maturity=0.25)
        assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=env()), 100.0 + 5.0)


def test_a_default_phoenix_annualizes_its_rebate():
    p = phoenix(maturity=0.25, payoff_config=PayoffConfig(rebate_rate=0.05))
    flat = phoenix(maturity=0.25, payoff_config=PayoffConfig(rebate_rate=0.05),
                   accrual=AccrualConfig(is_annualized_rebate=False))
    e = env()
    assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=e) - 100.0, 5.0 * 0.25)
    assert is_close(flat.get_maturity_payoff_v0(100.0, pricing_env=e) - 100.0, 5.0)


def test_a_default_ko_reset_annualizes_its_rebate_over_schedule_one_and_keeps_its_knock_in_flat():
    p = ko_reset()                                  # schedule 1 ends at 0.5, schedule 2 at 0.75
    e = env()
    assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=e) - 100.0, 5.0 * 0.5)
    assert is_close(p.get_maturity_payoff_v1(60.0, pricing_env=e), 60.0)


def test_the_ko_reset_tracker_reads_the_v0_tenor_of_a_default_contract():
    t = KOResetLifecycleTracker(product=ko_reset(), quantity=1.0, start_date=pd.Timestamp(T0))
    assert is_close(t.pre_contract_tenor(env()), 0.5)
    assert t.final_contract_tenor(env()) is None    # the knock-in payoff is flat


# -- a certain-V0 contract against its closed form (100 + rebate x tenor) e^{-rT} --------------------------------

def _certain_v0(maturity=2.0):
    """No knock-in, a knock-out barrier no path reaches: every path pays V0 at maturity."""
    ko = times(*[k / 4 for k in range(1, int(4 * maturity) + 1)], level=1.0e4)
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=maturity,
                          barrier_config=BarrierConfig(ko_barrier=1.0e4, ko_rate=0.12, ko_observation_schedule=ko),
                          payoff_config=PayoffConfig(rebate_rate=0.05), accrual_config=AccrualConfig())


def _rate_env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.02), valuation_date=T0)


CLOSED_FORM = (100.0 + 0.05 * 100.0 * 2.0) * math.exp(-0.02 * 2.0)


@pytest.mark.parametrize("engine, rel", [
    (lambda: SnowballQuadEngine(), 1e-6),
    (lambda: SnowballQuadEngineV2(), 1e-6),
    (lambda: SnowballMCEngine(MCParams(num_paths=2000, seed=7)), 1e-9),
    (lambda: SnowballPDESolver(PDEParams()), 1e-4),
])
def test_a_certain_v0_default_contract_prices_at_its_closed_form(engine, rel):
    price = engine().price(_certain_v0(), _rate_env())
    assert abs(price - CLOSED_FORM) <= rel * CLOSED_FORM, (price, CLOSED_FORM)


# -- the construction check is R5 alone, on the effective flags ---------------------------------------------------

def test_a_numeric_contract_with_an_explicitly_annualized_rebate_and_no_dates_is_accepted():
    """The old check refused it ("initial_date or settlement_date required ..."); its accrual runs on the
    contract clock (accrued_offset + maturity), which R5 never refuses."""
    p = snowball(accrual=AccrualConfig(is_annualized_rebate=True, is_annualized_ki=True), maturity=0.25)
    assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=env()) - 100.0, 5.0 * 0.25)


def test_a_dated_contract_whose_default_rebate_needs_an_accrual_start_is_refused_by_r5():
    """Knock-out cash flat, rebate unset: the rebate follows is_annualized=True and its tenor needs initial_date."""
    with pytest.raises(ValidationError, match="R5"):
        snowball(ko=dates(30, 60, 90), accrual=AccrualConfig(is_annualized_ko=False), maturity=None,
                 maturity_date=T0 + timedelta(days=90))


def test_a_dated_contract_with_a_flat_rebate_needs_no_accrual_start():
    p = snowball(ko=dates(30, 60, 90), accrual=AccrualConfig(is_annualized_ko=False, is_annualized_rebate=False),
                 maturity=None, maturity_date=T0 + timedelta(days=90))
    assert is_close(p.get_maturity_payoff_v0(100.0, pricing_env=env()) - 100.0, 5.0)
