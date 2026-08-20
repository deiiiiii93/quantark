"""`disable_ko_after_ki` in the PDE solvers.

The flag means what it says: once the trade knocks in, the knock-out barrier
stops applying. The two-surface solver expresses "knocked in" as the V1
surface, so honouring the flag means the KO payoff is written to V0 only.
`snowball_pde_solver` wrote it to both, so the flag did nothing at all -- the
engine silently priced a different product, agreeing with its own flag-off
price to ten significant figures while QUAD and Monte Carlo both moved.

These tests need no benchmark. If the valuation spot already sits below the KI
barrier the trade is knocked in from the start, so with the flag set no
knock-out can ever pay and the KO rate becomes economically irrelevant. That
is an exact invariant, not a tolerance.
"""

import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.modelvalidation.builders.equity_snowball import make_environment
from quantark.util.enum import ObservationType

ENV = {"spot": 100.0, "vol": 0.22, "rate": 0.025, "div_yield": 0.03}
KNOCKED_IN_ENV = dict(ENV, spot=80.0)  # below the 85 KI barrier: already in
MONTHS, MATURITY = 12, 1.0
DATES = [MATURITY * (i + 1) / MONTHS for i in range(MONTHS)]


def snowball(ko_rate: float, disable: bool) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=ko_rate,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=DATES,
            ki_barrier=85.0,
            ki_continuous=True,
            disable_ko_after_ki=disable,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.15),
        contract_multiplier=1.0,
        maturity=MATURITY,
    )


def price(product, env=None) -> float:
    solver = SnowballPDESolver(params=PDEParams(accuracy="standard"))
    return solver.calculate_greeks(product, make_environment(env or KNOCKED_IN_ENV))["price"]


def test_ko_rate_is_irrelevant_once_disabled_and_knocked_in():
    """The invariant: no KO can pay, so its rate cannot enter the price."""
    assert price(snowball(0.15, disable=True)) == pytest.approx(
        price(snowball(0.02, disable=True)), abs=1e-9
    )


def test_the_invariant_is_not_vacuous():
    """With the flag off the KO survives the knock-in, so the rate must matter."""
    assert price(snowball(0.15, disable=False)) != pytest.approx(
        price(snowball(0.02, disable=False)), abs=1e-6
    )


def test_the_flag_moves_the_price():
    """Disabling a KO that would otherwise pay cannot leave the price alone."""
    assert price(snowball(0.15, disable=True)) != pytest.approx(
        price(snowball(0.15, disable=False)), abs=1e-6
    )


def test_flag_off_is_untouched_at_the_certified_cell():
    """Regression guard: the certified configuration must not move at all."""
    assert price(snowball(0.15, disable=False), env=ENV) == pytest.approx(
        96.46065192554826, abs=1e-9
    )


def phoenix(ko_rate: float, disable: bool):
    product = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=MATURITY,
        ko_barrier=103.0, ko_rate=ko_rate, ki_barrier=85.0,
        coupon_barrier=85.0, coupon_rate=0.02, num_observations=MONTHS,
        memory_coupon=False, disable_ko_after_ki=disable,
    )
    return product


def test_phoenix_ko_rate_is_irrelevant_once_disabled_and_knocked_in():
    solver = PhoenixPDESolver(params=PDEParams(accuracy="standard"))

    def px(rate):
        return solver.calculate_greeks(
            phoenix(rate, disable=True), make_environment(KNOCKED_IN_ENV)
        )["price"]

    assert px(0.15) == pytest.approx(px(0.02), abs=1e-9)
