"""PDEEngine must expose the event stats its solvers already compute.

The facade dispatches price() to a product-specific solver but inherited
BaseEngine.calculate_event_stats, which returns None for "unsupported".
SnowballPDESolver implements it, so the facade was hiding a working result --
invisible until the replay layer's fail-closed event-stats default turned
that None into a hard error (0/27 fleet replays failed with "event-stats
engine PDEEngine returned no stats").

Fixture parameters mirror the study's own term sheet
(example/mo_volmodels/11_pde_convergence_gate.py): 3Y maturity, KO at 103% of
spot with a 3-month lockout (monthly observations thereafter), KI at 75%.
The PDE grid is deliberately coarse (GridConfig override) so the test runs
in a couple of seconds rather than the ~45s a default-resolution snowball
solve takes.
"""
import dataclasses
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import AsianOption, SnowballOption
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

# Study term sheet (11_pde_convergence_gate.py): 3Y, KO 103%, KI 75%, 3-month
# lockout, monthly KO observations (34 dates: months 3..36).
SPOT = 100.0
MATURITY_YEARS = 3.0
LOCKOUT_MONTHS = 3
MATURITY_MONTHS = 36
KO_PCT = 1.03
KI_PCT = 0.75
KO_RATE = 0.15

# Deliberately coarse: this is a unit test for delegation, not a convergence
# check. A default-resolution (accuracy="standard") solve of this product
# takes ~45s; this grid runs in ~1-2s.
_COARSE_GRID = GridConfig(points=60, steps_per_day=1.0)


def _ko_observation_dates():
    return [m / 12.0 for m in range(LOCKOUT_MONTHS, MATURITY_MONTHS + 1)]


@pytest.fixture
def snowball_product() -> SnowballOption:
    return create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=MATURITY_YEARS,
        ko_barrier=KO_PCT * SPOT,
        ko_rate=KO_RATE,
        ki_barrier=KI_PCT * SPOT,
        num_observations=len(_ko_observation_dates()),
        ko_observation_dates=_ko_observation_dates(),
        rebate_rate=KO_RATE,
        include_principal=False,
    )


@pytest.fixture
def snowball_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 5),
    )


@pytest.fixture
def coarse_pde_params() -> PDEParams:
    return PDEParams(grid=_COARSE_GRID)


def test_pde_engine_returns_the_solver_event_stats(
    snowball_product, snowball_env, coarse_pde_params
):
    stats = PDEEngine(params=coarse_pde_params).calculate_event_stats(
        snowball_product, snowball_env
    )
    assert stats is not None, "facade swallowed the solver's event stats"
    assert len(stats.ko_times) == len(_ko_observation_dates())
    total_ko = float(sum(stats.ko_probability))
    assert 0.0 <= total_ko <= 1.0 + 1e-9


def test_pde_engine_event_stats_match_the_solver_directly(
    snowball_product, snowball_env, coarse_pde_params
):
    """Delegation must not transform or truncate the result.

    Compares all 13 fields ``AutocallableEventStats`` carries, including the
    five KI-breakdown fields (``ki_times``, ``ki_event_probability``,
    ``ki_survival_probability``, ``ki_ever_probability``,
    ``ki_survive_knocked_in_probability``) that drive the per-date KI row
    structure the 2026-08-01 PDE event-stats re-baseline changed most (7 rows
    -> 6 in the affected goldens). A facade that mangled or dropped only the
    KI breakdown -- while leaving ``pv``/KO fields intact -- would slip past
    a check limited to those; it will not slip past this one.
    """
    facade = PDEEngine(params=coarse_pde_params).calculate_event_stats(
        snowball_product, snowball_env
    )
    direct = SnowballPDESolver(params=coarse_pde_params).calculate_event_stats(
        snowball_product, snowball_env
    )
    assert facade is not None and direct is not None

    field_names = {f.name for f in dataclasses.fields(direct)}
    assert field_names == {
        "pv",
        "ko_times",
        "ko_probability",
        "survival_probability",
        "expected_discounted_ko_cashflow",
        "ki_probability",
        "expected_discounted_maturity_cashflow",
        "reconciliation_error",
        "ki_times",
        "ki_event_probability",
        "ki_survival_probability",
        "ki_ever_probability",
        "ki_survive_knocked_in_probability",
    }, "AutocallableEventStats field set changed -- update this test's coverage"

    assert facade.pv == pytest.approx(direct.pv)
    assert facade.ko_times == pytest.approx(direct.ko_times)
    assert facade.ko_probability == pytest.approx(direct.ko_probability)
    assert facade.survival_probability == pytest.approx(direct.survival_probability)
    assert facade.expected_discounted_ko_cashflow == pytest.approx(
        direct.expected_discounted_ko_cashflow
    )
    assert facade.ki_probability == pytest.approx(direct.ki_probability)
    assert facade.expected_discounted_maturity_cashflow == pytest.approx(
        direct.expected_discounted_maturity_cashflow
    )
    assert facade.reconciliation_error == pytest.approx(
        direct.reconciliation_error, abs=1e-8
    )

    # KI breakdown fields (ki_times/ki_event_probability/ki_survival_probability
    # are empty arrays here since this fixture uses continuous KI monitoring,
    # which populates no per-date KI rows -- still worth comparing, since a
    # facade that fabricated or dropped rows would show up as a shape/content
    # mismatch here, empty or not).
    assert facade.ki_times == pytest.approx(direct.ki_times)
    assert facade.ki_event_probability == pytest.approx(direct.ki_event_probability)
    assert facade.ki_survival_probability == pytest.approx(
        direct.ki_survival_probability
    )

    # ki_ever_probability / ki_survive_knocked_in_probability are
    # Optional[float] -- None if an engine does not compute them for a given
    # product. Assert None-ness matches either way (a facade that silently
    # dropped a computed value to None would be caught), and compare values
    # when present.
    assert (facade.ki_ever_probability is None) == (
        direct.ki_ever_probability is None
    ), (facade.ki_ever_probability, direct.ki_ever_probability)
    if facade.ki_ever_probability is not None:
        assert facade.ki_ever_probability == pytest.approx(direct.ki_ever_probability)

    assert (facade.ki_survive_knocked_in_probability is None) == (
        direct.ki_survive_knocked_in_probability is None
    ), (
        facade.ki_survive_knocked_in_probability,
        direct.ki_survive_knocked_in_probability,
    )
    if facade.ki_survive_knocked_in_probability is not None:
        assert facade.ki_survive_knocked_in_probability == pytest.approx(
            direct.ki_survive_knocked_in_probability
        )


def test_pde_engine_calculate_event_stats_propagates_unsupported_product_type(
    snowball_env, coarse_pde_params
):
    """Judgement call: PDEEngine.calculate_event_stats propagates the
    ValidationError _get_solver raises for a product type outside
    PRODUCT_SOLVER_MAP, rather than swallowing it into None.

    price() already propagates this exact ValidationError for the same
    "PDEEngine cannot handle this product type at all" condition; making
    calculate_event_stats swallow it into a bare None would look identical
    to a *supported* product with no event-stats implementation (e.g. a
    EuropeanVanillaOption through EuropeanPDESolver), silently inviting a
    caller with event_stats_fallback='mc' into an MC fallback for a product
    this engine was never asked to price in the first place.
    """
    unsupported_product = AsianOption(
        strike=SPOT, option_type=OptionType.CALL, maturity=1.0
    )
    engine = PDEEngine(params=coarse_pde_params)
    with pytest.raises(ValidationError, match="does not support product type"):
        engine.calculate_event_stats(unsupported_product, snowball_env)
