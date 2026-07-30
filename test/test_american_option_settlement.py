"""American exercise obstacles with determination/payment separation."""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
from quantark.asset.equity.engine.pde import (
    AmericanPDESolver,
    GridConfig,
)
from quantark.asset.equity.engine.settlement_support import (
    build_american_exercise_date_grid,
    resolve_american_exercise_timings,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.errors import CapabilityError
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import Calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError


VALUATION_DATE = datetime(2026, 1, 2)
EXERCISE_DATE = datetime(2026, 1, 8)
RATE = 0.05
NUMERIC_LAG = 0.10


def _env(
    *,
    spot: float = 80.0,
    rate: float = RATE,
    calendar: Calendar | None = None,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=VALUATION_DATE,
        calendar=calendar,
    )


def _option(
    *,
    convention=None,
    date_based: bool = False,
    contract_multiplier: float = 1.0,
) -> AmericanOption:
    kwargs = (
        {"exercise_date": EXERCISE_DATE}
        if date_based
        else {"maturity": 1.0}
    )
    return AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        contract_multiplier=contract_multiplier,
        settlement_convention=convention,
        **kwargs,
    )


def _year_fraction_lag() -> SettlementConvention:
    return SettlementConvention(
        lag=NUMERIC_LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


def _business_day_lag(calendar: Calendar) -> SettlementConvention:
    return SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        calendar=calendar,
    )


def _mc() -> AmericanOptionMCEngine:
    return AmericanOptionMCEngine(
        params=MCParams(
            num_paths=2048,
            time_steps=24,
            seed=17,
            use_antithetic=True,
        )
    )


def _pde() -> AmericanPDESolver:
    return AmericanPDESolver(
        PDEParams(
            grid=GridConfig(
                points=121,
                steps_per_day=1.0,
                day_count=365,
            )
        )
    )


def test_numeric_exercise_nodes_use_curve_exact_delay_ratios():
    env = _env()
    times = np.array([0.0, 0.25, 0.75, 1.0])
    timings = resolve_american_exercise_timings(
        _option(convention=_year_fraction_lag()),
        env,
        times,
    )

    expected = np.array(
        [
            env.get_discount_factor(t + NUMERIC_LAG)
            / env.get_discount_factor(t)
            for t in times
        ]
    )
    assert timings.eligible.tolist() == [True, True, True, True]
    assert timings.delay_dfs == pytest.approx(expected)
    assert timings.payment_times == pytest.approx(times + NUMERIC_LAG)


def test_terminal_settlement_date_applies_only_to_maturity_node():
    env = _env()
    product = AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        exercise_date=EXERCISE_DATE,
        settlement_date=datetime(2026, 1, 12),
    )
    times = np.array([0.0, product.get_maturity(env)])
    timings = resolve_american_exercise_timings(product, env, times)

    assert timings.delay_dfs[0] == pytest.approx(1.0)
    assert timings.payment_times[0] == pytest.approx(0.0)
    assert timings.payment_times[-1] > times[-1]


def test_business_day_nodes_are_anchored_to_real_exercise_dates():
    calendar = Calendar()
    env = _env(calendar=calendar)
    product = _option(
        convention=_business_day_lag(calendar),
        date_based=True,
    )
    dates, times = build_american_exercise_date_grid(product, env)
    timings = resolve_american_exercise_timings(
        product,
        env,
        times,
        exercise_dates=dates,
    )

    assert dates[0] == VALUATION_DATE
    assert dates[-1] == EXERCISE_DATE
    assert timings.node_dates == dates
    assert timings.payment_dates[0] == datetime(2026, 1, 6)
    assert timings.payment_dates[-1] == datetime(2026, 1, 12)


@pytest.mark.parametrize(
    ("factory", "timings_attribute"),
    [
        (_mc, "_last_exercise_timings"),
        (_pde, "_exercise_timings"),
    ],
)
def test_engines_price_business_day_settlement_on_date_nodes(
    factory,
    timings_attribute,
):
    calendar = Calendar()
    env = _env(calendar=calendar)
    product = _option(
        convention=_business_day_lag(calendar),
        date_based=True,
    )
    engine = factory()

    price = engine.price(product, env)
    timings = getattr(engine, timings_attribute)
    _, expected_times = build_american_exercise_date_grid(product, env)

    assert np.isfinite(price)
    assert price >= 0.0
    assert timings.node_dates[0] == VALUATION_DATE
    assert timings.node_dates[-1] == EXERCISE_DATE
    assert timings.payment_dates[0] == datetime(2026, 1, 6)
    assert timings.payment_dates[-1] == datetime(2026, 1, 12)
    assert timings.node_times[timings.eligible] == pytest.approx(
        expected_times
    )


def test_prepared_pde_session_preserves_date_aligned_settlement_grid():
    calendar = Calendar()
    env = _env(calendar=calendar)
    product = _option(
        convention=_business_day_lag(calendar),
        date_based=True,
    )
    engine = _pde()
    direct = engine.price(product, env)

    with PricingSession() as session:
        outcome = session.execute(
            engine,
            PricingRequest(product=product, pricing_env=env),
        )

    assert outcome.value == direct
    assert outcome.manifest.adapter_id != "legacy-price"


@pytest.mark.parametrize("engine", [_mc(), _pde()])
def test_numeric_american_rejects_day_based_settlement(engine):
    calendar = Calendar()
    product = _option(convention=_business_day_lag(calendar))

    with pytest.raises(
        ValidationError,
        match="authoritative exercise_date",
    ):
        engine.price(product, _env(calendar=calendar))


@pytest.mark.parametrize("factory", [_mc, _pde])
def test_zero_lag_is_exact_price_identity(factory):
    env = _env()
    immediate = factory().price(_option(), env)
    explicit_zero = factory().price(
        _option(convention=SettlementConvention()),
        env,
    )

    assert explicit_zero == immediate


def test_lsm_uses_delayed_values_in_exercise_comparison():
    engine = _mc()
    product = _option()
    paths = np.array(
        [
            [80.0, 80.0, 80.0],
            [120.0, 120.0, 120.0],
        ]
    )
    payoffs = engine._lsm_discounted_payoffs(
        product=product,
        paths=paths,
        discount_factors=np.ones(2),
        strike=100.0,
        exercise_delay_dfs=np.array([0.95, 0.90, 0.80]),
    )

    assert payoffs == pytest.approx([18.0, 0.0])


def test_delayed_settlement_reduces_deep_itm_put_obstacle_and_value():
    env = _env(spot=60.0)
    immediate = _pde().price(_option(), env)
    delayed = _pde().price(
        _option(convention=_year_fraction_lag()),
        env,
    )

    assert delayed < immediate
    assert delayed >= (
        _option().intrinsic_value(env.spot)
        * np.exp(-RATE * NUMERIC_LAG)
    )


@pytest.mark.parametrize("factory", [_mc, _pde])
def test_delayed_american_price_scales_once_with_contract_multiplier(factory):
    env = _env()
    unit_price = factory().price(
        _option(convention=_year_fraction_lag()),
        env,
    )
    contract_price = factory().price(
        _option(
            convention=_year_fraction_lag(),
            contract_multiplier=10.0,
        ),
        env,
    )

    assert contract_price == pytest.approx(10.0 * unit_price)


def test_rqmc_uses_delayed_exercise_timings():
    engine = AmericanOptionMCEngine(
        params=MCParams(
            num_paths=128,
            time_steps=8,
            seed=17,
            rqmc_min_batches=2,
            rqmc_max_batches=2,
            rqmc_target_std=1.0e-12,
        ),
        method="randomized_quasi",
    )

    price = engine.price(
        _option(convention=_year_fraction_lag()),
        _env(),
    )

    assert np.isfinite(price)
    assert engine.get_last_rqmc_result().batches_used == 2


def test_analytical_american_rejects_delayed_exercise():
    with pytest.raises(
        CapabilityError,
        match="american_exercise",
    ):
        AmericanOptionAnalyticalEngine().price(
            _option(convention=_year_fraction_lag()),
            _env(),
        )
