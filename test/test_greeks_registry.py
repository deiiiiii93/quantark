"""Registry as the single source of greek validation/aliases/defaults."""

import pytest

from quantark.asset.equity.riskmeasures.greeks.registry import (
    ALIASES,
    DEFAULT_SET,
    REGISTRY,
    normalize_greeks,
)
from quantark.util.exceptions import ValidationError


def test_registry_matches_incumbent_surface():
    assert DEFAULT_SET == {
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "dividend_rho",
        "convexity_theta",
        "r_theta",
        "q_theta",
    }
    assert ALIASES["deltadq"] == "delta_q"
    assert ALIASES["rhoq"] == "dividend_rho"
    assert "vanna" in REGISTRY
    assert REGISTRY["delta"].linear_value == 1.0
    assert REGISTRY["gamma"].linear_value == 0.0


def test_normalize_resolves_aliases_to_canonical_keys():
    requests = normalize_greeks(["deltadq", "div_rho"])
    assert {req.key for req in requests} == {"delta_q", "dividend_rho"}
    assert all(req.clock is None for req in requests)


def test_normalize_none_and_empty():
    assert normalize_greeks(None) is None
    assert normalize_greeks([]) == set()


def test_normalize_rejects_unknown():
    with pytest.raises(ValidationError):
        normalize_greeks(["not_a_greek"])


def test_normalize_rejects_bad_type():
    with pytest.raises(ValidationError):
        normalize_greeks([3.14])


def test_clock_qualified_normalization():
    requests = normalize_greeks(["theta_1td", "veta_1d"])
    by_key = {req.key: req for req in requests}
    assert by_key["theta_1td"].canonical == "theta"
    assert by_key["theta_1td"].clock == "1td"
    assert by_key["vega_theta_1d"].canonical == "vega_theta"
    assert by_key["vega_theta_1d"].clock == "1d"


def test_clock_qualifier_rejected_for_non_time_greeks():
    with pytest.raises(ValidationError):
        normalize_greeks(["vanna_1td"])
    with pytest.raises(ValidationError):
        normalize_greeks(["speed_1d"])


def test_every_registered_name_is_computable():
    """The anti-drift test: every canonical name, alias, and clock-qualified
    form must produce its key when requested individually. This is the test
    that makes the charm/color silent-miss class of bug unrepresentable."""
    from datetime import datetime

    from quantark.asset.equity.engine.analytical.black_scholes_engine import (
        BlackScholesEngine,
    )
    from quantark.asset.equity.product.option.european_vanilla_option import (
        EuropeanVanillaOption,
    )
    from quantark.asset.equity.riskmeasures.greeks_calculator import (
        GreeksCalculator,
    )
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment
    from quantark.util.calendar import CalendarType, create_calendar
    from quantark.util.enum import OptionType

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 6, 26),
        calendar=create_calendar(CalendarType.CHINA_SSE),
    )
    product = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    calc = GreeksCalculator()
    engine = BlackScholesEngine()

    names = []
    for greek_def in REGISTRY.values():
        names.append(greek_def.name)
        names.extend(greek_def.aliases)
        if greek_def.supports_clock:
            names.append(f"{greek_def.name}_1d")
            names.append(f"{greek_def.name}_1td")

    for name in names:
        result = calc.calculate_numerical_greeks(
            product, env, engine, greeks=[name]
        )
        requests = normalize_greeks([name])
        expected_key = next(iter(requests)).key
        assert expected_key in result, f"{name!r} produced no {expected_key!r}"


def test_greek_def_carries_no_dead_dispatch_fields():
    """Dispatch order lives in the facade's hand-ordered chain (call order is
    part of the compatibility contract); the registry must not advertise
    per-greek callables or dependency lists that nothing reads."""
    from quantark.asset.equity.riskmeasures.greeks.registry import GreekDef

    assert set(GreekDef.__dataclass_fields__) == {
        "name", "aliases", "analytical_auto", "default", "linear_value", "supports_clock",
    }


def test_linear_order_matches_default_set():
    from quantark.asset.equity.riskmeasures.greeks.numerical import LINEAR_ORDER

    assert len(LINEAR_ORDER) == len(DEFAULT_SET)
    assert set(LINEAR_ORDER) == set(DEFAULT_SET)
