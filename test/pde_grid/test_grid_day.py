"""The grid day travels on the request (spec 3A §5.5)."""
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.grid import GridBinder, GridRequest, MarketSnapshot, validate_external_layout
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.util.exceptions import ValidationError

from pde_3a.products import SNOWBALL
from term_structure_benchmarks import make_term_env


def _req(day_count, events=(0.5,)):
    return GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,), hard_lower=None,
                       hard_upper=None, event_times=events, day_count=day_count)


def test_the_request_carries_the_environments_business_days():
    env = make_term_env("flat")
    env.bus_days_in_year = 244
    plan = SnowballPDESolver(PDEParams()).explain(make_snowball(SNOWBALL), env)
    assert plan.to_dict()["time"]["request"].day_count == 244


@pytest.mark.parametrize("bad", [0, -1, 1.5, None])
def test_day_count_is_a_positive_int(bad):
    with pytest.raises(ValidationError):
        _req(bad)


def test_a_frozen_layout_refuses_another_day_count():
    market = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.02, q_ref=0.0, bus_days_in_year=252)
    layout = GridBinder("fast", None).bind(_req(252), market)
    with pytest.raises(ValidationError, match="day_count"):
        validate_external_layout(layout, _req(244), market)


def test_bind_shared_refuses_mixed_day_counts():
    market = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.02, q_ref=0.0, bus_days_in_year=252)
    with pytest.raises(ValidationError, match="day_count"):
        GridBinder("fast", None).bind_shared([_req(252), _req(244)], market)


def test_step_counts_follow_the_environment():
    from dataclasses import replace

    from quantark.asset.equity.engine.pde import EuropeanPDESolver
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType
    euro = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    env = make_term_env("flat")
    t252 = EuropeanPDESolver(PDEParams()).explain(euro, env).time
    t244 = EuropeanPDESolver(PDEParams()).explain(euro, replace(env, bus_days_in_year=244)).time
    assert (t252.requested_steps, t244.requested_steps) == (1008, 976)   # spec §4.7: a 244 environment, 976 steps
    # flip (f): half-step (1, 1) splits the terminal interval, one delivered step more
    assert (t252.actual_steps, t244.actual_steps) == (1009, 977)


def test_griddconfig_day_count_is_removed():
    from quantark.asset.equity.engine.pde.grid.config import GridConfig
    with pytest.raises(ValidationError, match="bus_days_in_year"):
        GridConfig(day_count=244)
