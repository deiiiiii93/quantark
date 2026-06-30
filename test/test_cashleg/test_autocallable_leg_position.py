"""Position-level PV + Greeks for AutocallableCashLeg, plus the quantity contract."""

from datetime import datetime

import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.cashleg.event_distribution import EventDistribution
from quantark.util.exceptions import ValidationError
from test_cashleg._autocallable_helpers import (
    make_env,
    make_snowball,
    make_phoenix,
    make_engine,
    future_event_times,
    make_margin_leg,
)


def _pos(product, engine, legs, quantity=1.0):
    return EquityPosition(
        product=product, quantity=quantity, entry_price=0.0, underlying="UND",
        engine=engine, entry_timestamp=datetime(2024, 1, 1), cash_legs=legs,
    )


@pytest.mark.parametrize("asset", ["snowball", "phoenix"])
@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_margin_leg_pv_and_greeks_finite(asset, kind):
    env = make_env()
    product = make_snowball() if asset == "snowball" else make_phoenix()
    engine = make_engine(kind, asset)
    leg = make_margin_leg(future_event_times(product, engine, env))
    pos = _pos(product, engine, [leg])
    greeks = pos.get_trade_greeks(env, GreeksCalculator())
    assert np.isfinite(pos.get_trade_value(env))
    assert np.isfinite(greeks["delta"]) and abs(greeks["delta"]) > 0.0
    assert np.isfinite(greeks["gamma"])


def test_quantity_scales_trade_value_linearly():
    env = make_env()
    product = make_snowball()
    engine = make_engine("pde", "snowball")
    obs = future_event_times(product, engine, env)
    v1 = _pos(product, engine, [make_margin_leg(obs)], quantity=1.0).get_trade_value(env)
    v3 = _pos(product, engine, [make_margin_leg(obs)], quantity=3.0).get_trade_value(env)
    assert abs(v3 - 3.0 * v1) <= 1e-6 * max(1.0, abs(v1))


def test_fail_loud_when_engine_emits_no_ko_stream():
    env = make_env()
    product = make_snowball()
    engine = make_engine("pde", "snowball")
    leg = make_margin_leg(future_event_times(product, engine, env))
    with pytest.raises(ValidationError):
        leg.value(
            EventDistribution.trivial(float(leg.terminal_settlement_time)), env, 0.0
        )
