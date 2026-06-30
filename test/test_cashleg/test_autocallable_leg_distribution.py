"""price_with_events COUPON mapping + COUPON-basis leg position valuation."""

from datetime import datetime

import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg,
    AutocallableLegType,
    AccrualBasis,
)
from test_cashleg._autocallable_helpers import (
    make_env,
    make_phoenix,
    make_engine,
    future_event_times,
)


@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_phoenix_price_with_events_emits_coupon_stream(kind):
    env = make_env()
    ph = make_phoenix()
    dist = make_engine(kind, "phoenix").price_with_events(
        ph, env, emit_distribution=True
    ).event_distribution
    assert EventType.COUPON in dist.probabilities
    assert np.asarray(dist.probabilities[EventType.COUPON]).size == dist.event_times.size


@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_phoenix_cashflow_decomposition_reconciles(kind):
    # pv == sum(ko_cf) + sum(coupon_cf) + maturity_cf: coupon PV is classified in
    # its own field, not folded into the maturity cashflow (native PDE/QUAD parity
    # with MC).
    env = make_env()
    ph = make_phoenix()
    s = make_engine(kind, "phoenix").calculate_event_stats(ph, env)
    recon = (
        float(np.sum(s.expected_discounted_ko_cashflow))
        + float(np.sum(s.expected_discounted_coupon_cashflow))
        + float(s.expected_discounted_maturity_cashflow)
    )
    assert s.expected_discounted_coupon_cashflow.size == s.ko_times.size
    assert abs(s.pv - recon) <= 1e-6 * max(1.0, abs(s.pv))


@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_coupon_basis_leg_prices_in_position(kind):
    env = make_env()
    ph = make_phoenix()
    engine = make_engine(kind, "phoenix")
    obs = future_event_times(ph, engine, env)
    leg = AutocallableCashLeg(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.BACKEND_INTEREST,
        notional=1_000_000.0,
        rate=0.03,
        observation_schedule=tuple(obs),
        accrual_factors=tuple(np.full(obs.size, 0.5)),
        settlement_schedule=tuple(obs),
        terminal_accrual_factor=0.0,
        terminal_settlement_time=float(obs[-1]),
        accrual_basis=AccrualBasis.COUPON,
    )
    pos = EquityPosition(
        product=ph, quantity=1.0, entry_price=0.0, underlying="UND",
        engine=engine, entry_timestamp=datetime(2024, 1, 1), cash_legs=[leg],
    )
    pv = pos.get_trade_value(env)
    assert np.isfinite(pv) and pv != 0.0
