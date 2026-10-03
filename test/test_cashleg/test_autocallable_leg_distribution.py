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


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_phoenix_cashflow_decomposition_reconciles(kind, memory):
    # pv == sum(ko_cf) + sum(coupon_cf) + maturity_cf. memory coupons now carry their
    # cash (spec 3B §5.3): PDE reports the coupon cashflow and the maturity leg holds
    # the redemption only. QUAD keeps its own behaviour: it omits a memory coupon's
    # cash (the maturity leg absorbs it) and still reconciles.
    # coupon_probability is always present.
    env = make_env()
    ph = make_phoenix(memory=memory)
    s = make_engine(kind, "phoenix").calculate_event_stats(ph, env)
    assert s.coupon_probability.size == s.ko_times.size
    recon = (
        float(np.sum(s.expected_discounted_ko_cashflow))
        + float(np.sum(s.expected_discounted_coupon_cashflow))
        + float(s.expected_discounted_maturity_cashflow)
    )
    assert abs(s.pv - recon) <= 1e-6 * max(1.0, abs(s.pv))
    if not memory:
        # Non-memory PDE/QUAD classify the coupon PV separately.
        if kind in ("pde", "quad"):
            assert s.expected_discounted_coupon_cashflow.size == s.ko_times.size
    elif kind == "pde":
        assert s.expected_discounted_coupon_cashflow.size > 0
        assert float(np.sum(s.expected_discounted_coupon_cashflow)) > 0.0
    elif kind == "quad":
        assert s.expected_discounted_coupon_cashflow.size == 0


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
