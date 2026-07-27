"""EquityPosition passes leg-required streams to price_with_events [§11.1].

A KO-only leg set must trigger a solve with KI columns pruned; a leg reading the
MATURITY_WITH_KI terminal bucket must force the KI columns on. Both must still
value correctly.
"""

from datetime import datetime

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg.autocallable_leg import (
    AccrualBasis,
    AutocallableCashLeg,
    AutocallableLegType,
)
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _leg(terminal_events):
    return AutocallableCashLeg(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.REBATE,
        notional=100.0,
        rate=0.05,
        accrual_basis=AccrualBasis.KO_MATURITY,
        terminal_events=frozenset(terminal_events),
    )


def _position(engine, leg):
    return EquityPosition(
        product=_snowball(),
        quantity=1.0,
        entry_price=100.0,
        underlying="TEST",
        engine=engine,
        entry_timestamp=datetime(2024, 1, 1),
        cash_legs=[leg],
    )


def _spy_streams(engine):
    seen = {}
    original = engine._compute_event_stats

    def wrapper(product, pricing_env, *, npv=None, streams=None):
        seen["streams"] = streams
        return original(product, pricing_env, npv=npv, streams=streams)

    engine._compute_event_stats = wrapper
    return seen


def test_ko_only_leg_prunes_ki_columns():
    engine = SnowballPDESolver(PDEParams())
    seen = _spy_streams(engine)
    pos = _position(engine, _leg({EventType.MATURITY_NO_KO}))
    value = pos.get_trade_value(_env())
    assert seen["streams"] == frozenset({EventType.KO, EventType.MATURITY_NO_KO})
    assert EventType.KI not in seen["streams"]
    assert EventType.MATURITY_WITH_KI not in seen["streams"]
    assert value == value  # finite / no raise


def test_with_ki_leg_forces_ki_columns():
    engine = SnowballPDESolver(PDEParams())
    seen = _spy_streams(engine)
    pos = _position(engine, _leg({EventType.MATURITY_WITH_KI}))
    pos.get_trade_value(_env())
    assert EventType.MATURITY_WITH_KI in seen["streams"]


def test_trade_value_breakdown_also_wires_streams():
    engine = SnowballPDESolver(PDEParams())
    seen = _spy_streams(engine)
    pos = _position(engine, _leg({EventType.MATURITY_NO_KO}))
    pos.get_trade_value_breakdown(_env())
    assert seen["streams"] == frozenset({EventType.KO, EventType.MATURITY_NO_KO})
