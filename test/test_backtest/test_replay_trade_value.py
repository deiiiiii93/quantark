"""Trade-mode product replay: cash legs booked by ``LegCashBook`` (ship spec 2026-10-23 portfolio-trade-value D9,
D11–D15; AC11, AC14).

AC11 as the same-environment identity of AC10, on every replay day ``d`` up to the knock-out:

    trade value before the close (the tracker's pre-close copy, legs rolled to ``τ_d`` keeping flows due today)
      + cash already paid
    == book MTM after the close + pending receivable PV + paid cash

so every leg flow that leaves the trade's value that day is booked, once, at the value it left with. The
fixtures mirror ``test_lifecycle_leg_booking.py``: whole-day observations ``k/365``, spot 100 then 110 on the
knock-out observation day.
"""
import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest.replay import (
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    HedgeSpec,
    ReplayBacktestConfig,
    ReplayBacktestEngine,
    ReplayProduct,
)
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import PaymentConvention, SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.booking import run_time
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.execution.errors import CapabilityError
from quantark.portfolio import EquityPosition
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

START = datetime(2026, 3, 16)
D = 1 / 365.0
OBS = [30, 60, 90]
KO_DAY = 60
PATH = [0, 5, 10, 15, 29, 30, 31, 45, 59, KO_DAY]
TAIL = [70, 90, 95]


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=90 * D, contract_multiplier=1_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[d * D for d in OBS],
                                                       ki_barrier=75.0, ki_continuous=True))


def _phoenix():
    """A Phoenix on the same schedule; its cash legs are KO_MATURITY basis like a Snowball's (user ruling
    2026-10-23)."""
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    return create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=OBS[-1] * D, ko_barrier=103.0,
                                   ki_barrier=75.0, coupon_barrier=80.0, coupon_rate=0.01,
                                   num_observations=len(OBS))


def _accrual(ko, basis, convention=PaymentConvention.AT_PERIOD_END):
    starts = np.array([0] + OBS[:-1]) * D
    ends = np.array(OBS) * D
    return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                      base=BaseAmount(value=100_000.0, mode=BaseAmountMode.ABSOLUTE),
                      schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                      payment_convention=convention, ko_behavior=ko, survival_basis=basis)


def _legs():
    obs = tuple(d * D for d in OBS)
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=10 * D),
        _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD),
        _accrual(KOBehavior.PAY_FULL_SCHEDULE, SurvivalBasis.COMPLETE_PERIOD),
        _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD, PaymentConvention.AT_MATURITY),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=90 * D,
                            terminal_settlement_time=90 * D, accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO})),
    ]


def _market(days):
    dates = [pd.Timestamp(START + timedelta(days=d)) for d in days]
    spots = [110.0 if d == KO_DAY else 100.0 for d in days]
    n = len(dates)
    futures = pd.DataFrame({"date": dates, "contract": ["FAR"] * n, "futures_price": spots,
                            "expiry_date": [pd.Timestamp("2030-01-01")] * n, "multiplier": [1.0] * n})
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [0.2] * n}),
        rate_data=pd.DataFrame({"date": dates, "rate": [0.03] * n}),
        futures_data=futures,
    )


def _config(kind="snowball", *, legs=None, trade_value=True, days=PATH, terminate=True, **product_kwargs):
    product = _phoenix() if kind == "phoenix" else _snowball()
    replay_product = ReplayProduct(product=product, quantity=1.0, position_id=0, has_lifecycle=True,
                                   cash_legs=_legs() if legs is None else legs, **product_kwargs)
    return ReplayBacktestConfig(
        products=[replay_product],
        market_data=_market(days),
        hedge=HedgeSpec(kind="spot"),
        engine_config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE, quad_params=QuadParams()),
        fixed_dividend_yield=0.01,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        terminate_on_lifecycle_end=terminate,
        trade_value=trade_value,
    )


def _record_pre_close_values(engine, original_legs):
    """Hook the note's lifecycle step: record its trade value before each close (the pre-close copy, legs rolled
    keeping flows due today), under that day's environment."""
    replay = engine._replays[0]
    observe = replay.apply_lifecycle_events
    values = {}

    def hooked(date, product, env, spot):
        if replay.lifecycle.alive:
            tau = run_time(START, date)
            position = EquityPosition(
                product=replay._tracker.product_for_pricing(date, env, initial=True), quantity=1.0,
                entry_price=0.0, underlying="equity_index", engine=engine._pricing_engines[0],
                entry_timestamp=START, position_id="note",
                cash_legs=[l for l in (leg.time_shift(tau, keep_landing=True) for leg in original_legs)
                           if l is not None])
            values[pd.Timestamp(date).normalize()] = position.get_trade_value(env)
        return observe(date, product, env, spot)

    replay.apply_lifecycle_events = hooked
    return values


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_value_plus_booked_cash_is_conserved_through_payments_and_the_knock_out(kind):
    config = _config(kind)
    original = list(config.products[0].cash_legs)
    engine = ReplayBacktestEngine(config)
    pre_close = _record_pre_close_values(engine, original)
    states = engine.run().states_df().set_index("date")
    paid_before = 0.0
    for day in PATH:
        date = pd.Timestamp(START + timedelta(days=day))
        row = states.loc[date]
        after = row["product_mtm"] + row["pending_receivable_pv"] + row["paid_cash"]
        assert after == pytest.approx(pre_close[date] + paid_before, rel=1e-10), f"day {day}"
        paid_before = row["paid_cash"]
    # entry: no initial_price / entry_trade_value given, so the trade is entered at its pre-close trade value
    assert engine._initial_book_value == pytest.approx(pre_close[pd.Timestamp(START)], rel=1e-12)
    ko_row = states.loc[pd.Timestamp(START + timedelta(days=KO_DAY))]
    assert not ko_row["alive"]
    assert ko_row["leg_receivable_pv"] > 0.0                       # PAY_FULL + AT_MATURITY flows still owed
    assert "leg_paid_cash" in states.columns


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_leg_receivables_pay_at_nominal_and_total_leg_cash_is_the_realized_flows(kind):
    engine = ReplayBacktestEngine(_config(kind, days=PATH + TAIL))
    results = engine.run()
    states = results.states_df().set_index("date")
    c = 0.04 * 100_000.0 * 30 * D                                  # every period is 30 days
    realized = (-250.0                                             # fee, day 10
                + c + c                                            # TRUNCATE/ENTER: periods starting before the KO
                + c + c + c                                        # PAY_FULL: every period
                + c + c                                            # AT_MATURITY TRUNCATE/ENTER, paid at maturity
                + 1_000.0                                          # FixedPayoff on KO
                + 10_000.0 * 0.05 * KO_DAY * D)                    # rebate at the KO observation
    assert engine.leg_paid_cash == pytest.approx(realized, rel=1e-13)
    assert states["leg_paid_cash"].iloc[-1] == engine.leg_paid_cash
    assert engine.leg_receivable_pv == 0.0
    # the run waits for the leg receivables: it ends on day 90, when the last of them pays
    assert states.index[-1] == pd.Timestamp(START + timedelta(days=90))
    assert results.get_summary()["termination_reason"] == "ko"
    # leg cash is part of the book's paid cash
    assert states["paid_cash"].iloc[-1] - states["leg_paid_cash"].iloc[-1] == pytest.approx(
        states.loc[pd.Timestamp(START + timedelta(days=KO_DAY)), "paid_cash"]
        - states.loc[pd.Timestamp(START + timedelta(days=KO_DAY)), "leg_paid_cash"], rel=1e-15)


def test_leg_receivables_outstanding_at_data_end_label_the_run_data_end():
    engine = ReplayBacktestEngine(_config(days=PATH + [70]))
    summary = engine.run().get_summary()
    assert summary["termination_reason"] == "data_end"
    assert summary["all_settled"] is False


def test_flag_off_with_legs_books_no_leg_cash_and_matches_the_legless_replay_bitwise():
    with_legs = ReplayBacktestEngine(_config(trade_value=False))
    states = with_legs.run().states_df().set_index("date")
    legless = ReplayBacktestEngine(_config(trade_value=False, legs=[])).run().states_df().set_index("date")
    pd.testing.assert_frame_equal(states, legless, check_exact=True)
    assert "leg_paid_cash" not in states.columns
    assert with_legs.leg_paid_cash == 0.0


def test_trade_mode_without_legs_matches_product_mode_bitwise():
    trade = ReplayBacktestEngine(_config(legs=[])).run().states_df().set_index("date")
    product = ReplayBacktestEngine(_config(trade_value=False, legs=[])).run().states_df().set_index("date")
    pd.testing.assert_frame_equal(trade, product, check_exact=True)


def test_entry_trade_value_is_the_trade_entry():
    engine = ReplayBacktestEngine(_config(days=PATH[:3], entry_trade_value=12_345.0))
    states = engine.run().states_df().set_index("date")
    assert engine._initial_book_value == 12_345.0
    first = states.iloc[0]
    assert first["product_pnl"] == pytest.approx(
        first["product_mtm"] + first["pending_receivable_pv"] + first["paid_cash"] - 12_345.0, rel=1e-15)


def test_a_product_price_cannot_enter_a_trade_with_cash_legs():
    with pytest.raises(ValidationError, match="entry_trade_value"):
        ReplayBacktestEngine(_config(days=PATH[:3], initial_price=50.0)).run()


def _record_leg_greeks(engine):
    """Hook the day's Greeks: recompute the legs' delta/gamma independently (trade-risk total less product, one
    bump loop) on the day's product, environment and live rolled legs."""
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    replay = engine._replays[0]
    original = replay.calculate_greeks
    seen = {}

    def hooked(product, env, *, engine=None):
        result = original(product, env, engine=engine)
        legs = list(engine_obj._leg_positions[0].cash_legs) if 0 in engine_obj._leg_positions else []
        if legs:
            position = EquityPosition(product=product, quantity=float(engine_obj._quantities[0]), entry_price=0.0,
                                      underlying="equity_index", engine=engine, entry_timestamp=START,
                                      position_id="note", cash_legs=legs)
            risk = position.get_trade_risk(env, GreeksCalculator(), greeks=["delta", "gamma"])
            seen[pd.Timestamp(env.valuation_date).normalize()] = (
                risk.total["delta"] - risk.product["delta"], risk.total["gamma"] - risk.product["gamma"])
        return result

    engine_obj = engine
    replay.calculate_greeks = hooked
    return seen


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_trade_mode_hedges_the_trade_total_delta(kind):
    """User decision 2026-10-23: replay in trade mode with legs hedges on product + legs Greeks."""
    engine = ReplayBacktestEngine(_config(kind))
    seen = _record_leg_greeks(engine)
    result = engine.run()
    states = result.states_df().set_index("date")
    greeks = result.greeks_df().set_index("date")
    legless = ReplayBacktestEngine(_config(kind, legs=[])).run().greeks_df().set_index("date")
    live = [d for d in seen if d in states.index]
    assert live, "the note must be alive on some days"
    for date in live:
        leg_delta, leg_gamma = seen[date]
        assert states.loc[date, "leg_delta"] == pytest.approx(leg_delta, rel=1e-12, abs=1e-12)
        assert states.loc[date, "leg_gamma"] == pytest.approx(leg_gamma, rel=1e-12, abs=1e-12)
        assert greeks.loc[date, "product_position_delta"] == pytest.approx(
            legless.loc[date, "product_position_delta"] + leg_delta, rel=1e-12, abs=1e-12)
    assert any(abs(seen[d][0]) > 0.0 for d in live)


def test_leg_greek_columns_exist_only_with_leg_books():
    states = ReplayBacktestEngine(_config(trade_value=False)).run().states_df()
    assert "leg_delta" not in states.columns and "leg_gamma" not in states.columns


def test_fixed_payoff_on_knock_in_is_refused():
    legs = [FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KI)]
    with pytest.raises(CapabilityError):
        ReplayBacktestEngine(_config(legs=legs, days=PATH[:3])).run()


def test_legs_on_a_replay_without_lifecycle_observation_are_refused():
    config = _config(days=PATH[:3])
    config.products[0].has_lifecycle = False
    with pytest.raises(CapabilityError):
        ReplayBacktestEngine(config)


def test_legs_on_a_non_autocallable_product_are_refused():
    from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    config = _config(days=PATH[:3])
    config.products[0].product = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=90 * D)
    with pytest.raises(CapabilityError):
        ReplayBacktestEngine(config)


def test_cash_legs_must_have_a_finite_entry_trade_value():
    with pytest.raises(ValidationError):
        ReplayProduct(product=_snowball(), quantity=1.0, position_id=0, has_lifecycle=True,
                      cash_legs=_legs(), entry_trade_value=float("nan"))


def test_replay_product_positional_slots_are_unchanged():
    product = ReplayProduct(_snowball(), 1.0, 0, True, 99.0)
    assert product.initial_price == 99.0 and product.cash_legs == [] and product.entry_trade_value is None
    assert copy.copy(product).cash_legs == []
