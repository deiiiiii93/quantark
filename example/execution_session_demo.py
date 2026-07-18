"""Execution-framework migration demo 1: sessions, price_many, batch backends.

Shows that ``PricingSession`` wraps existing engines WITHOUT changing their
numbers: session PV == direct PV, a small book prices through ``price_many``,
and a DCN fixed-batch run is bit-identical between the serial and threads
batch backends. Also prints the reproducibility manifest every outcome
carries.

Run:  python example/execution_session_demo.py     (finishes in seconds)
Docs: docs/execution/README.md
"""
import dataclasses
from datetime import datetime

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc import EuropeanMCEngine
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.dcn_option import DCNDirection, DCNOption
from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.execution import (
    ExecutionPolicy,
    ExecutorSelection,
    PricingRequest,
    PricingSession,
    default_context,
)
from quantark.execution.contracts import PricingOperation
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType


def flat_env(valuation_date):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.ACT_365,
    )


def dcn_product():
    calendar = create_calendar(CalendarType.CHINA_SSE)
    schedule = build_dcn_schedule(
        calendar=calendar,
        initial_date=datetime(2023, 1, 3),
        valuation_date=datetime(2023, 1, 3),
        maturity_date=datetime(2025, 1, 3),
        tenor_months=24, lock_months=3, ko_lock_months=3,
        coupon_settlement_offset=2, ko_settlement_offset=2,
        settlement_date=datetime(2025, 1, 7),
    )
    return DCNOption(
        direction=DCNDirection.BUYER, notional=1_000_000.0,
        initial_price=100.0, coupon_barrier_ratio=0.80, ko_barrier_ratio=1.00,
        ki_barrier_ratio=0.75, ki_put_strike_ratio=1.10, coupon_rate=0.12,
        ko_coupon_rate=0.12, participation=1.0, coupon_counted_days=30,
        coupon_days_denom=360, schedule=schedule,
        settlement_date=datetime(2025, 1, 7),
    )


def batch_context(backend, workers):
    """A session context selecting the fixed-batch MC backend."""
    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend=backend, workers=workers)
        ),
    )


def main():
    env = flat_env(datetime(2024, 1, 1))
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    # 1. Session price == direct price (same code path, same numbers).
    engine = EuropeanMCEngine(params=MCParams(num_paths=20_000, seed=42))
    direct = engine.price(option, env)
    with PricingSession() as session:
        via_session = session.price(engine, option, env)
    assert via_session == direct
    print(f"1. session == direct: {via_session:.6f}")

    # 2. price_many over a small book (caller order preserved).
    book = [
        (BlackScholesEngine(), PricingRequest(product=option, pricing_env=env)),
        (engine, PricingRequest(product=option, pricing_env=env)),
    ]
    with PricingSession() as session:
        values = session.price_many(book)  # fail-fast path returns raw PVs
    print("2. price_many:", [f"{float(v):.6f}" for v in values])

    # 3. DCN fixed-batch MC: serial and threads backends are BIT-IDENTICAL.
    product = dcn_product()
    dcn_env = flat_env(datetime(2023, 1, 3))
    request = PricingRequest(
        product=product, pricing_env=dcn_env,
        operation=PricingOperation.PRICE_DETAILED,
    )

    def dcn_engine():
        return DCNMCEngine(num_paths=2**14, seed=42, num_batches=8)

    direct_pv = dcn_engine().price_detailed(product, dcn_env).pv
    with PricingSession(batch_context("serial", 1)) as session:
        serial_outcome = session.execute(dcn_engine(), request)
    with PricingSession(batch_context("threads", 4)) as session:
        threads_outcome = session.execute(dcn_engine(), request)
    assert serial_outcome.value.pv == direct_pv
    assert threads_outcome.value.pv == direct_pv  # bit-identical, not "close"
    print(f"3. DCN serial == threads == direct: {direct_pv:.6f}")

    # 4. Every outcome explains itself: reproducibility manifest.
    manifest = threads_outcome.manifest
    print("4. manifest:", manifest.schema_version, "|", manifest.adapter_id,
          "|", manifest.engine_class_path)
    print("   resolved policy:", dict(manifest.resolved_policy))


if __name__ == "__main__":
    main()
