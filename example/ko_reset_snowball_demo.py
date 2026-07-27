"""
KO-reset snowball pricing demo using the example valuation parameters
from docs/敲出重置雪球期权结构报备.docx (section: 三、估值参数).

Notes:
- KI is modeled as discrete daily (business days).
- Post-KI schedule is ABSOLUTE (fixed calendar dates).
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path


from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.param import BumpConfig, EngineParams, MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    KnockOutResetSnowballOption,
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import ObservationType, PostKOScheduleMode
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod


def business_day_time(calendar, start_date, end_date, bus_days_in_year):
    # "含头含尾" -> include start and end business days
    days = calendar.count_business_days(
        start_date, end_date, include_start=True, include_end=True
    )
    return days / float(bus_days_in_year)


def build_observation_schedule(dates, barrier, ko_rate, valuation_date, calendar, bus_days_in_year):
    records = [
        ObservationRecord(
            observation_date=dt,
            observation_time=business_day_time(
                calendar, valuation_date, dt, bus_days_in_year
            ),
            barrier=barrier,
            return_rate=ko_rate,
        )
        for dt in dates
    ]
    return ObservationSchedule(records=records)


def build_business_day_ki_schedule(start_date, end_date, barrier, valuation_date, calendar, bus_days_in_year):
    records = []
    current = start_date
    while current <= end_date:
        if calendar.is_business_day(current):
            records.append(
                ObservationRecord(
                    observation_date=current,
                    observation_time=business_day_time(
                        calendar, valuation_date, current, bus_days_in_year
                    ),
                    barrier=barrier,
                )
            )
        current += timedelta(days=1)
    return ObservationSchedule(records=records)


def business_day_offsets(calendar, start_date, dates):
    offsets = []
    for curr in dates:
        offsets.append(
            calendar.count_business_days(
                start_date, curr, include_start=True, include_end=True
            )
        )
    return offsets


def main():
    initial_price = 8340.11
    notional = 50_000_000.0
    contract_multiplier = notional / initial_price

    valuation_date = datetime(2026, 1, 21)
    initial_date = valuation_date
    bus_days_in_year = 244
    calendar = create_calendar(CalendarType.CHINA, year_range=(2025, 2031))

    pre_ko_rate = 0.15
    post_ko_rate = 0.03
    pre_ko_barrier = 1.03 * initial_price
    post_ko_barrier = 0.95 * initial_price

    ki_barrier = 0.80 * initial_price

    pre_ko_dates = [
        datetime(2026, 4, 21),
        datetime(2026, 5, 21),
        datetime(2026, 6, 22),
        datetime(2026, 7, 21),
        datetime(2026, 8, 21),
        datetime(2026, 9, 21),
        datetime(2026, 10, 21),
        datetime(2026, 11, 23),
        datetime(2026, 12, 21),
        datetime(2027, 1, 21),
        datetime(2027, 2, 22),
        datetime(2027, 3, 22),
        datetime(2027, 4, 21),
        datetime(2027, 5, 21),
        datetime(2027, 6, 21),
        datetime(2027, 7, 21),
        datetime(2027, 8, 23),
        datetime(2027, 9, 21),
        datetime(2027, 10, 21),
        datetime(2027, 11, 22),
        datetime(2027, 12, 21),
        datetime(2028, 1, 21),
    ]

    post_ko_dates = [
        datetime(2026, 4, 21),
        datetime(2026, 5, 21),
        datetime(2026, 6, 22),
        datetime(2026, 7, 21),
        datetime(2026, 8, 21),
        datetime(2026, 9, 21),
        datetime(2026, 10, 21),
        datetime(2026, 11, 23),
        datetime(2026, 12, 21),
        datetime(2027, 1, 21),
        datetime(2027, 2, 22),
        datetime(2027, 3, 22),
        datetime(2027, 4, 21),
        datetime(2027, 5, 21),
        datetime(2027, 6, 21),
        datetime(2027, 7, 21),
        datetime(2027, 8, 23),
        datetime(2027, 9, 21),
        datetime(2027, 10, 21),
        datetime(2027, 11, 22),
        datetime(2027, 12, 21),
        datetime(2028, 1, 21),
        datetime(2028, 2, 21),
        datetime(2028, 3, 21),
        datetime(2028, 4, 21),
        datetime(2028, 5, 22),
        datetime(2028, 6, 21),
        datetime(2028, 7, 21),
        datetime(2028, 8, 21),
        datetime(2028, 9, 21),
        datetime(2028, 10, 23),
        datetime(2028, 11, 21),
        datetime(2028, 12, 21),
        datetime(2029, 1, 22),
        datetime(2029, 2, 21),
        datetime(2029, 3, 21),
        datetime(2029, 4, 23),
        datetime(2029, 5, 21),
        datetime(2029, 6, 21),
        datetime(2029, 7, 23),
        datetime(2029, 8, 21),
        datetime(2029, 9, 21),
        datetime(2029, 10, 22),
        datetime(2029, 11, 21),
        datetime(2029, 12, 21),
        datetime(2030, 1, 21),
    ]

    pre_ko_schedule = build_observation_schedule(
        pre_ko_dates,
        pre_ko_barrier,
        pre_ko_rate,
        valuation_date,
        calendar,
        bus_days_in_year,
    )
    post_ko_schedule = build_observation_schedule(
        post_ko_dates,
        post_ko_barrier,
        post_ko_rate,
        valuation_date,
        calendar,
        bus_days_in_year,
    )

    ki_schedule = build_business_day_ki_schedule(
        valuation_date + timedelta(days=1),
        pre_ko_dates[-1],
        ki_barrier,
        valuation_date,
        calendar,
        bus_days_in_year,
    )

    barrier_config = BarrierConfig(
        ko_barrier=pre_ko_barrier,
        ko_rate=pre_ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=pre_ko_schedule,
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ki_schedule,
        ki_continuous=False,
    )

    post_barrier_config = BarrierConfig(
        ko_barrier=post_ko_barrier,
        ko_rate=post_ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=post_ko_schedule,
    )

    payoff_config = PayoffConfig(
        rebate_rate=0.15,
        include_principal=False,
        participation_rate=1.0,
    )
    accrual_config = AccrualConfig(
        is_annualized=True,
        is_annualized_rebate=True,
        is_annualized_ki=False,
    )

    product = KnockOutResetSnowballOption(
        initial_price=initial_price,
        strike=initial_price,
        barrier_config=barrier_config,
        post_barrier_config=post_barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        contract_multiplier=contract_multiplier,
        maturity=4.0,
        initial_date=initial_date,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=initial_price),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.04),
        valuation_date=valuation_date,
        bus_days_in_year=bus_days_in_year,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        calendar=calendar,
    )

    print(
        "Pre-KO business-day offsets from valuation:",
        business_day_offsets(calendar, valuation_date, pre_ko_dates),
    )
    print(
        "Post-KO business-day offsets from valuation:",
        business_day_offsets(calendar, valuation_date, post_ko_dates),
    )
    base_params = MCParams(num_paths=20000, time_steps=244, use_business_day_grid=True)
    doc_value = 689_803.6242
    doc_delta_cash = -27_467_316.0375
    doc_vega_1pct = 210_951.6748
    position = -1.0  # Seller (short) position
    greeks_calc = GreeksCalculator(
        params=EngineParams(bump_config=BumpConfig(spot_bump=0.01, vol_bump=0.01))
    )
    results = []

    def add_row(label, seller_pv, diff, diff_pct, delta_cash, vega_1pct):
        results.append(
            {
                "Engine": label,
                "Seller PV": seller_pv,
                "Doc PV": doc_value,
                "Diff vs Doc": diff,
                "Diff %": diff_pct,
                "Delta Cash": delta_cash,
                "Doc Delta Cash": doc_delta_cash,
                "Delta Cash Diff": delta_cash - doc_delta_cash,
                "Delta Cash Diff %": (
                    (delta_cash - doc_delta_cash) / doc_delta_cash
                    if doc_delta_cash != 0
                    else 0.0
                ),
                "Vega 1%": vega_1pct,
                "Doc Vega 1%": doc_vega_1pct,
                "Vega 1% Diff": vega_1pct - doc_vega_1pct,
                "Vega 1% Diff %": (
                    (vega_1pct - doc_vega_1pct) / doc_vega_1pct
                    if doc_vega_1pct != 0
                    else 0.0
                ),
            }
        )

    def compute_greeks(engine):
        greeks = greeks_calc.calculate(
            product, pricing_env, engine, method="numerical", greeks=["delta", "vega"]
        )
        delta = greeks.get("delta", 0.0)
        vega = greeks.get("vega", 0.0)
        delta_cash = position * delta * pricing_env.spot
        vega_1pct = position * vega  # vega is per 1% vol bump
        return delta_cash, vega_1pct

    def run_engine(label, method, params=None):
        engine = SnowballMCEngine(params=params or base_params, method=method)
        price = engine.price(product, pricing_env)
        seller_pv = position * price
        diff = seller_pv - doc_value
        diff_pct = diff / doc_value if doc_value != 0 else 0.0
        delta_cash, vega_1pct = compute_greeks(engine)
        add_row(label, seller_pv, diff, diff_pct, delta_cash, vega_1pct)

    run_engine("Pseudo", EngineType.MONTE_CARLO(MonteCarloMethod.PSEUDO))
    run_engine("QMC", EngineType.MONTE_CARLO(MonteCarloMethod.QUASI))
    run_engine("RQMC", EngineType.MONTE_CARLO(MonteCarloMethod.RANDOMIZED_QUASI))

    def run_pde(label: str, params: PDEParams) -> None:
        solver = KOResetSnowballPDESolver(params=params)
        price = solver.price(product, pricing_env)
        seller_pv = position * price
        diff = seller_pv - doc_value
        diff_pct = diff / doc_value if doc_value != 0 else 0.0
        delta_cash, vega_1pct = compute_greeks(solver)
        add_row(label, seller_pv, diff, diff_pct, delta_cash, vega_1pct)

    run_pde("PDE", PDEParams(grid_size=180, time_steps=120))
    run_pde(
        "PDE-Fixed1000",
        PDEParams(
            grid_size=1000,
            time_steps=1000,
        ),
    )

    def run_quad(label: str, params: QuadParams) -> None:
        engine = KOResetSnowballQuadEngine(params=params)
        price = engine.price(product, pricing_env)
        seller_pv = position * price
        diff = seller_pv - doc_value
        diff_pct = diff / doc_value if doc_value != 0 else 0.0
        delta_cash, vega_1pct = compute_greeks(engine)
        add_row(label, seller_pv, diff, diff_pct, delta_cash, vega_1pct)

    run_quad(
        "QUAD-Alt2",
        QuadParams(
            grid_points=1401,
            num_std_devs=10.5,
            stability_preset="conservative",
            align_priority="auto",
            event_smoothing_mode="fixed",
            event_smoothing_cells=1,
        ),
    )

    headers = [
        "Engine",
        "Seller PV",
        "Doc PV",
        "Diff vs Doc",
        "Diff %",
        "Delta Cash",
        "Doc Delta Cash",
        "Delta Cash Diff",
        "Delta Cash Diff %",
        "Vega 1%",
        "Doc Vega 1%",
        "Vega 1% Diff",
        "Vega 1% Diff %",
    ]
    fmt = {
        "Seller PV": lambda x: f"{x:,.4f}",
        "Doc PV": lambda x: f"{x:,.4f}",
        "Diff vs Doc": lambda x: f"{x:,.4f}",
        "Diff %": lambda x: f"{x:+.4%}",
        "Delta Cash": lambda x: f"{x:,.4f}",
        "Doc Delta Cash": lambda x: f"{x:,.4f}",
        "Delta Cash Diff": lambda x: f"{x:,.4f}",
        "Delta Cash Diff %": lambda x: f"{x:+.4%}",
        "Vega 1%": lambda x: f"{x:,.4f}",
        "Doc Vega 1%": lambda x: f"{x:,.4f}",
        "Vega 1% Diff": lambda x: f"{x:,.4f}",
        "Vega 1% Diff %": lambda x: f"{x:+.4%}",
    }
    widths = {h: len(h) for h in headers}
    for row in results:
        for h in headers:
            val = row[h]
            text = fmt[h](val) if h in fmt else str(val)
            widths[h] = max(widths[h], len(text))

    header_line = "  ".join(f"{h:<{widths[h]}}" for h in headers)
    sep_line = "  ".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in results:
        parts = []
        for h in headers:
            val = row[h]
            text = fmt[h](val) if h in fmt else str(val)
            align = "<" if h == "Engine" else ">"
            parts.append(f"{text:{align}{widths[h]}}")
        print("  ".join(parts))

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ko_reset_snowball_demo_results.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
