"""
Price the external variable-KI Phoenix case with multiple engines.

Case source:
    external/ki_variant_phx/phoenix_example.txt

Runs the same product setup with:
    - MC (100K paths)
    - QMC (100K paths)
    - RQMC (total 100K paths)
    - PDE
    - QUAD

and prints / writes a comparison table.

Usage:
    python example/phoenix_external_case_compare.py
    python example/phoenix_external_case_compare.py --output external/ki_variant_phx/my_results.md
    python example/phoenix_external_case_compare.py --mc-paths 2000 --qmc-paths 2000 --rqmc-total-paths 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional


from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import (
    CalendarType,
    DayCountConvention,
    calculate_year_fraction,
    create_calendar,
)
from quantark.util.enum import CouponPayType, ObservationAggregation, ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod


# =========================
# External Case Parameters
# =========================

START_DATE = datetime(2024, 12, 2)
END_DATE = datetime(2026, 12, 2)
KI_SWITCH_DATE = datetime(2025, 12, 2)

INITIAL_PRICE = 5943.62
STRIKE = 5943.62
NOTIONAL = 1_000_000.0
CONTRACT_MULTIPLIER = NOTIONAL / INITIAL_PRICE

RATE = 0.025
DIV_YIELD = 0.05
VOL = 0.25

KO_BARRIER = 6121.9286
COUPON_BARRIER = 4754.896
COUPON_RATE = 0.12
BUS_DAYS_IN_YEAR = 244
BUSINESS_CALENDAR = create_calendar(CalendarType.CHINA_SSE, year_range=(2024, 2026))

# User setting: no extra KO interest
KO_RATE = 0.0

# Lock period = 3 months -> first two coupon observations cannot KO.
LOCKOUT_KO_BARRIER = 1.0e12
KI_BARRIER_PRE_SWITCH = 1.0e-8  # Approximation of "0%" KI barrier before switch date.
KI_BARRIER_POST_SWITCH = 4457.715

COUPON_AND_KO_OBS_DATES = [
    "2025-01-02",
    "2025-02-05",
    "2025-03-03",
    "2025-04-02",
    "2025-05-06",
    "2025-06-03",
    "2025-07-02",
    "2025-08-04",
    "2025-09-02",
    "2025-10-09",
    "2025-11-03",
    "2025-12-02",
    "2026-01-05",
    "2026-02-02",
    "2026-03-02",
    "2026-04-02",
    "2026-05-06",
    "2026-06-02",
    "2026-07-02",
    "2026-08-03",
    "2026-09-02",
    "2026-10-08",
    "2026-11-02",
    "2026-12-02",
]


@dataclass
class EngineRunResult:
    case_name: str
    engine: str
    price: float
    std_error: Optional[float]
    paths: Optional[int]
    batches: Optional[int]
    runtime_sec: float
    delta_cash: Optional[float] = None
    vega_1pct: Optional[float] = None
    diff_vs_mc: Optional[float] = None
    rel_diff_vs_mc: Optional[float] = None


def _to_dt(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def create_pricing_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=INITIAL_PRICE),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV_YIELD),
        valuation_date=START_DATE,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=BUS_DAYS_IN_YEAR,
        calendar=BUSINESS_CALENDAR,
    )


def build_ko_schedule() -> ObservationSchedule:
    records: List[ObservationRecord] = []
    for idx, date_str in enumerate(COUPON_AND_KO_OBS_DATES):
        obs_date = _to_dt(date_str)
        obs_time = calculate_year_fraction(
            START_DATE,
            obs_date,
            DayCountConvention.BUSINESS_DAYS,
            bus_days_in_year=BUS_DAYS_IN_YEAR,
            calendar=BUSINESS_CALENDAR,
        )
        ko_barrier = LOCKOUT_KO_BARRIER if idx < 2 else KO_BARRIER
        records.append(
            ObservationRecord(
                observation_time=obs_time,
                observation_date=obs_date,
                barrier=ko_barrier,
                return_rate=KO_RATE,
                is_rate_annualized=True,
                initial_date=START_DATE,
                day_count_convention=DayCountConvention.BUSINESS_DAYS,
            )
        )
    return ObservationSchedule(
        records=records,
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )


def build_business_day_ki_schedule() -> tuple[ObservationSchedule, List[float]]:
    records: List[ObservationRecord] = []
    barriers: List[float] = []

    num_days = (END_DATE - START_DATE).days
    for day_idx in range(1, num_days + 1):
        obs_date = START_DATE + timedelta(days=day_idx)
        if not BUSINESS_CALENDAR.is_business_day(obs_date):
            continue
        obs_time = calculate_year_fraction(
            START_DATE,
            obs_date,
            DayCountConvention.BUSINESS_DAYS,
            bus_days_in_year=BUS_DAYS_IN_YEAR,
            calendar=BUSINESS_CALENDAR,
        )
        barrier = (
            KI_BARRIER_PRE_SWITCH
            if obs_date < KI_SWITCH_DATE
            else KI_BARRIER_POST_SWITCH
        )
        records.append(
            ObservationRecord(
                observation_time=obs_time,
                observation_date=obs_date,
                barrier=barrier,
            )
        )
        barriers.append(barrier)

    return (
        ObservationSchedule(
            records=records,
            aggregation_mode=ObservationAggregation.ACCUMULATE,
        ),
        barriers,
    )


def create_case_product(participation_rate: float) -> PhoenixOption:
    ko_schedule = build_ko_schedule()
    ki_schedule, ki_barriers = build_business_day_ki_schedule()

    barrier_config = BarrierConfig(
        ko_barrier=KO_BARRIER,
        ko_rate=KO_RATE,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=ko_schedule,
        ki_barrier=ki_barriers,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ki_schedule,
        ki_continuous=False,
        # KI hit does not terminate the trade; KO remains active after KI.
        disable_ko_after_ki=False,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=COUPON_BARRIER,
        coupon_rate=COUPON_RATE,
        coupon_pay_type=CouponPayType.INSTANT,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        memory_coupon=False,
        # Equal monthly coupon accrual (30/360-style monthly equalization proxy)
        fixed_coupon_year_fraction=1.0 / 12.0,
    )
    payoff_config = PayoffConfig(
        rebate_rate=0.0,
        include_principal=False,
        participation_rate=participation_rate,
    )
    accrual_config = AccrualConfig(
        coupon_pay_type=CouponPayType.INSTANT,
        is_annualized=True,
    )

    maturity = calculate_year_fraction(
        START_DATE,
        END_DATE,
        DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=BUS_DAYS_IN_YEAR,
        calendar=BUSINESS_CALENDAR,
    )

    return PhoenixOption(
        initial_price=INITIAL_PRICE,
        strike=STRIKE,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        contract_multiplier=CONTRACT_MULTIPLIER,
        maturity=maturity,
        initial_date=START_DATE,
        annualization_day_count=DayCountConvention.BUSINESS_DAYS,
    )


def run_engine(engine_name: str, engine, product: PhoenixOption, env: PricingEnvironment) -> EngineRunResult:
    started = time.perf_counter()
    price = engine.price(product, env)
    runtime_sec = time.perf_counter() - started

    std_error: Optional[float] = None
    paths: Optional[int] = None
    batches: Optional[int] = None

    if isinstance(engine, PhoenixMCEngine):
        last = engine.get_last_result()
        if last is not None:
            std_error = last.std_error
            paths = last.num_paths
            batches = last.batches_used

    delta_cash: Optional[float] = None
    vega_1pct: Optional[float] = None

    # Finite-difference sensitivities:
    # - DeltaCash: central bump at spot +/-1%, reported as delta * spot
    # - Vega(1%): central bump at vol +/-1 vol point, reported as 1% vol price change
    try:
        spot_bump_rel = 0.01
        spot_up = env.spot * (1.0 + spot_bump_rel)
        spot_dn = env.spot * (1.0 - spot_bump_rel)
        if spot_dn <= 0.0:
            raise ValueError("Spot down bump produced non-positive spot.")

        env_spot_up = PricingEnvironment(
            spot_quote=SpotQuote(spot=spot_up),
            vol_surface=env.vol_surface,
            rate_curve=env.rate_curve,
            div_yield=env.div_yield,
            valuation_date=env.valuation_date,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
            calendar=env.calendar,
        )
        env_spot_dn = PricingEnvironment(
            spot_quote=SpotQuote(spot=spot_dn),
            vol_surface=env.vol_surface,
            rate_curve=env.rate_curve,
            div_yield=env.div_yield,
            valuation_date=env.valuation_date,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
            calendar=env.calendar,
        )
        price_spot_up = engine.price(product, env_spot_up)
        price_spot_dn = engine.price(product, env_spot_dn)
        delta_cash = (price_spot_up - price_spot_dn) / (2.0 * spot_bump_rel)

        maturity = product.get_maturity(env)
        base_vol = env.get_vol(product.strike, maturity)
        vol_bump_abs = 0.01
        vol_up = base_vol + vol_bump_abs
        vol_dn = max(1.0e-6, base_vol - vol_bump_abs)

        env_vol_up = PricingEnvironment(
            spot_quote=env.spot_quote,
            vol_surface=FlatVolSurface(volatility=vol_up),
            rate_curve=env.rate_curve,
            div_yield=env.div_yield,
            valuation_date=env.valuation_date,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
            calendar=env.calendar,
        )
        env_vol_dn = PricingEnvironment(
            spot_quote=env.spot_quote,
            vol_surface=FlatVolSurface(volatility=vol_dn),
            rate_curve=env.rate_curve,
            div_yield=env.div_yield,
            valuation_date=env.valuation_date,
            day_count_convention=env.day_count_convention,
            bus_days_in_year=env.bus_days_in_year,
            calendar=env.calendar,
        )
        price_vol_up = engine.price(product, env_vol_up)
        price_vol_dn = engine.price(product, env_vol_dn)
        vega_1pct = 0.5 * (price_vol_up - price_vol_dn)
    except Exception:
        # Keep output robust for engines/configs that fail under bump runs.
        delta_cash = None
        vega_1pct = None

    return EngineRunResult(
        case_name="",
        engine=engine_name,
        price=price,
        std_error=std_error,
        paths=paths,
        batches=batches,
        runtime_sec=runtime_sec,
        delta_cash=delta_cash,
        vega_1pct=vega_1pct,
    )


def run_case(
    case_name: str,
    participation_rate: float,
    env: PricingEnvironment,
    args: argparse.Namespace,
) -> List[EngineRunResult]:
    product = create_case_product(participation_rate=participation_rate)

    mc_engine = PhoenixMCEngine(
        params=MCParams(
            num_paths=args.mc_paths,
            time_steps=252,
            seed=args.seed,
        ),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.PSEUDO),
    )
    qmc_engine = PhoenixMCEngine(
        params=MCParams(
            num_paths=args.qmc_paths,
            time_steps=252,
            seed=args.seed,
        ),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
    )

    rqmc_batches = max(1, args.rqmc_batches)
    rqmc_per_batch_paths = max(1, args.rqmc_total_paths // rqmc_batches)
    rqmc_engine = PhoenixMCEngine(
        params=MCParams(
            num_paths=rqmc_per_batch_paths,
            time_steps=252,
            seed=args.seed,
            rqmc_min_batches=rqmc_batches,
            rqmc_max_batches=rqmc_batches,
            rqmc_target_std=1.0e-12,
            rqmc_paths_mode="per_batch",
        ),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.RANDOMIZED_QUASI),
    )

    pde_engine = PhoenixPDESolver(
        params=PDEParams(
            grid_size=args.pde_grid,
            time_steps=args.pde_steps,
            auto_grid=True,
            time_grid_type="event_aligned",
            event_min_steps_per_interval=1,
            max_time_steps=args.pde_max_steps,
            log_dx_target=args.pde_log_dx,
        )
    )
    quad_engine = PhoenixQuadEngine(
        params=QuadParams(
            grid_points=args.quad_grid,
            num_std_devs=args.quad_std_devs,
        )
    )

    engines = [
        ("MC(100K)", mc_engine),
        ("QMC(100K)", qmc_engine),
        ("RQMC(total 100K)", rqmc_engine),
        ("PDE", pde_engine),
        ("QUAD", quad_engine),
    ]

    results: List[EngineRunResult] = []
    for name, engine in engines:
        row = run_engine(name, engine, product, env)
        row.case_name = case_name
        results.append(row)

    mc_price = next((r.price for r in results if r.engine == "MC(100K)"), None)
    if mc_price is not None:
        for row in results:
            row.diff_vs_mc = row.price - mc_price
            row.rel_diff_vs_mc = (
                row.diff_vs_mc / mc_price if abs(mc_price) > 1.0e-14 else None
            )
    return results


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return f"{value:,d}"


def build_markdown_table(rows: List[EngineRunResult]) -> str:
    lines = [
        "| Case | Engine | Price | DeltaCash | Vega(1%) | StdErr | Paths | Batches | Runtime(s) | Diff vs MC | Rel Diff vs MC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rel_diff = "-"
        if row.rel_diff_vs_mc is not None:
            rel_diff = f"{row.rel_diff_vs_mc:.4%}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.case_name,
                    row.engine,
                    _fmt_float(row.price, 4),
                    _fmt_float(row.delta_cash, 4),
                    _fmt_float(row.vega_1pct, 4),
                    _fmt_float(row.std_error, 6),
                    _fmt_int(row.paths),
                    _fmt_int(row.batches),
                    _fmt_float(row.runtime_sec, 3),
                    _fmt_float(row.diff_vs_mc, 4),
                    rel_diff,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(args: argparse.Namespace, rows: List[EngineRunResult]) -> str:
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_hint = "external/ki_variant_phx/phoenix_example.txt"
    lines = [
        "# Phoenix External Case Engine Comparison",
        "",
        f"- Generated at: {run_time}",
        f"- Case source: `{source_hint}`",
        "- Product: Variable-KI Phoenix (lock period represented via unreachable KO barriers for first 2 observations)",
        "- KI monitoring: Business-day-only discrete observations (China SSE calendar)",
        "- KI behavior: KI hit does not terminate product, and KO remains active after KI",
        f"- Day-count convention: BUSINESS_DAYS with {BUS_DAYS_IN_YEAR} business days/year",
        f"- KO extra interest (ko_rate): {KO_RATE:.2%}",
        "- Coupon accrual mode: fixed 1/12 per period",
        "- DeltaCash: central spot bump +/-1% (reported as delta * spot)",
        "- Vega(1%): central vol bump +/-1 vol point (reported as 1% vol price change)",
        f"- MC paths: {args.mc_paths:,}",
        f"- QMC paths: {args.qmc_paths:,}",
        (
            "- RQMC paths: "
            f"{args.rqmc_total_paths:,} total "
            f"({args.rqmc_batches} batches x {max(1, args.rqmc_total_paths // max(1, args.rqmc_batches)):,} paths)"
        ),
        f"- PDE params: grid={args.pde_grid}, steps={args.pde_steps}, max_time_steps={args.pde_max_steps}",
        f"- QUAD params: grid_points={args.quad_grid}, num_std_devs={args.quad_std_devs}",
        "",
        "## Comparison Table",
        "",
        build_markdown_table(rows),
        "",
        "## External Reference Values (from case text)",
        "",
        "- Participation 100%: PDE 15,724.1284, MC benchmark 15,944.3999",
        "- Participation 60%: PDE -17,952.3251, MC benchmark -17,668.9723",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price external Phoenix example with MC/QMC/RQMC/PDE/QUAD and output comparison table."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external/ki_variant_phx/phoenix_engine_comparison_results.md"),
        help="Output markdown file path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for MC engines.")

    # Requested path settings
    parser.add_argument("--mc-paths", type=int, default=100_000, help="MC paths.")
    parser.add_argument("--qmc-paths", type=int, default=100_000, help="QMC paths.")
    parser.add_argument(
        "--rqmc-total-paths",
        type=int,
        default=100_000,
        help="RQMC total paths (distributed across rqmc-batches).",
    )
    parser.add_argument(
        "--rqmc-batches",
        type=int,
        default=10,
        help="RQMC batch count used to realize total paths.",
    )

    parser.add_argument("--pde-grid", type=int, default=700, help="PDE spatial grid size.")
    parser.add_argument("--pde-steps", type=int, default=360, help="PDE time steps.")
    parser.add_argument(
        "--pde-max-steps",
        type=int,
        default=5000,
        help="PDE max time steps when auto grid is enabled.",
    )
    parser.add_argument(
        "--pde-log-dx",
        type=float,
        default=0.0025,
        help="PDE log-dx target for auto grid.",
    )
    parser.add_argument("--quad-grid", type=int, default=1001, help="QUAD grid points.")
    parser.add_argument(
        "--quad-std-devs",
        type=float,
        default=10.0,
        help="QUAD log-domain width in standard deviations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = create_pricing_env()

    all_rows: List[EngineRunResult] = []
    all_rows.extend(
        run_case("KI participation 100%", participation_rate=1.0, env=env, args=args)
    )
    all_rows.extend(
        run_case("KI participation 60%", participation_rate=0.6, env=env, args=args)
    )

    report = build_report(args, all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved comparison report to: {args.output}")


if __name__ == "__main__":
    main()
