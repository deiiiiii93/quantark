"""DCN MC<->PDE cross-validation gate report generator (spec acceptance #3).

Runs both sample contracts on the flat Q1 market through the MC and PDE
engines at the decided validation settings, evaluates the gate
|dPV| < max(3*stderr, 5bp*N), checks flat-curve recovery, and writes
model-validation-output/dcn/gate_report.json + GATE_REPORT.md.

Every number in the artifacts comes from this run; none are hand-computed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.product.option.dcn_option import DCNDirection, DCNOption
from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.priceenv.flat_builders import build_flat_env
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar

# --- flat Q1 market and the two sample contracts (problem statement) -------
FLAT = dict(r=0.0356, q=0.1406, sigma=0.184)
SPOT = 6000.0
GATE_SETTINGS = {"DCN-A": (2 ** 20, 16), "DCN-B": (2 ** 21, 32)}

CONTRACTS = {
    "DCN-A": dict(
        initial_date=datetime(2023, 1, 3), valuation_date=datetime(2023, 1, 3),
        maturity_date=datetime(2025, 1, 3), tenor_months=24, lock_months=3,
        ko_lock_months=3, coupon_settlement_offset=2, ko_settlement_offset=2,
        settlement_date=datetime(2025, 1, 7), ki_put_strike_ratio=1.10,
    ),
    "DCN-B": dict(
        initial_date=datetime(2023, 1, 3), valuation_date=datetime(2023, 1, 3),
        maturity_date=datetime(2026, 1, 5), tenor_months=36, lock_months=3,
        ko_lock_months=6, coupon_settlement_offset=0, ko_settlement_offset=0,
        settlement_date=datetime(2026, 1, 7), ki_put_strike_ratio=1.15,
    ),
}
COMMON = dict(
    notional=1_000_000.0, initial_price=SPOT, coupon_barrier_ratio=0.80,
    ko_barrier_ratio=1.00, ki_barrier_ratio=0.75, coupon_rate=0.12,
    ko_coupon_rate=0.12, participation=1.0, coupon_counted_days=30,
    coupon_days_denom=360,
)


def _flat_env():
    from quantark.param import (
        ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
    )
    from quantark.priceenv import PricingEnvironment

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT),
        vol_surface=FlatVolSurface(volatility=FLAT["sigma"]),
        rate_curve=FlatRateCurve(rate=FLAT["r"]),
        div_yield=ContinuousDividendYield(div_yield=FLAT["q"]),
        valuation_date=datetime(2023, 1, 3),
        day_count_convention=DayCountConvention.ACT_365,
    )


def _build_product(name: str) -> DCNOption:
    c = CONTRACTS[name]
    calendar = create_calendar(CalendarType.CHINA_SSE)
    schedule = build_dcn_schedule(
        calendar=calendar,
        **{k: c[k] for k in (
            "initial_date", "maturity_date", "tenor_months", "lock_months",
            "ko_lock_months", "coupon_settlement_offset",
            "ko_settlement_offset", "valuation_date", "settlement_date",
        )},
    )
    return DCNOption(
        direction=DCNDirection.BUYER,
        ki_put_strike_ratio=c["ki_put_strike_ratio"],
        schedule=schedule,
        settlement_date=c["settlement_date"],
        **COMMON,
    )


def main() -> None:
    out_dir = Path("model-validation-output/dcn")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _flat_env()
    report = {"market": FLAT, "spot": SPOT, "contracts": {}}

    for name in ("DCN-A", "DCN-B"):
        product = _build_product(name)
        paths, batches = GATE_SETTINGS[name]
        mc = DCNMCEngine(
            num_paths=paths, num_batches=batches, seed=42
        ).price_detailed(product, env)
        pde = DCNPDEEngine().price_detailed(product, env)
        notional = COMMON["notional"]
        gate = max(3.0 * mc.std_error, 5e-4 * notional)
        diff = abs(mc.pv - pde.pv)

        # flat recovery: same seeds through flat term-structure curves
        curve_env = build_flat_env(
            spot=SPOT, tenors=[0.5, 1.0, 1.5, 2.1, 3.1],
            valuation_date=datetime(2023, 1, 3), **FLAT,
        )
        pv_curves = DCNMCEngine(
            num_paths=2 ** 16, seed=42
        ).price(product, curve_env)
        pv_flat = DCNMCEngine(num_paths=2 ** 16, seed=42).price(product, env)
        flat_residual = abs(pv_curves - pv_flat)

        report["contracts"][name] = {
            "mc": mc.to_dict(),
            "pde": pde.to_dict(),
            "gate": {
                "definition": "|dPV| < max(3*stderr, 5bp*N)",
                "stderr_requirement_bp_of_notional": 2.0,
                "stderr_bp": mc.std_error / notional * 1e4,
                "abs_diff": diff,
                "gate_value": gate,
                "passed": bool(diff < gate),
            },
            "flat_recovery_residual": flat_residual,
        }
        status = "PASS" if diff < gate else "FAIL"
        print(f"{name}: MC={mc.pv:,.1f} +/- {mc.std_error:,.1f}  "
              f"PDE={pde.pv:,.1f}  |dPV|={diff:,.1f}  gate={gate:,.1f}  "
              f"[{status}]  flat-recovery={flat_residual:.2e}")

    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# DCN MC <-> PDE Cross-Validation Gate Report",
        "",
        f"Flat market: r={FLAT['r']:.4f}, q={FLAT['q']:.4f}, "
        f"sigma={FLAT['sigma']:.4f}, S0={SPOT:,.0f}",
        "",
        "Gate (decided at kickoff): `|dPV| < max(3*MC stderr, 5bp*N)`; "
        "MC stderr must be < 2bp*N (raise paths, never widen the gate).",
        "",
        "| Contract | Paths | MC PV | stderr | PDE PV | \\|dPV\\| | Gate | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in report["contracts"].items():
        g, mc_d, pde_d = c["gate"], c["mc"], c["pde"]
        lines.append(
            f"| {name} | {mc_d['num_paths']:,} | {mc_d['pv']:,.1f} "
            f"| {mc_d['std_error']:,.1f} | {pde_d['pv']:,.1f} "
            f"| {g['abs_diff']:,.1f} | {g['gate_value']:,.1f} "
            f"| {'PASS' if g['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Leg decomposition (signed, BUYER)",
        "",
        "| Contract | Fixed coupons | KO coupons | Loss leg | KI prob | KO prob |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in report["contracts"].items():
        m = c["mc"]
        lines.append(
            f"| {name} | {m['pv_fixed_coupons']:,.1f} "
            f"| {m['pv_ko_coupons']:,.1f} | {m['pv_loss_leg']:,.1f} "
            f"| {m['ki_probability']:.4f} | {m['ko_probability']:.4f} |"
        )
    lines += [
        "",
        "Flat recovery (same seed, flat term-structure curves vs flat "
        "scalars): "
        + ", ".join(
            f"{n}: {c['flat_recovery_residual']:.2e}"
            for n, c in report["contracts"].items()
        ),
        "",
        "Determinism: fixed seed 42 throughout; engines bit-reproduce in the "
        "same environment (asserted in test_dcn_mc_engine / test_dcn_pde_solver).",
        "",
        "_Generated by example/dcn_gate_report_demo.py — all numbers from "
        "the run, none hand-computed._",
    ]
    (out_dir / "GATE_REPORT.md").write_text("\n".join(lines))
    print(f"wrote {out_dir}/gate_report.json and {out_dir}/GATE_REPORT.md")


if __name__ == "__main__":
    main()
