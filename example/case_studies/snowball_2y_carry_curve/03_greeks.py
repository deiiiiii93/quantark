"""Stage 03 — static Greeks: flat q vs CarryCurve, per tail convention.

Reads data/market_20261105.json and results/pricing.json (the solved coupon);
writes results/greeks.json, per tail:
- curve model: ``calculate_carry_curve_buckets`` — spot delta, pinned-forward
  delta, delta per pillar (∂PV/∂F_i) with hedge lots (IM 200/pt, MO2709 one
  C−P pair = 100·DF), rho_q per pillar and parallel, the two identity
  residuals; the same table at halved bumps (AC5 stability);
- flat model: spot delta, spot-equivalent IM lots, parallel rho_q;
- forward-interval rho_q on (0,T1], …, (T5,T_mat] (where the exposure lives);
- scenarios: ΔPV under the curve vs the flat model (q_flat re-read off the
  shocked curve) vs the flat Greek's prediction, and the missed bp.

Run:
    python example/case_studies/snowball_2y_carry_curve/03_greeks.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _common():
    name = "snowball_2y_cc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def main() -> int:
    from quantark.asset.equity.riskmeasures import calculate_carry_curve_buckets
    from quantark.param import ContinuousDividendYield

    c = _common()
    started = time.time()
    snap = c.load_snapshot()
    pricing = c.read_json(c.RESULTS_DIR / "pricing.json")
    s0 = float(snap["spot"])
    terms = c.build_terms()
    T = terms.maturity_years
    product = c.build_product(pricing["coupon"]["value"], s0, terms)
    quad = c.quad_engine()
    labels = c.pillar_labels(snap)
    mults = c.lot_multipliers(snap)
    rate = c.rate_curve()

    def pv(div_yield, spot=None) -> float:
        return float(quad.price(product, c.make_env(snap, div_yield, spot)))

    tails = {}
    for tail in c.TAILS:
        curve = c.build_curve(snap, tail)
        view = curve.to_dividend_yield(rate)
        env = c.curve_env(snap, tail)
        report = calculate_carry_curve_buckets(
            product, env, quad, curve, labels=labels, lot_multipliers=mults,
            spot_bump_rel=c.SPOT_BUMP_REL, q_bump=c.Q_BUMP,
        )
        half = calculate_carry_curve_buckets(
            product, env, quad, curve, labels=labels, lot_multipliers=mults,
            spot_bump_rel=0.5 * c.SPOT_BUMP_REL, forward_bump_rel=0.5e-3, q_bump=0.5 * c.Q_BUMP,
        )

        # flat model
        q_flat = c.flat_q(curve, T)
        flat = ContinuousDividendYield(q_flat)
        pv_flat = pv(flat)
        s_up, s_dn = s0 * (1 + c.SPOT_BUMP_REL), s0 * (1 - c.SPOT_BUMP_REL)
        flat_delta = (pv(flat, s_up) - pv(flat, s_dn)) / (s_up - s_dn)
        flat_rho = (pv(ContinuousDividendYield(q_flat + c.Q_BUMP))
                    - pv(ContinuousDividendYield(q_flat - c.Q_BUMP))) / (2 * c.Q_BUMP) * 0.01

        # forward-interval rho_q (curve model)
        edges = c.interval_edges(curve, T)
        interval_labels = [f"0-{labels[0]}"] + [f"{a}-{b}" for a, b in zip(labels, labels[1:])] + [f"{labels[-1]}-T_mat"]
        intervals = []
        for (a, b), lab in zip(zip(edges, edges[1:]), interval_labels):
            up = pv(c.ForwardIntervalBumpDividendYield(view, a, b, c.Q_BUMP))
            dn = pv(c.ForwardIntervalBumpDividendYield(view, a, b, -c.Q_BUMP))
            intervals.append({"label": lab, "start": a, "end": b, "rho_q": (up - dn) / (2 * c.Q_BUMP) * 0.01})
        interval_sum = sum(x["rho_q"] for x in intervals)

        # scenarios
        base_curve_pv = report.base_pv
        scenario_defs = [("parallel q +1%", view.parallel_shifted(c.SCENARIO_SHOCK))]
        for i, lab in enumerate(labels):
            scenario_defs.append((f"{lab} q +1%", c.node_yield_bumped(curve, {i: c.SCENARIO_SHOCK}).to_dividend_yield(rate)))
        scenario_defs.append(("front steepener: IM2611+IM2612 q +2%",
                              c.node_yield_bumped(curve, {0: c.STEEPENER_SHOCK, 1: c.STEEPENER_SHOCK}).to_dividend_yield(rate)))
        scenario_defs.append(("tail only: forward q +1% beyond last pillar",
                              c.ForwardIntervalBumpDividendYield(view, curve.tenors[-1], math.inf, c.SCENARIO_SHOCK)))
        scenarios = []
        for name, shocked in scenario_defs:
            q_flat_new = float(shocked.get_yield(T))
            d_curve = pv(shocked) - base_curve_pv
            d_flat = pv(ContinuousDividendYield(q_flat_new)) - pv_flat
            predicted = flat_rho * (q_flat_new - q_flat) / 0.01
            scenarios.append({
                "scenario": name,
                "dq_at_maturity": q_flat_new - q_flat,
                "dpv_curve_bp": c.bp(d_curve),
                "dpv_flat_bp": c.bp(d_flat),
                "flat_greek_prediction_bp": c.bp(predicted),
                "missed_bp": c.bp(d_curve - d_flat),
            })

        rep = asdict(report)
        rep["buckets"] = [asdict(b) for b in report.buckets]
        tails[tail] = {
            "label": c.TAILS[tail][1],
            "curve": rep,
            "curve_half_bumps": {"spot_delta": half.spot_delta, "pinned_forward_delta": half.pinned_forward_delta,
                                 "parallel_rho_q": half.parallel_rho_q,
                                 "buckets": [{"delta": b.delta, "rho_q": b.rho_q} for b in half.buckets]},
            "curve_spot_lots": -report.spot_delta / c.IM_MULTIPLIER,
            "pinned_forward_lots": -report.pinned_forward_delta / c.IM_MULTIPLIER,
            "flat": {"q": q_flat, "pv": pv_flat, "spot_delta": flat_delta,
                     "spot_lots": -flat_delta / c.IM_MULTIPLIER, "parallel_rho_q": flat_rho},
            "forward_interval_rho": intervals,
            "forward_interval_sum": interval_sum,
            "scenarios": scenarios,
        }
        print(f"\n== {tail}: q_flat {q_flat:.4%}")
        print(f"   spot delta  curve {report.spot_delta:12,.1f}  flat {flat_delta:12,.1f}  CNY/pt"
              f"  (lots {-report.spot_delta / 200:7.2f} vs {-flat_delta / 200:7.2f});"
              f" pinned-fwd {report.pinned_forward_delta:10,.1f}; identity resid {report.delta_identity_residual:.2f}")
        for b in report.buckets:
            print(f"   {b.label:12s} dPV/dF {b.delta:12,.1f}  lots {b.hedge_lots:8.2f}  rho_q {c.bp(b.rho_q):8.2f} bp"
                  f"{'  (tail)' if b.extrapolated_tail else ''}")
        print(f"   rho_q parallel curve {c.bp(report.parallel_rho_q):8.2f} bp  flat {c.bp(flat_rho):8.2f} bp;"
              f" sum resid {c.bp(report.rho_sum_residual):.3f} bp; intervals sum {c.bp(interval_sum):8.2f} bp")
        print("   intervals " + "  ".join(f"{x['label']} {c.bp(x['rho_q']):.2f}" for x in intervals))
        for s in scenarios:
            print(f"   {s['scenario']:45s} curve {s['dpv_curve_bp']:8.2f}  flat {s['dpv_flat_bp']:8.2f}"
                  f"  missed {s['missed_bp']:8.2f} bp")

    payload = {
        "valuation_date": snap["valuation_date"],
        "coupon": pricing["coupon"]["value"],
        "notional": c.NOTIONAL,
        "spot": s0,
        "engine": {"quad_readout": c.QUAD_READOUT, "quad_grid_points": quad.params.grid_points},
        "bumps": {"spot_rel": c.SPOT_BUMP_REL, "forward_rel": 1e-3, "q": c.Q_BUMP,
                  "rho_unit": "+1% q", "scenario_shock": c.SCENARIO_SHOCK},
        "pillars": [{"label": lab, "lot_multiplier": m} for lab, m in zip(labels, mults)],
        "tails": tails,
        "runtime_seconds": time.time() - started,
    }
    out = c.RESULTS_DIR / "greeks.json"
    c.write_json(out, payload)
    print(f"\nwrote {out} in {payload['runtime_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
