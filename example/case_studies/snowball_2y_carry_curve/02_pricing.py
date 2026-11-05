"""Stage 02 — price the 2Y snowball under the curve and the flat q, per tail.

Reads data/market_20261105.json; writes results/pricing.json:
- the term sheet and the coupon solved to PV = 0 under the curve with the
  hold-forward tail (decision 9), held fixed for every cell;
- per tail convention: the carry curve (nodes, q per pillar, q(T_mat), the
  tail leverage ∂B(T_mat)/∂B_i), curve PV, flat PV, the gap in bp and as a
  coupon equivalent, and QUAD event statistics for both models;
- QUAD grid doubling on the resolved grid, PDE cross-check per cell;
- the mark-sensitivity rows (MO close; all-CFFEX-settlement chain).

Run:
    python example/case_studies/snowball_2y_carry_curve/02_pricing.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
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
    c = _common()
    started = time.time()
    snap = c.load_snapshot()
    s0 = float(snap["spot"])
    terms = c.build_terms()
    T = terms.maturity_years
    quad = c.quad_engine()

    base_env = c.curve_env(snap, c.COUPON_TAIL)
    sol = c.solve_coupon(lambda cp: quad.price(c.build_product(cp, s0, terms), base_env))
    product = c.build_product(sol.coupon, s0, terms)
    grid = c.resolved_quad_grid(quad, product, base_env)
    quad_fine = c.quad_engine(grid_points=2 * grid + 1)
    grid_fine = c.resolved_quad_grid(quad_fine, product, base_env)
    pde = c.pde_engine()
    print(f"coupon {sol.coupon:.6%}  (PV {sol.pv:.2e}, annuity {sol.annuity:,.0f} CNY per unit coupon)")

    def cell(env):
        pv = float(quad.price(product, env))
        stats = quad.calculate_event_stats(product, env)
        return {
            "pv": pv,
            "pv_bp": c.bp(pv),
            "pv_bp_fine_grid": c.bp(float(quad_fine.price(product, env))),
            "pv_bp_pde": c.bp(float(pde.price(product, env))),
            "events": c.event_summary(stats, T),
        }

    tails = {}
    for tail, (ext, label) in c.TAILS.items():
        curve = c.build_curve(snap, tail)
        view = curve.to_dividend_yield(c.rate_curve())
        q_flat = c.flat_q(curve, T)
        curve_cell = cell(c.curve_env(snap, tail))
        flat_cell = cell(c.flat_env(snap, tail, T))
        gap = flat_cell["pv"] - curve_cell["pv"]
        tails[tail] = {
            "label": label,
            "extrapolation": ext.value,
            "q_flat": q_flat,
            "flat_forward_matches_curve": c.flat_forward_matches(curve, q_flat, T, s0),
            "nodes": [{"label": lab, "tenor": t, "carry": b, "q": float(view.get_yield(t)),
                       "forward": float(curve.forward(s0, t))}
                      for lab, (t, b) in zip(c.pillar_labels(snap), curve.nodes)],
            "tail_leverage": c.tail_leverage(curve, T),
            "curve": curve_cell,
            "flat": flat_cell,
            "gap_bp": c.bp(gap),
            "gap_bp_fine_grid": flat_cell["pv_bp_fine_grid"] - curve_cell["pv_bp_fine_grid"],
            "gap_bp_pde": flat_cell["pv_bp_pde"] - curve_cell["pv_bp_pde"],
            "coupon_equiv_gap": gap / sol.annuity,
        }
        print(f"{tail:13s} q_flat {q_flat:.4%}  curve {curve_cell['pv_bp']:8.2f} bp  flat {flat_cell['pv_bp']:8.2f} bp"
              f"  gap {c.bp(gap):7.2f} bp = {gap / sol.annuity:+.4%} coupon"
              f"  | PDE gap {tails[tail]['gap_bp_pde']:7.2f}  fine-grid gap {tails[tail]['gap_bp_fine_grid']:7.2f}")

    marks = {}
    for mark in c.MARKS:
        if mark == "base":
            continue
        rows = {}
        for tail in c.TAILS:
            curve = c.build_curve(snap, tail, mark)
            pv_c = float(quad.price(product, c.curve_env(snap, tail, mark)))
            pv_f = float(quad.price(product, c.flat_env(snap, tail, T, mark)))
            rows[tail] = {"q_flat": c.flat_q(curve, T), "curve_bp": c.bp(pv_c), "flat_bp": c.bp(pv_f),
                          "gap_bp": c.bp(pv_f - pv_c),
                          "q_nodes": [float(curve.to_dividend_yield(c.rate_curve()).get_yield(t))
                                      for t in curve.tenors]}
        marks[mark] = {"description": c.MARKS[mark], "tails": rows}
        print(f"mark {mark}: " + "  ".join(f"{t} gap {r['gap_bp']:.2f} bp (curve {r['curve_bp']:.2f})"
                                          for t, r in rows.items()))

    payload = {
        "valuation_date": snap["valuation_date"],
        "spot": s0,
        "rate": c.RATE,
        "vol": snap["vol"]["value"],
        "notional": c.NOTIONAL,
        "terms": {
            "maturity_date": terms.maturity_date.isoformat(),
            "maturity_years": T,
            "ko_pct": c.KO_PCT, "ki_pct": c.KI_PCT,
            "ko_dates": [d.isoformat() for d in terms.ko_dates],
            "n_ki_observations": len(terms.ki_times),
        },
        "coupon": {"value": sol.coupon, "pv": sol.pv, "annuity": sol.annuity,
                   "tolerance": sol.tolerance, "solved_under": f"curve/{c.COUPON_TAIL}"},
        "engine": {"quad_readout": c.QUAD_READOUT, "quad_grid_resolved": grid,
                   "quad_grid_fine_resolved": grid_fine, "pde_accuracy": "high"},
        "tails": tails,
        "marks": marks,
        "runtime_seconds": time.time() - started,
    }
    out = c.RESULTS_DIR / "pricing.json"
    c.write_json(out, payload)
    print(f"wrote {out} in {payload['runtime_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
