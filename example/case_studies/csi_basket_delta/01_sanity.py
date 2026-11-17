"""
Stage 1 — sanity: the pricer reproduces what it must before any sweep is read.

* one-name basket == Black-Scholes (MM exactly, MC within 3 SE), with and
  without a decrement and a mid-period stub, call and put;
* the pathwise Euler identity sum_i S_i dV/dS_i = I dV/dI per path;
* MC sample moments of I_T == the closed form, for every rebalance rule and the
  singular / extreme correlations (uniform rho = 1, two-name rho = -1);
* the exchangeable control (all methods agree) and the concentration basket
  (identical vols, unequal weights: the heavier name carries more delta).

Writes results/01_sanity.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as cm  # noqa: E402

RESULTS = HERE / "results"
REPLICATE_SEEDS = (1000, 5000, 9000, 77777)


def _bs(s, k, t, sigma, r, q, is_call):
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / v
    d2 = d1 - v
    if is_call:
        return (
            s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2),
            math.exp(-q * t) * norm.cdf(d1),
        )
    return (
        k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1),
        -math.exp(-q * t) * norm.cdf(-d1),
    )


def single_name_checks() -> list:
    rows = []
    cases = [
        ("rule A, d = 0", "none", 0.0, 0.0, 1.0),
        ("monthly, d = 5%, 0.4P elapsed, +7%", "monthly", 0.05, 0.4, 1.07),
    ]
    for label, rule, dec, frac, move in cases:
        for is_call in (True, False):
            basket = cm.CSIBasket(
                names=("A",),
                target_weights=np.array([1.0]),
                vols=np.array([0.3]),
                div_yields=np.array([0.02]),
                corr=np.array([[1.0]]),
                rate=0.025,
                decrement=dec,
                rebalance=rule,
            )
            period = cm.period_length(rule) or 0.0
            state = cm.CSIState(1000.0, np.array([50.0]), np.array([move]), frac * period)
            option = cm.CSIOption(1.0, 1.0, is_call)
            q = cm.state_quantities(basket, state)
            n, s = q.shares[0], q.spots[0]
            bs_p, bs_d = _bs(s, option.strike(q.index_level) / n, 1.0, 0.3, 0.025, 0.02 + dec, is_call)
            mm = cm.mm_price_and_deltas(basket, state, option)
            mc = cm.mc_price_and_deltas(basket, state, option, cm.STUDY_PATHS, cm.STUDY_BATCHES)
            rows.append(
                {
                    "case": label,
                    "is_call": is_call,
                    "bs_price": n * bs_p,
                    "mm_price": mm.price,
                    "mc_price": mc.price,
                    "mc_price_se": mc.price_se,
                    "mm_price_rel_err": mm.price / (n * bs_p) - 1.0,
                    "mc_price_z": (mc.price - n * bs_p) / mc.price_se,
                    "bs_delta": n * bs_d,
                    "mm_delta": float(mm.deltas[0]),
                    "mc_delta": float(mc.deltas_pathwise[0]),
                    "mc_delta_z": float((mc.deltas_pathwise[0] - n * bs_d) / mc.deltas_pathwise_se[0]),
                }
            )
    return rows


def euler_check() -> dict:
    basket = cm.base_basket("monthly", decrement=0.05)
    moves = np.ones(5)
    moves[basket.names.index("TECH")] = 1.3
    state = cm.default_state(basket, elapsed=0.4 * cm.period_length("monthly"), moves=moves)
    option = cm.CSIOption(1.0, 1.0)
    mc = cm.mc_price_and_deltas(basket, state, option, cm.STUDY_PATHS, cm.STUDY_BATCHES)
    q = cm.state_quantities(basket, state)
    return {
        "case": "base, monthly, d = 5%, 0.4P elapsed, TECH +30%",
        "euler_max_rel_err_per_path": mc.euler_max_rel_err,
        "sum_S_delta": float(np.dot(q.spots, mc.deltas_pathwise)),
        "I_delta_index": mc.delta_index * q.index_level,
    }


def moment_checks() -> list:
    rows = []
    cases = [
        ("base, rule A", cm.base_basket("none"), 0.0),
        ("base, quarterly", cm.base_basket("quarterly"), 0.0),
        ("base, monthly", cm.base_basket("monthly"), 0.0),
        ("base, weekly", cm.base_basket("weekly"), 0.0),
        ("base, daily", cm.base_basket("daily"), 0.0),
        ("base, monthly, d = 5%, 0.4P", cm.base_basket("monthly", decrement=0.05), 0.4),
        ("uniform rho = 1 (singular)", cm.base_basket("none", corr=cm.uniform_corr(5, 1.0)), 0.0),
        ("uniform rho = -0.25 (PSD floor)", cm.base_basket("none", corr=cm.uniform_corr(5, -0.25)), 0.0),
        ("two-name rho = -1", cm.two_name_basket(-1.0), 0.0),
        ("two-name rho = -1, monthly, 0.4P", cm.two_name_basket(-1.0, "monthly"), 0.4),
    ]
    for label, basket, frac in cases:
        period = cm.period_length(basket.rebalance) or 0.0
        state = cm.default_state(basket, elapsed=frac * period)
        option = cm.CSIOption(1.0, 1.0)
        mc = cm.mc_price_and_deltas(basket, state, option, cm.STUDY_PATHS, cm.STUDY_BATCHES)
        mm = cm.mm_price_and_deltas(basket, state, option)
        row = {
            "case": label,
            "m1_closed": mm.m1,
            "m1_mc": mc.sample_m1,
            "m1_z": (mc.sample_m1 - mm.m1) / mc.sample_m1_se,
            "m2_closed": mm.m2,
            "m2_mc": mc.sample_m2,
            "m2_z": (mc.sample_m2 - mm.m2) / mc.sample_m2_se,
            "euler_max_rel_err": mc.euler_max_rel_err,
            "price_mc": mc.price,
            "price_mc_se": mc.price_se,
            "price_mm": mm.price,
            "replicates": [],
        }
        # A |z| > 3 among ~20 checks happens by chance; replicate it with
        # independent scrambles (seeds spaced by >= n_batches) and keep both.
        if max(abs(row["m1_z"]), abs(row["m2_z"])) > cm.SE_MULTIPLE:
            for seed in REPLICATE_SEEDS:
                rep = cm.mc_price_and_deltas(
                    basket, state, option, cm.STUDY_PATHS, cm.STUDY_BATCHES, seed=seed
                )
                row["replicates"].append(
                    {
                        "seed": seed,
                        "m1_z": (rep.sample_m1 - mm.m1) / rep.sample_m1_se,
                        "m2_z": (rep.sample_m2 - mm.m2) / rep.sample_m2_se,
                    }
                )
        rows.append(row)
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    out = {
        "stage": "01_sanity",
        "single_name": single_name_checks(),
        "euler": euler_check(),
        "moments": moment_checks(),
        "control": cm.evaluate_scenario(
            cm.control_basket("monthly"),
            cm.default_state(cm.control_basket("monthly")),
            cm.CSIOption(1.0, 1.0),
            "exchangeable control, monthly, at a rebalance date, 1Y ATM call",
        ),
        "concentration": cm.evaluate_scenario(
            cm.concentration_basket(),
            cm.default_state(cm.concentration_basket()),
            cm.CSIOption(1.0, 1.0),
            "concentration basket (equal vols, base weights), rule A, 1Y ATM call",
        ),
    }
    path = RESULTS / "01_sanity.json"
    path.write_text(json.dumps(out, indent=1))
    worst_z = max(abs(r["mc_price_z"]) for r in out["single_name"])
    worst_m = max(max(abs(r["m1_z"]), abs(r["m2_z"])) for r in out["moments"])
    print(f"wrote {path}")
    print(f"single-name: max |MM/BS - 1| = {max(abs(r['mm_price_rel_err']) for r in out['single_name']):.2e}, max |MC z| = {worst_z:.2f}")
    print(f"moments: max |z| = {worst_m:.2f}; Euler per path = {out['euler']['euler_max_rel_err_per_path']:.1e}")


if __name__ == "__main__":
    main()
