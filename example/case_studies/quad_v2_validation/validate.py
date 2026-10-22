"""Reproducible QUAD V2 accuracy, consistency and workload qualification.

Run from the repository root with one BLAS/OMP/Accelerate thread. This uses
the frozen independent audit inputs/results without modifying them.
"""
from dataclasses import asdict, replace
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "example/gaussian_quad_comparison"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUDIT))
from contracts import cases
from study import product, environment, reference
from quantark.asset.equity.engine.quad import SnowballQuadEngine, SnowballQuadEngineV2
from quantark.asset.equity.param import QuadParams, QuadV2Params
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator


def timed(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return dict(
        median_s=statistics.median(samples),
        min_s=min(samples),
        max_s=max(samples),
        samples_s=samples,
    )


def traced_peak(fn):
    """Separate allocation probe, outside all latency samples (not process RSS)."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def gate(got, expected):
    errors = {
        k: float(np.max(np.abs(np.asarray(got[k]) - expected[k]))) for k in expected
    }
    passed = {
        "price": bool(
            np.all(np.abs(np.asarray(got["price"]) - expected["price"]) <= 1e-4)
        ),
        "delta": bool(
            np.all(
                np.abs(np.asarray(got["delta"]) - expected["delta"])
                <= 1e-5 + 1e-4 * np.abs(expected["delta"])
            )
        ),
    }
    if "gamma" in expected:
        passed["gamma"] = bool(
            np.all(
                np.abs(np.asarray(got["gamma"]) - expected["gamma"])
                <= 1e-6 + 1e-3 * np.abs(expected["gamma"])
            )
        )
    return dict(max_abs_errors=errors, passed=passed, all_passed=all(passed.values()))


def matrix():
    saved = json.loads((AUDIT / "results/matrix.json").read_text())
    refs = {r["case"]["name"]: r["reference"]["fine"] for r in saved["results"]}
    result = []
    for c in cases():
        engine = SnowballQuadEngineV2()
        p, e = product(c), environment(c)
        got = engine.calculate_point_greeks(p, e)
        expected = {k: np.array(refs[c.name][k]) for k in ("price", "delta")}
        row = dict(
            name=c.name,
            case=asdict(c),
            outputs=got,
            reference={k: v.tolist() for k, v in expected.items()},
            gate=gate(got, expected),
            diagnostics=dict(engine._prepared.diagnostics),
        )
        assert row["gate"]["all_passed"], row
        result.append(row)
        print("matrix", c.name, flush=True)
    return result


def workload(c, repeats):
    p, e = product(c), environment(c)
    spots = np.linspace(0.8 * c.spot, 1.2 * c.spot, 101)
    risk_names = ["delta", "gamma", "vega", "rho", "dividend_rho", "theta"]
    fine_params = QuadV2Params(order=10, cells_per_sd=3.0, domain_sd=12.0)
    fine = SnowballQuadEngineV2(fine_params)
    fc = fine.prepare(p, e, np.r_[spots, c.spot * 0.99, c.spot * 1.01])
    curve_ref = fc.evaluate(spots)
    pv = fc.evaluate([c.spot * 0.99, c.spot, c.spot * 1.01])["price"]
    desk_ref = dict(
        price=pv[1],
        delta=(pv[2] - pv[0]) / (0.02 * c.spot),
        gamma=(pv[2] - 2 * pv[1] + pv[0]) / (0.01 * c.spot) ** 2,
    )

    def risk_run(cls, params):
        return GreeksCalculator(params).calculate(p, e, cls(params), greeks=risk_names)

    risk_ref = risk_run(SnowballQuadEngineV2, fine_params)

    def risk_gate(got):
        errors = {k: abs(got[k] - risk_ref[k]) for k in risk_names}
        passed = {
            k: err
            <= (
                1e-6 + 1e-3 * abs(risk_ref[k])
                if k == "gamma"
                else 1e-5 + 1e-4 * abs(risk_ref[k])
                if k == "delta"
                else 1e-4 + 1e-4 * abs(risk_ref[k])
            )
            for k, err in errors.items()
        }
        return dict(
            max_abs_errors=errors, passed=passed, all_passed=all(passed.values())
        )

    def checks(cls, params):
        engine = cls(params)
        desk = engine.calculate_greeks(p, e)
        curve = engine.calculate_spot_greeks_curve(p, e, spots)
        curve_values = {
            k: np.array([r[k] for r in curve]) for k in ("price", "delta", "gamma")
        }
        risks = risk_run(cls, params)
        return dict(
            desk=gate(desk, desk_ref),
            curve=gate(
                curve_values, {k: curve_ref[k] for k in ("price", "delta", "gamma")}
            ),
            risk=risk_gate(risks),
            risk_outputs=risks,
        )

    ladder = []
    selected = {}
    for n in (1001, 2001, 4001, 8001, 16001):
        params = QuadParams(grid_points=n, readout="transition")
        gates = checks(SnowballQuadEngine, params)
        ladder.append(dict(grid_points=n, **gates))
        for kind in ("desk", "curve", "risk"):
            if kind not in selected and gates[kind]["all_passed"]:
                selected[kind] = (params, gates[kind])
        print(
            "qualify",
            c.name,
            n,
            {k: gates[k]["all_passed"] for k in ("desk", "curve", "risk")},
            flush=True,
        )
        if len(selected) == 3:
            break
    for kind in ("desk", "curve", "risk"):
        if kind not in selected:
            selected[kind] = (params, gates[kind])
    output = {}
    v2params = QuadV2Params()
    v2gates = checks(SnowballQuadEngineV2, v2params)
    for name, cls in [
        ("v1_selected", SnowballQuadEngine),
        ("v2", SnowballQuadEngineV2),
    ]:
        settings = {
            k: (selected[k][0] if name == "v1_selected" else v2params)
            for k in ("desk", "curve", "risk")
        }
        result = {
            f"{k}_gate": selected[k][1] if name == "v1_selected" else v2gates[k]
            for k in settings
        }
        result["settings"] = {k: str(v) for k, v in settings.items()}
        result["cold_price_desk_greeks"] = timed(
            lambda: cls(settings["desk"]).calculate_greeks(p, e), repeats
        )
        result["cold_curve_101"] = timed(
            lambda: cls(settings["curve"]).calculate_spot_greeks_curve(p, e, spots),
            repeats,
        )
        result["cold_full_risk_batch"] = timed(
            lambda: risk_run(cls, settings["risk"]), repeats
        )
        result["curve_tracemalloc_peak_bytes"] = traced_peak(
            lambda: cls(settings["curve"]).calculate_spot_greeks_curve(p, e, spots)
        )
        if name == "v2":
            context = cls(v2params).prepare(p, e, spots)
            result["cold_price_point_greeks"] = timed(
                lambda: cls(v2params).calculate_point_greeks(p, e), repeats
            )
            result["curve_diagnostics"] = dict(context.diagnostics)
            result["cold_prepare"] = timed(
                lambda: cls(v2params).prepare(p, e, spots), repeats
            )
            result["warm_curve_101"] = timed(lambda: context.evaluate(spots), repeats)
            result["warm_point"] = timed(lambda: context.evaluate([c.spot]), repeats)
        output[name] = result
        print("benchmark", c.name, name, flush=True)
    return dict(
        name=c.name,
        spots=spots.tolist(),
        v1_qualification_ladder=ladder,
        results=output,
    )


def independent_risk(c):
    """Matched desk bumps through the frozen independent Gaussian reference."""
    params = QuadV2Params()
    bumps = params.get_effective_bump_config()
    s = c.spot
    center = reference(c, "fine", spots=[s * 0.99, s, s * 1.01])["price"]
    ref = dict(
        delta=(center[2] - center[0]) / (s * 0.02),
        gamma=(center[2] - 2 * center[1] + center[0]) / (s * 0.01) ** 2,
    )
    up = reference(replace(c, vol=c.vol + bumps.vol_bump), "fine")["price"][0]
    # The desk facade reports the one-sided PV change for its configured
    # volatility bump. Preserve that convention when building the reference.
    ref["vega"] = up - center[1]
    ref["rho"] = (
        (
            reference(replace(c, rate=c.rate + bumps.rate_bump), "fine")["price"][0]
            - center[1]
        )
        * 0.01
        / bumps.rate_bump
    )
    ref["dividend_rho"] = (
        (
            reference(replace(c, div=c.div + bumps.div_bump), "fine")["price"][0]
            - center[1]
        )
        * 0.01
        / bumps.div_bump
    )
    dt = 1 / 365
    future = [(t - dt, h) for t, h in zip(c.ko_times, c.ko_levels) if t > dt]
    aged = replace(
        c,
        maturity=c.maturity - dt,
        age=c.age + dt,
        ko_times=tuple(t for t, h in future),
        ko_levels=tuple(h for t, h in future),
        ki_times=tuple(t - dt for t in c.ki_times if t > dt),
    )
    ref["theta"] = reference(aged, "fine")["price"][0] - center[1]
    got = GreeksCalculator(params).calculate(
        product(c), environment(c), SnowballQuadEngineV2(params), greeks=list(ref)
    )
    errors = {k: got[k] - ref[k] for k in ref}
    assert max(abs(v) for v in errors.values()) < 1e-6, (c.name, errors)
    return dict(
        name=c.name,
        reference=ref,
        outputs=got,
        errors=errors,
        bump_config=asdict(bumps),
    )


def independent_curve(c):
    """101 PV/delta points and a gamma ladder from independent reference deltas."""
    spots = np.linspace(0.8 * c.spot, 1.2 * c.spot, 101)
    bumps = np.array([1e-4, 5e-5, 2.5e-5])
    queries = np.concatenate(
        [spots] + [s for b in bumps for s in (spots * (1 - b), spots * (1 + b))]
    )
    results = {}
    for level in ("medium", "fine"):
        ref = reference(c, level, spots=queries)
        delta = np.asarray(ref["delta"])
        gamma = []
        for i, b in enumerate(bumps):
            down, up = delta[101 + i * 202 : 303 + i * 202].reshape(2, 101)
            gamma.append((up - down) / (2 * b * spots))
        gamma = np.asarray(gamma)
        extrapolated = (4 * gamma[1:] - gamma[:-1]) / 3
        results[level] = dict(
            price=ref["price"][:101],
            delta=delta[:101].tolist(),
            gamma=extrapolated[-1].tolist(),
            gamma_ladder=gamma.tolist(),
            gamma_extrapolation_difference=(
                extrapolated[-1] - extrapolated[-2]
            ).tolist(),
            settings=ref["settings"],
        )
    context = SnowballQuadEngineV2().prepare(product(c), environment(c), spots)
    got = context.evaluate(spots)
    expected = {k: np.asarray(results["fine"][k]) for k in ("price", "delta", "gamma")}
    checks = gate(got, expected)
    uncertainty = {
        k: float(np.max(np.abs(np.asarray(results["fine"][k]) - results["medium"][k])))
        for k in expected
    }
    uncertainty["gamma_bump"] = float(
        np.max(np.abs(results["fine"]["gamma_extrapolation_difference"]))
    )
    assert checks["all_passed"], (c.name, checks)
    # This is observed refinement evidence, not a certified bound. Reserve
    # most of the acceptance budget for the engine, not reference variation.
    assert uncertainty["price"] < 1e-5 and uncertainty["delta"] < 1e-6, uncertainty
    assert max(uncertainty["gamma"], uncertainty["gamma_bump"]) < 1e-7, uncertainty
    print("independent curve", c.name, checks["max_abs_errors"], flush=True)
    return dict(
        name=c.name,
        spots=spots.tolist(),
        relative_bumps=bumps.tolist(),
        outputs={k: got[k].tolist() for k in expected},
        references=results,
        reference_refinement=uncertainty,
        gate=checks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["matrix", "benchmark", "risk", "curves", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    source = ROOT / "quantark/asset/equity/engine/quad"
    sources = list(source.rglob("*.py")) + [
        ROOT / "quantark/asset/equity/param/quad_v2_params.py",
        ROOT / "quantark/asset/equity/param/engine_params.py",
        Path(__file__),
    ]
    data = dict(
        metadata=dict(
            date=datetime.now().astimezone().isoformat(),
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            thread_env={
                k: os.getenv(k)
                for k in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            args=vars(args),
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(sources)
            },
            reference_sha256=hashlib.sha256(
                (AUDIT / "quad_reference_snapshot.py").read_bytes()
            ).hexdigest(),
            caveat="Matrix uses a separately qualified saved reference; backend/refinement agreement alone is not independent validation. Performance gates are per workload, not a universal guarantee.",
        ),
        results={},
    )
    destination = Path(__file__).with_name(f"{args.phase}_results.json")
    if args.phase in ("matrix", "all"):
        data["results"]["matrix"] = matrix()
        destination.write_text(json.dumps(data, indent=2) + "\n")
    if args.phase in ("risk", "all"):
        data["results"]["risk"] = [
            independent_risk(c)
            for c in cases()
            if c.name in ("monthly", "aged_five_days")
        ]
        destination.write_text(json.dumps(data, indent=2) + "\n")
    if args.phase in ("curves", "all"):
        data["results"]["curves"] = [
            independent_curve(c)
            for c in cases()
            if c.name in ("monthly", "daily", "aged_five_days", "three_years")
        ]
        destination.write_text(json.dumps(data, indent=2) + "\n")
    if args.phase in ("benchmark", "all"):
        data["results"]["benchmark"] = []
        for c in cases():
            if c.name not in ("monthly", "daily", "aged_five_days", "three_years"):
                continue
            data["results"]["benchmark"].append(workload(c, args.repeats))
            destination.write_text(json.dumps(data, indent=2) + "\n")
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    main()
