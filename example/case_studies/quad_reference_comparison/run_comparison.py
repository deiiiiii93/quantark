"""Reproduce the reference/QuantArk autocallable comparison.

Run from the repo root with .venv/bin/python. Reference source is read from
docs/quad/ref_scripts without editing it. AST loading removes only obsolete
imports and the unused Numba decorator; scipy FFT and all pricing function
bodies are unchanged. The adjusted Snowball comparison calls the original
core with undiscounted, correctly partitioned contractual cashflows.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from independent_mc import (Case, TERM_TIMES, TERM_RATES, TERM_DIVS, TERM_VOLS,
                            coupon_only_exact, estimate, vanilla_put)
from quantark.asset.equity.engine.quad import SnowballQuadEngine, PhoenixQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType, ProtectionType


class ObsFreqType(Enum):
    DAILY = "daily"
    EUROPEAN = "euro"
    CUSTOM = "custom"


def load_reference():
    ns = {"ObsFreqType": ObsFreqType, "__name__": "audit_reference"}
    for filename in ["option_quad.py", "snowball_quad.py", "phoenix_quad_v2.py"]:
        path = ROOT / "docs/quad/ref_scripts" / filename
        tree = ast.parse(path.read_text(), str(path))
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("asset.", "util.", "numba")):
                continue
            if isinstance(node, ast.Import) and any(x.name == "pyfftw" for x in node.names):
                continue
            if isinstance(node, ast.If):  # script demo guard only
                continue
            if isinstance(node, ast.FunctionDef) and node.name == "calculate_integral_simpson_numba":
                node.decorator_list = []
            body.append(node)
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


REF = load_reference()


def environment(c):
    if c.term:
        r = LinearRateCurve(list(zip(TERM_TIMES, TERM_RATES)))
        q = TermStructureDividendYield(TERM_TIMES.tolist(), TERM_DIVS.tolist())
        v = TermStructureVolSurface(TERM_TIMES.tolist(), TERM_VOLS.tolist())
    else:
        r, q, v = FlatRateCurve(c.rate), ContinuousDividendYield(c.div), FlatVolSurface(c.vol)
    return PricingEnvironment(spot_quote=SpotQuote(c.spot), rate_curve=r,
                              div_yield=q, vol_surface=v, valuation_date=datetime(2026, 9, 11))


def product(c):
    barrier = BarrierConfig(ko_barrier=list(c.ko_levels), ko_rate=c.coupon if c.product == "snowball" else 0,
        ko_observation_dates=list(c.ko_times), ki_barrier=c.ki_level,
        ki_observation_type=ObservationType.CONTINUOUS if c.continuous_ki else ObservationType.DISCRETE,
        ki_observation_dates=None if c.continuous_ki else list(c.ki_times), ki_continuous=c.continuous_ki)
    payoff = PayoffConfig(rebate_rate=c.coupon if c.product == "snowball" else 0,
        include_principal=False, protection_type=ProtectionType.NONE if c.loss_cap is None else ProtectionType.PARTIAL,
        protection_rate=0 if c.loss_cap is None else c.loss_cap / 100)
    kwargs = dict(initial_price=100., strike=c.strike, barrier_config=barrier, payoff_config=payoff,
                  accrual_config=AccrualConfig(is_annualized_ko=True, is_annualized_rebate=True,
                                               is_annualized_ki=False),
                  contract_multiplier=1., maturity=c.maturity, tenor=c.maturity, is_reverse=c.reverse)
    if c.product == "snowball":
        p = SnowballOption(**kwargs)
    else:
        p = PhoenixOption(**kwargs, coupon_config=CouponBarrierConfig(coupon_barrier=c.coupon_level,
            coupon_rate=c.coupon, coupon_pay_type=CouponPayType.INSTANT, memory_coupon=c.memory))
    if c.already_ki:
        p._otc_lifecycle_knocked_in = True
    return p


def qa(c, n=1001, **options):
    p, env = product(c), environment(c)
    cls = SnowballQuadEngine if c.product == "snowball" else PhoenixQuadEngine
    engine = cls(QuadParams(grid_points=n, **options))
    value = engine.price(p, env)
    actual = len(engine._last_spot_greeks_grid[0]) if hasattr(engine, "_last_spot_greeks_grid") else None
    return {"price": value, "actual_grid": actual, "convergence": getattr(engine, "_last_convergence_info", None)}


def reference_raw(c, n=1000, nt=None, market_spot=100.):
    if c.term or c.memory or c.continuous_ki or c.reverse:
        raise NotImplementedError("No matching reference wrapper for this contract/model")
    kwargs = dict(bus_days=int(round(c.maturity * 252)), cal_days=c.maturity * 252,
        ko_bus_days=np.array(c.ko_times) * 252, ko_cal_days=np.array(c.ko_times) * 252,
        ko_prices=c.ko_levels, ki_price=c.ki_level, spot=c.spot, r=c.rate, q=c.div, vol=c.vol,
        notional=100., initial=100., strike=c.strike, grid_x=n, grid_t=nt,
        is_knocked_in=c.already_ki, protection_rate=0 if c.loss_cap is None else 1 - c.loss_cap / 100,
        day_count_basis={"calendar": 252, "trading": 252}, market_spot=market_spot)
    if c.product == "snowball":
        return REF["price_snowball"](**kwargs, ko_rate=c.coupon, coupon_rate=c.coupon,
            ki_obs_type=ObsFreqType.CUSTOM, ki_bus_days=np.array(c.ki_times) * 252)
    return REF["price_phoenix_v2"](**kwargs, div_prices=[c.coupon_level] * len(c.ko_times),
        div_rate=c.coupon, ki_obs_type="custom", ki_dates=np.array(c.ki_times) * 252,
        coupon_after_ki=True, pay_type="instant")


def aligned_steps(c):
    for n in range(3, max(3000, int(c.maturity * 252) + 1)):
        x = np.array(c.ko_times + c.ki_times) / c.maturity * n
        if np.all(np.abs(x - np.round(x)) < 1e-9):
            return n
    raise ValueError("No modest uniform grid exactly contains the event dates")


def reference_core(c, n=1001, nt=None):
    """Original core; corrected cashflow adapter, not the supplied wrapper.

    Event dates must fit the uniform grid. Converting mathematically identical
    dates to that grid's stored float prevents one-ULP searchsorted delays.
    """
    if c.product != "snowball" or c.term or c.continuous_ki or c.reverse:
        raise NotImplementedError("Adjusted core adapter covers discrete flat-GBM Snowball only")
    nt = aligned_steps(c) if nt is None else nt
    grid = np.linspace(0, c.maturity, nt + 1)
    def indices(times):
        scaled = np.array(times) / c.maturity * nt
        if np.max(np.abs(scaled - np.round(scaled)), initial=0) > 1e-8:
            raise ValueError("Uniform time grid does not contain the contractual dates")
        return np.round(scaled).astype(int)
    ko_idx, ki_idx = indices(c.ko_times), indices(c.ki_times)

    def leg(kind, strike=None):
        upper = np.full(nt + 1, np.inf)
        lower = np.zeros(nt + 1)
        upper[ko_idx] = c.ko_levels
        if kind in ("dko_put", "rebate"):
            lower[ki_idx] = c.ki_level
        factors = {name: np.zeros(nt + 1) for name in ("asset1", "asset2", "asset3", "cash1", "cash2", "cash3")}
        if kind == "ko":
            factors["cash3"][ko_idx] = 100 * c.coupon * np.array(c.ko_times)
        elif kind in ("uo_put", "dko_put"):
            upper[-1] = min(upper[-1], strike)
            if lower[-1] >= upper[-1]:
                return 0.0
            factors["asset2"][-1] = -1.
            factors["cash2"][-1] = strike
        else:
            factors["cash2"][-1] = 100 * c.coupon * c.maturity
        pricer = REF["QuadratureOptionPricer"](n, nt, c.maturity, c.spot, c.rate, c.div, c.vol)
        return pricer.price(dict(ko_prices_u=upper[1:], ko_prices_l=lower[1:],
            ko_dates_u=grid[1:], ko_dates_l=grid[1:], factors=factors))

    ko = leg("ko")
    rebate = 0 if c.already_ki else leg("rebate")
    def downside(k):
        return -leg("uo_put", k) + (0 if c.already_ki else leg("dko_put", k))
    loss = downside(c.strike)
    if c.loss_cap is not None:
        loss -= downside(max(0., c.strike - c.loss_cap))
    return {"price": ko + rebate + loss, "ko": ko, "rebate": rebate, "loss": loss, "time_steps": nt}


def captured(fn):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn()
            if isinstance(result, (int, float, np.floating)):
                result = {"price": float(result)}
        except Exception as e:
            result = {"error": type(e).__name__ + ": " + str(e)}
    result["elapsed_s"] = time.perf_counter() - start
    if caught:
        result["warnings"] = list(dict.fromkeys(str(x.message) for x in caught))[:10]
    return result


def cases():
    base = Case("snow_monthly")
    daily = [i / 252 for i in range(1, 253)]
    out = [base, replace(base, name="snow_daily", ki_times=daily)]
    for name, kw in [
        ("low_vol", dict(vol=.05)), ("high_vol", dict(vol=.60)),
        ("low_spot", dict(spot=82.)), ("near_ki", dict(spot=75.1)), ("high_spot", dict(spot=110.)),
        ("negative_rate", dict(rate=-.02)), ("high_rate", dict(rate=.10)),
        ("high_dividend", dict(div=.15)), ("already_ki", dict(already_ki=True)),
        ("partial_protection", dict(loss_cap=20.)),
        ("step_down", dict(ko_levels=np.linspace(110, 90, 12).tolist())),
        ("irregular", dict(ko_times=(np.array([17,43,64,85,110,132,150,175,196,217,238,252]) / 252).tolist(), ki_times=daily)),
        ("continuous", dict(continuous_ki=True)), ("term_curve", dict(term=True)),
    ]:
        out.append(replace(base, name="snow_" + name, **kw))
    for months in [3, 36]:
        ts = [i / 12 for i in range(1, months + 1)]
        out.append(replace(base, name=f"snow_{months}m", maturity=months / 12,
                           ko_times=ts, ko_levels=[103.] * months, ki_times=ts))
    phoenix = replace(base, name="phoenix_monthly", product="phoenix")
    out.extend([phoenix, replace(phoenix, name="phoenix_daily", ki_times=daily),
        replace(phoenix, name="phoenix_coupon_near_ko", coupon_level=102.9),
        replace(phoenix, name="phoenix_coupon_above_ko", coupon_level=110.),
        replace(phoenix, name="phoenix_memory", memory=True),
        replace(phoenix, name="phoenix_term_curve", term=True)])
    out.extend([replace(base, name="snow_reverse", reverse=True, ko_levels=[97.] * 12, ki_level=125.),
                replace(phoenix, name="phoenix_reverse", reverse=True, ko_levels=[97.] * 12,
                        ki_level=125., coupon_level=115.)])
    return out


def controls():
    out = []
    for r in [0., .03, .10, -.02]:
        c = Case(f"vanilla_put_r{r}", rate=r, coupon=0., already_ki=True,
                 ko_times=[1.], ko_levels=[1e8], ki_times=[1.])
        out.append({"case": asdict(c), "exact": -vanilla_put(c),
            "reference": captured(lambda: reference_raw(c, 4001, 48)),
            "adjusted_core": captured(lambda: reference_core(c, 4001, 48)),
            "quantark": captured(lambda: qa(c, 4001))})
    for times in [[1.], [.25,.5,.75,1.], [i / 12 for i in range(1,13)]]:
        c = Case(f"coupon_only_{len(times)}", product="phoenix", ko_times=times,
                 ko_levels=[1e8] * len(times), ki_level=.0001, ki_times=[1.], coupon_level=80.)
        out.append({"case": asdict(c), "exact": coupon_only_exact(c),
                    "reference": captured(lambda: reference_raw(c, 4001, 48)),
                    "quantark": captured(lambda: qa(c, 4001))})
    for row in out:
        row["mc"] = captured(lambda row=row: estimate(Case(**row["case"]), power=17, replicates=8))
    for name, kwargs, exact in [
        ("deterministic_no_event", dict(spot=100.), 12.),
        ("deterministic_ko", dict(spot=110.), 1.),
        ("deterministic_ki", dict(spot=60.), -40.),
        ("deterministic_protected", dict(spot=60., loss_cap=20.), -20.),
        ("deterministic_phoenix", dict(product="phoenix", spot=100.), 12.),
    ]:
        c = Case(name, rate=0., div=0., vol=0., **kwargs)
        value = estimate(c, power=5, replicates=4)
        assert abs(value["mean"] - exact) < 1e-10, (name, value, exact)
        out.append({"case": asdict(c), "exact": exact, "mc": value})
    return out


def metadata(args):
    paths = [ROOT / "docs/quad/ref_scripts" / x for x in ("option_quad.py", "snowball_quad.py", "phoenix_quad_v2.py")]
    paths += list((ROOT / "quantark/asset/equity/engine/quad").glob("*.py"))
    paths += list(Path(__file__).parent.glob("*.py"))
    return {"date": datetime.now().isoformat(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
        "args": vars(args), "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["controls", "matrix", "ladders", "benchmark"], default="controls")
    parser.add_argument("--power", type=int, default=17)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--only", default="")
    parser.add_argument("--out", default=str(Path(__file__).with_name("results")))
    args = parser.parse_args()
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    data = {"metadata": metadata(args), "results": []}
    target = path / (args.phase + ("_" + args.only if args.only else "") + ".json")
    def save(row):
        data["results"].append(row)
        target.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n")
        print(json.dumps(row), flush=True)
    selected = [c for c in cases() if not args.only or c.name in args.only.split(",")]
    if args.phase == "controls":
        for row in controls():
            save(row)
    elif args.phase == "matrix":
        for c in selected:
            row = {"case": asdict(c)}
            row["mc"] = captured(lambda: estimate(c, args.power, args.replicates))
            row["reference_default"] = captured(lambda: reference_raw(c))
            row["reference_4001"] = captured(lambda: reference_raw(c, 4001))
            row["core_1001"] = captured(lambda: reference_core(c, 1001))
            row["core_4001"] = captured(lambda: reference_core(c, 4001))
            row["quantark_default"] = captured(lambda: qa(c))
            row["quantark_4001"] = captured(lambda: qa(c, 4001))
            save(row)
    elif args.phase == "ladders":
        for c in selected:
            for n in [201, 501, 1001, 2001, 4001, 8001]:
                save({"case": c.name, "grid": n,
                      "raw": captured(lambda: reference_raw(c, n)),
                      "core": captured(lambda: reference_core(c, n)),
                      "quantark": captured(lambda: qa(c, n))})
    elif args.phase == "benchmark":
        for c in selected:
            for n in [501, 1001, 4001]:
                for name, fn in [("raw", lambda: reference_raw(c, n)), ("core", lambda: reference_core(c, n)), ("quantark", lambda: qa(c, n))]:
                    warm = captured(fn)
                    if "error" in warm:
                        save({"case": c.name, "grid": n, "engine": name, **warm})
                        continue
                    samples = []
                    for _ in range(7):
                        t0 = time.perf_counter()
                        fn()
                        samples.append(time.perf_counter() - t0)
                    save({"case": c.name, "grid": n, "engine": name, "samples_s": samples,
                          "median_s": statistics.median(samples), "min_s": min(samples),
                          "max_s": max(samples), "actual_grid": warm.get("actual_grid"),
                          "time_steps": warm.get("time_steps")})
    print("Saved", target, flush=True)


if __name__ == "__main__":
    main()
