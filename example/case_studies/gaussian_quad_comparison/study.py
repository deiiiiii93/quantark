"""Audit the exact supplied Gaussian reference against SnowballQuadEngine.

No production engine changes. The reference snapshot is byte-for-byte copied.
Use --phase controls|matrix|mc|diagnostics|benchmark; each writes a separate
JSON artifact with source hashes, environment, timings and numerical inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import quad_reference_snapshot as ref
from contracts import Case, cases, european_put, mc
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType


def environment(c, spot=None):
    return PricingEnvironment(spot_quote=SpotQuote(c.spot if spot is None else float(spot)),
        rate_curve=FlatRateCurve(c.rate),div_yield=ContinuousDividendYield(c.div),
        vol_surface=FlatVolSurface(c.vol),valuation_date=datetime(2026,9,11))


def product(c):
    p = SnowballOption(initial_price=100.,strike=100.,contract_multiplier=1.,
        maturity=c.maturity,tenor=c.age+c.maturity,
        barrier_config=BarrierConfig(ko_barrier=list(c.ko_levels),ko_rate=c.coupon,
            ko_observation_dates=list(c.ko_times),ki_barrier=c.ki,
            ki_observation_dates=list(c.ki_times),ki_observation_type=ObservationType.DISCRETE,
            ki_continuous=False,disable_ko_after_ki=c.disable_ko_after_ki),
        payoff_config=PayoffConfig(rebate_rate=c.coupon,include_principal=c.principal,
            protection_type=ProtectionType.NONE if c.loss_cap is None else ProtectionType.PARTIAL,
            protection_rate=0. if c.loss_cap is None else c.loss_cap/100.),
        accrual_config=AccrualConfig(is_annualized_ko=True,is_annualized_rebate=True,is_annualized_ki=False,
                                    accrual_factors=[c.age+t for t in c.ko_times]))
    if c.already_ki:
        p._otc_lifecycle_knocked_in = True
    return p


def reference_deal(c):
    # Reference public API only marks alive paths. An already-KI contract
    # is algebraically reduced to A=B, with KO removed when disabled after KI.
    ko_times, ko_levels = c.ko_times,c.ko_levels
    if c.already_ki and c.disable_ko_after_ki:
        ko_times,ko_levels = (),()
    return ref.Deal(maturity=c.maturity,
        payoff_alive=c.ki_payoff if c.already_ki else c.alive_payoff,payoff_ki=c.ki_payoff,
        ki_barrier=None if c.already_ki else c.ki,ki_times=c.ki_times,
        ko_times=ko_times,ko_barriers=ko_levels,ko_payoffs=tuple(c.ko_payoff(t) for t in ko_times),
        ko_survives_ki=not c.disable_ko_after_ki,
        terminal_breaks=(100.,) if c.loss_cap is None else (100.-c.loss_cap,100.))


def check_contract(c):
    p,e=product(c),environment(c)
    s=np.array([1.,50.,74.999,75.,89.99,90.,99.99,100.,100.01,103.,150.])
    errors=[]
    for got, expected in [(np.array([p.get_maturity_payoff_v0(float(x),e) for x in s]),c.alive_payoff(s)),
                          (np.array([p.get_maturity_payoff_v1(float(x),e) for x in s]),c.ki_payoff(s))]:
        errors.append(float(np.max(np.abs(got-expected))))
    ko=p.resolve_ko_observations(e)
    errors.append(float(np.max(np.abs([x.payoff-c.ko_payoff(t) for x,t in zip(ko,c.ko_times)]))))
    assert len(ko)==len(c.ko_times)
    np.testing.assert_allclose([x.observation_time for x in ko],c.ko_times,rtol=0,atol=1e-12)
    np.testing.assert_allclose([x.barrier for x in ko],c.ko_levels,rtol=0,atol=1e-12)
    ki=p.resolve_ki_observations(e)
    np.testing.assert_allclose([x.observation_time for x in ki],c.ki_times,rtol=0,atol=1e-12)
    assert max(errors)<1e-10, (c.name,errors)
    return max(errors)


def reference(c, level="default", spots=None, **options):
    settings={"default":dict(cells_per_sd=2.,n_q=8,span_sd=10.,n_gl=240,n_sd=9.),
              "medium":dict(cells_per_sd=3.,n_q=10,span_sd=10.,n_gl=360,n_sd=9.),
              "fine":dict(cells_per_sd=4.,n_q=12,span_sd=12.,n_gl=480,n_sd=10.)}[level]
    settings.update(options)
    s=np.asarray([c.spot] if spots is None else spots,dtype=float)
    d=reference_deal(c)
    # Record actual reference mesh and refuse pathological allocation in this
    # audit harness; no changes to the supplied implementation are made.
    ts=ref._event_times(d)
    steps=np.diff(np.r_[0.,ts])
    h=c.vol*math.sqrt(float(steps.min()))/settings["cells_per_sd"]
    reach=max(settings["span_sd"],settings["n_sd"]+.5)*c.vol*math.sqrt(c.maturity)
    if h<=0 or (np.ptp(np.log(s))+2*reach)/h>100_000:
        raise ValueError("audit allocation guard: zero variance or near-coincident event times")
    mesh=ref._build_mesh(float(np.log(s).min())-reach,float(np.log(s).max())+reach,h,
                         settings["n_q"],align=math.log(d.terminal_breaks[0]))
    pv,delta=ref.value_and_delta_operator(d,ref.Market(c.rate,c.div,c.vol),s,**settings)
    return dict(price=pv.tolist(),delta=delta.tolist(),cells=mesh.n_cells,
                nodes=mesh.n_cells*mesh.n_q,h=mesh.h,settings=settings)


def panel(c, **settings):
    pv,delta=ref.value_and_delta(reference_deal(c),ref.Market(c.rate,c.div,c.vol),c.spot,**settings)
    return dict(price=pv,delta=delta,settings=settings)


def quantark(c,n=1001,mode="legacy_linear",bumps=(),**options):
    p=product(c)
    eng=SnowballQuadEngine(QuadParams(grid_points=n,readout=mode,**options))
    pv=float(eng.price(p,environment(c)))
    grid=getattr(eng,"_last_spot_greeks_grid",None)
    result=dict(price=pv,actual_grid=len(grid[0]) if grid is not None else None,
                mode=mode,requested_grid=n,bumps={},
                convergence=getattr(eng,"_last_convergence_info",None))
    for b in bumps:
        down=float(eng.price(p,environment(c,c.spot*(1-b))))
        up=float(eng.price(p,environment(c,c.spot*(1+b))))
        result["bumps"][str(b)]=dict(down=down,up=up,delta=(up-down)/(2*c.spot*b),
                                      gamma=(up-2*pv+down)/(c.spot*b)**2)
    return result


def capture(fn):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            result=fn()
        except Exception as e:
            result=dict(error=type(e).__name__+": "+str(e))
    result["elapsed_s"]=time.perf_counter()-start
    if w:
        result["warnings"]=list(dict.fromkeys(str(x.message) for x in w))[:8]
    return result


def controls():
    b=Case()
    for rate in [-.02,0.,.03,.10]:
        c=replace(b,name=f"vanilla_put_{rate}",rate=rate,coupon=0.,already_ki=True,
                  ko_times=(1.,),ko_levels=(1e8,),ki_times=(1.,))
        p,d=european_put(c)
        yield dict(case=asdict(c),contract_error=check_contract(c),exact_price=-p,exact_delta=-d,
                   reference=capture(lambda:reference(c)),panel=capture(lambda:panel(c,n_grid=400,n_gl=240)),
                   quantark=capture(lambda:quantark(c,4001,"transition",(.001,.0001))))
    # Single terminal observation has an independent closed form with two
    # truncated lognormal moments; it exercises the KI payoff discontinuity.
    for spot in [74.99,90.,100.,103.01]:
        c=replace(b,name=f"one_event_{spot}",spot=spot,ko_times=(1.,),ko_levels=(103.,),ki_times=(1.,))
        def exact(s):
            sd=c.vol*math.sqrt(c.maturity)
            def d2(h):return (math.log(s/h)+(c.rate-c.div-c.vol*c.vol/2)*c.maturity)/sd
            df=math.exp(-c.rate*c.maturity)
            return df*(c.ko_payoff(1.)*norm.cdf(d2(103.)) + c.alive_payoff(100.)*(norm.cdf(d2(75.))-norm.cdf(d2(103.))) - 100*norm.cdf(-d2(75.))) + s*math.exp(-c.div*c.maturity)*norm.cdf(-d2(75.)-sd)
        eps=c.spot*1e-5
        yield dict(case=asdict(c),contract_error=check_contract(c),exact_price=exact(c.spot),
                   exact_delta=(exact(c.spot+eps)-exact(c.spot-eps))/(2*eps),
                   reference=capture(lambda:reference(c)),panel=capture(lambda:panel(c,n_grid=400,n_gl=240)),
                   quantark=capture(lambda:quantark(c,4001,"transition",(.001,.0001))))
    # Semigroup: inert observation steps must preserve a European payoff.
    c=replace(b,name="no_event_semigroup",coupon=0.,already_ki=True,ko_levels=(1e8,)*12)
    p,d=european_put(c)
    yield dict(case=asdict(c),exact_price=-p,exact_delta=-d,
               reference=capture(lambda:reference(c)),panel=capture(lambda:panel(c,n_grid=1200,n_gl=100,span=2.5)),
               quantark=capture(lambda:quantark(c,4001,"transition",(.0001,))))
    for c in [b,replace(b,name="lowvol_drift_domain",vol=.005,div=.40),
              replace(b,name="zero_vol",vol=0.),replace(b,name="partial_protection",loss_cap=20.)]:
        yield dict(case=asdict(c),contract_error=check_contract(c) if c.vol>0 else None,
                   reference=capture(lambda:reference(c)),
                   quantark=capture(lambda:quantark(c)),
                   **({"panel":capture(lambda:panel(c,n_grid=1200,n_gl=100,span=2.5))} if c.vol>0 else {}))


def metadata(args):
    paths=list(HERE.glob("*.py"))+list((ROOT/"quantark/asset/equity/engine/quad").glob("*.py"))
    paths += [ROOT/"quantark/asset/equity/engine/base_engine.py",ROOT/"quantark/asset/equity/param/engine_params.py"]
    return dict(date=datetime.now().isoformat(),git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
                thread_env={k:os.getenv(k) for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]},
                args=vars(args),sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["controls","matrix","mc","diagnostics","benchmark"],required=True)
    parser.add_argument("--only",default="")
    parser.add_argument("--out",default=str(HERE/"results"))
    parser.add_argument("--power",type=int,default=18)
    parser.add_argument("--replicates",type=int,default=8)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    target=out/(args.phase+("_"+args.only if args.only else "")+".json")
    data=dict(metadata=metadata(args),results=[])
    def save(row):
        data["results"].append(row)
        target.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
        print(json.dumps(row),flush=True)
    selected=[c for c in cases() if not args.only or c.name in args.only.split(",")]
    if args.phase=="controls":
        for row in controls():save(row)
    elif args.phase=="matrix":
        for c in selected:
            row=dict(case=asdict(c),contract_error=check_contract(c))
            row["reference"]={level:capture(lambda level=level:reference(c,level)) for level in ["default","medium","fine"]}
            row["quantark"]={}
            for n in [1001,2001,4001,8001]:
                for mode in ["legacy_linear","transition"]:
                    row["quantark"][f"{n}_{mode}"]=capture(lambda:quantark(c,n,mode,(.01,.001,.0001)))
            save(row)
    elif args.phase=="mc":
        for c in selected:save(dict(case=asdict(c),mc=capture(lambda:mc(c,args.power,args.replicates))))
    elif args.phase=="diagnostics":
        for c in selected:
            row=dict(case=asdict(c))
            row["reference_domain"]={str(sd):capture(lambda:reference(c,"medium",span_sd=sd)) for sd in [10.,12.,16.]}
            row["reference_final_order"]={str(n):capture(lambda:reference(c,"medium",n_gl=n)) for n in [120,240,480,960]}
            row["quantark_domain"]={str(sd):capture(lambda:quantark(c,4001,"transition",(.0001,),num_std_devs=sd)) for sd in [8.,10.,14.]}
            row["auto_converge"]=capture(lambda:quantark(c,1001,"transition",auto_converge=True))
            row["no_filter"]=capture(lambda:quantark(c,1001,"transition",(.0001,),fft_filter_alpha=0.))
            row["nodal"]=capture(lambda:quantark(c,1001,"legacy_linear",(.0001,),event_projection="nodal",integration_rule="simpson"))
            save(row)
    elif args.phase=="benchmark":
        for c in selected:
            p,e=product(c),environment(c)
            for n in [1001,4001,8001]:
                for mode in ["legacy_linear","transition"]:
                    eng=SnowballQuadEngine(QuadParams(grid_points=n,readout=mode))
                    for workload,fn in [("price",lambda:eng.price(p,e)),("price_delta_gamma",lambda:eng.calculate_greeks(p,e))]:
                        fn()
                        samples=[]
                        for _ in range(7):
                            start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
                        save(dict(case=c.name,engine="quantark",n=n,mode=mode,workload=workload,
                                  median_s=statistics.median(samples),min_s=min(samples),max_s=max(samples),samples_s=samples))
            for level in ["default","fine"]:
                reference(c,level)
                samples=[]
                for _ in range(7):
                    start=time.perf_counter();reference(c,level);samples.append(time.perf_counter()-start)
                save(dict(case=c.name,engine="reference",level=level,workload="price_analytic_delta",
                          median_s=statistics.median(samples),min_s=min(samples),max_s=max(samples),samples_s=samples))
            spots=np.linspace(c.spot*.9,c.spot*1.1,101)
            eng=SnowballQuadEngine(QuadParams(grid_points=4001,readout="transition"))
            for label,fn in [("reference_spot_curve",lambda:reference(c,"default",spots=spots)),
                             ("quantark_grid_curve",lambda:eng.calculate_spot_greeks_curve(p,e,spots))]:
                fn();samples=[]
                for _ in range(5):
                    start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
                save(dict(case=c.name,engine=label,workload="101_spot_curve",median_s=statistics.median(samples),samples_s=samples))
    print("Saved",target,flush=True)


if __name__=="__main__":main()
