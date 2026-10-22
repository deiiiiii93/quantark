"""Reproducible edge, Greek, surface and event-decomposition diagnostics."""
import argparse
from dataclasses import replace
import json
import math
import time

import numpy as np
from study import (HERE, Case, ref, reference, reference_deal, panel, quantark,
                   capture, check_contract, product, environment, metadata,
                   SnowballQuadEngine, QuadParams)
from contracts import cases, european_put, mc


def edges():
    c=replace(Case(),vol=.005,div=.40,name="lowvol_drift_domain")
    yield dict(name=c.name,case=c.__dict__,negative_put_control=-european_put(c)[0],
        reference={str(sd):capture(lambda:reference(c,"default",span_sd=sd)) for sd in [10.,20.,40.,80.,100.,120.]},
        quantark={str(n):capture(lambda:quantark(c,n,"transition",(.0001,))) for n in [1001,4001,8001]},
        panel=capture(lambda:panel(c,n_grid=2400,n_gl=240,span=2.5)))
    c=replace(Case(),name="near_ko_first_observation",spot=103.001,
              ko_times=(.0001,.5,1.),ko_levels=(103.,)*3,ki_times=(1.,))
    yield dict(name=c.name,case=c.__dict__,contract_error=check_contract(c),
        panel={str(n):capture(lambda:panel(c,n_grid=n,n_gl=240,span=2.5)) for n in [1200,2400,4800]},
        quantark={f"{n}_{mode}":capture(lambda:quantark(c,n,mode,(.01,.001,.0001)))
                  for n in [1001,2001,4001,8001,16001] for mode in ["legacy_linear","transition"]})
    d=reference_deal(Case())
    d=replace(d,ki_times=(.25,.5,.75,1.),ko_times=(.25+1e-14,.5,.75,1.),
              ko_barriers=(103.,)*4,ko_payoffs=(3.,6.,9.,12.))
    steps=np.diff(np.r_[0.,ref._event_times(d)])
    yield dict(name="near_coincident_dates",min_step=float(steps.min()),
        implied_reference_cells=20/(float(steps.min())**.5/2),note="preflight only; not allocated")
    # Genuine zero-variance controls exercise the MC independently of pricing.
    for spot,expected in [(100.,12.),(110.,1.),(60.,-40.)]:
        c=replace(Case(),name="deterministic_mc",vol=0.,rate=0.,div=0.,spot=spot)
        r=mc(c,power=5,replicates=4)
        np.testing.assert_allclose(r['mean'],expected,rtol=0,atol=1e-12)
        yield dict(name=c.name,spot=spot,exact=expected,result=r)
    c=replace(Case(),name="partial_protection",loss_cap=20.)
    yield dict(name=c.name,case=c.__dict__,
        panel={str(n):capture(lambda:panel(c,n_grid=n,n_gl=240,span=2.5)) for n in [1200,2400,4800]},
        quantark={str(n):capture(lambda:quantark(c,n,"transition",(.0001,))) for n in [1001,4001,8001]})


def greeks(c):
    bumps=[.01,.001,.0001,.00001]
    s=np.array([c.spot]+[c.spot*(1+k*b) for b in bumps for k in [-1,1]])
    r=reference(c,"fine",spots=s)
    x=np.array(r['price']);delta=np.array(r['delta'])
    matched={}
    for i,b in enumerate(bumps):
        down,up=x[1+2*i:3+2*i]
        matched[str(b)]=dict(down=down,up=up,delta=(up-down)/(2*c.spot*b),
            gamma=(up-2*x[0]+down)/(c.spot*b)**2,
            gamma_from_delta=(delta[2+2*i]-delta[1+2*i])/(2*c.spot*b))
    p,e=product(c),environment(c)
    # Probe one common narrow spot interval around KI or the current mark.
    spots=c.spot*np.linspace(.995,1.005,21)
    rr=reference(c,"fine",spots=spots)
    curves={}
    for mode in ["legacy_linear","transition"]:
        eng=SnowballQuadEngine(QuadParams(grid_points=1001,readout=mode))
        curve=eng.calculate_spot_greeks_curve(p,e,spots)
        direct=[quantark(c,1001,mode,(.0001,),**{}) if float(s)==c.spot else
                quantark(replace(c,spot=float(s)),1001,mode,(.0001,)) for s in spots]
        curves[mode]=dict(curve=curve,direct=direct,
            curve_price_error_bp=float(100*np.max(np.abs(np.array([v['price'] for v in curve])-rr['price']))),
            curve_delta_error=float(np.max(np.abs(np.array([v['delta'] for v in curve])-rr['delta']))),
            direct_price_error_bp=float(100*np.max(np.abs(np.array([v['price'] for v in direct])-rr['price']))),
            direct_delta_error=float(np.max(np.abs(np.array([v['bumps']['0.0001']['delta'] for v in direct])-rr['delta']))),
            curve_vs_direct_price_bp=float(100*np.max(np.abs(np.array([v['price'] for v in direct])-np.array([v['price'] for v in curve])))))
    return dict(name=c.name,case=c.__dict__,reference=r,matched_bumps=matched,
                curve_spots=spots.tolist(),curve_reference=rr,curves=curves)


def events(c):
    out={}
    for mode in ["legacy_linear","transition"]:
        eng=SnowballQuadEngine(QuadParams(grid_points=2001,readout=mode))
        p,e=product(c),environment(c)
        pv=eng.price(p,e)
        s=eng.calculate_event_stats(p,e)
        out[mode]=dict(price=pv,event_pv=s.pv,ko_pv=float(np.sum(s.expected_discounted_ko_cashflow)),
            maturity_pv=s.expected_discounted_maturity_cashflow,reconciliation=s.reconciliation_error,
            ko_probability_sum=float(np.sum(s.ko_probability)),
            min_ko_probability=float(np.min(s.ko_probability)),
            cashflow_sum=float(np.sum(s.expected_discounted_cashflows)))
    return dict(name=c.name,results=out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['edges','greeks','events'],required=True)
    p.add_argument('--only',default='');a=p.parse_args()
    target=HERE/'results'/(a.phase+('_'+a.only if a.only else '')+'.json')
    data=dict(metadata=metadata(a),results=[])
    chosen=[c for c in cases() if not a.only or c.name in a.only.split(',')]
    it=edges() if a.phase=='edges' else (capture(lambda c=c:(greeks(c) if a.phase=='greeks' else events(c))) for c in chosen)
    for row in it:
        data['results'].append(row);target.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        print(row.get('name',row.get('error')),flush=True)
    print('Saved',target)


if __name__=='__main__':main()
