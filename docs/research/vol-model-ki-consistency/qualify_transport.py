"""Payoff-linearity repair and variance-grid refinement for characteristic ADI."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'test')]
from pde_statistics.test_vol_model_ki_consistency import HP,note,_flat_env
from test_asset.equity.engine_quad.test_autocallable_probability_consistency import unit_overlay
from quantark.asset.equity.engine.pde import HestonSnowballPDESolver
from quantark.asset.equity.engine.pde import adi_columns
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.util.numerical.tridiag import tridiag_backend

rows=[]
original=adi_columns._linear_variance_transport
try:
    for method in ('clipped_cubic_control','linear_characteristic'):
        adi_columns._linear_variance_transport=(lambda core,U,h:HestonSLVADICore._advect_v(core,U,h)) if method=='clipped_cubic_control' else original
        for spot in (80.,90.,100.):
            env,p=_flat_env(s0=spot),note('snowball')
            e=HestonSnowballPDESolver(HP,n_x=80,n_v=24,n_t=48,v_drift_scheme='semi_lagrangian')
            base=e.price(p,env)
            s=e.calculate_event_stats(p,env)
            inc=e.price_folded(p,env,unit_overlay(p,env,'ki'))-base
            rows.append(dict(method=method,spot=spot,pv=base,ki_settlement=s.ki_survive_knocked_in_probability,folded_increment=inc,difference=inc-s.ki_survive_knocked_in_probability))
finally:
    adi_columns._linear_variance_transport=original

refinement=[]
for scheme,n_v in [('semi_lagrangian',48),('semi_lagrangian',96),('semi_lagrangian',192),('adaptive_upwind',384)]:
    e=HestonSnowballPDESolver(HP,n_x=160,n_v=n_v,n_t=192,v_drift_scheme=scheme)
    s=e.calculate_event_stats(note('snowball'),_flat_env(s0=90.))
    refinement.append(dict(scheme=scheme,n_x=160,n_v=n_v,n_t=192,pv=s.pv,ki_settlement=s.ki_survive_knocked_in_probability,ki_ever=s.ki_ever_probability))
    print(refinement[-1],flush=True)
ref=refinement[-1]
checks=[]
for r in refinement[1:3]:
    for field in ('pv','ki_settlement','ki_ever'):
        allowance=.01 if field=='pv' else .0005
        checks.append(dict(n_v=r['n_v'],field=field,difference=abs(r[field]-ref[field]),allowance=allowance,passed=abs(r[field]-ref[field])<=allowance))
result=dict(backend=tridiag_backend(),linearity=rows,refinement=refinement,checks=checks,qualified_n_v=192,passed=all(c['passed'] for c in checks if c['n_v']==192))
Path(__file__).with_name('transport.json').write_text(json.dumps(result,indent=2)+'\n')
assert max(abs(r['difference']) for r in rows if r['method']=='linear_characteristic')<2e-10
if not result['passed']:raise SystemExit(1)
