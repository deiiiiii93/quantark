import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'test')]
from test_asset.equity.engine_quad.test_autocallable_probability_consistency import CASES,env
from test_asset.equity.engine_quad.test_continuous_ki_capability_consistency import term_market
from quantark.asset.equity.param import QuadParams,PDEParams,MCParams,QuadV2Params
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
rows=[]
for case in ['flat_overlap_disabled','piecewise_terms','smooth_terms']:
 for family,(factory,spec,quad,v2) in CASES.items():
  spec=dict(spec,ki_monitoring='continuous')
  if family=='ko_reset':spec['ki_continuous']=True
  e=env() if case=='flat_overlap_disabled' else term_market()
  if case=='flat_overlap_disabled':
   spec['disable_ko_after_ki']=True
   if family=='ko_reset':spec.update(pre_ko_barrier=80.,ki_barrier=85.)
   else:spec.update(ko_barrier=80.,ki_barrier=85.)
  if case=='smooth_terms':
   from quantark.param.div import TermStructureDividendYield
   e.div_yield=TermStructureDividendYield(times=[.12,.29,.5,1.],yields=[.01,.05,.02,.02])
  p=factory(spec);pc=KOResetSnowballPDESolver if family=='ko_reset' else PhoenixPDESolver if family.startswith('phoenix') else SnowballPDESolver
  mc=PhoenixMCEngine if family.startswith('phoenix') else SnowballMCEngine
  opts=dict(continuous_term_structure='piecewise_constant',continuous_steps_per_year=96)
  engines=[quad(QuadParams(grid_points=1001,**opts)),v2(QuadV2Params(**opts)),v2(QuadV2Params(order=10,cells_per_sd=3.,**opts)),pc(PDEParams(accuracy='high',**opts)),mc(MCParams(num_paths=100000,seed=917,**opts))]
  for i,eng in enumerate(engines):
   t=time.perf_counter()
   try:
    s=eng.calculate_event_stats(p,e)
    row=dict(case=case,family=family,engine=type(eng).__name__,reference=(i==2),pv=s.pv,ki_ever=s.ki_ever_probability,ki_settlement=s.ki_survive_knocked_in_probability,ko=float(s.pre_ko_probability_total+s.post_ko_probability_total) if family=='ko_reset' else float(s.ko_probability.sum()),seconds=time.perf_counter()-t)
   except Exception as ex:row=dict(case=case,family=family,engine=type(eng).__name__,error=str(ex))
   rows.append(row);print(json.dumps(row),flush=True)
   (ROOT/'docs/research/continuous-ki-consistency/qualification.json').write_text(json.dumps(rows,indent=2)+'\n')

# Independent acceptance gates for the recorded actual-engine outputs.
import math
maximum_price = maximum_probability = 0.0
for case in {row['case'] for row in rows}:
    for family in {row['family'] for row in rows}:
        group = [row for row in rows if row['case'] == case and row['family'] == family]
        reference = next(row for row in group if row.get('reference'))
        for row in group:
            assert 'error' not in row, row
            if 'MC' not in row['engine']:
                error = abs(row['pv'] - reference['pv'])
                maximum_price = max(maximum_price, error)
                assert error <= .01, row
            for key in ('ki_ever', 'ki_settlement', 'ko'):
                error = abs(row[key] - reference[key])
                if 'MC' in row['engine']:
                    bound = 1.96 * math.sqrt(reference[key] * (1-reference[key]) / 100000) + .0005
                else:
                    bound = .0005
                    maximum_probability = max(maximum_probability, error)
                assert error <= bound, (case, family, row['engine'], key, error, bound)
(ROOT/'docs/research/continuous-ki-consistency/gates.json').write_text(json.dumps({
    'rows': len(rows), 'deterministic_price_gate_per_100': .01,
    'deterministic_probability_gate': .0005,
    'mc_probability_gate': '95% binomial sampling interval plus deterministic probability gate',
    'max_deterministic_price_difference': maximum_price,
    'max_deterministic_probability_difference': maximum_probability,
    'passed': True,
}, indent=2)+'\n')
