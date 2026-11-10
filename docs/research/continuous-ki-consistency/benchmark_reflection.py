"""Matched-grid cold price/statistics timing and native-kernel parity.
Run from the repository: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python docs/research/continuous-ki-consistency/benchmark_reflection.py
"""
import sys,json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'test')]
from test_asset.equity.engine_quad.test_autocallable_probability_consistency import CASES,env
from quantark.asset.equity.param import QuadV2Params

rows=[]
for family in ['snowball','phoenix_memory','ko_reset']:
 factory,spec,_,cls=CASES[family]
 spec=dict(spec,ki_monitoring='continuous')
 if family=='ko_reset':spec['ki_continuous']=True
 product,market=factory(spec),env()
 for backend in ['pointwise','auto']:
  params=QuadV2Params(continuous_grid_backend=backend)
  record={'family':family,'backend':backend,'order':params.order,'cells_per_sd':params.cells_per_sd}
  for op in ['price','stats']:
   samples=[]
   for repeat in range(3):
    engine=cls(params)
    start=time.perf_counter()
    value=engine.price(product,market) if op=='price' else engine.calculate_event_stats(product,market)
    samples.append(time.perf_counter()-start)
   record[op+'_cold_seconds']=statistics.median(samples)
   if op=='price':record['price']=value
   else:
    record.update(ki_ever=value.ki_ever_probability,ki_settlement=value.ki_survive_knocked_in_probability)
  rows.append(record);print(json.dumps(record),flush=True)
  (ROOT/'docs/research/continuous-ki-consistency/reflection_timing.json').write_text(json.dumps(rows,indent=2)+'\n')
# Compare the actual methods at identical grids, not timing at unlike accuracy.
for family in ['snowball','phoenix_memory','ko_reset']:
 old,new=[r for r in rows if r['family']==family]
 assert abs(old['price']-new['price'])<1e-8
 assert abs(old['ki_ever']-new['ki_ever'])<1e-8
 assert abs(old['ki_settlement']-new['ki_settlement'])<1e-8
