"""Matched-grid cold-call timings and scalar/batched ADI parity."""
import json
import statistics
import sys
import time
import types
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'test')]
from pde_statistics.test_vol_model_ki_consistency import HP, note, _flat_env, CLASSES
from quantark.util.numerical.tridiag import tridiag_backend

# The prior production price implementation is read from the checkout's HEAD.
# Only price comparisons use it; it has no native statistics or memory Phoenix.
legacy={}
for family, file in [('snowball','snowball_vol_pde_solvers.py'),('phoenix','phoenix_vol_pde_solvers.py')]:
    name='quantark.asset.equity.engine.pde._ki_baseline_'+family
    module=types.ModuleType(name)
    module.__package__='quantark.asset.equity.engine.pde'
    sys.modules[name]=module
    source=subprocess.check_output(['git','show','HEAD:quantark/asset/equity/engine/pde/'+file],cwd=ROOT,text=True)
    exec(compile(source,name,'exec'),module.__dict__)
    legacy[family]=getattr(module,'Heston'+family.title()+'PDESolver')

rows=[]
for family in ('snowball','phoenix','phoenix_memory','ko_reset'):
    cls=CLASSES['heston'][2 if family=='ko_reset' else 1 if 'phoenix' in family else 0]
    p,env=note(family),_flat_env()
    for operation in ('price','statistics'):
        configs=['scalar','auto']+(['prior'] if family in legacy and operation=='price' else [])
        values={};samples={}
        for config in configs:
            timings=[]
            for repeat in range(3):
                factory=legacy[family] if config=='prior' else cls
                kwargs={} if config=='prior' else dict(column_backend=config)
                e=factory(HP,n_x=120,n_v=24,n_t=96,**kwargs)
                start=time.perf_counter()
                if operation=='price':
                    values[config]=dict(pv=e.price(p,env))
                else:
                    s=e.calculate_event_stats(p,env)
                    values[config]=dict(pv=s.pv,ki=s.ki_survive_knocked_in_probability,ever=s.ki_ever_probability)
                timings.append(time.perf_counter()-start)
            samples[config]=timings
        parity={k:{f:abs(v-values['auto'][f]) for f,v in values[k].items()} for k in configs if k!='auto'}
        row=dict(family=family,operation=operation,backend=tridiag_backend(),grid=dict(n_x=120,n_v=24,n_t=96),seconds={k:statistics.median(v) for k,v in samples.items()},samples=samples,values=values,differences=parity)
        rows.append(row)
        print(json.dumps(row),flush=True)
Path(__file__).with_name('timing.'+tridiag_backend()+'.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
    assert max(row['differences']['scalar'].values()) < 2e-11
