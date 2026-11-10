"""Reproduce native vol-model KI qualification with actual engine outputs."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
from pde_statistics.test_vol_model_ki_consistency import CLASSES, HP, note
from test_asset.equity.engine_pde.test_vol_pde_continuous_ki import _skewed_env, _flat_env
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import LocalVolSnowballMCEngine, HestonSnowballMCEngine, HestonSLVSnowballMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import LocalVolPhoenixMCEngine, HestonPhoenixMCEngine, HestonSLVPhoenixMCEngine
from quantark.asset.equity.param import PDEParams, MCParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.param.div import ForwardCarryCurve

OUT = Path(__file__).parent
PATHS = 100000
records = []
leverage = LeverageSurface(time_grid=np.linspace(0, 1, 5),
    strike_grid=100*np.exp(np.linspace(-.8, .8, 13)),
    leverage_grid=np.tile(1.08-.12*np.linspace(-.8, .8, 13), (5, 1)))


def make_engine(model, family, config):
    index = 2 if family == 'ko_reset' else 1 if 'phoenix' in family else 0
    if config != 'mc':
        cls = CLASSES[model][index]
        if model == 'lv':
            return cls(PDEParams(accuracy='high' if config == 'fine' else 'standard'))
        return cls(HP, **({'leverage_surface': leverage} if model == 'slv' else {}),
            n_x=240 if config == 'fine' else 160, n_v=48 if config == 'fine' else 32,
            n_t=384 if config == 'fine' else 192)
    cls = ({'lv': LocalVolPhoenixMCEngine, 'heston': HestonPhoenixMCEngine,
            'slv': HestonSLVPhoenixMCEngine} if index == 1 else
           {'lv': LocalVolSnowballMCEngine, 'heston': HestonSnowballMCEngine,
            'slv': HestonSLVSnowballMCEngine})[model]
    kw = dict(params=MCParams(num_paths=PATHS, seed=917, use_antithetic=False), ki_bridge_nodes_per_day=2)
    if model != 'lv':
        kw['model_params'] = HP
    if model == 'slv':
        kw['leverage_surface'] = leverage
    return cls(**kw)


for model in CLASSES:
    env = _skewed_env() if model == 'lv' else _flat_env()
    env.div_yield = ForwardCarryCurve([(0.12, .012), (.29, -.004), (.5, .008), (1., .016)]).to_dividend_yield(env.rate_curve)
    for family in ('snowball', 'phoenix', 'phoenix_memory', 'ko_reset'):
        product = note(family)
        for config in ('coarse', 'fine', 'mc'):
            e = make_engine(model, family, config)
            start = time.perf_counter()
            s = e.calculate_event_stats(product, env)
            row = dict(model=model, family=family, config=config, seconds=time.perf_counter()-start,
                pv=s.pv, ki_ever=s.ki_ever_probability, ki_settlement=s.ki_survive_knocked_in_probability,
                ko=float(s.pre_ko_probability_total+s.post_ko_probability_total if family == 'ko_reset' else s.ko_probability.sum()),
                reconciliation=s.reconciliation_error)
            if config == 'mc':
                row['paths'] = PATHS
                row['seed'] = 917
                row['price_check'] = e.price(product, env)
                row['price_standard_error'] = e.get_last_std_error()
                if row['price_standard_error'] is None:
                    raise RuntimeError('MC price did not expose its sampling error')
            else:
                row['grid'] = {'accuracy': e.params.accuracy} if model == 'lv' else dict(n_x=e.n_x, n_v=e.n_v, n_t=e.n_t)
            records.append(row)
            (OUT/'qualification.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps(row), flush=True)

checks=[]
for model in CLASSES:
    for family in ('snowball', 'phoenix', 'phoenix_memory', 'ko_reset'):
        rows={r['config']:r for r in records if r['model']==model and r['family']==family}
        ref=rows['fine']
        for config in ('coarse','mc'):
            r=rows[config]
            for field in ('ki_ever','ki_settlement','ko','pv'):
                delta=abs(r[field]-ref[field])
                allowance=.01 if field=='pv' else .0005
                if config=='mc':
                    if field=='pv':
                        se=r['price_standard_error']
                        allowance += 1.96*se if se is not None else 0.
                    else:
                        allowance += 1.96*np.sqrt(max(r[field]*(1-r[field]), 0.)/PATHS)
                checks.append(dict(model=model,family=family,config=config,field=field,difference=delta,allowance=allowance,passed=bool(delta<=allowance)))
(OUT/'gates.json').write_text(json.dumps(dict(checks=checks,passed=all(c['passed'] for c in checks)),indent=2)+'\n')
print('gates',sum(c['passed'] for c in checks),'/',len(checks),flush=True)
if not all(c['passed'] for c in checks):
    raise SystemExit(1)
