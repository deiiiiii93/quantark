"""Panel convergence and FFT-length performance sensitivity, run in isolation."""
import argparse
import json
import statistics
import time

from study import HERE, metadata, capture, panel, product, environment, SnowballQuadEngine, QuadParams
from contracts import cases


def main():
    args=argparse.Namespace(phase='extra_checks')
    data=dict(metadata=metadata(args),results=[])
    target=HERE/'results/extra_checks.json'
    def save(x):
        data['results'].append(x);target.write_text(json.dumps(data,indent=2)+'\n')
        print(x.get('name'),x.get('kind'),x.get('n'),flush=True)
    for c in cases():
        if c.name in ['monthly','aged_five_days','high_vol','step_down']:
            for n in [600,1200,2400,4800]:
                save(dict(name=c.name,kind='panel_convergence',n=n,
                          **capture(lambda:panel(c,n_grid=n,n_gl=240,span=2.5 if c.vol<.5 else 8.))))
        if c.name in ['monthly','daily']:
            p,e=product(c),environment(c)
            for n in [1013,7813]:
                eng=SnowballQuadEngine(QuadParams(grid_points=n,readout='transition'))
                value=eng.price(p,e);samples=[]
                for _ in range(7):
                    t=time.perf_counter();eng.price(p,e);samples.append(time.perf_counter()-t)
                actual=len(eng._last_spot_greeks_grid[0])
                save(dict(name=c.name,kind='fft_length',n=n,actual_grid=actual,fft_length=4*actual-2,price=value,
                          median_s=statistics.median(samples),samples_s=samples))
        if c.name in ['monthly','aged_five_days']:
            panel(c);samples=[]
            for _ in range(7):
                t=time.perf_counter();panel(c);samples.append(time.perf_counter()-t)
            save(dict(name=c.name,kind='panel_default_timing',median_s=statistics.median(samples),samples_s=samples))
    print('Saved',target)


if __name__=='__main__':main()
