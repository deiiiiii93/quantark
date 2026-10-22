"""Bounded QUAD V2 feasibility probe, not a production pricing engine.

Compare FFT and banded application of the exact same saved-reference kernel.
Full-price runs temporarily replace only _apply in a single-threaded process.
The reference's domain/input/contract limits are deliberately not addressed
by this experiment. Backend agreement alone is not external validation.
"""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import scipy
from scipy.fft import rfft, irfft, next_fast_len
from scipy.stats import norm

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent/'gaussian_quad_comparison'
sys.path.insert(0,str(AUDIT))
import quad_reference_snapshot as ref
from contracts import cases


class FFTApply:
    """Linear block convolution with the original boundary extension.

    y[i,r] = sum_d sum_q state_extended[i+d,q] kernel[d,q,r].
    Reverse the kernel to express this correlation as linear convolution.
    Cache transforms only for one solve and retain kernel objects so Python
    object-id reuse cannot produce a false cache hit.
    """
    def __init__(self):
        self.transforms={}

    def __call__(self,mesh,kernel,state,band):
        padded=np.pad(state,((band,band),(0,0)),mode='edge')
        length=next_fast_len(len(padded)+len(kernel)-1)
        key=(id(kernel),length)
        cached=self.transforms.get(key)
        if cached is None:
            transformed=rfft(kernel[::-1],n=length,axis=0)
            self.transforms[key]=(kernel,transformed)
        else:
            transformed=cached[1]
        spectrum=rfft(padded,n=length,axis=0)
        result=irfft(np.einsum('fq,fqr->fr',spectrum,transformed,optimize=True),
                     n=length,axis=0)
        return result[2*band:2*band+mesh.n_cells]


ORIGINAL_APPLY=ref._apply


@contextmanager
def backend(kind):
    ref._apply=ORIGINAL_APPLY if kind=='banded' else FFTApply()
    try:yield
    finally:ref._apply=ORIGINAL_APPLY


def deal(c):
    return ref.Deal(maturity=c.maturity,payoff_alive=c.alive_payoff,payoff_ki=c.ki_payoff,
        ki_barrier=c.ki,ki_times=c.ki_times,ko_times=c.ko_times,ko_barriers=c.ko_levels,
        ko_payoffs=tuple(c.ko_payoff(t) for t in c.ko_times),terminal_breaks=(100.,))


def full_price(c,kind):
    with backend(kind):
        return ref.value_and_delta_operator(deal(c),ref.Market(c.rate,c.div,c.vol),
            [c.spot],cells_per_sd=2.,n_q=8)


def timed(fn,repeats=7):
    fn();samples=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();samples.append(time.perf_counter()-start)
    return dict(median_s=statistics.median(samples),min_s=min(samples),max_s=max(samples),samples_s=samples)


def operator_checks():
    rng=np.random.default_rng(20260911)
    for cells in [160,640,1600]:
        mesh=ref._build_mesh(-2.,2.,4/cells,8)
        for target_band in [8,32,128]:
            dt=(target_band*mesh.h/(9*.2))**2
            op=ref._Operator(mesh,ref.Market(.03,.01,.2));b=op.band(dt);k=op.kernel(dt)
            values=rng.normal(size=(mesh.n_cells,mesh.n_q))
            fft=FFTApply()
            a=ORIGINAL_APPLY(mesh,k,values,b);z=fft(mesh,k,values,b)
            gap=float(np.max(abs(a-z)))
            np.testing.assert_allclose(z,a,rtol=2e-12,atol=2e-12)
            yield dict(kind='operator',cells=mesh.n_cells,band=b,n_q=8,max_abs_gap=gap,
                banded=timed(lambda:ORIGINAL_APPLY(mesh,k,values,b)),
                fft=timed(lambda:fft(mesh,k,values,b)),
                banded_window_bytes=mesh.n_cells*(2*b+1)*mesh.n_q*8,
                fft_length=next_fast_len(mesh.n_cells+4*b))


def gamma_checks():
    """Check log-coordinate gamma conversion against analytical vanilla puts."""
    out=[]
    for spot in [75.,100.,125.]:
        for sigma in [.005,.2,.6]:
            for maturity in [.0001,.5,2.]:
                rate,div=.03,.01
                x=np.log(spot);m=(rate-div-sigma*sigma/2)*maturity
                v=sigma*sigma*maturity;discount=np.exp(-rate*maturity)
                query=np.array([x]);breaks=[np.log(100.)]
                payoff=lambda y:np.maximum(100.-np.exp(y),0.)
                kw=dict(n_gl=240,n_sd=10.)
                value=discount*ref._panel_integral(query,payoff,breaks,m,v,derivative=False,**kw)[0]
                dx=discount*ref._panel_integral(query,payoff,breaks,m,v,derivative=True,**kw)[0]
                dxx=discount*ref._panel_integral(query,
                    lambda y:payoff(y)*((y-x-m)**2/(v*v)-1/v),breaks,m,v,
                    derivative=False,**kw)[0]
                raw_gamma=(dxx-dx)/(spot*spot)
                # A large affine cash/asset payoff cancels in gamma and can
                # lose precision when v is tiny. Integrate that part exactly
                # and apply density derivatives only to the residual.
                forward=spot*np.exp((rate-div)*maturity)
                if forward<100.:
                    residual=lambda y:np.maximum(np.exp(y)-100.,0.)
                    correction_p=100*discount-spot*np.exp(-div*maturity)
                    correction_d=-np.exp(-div*maturity)
                else:
                    residual=payoff;correction_p=0.;correction_d=0.
                r0=discount*ref._panel_integral(query,residual,breaks,m,v,derivative=False,**kw)[0]
                r1=discount*ref._panel_integral(query,residual,breaks,m,v,derivative=True,**kw)[0]
                r2=discount*ref._panel_integral(query,
                    lambda y:residual(y)*((y-x-m)**2/(v*v)-1/v),breaks,m,v,
                    derivative=False,**kw)[0]
                value=r0+correction_p
                delta=r1/spot+correction_d
                gamma=(r2-r1)/(spot*spot)
                d1=(np.log(spot/100)+(rate-div+sigma*sigma/2)*maturity)/np.sqrt(v)
                d2=d1-np.sqrt(v)
                exact_p=100*np.exp(-rate*maturity)*norm.cdf(-d2)-spot*np.exp(-div*maturity)*norm.cdf(-d1)
                exact_d=-np.exp(-div*maturity)*norm.cdf(-d1)
                exact_g=np.exp(-div*maturity)*norm.pdf(d1)/(spot*np.sqrt(v))
                np.testing.assert_allclose(value,exact_p,rtol=2e-8,atol=2e-10)
                np.testing.assert_allclose(delta,exact_d,rtol=2e-8,atol=2e-8)
                np.testing.assert_allclose(gamma,exact_g,rtol=2e-8,atol=2e-6)
                out.append(dict(spot=spot,sigma=sigma,maturity=maturity,price=value,
                    delta=delta,gamma=gamma,price_error=value-exact_p,
                    delta_error=delta-exact_d,gamma_error=gamma-exact_g,
                    raw_gamma_error=raw_gamma-exact_g))
    return dict(kind='analytic_gamma_controls',checks=len(out),results=out)


def main():
    target=HERE/'results.json'
    result=dict(metadata=dict(date=datetime.now().isoformat(),python=sys.version,
        platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        reference_sha256=hashlib.sha256((AUDIT/'quad_reference_snapshot.py').read_bytes()).hexdigest(),
        experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),results=[])
    def save(row):
        result['results'].append(row);target.write_text(json.dumps(result,indent=2)+'\n')
        print(row.get('name',row['kind']),flush=True)
    for row in operator_checks():save(row)
    save(gamma_checks())
    for c in cases():
        if c.name not in ['monthly','daily','aged_five_days','step_down','three_years']:continue
        a=full_price(c,'banded');b=full_price(c,'fft')
        np.testing.assert_allclose(b,a,rtol=2e-11,atol=2e-10)
        save(dict(kind='full_price',name=c.name,price=float(a[0][0]),delta=float(a[1][0]),
            price_gap=float(b[0][0]-a[0][0]),delta_gap=float(b[1][0]-a[1][0]),
            banded=timed(lambda:full_price(c,'banded'),repeats=5),
            fft=timed(lambda:full_price(c,'fft'),repeats=5)))
    print('Saved',target,flush=True)


if __name__=='__main__':main()
