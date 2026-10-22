"""Build the report, CSV tables and figures from saved experiment results."""
from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
ROOT=HERE.parents[1]


def read(name):
    return json.loads((RESULTS/name).read_text())


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                       '| '+' | '.join(['---']*len(headers))+' |']+
                      ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    matrix=read('matrix.json');rows=matrix['results'];byname={r['case']['name']:r for r in rows}
    bench=read('benchmark_monthly,daily,aged_five_days,three_years.json')['results']
    extra=read('extra_checks.json')['results']
    mc=read('mc_monthly,daily,daily_high_carry,near_ki_above,aged_five_days,high_vol,step_down,ki_ko_close,ko_disabled_after_ki.json')['results']
    greeks=read('greeks_monthly,near_ki_above,aged_five_days,ki_ko_close.json')['results']
    edges=read('edges.json')['results']
    assert len(rows)==25 and len(bench)==64 and len(mc)==9
    assert not any('error' in v for r in rows for group in ['reference','quantark'] for v in r[group].values())
    price_rows=[]
    for r in rows:
        p=r['reference']['fine']['price'][0]
        price_rows.append([r['case']['name'],f'{p:.9f}',
            r['quantark']['1001_legacy_linear']['actual_grid'],
            *[f"{100*(r['quantark'][key]['price']-p):+.5f}" for key in
              ['1001_legacy_linear','1001_transition','4001_transition','8001_transition']]])
    aggregate=[]
    for n in [1001,2001,4001,8001]:
        for mode in ['legacy_linear','transition']:
            err=np.array([100*(r['quantark'][f'{n}_{mode}']['price']-r['reference']['fine']['price'][0]) for r in rows])
            de=np.array([r['quantark'][f'{n}_{mode}']['bumps']['0.0001']['delta']-r['reference']['fine']['delta'][0] for r in rows])
            aggregate.append([n,mode,f'{np.median(abs(err)):.5f}',f'{max(abs(err)):.5f}',
                              f'{np.median(abs(de)):.7f}',f'{max(abs(de)):.7f}'])
    def timing(case,engine,n=None,mode=None,level=None,workload='price'):
        return next(x['median_s'] for x in bench if x['case']==case and x['engine']==engine
                    and x.get('n')==n and x.get('mode')==mode and x.get('level')==level and x['workload']==workload)
    perf=[]
    for c in ['monthly','daily','aged_five_days','three_years']:
        perf.append([c,*[f"{1000*timing(c,'quantark',n,'transition'):.2f}" for n in [1001,4001,8001]],
                     f"{1000*timing(c,'reference',level='default',workload='price_analytic_delta'):.2f}",
                     f"{1000*timing(c,'reference',level='fine',workload='price_analytic_delta'):.2f}"])
    curve_rows=[]
    for c in ['monthly','daily','aged_five_days','three_years']:
        curve_rows.append([c,f"{1000*timing(c,'reference_spot_curve',workload='101_spot_curve'):.2f}",
                           f"{1000*timing(c,'quantark_grid_curve',workload='101_spot_curve'):.2f}"])
    mc_rows=[]
    for r in mc:
        c=r['case']['name'];m=r['mc'];p=byname[c]['reference']['fine']['price'][0]
        assert m['lo95']<=p<=m['hi95'],c
        for control in ['forward','vanilla_put']:
            assert m[control]['lo95']<=m[control]['exact']<=m[control]['hi95'],(c,control)
        mc_rows.append([c,f'{p:.6f}',f"{m['mean']:.6f}",f"[{m['lo95']:.6f}, {m['hi95']:.6f}]",f"{(p-m['mean'])/m['se']:+.2f}"])
    headers=['Scenario','Reference PV / 100','QA actual N for requested 1001',
             'QA 1001 linear error bp','QA 1001 transition error bp','QA 4001 transition error bp','QA 8001 transition error bp']
    with (RESULTS/'price_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(price_rows)
    with (RESULTS/'timing_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scenario','qa_1001_transition_price_ms','qa_4001_transition_price_ms',
                                  'qa_8001_transition_price_ms','reference_default_price_delta_ms','reference_fine_price_delta_ms']);w.writerows(perf)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold'})
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for mode,color,label in [('legacy_linear','#d97706','Linear'),('transition','#2563eb','Transition')]:
        for stat,ls in [('max','-'),('median','--')]:
            vals=[]
            for n in [1001,2001,4001,8001]:
                e=[abs(100*(r['quantark'][f'{n}_{mode}']['price']-r['reference']['fine']['price'][0])) for r in rows]
                vals.append(max(e) if stat=='max' else np.median(e))
            axs[0,0].loglog([1001,2001,4001,8001],vals,marker='o',ls=ls,color=color,label=f'{label}: {stat}')
    axs[0,0].set(title='25 matched cases: price error',xlabel='Requested grid points',ylabel='Absolute error, bp of initial notional')
    axs[0,0].legend(fontsize=8);axs[0,0].grid(alpha=.15)
    g=next(x for x in greeks if x['name']=='monthly');q=byname['monthly']['quantark']
    for mode,color,label in [('legacy_linear','#d97706','QuantArk linear'),('transition','#2563eb','QuantArk transition')]:
        axs[0,1].semilogx([1.,.1,.01],[q['1001_'+mode]['bumps'][str(b)]['delta'] for b in [.01,.001,.0001]],'o-',color=color,label=label)
    axs[0,1].semilogx([1.,.1,.01],[g['matched_bumps'][str(b)]['delta'] for b in [.01,.001,.0001]],'s--',color='#059669',label='Reference, same bump')
    axs[0,1].axhline(g['reference']['delta'][0],color='#111827',ls=':',label='Reference point delta')
    axs[0,1].set(title='Monthly note: reducing the bump can expose error',xlabel='Central spot bump (%)',ylabel='Delta per index point');axs[0,1].invert_xaxis();axs[0,1].legend(fontsize=8)
    edge=edges[0];widths=[10,20,40,80,100,120]
    axs[1,0].plot(widths,[edge['reference'][str(float(x))]['price'][0] for x in widths],'o-',color='#7c3aed',label='Reference operator')
    axs[1,0].axhline(edge['negative_put_control'],color='#111827',ls='--',label='Analytical limit / converged panel')
    axs[1,0].set(title='Reference failure: low volatility, strong carry',xlabel='Reference span_sd',ylabel='PV per 100');axs[1,0].legend(fontsize=8)
    edge=edges[1];p=edge['panel']['4800']['price']
    for mode,color,label in [('legacy_linear','#d97706','Linear'),('transition','#2563eb','Transition')]:
        ns=[1001,2001,4001,8001,16001]
        axs[1,1].loglog(ns,[100*abs(edge['quantark'][f'{n}_{mode}']['price']-p) for n in ns],'o-',color=color,label=label)
    axs[1,1].set(title='QuantArk failure: very short first KO interval',xlabel='Requested grid points',ylabel='Absolute price error, bp');axs[1,1].legend(fontsize=8);axs[1,1].grid(alpha=.15)
    fig.savefig(HERE/'accuracy.png',dpi=170);fig.savefig(HERE/'accuracy.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');x=np.arange(4);width=.18
    for j,(key,label,color) in enumerate([(1,'QuantArk 1001: price','#93c5fd'),(2,'QuantArk 4001: price','#3b82f6'),
                                        (3,'QuantArk 8001: price','#1d4ed8'),(4,'Reference default: price + delta','#059669')]):
        ax.bar(x+(j-1.5)*width,[float(r[key]) for r in perf],width,label=label,color=color)
    ax.set_yscale('log');ax.set_xticks(x,[r[0].replace('_',' ') for r in perf]);ax.set_ylabel('Median milliseconds, log scale')
    ax.set_title('Measured latency depends on accuracy and grid size');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15)
    fig.savefig(HERE/'performance.png',dpi=170);fig.savefig(HERE/'performance.svg');plt.close(fig)

    max_ref_p=max(abs(r['reference']['default']['price'][0]-r['reference']['fine']['price'][0]) for r in rows)
    max_ref_d=max(abs(r['reference']['default']['delta'][0]-r['reference']['fine']['delta'][0]) for r in rows)
    panel_rows=[]
    for c in ['monthly','aged_five_days','high_vol','step_down']:
        p=byname[c]['reference']['fine']['price'][0]
        panel_rows.append([c,*[f"{next(x['price'] for x in extra if x['name']==c and x['kind']=='panel_convergence' and x['n']==n)-p:+.3e}" for n in [600,1200,2400,4800]]])
    fft_rows=[]
    for c in ['monthly','daily']:
        p=byname[c]['reference']['fine']['price'][0]
        v=next(x for x in extra if x['name']==c and x['kind']=='fft_length' and x['n']==7813)
        fft_rows.append([c,f"{1000*timing(c,'quantark',8001,'transition'):.2f}",f"{1000*v['median_s']:.2f}",f"{100*(v['price']-p):+.5f}"])
    specrows=[]
    for r in rows:
        c=r['case'];specrows.append([c['name'],c['spot'],f"{c['maturity']:.6f}",f"{100*c['rate']:g}%",f"{100*c['div']:g}%",f"{100*c['vol']:g}%",c['ki'],len(c['ko_times']),len(c['ki_times'])])

    text=rf'''# Gaussian quadrature reference versus QuantArk Snowball QUAD

**The reference operator is the stronger numerical benchmark for the shared flat-GBM, discrete-observation contracts; QuantArk is the broader production engine and usually the faster default scalar pricer. Neither is reliable across all scenarios without numerical controls.** The reference can silently fail when drift moves the relevant states outside its default domain. QuantArk can materially misprice a short first KO interval because its adaptive grid considers KI spacing only. These two failures prevent an unconditional winner.

This report compares the exact file supplied in the request, not the older implementations in `docs/quad/ref_scripts`. The supplied file contains both `value_and_delta` (panel integration) and `value_and_delta_operator` (a reusable banded operator). Most reference results below use the operator; panel results are explicitly labelled. QuantArk is evaluated at commit `{matrix['metadata']['git_head']}`. The reference was copied byte for byte to [quad_reference_snapshot.py](quad_reference_snapshot.py), SHA-256 `70938c07768f0512765bed99ca9d2c50dd561bb649fe595ebda90bbe3d090ac7`.

| Question | Finding |
| --- | --- |
| Accuracy in the 25 shared scenarios | Reference operator wins. It agrees with its refinement ladder, analytical controls, a separately discretized panel formulation, and Monte Carlo cross-checks. |
| Default scalar pricing speed | QuantArk wins for monthly, daily, and short-dated cases in the measured set; the reference operator is slightly faster for the three-year case. |
| High-accuracy price and delta | Reference operator is usually preferable in its supported scope. It avoids repeated bump repricing and is faster than the tested 4001/8001 QuantArk grids on several workloads. |
| Smooth point Greeks | Reference density delta wins. QuantArk's transition readout substantially improves results, but event projection, bump size, and moving mesh phase still matter. |
| Contract and workflow coverage | QuantArk wins: term inputs, reverse notes, continuous KI, settlement timing, lifecycle and event APIs. Some features require other QuantArk engines. |
| Robustness across all stresses | No unconditional winner. Both have concrete failure modes documented below. |

**What is actually matched.** The base contract has initial price and strike 100, one year remaining, r=3%, q=1%, volatility 20%, annual coupon 12%, KO=103 monthly and KI=75 monthly. The alive maturity payment is 12; the KI payment is `min(S_T-100,0)`; an early KO pays `12*t`. Both engines receive exactly the same observation times, barriers and signed cashflows. Payoff and resolved-schedule checks pass within floating-point precision for every matrix case. The reference and Monte Carlo payoffs are implemented independently of QuantArk.

All values are per 100 initial notional. **One basis point is 0.01 price unit.** Errors in basis points equal `100*(candidate-reference)`. Values normally exclude principal, prepayment, margin and funding legs; one explicitly labelled case includes principal at contractual redemption. For aged trades, accrued KO factors are explicitly `age+t` and the maturity rebate uses full contractual tenor. Already-KI reference cases reduce the two continuation branches to the KI branch; the supplied reference does not expose a lifecycle-state argument. Current spot below KI between observations does not create a historical KI.

**Why the methods differ.** Under the shared model, `X=log(S)` has an exact Gaussian increment with `m=(r-q-sigma²/2)*dt`, `v=sigma²*dt`. Both methods implement discounted backward expectation of an event-transformed value function. There is no Euler time-step bias in this comparison, and monitoring dates are not changed during numerical refinement.

The reference integrates smooth surviving and knocked-in continuation branches separately, splitting integration at active event thresholds. Its delta uses the differentiated Gaussian density:

$$V(S)=D\int F(y)p(y\mid\log S)\,dy,\qquad
\Delta(S)=\frac{{D}}{{S}}\int F(y)\frac{{y-\log S-m}}{{v}}p(y\mid\log S)\,dy.$$

The panel version evaluates Gauss–Legendre quadrature separately for each output point and uses cubic splines for smooth continuation. The operator version uses cellwise Gauss nodes, polynomial continuation, reusable banded transition kernels and sub-cell event integrals; constant KO regions use Gaussian CDFs. It does not interpolate a function across a KI or KO jump. A whole vector of spot queries shares one backward sweep.

QuantArk uses two nodal value arrays on a uniform log-price mesh, FFT convolution with trapezoid integration and spectral filtering, and cell-average event projection. Its default `legacy_linear` price interpolates the final nodal array in log spot. The optional `transition` readout evaluates the last discrete transition at the requested spot. This fixes the final interpolation defect but still uses the already projected event arrays. QuantArk's scalar delta/gamma come from central bump repricing; the default spot bump is 1%.

The panel cost is approximately `O(M*N*G*J)`, for M event intervals, N continuation points, G quadrature order and J event panels. The reference operator costs roughly `O(M*C*B*q²)` and materializes work arrays of size `O(C*B*q)`, where C is cells and B the transition bandwidth. QuantArk's discrete-KI sweep is roughly `O(M*N*log N)`, plus product construction/payoff work. These complexity expressions do not predict the timing winner at the small and medium grids actually used here; FFT factorization and dense-kernel implementation constants matter.

**Reference qualification.** Reference default settings are `(cells_per_sd=2, n_q=8, span_sd=10, n_gl=240, n_sd=9)`; medium uses `(3,10,10,360,9)`; fine uses `(4,12,12,480,10)`. The 25-case maximum default-to-fine difference is {max_ref_p:.3e} price units and {max_ref_d:.3e} delta units. This is a convergence observation, not a claimed universal error bound of that size. The drift-domain counterexample below shows why self-convergence is insufficient.

Four vanilla-put controls at rates -2%, 0%, 3%, and 10% agree with analytical prices and deltas to floating-point accuracy. Four single-observation KI/payoff controls agree with independent truncated-lognormal formulas; their analytical-price differences are below 3e-14. A no-event semigroup control agrees with a single European expectation. The same Monte Carlo harness also passes three deterministic zero-variance cashflow checks. These MC controls do not imply either quadrature engine supports zero volatility.

The panel formulation converges independently toward the operator on representative cases. Entries are signed panel PV minus fine operator PV, in price units:

{table(['Case','600 points','1200 points','2400 points','4800 points'],panel_rows)}

Five representative cases also pass independent domain and final-quadrature-order checks: reference span 10/12/16 and final Gauss order 120/240/480/960 give stable prices. QuantArk domain changes at fixed N also change mesh spacing and event phase, so they are sensitivity diagnostics rather than pure tail-truncation tests.

Monte Carlo uses exact GBM transitions at the contractual events, 8 independent scrambled Sobol replicates of 524,288 paths, **4,194,304 paths per case**, seed 20260911 plus replicate offsets. The 95% intervals use Student t with 7 degrees of freedom across replicate means. All nine reference prices, all nine analytical forwards and all nine vanilla-put controls lie inside their corresponding intervals. The intervals are wider than many deterministic-engine differences and cannot establish sub-basis-point accuracy by themselves.

{table(['Case','Reference','MC mean','MC 95% interval','Reference minus MC, SE'],mc_rows)}

**Price accuracy across the shared scenarios.** These are signed errors against the fine qualified operator, with no extrapolation or tolerance adjustment. `1001` is the requested QuantArk grid; its actual grid can be larger due to automatic KI resolution. Both engines price every contractual discrete observation in this matrix. BGK approximation is not enabled.

{table(headers,price_rows)}

At default settings the maximum absolute price error is 1.010 bp; switching only the readout lowers that maximum to 0.537 bp. At 8001 points the maximum is approximately 0.011 bp. A small price error does not imply an accurate delta. Aggregate delta errors below compare 0.01% central spot bumps with reference point delta; the matched reference-bump diagnostic is retained separately to distinguish finite-bump error.

{table(['Requested N','Readout','Median absolute PV error, bp','Maximum absolute PV error, bp','Median absolute delta error','Maximum absolute delta error'],aggregate)}

Refinement is not reliably monotone for an individual case. For the monthly contract, transition prices at 1001/2001/4001/8001 are 1.910333140 / 1.909919916 / 1.910256384 / 1.910340351 versus 1.910364003. The lucky coarse-grid cancellation at 1001 does not certify its neighboring grid sizes. The step-down case retains a delta error around 0.00167 at 1001 with transition readout despite a price error of only -0.122 bp.

![Accuracy and failure diagnostics](accuracy.png)

**Delta and gamma consistency.** For the monthly case, the reference point delta is 0.0075685524. QuantArk at 1001 points demonstrates why simply shrinking the bump is insufficient:

| Relative spot bump | Reference using same bump | QuantArk linear delta | QuantArk transition delta |
| --- | --- | --- | --- |
| 1% | 0.008925082 | 0.009422224 | 0.008950041 |
| 0.1% | 0.007582146 | 0.002885380 | 0.007608286 |
| 0.01% | 0.007568688 | -0.001406697 | 0.007594840 |

The linear readout's small-bump delta has the wrong sign here. Its 0.01%-bump gamma is +0.0000141 versus the reference -0.0392434; transition gives -0.0392093. Linear interpolation in log spot creates a piecewise `a+b*log(S)` price, so within a cell the computed curvature can be dominated by the interpolant rather than the option's economic curvature.

For the aged five-day note near KI, point delta is 4.81574447 while the reference's own 1% central-bump delta is 4.68457470, a 2.72% difference. QuantArk transition gives 4.68414475 at that same bump, close to the matched reference. Much of the gap to point delta in this example is the bump convention. At a 0.01% bump, transition gives 4.81523760 and the matched reference gives 4.81573075. Use common bump ladders before attributing a hedge discrepancy to the engine.

The reference returns analytic density delta but no public analytic gamma; gamma validation here differentiates its delta along a converged spot-bump ladder. Neither engine's vega, rho, theta or bucket risks are certified by this report.

**API and event consistency.** QuantArk's `calculate_spot_greeks_curve` uses the stored nodal grid and numerical gradients even when scalar pricing uses `readout="transition"`. On the monthly 21-spot probe, its curve and scalar transition prices differ by up to 0.190 bp; maximum curve delta error is 0.000911, versus 0.0000424 for directly repriced transition deltas on the same spots. It is a useful fast approximation with different finite-grid behavior, not the identical scalar pricing function. The reference vector query shares the same density readout for every spot; mesh-domain changes from including more query spots should still be checked.

QuantArk's event PV, cashflow sum and scalar PV reconcile in all three tested cases and both readouts. However, `expected_discounted_maturity_cf = pv - sum(ed_ko_cf)` in the implementation: zero reconciliation error is partly an accounting identity. It does not independently validate KO probabilities, maturity attribution or total PV. Explicit event statistics are a useful QuantArk capability that the supplied reference does not expose.

**Failure 1: reference domain excludes the economically relevant states.** Set S=100, T=1, r=3%, q=40%, volatility=0.5%, monthly KO=103, monthly KI=75 and coupon=12%. The operator chooses a spot-centered domain based on `span_sd*sigma*sqrt(T)` without drift. Its default log half-width is only 0.05, but expected log drift is about -0.37 and KI lies at log(0.75)=-0.288. Refining cell order inside that domain cannot recover the missing regime.

| Reference span_sd | Reference PV |
| --- | --- |
| 10 | +11.64534640 |
| 20 | +11.64534640 |
| 40 | +11.64534640 |
| 80 | -30.01254875 |
| 100 | -30.01254875 |
| 120 | -30.01254875 |

The wide-domain panel gives -30.0125487513, QuantArk transition at 4001 gives -30.0125487443, and the negative European-put limiting control gives -30.0125487513. Default-reference error is about **41.658 price units, or 4165.79 bp of notional**. The reference's default 10-to-12 domain/refinement ladder alone would falsely look stable. A robust reference must include drift and relevant event/payoff levels in domain construction and actively test tails.

**Failure 2: QuantArk misses a short KO diffusion interval in adaptive sizing.** Set S=103.001, r=3%, q=1%, volatility=20%, T=1, KO times `[0.0001,0.5,1]` at 103, and KI only at T at 75. The first KO is roughly 0.0252 trading day ahead. Panel prices at 1200/2400/4800 converge to 3.2277326782; analytic-density delta converges to -12.5304812116. QuantArk's grid resolver receives only KI times and therefore does not refine for the first KO.

| Requested N | Linear PV | Transition PV | Transition price error, bp |
| --- | --- | --- | --- |
| 1001 | 3.976585407 | 3.981385376 | +75.36527 |
| 2001 | 3.232454262 | 3.231269161 | +0.35365 |
| 4001 | 3.229139208 | 3.228242819 | +0.05101 |
| 8001 | 3.228117058 | 3.227841365 | +0.01087 |
| 16001 | 3.227830569 | 3.227760067 | +0.00274 |

Transition readout is not a substitute for resolving the diffusion kernel. Size the mesh from every merged KO/KI/maturity interval and its forward variance. The supplied reference operator also becomes costly when one tiny interval forces an extremely fine mesh for all other intervals; the panel form is the practical control used here.

**Other limits and convergence controls.** The operator's exact-float set union does not merge near-coincident event dates, although its event lookup uses a tolerance. A KO at `0.25+1e-14` beside KI at `0.25` implies about 400 million reference cells at default resolution. This was diagnosed from mesh sizing without allocating that mesh. Dates need canonicalization before calling the reference. Exact zero volatility is unsupported by these public Snowball pricing paths; QuantArk rejects it and the reference lacks a deterministic branch.

QuantArk's `auto_converge` checks successive PVs, not a certified error bound or Greek accuracy. In the near-coincident KI=102.9 / KO=103 case it accepts 4001 points with an estimated 0.01482 bp difference, while error to the qualified reference is 0.08507 bp. Its tolerance also scales with total PV, including any principal. Use independent error targets in notional units, domain/phase tests and Greek convergence in addition to this stop rule.

**Performance at the actual workloads.** Timings below are median wall-clock milliseconds over seven repetitions after warmup. Python/import initialization is excluded, and pricing runs rebuild their numerical work rather than retrieving a cached quote. QuantArk product/environment construction is outside timed engine calls; the reference timing includes its thin adapter and mesh metadata construction. All study timing jobs run serially with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`; host scheduling and thermal noise remain. Environment: Python 3.11.8, NumPy 2.4.6, SciPy 1.17.1, macOS ARM64. Every timing sample is saved.

The QuantArk columns return **price only**; reference columns return **price and analytic delta together**, its public operator workload. The reference default is already converged in this matrix; equal node counts would not be an equal-accuracy comparison.

{table(['Case','QA transition 1001, ms','QA transition 4001, ms','QA transition 8001, ms','Reference default PV+delta, ms','Reference fine PV+delta, ms'],perf)}

QuantArk default linear scalar medians are {1000*timing('monthly','quantark',1001,'legacy_linear'):.2f} ms monthly, {1000*timing('daily','quantark',1001,'legacy_linear'):.2f} ms daily, {1000*timing('aged_five_days','quantark',1001,'legacy_linear'):.2f} ms aged-five-day, and {1000*timing('three_years','quantark',1001,'legacy_linear'):.2f} ms three-year; use the raw JSON for ranges and exact values. Reference default operator is about 103 times faster than its own default panel function on the monthly case: 8.62 ms versus 884.54 ms. On the aged case the panel takes 224.56 ms versus 6.96 ms for the operator. Choosing the panel function would give a very different performance ranking for the same supplied file.

Price-and-Greek timing is also recorded separately. Monthly QuantArk transition at 1001 takes 14.59 ms for price/delta/gamma via three repricings versus reference price/delta at 8.62 ms. This comparison includes an additional gamma output for QuantArk and uses its default finite bump; it is not a claim that the delivered Greeks have equal accuracy.

![Performance comparison](performance.png)

A 101-spot curve reuses one sweep in each API. Reference returns pointwise density deltas; QuantArk at 4001 uses interpolation and grid gradients, with the consistency limitation above:

{table(['Case','Reference default 101-spot PV+delta, ms','QuantArk 4001 grid curve, ms'],curve_rows)}

FFT length is a material implementation sensitivity. At 8001 nodes QuantArk uses transform length 32002; its large prime factor makes NumPy FFT slower. At 7813 nodes the transform length is 31250, which factors into small primes. Without changing engine code:

{table(['Case','QA 8001 transition price, ms','QA 7813 transition price, ms','QA 7813 error, bp'],fft_rows)}

Thus the tested high-grid latency improves by roughly threefold just from a nearby grid size. Mesh-phase cancellation also changes prices, so grid selection must be justified by accuracy, not chosen solely for speed or a lucky price. The reference operator remains faster than these 7813-point QuantArk measurements in the two tested cases. A production FFT implementation using suitable padded lengths could improve this tradeoff further; that optimization was not implemented here.

**Coverage is a separate dimension from numerical accuracy.** A missing feature is not a mispricing comparison. The following combines inspected API coverage with the focused regression tests; it is not external certification of every feature.

| Feature | Supplied reference panel | Supplied reference operator | QuantArk Snowball QUAD |
| --- | --- | --- | --- |
| Flat r/q/vol, standard down-KI/up-KO | Yes | Yes | Yes |
| Irregular discrete dates, KO step-down | Yes, with date hygiene | Yes, short steps can be expensive | Yes, but adaptive sizing gap noted above |
| Multiple terminal payoff kinks / loss cap | Caller supplies all breaks | Explicitly rejects more than one terminal break | Product payoff support; loss-cap comparison run |
| Already KI | Algebraic payoff adapter required | Algebraic payoff adapter required | Lifecycle support |
| Reverse up-KI/down-KO | No native direction parameter | No native direction parameter | Yes |
| Time-varying deterministic rates/carry/variance | Market accepts scalar values only | Market accepts scalar values only | Per-interval term inputs |
| Continuous KI / BGK approximation | No | No | Brownian bridge; BGK is explicit opt-in; transition readout refuses continuous KI |
| Settlement delays | Caller must pre-discount/resolve cashflows | Same | Native event and terminal timing |
| Event probabilities / cashflow distribution | No public API | No public API | Yes |
| Valuation-date observation / expiry handling | Positive future times required | Positive future times required | Native lifecycle and immediate-event handling |
| Phoenix coupons, memory, KO reset | Not represented by this Deal | Not represented by this Deal | Requires the separate appropriate QuantArk engine; not this comparison's Snowball engine |
| Heston / local-volatility dynamics | No | No | This Gaussian QUAD engine is not a stochastic/local-vol solver |

**Recommended use.** Use QuantArk for production integration and broad contractual coverage. For ordinary prices its default configuration is a useful fast starting point, with an explicit per-product accuracy gate. For discrete-KI delta work, evaluate `readout="transition"`, choose spatial resolution using the full event calendar, and validate matched bump ladders; do not silently change existing marks, goldens or hedge sizes on the basis of this report. Use the supplied operator as an independent price/delta benchmark for its supported contracts only after a drift-aware domain check, date canonicalization and an analytical/MC control. Use the panel version when the operator cannot handle multiple payoff breaks or when a tiny interval makes its mesh impractical.

The highest-priority QuantArk improvements are complete-event adaptive sizing, consistent scalar/curve readout, and convergence checks that separate PV accuracy from Greek accuracy. The highest-priority reference improvements are drift-aware domain construction, finite/positive input checks, tolerant event-time merging, explicit lifecycle/direction semantics and a deterministic zero-variance path. FFT-length optimization is worthwhile after correctness. No production changes are included in this analysis.

**Reproduction and artifacts.** Run from the repository root. The original external reference is preserved in a local snapshot so the study does not depend on the temporary directory surviving. Full inputs, source hashes, actual grids, bumped prices, confidence intervals and timing repetitions are in [results](results/). CSV exports are [price_comparison.csv](results/price_comparison.csv) and [timing_comparison.csv](results/timing_comparison.csv). The 74 focused existing tests passed; output is [pytest.txt](results/pytest.txt).

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python example/gaussian_quad_comparison/study.py --phase controls
.venv/bin/python example/gaussian_quad_comparison/study.py --phase matrix
.venv/bin/python example/gaussian_quad_comparison/study.py --phase mc --only monthly,daily,daily_high_carry,near_ki_above,aged_five_days,high_vol,step_down,ki_ko_close,ko_disabled_after_ki --power 19 --replicates 8
.venv/bin/python example/gaussian_quad_comparison/study.py --phase diagnostics --only monthly,near_ki_above,aged_five_days,high_vol,ki_ko_close
.venv/bin/python example/gaussian_quad_comparison/supplement.py --phase edges
.venv/bin/python example/gaussian_quad_comparison/supplement.py --phase greeks --only monthly,near_ki_above,aged_five_days,ki_ko_close
.venv/bin/python example/gaussian_quad_comparison/supplement.py --phase events --only monthly,near_ki_above,aged_five_days
.venv/bin/python example/gaussian_quad_comparison/study.py --phase benchmark --only monthly,daily,aged_five_days,three_years
.venv/bin/python example/gaussian_quad_comparison/extra_checks.py
.venv/bin/python example/gaussian_quad_comparison/build_report.py
.venv/bin/python -m pytest -n0 -q test/test_snowball_quad_engine.py test/test_quad_readout.py test/test_quad_term_structure_engines.py test/test_snowball_lifecycle_ki.py test/test_quad_event_stats_smoothing.py
```

These are a scenario study and identified failure cases, not exhaustive engine-release certification. Timing is machine dependent. Reference self-convergence is not a statistical confidence interval. Monte Carlo intervals describe sampling uncertainty, not numerical model misspecification. No live market calibration, settlement-specific external repricing, continuous-KI external accuracy ladder, smile dynamics, or non-spot Greek certification is claimed.

**Scenario details.** Full arrays and coupon/principal/lifecycle flags are in matrix.json. All unspecified KO levels are 103; step-down uses 110 to 90 over 12 dates; irregular daily uses day indices 17/43/64/85/110/132/150/175/196/217/238/252. Daily high-carry and related near-KI/aged cases use an 18% annual coupon; other cases use 12%. Aged cases preserve a one-year original tenor.

{table(['Scenario','Spot','Remaining T','r','q','vol','KI','KO count','KI count'],specrows)}

**Implementation evidence.** The relevant code locations are [reference panel](quad_reference_snapshot.py#L169), [reference operator](quad_reference_snapshot.py#L499), [reference mesh/domain](quad_reference_snapshot.py#L517), [QuantArk event projection and readout]({ROOT}/quantark/asset/equity/engine/quad/snowball_quad_engine.py:331), [adaptive grid]({ROOT}/quantark/asset/equity/engine/quad/snowball_quad_engine.py:1460), [term inputs]({ROOT}/quantark/asset/equity/engine/quad/term_inputs.py:29), [bump Greeks]({ROOT}/quantark/asset/equity/engine/base_engine.py:221), [grid curve]({ROOT}/quantark/asset/equity/engine/base_engine.py:298), [residual maturity cashflow]({ROOT}/quantark/asset/equity/engine/quad/snowball_quad_engine.py:1264), and [QUAD defaults]({ROOT}/quantark/asset/equity/param/engine_params.py:767).
'''
    (HERE/'REPORT.md').write_text(text)
    print('Wrote',HERE/'REPORT.md')
    print('Assertions passed: 25 matched cases, 64 timing rows, 9 MC and 18 analytical MC controls.')


if __name__=='__main__':main()
