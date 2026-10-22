"""Independent contracts and exact-GBM Monte Carlo for the Gaussian QUAD audit.

No QuantArk or quadrature-reference imports. Values are per 100 initial
notional. Default cashflows exclude principal, margin and funding legs.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm, qmc, t as student_t


@dataclass(frozen=True)
class Case:
    name: str = "monthly"
    spot: float = 100.0
    maturity: float = 1.0
    rate: float = 0.03
    div: float = 0.01
    vol: float = 0.20
    coupon: float = 0.12
    ki: float = 75.0
    ko_times: tuple = tuple(i / 12 for i in range(1, 13))
    ko_levels: tuple = (103.0,) * 12
    ki_times: tuple = tuple(i / 12 for i in range(1, 13))
    already_ki: bool = False
    disable_ko_after_ki: bool = False
    principal: bool = False
    age: float = 0.0
    loss_cap: float | None = None

    def alive_payoff(self, s):
        return np.full_like(np.asarray(s, dtype=float),
                            100 * (self.coupon * (self.age + self.maturity) + self.principal))

    def ki_payoff(self, s):
        loss = np.maximum(100.0 - np.asarray(s), 0.0)
        if self.loss_cap is not None:
            loss = np.minimum(loss, self.loss_cap)
        return 100.0 * self.principal - loss

    def ko_payoff(self, t):
        return 100.0 * (self.coupon * (self.age + t) + self.principal)


def cases():
    b = Case()
    daily = tuple(i / 252 for i in range(1, 253))
    carry = replace(b, name="daily_high_carry", ki_times=daily,
                    ki=90., vol=.26, rate=.02, div=.14, coupon=.18)
    short_times = tuple(i / 252 for i in range(1, 6))
    short = replace(carry, name="aged_five_days", maturity=5/252, age=247/252,
                    ki_times=short_times, ko_times=(5/252,), ko_levels=(103.,), spot=90.018)
    out = [b, replace(b, name="daily", ki_times=daily),
           replace(b, name="weekly", ki_times=tuple(i/52 for i in range(1,53))),
           carry, replace(carry, name="near_ki_above", spot=90.018),
           replace(carry, name="near_ki_below_alive", spot=89.982),
           short, replace(short, name="aged_already_ki", already_ki=True),
           replace(b, name="low_spot", spot=60.),
           replace(b, name="high_spot", spot=110.),
           replace(b, name="low_vol", vol=.03),
           replace(b, name="high_vol", vol=.60),
           replace(b, name="negative_rate", rate=-.02),
           replace(b, name="high_rate", rate=.10),
           replace(b, name="negative_carry", div=-.05),
           replace(b, name="already_ki", already_ki=True),
           replace(b, name="ko_disabled_after_ki", disable_ko_after_ki=True),
           replace(b, name="principal_included", principal=True),
           replace(b, name="step_down", ko_levels=tuple(np.linspace(110.,90.,12))),
           replace(b, name="irregular_daily", ki_times=daily,
                   ko_times=tuple(i/252 for i in (17,43,64,85,110,132,150,175,196,217,238,252))),
           replace(b, name="three_months", maturity=.25, ko_times=(1/12,2/12,3/12),
                   ki_times=(1/12,2/12,3/12), ko_levels=(103.,)*3),
           replace(b, name="three_years", maturity=3., ko_times=tuple(i/12 for i in range(1,37)),
                   ki_times=tuple(i/12 for i in range(1,37)), ko_levels=(103.,)*36),
           replace(b, name="ki_ko_close", ki=102.9),
           replace(b, name="one_day_to_maturity", maturity=1/252, age=251/252,
                   spot=99.98, ko_times=(1/252,), ki_times=(1/252,), ko_levels=(103.,)),
           replace(b, name="ki_only_at_maturity", ki_times=(1.,))]
    return out


def european_put(c):
    sd = c.vol * math.sqrt(c.maturity)
    if sd == 0:
        forward = c.spot * math.exp((c.rate-c.div)*c.maturity)
        return (math.exp(-c.rate*c.maturity)*max(100.-forward,0.),
                -math.exp(-c.div*c.maturity) if forward < 100. else 0.)
    d1 = (math.log(c.spot/100) + (c.rate-c.div+c.vol*c.vol/2)*c.maturity)/sd
    d2 = d1-sd
    value = 100*math.exp(-c.rate*c.maturity)*norm.cdf(-d2) - c.spot*math.exp(-c.div*c.maturity)*norm.cdf(-d1)
    delta = -math.exp(-c.div*c.maturity)*norm.cdf(-d1)
    return float(value), float(delta)


def mc(c, power=18, replicates=8, seed=20260911):
    """Exact transitions at event dates, independent scrambled Sobol replicates.

    Student-t confidence intervals use replicate estimates, not an iid-path
    formula applied to Sobol points. Forward and vanilla controls use the
    same paths, including paths already terminated by the autocallable.
    """
    times = np.array(sorted(set(c.ki_times + c.ko_times + (c.maturity,))))
    dt = np.diff(np.r_[0., times])
    ko_map = {round(t,12): (h,c.ko_payoff(t)) for t,h in zip(c.ko_times,c.ko_levels)}
    ki_set = {round(t,12) for t in c.ki_times}
    samples = []
    for rep in range(replicates):
        sobol = qmc.Sobol(len(times), scramble=True, seed=seed+1009*rep)
        sums = np.zeros(3)
        for start in range(0, 2**power, 2**13):
            size = min(2**13, 2**power-start)
            z = ndtri(np.clip(sobol.random(size), 1e-15, 1-1e-15))
            s = np.full(size, c.spot)
            active = np.ones(size, dtype=bool)
            hit = np.full(size, c.already_ki)
            cash = np.zeros(size)
            for j,t in enumerate(times):
                s *= np.exp((c.rate-c.div-c.vol*c.vol/2)*dt[j]+c.vol*math.sqrt(dt[j])*z[:,j])
                ko = ko_map.get(round(float(t),12))
                if ko is not None:
                    terminate = active & (s >= ko[0])
                    if c.disable_ko_after_ki:
                        terminate &= ~hit
                    cash += terminate*ko[1]*math.exp(-c.rate*t)
                    active &= ~terminate
                if round(float(t),12) in ki_set:
                    hit |= active & (s <= c.ki)
            cash += active*np.where(hit,c.ki_payoff(s),c.alive_payoff(s))*math.exp(-c.rate*c.maturity)
            sums += (cash.sum(), s.sum(), (np.maximum(100-s,0)*math.exp(-c.rate*c.maturity)).sum())
        samples.append((sums/2**power).tolist())
    a = np.asarray(samples)
    def summary(x):
        mean = float(x.mean())
        se = float(x.std(ddof=1)/math.sqrt(replicates))
        half = float(student_t.ppf(.975,replicates-1)*se)
        return dict(mean=mean,se=se,half95=half,lo95=mean-half,hi95=mean+half)
    return dict(**summary(a[:,0]), paths=2**power*replicates,replicates=replicates,
                seed=seed,replicate_estimates=samples,
                forward=dict(**summary(a[:,1]),exact=c.spot*math.exp((c.rate-c.div)*c.maturity)),
                vanilla_put=dict(**summary(a[:,2]),exact=european_put(c)[0]))
