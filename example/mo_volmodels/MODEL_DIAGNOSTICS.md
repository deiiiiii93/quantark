# Heston identification diagnostics for MO options

This note defines what the MO example suite measures when it reports a Heston
Jacobian, singular-value decomposition (SVD), multiplier bootstrap, and
cross-date fit. These are diagnostics of the saved calibration experiment. They
do not establish that the fitted dynamics are unique, stable outside the chosen
box, or suitable for production risk.

There are two deliberately separate evidence cohorts:

- `04_heston_calibration.py` diagnoses the saved intraday Sina midpoint
  snapshot and its SABR-prepared OTM-IV target.
- `10_calibration_diagnostics.py` diagnoses independently dated official CFFEX
  end-of-day settlement cross sections without interpolation, extrapolation, or
  smile smoothing.

Midpoints and settlements are different price fields with different execution
semantics. The suite never combines them into one calibration panel.

## Quote-only, bound-aware Jacobian

Let

\[
\theta=(v_0,\kappa,\bar v,\sigma,\rho)
\]

be the Heston parameter vector and let
\(\widehat\sigma_i(\theta)\) be the model implied volatility at calibration node
\(i\). For the intraday fit, the reported quote matrix is

\[
J_{ij}=\frac{\partial\widehat\sigma_i(\theta)}{\partial\theta_j}.
\]

It differentiates model implied vols only. In particular, the soft Feller
penalty is excluded, so regularization cannot masquerade as information supplied
by option prices. The derivative of a residual would differ only by a sign;
that sign does not change singular values or rank.

The settlement-date objective assigns every expiry total weight one. Its SVD
therefore uses the objective-consistent row-weighted matrix

\[
J_w=\operatorname{diag}(\sqrt{w_1},\ldots,\sqrt{w_n})J.
\]

This prevents a denser expiry from dominating the identification diagnostic for
the same reason it does not dominate the fit. The cross-date JSON retains both
the weighted matrix used for SVD and the unweighted market-IV matrix for audit.

For an interior parameter, column \(j\) uses the second-order central stencil

\[
J_{\cdot j}\approx
\frac{f(\theta+h_j e_j)-f(\theta-h_j e_j)}{2h_j}.
\]

At a lower or upper box boundary it uses a second-order one-sided stencil:

\[
\frac{-3f(\theta)+4f(\theta+h_j e_j)-f(\theta+2h_j e_j)}{2h_j},
\qquad
\frac{3f(\theta)-4f(\theta-h_j e_j)+f(\theta-2h_j e_j)}{2h_j}.
\]

The step is parameter-relative but cannot become negligible:

\[
h_j=\min\left(
\max\left(10^{-4}|\theta_j|,10^{-6}(u_j-l_j)\right),
\tfrac14(u_j-l_j)
\right).
\]

Every perturbation must return the same finite node vector. A failed pricing or
IV inversion aborts the diagnostic; the implementation does not replace an
invalid Jacobian column with zeros or NaNs.

## Scaling and SVD semantics

Heston parameters have different units, so an unscaled condition number is not
an invariant property of the calibration. For a positive scale vector \(s\), the
intraday suite decomposes

\[
J_s=J\,\operatorname{diag}(s)=U\Sigma V^\mathsf{T}.
\]

The cross-date panel instead decomposes
\(J_{w,s}=J_w\operatorname{diag}(s)\). Thus both the parameter-unit policy and
the equal-total-per-expiry row policy are fixed across dates.

It records four explicit policies:

| Policy | Scale vector | Intended use |
|---|---|---|
| `raw` | \((1,1,1,1,1)\) | Reproduce derivatives in native parameter units only |
| `fit_relative` | fitted magnitudes with floors \((10^{-4},0.1,10^{-4},0.01,0.1)\) | Interpret sensitivity near this particular fit |
| `fixed_economic` | \((0.01,1,0.01,0.1,0.1)\) | Like-for-like comparison across dates |
| `bound_span` | \(u-l\) | Show sensitivity over the configured optimization box |

The condition number is \(\kappa_J=s_1/s_5\) when the smallest singular value is
above the machine-rank tolerance. The numerical rank uses a floating-point
tolerance. The separately reported policy-effective rank counts singular values
larger than \(10^{-3}s_1\). Right singular vectors identify parameter
*combinations* that are weakly or strongly visible locally; their signs are
canonicalized only to make artifacts deterministic.

Only `fixed_economic` condition numbers should be compared through time. A full
five-column local rank is not proof of global identification: the objective may
still have distant near-equivalent minima, active bounds, or material instability
under small changes in node influence. Conversely, a large condition number is a
warning under the stated scale policy, not a universal model constant.

## Maturity-stratified multiplier bootstrap

For the current-date prepared target, replicate \(b\) draws independent
multiplier weights

\[
z_i^{(b)}\sim\operatorname{Exp}(1),\qquad
w_i^{(b)}=z_i^{(b)}
\frac{n_m}{\sum_{k\in m}z_k^{(b)}}
\quad(i\in m),
\]

where \(m\) is a maturity stratum and \(n_m\) is its original node count. Thus
each maturity retains its original total weight while influence is redistributed
among its nodes. Each replicate uses the same target nodes, initial parameter
vector, bounds, Feller policy, pricing method, tolerances, and evaluation budget
as the main fit.

The JSON artifact retains every successful and failed solve, the random seed,
parameter quantiles, covariance/correlation when defined, bound-hit rates,
Feller-pass fraction, and both full-sample and reweighted RMSE distributions.

This is **conditional node-influence evidence**, not a statistical confidence
interval. The saved smile is one prepared cross section: its nodes are dependent,
the smoothing step is held fixed, and no quote-time sampling process is modeled.
Bootstrap quantiles therefore answer “how much does this configured optimum move
when the influence of these nodes changes?”, not “what is the population
confidence interval for Heston parameters?”

## Official CFFEX settlement cross-date cohort

`01_fetch_mo_settlement_history.py` freezes one official CFFEX daily-statistics
CSV per requested trading date as
`data/mo_settlement_snapshot_YYYYMMDD.json`. Each snapshot records the official
URL, a SHA-256 digest of the source bytes, close and settlement as distinct
fields, volume, open interest, and the expiry calendar used by the study.
Calibration uses settlement only.

The source is genuine exchange end-of-day history, but a settlement value is not
an executable bid/ask midpoint. It cannot measure spread width, staleness within
the session, or the price at which a desk could have traded. See the
[CFFEX historical-data service](https://www.cffex.com.cn/lssjfw/),
[historical-data download page](https://www.cffex.com.cn/lssjxz/), and
[CSI 1000 index-option product page](https://www.cffex.com.cn/zz1000gzqq/).

### Parity normalization

For each expiry, ordinary least squares fits listed call-put pairs to

\[
C(K,T)-P(K,T)=DF(T)\,[F(T)-K]
=a+bK,
\]

so \(DF=-b\) and \(F=a/DF\). The selected OTM settlement is converted to a
call-equivalent price and normalized as

\[
k=\frac{K}{F},\qquad
c=\frac{C_{\mathrm{equiv}}}{DF\,F}.
\]

The normalized calibration then uses spot \(1\), rates \(0\), and strike \(k\).
This removes the date-specific forward and discount level without pretending
that contract strikes or expiries align across dates. Market IVs are inverted
directly at observed nodes. There is no interpolation, extrapolation, or surface
smoothing.

### Admission, liquidity, and coverage gates

A date is calibrated only when all of the following hold:

- maturities are between 7 and 365 calendar days under ACT/365;
- at least five expiries and at least 80 usable OTM nodes remain;
- the selected OTM side has positive daily volume and positive open interest;
- each admitted expiry retains at least one liquid OTM put and one liquid OTM
  call;
- the full-wing parity pillar has an absolute annualized implied rate no larger
  than 10% and parity RMSE no larger than 1% of forward;
- each admitted expiry has valid finite IV inversion;
- each expiry receives total objective weight one, preventing a denser strike
  ladder from dominating solely through node count.

The artifact records the minimum and maximum observed log-moneyness by expiry.
The two-sided rule guarantees presence, not balanced depth or a universal wing
cutoff. Node and expiry counts are necessary coverage gates but not proof of
balanced wing liquidity; the stored call/put counts and moneyness ranges must
still be inspected.

Full-wing OLS remains the frozen normalization. A separate diagnostic refits up
to nine strikes nearest the primary forward and records the forward and implied-
rate differences; it never replaces the primary pillar. This exposes the
short-tenor amplification of small parity-slope changes.

The raw normalized call-equivalent rows are also checked for non-increasing call
prices and convexity using strike-spacing-aware adjacent slopes. Violation counts
and details are persisted and aggregated, but no node is smoothed, repaired, or
rejected by this diagnostic. Consequently, the cross-date panel must not be
described as a static-arbitrage-clean surface study.

Expiry is the third Friday rolled to the next trading day. The frozen cohort
records explicit holiday overrides (currently the June 2026 contract expiry on
2026-06-22) rather than consulting today's calendar and silently changing old
maturities.

The aggregate panel then applies a strict comparability gate: source class and
price field must match, trade dates and source hashes must be unique, the best
fit must have succeeded, and the complete calibration configuration must equal
the explicitly persisted required configuration. Excluded candidates and their
reasons remain in the artifact. Direct aggregation without an explicit expected
configuration uses a deterministic majority fingerprint with a lexical tie-break,
so configuration selection does not depend on input order.

The current candidate dates are an explicitly requested short panel, not a
systematic daily or weekly sample of the full market history. The artifact saves
the requested tags, but the six admitted dates from 2026-04-30 through
2026-07-15 are preliminary market-development evidence. A representative claim
requires a documented sampling rule over a substantially longer regime window.

Cross-date output reports parameter dispersion, fit-error dispersion, Feller
ratios, bound-hit frequencies, and the range of fixed-economic-scale SVD
condition numbers/effective ranks. It is a stability panel, not a formal time
series model and not a pass/fail theorem. Contract rolls, changing liquidity,
settlement conventions, discrete tick effects, and structural regime changes
can all move the result.

All reported cross-date RMSE values are conditional on the frozen economic box.
Active bounds therefore qualify the result: the panel measures fit capacity
inside that box, not unrestricted Heston capacity. A documented bound-stress or
free-parameter comparison is still required before assigning poor fit solely to
the model family.

## Reproducible commands

```bash
# Freeze already-downloaded official CSV files (offline/reproducible path).
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_settlement_history.py \
  --dates 20260430 20260515 20260615 20260630 20260706 20260715 20260720 \
  --input-dir /path/to/cffex-daily-csv

# Current intraday prepared-target fit plus Jacobian/SVD/bootstrap evidence.
.venv/bin/python example/mo_volmodels/04_heston_calibration.py \
  --tag latest --bootstrap-reps 32 --bootstrap-seed 20260721

# Separate official-settlement cross-date panel.
.venv/bin/python example/mo_volmodels/10_calibration_diagnostics.py \
  --tags 20260430 20260515 20260615 20260630 20260706 20260715 20260720 \
  --output-tag latest
```

Principal artifacts are `data/mo_calib_heston_latest.json` and
`data/mo_calibration_diagnostics_latest.{json,csv}`. Read their saved
configuration and exclusions before comparing headline metrics.

## References

- Cui, del Baño Rollin, and Germano,
  [*Full and fast calibration of the Heston stochastic volatility model*](https://eprints.lse.ac.uk/83754/1/Germano_Full%20and%20fast%20calibration_2017.pdf),
  *European Journal of Operational Research* 263 (2017), 625–638.
- [CFFEX CSI 1000 index-option product and contract rules](https://www.cffex.com.cn/zz1000gzqq/).
- [CFFEX historical-data service](https://www.cffex.com.cn/lssjfw/) and
  [historical-data download](https://www.cffex.com.cn/lssjxz/).
