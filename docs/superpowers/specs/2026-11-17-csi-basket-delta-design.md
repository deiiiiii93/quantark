# CSI basket delta: whole-index vs per-name — design

Date: 2026-11-17 · Status: approved; amended by design review (the run spec's rulings D12–D22 are folded in below) · Location of the study:
`example/case_studies/csi_basket_delta/`

## 1. Intent

The desk is starting a **customized strategy index (CSI)** business: a CSI is
a weighted basket of shares, ETFs or indices, and the desk sells OTC options
on it. The open question is how to manage the delta:

1. **Whole-index:** treat the CSI as one underlying, compute one delta Δ_I,
   and trade it as a single basket ("combination") order.
2. **Per-name:** compute a delta per constituent and hedge each name, where
   the per-name delta comes from either
   (a) a **linear split** of Δ_I by the weights, or
   (b) **finite difference** (FD) on each constituent, one at a time.

This case study compares **the deltas themselves**, as a static pricing
comparison across market scenarios. It starts with a European vanilla. The
result should tell the desk when the methods agree, when they don't, by how
much (in CNY), and why.

### Framing facts the study rests on

- A whole-index order of Δ_I index units buys Δ_I·nᵢ shares of name i, where
  nᵢ is today's (drifted) share count per index unit. So method 1 is the same
  trade as method 2(a) **with current share counts**. The real comparison is
  current-share split vs target-weight split vs a per-name delta from a model
  that sees the names.
- Chain rule: Δ_I is the derivative of V along a proportional bump of all
  names, so Σᵢ Sᵢ·∂V/∂Sᵢ = I·Δ_I **exactly**. Every correct method carries
  the same *total* delta notional. A per-name model only redistributes it
  between names, so the residual risk is to names *diverging*, not to the
  market moving.

## 2. Goals, non-goals, anti-goals

**Goals**

- One pricer covering three index rules through a rebalance-frequency
  setting: fixed shares (no rebalance in the option's life), periodic
  target-weight rebalance, and daily rebalance.
- An exact MC reference delta per name, plus a fast moment-matched
  lognormal model. Score every delta method against the reference, and report
  the moment-matched model's own error separately.
- Scenario sweeps that show what drives the gap: moneyness, maturity,
  rebalance frequency, position in the rebalance period, drift since the
  last rebalance, vol dispersion, correlation level, and dividend-yield
  dispersion.
- A Bloomberg × Swiss HTML report (`docs/UI_LANGUAGE.md`) with a README.

**Non-goals**

- Real market data (the basket is stylised and synthetic).
- Hedge P&L backtests, trading costs, lot sizes.
- Futures basis for index-type constituents.
- Exotic payoffs (snowballs on a CSI come later).
- Smile / local vol / stochastic vol: all names are GBM.
- Total-return or net-total-return index variants: **price return only**.
- Promoting the pricer into `quantark` (possible later, not in this study).
- Decrement studies: d = 0 in every sweep. `01_sanity.py` runs one d = 5%
  variant only to check the moments and the Euler identity.

**Anti-goals** (failure even if it runs)

- Declaring a gap "real" when it is inside MC noise. Every gap is reported
  with its standard error, and a gap counts as nonzero only above 3 SE.
- Using the moment-matched model as the truth. MC is the reference.
- Inventing a fallback when the maths can't be done right. Leave a `# TODO`
  and stop (standing repo rule).

## 3. Model

### 3.1 Index definition

- Names i = 1..N follow GBM under Q with drift r − qᵢ, vol σᵢ and
  correlation matrix ρ. The rate r is flat.
- Rebalance dates t₁ < t₂ < … < t_M fall inside (now, T]. The rebalance
  frequency is a parameter: `none`, quarterly, monthly, weekly or daily, on a
  244-trading-day year.
- Target weights wᵢ sum to 1. The decrement is d p.a. (default 0).
- Between rebalances: I(t) = e^{−d(t−t_k)}·Σᵢ nᵢ(k)·Sᵢ(t).
- At rebalance t_k the share counts reset to nᵢ(k) = wᵢ·I(t_k)/Sᵢ(t_k).
- So each full period's factor is
  R_k = e^{−dΔ_k}·Σᵢ wᵢ·Sᵢ(t_{k+1})/Sᵢ(t_k). These factors are independent
  across periods.
- **Current state:** we are part-way through a period. The state is the
  last-rebalance level I(t₀⁻), the name moves since then
  mᵢ = Sᵢ(now)/Sᵢ(t₀⁻), and the elapsed fraction of the period. That gives
  today's share counts nᵢ = wᵢ·I(t₀⁻)/Sᵢ(t₀⁻) and today's level
  I(now) = Σ nᵢSᵢ(now). Current weights are nᵢSᵢ(now)/I(now), which are not
  equal to wᵢ.
- **Decrement accrued since t₀ is folded into nᵢ:**
  nᵢ = e^{−d·elapsed}·wᵢ·I(t₀⁻)/Sᵢ(t₀⁻). Then I(now) = Σ nᵢSᵢ(now) holds
  exactly, and the stub is I(t₁) = e^{−dτ}·Σ nᵢSᵢ(t₁) with
  **τ = t₁ − now** throughout §3.3–3.4. The pathwise factor uses this same
  e^{−dτ}, so the per-path Euler identity holds. *(Builder's call.)*
- Fixed shares (rule A) is the case where no rebalance date falls inside the
  option's life.

### 3.2 Product

European call and put on I(T) with strike K. **Every scenario is struck
today:** K is set as a fraction of I(now), and units = 1e8 / I(now) (1e8 CNY
of index notional today). This holds moneyness fixed across scenarios, so the
drift sweep (§5 sweep 3) changes only the current weights, never the
moneyness. Moneyness has its own sweep. *(Builder's call.)*

### 3.3 MC reference

- Simulate exactly at the stub end t₁, at every later rebalance date, and at
  T. This is exact GBM sampling, so there is no time-stepping error.
- Use Sobol QMC from `quantark.montecarlo` (the multi-asset Brownian bridge
  is available) with RQMC batches for standard errors.
- The Sobol dimension count is N × steps. Daily for 1Y with 5 names is about
  1,220, well under the 21,201 cap.
- **Per-name reference delta: pathwise**, which is unbiased with no bump
  noise. Only the stub depends on Sᵢ(now):
  - ∂I_T/∂Sᵢ(now) = e^{−dτ}·nᵢ·Gᵢ·I_T/I(t₁), where Gᵢ = Sᵢ(t₁)/Sᵢ(now).
  - Call: Δᵢ = e^{−rT}·E[1{I_T>K}·∂I_T/∂Sᵢ]. Put: use −1{I_T<K}.
  - Per path, Σᵢ Sᵢ·∂I_T/∂Sᵢ = I_T exactly. This is the Euler identity at
    floating-point precision, and it is a test.
- **Whole-index delta Δ_I** from the same paths:
  e^{−rT}·E[1{I_T>K}·I_T/I(now)].

### 3.4 Moment-matched model

- Closed-form moments as products across periods:
  - Stub: E[I(t₁)] = e^{−dτ}Σ nᵢSᵢe^{(r−qᵢ)τ} and
    E[I(t₁)²] = e^{−2dτ}Σᵢⱼ nᵢnⱼSᵢSⱼe^{(2r−qᵢ−qⱼ)τ}e^{ρᵢⱼσᵢσⱼτ}.
  - Full periods: the same formulas with wᵢ in place of nᵢSᵢ/I.
- Match a lognormal: F = M₁ and σ²_eff·T = ln(M₂/M₁²). Price with Black on
  F.
- Δ_I and the per-name deltas must be noise-free. Use an analytic chain
  rule, with a cross-check against FD on the closed form.

## 4. Delta methods scored

All methods are scored against the MC pathwise reference Δᵢ^ref.

| # | Method | Model |
|---|--------|-------|
| M1 | Δ_I × today's share counts nᵢ (the whole-index order) | MC |
| M2 | Δ_I × target weights: wᵢ·I/Sᵢ shares (the shortcut) | MC |
| M3 | per-name central FD, bump 1% of Sᵢ, common random numbers | MC |
| M4 | M1 on the moment-matched model | MM |
| M5 | per-name derivative on the moment-matched model | MM |

**Metrics, per name and method:**

- Delta error in CNY per 1% move of name i: (Δᵢ^m − Δᵢ^ref)·Sᵢ·units·1%.
- The same error relative to the reference delta cash.
- Aggregate **residual daily P&L σ**: √(eᵀΣ_daily e), where
  eᵢ = (Δᵢ^m − Δᵢ^ref)·Sᵢ·units and Σ_daily is the daily return covariance.
  This is the noise a desk carries by using method m instead of the
  reference. It replaces the "names diverge by 1σ" scenario from the
  conversation with a single number that already weighs every divergence
  direction. *(Builder's call; noted for review.)*
- Every number carries its MC SE.
- **Hedge-delta invariant:** every delta holds K and units at the unbumped
  I(now). An FD bumps only one name's price, with the strike passed
  explicitly.
- The residual σ is bias-corrected: Q = ēᵀΣē − tr(Σ·Cov(ē)), with a
  delta-method SE. It counts as resolved only when Q > 3 SE.

Also report the price, Δ_I and total delta cash for each model, so the
moment-matching error is visible.

**PV is a comparison target in its own right** (added at the user's request
during the build).
- The report puts the fast model's PV error, (MM − MC) in % and in CNY on
  1e8 notional, next to the delta findings.
- It plots the error against moneyness, tenor, correlation and vol dispersion.
- A PV gap counts as resolved only above 3 MC SE.

## 5. Scenarios

**Base basket** (stylised; weights sum to 1):

| name | type | weight | σ | q |
|------|------|--------|---|---|
| LC1 | large-cap stock | 0.25 | 25% | 2.5% |
| LC2 | large-cap stock | 0.20 | 25% | 3.0% |
| TECH | high-beta tech stock | 0.15 | 45% | 0.3% |
| ETF | broad index ETF | 0.30 | 18% | 1.5% |
| GOLD | gold / bond ETF | 0.10 | 10% | 0% |

- Correlations: 0.6 among the equities and the ETF (0.85 between the ETF and
  the large caps), and 0.05 between GOLD and everything else.
- r = 2%.
- The exact matrix is the builder's to finalise. It must be positive
  definite, and that gets checked.

**Control basket (exchangeable):** **equal weights** (0.20 each), all
σ = 25%, all q = 1.5%, uniform ρ = 0.5, at a rebalance date. Every name is
interchangeable, so M1 = M2 = M3 = reference within SE, and M5 = M4 exactly by
symmetry. The moment-matched Δ_I is **not** asserted equal to the MC Δ_I,
because a 5-name basket is not lognormal even when exchangeable. That gap is
reported as a finding.

**Concentration basket:** the same as the control but with the base
(unequal) weights. This is **not** a control. Under the Sᵢ-numeraire measure,
the index log-drift shifts by about ρσ²T + (1−ρ)σ²T·cᵢ, where cᵢ is the
current weight. So a heavier name carries a larger delta per share even with
identical vols, and the M1 gap is nonzero (ETF vs GOLD drift spread ≈ 0.6% at
1Y ATM). The report shows this as a cause of its own, separate from vol
dispersion: **concentration**. *(Correction found in review.)*

**Sweeps**

1. Moneyness: K / I(now) from 80% to 120%, call and put. Maturities 1M,
   3M, 6M, 1Y. Rule A.
2. Rebalance frequency: none / quarterly / monthly / weekly / daily, at 1Y
   ATM. Run at the period start, mid-period and just before a rebalance.
3. Drift since last rebalance, under the **quarterly** rule: TECH +30% (and
   −30%), others flat, mid-period. This is where M2 should break.
4. Vol dispersion, scaled in log space so vols stay positive:
   σᵢ(λ) = σ̄_g·exp(λ·(ln σᵢ − mean ln σ)), λ ∈ {0, 0.5, 1, 1.5, 2}.
5. Correlation level, including the extremes:
   - Uniform ρ on all 5 names: {−0.25 (the PSD floor −1/(N−1)), 0, 0.2,
     0.4, 0.6, 0.8, 0.9, 1}.
   - A 2-name basket (EQ σ = 30%, w = 0.6; HEDGE σ = 15%, w = 0.4) over
     ρ ∈ {−1, −0.5, 0, 0.5, 1}, because ρ = −1 is only feasible pairwise.
   - Singular (PSD) matrices are factored by an eigen square root.
   - At each interior point, a **correlation model-risk band**: how far REF
     moves under ρ ± 0.1, shown next to the method gaps.
6. Dividend-yield dispersion: show that even the forward part of the delta
   splits differently when the qᵢ differ (M1 ≠ reference for a deep-ITM call).

## 6. Deliverables and layout

```
example/case_studies/csi_basket_delta/
  README.md                    question, setup, findings, how to run
  _common.py                   basket / state dataclasses, MC + MM pricers, delta methods
  01_sanity.py                 identities, single-name → BS, control basket, moment check
  02_fixed_shares.py           sweep 1 (+ dividend dispersion, sweep 6)
  03_rebalance.py              sweeps 2 and 3
  04_dispersion_correlation.py sweeps 4 and 5
  05_report.py                 HTML report from results/*.json
  results/                     JSON per script + csi_basket_delta_report.html
test/test_csi_basket_delta.py  fast pins on _common.py
```

- The report starts from the CSS tokens and SVG helpers in
  `example/case_studies/snowball_ftp_study/05_report.py`. It needs a frozen
  navigator and a light/dark switcher.
- The study follows repo conventions: `quantark.util.numerical` for
  comparisons and safe maths, `ValidationError` for bad inputs, dataclasses
  and type hints, and vectorised NumPy.
- The test loads `_common.py` by file path, like
  `test/test_snowball_ftp_study.py`.

## 7. Acceptance checks (the test file)

1. **Single-name basket:** the MM price and delta equal Black–Scholes to
   floating-point tolerance. MC is within 3 SE.
2. **Euler identity:** per path, Σᵢ Sᵢ·∂I_T/∂Sᵢ = I_T to fp precision. In
   expectation, Σ Sᵢ·Δᵢ^ref = I·Δ_I.
3. **Exchangeable control** (equal weights, vols, q and ρ): M1, M2 and M3
   match the reference within 3 SE (plus the FD bias allowance for M3). M5 = M4
   by symmetry.
   **Concentration basket:** the sign of the M1 gap is right (the heavier
   name has the larger reference delta per share).
4. **Moments:** MC sample means of I_T and I_T² match the closed form within
   3 SE, for rules A, B and C.
5. **Daily rebalance:** the rule-C M1 gap is smaller than the rule-A gap for
   the same basket and strike (the gap shrinks as rebalancing gets more
   frequent). The MM-vs-MC price difference is **reported, not asserted to be
   zero**: the index is only approximately lognormal, so that gap is a
   finding.
6. **Pathwise vs FD:** at the base scenario, M3 matches the reference within
   3 SE plus the O(h²) bump bias.

Keep the tests fast: small path counts and coarse grids. Tests are pinned
at **fixed** small path counts and seeds. Only exact identities (checks 1
MM, 2, 4) and true symmetries (check 3 control) are asserted as "within
3 SE". Effects that are real but small are asserted by sign or by ordering,
never as "equals zero within SE", so the tests don't start failing when the
study runs at higher path counts. *(Builder's call.)*

## 8. Approaches considered

1. **MC only.** Exact for any rule, but FD deltas are noisy and it gives no
   fast model to judge.
2. **Moment-matched lognormal only.** Fast and smooth, but it can't tell a
   real gap from approximation error.
3. **Both, with MC as the reference and moment-matched as the desk-style
   model. Chosen.** It follows the repo rule (exact by default, approximations
   opt-in and measured) and also answers whether the fast model is good
   enough for daily hedging.

The user also chose **one pricer for all three index rules** (fixed shares,
periodic rebalance, daily rebalance) via a rebalance-frequency setting, with
an optional decrement defaulting to zero.

## 9. Left to the builder

- RQMC batch count and path count: whatever makes the reported gaps sit
  well above 3 SE, or proves them zero.
- The exact correlation matrix (positive definite), and the dates and
  day-count details on the 244-day year.
- Chart selection inside the report.
- Whether the MM per-name derivative is a hand-written chain rule or
  autodiff-style. It must be noise-free and cross-checked by FD on the closed
  form.

## 10. Repo notes

- Most `example/case_studies/*` directories are untracked locally. Tracking
  this one is the user's call at landing time.
- New files under `docs/` need `git add -f` (public-repo ignore rules).
- No desk data: the basket is synthetic.
