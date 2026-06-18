# SA-CVA — Basel standardised approach for CVA risk (MAR50.27–50.77)

Regulatory reference for `quantark/sacva`. Source: Basel Committee on Banking
Supervision, MAR50 (consolidated framework), paragraphs 50.27–50.77. This file is
the module-local copy of the methodology; the verbatim standard text the module
was built from is in `docs/sacva.md`.

## Nature of SA-CVA

SA-CVA is an adaptation of the FRTB standardised approach for market risk (MAR21)
to CVA risk. It is a **sensitivity-based aggregation (SBA)** method. Differences
from the market-risk SA (MAR50.27): reduced risk-factor granularity; no default
risk and no curvature risk.

The bank supplies, per risk factor `k`, the sensitivity of the aggregate
regulatory CVA (`S_k^CVA`) and of the eligible hedges (`S_k^Hdg`). This module
**consumes** those sensitivities (MAR50.29) — it does not compute CVA or run an
exposure model.

## Aggregation (MAR50.51–50.53)

Weighted sensitivities (MAR50.51–50.52):

```
WS_k^CVA = RW_k · S_k^CVA
WS_k^Hdg = RW_k · S_k^Hdg
WS_k     = WS_k^CVA − WS_k^Hdg
```

Bucket capital, with hedging-disallowance `R = 0.01` (MAR50.53(1)):

```
K_b = sqrt( max(0, Σ WS_k² + Σ_{k≠l} ρ_kl WS_k WS_l) + R · Σ (WS_k^Hdg)² )
```

Risk-class capital, with `S_b = max(−K_b, min(Σ WS_k, K_b))` and `m_CVA = 1`
(MAR50.41, 50.53(2)–(3)):

```
K = m_CVA · sqrt( max(0, Σ K_b² + Σ_{b≠c} γ_bc S_b S_c) )
```

Delta capital = simple sum of `K` over the 6 delta risk classes (MAR50.43); vega
capital = simple sum over the 5 vega risk classes (MAR50.45). **SA-CVA =
delta + vega** (MAR50.42). Counterparty credit spread has **no vega** (MAR50.63).

## Risk classes and factor structure

| Risk class | Vega | Buckets | Factors per bucket |
|---|---|---|---|
| Interest rate | yes | per currency | multi (specified: 1/2/5/10/30y + inflation; other: parallel-yield + inflation; vega: rate-vol + inflation-vol) |
| FX | yes | per non-reporting currency | single |
| Counterparty credit | no | 1–8 (8 = qualified index) | multi (entity × tenor 0.5/1/3/5/10y) |
| Reference credit | yes | 1–17 (16/17 = qualified index) | single |
| Equity | yes | 1–13 (12/13 = qualified index) | single |
| Commodity | yes | 1–11 | single |

## Tables (decimals)

### Interest rate (MAR50.54–50.58)
- Cross-bucket γ = 0.5.
- Specified-ccy = reporting ccy ∪ {USD, EUR, GBP, AUD, CAD, SEK, JPY}.
- Specified delta RW: 1y 0.0111, 2y 0.0093, 5y 0.0074, 10y 0.0074, 30y 0.0074,
  inflation 0.0111.
- Specified tenor ρ (Table 4): 1–2 0.91, 1–5 0.72, 1–10 0.55, 1–30 0.31,
  2–5 0.87, 2–10 0.72, 2–30 0.45, 5–10 0.91, 5–30 0.68, 10–30 0.83; any
  tenor↔inflation 0.40.
- Other-ccy delta RW 0.0158 (yield & inflation); ρ(yield, inflation) 0.40.
- Vega RW 1.00; ρ(rate-vol, inflation-vol) 0.40.

### FX (MAR50.59–50.62)
Cross-bucket γ = 0.6. Delta RW 0.11. Vega RW 1.00.

### Counterparty credit spread (MAR50.63–50.65) — delta only
Cross-bucket γ (Table 6), symmetric, unit diagonal:

```
       1     2     3     4     5     6     7     8
1    1.00  0.10  0.20  0.25  0.20  0.15  0.00  0.45
2          1.00  0.05  0.15  0.20  0.05  0.00  0.45
3                1.00  0.20  0.25  0.05  0.00  0.45
4                      1.00  0.25  0.05  0.00  0.45
5                            1.00  0.05  0.00  0.45
6                                  1.00  0.00  0.45
7                                        1.00  0.00
8                                              1.00
```

Delta RW (Table 7): bucket 1a IG 0.005 / HY-NR 0.020; 1b 0.010 / 0.040;
2 0.050 / 0.120; 3 0.030 / 0.070; 4 0.030 / 0.085; 5 0.020 / 0.055;
6 0.015 / 0.050; 7 0.050 / 0.120; 8 0.015 / 0.050.

Intra-bucket ρ = ρ_tenor · ρ_name · ρ_quality (MAR50.65(4)–(5)):
- ρ_tenor: 1.00 same tenor, else 0.90.
- ρ_name (buckets 1–7): 1.00 same name, 0.90 distinct but legally related
  (non-null equal legal-entity group), 0.50 otherwise. (bucket 8): 1.00 same
  index & series, 0.90 same index distinct series, 0.80 otherwise.
- ρ_quality: 1.00 same quality (IG/IG or HY-NR/HY-NR), else 0.80.

### Reference credit spread (MAR50.66–50.69) — delta + vega
Buckets 1–7 IG sectors, 8–14 HY/NR sectors, 15 other, 16 IG index, 17 HY index.
Single delta/vega factor per bucket.

Sector cross-bucket γ (Table 9), sector s∈1..7:

```
     s1    s2    s3    s4    s5    s6    s7
s1  1.00  0.75  0.10  0.20  0.25  0.20  0.15
s2  0.75  1.00  0.05  0.15  0.20  0.15  0.10
s3  0.10  0.05  1.00  0.05  0.15  0.20  0.05
s4  0.20  0.15  0.05  1.00  0.20  0.25  0.05
s5  0.25  0.20  0.15  0.20  1.00  0.25  0.05
s6  0.20  0.15  0.20  0.25  0.25  1.00  0.05
s7  0.15  0.10  0.05  0.05  0.05  0.05  1.00
```
Composite γ(b,c): buckets 1–14 use the sector value; if the two buckets differ in
credit quality (IG vs HY/NR) the value is **halved** (MAR50.67(2)). Bucket 15
correlates 0 with everything else; γ(16,17)=0.75; γ({16,17}, 1–14)=0.45;
γ(15, 16/17)=0.

Delta RW (Table 10) by bucket 1–17: 0.005, 0.010, 0.050, 0.030, 0.030, 0.020,
0.015, 0.020, 0.040, 0.120, 0.070, 0.085, 0.055, 0.050, 0.120, 0.015, 0.050.
Vega RW 1.00.

### Equity (MAR50.70–50.73) — delta + vega
Single factor per bucket. Cross-bucket γ: 0.15 between any two distinct buckets
1–10; 0.75 between 12 and 13; 0.45 between {12,13} and 1–10; 0.00 for any pair
involving bucket 11.

Delta RW (Table 12) by bucket 1–13: 0.55, 0.60, 0.45, 0.55, 0.30, 0.35, 0.40,
0.50, 0.70, 0.50, 0.70, 0.15, 0.25. Vega RW 0.78 for large-cap buckets
{1–8, 12}, else 1.00.

### Commodity (MAR50.74–50.77) — delta + vega
Single factor per bucket. Cross-bucket γ: 0.20 between any two distinct buckets
1–10; 0.00 for any pair involving bucket 11.

Delta RW (Table 14) by bucket 1–11: 0.30, 0.35, 0.60, 0.80, 0.40, 0.45, 0.20,
0.35, 0.25, 0.35, 0.50. Vega RW 1.00.

## Numerical convention

The `max(0, ·)` floor on the bucket and cross-bucket radicands is the prescribed
SBA numerical treatment: the supervisory correlation matrices (ρ, γ) are not
guaranteed positive semi-definite, so the radicand can be negative and is floored
at zero. This is the regulator's own SBA semantics, not an approximation.

## Computing CVA and sensitivities from a portfolio (MAR50.32–50.35)

The SBA engine above consumes `CVASensitivity` records. Those can be supplied
directly, or **computed from a real trade portfolio** by the integration layer
(`quantark.sacva.SACVAEngine`), which wires the existing pricing engines "like SIMM
does":

1. **Exposure (MAR50.32–50.35).** `MonteCarloExposureEngine` simulates risk-neutral
   GBM spot paths (one constant-vol factor per underlying; FX drift `r_dom − r_for`)
   and reprices each trade on a deterministic value surface at every exposure node —
   the analytic surface re-evaluates the trade's own engine at the rolled-down
   `(spot, τ)` (no nested MC, no LSMC). Pathwise values are aggregated to a
   discounted EPE profile with netting **within** enforceable sets and summed
   **across** sets (MAR50.35). Risk-neutral drift is mandatory (MAR50.34(1)); the
   profile is tagged `RISK_NEUTRAL` / `regulatory_eligible`. A real-world / PFE
   `HistoricalExposureEngine` (non-eligible) is a separate, parallel backend.

2. **Unilateral CVA (MAR50.32).** `RegulatoryCVAEngine` integrates
   `CVA = ELGD · Σ_i ½(EE*_{i−1}+EE*_i)·(S(t_{i−1}) − S(t_i))`, where `EE*` is the
   discounted expected positive exposure and `S` the counterparty survival
   probability (`ELGD = 1 − R`).

3. **Sensitivities.**
   - *Counterparty credit-spread delta* (MAR50.63, per entity × tenor): one-sided 1bp
     key-rate hazard bump (`Δλ = 1bp / ELGD`, chain rule `s = λ(1−R)`) re-running
     step 2 only — the exposure is invariant to the counterparty hazard, so no MC
     re-run is needed. Divisor `1e-4`.
   - *Equity / FX spot delta + vega* (MAR50.59, MAR50.70, single factor per equity
     bucket / foreign currency): a +1% relative bump to the factor's spot /
     volatility **moves the exposure**, so each is a portfolio-wide re-run of the MC
     exposure (with common random numbers) for every counterparty exposed to that
     factor, summing ΔCVA. Divisor `1e-2`. FX spot is a GBM factor (rate=domestic,
     div=foreign); a trade declares one market factor — `equity_bucket` XOR
     `fx_currency` — and `fx_currency` may not equal the reporting currency.
   - *Eligible-hedge market value* (`S_k^Hdg`, MAR50.29): each `CVAHedge` is priced
     and bumped on the SAME factor; the sensitivity is emitted with `s_cva=0` so the
     SBA risk-factor netting forms `WS = RW·(s_cva − s_hdg)` and the hedge
     disallowance `R·Σ(WS^Hdg)²`.

`SACVAEngine.compute(portfolio)` runs 1→3 and feeds the resulting `CVASensitivity`
records to the unchanged SBA calculator. v1 covers equity and reporting-vs-foreign-FX
spot under deterministic rates, single reporting currency, uncollateralized; it emits
counterparty credit-spread delta, equity/FX spot delta+vega, and eligible-hedge MV
sensitivities (every trade must declare its market factor for the market legs —
all-or-none). IR market sensitivities need a key-rate-bumpable term curve and are a
scoped extension. See `example/sacva_portfolio_demo.py`.

### Stateful (autocallable) exposure (MAR50.32, snowball)

Path-dependent trades (`SnowballOption` priced by `SnowballQuadEngine`) are valued
without re-pricing per node. The QUAD engine runs a two-regime backward recursion —
`v_out` (not yet knocked in) and `v_in` (knock-in has occurred) — on one
inception-anchored, full-maturity-width spot grid; with the opt-in
`record_backward_grids` flag it now exposes those per-observation continuation
surfaces. `build_snowball_surface` reads them into a per-`(t, state)`
`GridValueSurface`, and `MonteCarloExposureEngine` resolves each path's knock-in
history and knock-out termination with `BarrierStateMachine` (Brownian-bridge
continuous KI), selecting `v_in`/`v_out` for live paths and zeroing knocked-out paths
(immediate settlement). The wide QUAD grid covers the simulated spot cloud, so no
extrapolation is needed. Correctness is pinned by a value-process martingale:
`E[ df·V_alive(t) + redemption·df at KO ] = price₀` at every node.

v1 stateful scope (raise, never approximate): a single `SnowballOption` per
counterparty (vanillas may net on its grid), plain `SnowballQuadEngine` only (Phoenix /
KO-reset carry richer state), constant KO/KI barriers, immediate KO settlement (delayed
settlement needs the `pending_receivable_exposure` machinery), `disable_ko_after_ki` and
BGK knock-in monitoring deferred. A counterparty's trades must share one reporting
discount curve and agree on per-underlying market data (one GBM factor per underlying),
else the engine raises rather than producing an order-dependent result. **IR** market
sensitivities remain a scoped extension (they need a key-rate-bumpable term curve) that
**raises** rather than silently approximating; FX spot delta+vega is supported.
