# How should a desk split the delta of an option on a custom strategy index?

The desk is starting a customized strategy index (CSI) business. A CSI is a
weighted basket of shares, ETFs or indices, and the desk sells OTC options on
it. To hedge, it has two choices:

1. **Whole-index.** Compute one index delta Δ_I and trade it as a single basket
   ("combination") order. That buys Δ_I·nᵢ shares of each name, where nᵢ is the
   index's current share count per unit. It is the same trade as splitting Δ_I
   across the names by their **current** weights.
2. **Per-name.** Compute a delta for each name and hedge each one separately,
   either by splitting Δ_I by weight or by bumping each name and repricing.

This study starts with a European vanilla on the CSI. It asks when the two
choices give different per-name hedges, by how much in CNY, and why.

**The short answer**

- Every correct method carries the **same total delta**, because
  Σ Sᵢ·∂V/∂Sᵢ = I·Δ_I exactly. Methods differ only in how that total is
  **split between names**. So the risk left over is to the names *diverging*,
  not to the market moving.
- Bumping names one at a time only gives a different answer if the pricer
  models each name: its own vol and its correlation with the others. On an
  index-only pricer with one index vol, bumping a name reproduces the weight
  split exactly. Here that's method M4.
- On a **fixed-share** CSI (no rebalance during the option's life), the weight
  split puts too little delta on volatile names and too much on calm ones. For
  a 1Y at-the-money call on 1e8 CNY of the stylised 5-name basket:
  - TECH (45% vol) is under-hedged by **10.9k CNY per 1% move**, which is
    **−12.1%** of its true delta.
  - GOLD (10% vol) is over-hedged by **6.8k CNY per 1%**, which is **+14.7%**.
  - The resulting daily P&L noise is **28.8k CNY** (1σ).
  - Per-name FD on a multi-name Monte Carlo removes it: its residual σ is
    not resolved (< 132 CNY/day at 3 SE).
- **Rebalancing turns the gap off.** A bump only lives until the next
  rebalance; after that the weights reset to target. So the gap scales with
  the time left in the current period. Measured as daily residual σ at the
  start of a period: no rebalance 28.8k, quarterly 7.2k, monthly 2.4k,
  weekly 0.6k, daily 0.1k CNY. For weekly or daily CSIs the whole-index order
  is fine.
- **Never split by *target* weights mid-period.** After TECH moves ±30% since
  the last rebalance, the target-weight shortcut leaves **47–49k CNY/day** of
  noise. The current-share split leaves 2.6–4.5k.
- **Correlation uncertainty is not the bottleneck.** Shifting the correlation
  by ±0.1 moves the reference per-name delta by at most about 1k CNY per 1%.
  The weight-split error it is compared against is 10–13k. Across uniform
  correlations from −0.25 (the lowest possible for 5 names) to +1, the
  weight-split residual σ stays between 25k and 31k CNY/day. In a 2-name
  basket the sign of ρ matters. At ρ = −1 the per-name split error is
  about 3.5× its ρ = +1 value: 40k vs 11k CNY per 1%. The residual σ is
  about 10× larger: 115k vs 11k CNY/day.
- **The fast single-index model misprices the smile, not just the level.**
  Matching an index lognormal to the basket's first two moments gives the
  wrong PV.
  - It overprices at-the-money and low strikes: a 1Y ATM call by **+0.75%
    (61k CNY per 1e8)**, a 1Y 80% put by **+10.3% (115k CNY)**.
  - It underprices high-strike calls: a 1Y 120% call by **−2.8% (−67k CNY)**,
    a 3M 120% call by −8.5%.
  - The reason: a sum of names with different vols is more right-skewed than
    any single lognormal.
  - Negative correlation and vol dispersion make it worse: **+5.9% (298k
    CNY)** for a 2-name stock/hedge basket at ρ = −1, and +9.0% at double the
    vol spread.
  - **Quote PV off the multi-name model**, especially for baskets that mix
    equities with gold, bonds or other hedge-like assets.
- **The fast moment-matched model is a middle ground for deltas.** Its per-name delta
  (M5) cuts the 1Y ATM residual σ from 28.8k to 11.2k CNY/day. It degrades in
  short-dated wings (TECH −17% at 1M, K/I = 120%) and at negative correlation
  (2-name ρ = −1: +14% and +23%).

Every figure here is computed by the scripts below, stored in `results/*.json`
and rendered by `05_report.py`. The figures in this README were copied from
the JSON at commit time.

## Setup

**The index.**
- Between rebalances it holds fixed share counts nᵢ. At each rebalance date the
  counts reset to wᵢ·I/Sᵢ, where wᵢ are the target weights.
- An optional decrement d p.a. accrues on the level (0 in every sweep).
- The rebalance rule ranges from `none` (fixed shares) through quarterly,
  monthly and weekly to daily (close to a constant mix).
- Times are on a 244-trading-day year.

**The model.**
- Each name follows its own GBM with its own vol, dividend yield and full
  correlation matrix.
- Singular matrices (ρ = 1) are factored by an eigen square root.
- The study basket is stylised and synthetic: no desk data.

| name | weight | vol | yield |
|---|---|---|---|
| LC1 large cap | 0.25 | 25% | 2.5% |
| LC2 large cap | 0.20 | 25% | 3.0% |
| TECH high beta | 0.15 | 45% | 0.3% |
| ETF broad index | 0.30 | 18% | 1.5% |
| GOLD gold/bond ETF | 0.10 | 10% | 0% |

Correlations are 0.6 among the equities, 0.85 between the ETF and the large
caps, and 0.05 for GOLD. r = 2%. Options are struck today on 1e8 CNY of index
notional.

**Methods**

| key | per-name delta | what it stands for |
|---|---|---|
| REF | MC pathwise, exact sampling at rebalance dates | the truth for this model |
| M1 | Δ_I × current shares | the whole-index order |
| M2 | Δ_I × target weights | a common shortcut |
| M3 | bump each name ±1%, reprice on the multi-name MC (common random numbers) | per-name FD |
| M4 | M1 on the moment-matched lognormal | index-only pricer with one index vol (bumping names gives this) |
| M5 | per-name derivative of the moment-matched lognormal | cheap per-name model |

**Metrics**
- **Gap:** method − REF, in CNY per 1% move of each name, with a paired SE
  across 32 RQMC batches of 2¹⁴ Sobol paths.
- **Resolved:** a gap counts as resolved only if it exceeds 3 SE. For M3 the
  bar also includes the O(h²) bump-bias allowance.
- **Residual σ:** the daily P&L standard deviation of the hedge mismatch,
  √(eᵀΣe), bias-corrected for MC noise.

## Findings in detail

The full tables and charts are in `results/csi_basket_delta_report.html`.

1. **Sanity** (`01_sanity.py`).
   - A one-name basket matches Black–Scholes. The moment-matched model agrees
     to 2e-14, and the MC is within 0.8 SE, for calls and puts, with and
     without a 5% decrement and a mid-period stub.
   - The Euler identity holds on every path to 6e-16.
   - MC moments match the closed form for every rebalance rule, ρ = 1, the
     5-name floor ρ = −0.25, and the 2-name ρ = −1. One check (quarterly,
     z = −3.7) fell outside 3 SE. It was replicated with four independent
     scrambles at |z| < 0.9, so it was a chance draw; both are kept in the
     JSON.
   - In the exchangeable control (equal weights, vols, yields, correlations)
     no method gap is resolved.
2. **Concentration alone opens a gap.** With identical vols but unequal
   weights, the heaviest name (ETF, 30%) is under-hedged by 1.4k CNY per 1%,
   and the lightest (GOLD, 10%) is over-hedged by 0.8k. A heavier name moves
   more with the index, so it carries more delta per share.
3. **Fixed shares** (`02_fixed_shares.py`).
   - The weight-split error grows with maturity and with strike. For TECH it
     goes from −0.1% (1M, K/I = 80%) to −20.8% (1Y, K/I = 120%). For GOLD it
     reaches +30.5%.
   - Puts mirror calls.
   - Dividend-yield dispersion alone splits even a deep in-the-money call
     differently. A forward's delta is nᵢe^{−qᵢT}, not proportional to nᵢ:
     ±1–2% of each name's delta with yields of 0–3%.
4. **Rebalancing** (`03_rebalance.py`). The gap is linear in the time left in
   the current period.
   - Mid-period is half the start-of-period value, and the last 5% is 5% of it.
   - After drift, the target-weight shortcut breaks: TECH +30% gives a −21%
     TECH error and 49k CNY/day.
5. **Drivers** (`04_dispersion_correlation.py`).
   - Vol dispersion is the main driver. Scaling the log-vol spread from 0× to
     2× takes the residual σ from 8.7k to 174k CNY/day.
   - Correlation level matters little for 5 names, and the ±0.1 model-risk
     band is about 10× smaller than the gap.
6. **PV and Δ_I: fast model vs multi-name MC.**
   - 83 of 96 scenario PVs differ by more than 3 SE. The fast model is high in
     61 of them and low in 22, and every low case is a high-strike option.
   - **Relative errors are largest on near-worthless short-dated wings.**
     The 1M 80% put is +168%, but only 60 CNY. The 1M 120% call is −19.6%.
     The 3M 80% put is +19.1%.
   - **In CNY the error peaks where vols disperse or correlation turns
     negative:** up to 942k CNY per 1e8 at double the vol spread.
   - **Its Δ_I is off by up to 17% on at-the-money options** (2-name,
     ρ = −1). On short-dated wings, where Δ_I is close to zero, it is off by
     up to 62% (1M 80% put).
   - Use it for speed and as a control variate, not as the reference for PV or
     deltas.

**Multiplicity caveat.**
- The study runs over 2,100 per-name gap tests. At a 3-SE bar, a few
  "resolved" labels are expected by chance.
- Some resolved gaps are also immaterial. Example: a 2 CNY gap on a deep
  in-the-money call is statistically real but meaningless for hedging.
- Read the CNY size, not just the flag.

**One-factor caveat.** With a single name, or with ρ = ±1, one shock drives
the whole basket, so the Sobol estimate is effectively one-dimensional. The
payoff kink then falls in a single stratum, and the batch SE of a delta
becomes unreliable: in tests it swung 15× across seeds. No conclusion here
depends on it. Those scenarios show either large, clearly resolved gaps or
"not resolved" FD gaps. The single-name tests allow the one stratum the kink
falls in, df·(K/S)/N.

## How to run

```bash
python example/case_studies/csi_basket_delta/01_sanity.py
python example/case_studies/csi_basket_delta/02_fixed_shares.py
python example/case_studies/csi_basket_delta/03_rebalance.py
python example/case_studies/csi_basket_delta/04_dispersion_correlation.py
python example/case_studies/csi_basket_delta/05_report.py
python -m pytest test/test_csi_basket_delta.py
```

Each stage takes seconds to a minute on a laptop. Results are deterministic
for a fixed seed.

## Files

| file | role |
|---|---|
| `_common.py` | basket, state and option contracts; moment-matched pricer; RQMC reference; delta methods; metrics; `evaluate_scenario` |
| `01_sanity.py` | Black–Scholes reductions, Euler identity, moments, control and concentration |
| `02_fixed_shares.py` | moneyness × maturity × call/put; dividend-yield dispersion |
| `03_rebalance.py` | rebalance frequency × position in period; drift since rebalance |
| `04_dispersion_correlation.py` | vol dispersion; correlation from −1 to 1; correlation model-risk band |
| `05_report.py` | Bloomberg × Swiss HTML report from the JSON |
| `results/` | one JSON per stage, plus `csi_basket_delta_report.html` |

Design: `docs/superpowers/specs/2026-11-17-csi-basket-delta-design.md`.
Tests: `test/test_csi_basket_delta.py`.

**Limits:**
- GBM names, so no smile and no jumps.
- Price-return index.
- One stylised basket.
- No trading costs, lot sizes or hedge backtests.

Hedging P&L over time with real A-share lots is the natural next study.
