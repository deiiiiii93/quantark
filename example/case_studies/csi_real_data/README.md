# Is a single-underlying model good enough for a CSI? A real-data test

The desk sells OTC options on customized strategy indices (CSIs): weighted baskets of A-shares and
ETFs. The cheapest way to run that book is to treat the CSI as **one underlying**:

- price it with **Black** on the index, using the vol of the index's own price history;
- hedge with **one basket order**, splitting the index delta Δ_I across the names by their current
  weights.

The sibling study [`csi_basket_delta`](../csi_basket_delta/) explains on a synthetic basket why the
CSI's **rebalancing rule** decides whether that is good enough. This study tests the same question on
**real prices**, using a **reference** model:

- a per-name Monte Carlo with each name's own historical vol and the historical correlation;
- per-name bump-and-reprice deltas;
- the real rebalance calendar.

It then lets the market be the judge: both models hedge the option daily along the real 2024–2026
path.

**Basket.** A public test case, not a real CSI:

| name | code | weight |
|---|---|---|
| Kweichow Moutai | 600519 | 25% |
| China Merchants Bank | 600036 | 20% |
| iFlytek (high-beta tech) | 002230 | 15% |
| CSI 300 ETF | 510300 | 30% |
| Gold ETF | 518880 | 10% |

**Options.** 1Y calls on 1e8 CNY of index notional. All figures below are in CNY on that notional.

## The short answer

- **The rule alone changes the product.** On the same prices from 2023-01 to 2026-11, the index ends
  at 1185 with no rebalancing (fixed shares) and at 1228–1239 when it is rebalanced, a 4.6% spread.
  Under fixed shares the weights drift by up to 14 points from target.
- **Fixed shares: the single-underlying model is not good enough.**
  - Its ATM price is 3.6% off the reference on average, ranging from −2.2% to +10.4%. The premium gap
    at the hedge starts averages +267k CNY.
  - Its per-name hedge under-hedges high-beta iFlytek by 8.1k CNY per 1% move on average.
  - Hedged daily on the real path, its P&L differs from the reference's by **14.4k CNY/day** (σ,
    noise-corrected; 90% CI 12.0–16.4k).
  - Most of that is the **split** across names (14.8k CNY/day), which only a per-name model fixes.
- **Frequent rebalancing: it is close to the reference.** The daily hedge difference shrinks with
  the rebalance frequency: 2.6k (quarterly), 1.1k (monthly), 0.8k (weekly) and **0.34k (daily)**
  CNY/day. The ATM PV gap is about 0.1–0.2%.
  - Under weekly and daily rebalancing what remains is mostly the **total-delta** part. It comes from
    the index vol and Black's Δ_I, not from the split: 0.76k vs 0.23k (weekly) and 0.32k vs 0.08k
    (daily) CNY/day. A better index-vol estimate addresses it, with no per-name model needed.
  - Under quarterly and monthly the two parts are comparable: their 90% CIs overlap.
- **The synthetic study's closed form holds on real calibrations.** The split error is predicted to
  first order by τ·wᵢ·I·[(βᵢ − 1)·σ_I²·I·Γ], where τ is the time left until the next rebalance date.
  On every rebalanced rule, regressing the measured split part on that formula gives a slope of
  0.99–1.00, with an rms residual of 1–12 CNY per 1%.
- **Every model difference above is small next to the hedging noise itself.** Each arm's daily hedge
  P&L has a σ of about 62k CNY/day under every rule. That is the noise of discrete daily delta
  hedging and daily re-marking, which both models share.

No pass/fail line is drawn. The report puts the figures next to the premium (about 6.9m CNY for the
ATM call), and what counts as good enough is the desk's call.

## What was run

| stage | script | output |
|---|---|---|
| 1. fetch | `01_fetch.py` (akshare, scratch env) | raw prices → cache **outside the repo** |
| 2. indices | `02_build_indices.py` | `results/02_indices.json`: 5 rules, levels, weights, vols |
| 3. Stage A | `03_gap_through_time.py` | `results/03_gap.json`: 35 dates × 5 rules × 3 strikes |
| 4. Stage B | `04_hedge_backtest.py` | `results/04_hedge.json`: 22 starts × 5 rules × 3 arms, daily |
| 5. report | `05_report.py` | `results/csi_real_data_report.html` (reads the JSON only) |

**Data.**
- Daily back-adjusted (`hfq`) closes from akshare 1.20.65, EastMoney (`stock_zh_a_hist_em`,
  `fund_etf_hist_em`). The calendar comes from `calendar_trade_date_hist_sina`.
- The range is 2023-01-03 to 2026-11-17. A bar fetched before 15:30 Beijing time is intraday, so it is
  dropped.
- No suspensions needed filling.
- Back-adjusted prices make the CSI a **total-return** index, so both models use q = 0 and no
  decrement.

**Rules.**
- Rebalancing happens at the close of the first session of each quarter, month or ISO week, or every
  session for the daily rule.
- The index starts at 1000 on 2023-01-03 at the target weights.

**Models.** Both are calibrated on the same trailing 244 sessions: demeaned daily log returns,
ddof = 1, annualised ×244.
- *Test:* Black at the index's own historical vol, with per-name hedge Δ_I·nᵢ.
- *Reference:* the synthetic study's RQMC engine, fed:
  - the per-name vols and correlation;
  - the real remaining rebalance schedule. One year is 244 sessions; sessions after the exchange
    calendar's horizon (2026-12-31) fall back to weekdays and are flagged per record.
  - CRN central bumps, with pathwise deltas kept for the exact Euler decomposition.

**Stage A.**
- Run on the 10th session of each month from 2024-01, at 90%, 100% and 110% strikes, with 8,192 × 32
  paths.
- The per-name gap (test − reference) splits exactly into two parts:
  - a **total-delta** part: wᵢ·I·(Δ_I^test − Δ_I^ref);
  - a **split** part: wᵢ·I·Δ_I^ref − Sᵢ·Δᵢ^ref.

**Stage B.**
- One ATM call is sold on each start date. Its strike and units are fixed at the start.
- Each day both models are re-marked, and three hedges are held:
  - **test:** Black;
  - **reference:** per-name bumps;
  - **attribution:** the reference's own Δ_I split by weights.
- Daily P&L is measured in excess of financing (r = 1.8%) and marked to the reference PV. So the
  difference between two arms is exactly their hedge difference:
  - reference − attribution = the split;
  - attribution − test = the total delta.

**MC noise.** The reference's deltas carry Monte Carlo noise (4,096 × 16 paths per day).
- Its variance is predicted every day from the engine's batch covariance and subtracted from each
  paired difference.
- An independent-seed probe with twice the paths validates the prediction: 490 predicted vs 466
  CNY/day measured.
- The CIs are a block bootstrap over calendar sessions, with every live start kept together.
- The report also shows a sensitivity, with the noise scaled to the measured level.

## How to run

```bash
pip install akshare          # only stage 1 needs it; not a project dependency
python example/case_studies/csi_real_data/01_fetch.py            # → ~/.cache/quantark/csi_real_data
python example/case_studies/csi_real_data/02_build_indices.py
python example/case_studies/csi_real_data/03_gap_through_time.py  # ~8 min, 4 workers
python example/case_studies/csi_real_data/04_hedge_backtest.py    # ~40 min, 4 workers
python example/case_studies/csi_real_data/05_report.py
python -m pytest test/test_csi_real_data.py -n0                   # offline, a few seconds
```

Set `CSI_REAL_DATA_CACHE` to move the cache; it must lie outside the repo. Results are deterministic
for fixed seeds, and the seed scheme is recorded in the JSON. Re-fetching after new closes changes
the data and so the results.

## Provenance and data policy

- Raw vendor prices are **not** committed. They live in the cache.
- Committed derived series:
  - each rule's daily index level and current weights;
  - Stage A and B outputs.
- Under the fixed-share rule, levels and weights let a reader rebuild each name's rebased price
  series. That is acceptable for this public test case.
- Share counts are not committed.

## Limits

- The reference is a GBM model on historical parameters, not market truth. It has no smile and no
  jumps, and uses no implied vols. Stage B's real-path P&L is the only model-free judge.
- The sample covers 2024 to 2026 only. Stage B's 22 one-year starts overlap heavily, so the terminal
  P&L per start is descriptive, and the daily paired statistic is the test.
- Daily hedging uses fractional shares at the close, with no costs or 100-share lots.
- Stage A's expiries in 2027 use weekday sessions (165 of 525 records).

Design: `docs/superpowers/specs/2026-11-18-csi-real-data-design.md`. Tests: `test/test_csi_real_data.py`.
