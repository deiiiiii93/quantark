# CSI on real data: rebalancing, a single-underlying desk model, and its hedge

Date: 2026-11-18 · Status: approved (built via /ship; refinements in the run spec D9–D14) · Follows `2026-11-17-csi-basket-delta-design.md`
(the synthetic study, landed separately as `example/case_studies/csi_basket_delta/`).

## Intent

The desk is starting a customized strategy index (CSI) business: weighted baskets of A-shares and
ETFs with OTC options sold on them. The synthetic study showed, in a stylised GBM world, *why*
rebalancing decides whether an index-level model is good enough (after each rebalance the option
sees the names only through the index level; the per-name split error is linear in the time to the
next rebalance; a constant-mix index is exactly lognormal). This study asks the same question on
**real prices**, with the framing the desk will actually use:

- **Reference ("truth")**: a per-name Monte Carlo with each name's historical vol and the historical
  correlation matrix; per-name bump-and-reprice as the delta reference.
- **Test model**: Black on the index as one underlying, with the vol estimated from the index's own
  historical series; total delta split linearly across names by current weights (the whole-index
  "combination" order).

and adds the judge no model controls: **hedging each model's deltas along the real 2024–now path**.

## Goals

1. Build a CSI from real daily prices under each rebalance rule (none / quarterly / monthly /
   weekly / daily) and show how much the rule alone changes the index path and its realized vol.
2. **Stage A, the gap through time**: on monthly evaluation dates from 2024-01 to now, calibrate
   both models on a trailing 1Y window and compare PV, total delta (Δ_I) and per-name deltas, per
   rule, in CNY on 1e8 notional.
3. **Stage B, hedge P&L on the real path**: for options sold on monthly start dates that have
   expired, hedge daily with each model's per-name deltas through expiry and compare the realized
   hedge error, per rule.
4. Decompose the per-name delta gap into total-delta error + split error (the synthetic study's
   decomposition) and check the split error against the first-order closed form
   `τ·wᵢ·I·[(βᵢ−1)σ_I²·I·Γ + (q̄−qᵢ)·Δ]` on real calibrations.

## Non-goals

- No new pricing library code in `quantark/`; this is a case study.
- No implied vols, smiles or vol surfaces; both models use historical parameters only.
- No dividend modelling beyond what back-adjusted prices imply (see Decisions).
- v1 has no trading costs or lot sizes (designed so they can be added; see "Later").
- The desk's real CSI baskets are out of scope (dropped by the user, 2026-11-18). The basket is a
  module-level definition for the public stand-in; no generic basket loader or untracked-results
  mode.

## Anti-goals

- Committing raw vendor price data to this public repo.
- A comparison that quietly favours one model (e.g. different calibration windows or different
  prices for the two models).

## Constraints

- Public repo: raw prices are cached **outside the repo**; only scripts, derived JSON and the HTML
  report are committed. Derived **daily** series (index levels per rule, weights, per-name deltas)
  may be committed: the basket is a public test case, not a real CSI (user, 2026-11-18).
- Data via akshare (Wind is unavailable). akshare is **not** added to project dependencies; the fetch
  script documents `pip install akshare`. akshare renamed functions in 1.20.4x; use the current
  names (`stock_zh_a_hist_em`, `fund_etf_hist_em` / `fund_etf_hist_sina`, `index_zh_hist_em`).
- Tests run offline on synthetic price frames; no test touches the network.
- Reuse the synthetic study's engine (`example/case_studies/csi_basket_delta/_common.py`):
  multi-name GBM RQMC with full correlation (singular allowed), CRN per-name FD, rebalance rules.
- Report follows the repo UI language (Bloomberg × Swiss, frozen nav, theme switcher) and typesets
  formulas as MathML like the synthetic report.
- Python 3.10–3.13 (CI); no backslashes inside f-string expressions.

## Approaches considered

1. **Extend the synthetic study's branch** with real-data stages and reorder its report. One branch,
   but it delays landing verified work and grows one study into two.
2. **Land the synthetic study; build a sibling study on a new branch** that imports the shared
   engine and is organised around the reference/test framing from the start. *(chosen)*
3. **Rewrite into one study**, synthetic and real, with M2–M5 and the moment-matched model demoted.
   Most coherent, but rewrites verified work and re-opens its findings.

Chosen: **2** (user's pick). The synthetic study is the mechanism; the real-data study is the
evidence. The new report links back to the synthetic one for derivations.

## Decisions (user)

- Basket: the public stand-in only (below). The real-CSI follow-up was dropped.
- Evaluation period 2024-01 to now; prices fetched from 2023-01 so the first trailing-1Y
  calibrations are complete.
- Measurement: Stage A (gap through time) then Stage B (hedge P&L on the real path).
- Location: new sibling study on a new branch after the synthetic study lands.

## Decisions (controller, correctable)

- **Basket** (weights as the synthetic study):

  | role | name | code | weight |
  |---|---|---|---|
  | large cap | Kweichow Moutai | 600519 | 25% |
  | large cap | China Merchants Bank | 600036 | 20% |
  | high-beta tech | iFlytek | 002230 | 15% |
  | broad index | CSI 300 ETF | 510300 | 30% |
  | hedge-like | Gold ETF | 518880 | 10% |

- **Prices**: back-adjusted (`hfq`) daily closes, so the CSI is a **total-return** index and every
  model uses dividend yield q = 0 with no decrement. Reason: avoids corporate-action handling and a
  dividend model; both models see the same prices. Consequence: the q̄ − qᵢ term of the closed form
  is zero here.
- **Calendar**: trading days where all five names have a close; a name suspended on a day carries
  its previous close (count and report suspensions).
- **Index construction**: base 1000 at the first trading day of 2023 at target weights. Rebalance at
  the close on: quarterly = first trading day of Jan/Apr/Jul/Oct; monthly = first trading day of
  each month; weekly = first trading day of each week; daily = every day; none = never. For rule
  *none* the weights therefore drift from 2023-01 onwards, which is exactly the "historical vol of
  a drifted index" effect the synthetic study could not measure.
- **Rate**: flat r = 1.8% p.a. for discounting and hedge financing (both models identical; the gap
  is insensitive to it). Record it in the results.
- **Calibration (both models, same window)**: trailing 244 trading days of daily log returns ending
  on the evaluation date. Reference: per-name sample vols and sample correlation (PSD-checked; the
  engine's eigen square root handles near-singular matrices). Test: sample vol of the CSI's own
  daily log returns under the same rule over the same window.
- **Reference model state on a real date: real rebalance schedule.** Real periods do not match the
  engine's fixed lengths (a quarter can have 64 trading days vs the engine's 61; some months have
  23), and `state_quantities` rejects `elapsed ≥ period`. So the engine gets an **optional explicit
  schedule**: the list of remaining rebalance times (in years on the 244-day clock, counted in real
  trading days from the evaluation date) plus maturity. When given, it replaces the
  period-derived `schedule_times`; when absent, behaviour is unchanged. This is an additive change
  to `csi_basket_delta/_common.py`; the synthetic study's tests must stay green and its results JSON
  must reproduce bit-identically. Real rebalance dates come from the exchange trading calendar
  (akshare `tool_trade_date_hist_sina`, which includes announced future sessions); beyond the
  calendar's horizon, weekdays are used and the report says so.
- **Evaluation and start dates**: the **10th trading day of each month**, after that day's close
  (and after any rebalance at that close). Not the 1st trading day: that is itself a monthly (and
  every third month a quarterly) rebalance date and would always catch those rules at their largest
  stub. Every result records each rule's stub τ on that date, so the position in the period is
  visible. The first evaluation date (10th trading day of Jan 2024) has ≈ 251 index returns behind
  it, so the trailing 244-return window is complete without moving the 2023 base date.
- **Options**: European calls on 1e8 CNY of index notional, 1Y, struck at moneyness 0.9 / 1.0 / 1.1
  of the index level on the evaluation date (Stage A). Stage B hedges the ATM call only.
- **Test model deltas**: Black Δ_I (on the forward with q = 0) times current shares nᵢ (= the
  whole-index order).
- **Reference deltas**: CRN central bump of each name on the MC (the synthetic study's M3), with the
  pathwise per-name delta also recorded.
- **Stage B mechanics**: sell one ATM 1Y call per model per rule per start date at that model's own
  PV. Each trading day at the close: recalibrate both models on the trailing window (as a desk would
  re-mark), recompute per-name deltas, trade to them (fractional shares, no costs), finance cash at
  r. At expiry pay the payoff on the real index level. Hedge error = terminal P&L, reported in CNY
  and as % of premium; also the daily P&L series to show *when* errors occur (rebalance dates,
  drift). Rebalance trades arise automatically because nᵢ jumps.
- **Stage B arms (three, same paths, same day)**: (1) test model: Black Δ_I × nᵢ; (2) reference:
  per-name MC bumps; (3) attribution arm: the reference MC's own Δ_I × nᵢ (the synthetic study's
  M1). (1) − (3) isolates the total-delta/vol error; (3) − (2) isolates the split error. Each arm's
  premium is its own model PV for (1) and (2); arm (3) uses the reference PV.
- **Stage B statistic.** Overlapping 1Y starts give only 2–3 independent terminal outcomes, so the
  headline is **daily**: the daily hedged-portfolio P&L of each arm, and the **paired daily
  difference** between arms on the same day and start (hundreds of observations), reported as std
  in CNY/day per 1e8 with a block-bootstrap CI (blocks of ≥ 20 days, by start). Terminal P&L per
  start is reported descriptively (table + chart), not as the test. No pass/fail threshold: the
  report states the CNY numbers next to the premium and leaves "good enough" to the desk.
- **Start dates (Stage B)**: the evaluation dates whose 1Y option has expired by the last data date
  (≈ 23). The 6M set is not built in v1; the daily paired statistic does not need it.
- **Compute budget**: Stage B is the expensive part (starts × ~244 days × 5 rules × MC with 10
  bumps). The builder picks paths/batches so the MC noise in the reference deltas adds less than
  10% to the **std of the paired daily difference (2) − (1)** (measured once on a few starts by
  doubling paths), and states the number. Caching per (date, rule) calibration is expected.

## Outputs

- `example/case_studies/csi_real_data/` with numbered stages:
  `01_fetch` (akshare → cache outside the repo), `02_build_indices` (5 rules, levels, weights,
  realized vols), `03_gap_through_time` (Stage A), `04_hedge_backtest` (Stage B), `05_report`
  (HTML), plus `_data.py`, a README, and offline tests in `test/test_csi_real_data.py`.
- Committed results JSON for the stand-in basket and the HTML report.

## Report story (layered: desk, then business)

1. The five index paths from the same prices, and how far apart they end; realized vol by rule;
   weight drift under *none*.
2. Stage A: PV, Δ_I and per-name gaps through time per rule; the split gap vs the closed form.
3. Stage B: hedge error per model per rule; when it happens (rebalance dates, drift episodes).
4. Business read: which rebalance rules make the single-underlying model good enough on real data,
   and what the reference model buys for the rest. Link to the synthetic study for the mechanism.

## Left to the builder

File layout inside the study, helper names, test design (offline fixtures), MC path counts within
the stated noise budget, chart choices within the UI language.

## Later (out of v1)

A-share 100-share lots and transaction costs in Stage B; implied-vol inputs; price-return index with
an explicit dividend model.
