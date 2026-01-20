# Autocallable Risk Profile Report (Snowball-first)

## 1. Executive Dashboard (Traffic Light)
*Goal: Instant situational awareness for the trader/manager.*
- **Vital Signs:** Table of current PV, Delta, Gamma, Vega, Theta.
- **Barrier Status:** 
  - Distance to KO (in % and in standard deviations `(ln(B/S) / (σ√T))`).
  - Distance to KI (in % and in σ).
  - Status Indicator: "Safe", "KO Danger", or "KI Danger".
- **Risk Interpretation:** Short paragraph explaining the dominant risk (e.g., "The product is currently near the KO barrier; expect high Gamma and potential delta-flip on observation.")

## 2. Product & Market Snapshot
- **Terms Summary:** Next observation date, current strike, barrier levels.
- **Market State:** Current Spot, ATM Vol, and Dividend/Basis level.

## 3. Barrier Risk (Zoom Analysis)
*Goal: Understand "cliff-edge" effects near barriers.*
- **Grids:** High-density grid `S ∈ Barrier * [0.98, 1.02]` (20 nodes).
- **Surfaces:** Gamma and Vega zoom plots.
- **Interpretation:** "Near-barrier Gamma spikes indicate significant hedging costs if the underlying oscillates around the barrier level."

## 4. Required Surfaces & Greeks
- **First Order:** Delta, Gamma, RhoQ (Dividend), RhoB (Basis).
- **Vol Surface Risk:** Vega, Vanna (Skew risk), Volga (Vol convexity).
- **Higher-Order Time Greeks:** Theta, Charm (Delta drift).
- **Interpretation:** "High Vanna suggests that your Delta hedge will become insufficient if volatility spikes—consider over-hedging the downside."

## 5. Scenario Ladder (Trader Matrix)
- **PnL Matrix:** 5x3 or 7x3 Spot vs. Vol ladder.
- **Worst-Case Highlight:** Identification of the most damaging scenario.
- **Interpretation:** "The portfolio is most vulnerable to a 'Gap Down' scenario (Spot -10%, Vol +5%); PnL loss is driven primarily by the short Gamma position."

## 6. Bucketed Greeks (Term-Structure)
- **Tenor Buckets:** 1M, 3M, 6M, 1Y.
- **Interpretation:** "The 6M-1Y bucket carries 80% of the Vega risk; a flattening of the vol curve will result in significant PnL gain/loss."

## 7. Risk-Neutral Event Stats
- **RN Probabilities:** P(KO), P(KI), P(Survive).
- **Cashflow Attribution:** Expected discounted coupons vs. Principal repayment.
- **Interpretation:** "65% probability of Early Termination (KO) before month 6. High probability of survival suggests a 'Coupon Collector' profile."

## 8. Historical & Stress Analysis
- **Historical Replay:** Factor shocks applied to today's parameters.
- **PnL Distribution:** Histograms of simulated PnL.
- **Interpretation:** "Under 2015-style crash scenarios, the product exhibits significant tail risk beyond the VaR 99% level."