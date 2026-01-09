# Backtest Module: Theory and Methodology

## Overview

The Backtest module in QuantArk serves as a rigorous simulation framework designed to evaluate the historical performance of hedging strategies and portfolio management algorithms. Unlike simple P&L calculators, it simulates the passage of time, market data updates, strategy signal generation, and trade execution with realistic constraints.

## Relationship to Other Risk Modules

The Backtest module is one of three complementary risk analysis frameworks in QuantArk, each addressing a different dimension of risk:

| Module | Time Dimension | Data Source | Risk Question Answered |
|--------|----------------|-------------|------------------------|
| **Stress Test** | Instantaneous (t=0) | Hypothetical shocks | "What is my loss if X happens now?" |
| **Dynamic Scenario** | Multi-day (t=0→T) | Hypothetical paths | "How does risk evolve under scenario X over T days?" |
| **Backtest** | Historical | Actual market data | "How would this strategy have performed historically?" |

The Backtest module is unique in that it uses **actual historical market data** to evaluate strategies. This provides empirical validation but is limited to historical patterns—unlike Dynamic Scenario, which can explore hypothetical futures, or Stress Test, which examines instantaneous tail risks.

## Core Framework

The backtesting engine operates on a time-stepped simulation model. For each time step in the simulation period:

1.  **Market Data Update**: The `PricingEnvironment` is updated with new market data (spot prices, volatilities, rates) from the `MarketDataAdapter`.
2.  **Portfolio Re-valuation**: All positions in the portfolio are marked-to-market using the updated environment. Greeks (Delta, Gamma, Vega, etc.) are recalculated.
3.  **Strategy Evaluation**: The active strategy (e.g., Delta Neutral, DV01 Neutral) analyzes the current portfolio state against its defined targets and constraints.
4.  **Signal Generation**: If a rebalancing trigger is met (e.g., delta limit breached), the strategy generates a list of required trades.
5.  **Execution & Cost Modeling**: Trades are executed against the market. Transaction costs are calculated based on the selected model (e.g., slippage, commissions) and deducted when reporting net P&L.
6.  **State Recording**: The system records the full state (P&L, exposures, trades) for post-analysis.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A professional technical flowchart of a quantitative backtesting engine. Left to right flow: 'Market Data Feed' -> 'Portfolio Re-valuation' -> 'Strategy Logic' -> 'Risk Check' -> 'Trade Execution' -> 'Performance Logging'. Use a clean, modern style with a professional blue and grey color scheme."

## Discrete-Time Hedging Theory

### Local Expansion and Residual P&L

Backtesting delta/DV01 hedges is fundamentally a **discrete-time** replication experiment: we rebalance at a set of timestamps, and the residual P&L comes from higher-order risks and model mismatch.

For a single underlying equity option portfolio value \(V(S, \sigma, t)\), a standard local expansion gives:

$$
\Delta V \approx \Delta \,\Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \,\Delta t + \text{Vega}\,\Delta \sigma + \cdots
$$

If we hedge with a delta-one instrument (spot/futures, \(\Delta_{\text{hedge}} \approx 1\)) and set hedge size \(h \approx -\Delta\), the first-order \(\Delta \,\Delta S\) term is reduced, but the residual remains:

$$
\Delta V_{\text{hedged}} \approx \frac{1}{2}\Gamma (\Delta S)^2 + \Theta \,\Delta t + \text{Vega}\,\Delta \sigma + \cdots \;-\; \text{Transaction Costs}
$$

This is where "gamma bleed", theta decay, vol moves, gaps, and costs show up in the backtest results.

### Expected Hedging Error Under GBM

Under the Black-Scholes assumptions with Geometric Brownian Motion, the discrete hedging error can be characterized analytically. For a delta-hedged option rebalanced at intervals of \(\Delta t\):

$$
\mathbb{E}[\text{P\&L}_{\text{hedged}}] \approx \frac{1}{2}\Gamma S^2 \sigma^2 \Delta t + \Theta \Delta t
$$

By the Black-Scholes PDE, \(\Theta + \frac{1}{2}\Gamma S^2 \sigma^2 = rV\), so the expected P&L equals the risk-free return in the continuous limit. However, **discrete rebalancing introduces variance**:

$$
\text{Var}(\text{P\&L}_{\text{hedged}}) \approx \frac{1}{2}\Gamma^2 S^4 \sigma^4 (\Delta t)^2 + \cdots
$$

This variance scales with \((\Delta t)^2\), meaning more frequent hedging reduces replication error proportionally to the square of the rebalancing interval—subject to transaction costs.

### The Optimal Hedging Interval Problem

Transaction costs create a fundamental tradeoff: more frequent hedging reduces gamma risk but increases costs. The **Whalley-Wilmott** framework characterizes this as a "no-trade region" around the target delta:

$$
|\Delta_{\text{net}} - \Delta_{\text{target}}| < \text{Threshold} \implies \text{No Hedge}
$$

The optimal threshold balances the expected marginal benefit of rebalancing against the marginal transaction cost. For proportional costs \(c\) (per unit notional), the optimal half-width \(h^*\) of the no-trade region scales as:

$$
h^* \sim \left(\frac{3c S}{2 |\Gamma| \sigma^2 \Delta t}\right)^{1/3}
$$

This predicts that larger costs, higher volatility, and longer intervals all widen the optimal no-trade region.

## Hedging Strategies

### Equity: Delta-Neutral Hedging
The primary goal of a Delta-Neutral strategy is to immunize the portfolio against small directional movements in the underlying asset.

*   **Theory**: The strategy monitors the portfolio's net delta ($\Delta_{net}$). 
    $$ \Delta_{net} = \sum_{i} \Delta_i \times Q_i $$
    Where $\Delta_i$ is the delta of instrument $i$ and $Q_i$ is the quantity.
*   **Rebalancing Logic**: A hedge trade is triggered when $|\Delta_{net}| > \text{Threshold}$ (optionally combined with time-based frequency rules). The hedge quantity is computed to bring $\Delta_{net}$ back to a target (usually 0).

    General sizing:
    $$
    Q_{hedge} = - \frac{\Delta_{net} - \Delta_{target}}{\Delta_{hedge\_instrument}} \times \text{hedge\_ratio}
    $$

    For spot/futures hedges in practice (\(\Delta_{hedge\_instrument} \approx 1\)):
    $$
    Q_{hedge} \approx -(\Delta_{net} - \Delta_{target}) \times \text{hedge\_ratio}
    $$

### Fixed Income: DV01-Neutral Hedging
For fixed income portfolios, the primary risk measure is DV01 (Dollar Value of an 01), representing the P&L change for a 1 basis point parallel shift in the yield curve.

*   **Theory**: The strategy protects against parallel rate shifts by neutralizing the portfolio's DV01.
    $$ \text{DV01}_{net} = \sum_{j} \text{DV01}_j $$
*   **Hedge Sizing (Futures)**: If a hedge futures contract has DV01 \(\text{DV01}_{fut}\) per contract, the contract count is:
    $$
    N_{contracts} = -\frac{\text{DV01}_{net} - \text{DV01}_{target}}{\text{DV01}_{fut}} \times \text{hedge\_ratio}
    $$
*   **Execution**: Hedging is typically performed using Treasury Futures (e.g., TU, FV, TY, US). The number of contracts is determined by the futures' specific DV01 and the portfolio's exposure.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A financial line chart comparing two equity curves on a dark background. Line 1: 'Unhedged Portfolio' (volatile, jagged, red). Line 2: 'Delta-Neutral Strategy' (smooth, stable, green). Style: Professional financial terminal UI, dark mode, clear grid lines."

## Transaction Cost Models

### Cost Components

Realistic backtesting requires accurate modeling of friction. QuantArk supports a comprehensive cost model:

1.  **Fixed Commission**: A flat fee per trade ticket.
2.  **Proportional Commission**: A fee based on notional value (bps).
3.  **Bid-Ask Spread**: The cost of crossing the spread, effectively paying half the spread on entry and exit.
4.  **Market Impact (Slippage)**: Modeled as a function of trade size relative to market liquidity. Large trades incur higher impact costs.

Conceptually:
$$
\text{Total Cost} = \text{Fixed} + (\text{Rate} \times |\text{Notional}|) + \text{Spread Cost} + \text{Slippage Cost}
$$

### The Cost-Liquidity Tradeoff

Transaction costs fundamentally alter the optimal hedging strategy. In the frictionless Black-Scholes world, continuous hedging eliminates all directional risk. With costs, the optimal strategy becomes **band-based**: hedge only when delta deviates beyond a threshold.

The economic intuition follows from marginal analysis:
- **Marginal benefit of hedging**: Reduces expected gamma P&L variance by \(\sim \Gamma^2 S^4 \sigma^4 \Delta t\)
- **Marginal cost of hedging**: Linear in trade size (for proportional costs)

When these are equated, we obtain the optimal no-trade region width. The Whalley-Wilmott result shows the threshold scales as \(c^{1/3}\)—meaning costs have a **sublinear impact** on optimal hedge width, but the effect is significant.

### Impact on Sharpe Ratio Optimization

Transaction costs reduce strategy Sharpe ratios through two mechanisms:

1. **Direct drag**: Expected P&L is reduced by the average cost per trade times hedge frequency
2. **Indirect variance increase**: Wider no-trade regions (to avoid costs) increase gamma risk exposure

For a strategy with \(N\) hedges over period \(T\) and average cost \(C\):
$$
\text{Sharpe}_{\text{net}} \approx \frac{\mu - \frac{N}{T}C}{\sigma}
$$

where \(\mu\) is the gross expected return and \(\sigma\) is volatility. This highlights the tension: more aggressive hedging (higher \(N/T\)) reduces \(\sigma\) but increases the cost drag.

## Performance Metrics

The module evaluates strategies using institutional-grade metrics:

### Risk-Adjusted Returns

**Sharpe Ratio** measures excess return per unit of risk:

$$
\text{Sharpe} = \frac{\mathbb{E}[R - R_f]}{\sigma_{R-R_f}} = \frac{\bar{R} - R_f}{\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(R_t - \bar{R})^2}}
$$

where \(R_t\) is the return at time \(t\), \(R_f\) is the risk-free rate, and \(\bar{R}\) is the mean return.

**Sortino Ratio** penalizes only downside volatility:

$$
\text{Sortino} = \frac{\mathbb{E}[R - R_f]}{\sigma_{\text{downside}}}, \quad \sigma_{\text{downside}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\min(R_t - R_f, 0)^2}
$$

### Drawdown Analysis

**Maximum Drawdown** measures the largest peak-to-trough decline:

$$
\text{MDD} = \max_{\tau_1 < \tau_2} \left( \frac{V_{\tau_1} - V_{\tau_2}}{V_{\tau_1}} \right)
$$

where \(V_t\) is portfolio value at time \(t\).

**Recovery Factor** relates total return to maximum drawdown:

$$
\text{Recovery Factor} = \frac{\text{Total Return}}{\text{MDD}}
$$

Higher values indicate better risk-adjusted performance.

### Hedging Efficiency

**Tracking Error** measures how well the hedge maintains its target:

$$
\text{TE}_{\Delta} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(\Delta_t - \Delta_{\text{target}})^2}
$$

For DV01 hedging, the analogous metric uses DV01 deviation.

**Hedge Frequency** captures trading intensity:

$$
\text{Hedge Frequency} = \frac{N_{\text{hedges}}}{T_{\text{days}}}
$$

This is crucial for cost management—higher frequency generally increases transaction cost drag.

**Hedge Effectiveness** quantifies variance reduction:

$$
\text{HE} = 1 - \frac{\sigma^2_{\text{hedged}}}{\sigma^2_{\text{unhedged}}}
$$

Values closer to 1 indicate more effective risk reduction.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A financial dashboard UI wireframe showing four modules: 1. 'Sharpe Ratio' (gauge widget), 2. 'Max Drawdown' (bar chart), 3. 'Win Rate' (pie chart), 4. 'Total Return' (large distinct text). Style: Dark mode professional trading interface, high contrast."

## Model Coverage (QuantArk Implementation)

The backtest module implements a protocol-based architecture that supports multiple asset classes through a common interface:

### Supported Strategies

| Strategy | Asset Class | Target Risk | Hedge Instrument |
|----------|-------------|-------------|------------------|
| `DeltaNeutralStrategy` | Equity | Delta → 0 | Spot or Futures |
| `DV01NeutralStrategy` | Fixed Income | DV01 → 0 | Treasury Futures |
| `ConvexityNeutralStrategy` | Fixed Income | Convexity → 0 | Bond/Futures mix |

### Architecture

The module uses Python protocols (`backtest/base.py`) to define contracts for:
- **`BaseBacktestEngine`**: Main simulation loop (time-stepping, state management)
- **`BaseStrategy`**: Strategy interface (trigger evaluation, hedge sizing)
- **`BaseHedgeExecutor`**: Trade execution and cost calculation
- **`BaseBacktestResults`**: Results access and metrics computation

This protocol-based design enables the same strategy classes to work across equity and fixed income portfolios.

### Configuration

Key configuration options affect calculations:
- **`hedge_frequency`**: Controls rebalancing cadence (threshold, daily, continuous)
- **`hedge_threshold`**: Sets the no-trade region width
- **`transaction_costs`**: Selects cost model (Zero, Fixed, Proportional, Slippage, Spread, Complete)
- **`save_state`**: Enables detailed state history for post-analysis

### References

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Whalley, A. E., & Wilmott, P. (1997). "An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs." *Mathematical Finance*.
