# Stress Test Module: Theory and Methodology

## Overview

Stress Testing is a critical risk management component used to evaluate the resilience of a portfolio under extreme but plausible market conditions. The Stress Test module in QuantArk implements a **Static Scenario Analysis** framework. It applies instantaneous shocks to market factors and re-evaluates the portfolio's value without the passage of time.

## Relationship to Other Risk Modules

The Stress Test module is one of three complementary risk analysis frameworks in QuantArk, each addressing a different dimension of risk:

| Module | Time Dimension | Data Source | Risk Question Answered |
|--------|----------------|-------------|------------------------|
| **Stress Test** | Instantaneous (t=0) | Hypothetical shocks | "What is my loss if X happens now?" |
| **Dynamic Scenario** | Multi-day (t=0→T) | Hypothetical paths | "How does risk evolve under scenario X over T days?" |
| **Backtest** | Historical | Actual market data | "How would this strategy have performed historically?" |

The Stress Test module is unique in its **instantaneous** nature: it examines tail risk at a single point in time, making it ideal for regulatory capital calculations and "what-if" analysis. Unlike Dynamic Scenario, it does not model the evolution of risk over time; unlike Backtest, it is not constrained by historical patterns.

## Theoretical Foundation

The core value proposition of a stress test is to answer the question: "How much would I lose if [X] happens right now?"

The valuation change (ΔV) is calculated as:
$$ \Delta V = V(S_{stressed}, \sigma_{stressed}, r_{stressed}, ...) - V(S_{0}, \sigma_{0}, r_{0}, ...) $$

Where:
*   $V(...)$ is the portfolio valuation function.
*   $S, \sigma, r$ represent Spot, Volatility, and Interest Rates.
*   Subscript $0$ denotes the current market state.
*   Subscript $stressed$ denotes the post-shock state.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A 3D risk surface plot (financial volatility plane). X-axis: 'Spot Price Change', Y-axis: 'Volatility Change', Z-axis: 'Portfolio P&L'. The surface features a dramatic 'cliff' or steep drop-off representing tail risk. Style: High-tech scientific visualization, heatmap coloring."

## Repricing vs. Sensitivity Approximation

### Full Repricing Approach

QuantArk stress tests are implemented as **full repricing under a shocked market state**: the engine constructs a stressed `PricingEnvironment` and re-values every position using its pricing engine.

### Taylor Expansion Approximation

It is useful to contrast this with a Greek-based approximation (often used as a fast diagnostic):

$$
\Delta V \approx \Delta \,\Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \text{Vega}\,\Delta \sigma + \text{Rho}\,\Delta r + \cdots
$$

### Error Bounds and When to Use Each

The Taylor approximation error depends on the shock size and the magnitude of higher-order derivatives:

$$
\epsilon_{\text{Taylor}} = \mathcal{O}(|\Delta S|^3) + \mathcal{O}(|\Delta \sigma|^2) + \cdots
$$

**Guidelines for method selection:**

| Shock Size | Taylor Approximation | Full Repricing |
|------------|---------------------|----------------|
| Small (±1-2%) | Excellent; error < 1% | Unnecessary overhead |
| Medium (±5-10%) | Good; error ~1-5% | Recommended for accuracy |
| Large (±20%+) | Poor; error can exceed 10% | Essential |

For barrier options, digital options, or other highly non-linear payoffs, full repricing is **always recommended** regardless of shock size—the Taylor expansion may miss critical discontinuities.

## Factor Correlation in Stress Scenarios

### The Spot-Vol Correlation Phenomenon

Real-world stress events exhibit strong correlations between risk factors. The most significant is the **negative spot-volatility correlation** observed during market crises:

$$
\rho_{S,\sigma} = \text{Corr}(\Delta S/S, \Delta \sigma) < 0 \quad \text{(during stress periods)}
$$

This "leverage effect" occurs because:
1. Falling equity prices increase financial leverage, raising default risk
2. Increased uncertainty drives demand for options as protection, pushing up implied volatility
3. Market makers delta-hedge selling, creating feedback loops

### Multi-Factor Scenario Design

Realistic stress scenarios must therefore combine shocks to correlated factors. For a "Market Crash" scenario:

$$
\begin{aligned}
\Delta S &= -20\% \\
\Delta \sigma &= +50\% \\
\Delta r &= -50\text{bps} \quad \text{(flight to quality)}
\end{aligned}
$$

The joint effect is often **non-additive** due to cross-gamma terms:

$$
\Delta V \approx \Delta \cdot \Delta S + \frac{1}{2}\Gamma (\Delta S)^2 + \text{Vega} \cdot \Delta \sigma + \text{Vanna} \cdot \Delta S \cdot \Delta \sigma + \cdots
$$

where **Vanna** = \(\frac{\partial^2 V}{\partial S \partial \sigma}\) captures the sensitivity of delta to volatility changes.

## Stress Architecture

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A system workflow diagram for Stress Testing. Three stages: 1. Inputs (Portfolio, Base Market, Scenarios) -> 2. Processing (Apply Shocks, Generate Stressed Envs, Full Repricing) -> 3. Outputs (P&L Impact, Greek Sensitivities, Capital Report). Style: Professional block diagram, clear input-process-output flow."

### 1. Stress Types

To provide flexibility, shocks can be applied in three ways:

*   **Percentage Shock**: A relative change (e.g., Equity Spot -20%).
    $$ X_{new} = X_{old} \times (1 + \text{shock}) $$
*   **Absolute Shock**: An additive change (e.g., Interest Rates +100bps).
    $$ X_{new} = X_{old} + \text{shock} $$
*   **Value Override**: Setting a specific level (e.g., Volatility = 80%).
    $$ X_{new} = \text{shock} $$

#### Composition Properties

The choice of stress type affects how multiple shocks compose. For **percentage shocks** applied sequentially:

$$
X_{n} = X_0 \prod_{i=1}^{n}(1 + \epsilon_i)
$$

This means the order of application matters for large shocks (non-commutative). For **absolute shocks**:

$$
X_{n} = X_0 + \sum_{i=1}^{n}\delta_i
$$

These are order-independent. The practical implication:
- Use **percentage shocks** for variables with exponential dynamics (spot prices, volatility)
- Use **absolute shocks** for rates and spreads where additive shifts are the market convention
- Use **value override** to set specific stress levels regardless of starting point

#### Additive vs. Multiplicative Shocks

For small shocks \(|\epsilon| \ll 1\), the distinction is negligible:

$$
X(1+\epsilon) \approx X + X\epsilon
$$

But for large stress scenarios (±20% or more), the difference matters:
- A 20% drop followed by a 20% rise: \(X \times 0.8 \times 1.2 = 0.96X\) (4% net loss)
- Versus absolute: \((X - 0.2X) + 0.2X = X\) (no net loss)

This reflects a real feature of markets: multiplicative shocks capture compounding effects.

### 2. Granularity Levels

Risk factors can be stressed at different scopes, reflecting different types of risk events:

*   **Portfolio Level**: Global shocks (e.g., "Global Market Crash" affects all equity assets). Use for systemic events.
*   **Underlying Level**: Idiosyncratic shocks (e.g., "AAPL earnings miss" affects only AAPL). Use for single-name risk.
*   **Position Level**: Specific adjustments targeting a position identifier. Use for what-if analysis on specific holdings.

**Theoretical justification**: This hierarchy maps to the fundamental theorem of asset pricing—systematic risk versus idiosyncratic risk. Portfolio-level shocks test systematic risk exposure, while underlying/position-level shocks test concentration risk.

## Supported Risk Factors
The module supports stressing a wide range of market drivers across asset classes:

*   **Equity Spot**: Direct percentage changes to the underlying asset price.
*   **Implied Volatility**: Shifts to the volatility surface. Can be a parallel shift (level) or a twist (skew/smile adjustment).
*   **Interest Rates**: Shifting the risk-free yield curve. Crucial for long-dated options and bond portfolios.
*   **Dividend Yield**: Adjusting the continuous dividend yield assumption.
*   **Credit Spreads**: For Fixed Income, widening or tightening the Z-spread or OAS over the risk-free curve.

## Scenario Construction

A **Scenario** is a collection of simultaneous stresses. Real-world crises rarely involve a single factor moving in isolation.

### Standard Scenarios
*   **Market Crash**: Typically characterized by a sharp drop in Spot prices and a spike in Volatility (Spot-Vol correlation is usually negative).
*   **Rate Hike/Cut**: Parallel shifts in the yield curve, affecting Fixed Income assets and the discounting of Equity derivatives.
*   **Liquidity Crisis**: widening of credit spreads and bid-ask spreads.

### Historical Scenarios
The module supports "Replay" scenarios based on historical data:
*   **Black Monday (1987)**: Extreme equity crash.
*   **2008 Financial Crisis**: Combined equity drop and volatility explosion.
*   **COVID-19 (2020)**: Rapid crash followed by high volatility regime.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A hierarchical structure diagram of a Stress Test Scenario. Top node: 'Scenario: Market Crash'. Middle layer (3 nodes): 'Stress 1: Spot Price -20%', 'Stress 2: Volatility +50%', 'Stress 3: Rates +100bps'. Bottom node: 'Portfolio Valuation Engine'. Arrows flow from Stresses to the Engine. Style: Clean professional flowchart."

## Model Coverage (QuantArk Implementation)

QuantArk applies stresses through parameter-specific adapters on `PricingEnvironment`. Out of the box, the stress engine supports:

### Supported Parameters

*   **spot**: Requires a spot quote; stressed value must remain positive.
*   **volatility / vol**: Supported for flat volatility surfaces (`FlatVolSurface`) only.
*   **rate**: Supported for flat curves and interpolated curves; applied as a parallel shift.
*   **key_rate**: Requires `tenor_bucket` metadata (e.g., `"5Y"`); applies bucketed changes on interpolated curves (falls back to parallel shift for flat curves).
*   **dividend_yield / div_yield / dividend**: Supported for flat/continuous dividend yield; stressed dividend yield is clamped at 0.0 if it would go negative.
*   **spread**: Currently mapped to the same mechanism as a rate shock (parallel shift proxy).

If a parameter is not supported, the engine raises an error unless a custom adapter is registered.

### Scenario Persistence

Scenarios can be persisted and restored via YAML/JSON serialization. This enables:
- **Scenario libraries**: Build reusable stress test suites
- **Version control**: Track scenario evolution
- **Sharing**: Distribute scenarios across teams
- **Regulatory reporting**: Document stress test assumptions

### Granularity in Practice

The granularity levels (PORTFOLIO/UNDERLYING/POSITION) enable sophisticated risk analysis:

| Use Case | Granularity | Example |
|----------|-------------|---------|
| Regulatory capital (e.g., FRTB) | Portfolio | Market-wide stress scenarios |
| Single-name concentration | Underlying | "What if TSLA drops 30%?" |
| Specific instrument analysis | Position | "What if this OTM call becomes ITM?" |

### Risk Aggregation

After calculating the P&L for each scenario, the module aggregates results to provide key risk insights:

*   **Worst Case Scenario**: Identifies the scenario causing the largest loss.
*   **Scenario Comparison**: Side-by-side view of P&L across different regimes (e.g., Bull vs. Bear market).
*   **Greek Sensitivity**: Analysis of how hedging parameters (Delta, Gamma) change under stress. For example, a portfolio might be Delta-neutral now, but become significantly short Delta after a market drop due to negative Gamma.

### References

*   Basel Committee on Banking Supervision (2016). "Minimum Capital Requirements for Market Risk."
*   Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
*   Cont, R., & Deguest, R. (2013). "Stress Testing Banks." *International Monetary Fund Working Paper*.
