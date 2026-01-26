# Dynamic Scenario Module: Theory and Methodology

## Overview

The Dynamic Scenario module extends risk analysis into the temporal dimension. Unlike static stress tests which are instantaneous, dynamic scenarios simulate the **evolution** of market factors over a multi-day horizon. This allows for the analysis of path-dependent risks, the effectiveness of dynamic hedging strategies, and the impact of time decay (Theta).

## Relationship to Other Risk Modules

The Dynamic Scenario module is one of three complementary risk analysis frameworks in QuantArk, each addressing a different dimension of risk:

| Module | Time Dimension | Data Source | Risk Question Answered |
|--------|----------------|-------------|------------------------|
| **Stress Test** | Instantaneous (t=0) | Hypothetical shocks | "What is my loss if X happens now?" |
| **Dynamic Scenario** | Multi-day (t=0→T) | Hypothetical paths | "How does risk evolve under scenario X over T days?" |
| **Backtest** | Historical | Actual market data | "How would this strategy have performed historically?" |

The Dynamic Scenario module is unique in its **forward-looking path-dependence**: it models how portfolio risk evolves over time under hypothetical scenarios. Unlike Stress Test (instantaneous snapshot) or Backtest (historical replay), it can explore hypothetical futures and analyze path-dependent effects like gamma bleed and theta burn.

## Theory of Path Generation

A dynamic scenario is defined by a sequence of market states over $T$ days. The module constructs these paths using two primary approaches:

### 1. Parametric Paths

These are synthetic paths constructed to test specific structural risks.

#### Geometric Brownian Motion (GBM) Paths

For equity spot dynamics with trend, the parametric form follows the GBM solution:

$$
S_t = S_0 \exp\left(\mu t + \sigma W_t\right)
$$

where:
- $\mu$ is the drift (expected return)
- $\sigma$ is the volatility
- $W_t$ is a Wiener process (Brownian motion)

In discrete time with daily steps $\Delta t = 1$:

$$
S_{t+1} = S_t \exp\left(\mu - \frac{1}{2}\sigma^2 + \sigma \sqrt{\Delta t} \, Z_t\right)
$$

where $Z_t \sim \mathcal{N}(0,1)$ are independent standard normals.

#### Mean-Reverting Paths (Ornstein-Uhlenbeck)

For volatilities and rates, mean reversion is often more appropriate:

$$
dX_t = \kappa(\theta - X_t)dt + \sigma dW_t
$$

The discrete form (Euler discretization):

$$
X_{t+1} = X_t + \kappa(\theta - X_t)\Delta t + \sigma\sqrt{\Delta t}\,Z_t
$$

where:
- $\theta$ is the long-run mean
- $\kappa$ is the mean reversion speed
- $\sigma$ is the volatility of the process

#### Volatility Regime Switching

For modeling vol-of-vol or sudden regime changes, the dynamics can be modeled as:

$$
\sigma_t = \sigma_0 \cdot \exp\left(\alpha \cdot \mathbb{I}_{t > t_{\text{switch}}}\right)
$$

where $\mathbb{I}$ is the indicator function for the regime switch time.

### 2. Fixed Income Path Dynamics

For yield curve evolution, the **Heath-Jarrow-Morton (HJM)** framework provides the theoretical foundation:

$$
df(t,T) = \alpha(t,T)dt + \sigma(t,T)dW_t
$$

where $f(t,T)$ is the instantaneous forward rate at time $t$ for maturity $T$.

In practice, scenarios often focus on the first three **principal components** of the yield curve:

1. **Level**: Parallel shift (explains ~70-80% of variance)
2. **Slope**: Steepening/flattening (explains ~10-15% of variance)
3. **Curvature**: Twist/hump (explains ~5-10% of variance)

### 3. Historical Bootstrapping (Future Scope)
Constructing paths by sampling from historical returns to preserve empirical correlation structures between assets and volatilities.

## Modeled Risk Factors
Dynamic scenarios allow for the evolution of complex market factors:

*   **Spot Dynamics**: Modeling drift (trend) and diffusion (volatility) of the asset price.
*   **Volatility Surface**: Evolution of the implied volatility surface over time.
    *   *Sticky Strike vs. Sticky Delta*: Handling how the skew moves relative to spot.
    *   *Vol-of-Vol*: Fluctuations in the volatility level itself.
*   **Yield Curve Principal Components**: Modeling the independent movements of Level, Slope, and Curvature for rates.
*   **Term Structure**: Evolution of forward rates and the "roll down" effect on bond prices.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A 3D scientific surface plot of a 'Volatility Surface'. Use visual indicators like semi-transparent 'ghost' surfaces or directional arrows to depict the surface rippling and changing shape over time (Vol-of-Vol). Style: High-tech 3D data visualization, blue and purple gradient."

## Path Representation in QuantArk (DayPath / DayStep)

In QuantArk, a dynamic scenario is represented as a **DayPath**: an ordered list of **DaySteps**, where each DayStep contains one or more **ParameterChanges** (spot/vol/rate/dividend, etc.). Each ParameterChange uses a stress type:

* **Percentage** (compound): \(X_t = X_{t-1}(1+\epsilon_t)\)
* **Absolute** (additive): \(X_t = X_{t-1}+\delta_t\)
* **Value** (override): \(X_t = \bar{X}_t\)

This makes the scenario definition deterministic and easy to audit: “what changed on each day, and why?”

## Dynamic Hedging Simulation

A key feature of this module is the ability to simulate **active portfolio management** along the path.

For each day $t$ in the scenario:
1.  **Advance Time**: Move valuation date to $t$. Time to maturity decreases ($T-t$).
2.  **Update Market**: Apply the scenario's defined market parameters for day $t$.
3.  **Evaluate Portfolio**: Calculate P&L and Risk Metrics (Greeks).
4.  **Hedge Logic**:
    *   Check Hedging Triggers (e.g., Is Delta deviation > limit?).
    *   If triggered, execute trades to rebalance to target.
    *   Record transaction costs and trade details.

Conceptually, the engine loop is:

1. Apply DayStep market changes to a cloned portfolio’s pricing environments
2. Advance valuation date (so time-to-maturity decreases)
3. Reprice portfolio and compute risk (Greeks / DV01)
4. If hedging enabled: run strategy trigger + sizing, execute hedge trades, then reprice again
5. Record day result (value, P&L, risk, trades, costs)

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A flowchart diagram of the Dynamic Scenario Simulation Loop. Visual flow: Start -> [Update Market Data (t)] -> [Reprice Portfolio] -> [Check Hedge Triggers] -> [Execute Trades] -> [Record Results] -> [Advance Time (t+1)] -> Loop back. Style: Technical process diagram, circular cycle, modern UI elements."

### Path Dependency Analysis
This framework captures risks that static analysis misses. The following are formal definitions of key path-dependent risks:

#### Gamma Bleed (Gamma P&L)

**Gamma bleed** refers to the cumulative P&L from gamma (convexity) as the underlying moves along the realized path. For a delta-hedged position:

$$
\text{Gamma P\&L} = \sum_{t=1}^{T} \frac{1}{2} \Gamma_{t-1} (\Delta S_t)^2
$$

In a **choppy market** (high realized volatility), gamma is positive and generates profit. In a **trending market**, the hedge fails to capture the full directional move.

The distinction between **expected gamma P&L** (based on implied volatility) and **realized gamma P&L** (based on actual path) is a key source of hedging P&L:

$$
\mathbb{E}[\text{Gamma P\&L}] \approx \frac{1}{2}\Gamma S^2 \sigma_{\text{imp}}^2 \Delta t
$$

$$
\text{Realized Gamma P\&L} = \frac{1}{2}\Gamma S^2 \sigma_{\text{realized}}^2 \Delta t
$$

#### Theta Burn (Time Decay)

**Theta burn** is the deterministic erosion of option value due to passage of time, independent of market movement:

$$
\Theta_t = \frac{\partial V}{\partial t}\bigg|_{S_t, \sigma_t}
$$

For a European call under Black-Scholes:

$$
\Theta = -\frac{S \phi(d_1)\sigma}{2\sqrt{T-t}} - rK e^{-r(T-t)} \Phi(d_2)
$$

The cumulative theta burn over the scenario horizon:

$$
\text{Total Theta} = \sum_{t=1}^{T} \Theta_t \cdot \Delta t
$$

Theta is typically **negative for long options** (time works against you) and **positive for short options** (time works in your favor).

#### Liquidity Holes

A **liquidity hole** occurs when market movement is faster than the rebalancing frequency. If the underlying gaps by more than the hedge threshold:

$$
|\Delta S_t| > \text{Threshold} \implies \text{Unhedged exposure}
$$

The expected loss from liquidity holes scales with:

$$
\mathbb{E}[\text{Gap Loss}] \approx \frac{1}{2}\Gamma \cdot \mathbb{E}[(\Delta S)^2 \cdot \mathbb{I}_{|\Delta S| > \text{Threshold}}]
$$

This risk is particularly acute for:
- Short gamma positions (selling options)
- Short-dated options (high gamma near expiry)
- Illiquid underlyings (large gaps possible)

## Fixed Income Dynamics

For Fixed Income portfolios, dynamic scenarios involve the evolution of the entire yield curve and key rates.

*   **Parallel Shifts**: Entire curve moves up/down over the period (e.g., Fed hiking cycle).
*   **Twists**: Short-end moves faster than long-end (Bear Flattener / Bull Steepener).
*   **Roll-down**: As time passes, bonds "roll down" the yield curve, changing their yield even if the curve is static. The module accounts for this aging process.

The **roll-down return** is the return on a bond that rolls down the yield curve over the scenario horizon. It is calculated as the difference between the final and initial bond prices, adjusted for interest payments received during the period.

**[Image Placeholder]**
> **Prompt for Nanobanana**: /diagram prompt: "A 3D wireframe surface plot representing a Yield Curve evolving over time. Axes: X='Tenor' (1M-30Y), Y='Time' (Day 1-30), Z='Interest Rate'. The mesh surface should visibly twist and shift along the Time axis to demonstrate curve evolution. Style: Technical scientific plot, clean wireframe lines."

## Model Coverage (QuantArk Implementation)

### Engine Architecture

Dynamic scenario engines are implemented for both **equity** and **fixed income** portfolios:

* **Equity engine** (`DynamicScenarioEngine`): Applies day-by-day changes to spot, flat volatility, flat rate, dividend yield (clamped at 0.0 if negative), and basis yield for futures; optionally runs a backtest strategy (e.g., delta-neutral) and charges transaction costs.
* **FI engine** (`FIDynamicScenarioEngine`): Applies day-by-day rate curve changes (including parallel and simple twist components via the path library); tracks DV01/duration/convexity and optionally performs DV01-based hedging.

### Scenario Construction

The module provides two primary interfaces for constructing scenarios:

* **PathBuilder**: A fluent API for constructing custom parametric paths. Methods include `spot_trend()`, `vol_decay()`, `rate_parallel_shift()`, `rate_steepener()`, etc.
* **PathLibrary / FIPathLibrary**: Predefined scenario factories including:
  - Equity: `consecutive_rally()`, `v_shaped_recovery()`, `volatility_spike_decay()`, `gradual_crash()`
  - Fixed Income: `parallel_shift()`, `steepener()`, `flattener()`, `rate_hike_cycle()`

### Hedging Integration

The module integrates with the backtest module's strategy framework:
- Strategies from `backtest/strategy/` can be used in dynamic scenarios
- The engine calls strategy lifecycle methods: `on_step()`, `should_hedge()`, `calculate_hedge_size()`, `on_hedge_executed()`
- Transaction costs are tracked and deducted from P&L

### Path Representation

In QuantArk, a dynamic scenario is represented as a **DayPath**: an ordered list of **DaySteps**, where each DayStep contains one or more **ParameterChanges** (spot/vol/rate/dividend, etc.). Each ParameterChange uses a stress type:

* **Percentage** (compound): \(X_t = X_{t-1}(1+\epsilon_t)\)
* **Absolute** (additive): \(X_t = X_{t-1}+\delta_t\)
* **Value** (override): \(X_t = \bar{X}_t\)

This makes the scenario definition deterministic and easy to audit: "what changed on each day, and why?"

### Extension Points

Large-scope "future" items (e.g., historical bootstrapping, full surface dynamics, sticky-delta vs sticky-strike rules) are best viewed as extension points: the current implementation is a deterministic scenario runner with explicit parameter updates and full repricing each day.

## Applications

1.  **Forward-Looking Risk**: Estimating P&L for next quarter under a "Soft Landing" vs "Recession" scenario.
2.  **Strategy Validation**: Proving that a hedging algorithm works not just instantaneously, but sustains performance over a volatile week.
3.  **Margin Simulation**: Estimating future margin calls based on portfolio evolution.

## References

*   Heath, D., Jarrow, R., & Morton, A. (1992). "Bond Pricing and the Term Structure of Interest Rates: A New Methodology for Contingent Claims Valuation." *Econometrica*.
*   Hull, J., & White, A. (1990). "Pricing Interest Rate Derivative Securities." *The Review of Financial Studies*.
*   Litterman, R., & Scheinkman, J. (1991). "Common Factors Affecting Bond Returns." *Journal of Fixed Income*.
