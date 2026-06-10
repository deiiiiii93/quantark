# Snowball Option Pricing via 2-Surface PDE Method

This document describes the theoretical method implemented in `SnowballPDE2S.py` for pricing Snowball options using a "Two-Surface" Partial Differential Equation (PDE) approach.

## 1. Overview

The core idea of the method is to solve for two separate price functions (surfaces) simultaneously:

1.  **$V_0(S, t)$**: The value of the option assuming the Knock-In (KI) event **has not** occurred yet.
2.  **$V_1(S, t)$**: The value of the option assuming the Knock-In (KI) event **has** already occurred.

The pricing engine evolves these two surfaces backward in time from maturity ($T$) to the present ($0$). The surfaces interact at specific observation dates or boundaries:
*   **Knock-In (KI) Event**: If the underlying price $S$ breaches the KI barrier, the value on the "Not Knocked-In" surface $V_0$ jumps to the value on the "Knocked-In" surface $V_1$.
*   **Knock-Out (KO) Event**: If $S$ breaches the KO barrier, both surfaces jump to the specified rebate/coupon value (the option terminates).

## 2. Mathematical Model

The underlying asset price $S$ is assumed to follow the Geometric Brownian Motion (Black-Scholes-Merton framework):

$$ dS_t = (r - q) S_t dt + \sigma S_t dW_t $$

Where:
*   $r$: Risk-free rate
*   $q$: Dividend yield
*   $\sigma$: Volatility

Inside the domain (between barriers and observation dates), both $V_0$ and $V_1$ satisfy the standard Black-Scholes PDE:

$$ \frac{\partial V}{\partial t} + (r - q) S \frac{\partial V}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - r V = 0 $$

## 3. Grid Discretization

The continuous problem is discretized using the Finite Difference Method (FDM).

### 3.1 Spatial Grid ($S$)
A **Non-Uniform Grid** is used to concentrate accuracy where it matters most. Grid points are clustered around:
*   Spot price ($S_0$)
*   Strike price ($K$)
*   Knock-In barrier ($B_{KI}$)
*   Knock-Out barrier ($B_{KO}$)

This is typically achieved using a coordinate transformation (e.g., Tavella-Randall) to map a uniform computational domain to the non-uniform physical domain.

### 3.2 Time Grid ($t$)
The time step is uniform ($\Delta t$), but the grid is constructed to ensure that all discrete **observation dates** (KI and KO dates) align exactly with time steps. This avoids interpolation errors across critical event dates.

## 4. Boundary and Terminal Conditions

### 4.1 Terminal Condition ($t = T$)
At maturity $T$, the payoffs are determined as follows:

*   **For $V_1(S, T)$ (Knocked-In):**
    The option behaves like a variation of a Put option (assuming standard Snowball).
    $$ V_1(S, T) = \text{Principal} + \text{Participation} \times \min\left(\frac{S_T - K}{S_0}, 0\right) \times N $$
    *Note: Partial or Full protection floors are applied here if defined.*

*   **For $V_0(S, T)$ (Not Knocked-In):**
    The option has survived without KI (and implicitly without KO, as we are at $T$). The payoff is typically the Principal plus a fixed Rebate/Coupon.
    $$ V_0(S, T) = \text{Principal} + \text{Rebate} $$

### 4.2 Spatial Boundary Conditions
*   **Lower Boundary ($S \to 0$):**
    *   $V_1(0, t)$: Discounted value of the worst-case payoff (e.g., deep ITM put).
    *   $V_0(0, t)$: Discounted value of principal (or similar, depending on product specs).
*   **Upper Boundary ($S \to \infty$):**
    *   Typically approximated by the KO Rebate or the intrinsic value, depending on whether $S_{max}$ is a KO barrier. In this implementation, it is often set to the KO Rebate value if $S \ge B_{KO}$.

## 5. Backward Induction Algorithm

The solving process iterates from $j = N_t$ (Maturity) down to $0$ (Today).

**Step 1: Evolve PDEs**
Solve the PDE for one time step $\Delta t$ for both $V_0$ and $V_1$ using the **Crank-Nicolson** scheme. This requires solving a tridiagonal linear system:
$$ \mathbf{M}_2 \mathbf{V}^{j} = \mathbf{M}_1 \mathbf{V}^{j+1} $$

**Step 2: Apply Knock-Out (KO) Jumps**
If $t_j$ is a Knock-Out observation date:
For grid points $S_k$ where KO is triggered (e.g., $S_k \ge B_{KO}$):
$$ V_0(S_k, t_j) = \text{KO\_Payoff}(t_j) $$
$$ V_1(S_k, t_j) = \text{KO\_Payoff}(t_j) $$
*(KO takes precedence over everything).*

**Step 3: Apply Knock-In (KI) Jumps**
If $t_j$ is a Knock-In observation date (or if KI is continuous):
For grid points $S_k$ where KI is triggered (e.g., $S_k \le B_{KI}$):
$$ V_0(S_k, t_j) = V_1(S_k, t_j) $$
*Explanation: If the price hits the KI barrier, the status changes from "Not Knocked-In" to "Knocked-In". Therefore, the value of being "Not Knocked-In" at that precise spot becomes the value of being "Knocked-In".*

## 6. Calculation of Final Price

After stepping back to $t=0$:
*   If the contract is already Knocked-In historically:
    $$ \text{Price} = V_1(S_{spot}, 0) $$
*   Otherwise:
    $$ \text{Price} = V_0(S_{spot}, 0) $$

This value is obtained via spline interpolation on the spatial grid if the current spot $S_{spot}$ does not align perfectly with a grid point.

## 7. Extensions in Code
The `SnowballPDE2S.py` also handles:
*   **Discrete vs Continuous Observations**: By adjusting when the "Jumps" (Step 2 and 3) are applied.
*   **Rebates**: Handled via boundary conditions and terminal payoffs.
