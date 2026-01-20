# Risk Metrics Reference

Comprehensive reference for risk metrics (Greeks) in QuantArk.

## Table of Contents

1. [Black-Scholes Greeks (Equity)](#black-scholes-greeks-equity)
2. [Black '76 Greeks (Bond Options)](#black-76-greeks-bond-options)
3. [Bond Risk Metrics](#bond-risk-metrics)
4. [Numerical Methods](#numerical-methods)
5. [Interpretation Guide](#interpretation-guide)

---

## Black-Scholes Greeks (Equity)

### Notation

| Symbol | Description |
|--------|-------------|
| S | Spot price |
| K | Strike price |
| T | Time to maturity (years) |
| r | Risk-free rate |
| q | Dividend yield |
| σ | Volatility |
| N(x) | Standard normal CDF |
| n(x) | Standard normal PDF |

### d1 and d2

```
d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

### Delta (Δ)

**Definition:** Rate of change of option price with respect to spot price.

**Formula:**
```
Call Delta = e^(-qT) × N(d1)
Put Delta  = -e^(-qT) × N(-d1) = e^(-qT) × (N(d1) - 1)
```

**Range:**
- Call: 0 to 1
- Put: -1 to 0

**QuantArk Implementation:**
```python
# From GreeksCalculator.calculate_analytical_greeks()
if product.is_call():
    delta = discount_div * N_d1
else:
    delta = -discount_div * stats.norm.cdf(-d1)
```

---

### Gamma (Γ)

**Definition:** Rate of change of delta with respect to spot price.

**Formula:**
```
Gamma = e^(-qT) × n(d1) / (S × σ × √T)
```

**Note:** Gamma is the same for calls and puts.

**QuantArk Implementation:**
```python
gamma = discount_div * n_d1 / (S * sigma * sqrt_T)
```

---

### Vega (ν)

**Definition:** Rate of change of option price with respect to volatility.

**Formula:**
```
Vega = S × e^(-qT) × √T × n(d1)
```

**Note:**
- Vega is the same for calls and puts
- QuantArk returns vega per 1% change (divided by 100)

**QuantArk Implementation:**
```python
vega = S * discount_div * n_d1 * sqrt_T / 100  # Per 1% vol change
```

---

### Theta (Θ)

**Definition:** Rate of change of option price with respect to time (time decay).

**Formula:**
```
Call Theta = -[S × σ × e^(-qT) × n(d1)] / (2√T) - r × K × e^(-rT) × N(d2) + q × S × e^(-qT) × N(d1)
Put Theta  = -[S × σ × e^(-qT) × n(d1)] / (2√T) + r × K × e^(-rT) × N(-d2) - q × S × e^(-qT) × N(-d1)
```

**Note:** QuantArk returns theta per day (divided by 365)

**QuantArk Implementation:**
```python
term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
if product.is_call():
    term2 = -r * K * discount_rf * N_d2
    term3 = q * S * discount_div * N_d1
else:
    term2 = r * K * discount_rf * stats.norm.cdf(-d2)
    term3 = -q * S * discount_div * stats.norm.cdf(-d1)
theta = (term1 + term2 + term3) / 365  # Per day
```

---

### Rho (ρ)

**Definition:** Rate of change of option price with respect to interest rate.

**Formula:**
```
Call Rho = K × T × e^(-rT) × N(d2)
Put Rho  = -K × T × e^(-rT) × N(-d2)
```

**Note:** QuantArk returns rho per 1% change (divided by 100)

**QuantArk Implementation:**
```python
if product.is_call():
    rho = K * T * discount_rf * N_d2 / 100
else:
    rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
```

---

## Black '76 Greeks (Bond Options)

### Notation

| Symbol | Description |
|--------|-------------|
| F | Forward bond price |
| K | Strike price |
| T | Time to option expiry |
| r | Risk-free rate |
| σ | Volatility |
| D | Discount factor = e^(-rT) |

### d1 and d2 (Black '76)

```
d1 = [ln(F/K) + (σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

### Delta (Bond Option)

**Formula:**
```
Call Delta = D × N(d1)
Put Delta  = D × (N(d1) - 1)
```

**QuantArk Implementation:**
```python
# From BondGreeksCalculator.calculate_analytical_greeks()
if option.is_call():
    delta = D * N_d1
else:
    delta = D * (N_d1 - 1)
greeks["delta"] = delta * option.notional
```

---

### Gamma (Bond Option)

**Formula:**
```
Gamma = D × n(d1) / (F × σ × √T)
```

---

### Vega (Bond Option)

**Formula:**
```
Vega = D × F × √T × n(d1)
```

Note: Returned per 1% volatility change.

---

## Bond Risk Metrics

### Duration

**Macaulay Duration:**
```
D_mac = Σ[t × w_t] / Price

where w_t = PV(CF_t) / Price
```

**Modified Duration:**
```
D_mod = D_mac / (1 + y/m)

where y = yield, m = compounding frequency
```

**Interpretation:** Approximate percentage price change for 1% yield change.

---

### Convexity

**Formula:**
```
Convexity = Σ[t × (t+1) × PV(CF_t)] / (Price × (1+y)²)
```

**Second-order price approximation:**
```
ΔP/P ≈ -D_mod × Δy + (1/2) × Convexity × (Δy)²
```

---

### DV01 (Dollar Value of 01)

**Definition:** Dollar change in price for 1 basis point (0.01%) yield change.

**Formula:**
```
DV01 = -D_mod × Price × 0.0001
```

Or numerically (central difference):
```
DV01 = [P(y - 0.0001) - P(y + 0.0001)] / 2
```

**QuantArk Implementation:**
```python
# From BondGreeksCalculator._calculate_dv01_fdm()
env_up.rate_curve = FlatRateCurve(rate=base_rate + 0.0001)
price_up = engine.price(option, volatility, valuation_date)
dv01 = base_price - price_up  # Price decrease for rate increase
```

---

## Numerical Methods

### Central Difference (Finite Difference)

**First Derivative (Delta):**
```
δV/δS ≈ [V(S + ΔS) - V(S - ΔS)] / (2ΔS)
```

**Second Derivative (Gamma):**
```
δ²V/δS² ≈ [V(S + ΔS) - 2V(S) + V(S - ΔS)] / (ΔS)²
```

### QuantArk Bump Sizes

| Greek | Bump Type | Default Size |
|-------|-----------|--------------|
| Delta | Relative spot bump | 1% (0.01) |
| Gamma | Relative spot bump | 1% (0.01) |
| Vega | Absolute vol bump | 1% (0.01) |
| Theta | Time bump | 1 day |
| Rho | Absolute rate bump | 1% (0.01) |
| DV01 | Absolute rate bump | 1bp (0.0001) |

### QuantArk Implementation (Numerical Greeks)

```python
# From GreeksCalculator.calculate_numerical_greeks()
bump = self.params.bump_size  # Default 0.01

# Delta: spot bump
env_up.spot_quote.spot *= 1 + bump
env_down.spot_quote.spot *= 1 - bump
delta = (price_up - price_down) / (2 * S * bump)

# Gamma: spot bump (same as delta)
gamma = (price_up - 2*base_price + price_down) / (S * bump)²

# Vega: 1% absolute vol bump
env_up.vol_surface = FlatVolSurface(current_vol + 0.01)
vega = price_up - base_price

# Theta: 1 day time bump
product_theta.maturity -= 1/365
theta = price_theta - base_price

# Rho: 1% absolute rate bump
env_up.rate_curve = FlatRateCurve(current_rate + 0.01)
rho = price_up - base_price
```

---

## Interpretation Guide

### Delta Interpretation

| Delta Value | Meaning | Position |
|-------------|---------|----------|
| 0.50 | ATM call | Neutral |
| 0.80 | ITM call | Bullish |
| 0.20 | OTM call | Speculative |
| -0.50 | ATM put | Neutral |
| -0.80 | ITM put | Bearish |

**Hedging:** To hedge delta, take opposite position in underlying.
```
Hedge ratio = -Delta × Notional / Underlying_Price
```

---

### Gamma Interpretation

| Gamma Behavior | Meaning |
|----------------|---------|
| High gamma | Price moves accelerate |
| Low gamma | Price moves linear |
| Gamma highest ATM | Maximum sensitivity at strike |
| Gamma → 0 far ITM/OTM | Little curvature |

**Gamma Hedging:** Requires options (underlying has zero gamma).

---

### Vega Interpretation

| Scenario | Vega Impact |
|----------|-------------|
| Long options | Positive vega (benefit from vol increase) |
| Short options | Negative vega (hurt by vol increase) |
| ATM options | Highest vega |
| ITM/OTM options | Lower vega |

**Vega Risk:** 1 vega point = $1 P&L per 1% vol change.

---

### Theta Interpretation

| Scenario | Theta Behavior |
|----------|----------------|
| Long options | Negative theta (time decay) |
| Short options | Positive theta (earn time decay) |
| ATM options | Highest theta decay |
| Near expiry | Theta accelerates |

**Theta-Gamma Tradeoff:** High gamma positions have high theta cost.

---

### Rho Interpretation

| Scenario | Rho Behavior |
|----------|--------------|
| Long calls | Positive rho (benefit from rate increase) |
| Long puts | Negative rho (hurt by rate increase) |
| Longer maturity | Higher rho sensitivity |

**Note:** Rho is typically the smallest Greek for equity options.

---

## Greek Sensitivities Table

| Greek | Spot ↑ | Vol ↑ | Time ↓ | Rate ↑ |
|-------|--------|-------|--------|--------|
| Delta (Call) | ↑ | ~ | ↓ ATM | ↑ |
| Delta (Put) | ↑ (less negative) | ~ | ↓ ATM | ↓ |
| Gamma | ↓ if ITM, ↑ if OTM | ↑ | ↑ near expiry | ~ |
| Vega | ~ | ~ | ↓ | ~ |
| Theta | ~ | ↑ | ~ | ~ |

---

## Edge Cases

### Near Expiry (T → 0)

| Greek | Behavior |
|-------|----------|
| Delta | → 1 (ITM call), 0 (OTM call), -1 (ITM put), 0 (OTM put) |
| Gamma | → ∞ at strike, 0 elsewhere |
| Vega | → 0 |
| Theta | → -∞ ATM, 0 otherwise |

### Deep ITM

| Greek | Behavior |
|-------|----------|
| Delta | → 1 (call), -1 (put) |
| Gamma | → 0 |
| Vega | → 0 |

### Deep OTM

| Greek | Behavior |
|-------|----------|
| Delta | → 0 |
| Gamma | → 0 |
| Vega | → 0 |

---

## Code Examples

### Calculate All Greeks (Equity)

```python
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams

# Setup
option = EuropeanVanillaOption(strike=100, option_type=OptionType.CALL, maturity=1.0)
engine = BlackScholesEngine()
calculator = GreeksCalculator(params=EngineParams(bump_size=0.01))

# Analytical Greeks (European only)
analytical = calculator.calculate_analytical_greeks(option, pricing_env)

# Numerical Greeks (any product)
numerical = calculator.calculate_numerical_greeks(option, pricing_env, engine)

# Compare
comparison = calculator.compare_greeks(analytical, numerical)
```

### Calculate Bond Sensitivities

```python
from asset.bond.riskmeasures import BondGreeksCalculator

calculator = BondGreeksCalculator(bump_size=0.01)

# Full Greeks
greeks = calculator.calculate_analytical_greeks(bond_option, pricing_env)

# Bond-specific sensitivities
sensitivities = calculator.calculate_bond_sensitivities(bond_option, pricing_env)
print(f"Option DV01: {sensitivities['option_dv01']:.4f}")
print(f"Option Duration: {sensitivities['option_duration']:.4f}")
print(f"Underlying DV01: {sensitivities['underlying_dv01']:.4f}")
```
