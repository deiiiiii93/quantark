# Black '76 Model for European Bond Options

## Overview
The Black '76 model is used to price European options on bond prices. It models the forward bond price as following a lognormal process.

## Formulas

### Option Price
```
Call = D(T) * [F * N(d1) - K * N(d2)]
Put  = D(T) * [K * N(-d2) - F * N(-d1)]
```

Where:
- D(T) = discount factor to option expiry
- F = forward bond price
- K = strike price
- N(x) = cumulative standard normal distribution

### d1 and d2
```
d1 = [ln(F/K) + σ²T/2] / (σ√T)
d2 = d1 - σ√T
```

Where:
- σ = volatility of bond price
- T = time to option expiry

## Implementation
- `BlackBondOptionEngine`: Main pricing engine
  - `price()`: Returns option price
  - `price_with_details()`: Returns detailed results (price, forward, Greeks)
  - `implied_volatility()`: Newton-Raphson solver for implied vol

## Forward Bond Price
Forward bond price is calculated as:
```
F = (Dirty Price - PV of coupons before expiry) * exp(r*T)
```

For clean price strikes, accrued interest at expiry is subtracted.