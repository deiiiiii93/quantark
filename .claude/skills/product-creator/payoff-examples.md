# Payoff Examples

Reference payoff specifications for common product types.

## Vanilla Options

### European Call
```
State: Single terminal payoff
Formula: max(S_T - K, 0)
Variables:
  - S_T: Spot at maturity
  - K: Strike
Path dependence: None
```

### European Put
```
State: Single terminal payoff
Formula: max(K - S_T, 0)
Path dependence: None
```

### American Option
```
States:
  1. Early exercise: max(S_t - K, 0) at any t <= T [call]
  2. Terminal: max(S_T - K, 0)
Optimal exercise: When intrinsic > time value
Path dependence: Stopping time decision
```

## Barrier Options

### Down-and-Out Call
```
States:
  1. Knocked out: 0 (if min(S_t) <= B at any observation)
  2. Not knocked out: max(S_T - K, 0)
Variables:
  - B: Down barrier (B < S_0)
  - K: Strike
Path dependence: Yes (barrier monitoring)
Observation: DISCRETE or CONTINUOUS
```

### Down-and-In Put
```
States:
  1. Not knocked in: 0 (if min(S_t) > B always)
  2. Knocked in: max(K - S_T, 0)
Variables:
  - B: Down barrier
Path dependence: Yes
```

### Up-and-Out Call
```
States:
  1. Knocked out: 0 (if max(S_t) >= B)
  2. Not knocked out: max(S_T - K, 0)
Variables:
  - B: Up barrier (B > S_0)
```

### Double Barrier Knock-Out
```
States:
  1. Knocked out: 0 (if S_t <= L or S_t >= U at any observation)
  2. Survived: max(S_T - K, 0) [for call]
Variables:
  - L: Lower barrier
  - U: Upper barrier
```

## Digital/Binary Options

### Cash-or-Nothing Call
```
States:
  1. ITM: Cash amount Q (if S_T > K)
  2. OTM: 0 (if S_T <= K)
Formula: Q * 1_{S_T > K}
```

### Asset-or-Nothing Call
```
States:
  1. ITM: S_T (if S_T > K)
  2. OTM: 0
Formula: S_T * 1_{S_T > K}
```

### One-Touch
```
States:
  1. Touched: Rebate R (if S_t >= B at any t)
  2. Never touched: 0
Path dependence: Yes (first passage time)
Payment timing: At touch or at maturity
```

### Double One-Touch
```
States:
  1. Touched: Rebate R (if S_t <= L or S_t >= U)
  2. Never touched: 0
Variables:
  - L: Lower barrier
  - U: Upper barrier
```

## Autocallable/Structured Products

### Standard Snowball
```
States:
  1. Knock-out at t_i: Principal + KO_rate * t_i (if S_{t_i} >= KO at observation i)
  2. V0 (never KO, never KI): Principal + Rebate (if never KO and never KI)
  3. V1 (never KO, KI happened): Principal + Participation * min(S_T/K - 1, 0)

Variables:
  - KO: Knock-out barrier (typically 103%)
  - KI: Knock-in barrier (typically 75%)
  - KO_rate: Annualized coupon rate
  - Rebate: Terminal rebate if no KO/KI
  - Participation: Downside participation after KI

Path dependence:
  - Discrete KO observations (monthly)
  - Continuous or discrete KI monitoring

Observation dates: [t_1, t_2, ..., t_n] year fractions
```

### Step-Down Snowball
```
Same as standard, but:
  - KO barrier decreases each period
  - KO_i = KO_initial - stepdown * i
```

### Parachute Snowball
```
Same as standard, but:
  - Last KO barrier = KI barrier
  - Guarantees exit if not knocked in
```

### Airbag Snowball
```
Same as standard, but V1 payoff modified:
  - If S_T < Airbag barrier:
    - Use reduced participation rate
    - May use different strike
```

### Reverse Snowball
```
Same as standard, but direction reversed:
  - KO: Down barrier (e.g., 97%)
  - KI: Up barrier (e.g., 125%)
  - Payoff: Participation * max(1 - S_T/K, 0) when KI
```

## Asian Options

### Average Price Call
```
State: Single terminal payoff
Formula: max(A - K, 0)
where A = (1/n) * sum(S_{t_i}) [arithmetic average]
or A = (prod(S_{t_i}))^(1/n) [geometric average]
Path dependence: Yes (averaging)
Averaging: Discrete or continuous
```

### Average Strike Call
```
Formula: max(S_T - A, 0)
where A is the average price
```

## Lookback Options

### Floating Strike Lookback Call
```
Formula: S_T - min(S_t for t in [0,T])
Payoff: Buy at lowest price, sell at final price
Path dependence: Yes (running minimum)
```

### Fixed Strike Lookback Call
```
Formula: max(max(S_t) - K, 0)
Payoff: Option on maximum realized price
```

## Spread/Multi-Asset

### Spread Option
```
Formula: max(S1_T - S2_T - K, 0)
Variables:
  - S1: First asset
  - S2: Second asset
  - K: Strike (spread threshold)
```

### Best-of Option
```
Formula: max(max(S1_T, S2_T, ...) - K, 0)
Payoff on best performing asset
```

### Worst-of Option
```
Formula: max(min(S1_T, S2_T, ...) - K, 0)
Payoff on worst performing asset
```

## Bond Products

### Fixed Rate Bond
```
Cashflows:
  - Coupons: C_i = Notional * Coupon_rate * DayCount(t_{i-1}, t_i) at t_i
  - Principal: Notional at T
Price: PV of all cashflows
```

### Floating Rate Note
```
Cashflows:
  - Coupons: C_i = Notional * (Index_rate + Spread) * DayCount at t_i
  - Principal: Notional at T
Index resets: At each accrual start date
```

### Callable Bond
```
Same as fixed bond, but:
  - Issuer can redeem at call_price on call_dates
  - Optimal call: When bond value > call_price
```

### Convertible Bond
```
Same as fixed bond, but:
  - Holder can convert to Conversion_ratio * shares
  - Conversion value: Conversion_ratio * Stock_price
  - Optimal conversion: When conversion value > bond value
```

## Interest Rate Swaps

### Vanilla IRS (Payer)
```
Legs:
  1. Fixed leg: Pay Notional * Fixed_rate * DayCount at each period
  2. Floating leg: Receive Notional * Float_rate * DayCount at each period
Net: sum(Float - Fixed) discounted
```

### Basis Swap
```
Legs:
  1. Pay: Index1 + Spread1
  2. Receive: Index2 + Spread2
Net: Spread differential present value
```
