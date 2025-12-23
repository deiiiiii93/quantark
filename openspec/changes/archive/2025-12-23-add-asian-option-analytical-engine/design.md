# Design: Asian Option Analytical Engine

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 AsianOptionAnalyticalEngine                 │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Method    │  │   Router    │  │   Greeks    │         │
│  │  Selector   │──│   Logic     │──│ Calculator  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Method Implementations              │       │
│  │                                                  │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │       │
│  │  │ Geometric│ │ Turnbull │ │  Levy    │        │       │
│  │  │ (Exact)  │ │ Wakeman  │ │ Approx   │        │       │
│  │  └──────────┘ └──────────┘ └──────────┘        │       │
│  │  ┌──────────┐ ┌──────────┐                      │       │
│  │  │ Curran   │ │ Discrete │                      │       │
│  │  │ Approx   │ │   HHM    │                      │       │
│  │  └──────────┘ └──────────┘                      │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Class Diagram

```python
class AsianOptionAnalyticalEngine(BaseEngine):
    """Analytical pricing engine for Asian options."""
    
    DEFAULT_METHOD: ClassVar[AsianAnalyticalMethod] = AsianAnalyticalMethod.TURNBULL_WAKEMAN
    
    def __init__(
        self,
        params: Optional[EngineParams] = None,
        method: Union[str, AsianAnalyticalMethod, tuple, None] = None,
    ):
        """
        Initialize engine with method selection.
        
        Args:
            params: Engine configuration
            method: Pricing method - can be:
                - EngineType.ANALYTICAL(AsianAnalyticalMethod.TURNBULL_WAKEMAN)
                - AsianAnalyticalMethod.TURNBULL_WAKEMAN
                - "turnbull_wakeman"
        """
    
    def price(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> float:
        """Price Asian option using selected analytical method."""
    
    def calculate_greeks(self, product: BaseEquityProduct, pricing_env: PricingEnvironment) -> Dict[str, float]:
        """Calculate option Greeks."""
    
    # Private method implementations
    def _price_geometric_continuous(self, ...) -> float: ...
    def _price_geometric_discrete(self, ...) -> float: ...
    def _price_turnbull_wakeman(self, ...) -> float: ...
    def _price_levy(self, ...) -> float: ...
    def _price_curran(self, ...) -> float: ...
    def _price_discrete_hhm(self, ...) -> float: ...
    def _price_floating_strike(self, ...) -> float: ...
```

## Method Selection Flow

```
                         ┌─────────────────┐
                         │  User Request   │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Method explicitly set?   │
                    └─────────────┬─────────────┘
                           Yes    │    No
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
         ┌───────────────────┐     ┌───────────────────────────┐
         │ Use explicit      │     │ Auto-select based on:     │
         │ method            │     │ - Geometric → KEMNA_VORST │
         └───────────────────┘     │ - Arithmetic→ TURNBULL_   │
                                   │              WAKEMAN      │
                                   └───────────────────────────┘
                                              │
                                              ▼
                                   ┌───────────────────────────┐
                                   │ Floating strike?          │
                                   └────────────┬──────────────┘
                                         Yes    │    No
                                   ┌────────────┴──────────────┐
                                   │                           │
                                   ▼                           ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │ Apply Henderson- │     │ Price directly   │
                        │ Wojakowski       │     └──────────────────┘
                        │ symmetry         │
                        └──────────────────┘
```

## Parameter Extraction

```python
def _extract_params(
    self,
    product: AsianOption,
    pricing_env: PricingEnvironment
) -> dict:
    """Extract and validate all pricing parameters."""
    
    S = pricing_env.spot  # Current spot price
    K = product.strike    # Strike price
    T = product.get_maturity(pricing_env)  # Time to maturity
    r = pricing_env.get_rate(T)            # Risk-free rate
    q = pricing_env.get_div_yield(T)       # Dividend yield
    b = r - q                               # Cost of carry
    sigma = pricing_env.get_vol(K, T)      # Volatility
    
    # Observation schedule
    past_prices, future_times, n_total = product.resolve_observations(pricing_env)
    n = n_total                            # Total observations
    m = len(past_prices)                   # Realized observations
    S_A = product.get_past_average(pricing_env) if m > 0 else 0.0  # Realized average
    t1 = future_times[0] if future_times else 0.0  # Time to first future fixing
    
    return {
        'S': S, 'K': K, 'T': T, 'r': r, 'q': q, 'b': b, 'sigma': sigma,
        'n': n, 'm': m, 'S_A': S_A, 't1': t1, 'future_times': future_times
    }
```

## In-Period Pricing Logic

When `m > 0` observations have already occurred:

```python
def _adjust_for_in_period(self, params: dict, is_call: bool) -> Tuple[float, float]:
    """
    Adjust strike and compute multiplier for in-period pricing.
    
    Returns:
        (adjusted_strike, price_multiplier)
    """
    n, m, X, S_A = params['n'], params['m'], params['K'], params['S_A']
    
    # Check if exercise is certain (for calls when avg > strike)
    if S_A > (n / m) * X:
        if is_call:
            # Certain exercise - return expected payoff
            return self._certain_exercise_value(params)
        else:
            # Put is worthless
            return 0.0, 0.0
    
    # Adjust strike
    X_adj = (n * X - m * S_A) / (n - m)
    multiplier = (n - m) / n
    
    return X_adj, multiplier
```

## Floating-Strike Transformation

Henderson-Wojakowski symmetry transforms floating-strike to fixed-strike:

```python
def _price_floating_strike(
    self,
    product: AsianOption,
    pricing_env: PricingEnvironment,
    params: dict
) -> float:
    """
    Price floating-strike Asian using symmetry.
    
    Floating call = Fixed put with:
        - r → r - b
        - b → -b
        - Strike = S (spot)
    """
    is_call = product.is_call()
    S, K = params['S'], params['K']
    r, b = params['r'], params['b']
    
    # Transform parameters
    r_new = r - b
    b_new = -b
    K_new = S  # Strike becomes current spot
    
    # Create transformed params
    params_transformed = params.copy()
    params_transformed['r'] = r_new
    params_transformed['b'] = b_new
    params_transformed['K'] = K_new
    
    # Price as opposite option type
    if is_call:
        # Floating call = Fixed put
        return self._price_fixed_put(params_transformed)
    else:
        # Floating put = Fixed call
        return self._price_fixed_call(params_transformed)
```

## Key Formulas

### Geometric Continuous (Kemna-Vorst)

```
σ_A = σ / √3
b_A = (b - σ²/6) / 2

Call = S × exp((b_A - r) × T) × N(d1) - K × exp(-r × T) × N(d2)
Put  = K × exp(-r × T) × N(-d2) - S × exp((b_A - r) × T) × N(-d1)

where:
    d1 = [ln(S/K) + (b_A + σ_A²/2) × T] / (σ_A × √T)
    d2 = d1 - σ_A × √T
```

### Turnbull-Wakeman (Arithmetic Approximation)

```
M₁ = [exp(b×T) - exp(b×t₁)] / [b × (T - t₁)]    (for b ≠ 0)
M₁ = 1                                           (for b = 0)

M₂ = 2×exp((2b+σ²)×T) / [(b+σ²)(2b+σ²)(T-t₁)²]
   + 2×exp((2b+σ²)×t₁) / [b×(T-t₁)²] × [1/(2b+σ²) - exp(b(T-t₁))/(b+σ²)]

b_A = ln(M₁) / T
σ_A = √[ln(M₂)/T - 2×b_A]

Then apply BSM formula with (b_A, σ_A)
```

### Discrete Arithmetic (Haug-Haug-Margrabe)

```
h = (T - t₁) / (n - 1)

E[A_T] = (S/n) × exp(b×t₁) × [1 - exp(b×h×n)] / [1 - exp(b×h)]

E[A_T²] = (S²/n²) × exp((2b+σ²)×t₁) × [
    [1 - exp((2b+σ²)×h×n)] / [1 - exp((2b+σ²)×h)]
    + 2/[1 - exp((b+σ²)×h)] × (
        [1 - exp(b×h×n)] / [1 - exp(b×h)]
        - [1 - exp((2b+σ²)×h×n)] / [1 - exp((2b+σ²)×h)]
    )
]

σ_A = √[(ln(E[A_T²]) - 2×ln(E[A_T])) / T]
F_A = E[A_T]

Call = exp(-r×T) × [F_A × N(d1) - X × N(d2)]
Put  = exp(-r×T) × [X × N(-d2) - F_A × N(-d1)]
```

## Numerical Stability Considerations

1. **b ≈ 0 handling**: Use L'Hôpital's rule-derived formulas when |b| < 1e-10
2. **Near-expiry**: Return intrinsic value when T < 1e-10
3. **Extreme moneyness**: Cap |ln(S/K)| to prevent overflow
4. **Moment calculation**: Use log-space arithmetic to prevent overflow in M₂
5. **Single fixing remaining**: Use adjusted BSM formula directly

## Greeks Calculation

### Analytical Greeks (where available)

For geometric average options, Greeks are available in closed form:
- Delta = ∂V/∂S using chain rule through d1
- Gamma = ∂²V/∂S² 
- Vega = ∂V/∂σ_A × ∂σ_A/∂σ

### Numerical Greeks (fallback)

For arithmetic approximations, use finite differences:
```python
bump = 0.01  # 1% bump
delta = (price(S*(1+bump)) - price(S*(1-bump))) / (2*S*bump)
gamma = (price(S*(1+bump)) - 2*price(S) + price(S*(1-bump))) / (S*bump)²
vega = (price(σ+0.01) - price(σ-0.01)) / 0.02
```

## Error Handling

```python
# Product type validation
if not isinstance(product, AsianOption):
    raise PricingError(
        f"AsianOptionAnalyticalEngine requires AsianOption, got {type(product).__name__}"
    )

# Parameter validation
if S <= 0:
    raise ValidationError(f"Spot price must be positive, got {S}")
if sigma <= 0:
    raise ValidationError(f"Volatility must be positive, got {sigma}")

# Method compatibility
if product.is_geometric() and method not in [KEMNA_VORST]:
    # Auto-upgrade to geometric method
    method = AsianAnalyticalMethod.KEMNA_VORST
    
if product.is_arithmetic() and method == KEMNA_VORST:
    raise ValidationError("KEMNA_VORST only applies to geometric averaging")

# Levy limitation
if method == LEVY and is_zero(b):
    raise ValidationError("Levy approximation does not support b=0; use TURNBULL_WAKEMAN")
```

## Test Cases from Literature

### Geometric Put (Haug Example)
```
S=80, K=85, T=0.25, r=0.05, b=0.08, σ=0.20
Expected: p = 4.6922
```

### Turnbull-Wakeman Arithmetic Put (Haug Example)
```
S=90, S_A=88, K=95, t1=0, T=0.25, T2=0.25, r=0.07, b=0.02, σ=0.25
Expected: p = 5.6093
```

### Levy Currency Option (Haug Example)
```
S=6.80, S_A=6.80, K=6.90, T=T2=0.5, r=0.07, b=-0.02, σ=0.14
Expected: c = 0.0944, p = 0.2237
```

### Discrete Asian HHM (Haug Example)
```
S=100, S_A=110, K=105, t1=0, T=0.5, n=360, m=180, r=0.07, b=0.02, σ=0.25
Expected: c = 2.0971
```
