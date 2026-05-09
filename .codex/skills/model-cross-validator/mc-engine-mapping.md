# MC Engine Mapping

Quick reference for finding Monte Carlo benchmark engines for different product types.

## Available MC Engines

| Product Type | MC Engine Class | Location | Notes |
|--------------|-----------------|----------|-------|
| European Vanilla | `EuroMCEngine` | `asset/equity/engine/mc/euro_mc_engine.py` | Standard GBM paths |
| Asian (Arithmetic) | `AsianOptionMCEngine` | `asset/equity/engine/mc/asian_option_mc_engine.py` | Discrete averaging |
| Asian (Geometric) | `AsianOptionMCEngine` | `asset/equity/engine/mc/asian_option_mc_engine.py` | Control variate |
| Snowball | `SnowballMCEngine` | `asset/equity/engine/mc/snowball_mc_engine.py` | Autocallable with KO |
| Phoenix | `PhoenixMCEngine` | `asset/equity/engine/mc/phoenix_mc_engine.py` | Coupon autocallable |

## Product to MC Engine Mapping

### Equity Options

| Analytical/PDE Engine | Corresponding MC Engine |
|-----------------------|------------------------|
| `BlackScholesEngine` | `EuroMCEngine` |
| `AmericanOptionAnalyticalEngine` | `EuroMCEngine` (exercise boundary) |
| `BarrierAnalyticalEngine` | *Needs creation* |
| `DigitalOptionAnalyticalEngine` | `EuroMCEngine` (binary payoff) |
| `OneTouchAnalyticalEngine` | *Needs creation* |
| `AsianOptionAnalyticalEngine` | `AsianOptionMCEngine` |
| `SnowballPDESolver` | `SnowballMCEngine` |
| `SnowballQuadEngine` | `SnowballMCEngine` |
| `PhoenixPDESolver` | `PhoenixMCEngine` |

### Fixed Income (Future)

| Engine | MC Benchmark |
|--------|--------------|
| Bond discount engines | N/A (deterministic) |
| Rate model engines | *Would need rate MC* |

## MC Engine Discovery Commands

```bash
# List all MC engines
ls -la asset/equity/engine/mc/*.py

# Find MC engine for a product
rg "class.*MCEngine" asset/equity/engine/mc/ -A 5

# Check what products an MC engine supports
rg "isinstance.*product" asset/equity/engine/mc/<engine>.py
```

## Creating Missing MC Engines

If no MC engine exists for a product:

1. Use `engine-creator` skill to create one
2. Follow MC engine patterns from existing engines
3. Key components:
   - Path generation (GBM, Heston, etc.)
   - Payoff calculation
   - Discounting
   - Standard error calculation

### MC Engine Template

```python
from asset.equity.engine.mc.base_mc_engine import BaseMCEngine

class NewProductMCEngine(BaseMCEngine):
    """Monte Carlo engine for [Product]."""

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int = 42,
        use_antithetic: bool = True,
    ):
        super().__init__(n_paths, n_steps, seed)
        self.use_antithetic = use_antithetic

    def price(self, product, pricing_env) -> float:
        # Generate paths
        paths = self._generate_paths(pricing_env)

        # Calculate payoffs
        payoffs = self._calculate_payoffs(paths, product, pricing_env)

        # Discount and average
        df = np.exp(-pricing_env.rate * product.get_maturity(pricing_env))
        price = df * np.mean(payoffs)

        return price

    def price_with_stats(self, product, pricing_env, n_paths=None):
        # ... return price, std_error, CI
        pass
```

## Recommended MC Configurations

### Standard Validation

```python
MC_CONFIG_STANDARD = {
    'n_paths': 100_000,
    'n_steps': 252,
    'seed': 42,
    'use_antithetic': True,
}
```

### High Precision (for discrepancies)

```python
MC_CONFIG_HIGH = {
    'n_paths': 1_000_000,
    'n_steps': 504,  # 2x daily
    'seed': 42,
    'use_antithetic': True,
}
```

### Quick Check

```python
MC_CONFIG_QUICK = {
    'n_paths': 10_000,
    'n_steps': 100,
    'seed': 42,
    'use_antithetic': True,
}
```

## Convergence Properties

Expected MC convergence rate: Error ∝ 1/√N

| Paths | Expected Std Error |
|-------|-------------------|
| 10,000 | ~1% |
| 100,000 | ~0.3% |
| 1,000,000 | ~0.1% |

## When MC Comparison is Not Applicable

1. **MC Engine being validated**: Cannot compare MC to MC
2. **Deterministic products**: Bonds with known cash flows
3. **Numerical instability**: Some exotic products
4. **Performance constraints**: Time-critical validation

In these cases, use alternative validation methods:
- Theoretical relationship checks
- Boundary condition verification
- Independent implementation (Developer B)
