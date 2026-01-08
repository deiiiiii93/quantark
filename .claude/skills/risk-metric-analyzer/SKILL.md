---
name: risk-metric-analyzer
description: |
  Analyze and report comprehensive risk metrics for financial products in QuantArk.
  Use when the user asks to:
  - Analyze risk metrics for a product (options, bonds, etc.)
  - Calculate Greeks for a derivative
  - Generate a risk report
  - Examine sensitivities (delta, gamma, vega, theta, rho, DV01)
  - Compare risk across products or scenarios
  Triggers: "risk analysis", "calculate greeks", "risk metrics", "sensitivity analysis", "risk report", "delta gamma"
---

# Risk Metric Analyzer Skill

Automatically generate comprehensive risk metric analysis reports for financial products following QuantArk patterns.

## When This Skill Activates

Claude should use this skill when:
- User asks to analyze risk metrics for a product
- User wants to calculate Greeks (delta, gamma, vega, theta, rho)
- User requests a risk report or sensitivity analysis
- User wants to compare risk profiles across products/scenarios
- User mentions DV01, duration, or other rate sensitivities for bonds

## Workflow

### Step 0: Dependency Validation (CRITICAL)

**Before any analysis, validate all required dependencies are installed.**

Import the dependency checker:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "risk-metric-analyzer"))
from dependency_checker import DependencyChecker, format_dependency_table

# Check dependencies
checker = DependencyChecker(python_path="quantark/bin/python")  # Use project's venv
missing_core, missing_optional, available = checker.check_all_dependencies()

# If core dependencies are missing, ASK USER before proceeding
if missing_core:
    print(checker.get_status_report())
    print(format_dependency_table(missing_core, missing_optional))
    
    # ⚠️ IMPORTANT: Always ask user for confirmation
    user_response = input("\n❓ Install missing core dependencies? [Y/n]: ")
    
    if user_response.lower() in ['y', 'yes', '']:
        for dep in missing_core:
            success, msg = checker.install_dependency(dep)
            if success:
                print(f"  ✓ {msg}")
            else:
                print(f"  ✗ {msg}")
                print(f"\n⚠️ Could not install {dep.name}. Please install manually:")
                print(f"   {checker.get_install_command(dep)}")
                return False  # Cannot proceed
    else:
        print("⚠️ Analysis cancelled - missing required dependencies.")
        return False
```

**Required Core Dependencies:**

| Package | Version | Used For |
|---------|---------|----------|
| NumPy | ≥1.20 | Numerical calculations, arrays |
| Pandas | ≥1.3 | Data manipulation, CSV export |
| Matplotlib | ≥3.5 | Visualizations, PDF reports |

**Optional Dependencies:**

| Package | Used For | Fallback |
|---------|----------|----------|
| Seaborn | Enhanced plots | Matplotlib styles |
| ReportLab | PDF via ReportLab | Matplotlib PDF backend |

**User Confirmation Dialog Template:**

```
⚠️ Missing Required Dependencies

The following packages must be installed for risk analysis:

Package        Used For
─────────────────────────────────────────────
NumPy          Numerical calculations and array operations
Pandas         Data manipulation and CSV output
Matplotlib     Visualization and PDF report generation

Install these packages now? [Y/n]: 

Installation will use: quantark/bin/python -m pip install <package>
```

**If Installation Fails:**
1. Display the error message
2. Provide manual installation command
3. Ask user to try again after manual install
4. Do NOT proceed without core dependencies

### Step 0.5: Output Folder Organization

**Organize outputs into dedicated task folders to avoid clutter.**

Each analysis run should create its own folder with structured subdirectories:

```
risk_metric_analysis/
├── european_call_spot_vol_20241230/     # Task-specific folder
│   ├── scripts/                          # Generated scripts
│   ├── data/                             # CSV data files
│   ├── visualizations/                   # PNG plots/charts
│   ├── reports/                          # PDF/MD final reports
│   └── README.md                         # Task description
├── american_put_basic_20241231/          # Another task
│   └── ...
└── barrier_pde_full_20241231/            # Yet another task
    └── ...
```

**Import and use the folder manager:**

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "risk-metric-analyzer"))
from folder_manager import FolderManager, get_analysis_folder_name

# Initialize folder manager
folder_mgr = FolderManager()

# Prepare product info for auto-naming
product_info = {
    'product_type': 'european_call',      # From user input
    'strike': 100,                         # Optional
    'maturity': 1.0,                       # Optional
    'analysis_type': 'spot_vol',           # From user request
}

# Get folder name interactively
analysis_folder, should_proceed = folder_mgr.get_folder_name_from_user(product_info)

if not should_proceed:
    print("Analysis cancelled.")
    return False

# Get path shortcuts for easy access
paths = folder_mgr.get_output_paths(analysis_folder)
# paths['scripts']      -> .../european_call_spot_vol_20241230/scripts
# paths['data']         -> .../european_call_spot_vol_20241230/data
# paths['visualizations'] -> .../european_call_spot_vol_20241230/visualizations
# paths['reports']      -> .../european_call_spot_vol_20241230/reports
```

**Folder Naming Convention:**

| Pattern | Example | Use Case |
|---------|---------|----------|
| `{product}_{analysis}_{date}` | `european_call_spot_vol_20241230` | Default auto-generated |
| `{product}_K{strike}_T{maturity}_{date}` | `european_call_K100_T1y_spot_vol_20241230` | With parameters |
| `{product}_{analysis}_{timestamp}` | `european_call_spot_vol_20241230_143522` | Multiple runs same day |
| User-provided | `my_analysis` | User customization |

**User Dialog Template:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output Folder Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Base directory: risk_metric_analysis/
📝 Suggested name:  european_call_spot_vol_20241230

Folder name [Press Enter for suggestion, or type custom name]:
```

**Conflict Resolution Dialog (if folder exists):**

```
⚠️  Folder 'european_call_spot_vol_20241230' already exists at:
   risk_metric_analysis/european_call_spot_vol_20241230
   Contains 24 files from previous analysis

Options:
  1. Override - Delete existing folder and create new
  2. New name  - Create with a new name (suggested below)
  3. Cancel    - Cancel this operation

  Suggested new name: european_call_spot_vol_20241230_143522

  Choose option [1/2/3] or provide custom name:
```

**Quick Helper for Non-Interactive Use:**

```python
from folder_manager import get_analysis_folder_name

# Generate a name without user interaction
folder_name = get_analysis_folder_name(
    product_type='european_call',
    analysis_type='spot_vol',
    strike=100,
    maturity=1.0,
    include_date=True
)
# Returns: 'european_call_K100_T1.0_spot_vol_20241230'
```

### Step 1: Identify Product Type and Confirm Requirements

**MANDATORY: Clarification Checklist**

Before generating any scripts or reports, confirm the following with the user:

| Item | Confirm | Default if not specified |
|------|---------|--------------------------|
| **Product type** | Exact product (European, American, Barrier, etc.) | Ask user - REQUIRED |
| **Engine preference** | Analytical, PDE, or Monte Carlo | Auto-select based on product |
| **Barrier monitoring** | Continuous or discrete (for barrier options) | Continuous |
| **Valuation date** | Specific date or "today" | `datetime.now()` |
| **Output conventions** | Per-contract vs total position | Per-contract for equity (includes contract_multiplier), PV per bond (includes denominator) |
| **Position size** | Number of contracts/bonds | Default 1; scale totals with position quantity only |
| **Requested metrics** | Specific Greeks or "all defaults" | All defaults for asset class |
| **Output format** | Markdown, PDF, DOCX, or CSV-only | Markdown + CSV |
| **Include visualizations** | Yes/No | No (unless requested) |

**Example confirmation dialogue:**
```
Before generating the risk analysis, please confirm:
1. Product: European Call Option (strike=100, maturity=1Y)
2. Engine: Analytical (BlackScholesEngine) - recommended for vanilla
3. Valuation: Today (2024-01-15)
4. Output: Per-unit Greeks (multiply by position size for total)
5. Metrics: Delta, Gamma, Vega, Theta, Rho
6. Format: Markdown report + CSV data

Proceed? [Y/n]
```

**Product Classification:**

| Product Type | Asset Class | Location |
|--------------|-------------|----------|
| European Vanilla Option | Equity | `asset/equity/product/option/european_vanilla_option.py` |
| American Option | Equity | `asset/equity/product/option/american_option.py` |
| Barrier Option | Equity | `asset/equity/product/option/barrier_option.py` |
| Double Barrier Option | Equity | `asset/equity/product/option/double_barrier_option.py` |
| One Touch Option | Equity | `asset/equity/product/option/one_touch_option.py` |
| Snowball Option | Equity | `asset/equity/product/option/snowball_option.py` |
| Spot Instrument | Equity | `asset/equity/product/deltaone/spot_instrument.py` |
| Futures | Equity | `asset/equity/product/deltaone/futures.py` |
| Fixed Bond | Bond | `asset/bond/product/couponbond/fixed_bond.py` |
| Floating Rate Note | Bond | `asset/bond/product/couponbond/frn.py` |
| Bond Option | Bond | `asset/bond/product/option/euro_short_term_bond_option.py` |
| Interest Rate Swap | Rate | `asset/rate/product/irs/interest_rate_swap.py` |

**If user's product is unclear:**
- Ask clarifying questions about product type
- Check product specifications (strike, maturity, barriers, etc.)

### Step 2: Select Pricing Engine

**Engine Selection Priority:**
1. **Analytical** - Use if available (fastest, most accurate for vanilla products)
2. **PDE** - Use for path-dependent products (barriers, American)
3. **Monte Carlo** - Use as fallback or for complex products (snowball, Asian)

**Engine Decision Matrix:**

| Product | Analytical | PDE | Monte Carlo |
|---------|------------|-----|-------------|
| European Vanilla Option | **BlackScholesEngine** (default) | EuropeanPDESolver | EuropeanMCEngine |
| American Option | **AmericanOptionAnalyticalEngine** (default) | AmericanPDESolver | - |
| Barrier Option | BarrierAnalyticalEngine (limited) | **BarrierPDESolver** (default) | BarrierMCEngine |
| Double Barrier Option | - | **DoubleBarrierPDESolver** (default) | - |
| One Touch Option | OneTouchAnalyticalEngine | **OneTouchPDESolver** (default) | - |
| Snowball Option | - | SnowballPDESolver | **SnowballMCEngine** (default) |
| Delta-One (Spot/Futures) | **DeltaOneEngine** (default) | - | - |
| Fixed/Floating Bond | **BondDiscountEngine** (default) | - | - |
| Bond Option | **BlackBondOptionEngine** (default) | - | - |
| Interest Rate Swap | **IRSDiscountEngine** (default) | - | - |

See [product-engine-mapping.md](product-engine-mapping.md) for detailed mapping.

**MC/PDE Greeks: Critical Guidance**

When using Monte Carlo or PDE engines for numerical Greeks, follow these practices:

1. **Fixed Random Seeds (Monte Carlo)**
   - **CRITICAL**: Use the same random seed for base and bumped valuations (common random numbers)
   - This reduces variance in Greek estimates by orders of magnitude
   - Set `MCParams(seed=42)` and reuse for all bump scenarios

2. **MC Greeks Are Noisy**
   - Gamma and Vega via MC have high variance; do not interpret small differences as "model risk"
   - Use QMC (Quasi-Monte Carlo) or RQMC for lower variance: `method=MonteCarloMethod.QUASI`
   - Increase path count for second-order Greeks: 500k+ paths recommended for gamma

3. **Barrier Options with Continuous Monitoring**
   - Use Brownian bridge barrier correction for continuously monitored barriers
   - PDE solvers with fine time grids are preferred over MC for barriers
   - For discrete barriers, ensure observation dates align with MC time steps

4. **PDE Grid Resolution**
   - Greeks extracted from PDE are grid-dependent; use Richardson extrapolation for accuracy
   - Time grid: 100+ steps for near-barrier options
   - Spatial grid: Use non-uniform grid if products are path-dependent, with refinement near barriers/strikes

### Step 3: Determine Risk Metrics

**Default Risk Metrics by Asset Class:**

**Equity Derivatives:**
- Delta (∂V/∂S) - Spot sensitivity
- Gamma (∂²V/∂S²) - Delta convexity
- Vega (∂V/∂σ) - Volatility sensitivity
- Theta (∂V/∂t) - Time decay
- Rho (∂V/∂r) - Rate sensitivity

**Bond Derivatives:**
- Delta (∂V/∂F) - Forward price sensitivity
- Gamma (∂²V/∂F²) - Convexity
- Vega (∂V/∂σ) - Volatility sensitivity
- Theta (∂V/∂t) - Time decay
- Rho (∂V/∂r) - Rate sensitivity
- DV01 - Dollar value of 1bp rate move
- Duration - Effective duration
- Convexity - Rate convexity

**Delta-One Products:**
- Delta = 1.0 (by definition)
- Gamma = 0.0
- Vega = 0.0
- Theta (for futures only)

See [risk-metrics-reference.md](risk-metrics-reference.md) for detailed formulas.

**Units and Scaling Conventions (CRITICAL)**

QuantArk uses a **two-stage scaling model** that must be understood for correct risk reporting:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE SCALING MODEL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Engine Output (Per Contract/Unit)                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Equity: price = theoretical_value × contract_multiplier      │ │
│  │ Bond:   price = PV (already includes denominator)            │ │
│  │ Greeks: Same scaling as price                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                           │                                     │
│                           ▼                                     │
│  Stage 2: Position Scaling                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ All Assets: market_value = engine_price × quantity          │ │
│  │ All Assets: position_greek = engine_greek × quantity        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Understanding Engine-Returned Values:**

| Metric | Equity Derivatives | Fixed Income |
|--------|-------------------|--------------|
| **Price** | `theoretical_value × contract_multiplier` | `PV` (already includes denominator) |
| **Delta** | `∂price/∂spot × contract_multiplier` | `∂price/∂yield × denominator` |
| **Gamma** | `∂²price/∂spot² × contract_multiplier` | `∂²price/∂yield² × denominator` |
| **Vega** | `∂price/∂vol × contract_multiplier` | `∂price/∂vol × denominator` |
| **Theta** | `∂price/∂time × contract_multiplier` | `∂price/∂time × denominator` |
| **Rho** | `∂price/∂rate × contract_multiplier` | (rate sensitivity via DV01) |
| **DV01** | N/A | `∂price/∂yield` per 1bp |

**Unit Reference Table:**

| Greek | Unit | What It Means | Scaling Applied |
|-------|------|---------------|-----------------|
| **Price** | Currency $ | Price of ONE contract/unit | × contract_multiplier (equity) |
| **Delta** | $ per $1 spot | P&L for $1 spot move per contract | × contract_multiplier (equity) |
| **Gamma** | $ per $1 spot² | Delta change per $1 spot move per contract | × contract_multiplier (equity) |
| **Vega** | $ per 1% vol | P&L for +1% vol change per contract | × contract_multiplier (equity) |
| **Theta** | $ per day | P&L for 1 day passage per contract | × contract_multiplier (equity) |
| **Rho** | $ per 1% rate | P&L for +1% rate change per contract | × contract_multiplier (equity) |
| **DV01** | $ per 1bp | P&L for +1bp rate change per bond | × denominator (bond) |

**IMPORTANT Reporting Conventions:**

When generating risk reports, ALWAYS clarify the scaling convention:

```markdown
## Risk Metrics Report

**Scaling Convention:**
- **Per-Unit Greeks**: Values are per contract/unit. Multiply by position quantity for total exposure.
- **Contract Multiplier**: 100 (equity options represent 100 shares)
- **Position Size**: 50 contracts

| Metric | Per-Unit | Position (50 contracts) |
|--------|----------|------------------------|
| Delta  | $52.30   | $2,615.00              |
| Gamma  | $1.25    | $62.50                 |
```

**For Equity Options:**
```python
# Engine returns price scaled by contract_multiplier
engine = BlackScholesEngine()
product = EuropeanVanillaOption(strike=100, contract_multiplier=100)
price_per_contract = engine.price(product, pricing_env)  # = theoretical_value × 100

# Position scales by quantity
position = EquityPosition(product, quantity=50, ...)
total_value = price_per_contract * 50  # Total position value
```

**For Bonds:**
```python
# Bond engines return price including denominator
engine = BondDiscountEngine()
bond = FixedBond(denominator=1000, ...)
price_per_bond = engine.dirty_price(bond, valuation_date)  # PV of $1000 notional

# Position scales by quantity
position = FIPosition(bond, quantity=10, ...)
total_value = price_per_bond * 10  # Total position value
```

**Common Mistakes to Avoid:**

| Mistake | Symptom | Correct Approach |
|---------|---------|------------------|
| Treating engine price as "per share" | Greeks too small | Engine returns contract-level prices |
| Double scaling by multiplier | Values too large | Engine already applied multiplier |
| Forgetting position quantity | Missing total exposure | `position_greek = engine_greek × quantity` |
| Comparing equity vs bond Greeks directly | Inconsistent magnitudes | Different conventions: equity (per contract) vs bond (per denominator) |

### Step 4: Define Product Parameters

**Required Parameters by Product Type:**

**European/American Options:**
```python
# Required
strike: float        # Strike price
option_type: OptionType  # CALL or PUT
maturity: float      # Time to maturity (years) OR
exercise_date: date  # Exercise date

# Optional (from pricing environment)
spot: float          # Spot price
volatility: float    # Implied volatility
rate: float          # Risk-free rate
div_yield: float     # Dividend yield

# Optional (contract scaling)
contract_multiplier: float  # Default 1.0, equity contract size
```

**Barrier Options:**
```python
# Additional to vanilla
barrier: float           # Barrier level
barrier_type: BarrierType  # DOWN_AND_OUT, UP_AND_OUT, etc.
rebate: float = 0.0      # Optional rebate
```

**Bond Options:**
```python
strike: float            # Strike price
option_type: OptionType  # CALL or PUT
expiry_date: date        # Option expiry
underlying: Bond         # Underlying bond
notional: float          # Contract size (number of bonds per option)
```

**If user doesn't provide parameters:**
- Use default values from product class
- Ask for essential missing parameters (strike, maturity)
- Use ATM strike if not specified: strike = spot

### Step 5: Set Up Pricing Environment

**Default Pricing Environment:**

```python
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from datetime import datetime

# Default values (use if not specified)
DEFAULT_SPOT = 100.0
DEFAULT_VOL = 0.20       # 20% volatility
DEFAULT_RATE = 0.05      # 5% risk-free rate
DEFAULT_DIV_YIELD = 0.02 # 2% dividend yield

pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=spot or DEFAULT_SPOT),
    vol_surface=FlatVolSurface(volatility=vol or DEFAULT_VOL),
    rate_curve=FlatRateCurve(rate=rate or DEFAULT_RATE),
    div_yield=ContinuousDividendYield(div_yield=div_yield or DEFAULT_DIV_YIELD),
    valuation_date=valuation_date or datetime.now(),
)
```

### Step 6: Create Calculation Scripts

**Script Location:** `risk_metric_analysis/risk_metric_calculation_scripts/`

**Script Naming:** `{product_name}_risk_analysis.py`

**Script Template:**

```python
"""
Risk Metric Analysis for {ProductName}
Generated by Risk Metric Analyzer Skill
"""
from datetime import datetime
from asset.{asset_class}.product.{category} import {ProductClass}
from asset.{asset_class}.engine.{engine_type} import {EngineClass}
from asset.{asset_class}.riskmeasures import {GreeksCalculatorClass}
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
import pandas as pd

# === Product Configuration ===
{product_config}

# === Pricing Environment ===
{pricing_env_config}

# === Create Product ===
product = {ProductClass}({product_params})

# === Create Engine ===
engine = {EngineClass}({engine_params})

# === Calculate Greeks (includes price) ===
calculator = {GreeksCalculatorClass}()
greeks = calculator.calculate_numerical_greeks(product, pricing_env, engine)
# Note: greeks dict already contains 'price' key - no need to call engine.price() separately

# Analytical Greeks (if available)
{analytical_greeks_code}

# === Output Results ===
results = {
    'Product': '{product_name}',
    **greeks  # Contains: price, delta, gamma, vega, theta, rho
}

# Save to CSV
df = pd.DataFrame([results])
df.to_csv('risk_metric_analysis/reports/{output_filename}_data.csv', index=False)

print("Risk Metrics for {product_name}:")
for key, value in results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")
```

### Step 7: Create Visualizations (Optional)

**If user requests visualizations:**

> **IMPORTANT:** Follow [plotting-style-guide.md](plotting-style-guide.md) for all visual standards including colors, fonts, layouts, and accessibility.

**Script Location:** `risk_metric_analysis/plotting_scripts/`

**Visualization Types:**

| Type | Factors | Plot Style | Use Case |
|------|---------|------------|----------|
| **1D Line Plot** | 1 factor | Line chart | Greeks vs single parameter |
| **2D Heatmap** | 2 factors | Color matrix | Greeks surface (Spot × Vol, Spot × Time) |
| **3D Surface** | 2 factors | 3D mesh/surface | Interactive Greek landscape |
| **3D Scatter** | 3 factors | Color-coded 3D points | Three-factor sensitivity |

**Single-Factor Visualizations (1D):**

1. **Greeks vs Spot** - Delta, Gamma, Vega as function of spot
2. **Greeks vs Time** - Greeks evolution over time to expiry
3. **Greeks vs Volatility** - Sensitivity to vol changes
4. **Greek Comparison** - Compare analytical vs numerical

**Multi-Factor Visualizations (2D/3D):**

5. **Spot × Volatility Heatmap** - Greek values across spot-vol grid (most common)
6. **Spot × Time Heatmap** - Greek evolution over spot and time
7. **Spot × Rate Heatmap** - Rate sensitivity across spot levels
8. **3D Surface: Spot × Vol × Greek** - Interactive 3D surface
9. **3D Surface: Spot × Time × Greek** - Time decay surface
10. **3-Factor Scatter: Spot × Vol × Time** - Color = Greek value

**Plotting Script Template:**

```python
"""
Risk Metric Visualization for {ProductName}
"""
import numpy as np
import matplotlib.pyplot as plt
from asset.{asset_class}.product.{category} import {ProductClass}
from asset.{asset_class}.engine.{engine_type} import {EngineClass}
from asset.{asset_class}.riskmeasures import GreeksCalculator

# ... setup code ...
from copy import deepcopy
from param import SpotQuote

# Spot range for analysis
spot_range = np.linspace(spot * 0.5, spot * 1.5, 50)

# Calculate Greeks for each spot level
deltas, gammas, vegas = [], [], []
for s in spot_range:
    env = deepcopy(pricing_env)
    env.spot_quote = SpotQuote(spot=s)  # Recreate to trigger validation
    greeks = calculator.calculate_numerical_greeks(product, env, engine)
    deltas.append(greeks['delta'])
    gammas.append(greeks['gamma'])
    vegas.append(greeks['vega'])

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(spot_range, deltas, 'b-', linewidth=2)
axes[0].axvline(x=strike, color='r', linestyle='--', label='Strike')
axes[0].set_xlabel('Spot Price')
axes[0].set_ylabel('Delta')
axes[0].set_title('Delta vs Spot')
axes[0].legend()
axes[0].grid(True)

# ... similar for gamma and vega ...

plt.tight_layout()
plt.savefig('risk_metric_analysis/reports/{output_filename}_greeks.png', dpi=150)
plt.close()
```

**2D Heatmap Template (Spot × Volatility):**

```python
"""
2D Heatmap: Greek Surface (Spot × Volatility)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from copy import deepcopy
from param import SpotQuote, FlatVolSurface

# Grid parameters
SPOT_RANGE = (spot * 0.7, spot * 1.3)  # +/- 30%
VOL_RANGE = (0.10, 0.50)               # 10% to 50%
GRID_SIZE = 25                          # 25x25 grid

# Create grids
spot_grid = np.linspace(*SPOT_RANGE, GRID_SIZE)
vol_grid = np.linspace(*VOL_RANGE, GRID_SIZE)
SPOT, VOL = np.meshgrid(spot_grid, vol_grid)

# Calculate Greeks for each (spot, vol) combination
greek_values = np.zeros_like(SPOT)
for i, v in enumerate(vol_grid):
    for j, s in enumerate(spot_grid):
        env = deepcopy(pricing_env)
        env.spot_quote = SpotQuote(spot=s)
        env.vol_surface = FlatVolSurface(volatility=v)
        greeks = calculator.calculate_numerical_greeks(product, env, engine)
        greek_values[i, j] = greeks['{greek_name}']  # e.g., 'delta', 'gamma', 'vega'

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))

# Use diverging colormap centered at zero for delta/gamma
if '{greek_name}' in ['delta', 'gamma']:
    norm = TwoSlopeNorm(vmin=greek_values.min(), vcenter=0, vmax=greek_values.max())
    cmap = 'RdBu_r'
else:
    norm = None
    cmap = 'viridis'

im = ax.pcolormesh(SPOT, VOL, greek_values, cmap=cmap, norm=norm, shading='auto')
cbar = fig.colorbar(im, ax=ax, label='{greek_name.capitalize()}')

# Mark current position and strike
ax.axvline(x=spot, color='white', linestyle='--', linewidth=1.5, label='Current Spot')
ax.axvline(x=strike, color='red', linestyle='-', linewidth=1.5, label='Strike')
ax.axhline(y=current_vol, color='white', linestyle=':', linewidth=1.5, label='Current Vol')

ax.set_xlabel('Spot Price', fontsize=12)
ax.set_ylabel('Volatility', fontsize=12)
ax.set_title('{greek_name.capitalize()} Heatmap: Spot × Volatility', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('risk_metric_analysis/reports/{output_filename}_{greek_name}_heatmap.png', dpi=150)
plt.close()
```

**3D Surface Template (Spot × Volatility × Greek):**

```python
"""
3D Surface: Greek Landscape (Spot × Volatility)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from param import SpotQuote, FlatVolSurface

# Grid parameters (same as heatmap)
SPOT_RANGE = (spot * 0.7, spot * 1.3)
VOL_RANGE = (0.10, 0.50)
GRID_SIZE = 30

spot_grid = np.linspace(*SPOT_RANGE, GRID_SIZE)
vol_grid = np.linspace(*VOL_RANGE, GRID_SIZE)
SPOT, VOL = np.meshgrid(spot_grid, vol_grid)

# Calculate Greeks
greek_values = np.zeros_like(SPOT)
for i, v in enumerate(vol_grid):
    for j, s in enumerate(spot_grid):
        env = deepcopy(pricing_env)
        env.spot_quote = SpotQuote(spot=s)
        env.vol_surface = FlatVolSurface(volatility=v)
        greeks = calculator.calculate_numerical_greeks(product, env, engine)
        greek_values[i, j] = greeks['{greek_name}']

# Create 3D surface
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Surface plot with color mapping
surf = ax.plot_surface(SPOT, VOL, greek_values, cmap='viridis',
                        edgecolor='none', alpha=0.9, antialiased=True)

# Add wireframe for clarity
ax.plot_wireframe(SPOT, VOL, greek_values, color='black', linewidth=0.3, alpha=0.3)

# Mark current position
ax.scatter([spot], [current_vol], [current_greek], color='red', s=100,
           marker='o', label='Current Position', zorder=5)

ax.set_xlabel('Spot Price', fontsize=11)
ax.set_ylabel('Volatility', fontsize=11)
ax.set_zlabel('{greek_name.capitalize()}', fontsize=11)
ax.set_title('{greek_name.capitalize()} Surface: Spot × Volatility', fontsize=14, fontweight='bold')

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='{greek_name.capitalize()}')

# Set viewing angle for best perspective
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('risk_metric_analysis/reports/{output_filename}_{greek_name}_3d_surface.png', dpi=150)
plt.close()
```

**3D Scatter Template (3 Factors: Spot × Vol × Time):**

```python
"""
3D Scatter: Three-Factor Sensitivity (Spot × Vol × Time)
Color represents Greek value
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from param import SpotQuote, FlatVolSurface

# Grid parameters for 3 factors
SPOT_RANGE = (spot * 0.8, spot * 1.2)
VOL_RANGE = (0.15, 0.40)
TIME_RANGE = (0.1, 1.0)  # 0.1 to 1 year
POINTS_PER_DIM = 10  # 10^3 = 1000 points

spot_grid = np.linspace(*SPOT_RANGE, POINTS_PER_DIM)
vol_grid = np.linspace(*VOL_RANGE, POINTS_PER_DIM)
time_grid = np.linspace(*TIME_RANGE, POINTS_PER_DIM)

# Collect all combinations
spots, vols, times, greek_vals = [], [], [], []

for t in time_grid:
    for v in vol_grid:
        for s in spot_grid:
            env = deepcopy(pricing_env)
            env.spot_quote = SpotQuote(spot=s)
            env.vol_surface = FlatVolSurface(volatility=v)

            # Adjust product maturity
            product_copy = deepcopy(product)
            product_copy.maturity = t

            try:
                greeks = calculator.calculate_numerical_greeks(product_copy, env, engine)
                spots.append(s)
                vols.append(v)
                times.append(t)
                greek_vals.append(greeks['{greek_name}'])
            except:
                pass  # Skip invalid combinations

spots = np.array(spots)
vols = np.array(vols)
times = np.array(times)
greek_vals = np.array(greek_vals)

# Create 3D scatter
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(spots, vols, times, c=greek_vals, cmap='viridis',
                     s=20, alpha=0.7, edgecolors='none')

ax.set_xlabel('Spot Price', fontsize=11)
ax.set_ylabel('Volatility', fontsize=11)
ax.set_zlabel('Time to Maturity', fontsize=11)
ax.set_title('3-Factor Sensitivity: Color = {greek_name.capitalize()}', fontsize=14, fontweight='bold')

fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='{greek_name.capitalize()}')
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('risk_metric_analysis/reports/{output_filename}_{greek_name}_3d_scatter.png', dpi=150)
plt.close()
```

**Spot × Time Heatmap Template (Time Decay Surface):**

```python
"""
2D Heatmap: Time Decay Surface (Spot × Time)
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from param import SpotQuote

# Grid parameters
SPOT_RANGE = (spot * 0.7, spot * 1.3)
TIME_RANGE = (0.01, maturity)  # Near-expiry to full maturity
GRID_SIZE = 30

spot_grid = np.linspace(*SPOT_RANGE, GRID_SIZE)
time_grid = np.linspace(*TIME_RANGE, GRID_SIZE)
SPOT, TIME = np.meshgrid(spot_grid, time_grid)

# Calculate Greeks
greek_values = np.zeros_like(SPOT)
for i, t in enumerate(time_grid):
    for j, s in enumerate(spot_grid):
        env = deepcopy(pricing_env)
        env.spot_quote = SpotQuote(spot=s)

        product_copy = deepcopy(product)
        product_copy.maturity = t

        try:
            greeks = calculator.calculate_numerical_greeks(product_copy, env, engine)
            greek_values[i, j] = greeks['{greek_name}']
        except:
            greek_values[i, j] = np.nan

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.pcolormesh(SPOT, TIME, greek_values, cmap='plasma', shading='auto')
cbar = fig.colorbar(im, ax=ax, label='{greek_name.capitalize()}')

ax.axvline(x=spot, color='white', linestyle='--', linewidth=1.5, label='Current Spot')
ax.axvline(x=strike, color='red', linestyle='-', linewidth=1.5, label='Strike')

ax.set_xlabel('Spot Price', fontsize=12)
ax.set_ylabel('Time to Maturity (years)', fontsize=12)
ax.set_title('{greek_name.capitalize()} Heatmap: Spot × Time', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('risk_metric_analysis/reports/{output_filename}_{greek_name}_time_heatmap.png', dpi=150)
plt.close()
```

### Step 8: Generate Report

**Report Location:** `risk_metric_analysis/reports/`

**Report Format Options:**
- **Markdown** (.md) - Default, human-readable
- **PDF** (.pdf) - Professional format
- **DOCX** (.docx) - Microsoft Word compatible

**Data Format:**
- **CSV** (.csv) - Raw calculation data

**Report Template Structure:**

```markdown
# Risk Metric Analysis Report

**Product:** {product_name}
**Analysis Date:** {date}
**Valuation Date:** {valuation_date}

## 1. Product Summary

| Parameter | Value |
|-----------|-------|
| Product Type | {product_type} |
| Strike | {strike} |
| Maturity | {maturity} |
| Option Type | {option_type} |
| ... | ... |

## 2. Market Data

| Parameter | Value |
|-----------|-------|
| Spot Price | {spot} |
| Volatility | {volatility}% |
| Risk-Free Rate | {rate}% |
| Dividend Yield | {div_yield}% |

## 3. Pricing Results

| Metric | Value |
|--------|-------|
| Price | {price} |
| Intrinsic Value | {intrinsic} |
| Time Value | {time_value} |

## 4. Risk Metrics (Greeks)

| Greek | Value | Interpretation |
|-------|-------|----------------|
| Delta | {delta} | {delta_interpretation} |
| Gamma | {gamma} | {gamma_interpretation} |
| Vega | {vega} | {vega_interpretation} |
| Theta | {theta} | {theta_interpretation} |
| Rho | {rho} | {rho_interpretation} |

## 5. Visualizations

{include_charts_if_requested}

## 6. Methodology Notes

- Pricing Engine: {engine_name}
- Greeks Method: {greeks_method}
- Bump Size: {bump_size}%

---
*Generated by QuantArk Risk Metric Analyzer*
```

See [report-templates.md](report-templates.md) for detailed templates.

## Output Structure

```
risk_metric_analysis/
├── risk_metric_calculation_scripts/
│   └── {product}_risk_analysis.py
├── plotting_scripts/
│   └── {product}_visualization.py (if requested)
└── reports/
    ├── {product}_report.md (or .pdf, .docx)
    ├── {product}_data.csv
    └── {product}_greeks.png (if visualizations requested)
```

## Example Usage

### Example 1: European Option Risk Analysis

User: "Analyze risk metrics for a European call option with strike 100, maturity 1 year"

**Generated Script:** `european_call_risk_analysis.py`

```python
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from util.enum import OptionType

# Product
option = EuropeanVanillaOption(
    strike=100.0,
    option_type=OptionType.CALL,
    maturity=1.0
)

# Engine and Greeks
engine = BlackScholesEngine()
calculator = GreeksCalculator()

# Price and Greeks
price = engine.price(option, pricing_env)
greeks = calculator.calculate_analytical_greeks(option, pricing_env)
```

### Example 2: Bond Option Risk Analysis

User: "Calculate DV01 and duration for my bond option"

**Generated Script:** `bond_option_risk_analysis.py`

```python
from asset.bond.product.option import EuroShortTermBondOption
from asset.bond.engine.analytical import BlackBondOptionEngine
from asset.bond.riskmeasures import BondGreeksCalculator

# Product (from user specification)
bond_option = EuroShortTermBondOption(...)

# Engine and Calculator
engine = BlackBondOptionEngine(pricing_env)
calculator = BondGreeksCalculator()

# Calculate Greeks and Bond Sensitivities
greeks = calculator.calculate_analytical_greeks(bond_option, pricing_env)
sensitivities = calculator.calculate_bond_sensitivities(bond_option, pricing_env)
```

## Reference Files

- [product-engine-mapping.md](product-engine-mapping.md) - Complete product-to-engine mapping
- [risk-metrics-reference.md](risk-metrics-reference.md) - Greek formulas and interpretation
- [report-templates.md](report-templates.md) - Report templates and formats
- [plotting-style-guide.md](plotting-style-guide.md) - Visual standards for all plots

## Risk Measures Implementation Reference

- Equity Greeks: `asset/equity/riskmeasures/greeks_calculator.py`
- Bond Greeks: `asset/bond/riskmeasures/bond_greeks_calculator.py`
- VaR Engines: `var/engines/`

## Notes

- Always validate products before pricing
- Use analytical Greeks when available (faster, more accurate)
- Numerical Greeks use central difference (bump_size=0.01 default)
- Edge cases handled: near expiry, deep ITM/OTM (volatility must be strictly positive)
- Reports should be educational - include interpretation of metrics
