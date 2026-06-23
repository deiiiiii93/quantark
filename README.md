# QuantArk — Financial Derivatives Pricing & Risk Library

[![tests](https://github.com/deiiiiii93/quantark/actions/workflows/tests.yml/badge.svg)](https://github.com/deiiiiii93/quantark/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/quantark)](https://pypi.org/project/quantark/)
[![Python](https://img.shields.io/pypi/pyversions/quantark)](https://pypi.org/project/quantark/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A modular, professional-grade Python library for pricing and risk management of financial derivatives.

## Overview

QuantArk spans five asset classes (equity, bond, rate, FX, credit) and a full
risk-and-capital stack, all built on a clean, modular architecture that
separates concerns across composable layers:

- **Products**: Define instrument specifications (options, swaps, bonds, CDS, etc.)
- **Processes**: Stochastic models (Black-Scholes-Merton, Heston, Local Vol, SLV, Garman-Kohlhagen, hazard-rate)
- **Engines**: Pricing algorithms (Analytical, Monte Carlo, PDE, Quadrature, Tree)
- **Parameters**: Market data (spot prices, volatility surfaces incl. SABR / Vanna-Volga, rate curves, dividends)
- **PriceEnv**: Unified pricing environment bundling all market data
- **RiskMeasures**: Greeks calculation (both analytical and numerical)
- **Risk & Capital**: VaR, stress testing, dynamic scenarios, hedging backtests, and regulatory capital (ISDA SIMM, SA-CCR, SA-CVA)

## Features

### Current Implementation

#### Equity derivatives
- **Vanilla**: European and American options (analytical BS93/BS02/BAW, MC LSM, PDE)
- **Asian, Digital, Barrier, Double-Barrier, One-Touch, Double-One-Touch** options
- **Autocallables**: Snowball, Phoenix, KO-Reset Snowball (MC, PDE, Quad engines)
- **Structured**: Range Accrual, Accumulator, Single/Double Sharkfin
- **Total Return Swaps**: single-asset, multi-asset, and dual-currency, with realized-cashflow accounting
- **Delta-One**: spot instruments and futures

#### Stochastic models & engines
- **Processes**: Black-Scholes-Merton (continuous dividend yield), Heston, Local Vol (Dupire), Stochastic Local Vol (SLV)
- **Engines**: Analytical (closed-form), Monte Carlo (QMC — Sobol, Brownian bridge, randomized QMC, antithetic/control-variate variance reduction), PDE (finite-difference, banded solvers), Quadrature, Tree
- **Shared MC layer** (`quantark/montecarlo/`): reusable QMC path generation across equity and FX

#### Fixed income
- **Bonds**: fixed-rate coupon bonds, FRNs, convertible bonds, bond forwards, bond futures, bond options
- **Interest rate**: swaps (IRS), FRAs, caps/floors, swaptions — with DV01 / key-rate risk

#### FX derivatives
- **Garman-Kohlhagen** vanilla, digital, and quanto (vanilla / digital) options with two-curve discounting
- **Path-dependent / exotic**: barrier, one-touch, sharkfin, range accrual (domestic / quanto / foreign), target-redemption forwards & notes (TARF / TARN)
- **Smile models**: SABR (Hagan) and Vanna-Volga, plus an FX-Heston analytical engine
- **Delta-one**: spot / forward / swap products

#### Credit derivatives
- **CDS** and **basket CDS** priced off a hazard-rate curve (model-agnostic engine), with hazard01 / CS01 sensitivities

#### Risk, scenario & regulatory capital
- **Portfolio VaR**: Historical (full revaluation), Parametric (variance-covariance with Greeks/DV01), and Monte Carlo — with risk-factor attribution and a backtesting framework
- **Stress testing**: scenario-based shocks across all asset classes
- **Dynamic scenarios**: multi-day path simulation
- **Hedging backtests**: delta / DV01 / convexity-neutral strategies (equity / FI / FX)
- **ISDA SIMM v2.6**: initial margin with CRIF, calibration, and dashboard
- **SA-CCR**: Basel counterparty EAD (RC + PFE)
- **SA-CVA**: Basel (MAR50) SBA CVA capital
- **Portfolios**: equity, fixed income, FX, and credit position tracking — one position object feeds the entire risk stack

#### Cross-cutting
- **Greeks**: analytical (closed-form) and numerical (finite-difference) — Delta, Gamma, Vega, Theta, Rho, DV01
- **Robust error handling**: professional exception hierarchy
- **Numerical stability**: shared `quantark.util.numerical` toolkit (protected math, tolerance-aware comparisons), careful boundary checking and validation

### Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Extensibility**: Easy to add new products, processes, and engines
3. **Type Safety**: Extensive use of dataclasses and type hints
4. **Validation**: Input validation at every level
5. **Professional Exception Handling**: Custom exception hierarchy for different error types

## Installation

```bash
pip install quantark
```

From source / latest development version:

```bash
pip install git+https://github.com/deiiiiii93/quantark
```

For development (editable install with test tooling):

```bash
git clone https://github.com/deiiiiii93/quantark && cd quantark
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

**Migration note**: imports use the `quantark.*` namespace
(`from quantark.util.enum import OptionType`). The historical flat
top-level packages (`asset`, `util`, `param`, …) still work through a
compatibility shim that aliases them to the same modules and emits a
`DeprecationWarning` — migrate existing code to `quantark.*` at your
convenience.

## Quick Start

```python
from quantark.asset.equity.product.option import EuropeanVanillaOption
from datetime import datetime

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

# Set up market data
spot = SpotQuote(spot=100.0)
vol = FlatVolSurface(volatility=0.20)  # 20% vol
rate = FlatRateCurve(rate=0.05)  # 5% risk-free rate
div = ContinuousDividendYield(div_yield=0.02)  # 2% dividend yield

pricing_env = PricingEnvironment(
    spot_quote=spot,
    vol_surface=vol,
    rate_curve=rate,
    div_yield=div,
    valuation_date=datetime(2024, 1, 1),
)

# Create a European call option
call_option = EuropeanVanillaOption(
    strike=100.0,
    maturity=1.0,  # 1 year
    option_type=OptionType.CALL
)

# Price the option
engine = BlackScholesEngine()
price = engine.price(call_option, pricing_env)
print(f"Call Price: ${price:.6f}")

# Calculate Greeks
greeks_calc = GreeksCalculator()
analytical_greeks = greeks_calc.calculate_analytical_greeks(
    call_option, pricing_env, price
)

print(f"Delta: {analytical_greeks['delta']:.6f}")
print(f"Gamma: {analytical_greeks['gamma']:.6f}")
print(f"Vega:  {analytical_greeks['vega']:.6f}")
print(f"Theta: {analytical_greeks['theta']:.6f} (per day)")
print(f"Rho:   {analytical_greeks['rho']:.6f}")

```

### FX Options

```python
from datetime import datetime

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType

fx_env = FxPricingEnvironment(
    valuation_date=datetime(2026, 6, 12),
    spot_quote=SpotQuote(spot=1.20),            # EUR/USD
    domestic_curve=FlatRateCurve(rate=0.05),    # USD
    foreign_curve=FlatRateCurve(rate=0.03),     # EUR
    vol_surface=FlatVolSurface(volatility=0.10),
)

option = FxVanillaOption(
    currency_pair=CurrencyPair("EUR", "USD"),
    strike=1.25,
    option_type=OptionType.CALL,
    maturity=1.0,
    notional_foreign=1_000_000.0,   # 1m EUR
)

engine = GarmanKohlhagenEngine()
print(f"Price: {engine.price(option, fx_env):,.2f} USD")
print(f"Delta: {engine.calculate_greeks(option, fx_env)['delta']:,.2f} EUR")
```

See `example/fx_vanilla_option_demo.py`, `example/fx_digital_option_demo.py`,
`example/fx_quanto_option_demo.py`, and `example/fx_deltaone_demo.py` for
digitals, quantos, and delta-one products.

For portfolio Value-at-Risk (parametric, historical, and Monte Carlo engines
with risk-factor attribution), see the runnable demos:
`example/parametric_var_demo.py`, `example/portfolio_var_demo.py`, and
`example/var_backtest_demo.py`.

## Batch-Friendly Engine Params

For QUAD/PDE batch pricing, you can use named presets, factory helpers, or YAML/JSON
configs instead of tuning many individual parameters.

```python
from quantark.asset.equity.param import make_quad_params, make_pde_params

quad_params = make_quad_params(profile="barrier_sensitive")
pde_params = make_pde_params(profile="balanced")
```

Engine parameter presets accept either preset names or explicit config objects; see the docstrings in `quantark/asset/equity/param/` for the full schema.

## RQMC Target Std Scaling (MC Benchmarking)

Randomized QMC uses a target standard error for adaptive batching. For large
notionals, prefer relative scaling to avoid overly strict absolute tolerances:

```python
from quantark.asset.equity.param import MCParams

mc_params = MCParams(
    num_paths=100000,
    rqmc_target_std=1e-4,  # 1 bp relative
    rqmc_target_std_mode="relative_notional",
    rqmc_paths_mode="total",
)
```

## Running the Demo

A comprehensive demonstration is provided:

```bash
python example/european_option_demo.py
```

The demo showcases:
1. European Call option pricing and Greeks
2. European Put option pricing and Greeks
3. Put-Call Parity verification
4. Comparison between analytical and numerical Greeks

### FX portfolio across the full risk stack

```bash
python example/fx_portfolio_risk_demo.py   # one FX book through all four risk modules
python example/fx_stress_test_demo.py      # FX stress testing (spot/vol/two-rate shocks)
python example/fx_var_demo.py              # FX VaR (parametric / historical / Monte Carlo)
python example/fx_dynamic_scenario_demo.py # multi-day FX path simulation
python example/fx_backtest_demo.py         # FX delta-neutral hedging backtest
```

## Project Structure

All library code lives under the `quantark/` package (canonical `quantark.*` imports).

```
quantark/
├── asset/              # Asset classes
│   ├── equity/
│   │   ├── engine/     # analytical/, mc/, pde/, quad/  (+ base, event_stats)
│   │   ├── process/    # bsm/  (Heston / LocalVol / SLV live in volmodels/)
│   │   ├── product/    # option/ (vanilla, exotics, autocallables), deltaone/, swap/ (TRS)
│   │   ├── lifecycle/  # shared autocallable lifecycle core
│   │   ├── param/      # engine parameters & profiles
│   │   ├── analysis/, report/, riskmeasures/
│   ├── bond/          # coupon bonds, FRNs, convertibles, forwards, futures, options
│   │   └── engine/    # analytical/, discount/, pde/, tree/, convertible/
│   ├── rate/          # IRS, FRA, cap/floor, swaption
│   ├── fx/            # Garman-Kohlhagen + exotics (barrier, sharkfin, range accrual, TARF/TARN)
│   │   └── engine/    # analytical/ (incl. Vanna-Volga, FX-Heston), mc/, pde/
│   └── credit/        # CDS, basket CDS (hazard-curve engine)
├── volmodels/          # Heston, Local Vol (Dupire), SLV kernels + Black-Scholes / curves
├── montecarlo/         # Shared QMC layer (Sobol, Brownian bridge, RQMC, variance reduction)
├── param/              # Market data: quote/, rrf/, div/, vol/ (incl. sabr/, vannavolga/)
├── priceenv/           # Pricing environments (equity & FX)
├── portfolio/          # Position tracking: equity/, fi/, fx/, credit/
├── var/                # Value-at-Risk (historical / parametric / monte_carlo) + backtest/
├── stresstest/         # Scenario-based stress testing (all asset classes)
├── dynamicscenario/    # Multi-day scenario simulation
├── backtest/           # Hedging-strategy backtesting (delta / DV01 / convexity-neutral)
├── simm/               # ISDA SIMM v2.6 initial margin (engines, crif, calibration, dashboard)
├── saccr/              # Basel SA-CCR counterparty EAD (RC + PFE)
├── sacva/              # Basel SA-CVA (MAR50) CVA capital
├── cashleg/            # Deterministic cash-flow legs for structured products
├── rfq/                # Request-for-quote service & registry
└── util/               # enum/, numerical/, calendar, market-data adapters, exceptions

example/                # 90+ runnable example scripts
test/                   # Unit-test suite
```

## Exception Hierarchy

QuantArk uses a professional exception hierarchy:

- `QuantArkException`: Base exception
  - `ValidationError`: Invalid input parameters
  - `NumericalError`: Numerical instability/convergence issues
  - `MarketDataError`: Missing or invalid market data
  - `PricingError`: General pricing failures

## Numerical Stability

The Black-Scholes engine includes extensive checks for numerical stability:

- Input parameter validation (spot, strike, volatility, rates)
- Overflow protection in exponential calculations
- Boundary condition handling (near expiry, deep ITM/OTM)
- Sanity checks on computed prices vs intrinsic values
- Extreme parameter detection and rejection

## Roadmap

### Delivered
- [x] American options (analytical and numerical methods)
- [x] Asian, barrier, digital, one-touch, and double-barrier options
- [x] Autocallables (Snowball, Phoenix, KO-Reset) and structured products (range accrual, accumulator, sharkfin)
- [x] Monte Carlo, PDE, Quadrature, and Tree engines
- [x] Heston, Local Volatility (Dupire), and Stochastic Local Vol models
- [x] FX derivatives (vanilla, exotic, TARF/TARN) with SABR & Vanna-Volga smiles
- [x] Credit derivatives (single-name and basket CDS)
- [x] Fixed income instruments (bonds, swaps, FRAs, caps/floors, swaptions)
- [x] Portfolio VaR, stress testing, dynamic scenarios, and hedging backtests
- [x] Regulatory capital: ISDA SIMM v2.6, SA-CCR, SA-CVA

### In progress / planned
- [ ] Multi-asset and hybrid derivatives
- [ ] Unified calibration framework
- [ ] Full XVA suite (FVA, MVA, KVA)
- [ ] Performance optimization (Cython, GPU)
- [ ] Real-time risk metrics & market data integration
- [ ] Portfolio optimization
- [ ] Cloud deployment support

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the existing code structure and style
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Validate inputs and handle edge cases
5. Add professional error handling

## Disclaimer

QuantArk is provided for research and educational purposes. It is not
investment advice, and no warranty is made as to the correctness of any
price, risk figure, or model output. Validate independently before any
production or trading use. See the LICENSE file for the full terms.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

QuantArk builds on decades of published quantitative-finance research. In particular:

- **Black-Scholes-Merton** option-pricing model — Fischer Black, Myron Scholes, and Robert Merton
- **American options** — Barone-Adesi & Whaley quadratic approximation; Bjerksund & Stensland (1993/2002); Longstaff & Schwartz least-squares Monte Carlo
- **Stochastic & local volatility** — Steven Heston (1993); Bruno Dupire local-volatility construction; stochastic-local-vol (SLV) calibration
- **FX volatility smiles** — Patrick Hagan et al. (SABR); the Vanna-Volga market approach
- **Credit** — standard ISDA hazard-rate / reduced-form CDS pricing conventions
- **Regulatory capital** — ISDA SIMM v2.6, and the Basel Committee's SA-CCR and SA-CVA (MAR50) standards
- **Numerical methods** — Sobol sequences, Brownian-bridge and randomized-QMC variance reduction, and finite-difference PDE schemes from standard computational-finance texts
- Greeks formulas from standard derivatives textbooks
- Design patterns inspired by QuantLib and similar professional libraries
