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

#### Intraday pricing
- **Timestamp valuation** (`quantark.intraday`): timezone-aware valuation instants on a trading-session calendar with a
  sub-day variance profile, confirmed and assumed (provisional) fixings, and a structured result that separates the
  contingent claim, pending receivables and paid cash
- **Routes** for digitals, barriers/one-touches and autocallables through QUAD V2, PDE, MC and admissible closed forms,
  each reporting its numerical status (PDE resolution, MC sampling error); point and desk-bump Greeks, a declared-step
  intraday theta, roll-through-events scenarios, spot curves and batch aggregation
- **Evidence**: an independent Gaussian-transition reference gates every route on a one-day-to-one-second
  time-to-fixing ladder; see `quantark/intraday/README.md` and the generated
  [intraday capability matrix](docs/execution/intraday-capability-matrix.md)

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

Optional accelerator (Numba): `pip install "quantark[accel]"` compiles the Monte Carlo QE
kernels and the PDE mesh builder. Results are bit-identical with or without it; a cold PDE
solve is faster with it.

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

For QUAD batch pricing, you can use named presets, factory helpers, or YAML/JSON
configs instead of tuning many individual parameters. PDE has no profile tier:
`PDEParams` is small enough to write out, and `PDEParams.from_config` reads the
same field names from a mapping, YAML, or JSON.

```python
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.param import PDEParams, make_quad_params

quad_params = make_quad_params(profile="barrier_sensitive")
pde_params = PDEParams(accuracy="standard")
pde_barrier = PDEParams(grid=GridConfig(points=600, steps_per_day=6.0))
pde_loaded = PDEParams.from_config({"accuracy": "high", "grid": {"time_steps": 800}})
```

Quad presets accept either preset names or explicit config objects; see the docstrings in `quantark/asset/equity/param/` for the full schema.

### PDE parameters (0.8.0)

Every 1D PDE route (`PDEEngine` and its solvers, the local-vol vanilla and barrier
solvers, DCN) is configured by one `PDEParams`, keyword arguments only:

| Field | When to change it | What happens |
|---|---|---|
| `accuracy` | A faster preview or a finer answer. | Selects the `GridConfig` preset: fast (200 pts, 4/day), standard (400, 4/day), high (800, 8/day). |
| `grid` | Override one preset field. | Each non-`None` `GridConfig` field replaces that preset value; the rest inherit. |
| `stepper` | A different time weighting or damping schedule, or you are writing a stepper. | `None` is the built-in `CrankNicolson()` (below). A custom stepper runs on every 1D route. |
| `operator` | You are writing an operator (an exponentially fitted stencil, say). | `None` is the built-in `CentralDifference()`: the central-difference tridiagonal with upwind rows on zero-diffusion steps. |
| `readout` | You are writing a readout. | `None` is the built-in `StandardReadout()`: linear price interpolation, delta and gamma from the local cubic. |
| `event_projection` | Validating the event treatment. | `None` is `"cell_average"` on every core route: the autocallables, the barrier / double-barrier / one-touch / double-one-touch discrete observations, and the DCN's daily dates. `"nodal"` assigns whole nodes (the 0.7.0 base-family mask). Continuous monitoring and a valuation-date observation stay nodal either way. |
| `cache_enabled` | Memory pressure, or a fresh solve every call. | The same numbers to round-off: a cached run shares one factorization across intervals whose `dt` differ by an ULP. |

`CrankNicolson` (`quantark.asset.equity.engine.pde.core.components`) carries the
scheme's settings:

| Setting | When to change it | What happens |
|---|---|---|
| `theta` (0.5) | Ringing that damping does not cure, or a more diffusive scheme. | 0.5 is Crank-Nicolson (second order), 1.0 backward Euler (first order). |
| `event_theta` (1.0) | Studying damping strength after events. | The weight on the event-damped steps after each interior event. |
| `damping` (`None`) | See "Damping" below. | `"half_step"`: each damped interval is split in two; both halves take `event_theta` below an event and 1.0 at the terminal node; `"full_step"`: whole damped steps. `None` is the route's form. |
| `terminal_damping_steps` (`None`) | Checking the payoff kink's smoothing. | Damped intervals off the payoff; `None` is the route's count, 0 disables. |
| `event_damping_steps` (`None`) | Studying damping strength after events. | Damped intervals below each interior event node; `None` is the route's count, 0 disables. |

The route defaults (`DAMPING_DEFAULTS`, form, terminal count, event count):

| Route | Default |
|---|---|
| Core, product without continuous knock-in | half-step, 1, 1 |
| Core, product with continuous knock-in | full-step, 1, 2 |
| 2D (Heston, SLV) | full-step, 1, 2 |

Every 1D route is a core route, DCN and the local-vol vanilla and barrier included (0.10.0
removed their own rows).

The class is read from the contract alone (`continuous_knock_in`), never from the spot or
the knocked-in state, so a spot bump cannot change it. **A count means intervals of the
form in force:** `CrankNicolson(event_damping_steps=2)` with no form gives two halved
intervals on every core solve without continuous knock-in, where 0.7.0 gave two whole
steps; under continuous knock-in it keeps its 0.7.0 meaning. `damping="full_step"` with
explicit counts restores the old meaning everywhere; on its own it takes the route's counts,
`(full_step, 1, 1)` on the core, which 0.7.0 never used. The 0.7.0 schedule is `pde_070` below.
A `CrankNicolson` subclass is a custom stepper, used as given: set all three fields.

```python
from quantark.asset.equity.engine.pde.core.components import CrankNicolson

pde_euler = PDEParams(stepper=CrankNicolson(theta=1.0))
pde_070 = PDEParams(stepper=CrankNicolson(damping="full_step", terminal_damping_steps=1, event_damping_steps=2))
pde_loaded = PDEParams.from_config({"accuracy": "high", "stepper": {"terminal_damping_steps": 0}})
```

**Damping: when to change it.** Time error at standard against each case's own limit
(pv unless stated; spec `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md` §4.2):

| Case | Full-step (1, 2) | Two intervals halved | Half-step (1, 1), the default | Undamped |
|---|---|---|---|---|
| European | −8.8e-7 | −8.8e-7 | −3.0e-7 | +2.8e-7 |
| Barrier, monthly | +3.6e-5 | +1.8e-5 | +7.4e-6 | −3.1e-6 |
| Snowball, monthly discrete KI | +9.5e-5 | +4.8e-5 | +2.3e-5 | −2.2e-6 |
| Snowball, daily KI | +3.0e-3 | +1.5e-3 | +6.3e-4 | +4.3e-5 |
| Snowball, daily KI, spot 86: delta | +1.1e-3 | +5.3e-4 | +1.5e-4 | −2.1e-4 |
| Snowball, daily KI, spot 86: gamma | −5.6e-4 | −2.2e-4 | +1.5e-4 | −1.2e-3 |

Undamped Crank-Nicolson has the smallest pv error and the worst gamma next to a barrier.
Under discrete knock-in the half-step default cuts the pv time error three to eight times
on the Phoenix and KO-reset too (flat market, daily KI: Phoenix +3.7e-3 → +7.4e-4,
KO-reset +1.9e-3 → +3.9e-4) and on the local-vol Snowball and Phoenix (calm surface,
daily KI: +9.6e-3 → +2.0e-3). It costs one extra step per event and one at the terminal
node: daily KI goes from 1,008 to 1,260 steps. Where it is worse, and stays the default:

- gamma one point above a daily knock-in barrier on the local-vol calm surface at
  standard (Snowball 3.7e-4 → 1.5e-3, Phoenix 1.8e-4 → 1.6e-3); pv improves seven times
  and delta eleven to thirteen times in the same cells, and the Snowball's gamma is better
  from 8 steps per day. Raise `steps_per_day` if that gamma matters;
- delta and gamma on the local-vol crash surface at spot 102.5%, under a coupon barrier at
  102.9% and a KO barrier at 103% (1.7e-5 to 4.9e-5 and 4.6e-5); pv improves six times.

The local-vol Phoenix takes the half-step default with the core, but no certification
study covers it: it is not a certified route.

Products with continuous knock-in keep full-step: there the first-passage correction
needs `sigma * sqrt(dt)` larger than a cell, and half-step moved seven of nine resolved
cases away from an exact reference. **On a continuous-knock-in product, refine `points`
together with `steps_per_day`**: time-only refinement drifts (Snowball `ordinary`
96.46045, 96.46211, 96.46310, 96.46417 at 8, 16, 32, 64 steps per day). The correction
carries an error of its own that joint refinement does not remove: about 5e-4 contracts in
pv on the certified cases, 9e-3 on a three-month Snowball, 1.3e-2 at 8% volatility one
point above the barrier.

**Projection.** A discrete observation of the barrier family enters each dual cell at the
record's exact level instead of overwriting whole nodes. Monthly observations, flat
market, against exact-GBM Monte Carlo (spec §4.6):

| Product | Nodal mask (0.7.0), 400 points | Dual-cell projection, 400 points |
|---|---|---|
| Up-and-out call, K 100, B 130 | −0.81%, −21 SE | −0.05%, −1.2 SE |
| Down-and-in put, K 100, B 85 | +0.22%, +8.8 SE | +0.05%, +2.2 SE (the far-field row; fixed in 0.8.0) |
| Double knock-out call, K 100, (80, 120) | +1.89%, +34 SE | +0.01%, +0.2 SE |
| One-touch up 115 | +0.03%, +1.5 SE (+0.9% at 200 points) | −0.02%, −0.9 SE |
| Double one-touch (85, 115) | −0.45%, −28 SE | −0.02%, −1.1 SE |

Greeks, largest error at 400 points over 93 knock-out and touch cells against an exact
recursion:

| First observation in | Quantity | Nodal (0.7.0) | Projection | Projection and half-step (0.8.0) |
|---|---|---|---|---|
| 1 month | pv | 4.1e-2 | 1.5e-4 | 1.7e-4 |
| | grid delta / gamma | 1.7e-3 / 2.2e-4 | 1.8e-5 / 1.9e-6 | 9.8e-6 / 1.7e-6 |
| 5 days | pv | 5.3e-2 | 3.2e-4 | 2.1e-4 |
| | grid delta / gamma | 1.9e-3 / 8.3e-4 | 1.8e-4 / 7.2e-5 | 4.9e-5 / 2.2e-5 |
| 1 day | pv | 4.5e-2 | 7.9e-3 | 1.1e-3 |
| | grid delta / gamma | 9.4e-3 / 9.7e-3 | 3.4e-3 / 9.1e-3 | 8.5e-4 / 1.8e-3 |

`GridConfig` fields: `points`, `steps_per_day` (a density per **business day of the
pricing environment**: `PricingEnvironment.bus_days_in_year`, carried on
`GridRequest.day_count`) or `time_steps` (a total budget over the horizon; exactly one
time basis is in force), `eps_crit`, `num_std`, `bounds`, `max_points`, `max_steps`.
A 244-day environment gives a one-year Snowball 972 steps where a 252-day one gives
1,008.

`max_steps` (12,000) caps the delivered step count, half-step nodes included. **A grid over
it raises** `ValidationError` naming the count it needs and both fixes, raise
`GridConfig(max_steps=...)` or lower `steps_per_day`; it is never scaled down. A
five-year `high` Snowball asks 10,080 and fits; on a 252-day environment the default cap
holds a little under six years at `high` and twelve at `standard` (half-step nodes count),
and a six-year monthly Snowball at `high` raises where 0.7.0 scaled it.

Since 0.10.0 every 1D route honours every field. The local-vol vanilla and barrier
solvers, `DCNPDEEngine` and `LocalVolDCNPDEEngine` march on the shared core with the
shared presets, the cell-average projection, the core's damping rows, the shared domain
rule and the built-in readout, and take a custom stepper, operator or readout like any
other 1D solver. `quantark.asset.equity.engine.pde.route_applicability.ROUTE_RULES` keeps
one row, the 2D Heston / SLV autocallables': they damp in whole steps, and an explicit
`damping="half_step"` raises `ValidationError` there. What the shared presets mean on
the moved routes:

- The local-vol vanilla and continuously monitored barrier take the core's time density,
  not 0.9.0's step budget (200 at `fast` and `standard`, 400 at `high`).
  `GridConfig(time_steps=200)` restores the `standard` budget at a cost
  (price units at spot 100) of at most 1.4e-5 on a vanilla and 3.4e-5 on a continuous
  barrier, but up to 5.4e-4 monthly, 4.3e-3 weekly and 2.6e-2 daily.
- A daily monitored barrier at `standard` (4 steps per day) carries about 1e-3 of time
  error: +9.3e-4 / +2.1e-4 / +1.1e-5 at 4 / 8 / 16 steps per day on a constant local-vol
  surface. Use `GridConfig(steps_per_day=8.0)` where that matters.
- The DCN takes the shared presets (200 / 400 / 800 points, 4 / 4 / 8 steps per day) in
  place of its own table: at most 0.38 / 0.35 / 0.10 bp of notional against an exact
  recursion on 24 cells, where 0.9.0's engine was off by up to 5.2 / 7.2 / 7.7 bp and did
  not converge in the point count. `DCNPDEResult.num_time_steps` is the delivered step
  count.
- An explicit `bounds` belongs well beyond a barrier: a far-field row is a constant, and
  next to a barrier a constant is wrong (by −0.09 to −0.13 on the autocallables' row and
  0.1 to 3.3 on the single barrier's).

`docs/engine_param_guide.md` gives the measurements behind these numbers.

**What will this solve do?** `explain(product, pricing_env)` on `PDEEngine` and every
1D solver returns the resolved plan from the same preparation `price` runs,
without marching: route, every setting with its source and applicability, space, time,
events, scheme, outputs, identities, and per-operation terminal outcomes. An excerpt of
`str(SnowballPDESolver(PDEParams()).explain(snowball, env))` for a one-year monthly
snowball with continuous knock-in in a flat environment:

```
[route]
  solver       SnowballPDESolver
  adapter      SnowballAdapter
  samples      scalar
  solve_state  (('ki_barrier', 85.0), ('ki_continuous', True), ('knocked_in', False), ('reverse', False))
[time]
  basis            steps_per_day
  value            4.0
  day_count        252
  requested_steps  1008
  actual_steps     1008
  mandatory_nodes  11
[scheme]
  stepper                CrankNicolson
  resolved               {'theta': 0.5, 'event_theta': 1.0, 'terminal_damping_steps': 1, 'event_damping_steps': 2, 'damping': 'full_step'}
  terminal_damped_steps  1
  event_damped_steps     22
  theta_counts           {0.5: 985, 1.0: 23}
  zero_diffusion_steps   0
[outputs]
  groups   ({'name': 'value', 'surfaces': ('alive', 'ki'), 'columns': 2, 'backend': 'banded', 'retention': 'full'},)
  readout  final column
```

Names removed in 0.8.0:

| Removed | Replacement |
|---|---|
| `GridConfig(day_count=...)` (raises `ValidationError`) | `PricingEnvironment(bus_days_in_year=...)`: the grid day is the environment's business days |
| `TimeLayout.fill_scaled`, `.cap_exceeded`; the same keys in `explain` and on `LifeSurfaceSolution` | none: a grid over `max_steps` raises |
| `groups.VALUE_NO_ZERO_DIFFUSION_OVERRIDE` | `groups.VALUE` (Phoenix now takes θ = 1 on zero-diffusion steps) |

The names removed in 0.6.0, 0.7.0 and 0.10.0 (among them `GridLayerMixin`, the local-vol
and DCN preset tables and `resolve_config(profiles=)`) are listed in `CHANGELOG.md`. How the solvers share
one core, and how to write a custom component, is in `docs/pde-core.md`.

### PDE event statistics (0.9.0)

The 1D Snowball, Phoenix and KO-reset solvers (and their local-vol variants) march the
statistics in the price's own groups, so the cashflows add up to the price:

```python
from quantark.asset.equity.engine.pde import SnowballPDESolver, HestonSnowballPDESolver

stats = SnowballPDESolver(PDEParams()).calculate_event_stats(snowball, env)
# pv = Σ knock-out cash + Σ coupon cash + the marched maturity leg, to round-off
assert abs(stats.reconciliation_error) <= 1e-10
stats.provenance          # StatisticsProvenance(statistics_model='flat_vol_1d', price_model='flat_vol_1d')

# 2D Heston / SLV solvers have no native statistics: they raise StatisticsRefused unless asked
# for the flat-vol 1D surrogate, whose provenance names both models
heston = HestonSnowballPDESolver(heston_params, params=PDEParams(), statistics="flat_vol_surrogate")
```

Field semantics (the knock-in fields, the Phoenix memory and EXPIRY coupon cash, the
KO-reset maturity buckets, the valuation-date rules) are in
`quantark/asset/equity/engine/docs/autocallable_event_stats_api.md`.

### Accrual time (unreleased)

An autocallable's cash depends on its accrual terms only. Each knock-out, coupon and knock-in
record carries an `AccrualPeriod` derived when the product is built: a numeric schedule accrues
its quoted fraction (`k/12` is 1/12 per period), a dated one the contract's day count from
`initial_date`. Ageing or re-timing an observation never moves the cash.

```python
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.enum import AccrualEndType

AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT)          # accrue to the payment date, not the fixing
AccrualConfig(accrual_calendar=cal, accrual_bus_days_in_year=244)   # required for a BUSINESS_DAYS accrual
```

A dated contract with annualized cash needs `initial_date`. After changing a built product's
accrual terms or schedules, call `product.validate()` to derive its periods again.

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
├── intraday/           # Timestamp valuation: sessions, variance profiles, provisional fixings, routes, Greeks
├── execution/          # PricingSession kernel: capability matrix, determinism policy, batch / scenarios
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
