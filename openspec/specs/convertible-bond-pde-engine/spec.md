# convertible-bond-pde-engine Specification

## Purpose
TBD - created by archiving change add-convertible-bond. Update Purpose after archive.
## Requirements
### Requirement: Jump-Diffusion PDE Engine (Bloomberg OVCV Model)
The system SHALL provide a `ConvertibleBondJumpDiffusionEngine` implementing the Bloomberg OVCV jump-diffusion model for pricing convertible bonds with explicit credit risk.

#### Scenario: Basic pricing with jump-diffusion
- **WHEN** `price()` is called on a `ConvertibleBond` with `ConvertibleBondJumpDiffusionEngine`
- **THEN** the engine returns the convertible bond price using the jump-diffusion PDE

#### Scenario: Jump-diffusion PDE formulation
- **WHEN** the PDE is solved
- **THEN** the equation `[r(t)+h(t)]B = B_t + 0.5*σ²*S²*B_SS + [r(t)-q(t)+η*h(t)]*S*B_S + h(t)*max[R*F, K(t)*S*(1-η)]` is solved

#### Scenario: Hazard rate incorporation
- **WHEN** hazard_rate λ is specified
- **THEN** the PDE includes credit-adjusted discounting term `(r+λ)*B` and default recovery term `λ*D`

#### Scenario: Stock jump on default
- **WHEN** stock_jump_on_default η is specified
- **THEN** the drift term includes `η*λ*S` adjustment and recovery includes `(1-η)*S*K` component

#### Scenario: Default recovery calculation
- **WHEN** default occurs
- **THEN** recovery value is `max(recovery_rate*face_value, conversion_ratio*stock*(1-η))`

### Requirement: Tsiveriotis-Fernandes Decomposition Engine
The system SHALL provide a `ConvertibleBondTFEngine` implementing the Tsiveriotis-Fernandes coupled PDE model that separates cash and equity components for correct credit treatment.

#### Scenario: Basic pricing with TF model
- **WHEN** `price()` is called on a `ConvertibleBond` with `ConvertibleBondTFEngine`
- **THEN** the engine returns the convertible bond price using coupled PDE system

#### Scenario: Coupled PDE system
- **WHEN** the TF model is solved
- **THEN** two coupled PDEs are solved: one for total value u, one for cash-only value v

#### Scenario: Credit-adjusted discounting of components
- **WHEN** solving the coupled system
- **THEN** equity component (u-v) is discounted at risk-free rate r, cash component v is discounted at risky rate (r+r_c)

#### Scenario: COCB (Cash-Only Convertible Bond) output
- **WHEN** `price_with_details(return_decomposition=True)` is called
- **THEN** both total value and cash-only component value are returned

#### Scenario: Boundary conditions for COCB
- **WHEN** conversion occurs at a boundary
- **THEN** v=0 (no cash component in equity)
- **WHEN** put is exercised
- **THEN** v=put_price (full value is cash)

### Requirement: PDE Grid Configuration
The system SHALL accept PDE configuration parameters for spatial and temporal discretization.

#### Scenario: Grid resolution specification
- **WHEN** `ConvertibleBondJumpDiffusionEngine` or `ConvertibleBondTFEngine` is initialized with `num_space_steps=200, num_time_steps=500`
- **THEN** the PDE grid has 200 spatial points and 500 time steps

#### Scenario: Default grid configuration
- **WHEN** a PDE engine is initialized without grid parameters
- **THEN** default of 100 space steps and 200 time steps is used

#### Scenario: Spatial domain specification
- **WHEN** `spot_min_mult=0.01, spot_max_mult=5.0` is specified
- **THEN** the spatial domain spans from 1% to 500% of current spot price

### Requirement: Time Stepping Schemes
The system SHALL support multiple time stepping schemes for PDE solution.

#### Scenario: Crank-Nicolson scheme
- **WHEN** `scheme="crank_nicolson"` is specified
- **THEN** the engine uses Crank-Nicolson (implicit-explicit average) for time stepping

#### Scenario: Implicit Euler scheme
- **WHEN** `scheme="implicit_euler"` is specified
- **THEN** the engine uses fully implicit Euler for time stepping

#### Scenario: Rannacher smoothing
- **WHEN** `rannacher_steps=4` is specified
- **THEN** the first 4 time steps use implicit Euler to smooth discontinuities at maturity

#### Scenario: Default scheme
- **WHEN** no scheme is specified
- **THEN** Crank-Nicolson is used by default

### Requirement: Boundary Conditions
The system SHALL implement appropriate boundary conditions for convertible bond PDE.

#### Scenario: Maturity boundary condition
- **WHEN** at maturity T
- **THEN** bond value is `max(face_value + accrued, conversion_ratio * stock_price)`

#### Scenario: High stock price boundary
- **WHEN** stock price is very high (deep in-the-money)
- **THEN** bond value approaches parity (conversion_ratio * stock_price)

#### Scenario: Low stock price boundary
- **WHEN** stock price is very low (deep out-of-the-money)
- **THEN** bond value approaches present value of remaining cash flows at risky rate

#### Scenario: Free boundary for American conversion
- **WHEN** conversion is optimal before maturity
- **THEN** the exercise boundary is computed as part of the solution

### Requirement: Coupon and Dividend Handling
The system SHALL incorporate coupon payments and dividends in the PDE solution.

#### Scenario: Coupon as source term
- **WHEN** coupon payment occurs at time t_i
- **THEN** coupon is added as Dirac delta source term in PDE at t_i

#### Scenario: Continuous dividend yield
- **WHEN** dividend_yield q is specified
- **THEN** the drift term in PDE is (r-q)*S*B_S

#### Scenario: Discrete dividends
- **WHEN** discrete dividends are specified
- **THEN** the PDE solution accounts for stock price jumps at dividend dates

### Requirement: Greeks from PDE Grid
The system SHALL compute Greeks directly from the PDE solution grid.

#### Scenario: Delta from grid
- **WHEN** `calculate_delta()` is called after pricing
- **THEN** delta is computed as ∂B/∂S from the spatial derivative on grid

#### Scenario: Gamma from grid
- **WHEN** `calculate_gamma()` is called after pricing
- **THEN** gamma is computed as ∂²B/∂S² from second spatial derivative on grid

#### Scenario: Theta from grid
- **WHEN** `calculate_theta()` is called after pricing
- **THEN** theta is computed as -∂B/∂t from the time derivative on grid

### Requirement: PricingEnvironment Integration
The system SHALL accept `PricingEnvironment` for market data input.

#### Scenario: Market data extraction
- **WHEN** `price(product, pricing_env)` is called
- **THEN** spot price, risk-free rate curve, and volatility surface are extracted from pricing_env

#### Scenario: Time-dependent parameters
- **WHEN** rate curve or vol surface have term structure
- **THEN** the PDE uses time-dependent r(t) and σ(t)

### Requirement: Validation and Error Handling
The system SHALL validate inputs and provide clear error messages.

#### Scenario: Missing market data
- **WHEN** pricing without required market data
- **THEN** a `MarketDataError` is raised with specific missing field

#### Scenario: Numerical instability detection
- **WHEN** explicit scheme is used with CFL condition violation
- **THEN** a `NumericalError` is raised suggesting implicit scheme or finer grid

#### Scenario: Product type validation
- **WHEN** a non-ConvertibleBond product is passed to the engine
- **THEN** a `ValidationError` is raised indicating unsupported product type

### Requirement: Numerical Stability
The system SHALL ensure numerical stability of PDE solutions.

#### Scenario: CFL condition check for explicit schemes
- **WHEN** explicit Euler is used
- **THEN** the engine validates CFL condition and warns if violated

#### Scenario: Smooth initial condition handling
- **WHEN** the payoff at maturity has discontinuities (conversion boundary)
- **THEN** Rannacher smoothing is applied by default

#### Scenario: Convergence verification
- **WHEN** `verify_convergence=True` is specified
- **THEN** the engine runs at two grid resolutions and reports relative error

### Requirement: Exact conversion probability output
The system SHALL compute and return an exact (within the PDE discretization) risk-neutral probability of eventual conversion for PDE-based convertible bond engines, consistent with the same optimal policy constraints used for pricing (conversion, call, put).

#### Scenario: Boundary cases
- **WHEN** the bond is configured so conversion is optimal immediately for all relevant stock prices
- **THEN** `conversion_probability` returned by the PDE engine is `1.0` (within numerical tolerance)
- **WHEN** conversion is never allowed during the valuation horizon
- **THEN** `conversion_probability` returned by the PDE engine is `0.0` (within numerical tolerance)

#### Scenario: Probability bounds
- **WHEN** a PDE engine computes conversion probability on a grid
- **THEN** the probability is in `[0, 1]`

### Requirement: Time-Dependent Parameters in PDE Solution
The system SHALL query interest rates and volatility at each time step during PDE backward induction, rather than using single values at maturity.

#### Scenario: Forward rate per time step
- **GIVEN** a `ConvertibleBondJumpDiffusionEngine` or `ConvertibleBondTFEngine` pricing a convertible bond
- **WHEN** the PDE time-stepping loop processes time step from $t$ to $t + \Delta t$
- **THEN** the engine uses `rate_curve.get_forward_rate(t, t + dt)` for the local interest rate

#### Scenario: Local volatility per time step
- **GIVEN** a PDE engine pricing a convertible bond
- **WHEN** the PDE time-stepping loop processes time $t$
- **THEN** the engine derives a per-step effective volatility `sigma_step(t, t+dt)` from implied vols via `pricing_env.get_vol(strike, time_to_maturity)` using total variance differences

#### Scenario: Non-flat rate curve produces different price
- **GIVEN** a convertible bond with 4-year maturity
- **AND** a stepped rate curve: 1% for years 0-2, 9% for years 2-4
- **WHEN** pricing with a PDE engine
- **THEN** the price differs from a flat 5% curve by more than 0.1% of face value

#### Scenario: Matrices built with time-local parameters
- **GIVEN** a PDE engine with Crank-Nicolson or implicit Euler scheme
- **WHEN** `_build_matrices()` is called for time step $n$
- **THEN** the method receives the forward rate and local volatility for that specific time step

