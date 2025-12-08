## ADDED Requirements
### Requirement: One-touch and no-touch analytical pricing
The system SHALL provide a closed-form analytical engine for one-touch and no-touch options supporting continuous, discrete (shifted), and expiry-only monitoring, with pay-at-hit vs pay-at-expiry handling for one-touch.

#### Scenario: Continuous monitoring payout
- **WHEN** pricing a `OneTouchOption` with `observation_type=CONTINUOUS`
- **THEN** the engine SHALL use the closed-form one-touch formula from `onetouch_analytical_engine.md` without tenor/365 scaling
- **AND** SHALL pay immediately when `payment_at_hit=True` (no discounting) or discount to expiry when `payment_at_hit=False`

#### Scenario: Discrete monitoring with barrier shift
- **WHEN** `observation_type=DISCRETE` with a regular observation schedule and fixed rebate across observations
- **THEN** the engine SHALL apply the beta barrier shift (β=0.5825971579) via `apply_barrier_shift(dt=freq)` before using the continuous formula
- **AND** SHALL raise `ValidationError` for irregular schedules or varying payoffs

#### Scenario: Expiry-only monitoring fallback
- **WHEN** `observation_type=EXPIRY`
- **THEN** the engine SHALL price using digital-style probabilities (one-touch pays rebate * exp(-rT) * P(hit), no-touch pays rebate * exp(-rT) * P(not hit))
- **AND** SHALL ignore `payment_at_hit` for expiry monitoring

#### Scenario: Ignore payment_at_hit for no-touch
- **WHEN** pricing a NO_TOUCH option with any observation type
- **THEN** the engine SHALL ignore `payment_at_hit` and only pay at expiry if the barrier has not been hit (continuous/discrete) or the terminal condition is not breached (expiry)

#### Scenario: Input validation and maturity edge
- **WHEN** spot, barrier, or volatility are non-positive, or maturity is negative
- **THEN** the engine SHALL raise `ValidationError`
- **AND** WHEN maturity is effectively zero, it SHALL return the immediate payoff based on whether the barrier is already hit for continuous/discrete monitoring

#### Scenario: Product type enforcement
- **WHEN** a non-`OneTouchOption` product is passed
- **THEN** the engine SHALL raise `PricingError`

