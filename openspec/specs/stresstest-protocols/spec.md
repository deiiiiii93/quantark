# stresstest-protocols Specification

## Purpose
TBD - created by archiving change add-fi-stresstest. Update Purpose after archive.
## Requirements
### Requirement: Base Stress Engine Protocol

The system SHALL provide a `BaseStressEngine` protocol that defines the common interface for running stress scenarios across asset classes.

The protocol MUST:
- Accept any `BasePortfolio` implementation plus a list of `Scenario` objects.
- Expose `run_static_scenarios()`, `supports_portfolio()`, and `evaluate_scenario()` hooks.
- Return results that satisfy the `StressResultEnvelope` contract.

#### Scenario: Equity Engine Implements Protocol
- **GIVEN** the existing equity stress engine
- **WHEN** it is refactored to implement `BaseStressEngine`
- **THEN** callers can continue invoking `run_static_scenarios()` without code changes
- **AND** the returned results satisfy the envelope contract.

#### Scenario: FI Engine Implements Protocol
- **GIVEN** a new FI stress engine
- **WHEN** it implements `BaseStressEngine`
- **THEN** it advertises support for `FIPortfolio`
- **AND** returns results that downstream exporters can consume via the same interface.

### Requirement: Stress Result Envelope

The system SHALL define a `StressResultEnvelope` contract that all stress engines MUST return.

The envelope MUST include:
- Baseline portfolio value and scenario-level P&L.
- Optional portfolio-level Greeks and/or FI risk measures stored in an `extra_metrics` namespace keyed by asset class.
- Serialized scenario metadata (curve shifts, spread bumps) for exporters.

#### Scenario: Equity Result Compatibility
- **GIVEN** the equity engine returns only P&L and Greeks
- **WHEN** the results are inspected via the envelope
- **THEN** the `extra_metrics` namespace is either empty or contains equity-specific fields
- **AND** FI-aware exporters can ignore missing FI metrics without errors.

#### Scenario: FI Result Extension
- **GIVEN** the FI engine returns DV01 and convexity vectors
- **WHEN** the envelope is serialized
- **THEN** the FI metrics appear under `extra_metrics["fi"]`
- **AND** reporters can render DV01 waterfalls alongside P&L.

### Requirement: Stress Metrics Adapter

The system SHALL provide a `StressMetricsAdapter` protocol that lets asset-specific engines calculate metrics from stressed portfolios.

The adapter MUST:
- Accept the stressed portfolio, baseline portfolio, and stressed pricing environments.
- Produce a dictionary of asset-specific metrics (e.g., DV01, convexity, gamma) plus attribution metadata.
- Be pluggable so new asset classes can register adapters without modifying the core engine.

#### Scenario: Default Equity Adapter
- **GIVEN** no custom adapter is provided
- **WHEN** an equity portfolio is stressed
- **THEN** the default adapter computes delta/gamma/vega as before
- **AND** metrics flow into the envelope.

#### Scenario: FI Adapter Injection
- **GIVEN** an FI engine registers a `StressMetricsAdapter`
- **WHEN** `run_static_scenarios()` executes
- **THEN** the adapter calculates DV01/convexity metrics per scenario
- **AND** the engine embeds them in the envelope without altering core logic.

