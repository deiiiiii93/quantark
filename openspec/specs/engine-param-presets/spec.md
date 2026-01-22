# engine-param-presets Specification

## Purpose
TBD - created by archiving change add-engine-param-presets. Update Purpose after archive.
## Requirements
### Requirement: Preset Profiles for Engine Params
The system SHALL provide named presets for QUAD and PDE parameter objects to support
batch-friendly configuration (`fast`, `balanced`, `accurate`, `barrier_sensitive`,
`reverse_sensitive`).

#### Scenario: Use a balanced QUAD preset
- **GIVEN** `profile="balanced"` for QUAD params
- **WHEN** a user requests QUAD params via the preset registry or factory
- **THEN** a `QuadParams` instance is produced with the balanced preset values

#### Scenario: Use a reverse-sensitive PDE preset
- **GIVEN** `profile="reverse_sensitive"` for PDE params
- **WHEN** a user requests PDE params via the preset registry or factory
- **THEN** a `PDEParams` instance is produced with the reverse-sensitive preset values

### Requirement: Factory Helpers for Params
The system SHALL provide factory helpers to construct `QuadParams` and `PDEParams`
using a profile and optional overrides.

#### Scenario: Override a preset
- **GIVEN** `profile="accurate"` and an override `{ "num_std_devs": 10 }`
- **WHEN** the factory creates QUAD params
- **THEN** the resulting params use the accurate preset values with `num_std_devs=10`

#### Scenario: Auto-tuning with product hints
- **GIVEN** a reverse Phoenix product and `reverse=True`
- **WHEN** the factory creates QUAD params
- **THEN** the params reflect reverse-aware alignment and smoothing defaults

### Requirement: Config Loaders for Batch Pipelines
The system SHALL support constructing `QuadParams` and `PDEParams` from dict/YAML/JSON
configs that specify a profile and optional overrides.

#### Scenario: Load from YAML config
- **GIVEN** a YAML file containing `profile: barrier_sensitive` and overrides
- **WHEN** a user calls `QuadParams.from_config(path)`
- **THEN** a `QuadParams` instance is created with the preset and overrides applied

#### Scenario: Unknown profile
- **GIVEN** `profile="ultra_fast"`
- **WHEN** the loader or factory is called
- **THEN** a `ValidationError` is raised listing valid profile names

