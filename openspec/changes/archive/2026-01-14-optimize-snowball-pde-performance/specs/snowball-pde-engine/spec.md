## ADDED Requirements
### Requirement: PDE performance profiling
The system SHALL provide a repeatable benchmark to measure Snowball PDE solve time across representative grid sizes.

#### Scenario: Benchmark execution
- **GIVEN** a standard snowball configuration
- **WHEN** the benchmark script is run
- **THEN** it reports timing for PDE solves at multiple grid/time step sizes

### Requirement: Efficient linear solves
The system SHALL minimize per-step linear solve overhead by reusing factorizations and/or using banded solvers when the operator is tridiagonal.

#### Scenario: Reuse factorization
- **GIVEN** repeated time steps with identical (dt, theta)
- **WHEN** time stepping proceeds
- **THEN** the solver reuses cached factorizations instead of recomputing them

#### Scenario: Banded solve path
- **GIVEN** a tridiagonal spatial operator
- **WHEN** the solver executes time stepping
- **THEN** it uses a banded/tridiagonal solve path for the linear system
