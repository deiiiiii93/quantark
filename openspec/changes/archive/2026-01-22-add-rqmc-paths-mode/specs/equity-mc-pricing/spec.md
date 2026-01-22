# equity-mc-pricing Specification (Delta)

## ADDED Requirements

### Requirement: RQMC Total Paths Mode
The system SHALL allow RQMC to interpret `num_paths` as a total expected path count
when configured, and compute a Sobol-ideal per-batch path count automatically.

#### Scenario: Per-batch mode (default)
- **GIVEN** `rqmc_paths_mode="per_batch"`
- **WHEN** RQMC pricing is executed
- **THEN** the engine uses `num_paths` per batch without adjustment

#### Scenario: Total paths mode
- **GIVEN** `rqmc_paths_mode="total"` and `num_paths=100000` with `rqmc_max_batches=4`
- **WHEN** RQMC pricing is executed
- **THEN** per-batch paths are set to `next_power_of_two(ceil(100000 / 4))`

