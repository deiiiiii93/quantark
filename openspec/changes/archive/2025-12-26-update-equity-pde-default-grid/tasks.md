## 1. Implementation
- [x] 1.1 Add `TimeGrid.build_event_aligned()` and support `time_grid_type="event_aligned"`
- [x] 1.2 Extend `PDEParams` with `auto_grid` and grid-tuning knobs (day-based/event-based)
- [x] 1.3 Implement feature-aware defaults in `BasePDESolver` (event-aligned time grid, adaptive space grid for barriers)
- [x] 1.4 Apply Rannacher smoothing after event times for default meshes
- [x] 1.5 Update/add unit tests for new params and time grid behavior
- [x] 1.6 Run `openspec validate update-equity-pde-default-grid --strict`
- [x] 1.7 Run `python -m pytest` (suite currently has unrelated failures)
