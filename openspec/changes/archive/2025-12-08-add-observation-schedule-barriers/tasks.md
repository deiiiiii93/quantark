## 1. Data model
- [x] 1.1 Add `ObservationRecord` with observation time, barrier inputs (single or upper/lower), rebate/return-rate/payoff fields, and validation.
- [x] 1.2 Add `ObservationSchedule` with ordered records, aggregation mode enum (stop-first-hit, accumulate, best, worst), and compatibility defaults.

## 2. Product integration
- [x] 2.1 Update `BarrierOption` to accept schedule, validate discrete monitoring, and map legacy fields when schedule absent.
- [x] 2.2 Update `DoubleBarrierOption` similarly for upper/lower barriers.
- [x] 2.3 Update `OneTouchOption` and `DoubleOneTouchOption` to accept schedule while keeping existing observation settings.

## 3. Engine support
- [x] 3.1 Ensure analytical/PDE/MC barrier pricing paths ingest `ObservationSchedule` semantics (hit detection per record, aggregation mode behavior).
- [x] 3.2 Add safeguards for mixed inputs (schedule + legacy fields) and raise validation errors on conflicts.

## 4. Tests & docs
- [x] 4.1 Add unit tests covering each aggregation mode across product types.
- [x] 4.2 Add examples/documentation snippets showing schedule usage and backward compatibility.

