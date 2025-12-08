## Context
- Discrete monitoring currently uses a single barrier level plus `observation_dates`; cannot express per-date barriers/returns.
- We need richer observation control for all barrier-like products (single, double, one-touch, double one-touch) without introducing separate products.

## Goals / Non-Goals
- Goals: Add an `ObservationSchedule` with per-date `ObservationRecord` data and aggregation modes (stop-first-hit, accumulate, best, worst) usable across barrier-like products; keep backward compatibility.
- Non-Goals: Rework continuous monitoring, redesign pricing engines, or change existing default payoffs.

## Decisions
- Reuse existing products; add optional `ObservationSchedule` instead of creating new discrete-only products.
- `ObservationRecord` carries observation time (year fraction or date), barrier data (single `barrier` or `upper_barrier`/`lower_barrier`), and payoff terms (rebate or return-rate style amount).
- `ObservationSchedule` is an ordered list of records with an aggregation mode:
  - `stop-first-hit`: evaluate records chronologically and stop at first hit.
  - `accumulate`: sum payoffs for each hit record.
  - `best`: take the maximum payoff among hit records.
  - `worst`: take the minimum payoff among hit records.
- Default/backward compatibility: if schedule is absent, retain existing `observation_type`/`observation_dates` semantics with uniform barriers/rebates.
- Applicability: schedule is available to BarrierOption, DoubleBarrierOption, OneTouchOption, DoubleOneTouchOption; continuous monitoring unchanged.

## Risks / Trade-offs
- Engine complexity increases to honor aggregation modes across product families → mitigate with clear validation and shared helpers.
- Mixed inputs (schedule + legacy fields) could conflict → detect and raise validation errors or define precedence.

## Open Questions
- Whether return-rate vs fixed-rebate needs explicit type tagging per record; assume both are supported with clear schema during implementation.

