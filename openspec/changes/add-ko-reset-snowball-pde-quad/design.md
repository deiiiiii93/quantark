## Context
KO-reset snowball options introduce a post-KI KO schedule that differs from the pre-KI schedule. The existing Snowball PDE/quad engines already implement two-surface V0/V1 logic and should be extended to apply different KO schedules per surface.

## Goals / Non-Goals
- Goals:
  - Provide deterministic PDE and quadrature pricing for `KnockOutResetSnowballOption`.
  - Reuse Snowball engine patterns (two-surface V0/V1 recursion).
  - Support pre-KI KO schedule on V0 and post-KI KO schedule on V1.
  - Integrate with `PDEEngine` dispatch.
- Non-Goals:
  - Implement REBASED post-KO schedules in PDE/quad (requires additional state).
  - Replace or refactor existing Snowball PDE/quad engines.

## Decisions
- Decision: Use combined two-surface logic (V0/V1) and apply pre/post KO schedules separately per surface.
  - Rationale: Matches existing Snowball engine pattern and minimizes new infrastructure.
- Decision: Support only `PostKOScheduleMode.ABSOLUTE` in PDE/quad.
  - Rationale: REBASED schedules depend on KI time and require extra state; this is out of scope.
- Decision: When `disable_ko_after_ki=True`, suppress post-KI KO application on V1.
  - Rationale: Consistency with Snowball behavior.

## Risks / Trade-offs
- REBASED mode will continue to rely on Monte Carlo; PDE/quad users must be warned via validation errors.

## Migration Plan
1. Add KO-reset PDE and quad engines.
2. Integrate with PDEEngine dispatcher and exports.
3. Add tests and docs.

## Open Questions
- None.
