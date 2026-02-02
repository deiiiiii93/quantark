## Context
Holiday calendars are loaded from mixed locations and only model national-level holidays. Exchange-level calendars are needed to reflect different holiday conventions across markets.

## Goals / Non-Goals
- Goals:
  - Centralize holiday CSV storage under `util/calendar/holidayfile/`.
  - Enforce a deterministic calendar-to-file naming rule.
  - Support exchange-level calendars alongside national calendars using combined names.
- Non-Goals:
  - Build a data ingestion pipeline for external holiday sources.
  - Redesign the business day adjustment API.

## Decisions
- Decision: Use combined calendar names for exchange-level calendars.
  - Format: `{national}_{exchange}` in lowercase (e.g., `china_sse`).
  - Rationale: Simple, explicit, file-name safe, and compatible with existing `CalendarType` string values.
- Decision: Holiday file names must match the calendar name exactly (case-sensitive) with `.csv` extension.
  - Rationale: Avoids ambiguous mapping and supports deterministic lookup.
- Decision: Keep existing national calendars (e.g., `china`) and add exchange-level variants.
  - Rationale: Preserves current usage while enabling exchange-specific conventions.
- Decision: If an exchange-level CSV is missing, fall back to the national calendar.
  - Rationale: Avoids hard failures while still allowing exchange-specific overrides.

## Risks / Trade-offs
- If exchange-level calendars are requested without a corresponding CSV, behavior must be defined (error vs fallback). This will be decided during implementation to avoid silent mispricing.

## Migration Plan
1. Create `util/calendar/holidayfile/`.
2. Move and rename existing holiday CSVs to follow the naming rule.
3. Update calendar loading logic and references in code/docs/tests.

## Open Questions
- None.
