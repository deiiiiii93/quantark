## MODIFIED Requirements

### Requirement: Holiday CSV Storage and Naming
The system SHALL load holiday CSV files from the `quantark.util` package
resources at `calendar/holidayfile/` (resolved via `importlib.resources`, not
filesystem paths derived from the repository layout), and the CSV filename
SHALL exactly match the calendar name with a `.csv` extension.

#### Scenario: Resolve holiday file by calendar name
- **WHEN** a calendar named `china_sse` is requested
- **THEN** the system loads the `calendar/holidayfile/china_sse.csv` resource
  from the `quantark.util` package

#### Scenario: Resolution works without a repository checkout
- **WHEN** quantark is installed non-editably into a fresh venv and a calendar
  named `china` is requested
- **THEN** the holiday CSV is found via package resources and the calendar
  loads its holidays
