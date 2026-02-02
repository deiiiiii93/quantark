# calendar-holiday-files Specification

## Purpose
TBD - created by archiving change add-exchange-holiday-calendars. Update Purpose after archive.
## Requirements
### Requirement: Holiday CSV Storage and Naming
The system SHALL load holiday CSV files from `util/calendar/holidayfile/`, and the CSV filename SHALL exactly match the calendar name with a `.csv` extension.

#### Scenario: Resolve holiday file by calendar name
- **WHEN** a calendar named `china_sse` is requested
- **THEN** the system loads `util/calendar/holidayfile/china_sse.csv`

### Requirement: Exchange-Level Calendar Naming
The system SHALL support exchange-level calendars in addition to national calendars, using a combined name format `{national}_{exchange}` in lowercase.

#### Scenario: Identify exchange-level calendar name
- **WHEN** the exchange-level calendar for Shanghai Stock Exchange is needed
- **THEN** the calendar name is `china_sse`

### Requirement: National and Exchange Calendar Coexistence
The system SHALL allow national and exchange-level calendars to coexist without name collisions.

#### Scenario: National and exchange calendars are distinct
- **WHEN** both `china` and `china_sse` calendars are defined
- **THEN** each calendar resolves to its own holiday CSV file

### Requirement: Exchange Calendar Fallback
The system SHALL fall back to the national calendar when an exchange-level holiday CSV is missing.

#### Scenario: Exchange CSV missing
- **WHEN** the `china_sse` holiday file is missing
- **THEN** the system uses the `china` calendar holidays

